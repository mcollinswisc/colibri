/* test_expert_width.c -- the routed-vs-widest expert width contract.
 *
 * expert_bytes_probe() must return the WIDEST expert width the container holds
 * (#766: ws[] slots are reused across layers, so a slot sized by that is safe).
 * expert_bytes_row() at c->first_dense must return the ROUTED width -- what one
 * expert costs in VRAM, and what pin_load() hands back to the host after
 * uploading it.
 *
 * They diverge only on a container that MIXES widths, which is exactly how
 * GLM-5.2 ships: int4 routed experts beside an int8 MTP head, measured at
 * 21.23 MB and 37.79 MB. Feeding the widest width to the VRAM staging prefix
 * under-fills the tier by that ratio (96 GB asked, 54 GB placed) and leaks the
 * difference into resident_bytes on every release, because
 * expert_host_release() debits the true qt_bytes.
 *
 * test_cap_mixed_width.c covers the other consumer of the same split -- the
 * per-row LRU cap and its divisor. This one covers the two primitives it is
 * built from, including the fallbacks that no fixture with a readable header
 * can reach.
 *
 * Both are pure functions of the tensor index, and st_find() falls back to a
 * linear scan when S->hidx is NULL, so this exercises them against a fabricated
 * index -- no model, no snapshot, no GPU. Portable, same as test_cap_precedence.c. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int failures = 0;

/* Synthetic widths in the ratio the real container has (37.2/21.0 = 1.77 against
 * the measured 37.79/21.23 = 1.78); round numbers keep the expectations obvious. */
#define ROUTED_W   6000000LL
#define ROUTED_QS  1000000LL
#define ROUTED_TOT (3*(ROUTED_W+ROUTED_QS))          /* 21.0 MB */
#define MTP_W     11400000LL
#define MTP_QS     1000000LL
#define MTP_TOT   (3*(MTP_W+MTP_QS))                 /* 37.2 MB */

#define MAX_T 32
static st_tensor g_t[MAX_T];
static int g_tn;

static void add_expert_layer(int layer, int64_t w, int64_t qs){
    const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++){
        char *n=malloc(256);
        snprintf(n,256,"model.layers.%d.mlp.experts.0.%s.weight",layer,suf[k]);
        g_t[g_tn].name=n; g_t[g_tn].nbytes=w; g_tn++;
        char *q=malloc(256);
        snprintf(q,256,"model.layers.%d.mlp.experts.0.%s.weight.qs",layer,suf[k]);
        g_t[g_tn].name=q; g_t[g_tn].nbytes=qs; g_tn++;
    }
}

/* first_dense=3 / n_layers=78 mirror GLM-5.2: layer 3 is the first MoE layer, and
 * the MTP head is indexed one past the last transformer layer. */
static void fixture(Model *m, int has_mtp){
    memset(m,0,sizeof(*m));
    memset(g_t,0,sizeof(g_t)); g_tn=0;
    m->c.first_dense=3; m->c.n_layers=78;
    m->c.hidden=5120;   m->c.moe_inter=1536;
    m->has_mtp=has_mtp;
    m->S.t=g_t; m->S.hidx=NULL;                       /* NULL hidx -> st_find linear-scans */
}
static void seal(Model *m){ m->S.n=g_tn; }

/* The width pin_load() sizes the VRAM staging prefix with. */
static int64_t routed_of(Model *m, int ebits){
    return expert_bytes_row(m,m->c.first_dense,ebits);
}

static void eq(const char *label, int64_t got, int64_t want){
    if(got!=want){
        fprintf(stderr,"FAIL %-44s -> %lld (want %lld)\n",
                label,(long long)got,(long long)want);
        failures++;
    }
}

int main(void){
    Model m;
    const int ebits=4;

    /* 1. MIXED WIDTHS -- the case that broke. The probe sees the int8 MTP row and
     *    must report it; the routed row must stay the int4 expert. */
    fixture(&m,1);
    add_expert_layer(m.c.first_dense,ROUTED_W,ROUTED_QS);
    add_expert_layer(m.c.n_layers,MTP_W,MTP_QS);
    seal(&m);
    eq("mixed: routed width is the MoE layer",   routed_of(&m,ebits),           ROUTED_TOT);
    eq("mixed: probe width is the MTP head",     expert_bytes_probe(&m,ebits),  MTP_TOT);
    eq("mixed: the MTP row reports its own",     expert_bytes_row(&m,m.c.n_layers,ebits), MTP_TOT);
    if(expert_bytes_probe(&m,ebits) <= routed_of(&m,ebits)){
        fprintf(stderr,"FAIL mixed: probe must exceed routed\n"); failures++;
    }

    /* 2. UNIFORM WIDTHS -- every single-width container, e.g. the per-row int4
     *    build of the same model. The two must agree exactly: that is the
     *    guarantee that separating them changes nothing outside case 1. */
    fixture(&m,0);
    add_expert_layer(m.c.first_dense,ROUTED_W,ROUTED_QS);
    seal(&m);
    eq("uniform: routed width",                  routed_of(&m,ebits),          ROUTED_TOT);
    eq("uniform: probe agrees with routed",      expert_bytes_probe(&m,ebits), ROUTED_TOT);

    /* 3. NARROWER MTP row. The probe is a MAX, not "whatever the MTP head is":
     *    a head narrower than the routed experts must not shrink the slot. */
    fixture(&m,1);
    add_expert_layer(m.c.first_dense,MTP_W,MTP_QS);    /* wide routed experts */
    add_expert_layer(m.c.n_layers,ROUTED_W,ROUTED_QS); /* narrow head */
    seal(&m);
    eq("narrow MTP: routed width unchanged",     routed_of(&m,ebits),          MTP_TOT);
    eq("narrow MTP: probe keeps the max",        expert_bytes_probe(&m,ebits), MTP_TOT);

    /* 4. NO EXPERT TENSORS -- an unreadable header. Both fall back to the same
     *    analytic estimate for a routed row. The MTP row deliberately does not:
     *    expert_bytes_row() doubles it there rather than guess it is as narrow as
     *    a routed row, since under-reserving is the direction that ends in an
     *    OOM-kill. pin_load() only ever asks for first_dense, so it never sees
     *    that branch -- assert it anyway, because that is why asking for the
     *    routed row by name is not the same as asking for "not the MTP one". */
    fixture(&m,1);
    seal(&m);
    { int64_t want = tbytes(m.c.moe_inter,m.c.hidden,ebits)*2
                   + tbytes(m.c.hidden,m.c.moe_inter,ebits);
      eq("no tensors: routed falls back",        routed_of(&m,ebits),          want);
      eq("no tensors: probe uses the same copy", expert_bytes_probe(&m,ebits), want);
      eq("no tensors: MTP row reserves double",  expert_bytes_row(&m,m.c.n_layers,ebits), want*2); }

    /* 5. MTP row present but the MoE layer is missing from the index: the probe
     *    must still report the head rather than the analytic fallback. */
    fixture(&m,1);
    add_expert_layer(m.c.n_layers,MTP_W,MTP_QS);
    seal(&m);
    eq("head only: probe reports the head",      expert_bytes_probe(&m,ebits), MTP_TOT);

    if(failures){
        fprintf(stderr,"expert width tests: %d FAILURE(S)\n",failures);
        return 1;
    }
    puts("expert width tests: ok");
    return 0;
}
