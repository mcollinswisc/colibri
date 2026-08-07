# The VRAM tier under-fills because the prefix is sized by the widest expert

Found on `dev@e323575` (v1.5.0), RTX PRO 6000 Blackwell (102.0 GB) / 62.7 GiB host.
Fixed in `fix(cuda): size the VRAM prefix by the routed expert, not the widest`.
Rebased onto `dev@08a499e`, which carries #869 -- see *Relationship to #869* below.

## Summary

`expert_bytes_probe()` ([c/telemetry.h](../c/telemetry.h)) returns the MAX expert
width the container holds. `7c28e18` (*"fix(plan): size the expert slot by the widest
width the container holds"*, 2026-08-03, #766) made it do that because slabs are
reused across slots and grow to the widest expert they have ever held, so a slot that
outlives one row has to be sized by the widest.

`pin_load()` then reused the same `eb` for two quantities that describe a single
**routed** expert, and both were wrong by the width ratio. The VRAM staging prefix is
the one place in the engine where a slab provably does *not* outlive its row: it is
freed the instant the upload lands.

## The widths, read from the container headers

A safetensors header is a `u64` length followed by that much JSON, so this needs no
library:

| | routed (layer 3) | MTP head (layer 78) | ratio |
|---|---|---|---|
| `glm52_i4_g64` | 6,291,456 + 786,432 scales, ×3 = **21.23 MB** | 12,582,912 + 8,192, ×3 = **37.77 MB** | 1.78 |
| `glm52_i4_row` | 6,291,456 + 8,192 scales, ×3 = **18.90 MB** | 12,582,912 + 8,192, ×3 = **37.77 MB** | 2.00 |

The MTP head is byte-identical in both: exactly 2× the routed weight bytes, i.e.
int8 against int4. Only the scale sidecar differs — per-group-of-64 for g64,
per-row for row. **Both containers are affected**, `row` slightly worse. An earlier
version of this note claimed `row` was immune on the grounds that its routed experts
are a uniform 18.9 MB; that was true and irrelevant, because the head is what the
probe returns.

## 1. The tier under-fills by the width ratio

    prefix_est=(int)(budget/eb)+g_cuda_ndev;

`prefix_est` is how many experts get staged, but only routed experts are ever
staged. Dividing the budget by the MTP width undercounts by the ratio above:

| container | `CUDA_EXPERT_GB` | before | after |
|---|---|---|---|
| g64 | 96 | 53.95 GB | **94.64 GB** |
| g64 | auto (96.9 measured) | — | **94.79 GB** |
| row | 60 | 30.04 GB | **60.00 GB** |

`auto` was affected identically: it sets `budget = safe_total`, real measured
headroom, then divides by the same inflated `eb`.

## 2. `resident_bytes` leaked the difference on every release

Staging credited `count*eb` while `expert_host_release()` debits the true
`qt_bytes`, so each released expert leaked `37.77 − 21.23 = 16.54 MB` — about
**75 GB** at a full tier. The consequences, in order:

1. `cap_for_ram()` drove the LRU cap to its floor of 1.
2. The pre-flight guard refused to start: `projected peak 81.3 GB ... exceeds the
   49.7 GB actually available`, on a run whose kernel-measured `VmHWM` was
   **22.77 GB**. `COLI_RAM_OVERCOMMIT=1` was required to use a large tier at all —
   overriding a real safety net because of an accounting error.

After the fix the projection is honest: 53.3 GB projected against 34.5 GB measured,
the gap being the LRU headroom that a longer session actually fills.

## The fix

- `prefix_est` and the staging round size divide by `expert_bytes_row()` at
  `c->first_dense` — the routed width — instead of the probe's MAX. Identical on any
  single-width container. `eb` stays for `npin` and `expert_avail()`, which size
  slots that *do* stay host-resident.
- The `resident_bytes` credit **sums the `qt_bytes` actually loaded** instead of
  estimating, so it cannot diverge from what release debits whatever widths a future
  container mixes. Same pattern the REPIN `gpu_swap` path already used.
- `[PIN] placement` sums the `qt_bytes` of slots whose `slab` is still non-NULL
  rather than multiplying a count by the widest width. #869 corrected `TIERS`,
  `[PROF] resident experts` and `[PROF] config` for this reason; this line lives
  inside `pin_load()` and was missed. `expert_host_release()` NULLs `slab`, so
  "still resident" needs no separate bookkeeping.
- Sizing the prefix correctly made a latent bug in the #730 round loop reachable:
  the inner upload loop stops at the budget, but the outer loop kept reading further
  rounds into host RAM that nothing then released. It now stops on the first round
  that places nothing, and the RAM suffix pass starts from that rank rather than
  `gpu_prefix`, so those slots become ordinary host pins instead of slots with a
  NULL slab that every lookup would read as expert 0.
- `c/tests/test_expert_width.c` pins the contract for both functions against a
  fabricated tensor index — `st_find()` linear-scans when `hidx` is NULL, so it
  needs no model, no snapshot and no GPU.

## 3. Still broken: whether the tier exists at all is a remainder

Not caused by `7c28e18` and **not fixed here**.

`autopin_lru_reserve()` reserves the LRU *before* sizing the pin plan, so when the
budget cannot afford the requested cap the surviving plan is a remainder. `pin_load()`
is only called when that remainder clears 0.5 GB — and because the CUDA tier is built
*inside* `pin_load()`, missing the threshold silently disables the VRAM tier
entirely, with no `[CUDA]` line and no error.

Measured after the fix, `--vram 85 --ram 54` at the default cap of 64:

    [PIN] auto: 17.6 GB plan capped to 0.4 GB to preserve the no-pin LRU cap 12/layer
    ... cap lowered 64->12 ...
    -> no tier, 0% pin hits, 0.59 tok/s

`--cap 6` gets the tier built. The fix does not help here because `expert_avail()`
runs *before* the pin, when `resident_bytes` is still just the dense weights — the
phantom accrued during `pin_load()`, after the plan had been sized. The tier should
not be contingent on the RAM pin plan clearing an unrelated floor.

#869 does not change this either. It cuts the per-slot cost by the width ratio, but
the reserve is clamped to `expert_available` regardless (`cap = min(requested,
affordable)`), so it still absorbs essentially the whole budget and the remainder
still lands under the floor. The narrower slot buys more slots, not a larger plan.

## Relationship to #869

#869 (*"plan: price each expert row at its own width, not the container's widest"*,
merged 2026-08-07) fixes the **other** consequence of `7c28e18`, reported as #856:
`cap_for_ram()` charged every per-row `ecache[layer]` slot the widest width, halving
the RAM cache (154 -> 77 slots per row on GLM-5.2). It is a disjoint change — it never
enters `pin_load()`, and the three lines quoted at the top of this note are still
`eb`-based on `dev` today. Its PR says so: *"the PIN path still charges
`resident_bytes` at the widest width [...] Separate change."*

Two things it gives this branch:

- `expert_bytes_row(m, layer, ebits)`, which at `c->first_dense` is exactly the
  routed-width helper this fix used to define for itself. That helper is gone; the
  fix calls #869's.
- `expert_load()` now **shrinks** slabs, not just grows them (25% hysteresis, arena
  slices exempt). That is what makes per-row accounting true rather than optimistic,
  since slabs migrate between rows through the `ws[]` swap.

Two things to re-measure because of it, **not yet done**:

- The cap. `cap_for_ram()`'s divisor on g64 falls from `77 x 37.77 = 2908 MB` to
  `75 x 21.23 + 37.77 = 1630 MB`, so `capmax` scales by **1.78x** against an unchanged
  `avail`: the raise this box saw to 8 should now land near 14. Every tok/s figure in
  this note and in `launch_chat.sh` predates that.
- The shrink's cost. On a mixed container it fires whenever an MTP slab reaches a
  routed row — `mmap`/`munmap` plus first-touch faults at 19-38 MB. #869 states it is
  unmeasured and asks for exactly this measurement. It only fires if the MTP row is
  actually loaded, so check `[MTP]` before reading any A/B.

## Reproduction

    make -C c colibri CUDA=1
    COLI_CUDA=1 COLI_GPU=0 CUDA_RELEASE_HOST=1 DIRECT=1 URING=1 \
    CUDA_EXPERT_GB=96 RAM_GB=54 CTX=4096 NGEN=8 COLI_TEMP=0 \
    SNAP=/mnt/hermes/models/glm52_i4_g64 COLI_PROMPT="hi" \
      ./c/colibri 6 4 4

Before: `VRAM 53.95 GB (budget 96.0 GB)` then `[RAM] refusing to start`, exit 2.
After: `VRAM 94.64 GB`, exit 0. Sample `VmHWM` from `/proc/<pid>/status` against
the `resident` figure in the `[RAM_GB=]` line to see the phantom.

Direct engine runs append to `<SNAP>/.coli_usage`, which autopin reads at the next
startup — back it up and restore it, or the next run measures a moving target.

## A note on measuring throughput here

Between-batch drift on this box is ~20%, far larger than the ±4% spread *within* a
warm batch: the same configuration measured 0.85 in one batch and 1.03 in another
with byte-identical hit rates and tier size. Upstream #718 reports the same. Any A/B
must **interleave** its arms inside one batch. Two configurations benchmarked as
separate batches are not comparable, however many repeats each one has.
