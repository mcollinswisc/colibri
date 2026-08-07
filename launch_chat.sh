#!/bin/sh
# Measured-best config for glm52_i4_g64 on this box (RTX PRO 6000 96 GiB / 62.7 GiB host).
# 1.10 tok/s median, 84.9 GB expert tier + dense on the card, peak RSS 33 GB, hit 69.5%.
#
# STALE SINCE THE #869 REBASE, in the direction of "probably better, unverified":
# cap_for_ram's divisor dropped 1.78x, so the raise below lands near 14 rather than 8,
# and expert_load now shrinks slabs as well as growing them (new allocator work on a
# mixed-width container).  The tier size is unaffected.  Re-measure with bench-toks.sh,
# interleaved, before trusting the tok/s above.  See docs/expert-width-vram-prefix.md.
#
#   --vram 85    means 85.  Before the fix on this branch it did not: the engine sized
#                the VRAM staging prefix with the WIDEST expert in the container (the
#                int8 MTP head, 37.77 MB) while routed experts cost 21.23 MB on the
#                card, so you had to ask for 151 to get 85.  See
#                docs/expert-width-vram-prefix.md.  --vram auto works too and lands
#                within a GB of this; 85 is written out so the number is reviewable.
#   --cap 6      still load-bearing, and NOT for the reason you would guess.  At the
#                default 64 the autopin LRU reserve eats the whole RAM budget, the pin
#                plan collapses to 0.4 GB, and main() only calls pin_load() when the
#                plan clears 0.5 GB -- so the CUDA tier, which is built inside
#                pin_load(), silently never happens.  Measured at the default: no
#                [CUDA] tier line at all, 0% pin hits, 0.59 tok/s.  The 6 is only a
#                request; cap_for_ram raises it from there, and #869 raised how far.
#
# Do NOT add PIN_GB: an explicit RAM pin raises hit rate but costs throughput
# (0.88 at PIN_GB=45, 0.86 at 62).  That is specific to explicit pins -- the LRU
# growing to cap 8 is a straight win, worth ~12% in a paired A/B against cap 1
# (8s less disk wait, no extra matmul).
#
# COLI_RAM_OVERCOMMIT is deliberately gone: the guard it used to override was firing
# on ~75 GB of phantom resident_bytes, which the fix removed.

. "${HOME}/env/colibri/bin/activate"

cd c
CUDA_RELEASE_HOST=1 DIRECT=1 URING=1 THINK=1 \
COLI_MODEL=/mnt/hermes/models/glm52_i4_g64/ \
  ./coli chat --gpu auto --vram 85 --ram 54 --cap 6
