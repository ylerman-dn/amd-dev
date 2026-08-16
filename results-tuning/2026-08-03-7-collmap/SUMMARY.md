# Acceptance map across 5 more collectives

Job 14059, `amd-mi355x-4`, 1 node, 8 ranks × 1 GPU, 4K–512M f=2, channels unset.
`-n 20 -w 5 -c 1 -A 1 -Z csv -X <file>`, `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=..._%p.log`
(standard measured-run config). 5 smoke + 37 matrix runs, **all rc=0**.

Collectives: all_gather, reduce_scatter, alltoall, broadcast, reduce. Per collective: 6 forced
algo/proto pairs + 1 unforced reference, plus a `PAT` probe for all_gather and reduce_scatter.

## Acceptance — what is honoured, per collective

| collective | honoured | never honoured |
|---|---|---|
| all_reduce *(batch 6)* | RING/SIMPLE, RING/LL, **TREE/LL** | LL128, TREE+SIMPLE, NVLS, PAT |
| all_gather | **nothing below 16M** (`Direct` path); RING/SIMPLE + RING/LL at ≥16M | TREE (any proto), LL128, PAT |
| reduce_scatter | RING/SIMPLE, RING/LL | TREE (any proto), LL128, PAT |
| broadcast | RING/SIMPLE, RING/LL | TREE (any proto), LL128 |
| reduce | RING/SIMPLE, RING/LL | TREE (any proto), LL128 |
| alltoall | **no algo/proto concept** — `-A 1` returns `N/A/N/A/N/A` at every size | — |

Three results worth separating out:

1. **TREE is honoured only for all_reduce**, and only with LL. Every other collective silently
   converts TREE to RING at every size.
2. **LL128 is never honoured by any collective at any size.** It always becomes SIMPLE.
3. **PAT is never honoured** — substituted to `Direct`/SIMPLE for all_gather and RING/SIMPLE for
   reduce_scatter. `tuning.cc` claims PAT is valid for exactly these two collectives; empirically
   it is not, on this build.

**all_gather is the outlier**: below 16M it reports `Direct`/SIMPLE regardless of what is
requested — including in the unforced reference — so forcing algo/proto there has *zero* effect.
It only becomes steerable at ≥16M, and then only RING/LL is a real alternative. This is the one
collective where acceptance is size-dependent.

## `-A 1` and INFO report different layers — not a conflict

At all_reduce 64M+, `-A 1` says `RING*` while INFO's per-call line says `Algo RING`. Checked
across all 18 sizes:

- base algo agrees **18/18**, proto agrees **18/18**
- string-identical algo only **14/18** — the four differences are exactly the WarpSpeed sizes

So `-A 1` reports the **RCCL addon layer** (`RING*` = WarpSpeed, `Direct`), and INFO's per-call
line reports the **NCCL enum** underneath it. `-A 1` is strictly the more informative of the two
about what ran, and `Direct` for all_gather is only visible there. Earlier wording in
`findings/01` said the two "matched at 18/18", which was imprecise — they agree on base algo and
proto, not on the addon marker.

## Defaults and real channel counts per collective

Unforced, real counts from INFO (`channel{Lo..Hi}`). Note the channel ceiling is **not** 56 for
every collective:

| collective | default proto switch | channel ceiling |
|---|---|---|
| all_reduce | LL → SIMPLE at 512K | 56, then 222–224 under WarpSpeed |
| reduce_scatter | LL → SIMPLE at 2M | 56, no WarpSpeed anywhere ≤512M |
| broadcast | LL → SIMPLE at 512K | **112** (via 86, 103, 108, 111) |
| reduce | LL → SIMPLE at 512K | **112** (identical to broadcast) |

broadcast and reduce behave identically at every size and use double all_reduce's channel
ceiling. Neither they nor reduce_scatter showed WarpSpeed at any size up to 512M — for
reduce_scatter that is worth noting against AMD's documented "ReduceScatter from 256MB", which we
did not observe.

**Parsing gotcha:** for all_gather and reduce_scatter the byte value in the INFO line is
**per-rank**, i.e. the `-b` size divided by 8, while the `-A 1` table row shows the total. Aligning
the two sources requires that conversion.

## Validating `unsupported_combos.yaml`

| its rule | verdict |
|---|---|
| all_gather: TREE unsupported | **correct** |
| reduce_scatter: TREE unsupported | **correct** |
| broadcast: RING only | **correct** |
| reduce: RING only | **correct** |
| alltoall: TREE unsupported | correct, trivially — no algo concept at all |
| TREE+SIMPLE @ 1 node | **correct** |

Six rules, all correct. But three significant gaps: **no LL128 rule** (never works, any
collective), **no PAT rule** (never works, and the file's cited source claims the opposite), and
nothing about all_gather being unsteerable below 16M.

## Practical consequence for the sweep

`autotune_config.yaml` would sweep 6 collectives × 2 algos × 3 protos = 36 combinations. Across
all six collectives only about **10 distinct real configurations exist**. Everything else either
duplicates a config under a different label or crashes (COLLNET, batch 6). For all_gather below
16M there is nothing to sweep at all.

## Files

37 CSVs + 42 stdout logs (incl. 5 Phase-0 smoke logs), `debug_files.tgz` (per-rank INFO logs).
