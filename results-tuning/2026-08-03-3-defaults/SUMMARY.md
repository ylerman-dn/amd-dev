# RCCL default algo/proto/nchannels across 4K–512M, single node

Job 14048, `amd-mi355x-4`, 1 node, 8 ranks × 1 GPU, all_reduce.
`all_reduce_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1` via
`srun --mpi=pmix --ntasks-per-node=8`. **Nothing forced** — no `NCCL_ALGO`, no `NCCL_PROTO`,
no `NCCL_MIN/MAX_NCHANNELS`. 3 repeats.

3/3 repeats rc=0, 18 rows each, `#wrong`=0 throughout. busbw is in-place, median of 3.

**`nch` column corrected 2026-08-03** after the INFO capture
(`results-tuning/2026-08-03-4-infocap/info.log`) showed that `-A 1`'s channel column reports the
planned ceiling, not what ran. `nch(-A)` is what the benchmark column printed; `nch(real)` is
the actual channel span from INFO (`channel{Lo..Hi}`, count = Hi−Lo+1). algo/proto were
confirmed correct in the `-A` column at every size.

| size | algo | proto | nch(-A) | **nch(real)** | median GB/s | min | max | spread |
|---|---|---|---|---|---|---|---|---|
| 4K | TREE | LL | 1 | 1 | 0.32 | 0.32 | 0.33 | 3.1% |
| 8K | TREE | LL | 1 | 1 | 0.55 | 0.54 | 0.55 | 1.8% |
| 16K | TREE | LL | 1 | 1 | 0.86 | 0.86 | 0.88 | 2.3% |
| 32K | TREE | LL | 2 | 2 | 1.71 | 1.69 | 1.72 | 1.8% |
| 64K | TREE | LL | 4 | 4 | 3.32 | 3.30 | 3.32 | 0.6% |
| 128K | TREE | LL | 8 | 8 | 6.35 | 6.34 | 6.36 | 0.3% |
| 256K | TREE | LL | 16 | 16 | 12.67 | 12.63 | 12.68 | 0.4% |
| 512K | RING | SIMPLE | 16 | 16 | 21.23 | 21.18 | 21.33 | 0.7% |
| 1M | RING | SIMPLE | 32 | 32 | 41.78 | 41.41 | 41.82 | 1.0% |
| 2M | RING | SIMPLE | 56 | **52** | 79.78 | 79.21 | 79.82 | 0.8% |
| 4M | RING | SIMPLE | 56 | **52** | 128.68 | 128.00 | 128.89 | 0.7% |
| 8M | RING | SIMPLE | 56 | **54** | 198.03 | 197.44 | 198.26 | 0.4% |
| 16M | RING | SIMPLE | 56 | 56 | 269.14 | 268.42 | 269.48 | 0.4% |
| 32M | RING | SIMPLE | 56 | 56 | 317.32 | 317.02 | 318.11 | 0.3% |
| 64M | `RING*` | SIMPLE | 56 | **222** | 355.65 | 355.54 | 356.13 | 0.2% |
| 128M | `RING*` | SIMPLE | 56 | **222** | 379.59 | 379.28 | 380.36 | 0.3% |
| 256M | `RING*` | SIMPLE | 56 | **223** | 390.65 | 390.21 | 391.25 | 0.3% |
| 512M | `RING*` | SIMPLE | 56 | **224** | 396.84 | 396.25 | 397.07 | 0.2% |

## Three transitions

1. **256K → 512K: TREE/LL → RING/SIMPLE.** Both algo and proto flip at the same boundary.
2. **Channel ramp**: 1 channel up to 16K, then 2, 4, 8, 16, 32, reaching 52 at 2M, 54 at 8M and
   56 at 16M–32M, then jumping to ~222–224 under WarpSpeed from 64M. The `-A 1` column hides
   both ends of that — see the note above the table.
   **48-vs-56 — settled: there is no 48-channel cap in effect here.** The INFO log shows real
   channel counts of 52, 54, 56, 222, 223 and 224 — every one above 48
   (`results-tuning/2026-08-03-4-infocap/info.log`). Exceeding 48 is direct proof the cap is not
   applied, regardless of what the binary's cap message says. An earlier draft of this file
   called this "unresolved" because the cap *message* never printed; that conflated two
   questions and was wrong.
   Still open, and narrower: **under what conditions the cap would fire.** Disassembly of
   `rcclRestrictMaxChannels` suggests gates on `nNodes >= 2` and gfx950, which would explain the
   single-node case (log confirms `nRanks 8 nNodes 1`) — but that rests on identifying a struct
   offset as `nNodes` by calibration, so it is inference, not fact. A 2-node run decides it.
3. **32M → 64M: RING → `RING*`.** WarpSpeed engages only at 64M and above — which matches AMD's
   documented behaviour exactly (auto mode applies to AllReduce and AllGather "from 64MB",
   ReduceScatter from 256MB). Below 64M, plain ring.

## Selection is deterministic

Zero selection mismatches across 3 repeats at all 18 sizes. That retroactively supports the
one-run-per-combo approach used in step 2 — for reading *selection*, a single run is sound.
It says nothing about timing, which still needs repeats.

## Spread is small here

0.2–3.1%, worst at the smallest sizes (4K at 3.1%), tightening to ~0.3% above 8M. Worth
recording because a previous session reported wide run-to-run spread at small-to-mid sizes as
the cause of "phantom winners". At 256K we measure 0.4% spread. That is not a refutation — those
runs were forced configs and possibly multi-node — but on this configuration, with `-n 20` and
one node, the noise floor is low enough that ~3 repeats should be sufficient for A/B work.

## Where tuning could plausibly help

Nothing here is a claim; these are the hypotheses this map suggests.

- **4K–256K (TREE/LL, 1–16 channels)**: the lowest-throughput region by far. Whether TREE/LL is
  actually optimal here is untested — RING/LL is an honoured alternative (step 2) and worth
  measuring.
- **The 256K/512K boundary**: RCCL flips two knobs at once. The crossover may not be at the
  ideal size.
- **512K–32M (plain RING/SIMPLE)**: WarpSpeed is not used here but is used above. It *can* be
  requested — `librccl.so` exposes `RCCL_WARP_SPEED_AUTO`, `RCCL_WARP_SPEED_FORCE_ENABLE`,
  `RCCL_WARP_SPEED_CU_COUNT` and per-collective thresholds `RCCL_WARP_SPEED_AR_THRESHOLD` /
  `_AG_` / `_RS_`, so the 64M onset is a movable threshold rather than a hard floor.
  But note what WarpSpeed actually buys: per AMD's docs it **reduces CU count by ~50% at equal
  performance**, so extending it downward is a compute-efficiency win, *not* necessarily a busbw
  win. An earlier draft of this file framed it as a throughput opportunity; that was wrong.
  It is still worth measuring, because freeing half the CUs matters for real workloads where
  comms and compute contend — but busbw alone will not show the benefit.
- **≥64M**: already at 355–397 GB/s on WarpSpeed, and step 2 found nothing beating default at
  256M/512M. Likely the least fruitful region for busbw.

Note also `Overriding %s algorithm with RING for nccl%s at %zu bytes as WarpSpeed is requested
and only supports RING` — WarpSpeed forces RING, which is why TREE never carries the asterisk.
