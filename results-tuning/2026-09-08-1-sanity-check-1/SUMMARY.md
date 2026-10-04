# SUMMARY — sanity-check-1 (2026-09-08): Process A by one command, grid 1..48

Command: `rccl_tune.py run --collectives all_reduce --scales 1 --name sanity-check-1` (tool 3b56455). Wall 2421 s = 40 min: search 29 min (63 search runs + 3 default runs, 22% fewer than the 81-run grid), A/B 9 min (9 reps/arm), node 8, job 21057, released by the tool.

Verdict: **9 of 18 sizes KEPT, all P(sup)=1.00** (4K..1M, gains +9..+73%); 9 DROPPED as landmines (2M..512M, -8..-35%) because the grid tops at 48 channels while RCCL's default runs 64-112 there. `all_reduce_1n.final.conf` = 7 rules (adjacent identical sizes merged). Expected for this grid; checks 2 and 3 raise the cap.

## Per size: default vs ours (A/B, `ab_out/allreduce_1n/per_size_stats.csv`; requested from `all_reduce_1n.conf`)

| size | default executed | ours requested | ours executed | default med | ours med | gain | P(sup) | verdict |
|---|---|---|---|---|---|---|---|---|
| 4096 | TREE/LL/1 | tree/ll/4 | TREE/LL/2 | 0.33 | 0.36 | +9.1% | 1.0 | KEPT |
| 8192 | TREE/LL/1 | tree/ll/48 | TREE/LL/4 | 0.57 | 0.71 | +24.6% | 1.0 | KEPT |
| 16384 | TREE/LL/1 | tree/ll/8 | TREE/LL/8 | 0.89 | 1.41 | +58.4% | 1.0 | KEPT |
| 32768 | TREE/LL/2 | tree/ll/40 | TREE/LL/16 | 1.76 | 2.78 | +58.0% | 1.0 | KEPT |
| 65536 | TREE/LL/4 | tree/ll/40 | TREE/LL/32 | 3.30 | 5.40 | +63.6% | 1.0 | KEPT |
| 131072 | TREE/LL/8 | tree/ll/40 | TREE/LL/32 | 6.56 | 9.65 | +47.1% | 1.0 | KEPT |
| 262144 | TREE/LL/16 | ring/ll/32 | RING/LL/32 | 13.07 | 18.50 | +41.5% | 1.0 | KEPT |
| 524288 | RING/SIMPLE/16 | ring/ll/48 | RING/LL/43 | 20.87 | 36.07 | +72.8% | 1.0 | KEPT |
| 1048576 | RING/SIMPLE/32 | ring/ll/40 | RING/LL/40 | 38.09 | 44.68 | +17.3% | 1.0 | KEPT |
| 2097152 | RING/SIMPLE/64 | ring/ll/48 | RING/LL/47 | 77.53 | 71.57 | -7.7% | 0.0 | dropped |
| 4194304 | RING/SIMPLE/103 | ring/simple/48 | RING/SIMPLE/47 | 130.22 | 109.34 | -16.0% | 0.0 | dropped |
| 8388608 | RING/SIMPLE/103 | ring/simple/48 | RING/SIMPLE/47 | 201.50 | 147.96 | -26.6% | 0.0 | dropped |
| 16777216 | RING/SIMPLE/108 | ring/simple/48 | RING/SIMPLE/48 | 274.15 | 188.27 | -31.3% | 0.0 | dropped |
| 33554432 | RING/SIMPLE/111 | ring/simple/48 | RING/SIMPLE/48 | 322.94 | 213.40 | -33.9% | 0.0 | dropped |
| 67108864 | RING/SIMPLE/111 | ring/simple/48 | RING/SIMPLE/48 | 358.94 | 234.56 | -34.7% | 0.0 | dropped |
| 134217728 | RING/SIMPLE/112 | ring/simple/48 | RING/SIMPLE/48 | 376.38 | 251.42 | -33.2% | 0.0 | dropped |
| 268435456 | RING/SIMPLE/112 | ring/simple/48 | RING/SIMPLE/48 | 385.63 | 263.44 | -31.7% | 0.0 | dropped |
| 536870912 | RING/SIMPLE/112 | ring/simple/48 | RING/SIMPLE/48 | 390.22 | 268.12 | -31.3% | 0.0 | dropped |

## Consistency: the two 'default' measurements agree

Search-stage default runs (3, `search_raw/d000*/merged_exec.csv`, env path, image flag removed) vs A/B default arm (9 reps, no plugin, flag removed): same executed algo/proto/channels at every size; busbw within ±3% (4K) and ±2% elsewhere.

| size | search default med | A/B default med | diff |
|---|---|---|---|
| 4096 | 0.34 | 0.33 | +3.0% |
| 8192 | 0.56 | 0.57 | -1.8% |
| 16384 | 0.88 | 0.89 | -1.1% |
| 32768 | 1.76 | 1.76 | +0.0% |
| 65536 | 3.33 | 3.30 | +0.9% |
| 131072 | 6.54 | 6.56 | -0.3% |
| 262144 | 13.04 | 13.07 | -0.2% |
| 524288 | 20.91 | 20.87 | +0.2% |
| 1048576 | 37.90 | 38.09 | -0.5% |
| 2097152 | 77.18 | 77.53 | -0.5% |
| 4194304 | 130.26 | 130.22 | +0.0% |
| 8388608 | 201.66 | 201.50 | +0.1% |
| 16777216 | 273.90 | 274.15 | -0.1% |
| 33554432 | 323.05 | 322.94 | +0.0% |
| 67108864 | 359.36 | 358.94 | +0.1% |
| 134217728 | 376.43 | 376.38 | +0.0% |
| 268435456 | 385.74 | 385.63 | +0.0% |
| 536870912 | 389.99 | 390.22 | -0.1% |

## Search winner (env path) vs the same rule through the plugin (A/B config arm)

| size | search winner busbw (median of 3) | plugin arm median | diff |
|---|---|---|---|
| 4096 | 0.38 | 0.36 | -5.3% |
| 8192 | 0.74 | 0.71 | -4.1% |
| 16384 | 1.43 | 1.41 | -1.4% |
| 32768 | 2.83 | 2.78 | -1.8% |
| 65536 | 5.43 | 5.40 | -0.6% |
| 131072 | 9.70 | 9.65 | -0.5% |
| 262144 | 18.65 | 18.50 | -0.8% |
| 524288 | 36.53 | 36.07 | -1.3% |
| 1048576 | 45.60 | 44.68 | -2.0% |
| 2097152 | 71.97 | 71.57 | -0.6% |
| 4194304 | 110.24 | 109.34 | -0.8% |
| 8388608 | 148.79 | 147.96 | -0.6% |
| 16777216 | 188.91 | 188.27 | -0.3% |
| 33554432 | 213.93 | 213.40 | -0.3% |
| 67108864 | 234.65 | 234.56 | -0.0% |
| 134217728 | 252.01 | 251.42 | -0.2% |
| 268435456 | 263.23 | 263.44 | +0.1% |
| 536870912 | 267.97 | 268.12 | +0.1% |

## Numbers that need a second look

- 8K picked 48 channels (executed 4). NOT the tie-break: ch48's three repeats were 0.76/0.72/0.74 (median 0.74) and ch8's 0.73/0.73/0.73 fell outside the 0.5% band (`all_reduce_1n/adaptive_report.json`), so ch48 won outright on one 0.76 sample. At <=16K the 0.5% tolerance is below the busbw print resolution (0.01 = 1.4% at 8K). Harmless here: the plugin arm executed TREE/LL/4 anyway.
- '9 reps/arm' = 9 launched; the medians use the cv-core-gated values, 7-8 at 7 sizes (n column in the table = n_def/n_cfg). Raw 9 are in the validate.log on node 8, not on this VM.
- 4K: search cell MIN=MAX=4 measured 0.38; the plugin rule ch4 executed 2 channels and measured 0.36 = the search's ch2 cell. The env path can force what the plugin path cannot reproduce at small sizes (reviewer's point).
- 512K: search winner 36.5 vs plugin arm 36.1, default 20.9 — a +73% window, same as the 09-07 grid found (+75.7%).

## By hand
See HANDS.md: launch line, log move, PLAN/HANDS/RUNLOG launch row, search_raw/ fetch (rsync of metrics/merged_exec/summary/command.txt only; dbg logs stay on node 8 /data/ylerman/rccl-tune-2026-09-08-sanity-check-1/), and this SUMMARY.
