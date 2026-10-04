# SUMMARY — sanity-check-2 (2026-09-08): same one command, grid 1..112

Command: `rccl_tune.py run --collectives all_reduce --scales 1 --name sanity-check-2 --grid 1,2,4,8,12,16,24,32,40,48,56,64,72,80,84,96,104,112` (tool efa0117). Wall 3814 s = 64 min: search 53 min (120 runs + 3 default runs, 25.9% under the 162-run grid), A/B 9m39s, node 8, job 21060, released by the tool.

Verdict: **10 of 18 sizes KEPT, all P(sup)=1.00** (4K..2M, +12..+74%); 8 DROPPED at >=4M. Of those, 16M..512M are true parity (same executed config as default, -0.0..-0.3%), 8M +0.2% (P 0.71), and **4M is a real -1.1% loss** (all 9 config values below all 9 default values, P(sup)=0.00): the search picked RING/SIMPLE/72 -> executed 64 where default executes 103. `all_reduce_1n.final.conf` = 8 rules.

Vs the 2026-09-07 flag-off A/B (`2026-09-07-1-fullgrid/ab_v4_noflag/`): same executed config at 14/18 sizes, config-arm medians within 1%; but 4M was KEPT yesterday (RING/LL/96 -> exec 94, +4.1%) and dropped today: check 2's search never ran RING/LL above 80 channels (`all_reduce_1n/live_runs.log`), the hill-climb stopped on the 72~80 plateau, so the 4M win on the grid was unreachable by construction (reviewer, REVIEW.md §4).

## Per size: default vs ours (`ab_out/allreduce_1n/per_size_stats.csv`; requested = `all_reduce_1n.conf`; search-time executed from `search_raw/r*/merged_exec.csv`)

| size | default executed | ours requested | executed in search | executed in A/B | default med | ours med | gain | P(sup) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | TREE/LL/1 | tree/ll/2 | 2 | TREE/LL/2 | 0.33 | 0.37 | +12.1% | 1.0 | KEPT |
| 8192 | TREE/LL/1 | tree/ll/48 | 4 | TREE/LL/4 | 0.56 | 0.71 | +26.8% | 1.0 | KEPT |
| 16384 | TREE/LL/1 | tree/ll/16 | 8 | TREE/LL/8 | 0.89 | 1.39 | +56.2% | 1.0 | KEPT |
| 32768 | TREE/LL/2 | tree/ll/48 | 16 | TREE/LL/16 | 1.75 | 2.77 | +58.3% | 1.0 | KEPT |
| 65536 | TREE/LL/4 | tree/ll/32 | 32 | TREE/LL/32 | 3.35 | 5.35 | +59.7% | 1.0 | KEPT |
| 131072 | TREE/LL/8 | tree/ll/64 | 64 | TREE/LL/64 | 6.60 | 10.62 | +60.9% | 1.0 | KEPT |
| 262144 | TREE/LL/16 | tree/ll/64 | 64 | TREE/LL/64 | 13.04 | 18.83 | +44.4% | 1.0 | KEPT |
| 524288 | RING/SIMPLE/16 | ring/ll/64 | 64 | RING/LL/64 | 20.94 | 36.46 | +74.1% | 1.0 | KEPT |
| 1048576 | RING/SIMPLE/32 | ring/ll/64 | 64 | RING/LL/64 | 38.29 | 60.58 | +58.2% | 1.0 | KEPT |
| 2097152 | RING/SIMPLE/64 | ring/ll/72 | 69 | RING/LL/69 | 77.41 | 88.47 | +14.3% | 1.0 | KEPT |
| 4194304 | RING/SIMPLE/103 | ring/simple/72 | 64 | RING/SIMPLE/64 | 130.19 | 128.77 | -1.1% | 0.0 | dropped |
| 8388608 | RING/SIMPLE/103 | ring/simple/96 | 94 | RING/SIMPLE/94 | 201.75 | 202.13 | +0.2% | 0.7099 | dropped |
| 16777216 | RING/SIMPLE/108 | ring/simple/112 | 108 | RING/SIMPLE/108 | 273.92 | 272.97 | -0.3% | 0.0278 | dropped |
| 33554432 | RING/SIMPLE/111 | ring/simple/112 | 111 | RING/SIMPLE/111 | 322.72 | 322.03 | -0.2% | 0.0494 | dropped |
| 67108864 | RING/SIMPLE/111 | ring/simple/112 | 111 | RING/SIMPLE/111 | 359.39 | 358.25 | -0.3% | 0.1032 | dropped |
| 134217728 | RING/SIMPLE/112 | ring/simple/112 | 112 | RING/SIMPLE/112 | 376.25 | 375.86 | -0.1% | 0.2099 | dropped |
| 268435456 | RING/SIMPLE/112 | ring/simple/112 | 112 | RING/SIMPLE/112 | 385.54 | 385.28 | -0.1% | 0.321 | dropped |
| 536870912 | RING/SIMPLE/112 | ring/simple/112 | 112 | RING/SIMPLE/112 | 389.93 | 389.76 | -0.0% | 0.2469 | dropped |

## Consistency of the default numbers

Three independent 'RCCL default' measurements, all flag-off, node 8: check-2 search default runs (3), check-2 A/B default arm (9), check-1 A/B default arm (9, one hour earlier).

| size | c2 search def | c2 A/B def | c1 A/B def | max spread |
|---|---|---|---|---|
| 4096 | 0.33 | 0.33 | 0.33 | 0.0% |
| 8192 | 0.57 | 0.56 | 0.57 | 1.8% |
| 16384 | 0.89 | 0.89 | 0.89 | 0.0% |
| 32768 | 1.76 | 1.75 | 1.76 | 0.6% |
| 65536 | 3.34 | 3.35 | 3.30 | 1.5% |
| 131072 | 6.58 | 6.60 | 6.56 | 0.6% |
| 262144 | 13.07 | 13.04 | 13.07 | 0.2% |
| 524288 | 20.91 | 20.94 | 20.87 | 0.3% |
| 1048576 | 37.95 | 38.29 | 38.09 | 0.9% |
| 2097152 | 76.55 | 77.41 | 77.53 | 1.3% |
| 4194304 | 129.87 | 130.19 | 130.22 | 0.3% |
| 8388608 | 201.22 | 201.75 | 201.50 | 0.3% |
| 16777216 | 273.57 | 273.92 | 274.15 | 0.2% |
| 33554432 | 322.80 | 322.72 | 322.94 | 0.1% |
| 67108864 | 358.85 | 359.39 | 358.94 | 0.2% |
| 134217728 | 376.05 | 376.25 | 376.38 | 0.1% |
| 268435456 | 385.70 | 385.54 | 385.63 | 0.0% |
| 536870912 | 389.74 | 389.93 | 390.22 | 0.1% |

## Notes

- Requested vs executed: 8K requests 48 and executes 4; 32K requests 48, executes 16; 2M requests 72, executes 69. Search and A/B agree everywhere (columns 4 and 5), so the env-var path and the plugin path trim alike here.

- 4K: the search picked ch2 this time (check 1 picked ch4; both execute 2). Tolerance 0.5% is below the print resolution at 4K; the conf carries whichever tied sample won.

- Vs check 1 (grid cap 48): 4K..64K and 512K agree within a few points (512K +74.1% vs +72.8%). 128K +60.9% vs +47.1% and 1M +58.2% vs +17.3%: check 1's 40-channel winners were grid-capped (executed 32/40 vs 64 now). 256K switched algo (RING/LL/32 -> TREE/LL/64).
- Null control: at 16M/32M/64M both arms executed the identical config yet the config arm reads -0.2..-0.35% with P(sup) 0.03/0.05/0.10 (yesterday 16M P=0.00). A possible ~0.3% penalty on the config arm (plugin path or arm order); NOT reproduced in check 3 (same-exec sizes P 0.23-0.47 there), so treat as unconfirmed.
- Tolerance 0.5% is below 3-repeat noise at every size, not only print resolution: 1M winner raw 59.91/61.64/61.65; 2M 89.93/91.74/93.07 (`search_raw/`).

## By hand
See HANDS.md.
