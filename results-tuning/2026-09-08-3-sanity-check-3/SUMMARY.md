# SUMMARY — sanity-check-3 (2026-09-08): same one command, grid 1..224 (25 values)

Command: `rccl_tune.py run --collectives all_reduce --scales 1 --name sanity-check-3 --grid 1,...,112,128,144,160,176,192,208,224` (tool efa0117). Wall 3847 s = 64 min: search 54 min (123 runs + 3 default runs, 45.3% under the 225-run grid), A/B 9m38s, node 8, job 21061, released by the tool.

Verdict: **10 of 18 sizes KEPT, all P(sup)=1.00** (4K..2M, +9..+75%, same set as check 2); 8 dropped at >=4M: 8M..512M parity (-0.5..+0.6%), and **4M a -73% landmine**: the search's winner requested 144 channels, and the tuner plugin executed 112 with throughput collapsing 129.9 -> 34.8 GB/s. `all_reduce_1n.final.conf` = 8 rules, identical structure to check 2's (channel labels differ at 4K/8K/16K/32K/64K where they execute the same counts).

## What >112 channels did (the point of this check)

- Env path (search): requesting 128 or 144 executed **128** channels at every size >=1M (`search_raw/r0117..r0122/merged_exec.csv`). So the env cap can exceed RCCL's own 112, up to 128; 144 is silently 128. And 128 executed is WORSE than the default at 8M-64M in the same runs: 8M 179.6 vs 201.4 (-11%), 16M 215.2 vs 273.2 (-21%), 32M 234.4 vs 322.4 (-27%), 64M 247.9 vs 359.2 (-31%); only 2M/4M gained ~+2%. Values 160..224 were never run: the climb moves the incumbent only on a >2% gain (`adaptive_search.py` improve_eps 2.0, hard-coded, not in POLICY); 128/112 = +2.25% at 4M made 128 the incumbent, 144/128 = +0.5% did not, so the search stopped at 144.
- Plugin path (A/B): the rule `ring,simple,144` at 4M executed **112** (`per_size_stats.csv` row 4M: cfg_applied_ch 144, cfg_exec RING/SIMPLE/112) and measured 34.79 vs default 129.89 (-73.2%, P(sup)=0.00). The two paths diverge above 112: env MIN/MAX creates 128 channels at init, the plugin can only pick among the 112 that exist. Whether '112 executed at 4M' is itself bad, or the over-request (144 > 112) trips a different RCCL branch, is NOT settled: nothing else measured 112 executed at 4M (env req 112 was trimmed to 103 -> 130.1). One measurement settles it: A/B the same conf with the 4M rule set to `ring,simple,112` (~130 = over-request pathology, clamp conf channels to the node max; ~35 = 112 at 4M is the problem). Proposed, not run. The validator's landmine gate dropped the rule either way.
- Search-stage 4M numbers, RING/SIMPLE by requested channels (median of runs, executed count):
  req 1->1: 6.5; req 8->8: 37.1; req 24->24: 77.1; req 40->40: 97.2; req 48->47: 109.7; req 56->52: 110.5; req 64->64: 130.0; req 72->64: 129.9; req 80->74: 129.8; req 84->74: 129.7; req 96->86: 130.6; req 104->103: 129.8; req 112->103: 130.1; req 128->128: 133.0; req 144->128: 133.7

## Per size: default vs ours (`ab_out/allreduce_1n/per_size_stats.csv`; requested from `all_reduce_1n.conf`; search-time executed from `search_raw/r*/merged_exec.csv`)

| size | default executed | ours requested | executed in search | executed in A/B | default med | ours med | gain | P(sup) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 4096 | TREE/LL/1 | tree/ll/32 | 2 | TREE/LL/2 | 0.34 | 0.37 | +8.8% | 1.0 | KEPT |
| 8192 | TREE/LL/1 | tree/ll/64 | 4 | TREE/LL/4 | 0.56 | 0.73 | +30.4% | 1.0 | KEPT |
| 16384 | TREE/LL/1 | tree/ll/72 | 8 | TREE/LL/8 | 0.89 | 1.42 | +59.6% | 1.0 | KEPT |
| 32768 | TREE/LL/2 | tree/ll/32 | 16 | TREE/LL/16 | 1.74 | 2.79 | +60.3% | 1.0 | KEPT |
| 65536 | TREE/LL/4 | tree/ll/40 | 32 | TREE/LL/32 | 3.34 | 5.36 | +60.5% | 1.0 | KEPT |
| 131072 | TREE/LL/8 | tree/ll/64 | 64 | TREE/LL/64 | 6.56 | 10.65 | +62.3% | 1.0 | KEPT |
| 262144 | TREE/LL/16 | tree/ll/64 | 64 | TREE/LL/64 | 13.05 | 18.71 | +43.4% | 1.0 | KEPT |
| 524288 | RING/SIMPLE/16 | ring/ll/64 | 64 | RING/LL/64 | 20.76 | 36.42 | +75.4% | 1.0 | KEPT |
| 1048576 | RING/SIMPLE/32 | ring/ll/64 | 64 | RING/LL/64 | 37.65 | 60.23 | +60.0% | 1.0 | KEPT |
| 2097152 | RING/SIMPLE/64 | ring/ll/72 | 69 | RING/LL/69 | 77.39 | 89.68 | +15.9% | 1.0 | KEPT |
| 4194304 | RING/SIMPLE/103 | ring/simple/144 | 128 | RING/SIMPLE/112 | 129.89 | 34.79 | -73.2% | 0.0 | dropped |
| 8388608 | RING/SIMPLE/103 | ring/simple/96 | 94 | RING/SIMPLE/94 | 200.95 | 202.18 | +0.6% | 0.8642 | dropped |
| 16777216 | RING/SIMPLE/108 | ring/simple/104 | 103 | RING/SIMPLE/103 | 273.30 | 272.06 | -0.5% | 0.0 | dropped |
| 33554432 | RING/SIMPLE/111 | ring/simple/112 | 111 | RING/SIMPLE/111 | 322.94 | 322.27 | -0.2% | 0.3951 | dropped |
| 67108864 | RING/SIMPLE/111 | ring/simple/112 | 111 | RING/SIMPLE/111 | 359.65 | 358.97 | -0.2% | 0.2284 | dropped |
| 134217728 | RING/SIMPLE/112 | ring/simple/112 | 112 | RING/SIMPLE/112 | 376.53 | 375.76 | -0.2% | 0.3951 | dropped |
| 268435456 | RING/SIMPLE/112 | ring/simple/112 | 112 | RING/SIMPLE/112 | 385.70 | 385.49 | -0.1% | 0.3704 | dropped |
| 536870912 | RING/SIMPLE/112 | ring/simple/112 | 112 | RING/SIMPLE/112 | 390.56 | 390.43 | -0.0% | 0.4683 | dropped |

## Consistency of the default numbers (all flag-off, node 8)

| size | c3 search def (3) | c3 A/B def (9) | c2 A/B def | c1 A/B def | max spread |
|---|---|---|---|---|---|
| 4096 | 0.33 | 0.34 | 0.33 | 0.33 | 3.0% |
| 8192 | 0.56 | 0.56 | 0.56 | 0.57 | 1.8% |
| 16384 | 0.89 | 0.89 | 0.89 | 0.89 | 0.0% |
| 32768 | 1.73 | 1.74 | 1.75 | 1.76 | 1.7% |
| 65536 | 3.32 | 3.34 | 3.35 | 3.30 | 1.5% |
| 131072 | 6.58 | 6.56 | 6.60 | 6.56 | 0.6% |
| 262144 | 13.07 | 13.05 | 13.04 | 13.07 | 0.2% |
| 524288 | 20.92 | 20.76 | 20.94 | 20.87 | 0.9% |
| 1048576 | 37.46 | 37.65 | 38.29 | 38.09 | 2.2% |
| 2097152 | 78.06 | 77.39 | 77.41 | 77.53 | 0.9% |
| 4194304 | 130.31 | 129.89 | 130.19 | 130.22 | 0.3% |
| 8388608 | 201.88 | 200.95 | 201.75 | 201.50 | 0.5% |
| 16777216 | 273.15 | 273.30 | 273.92 | 274.15 | 0.4% |
| 33554432 | 322.69 | 322.94 | 322.72 | 322.94 | 0.1% |
| 67108864 | 359.56 | 359.65 | 359.39 | 358.94 | 0.2% |
| 134217728 | 376.03 | 376.53 | 376.25 | 376.38 | 0.1% |
| 268435456 | 385.63 | 385.70 | 385.54 | 385.63 | 0.0% |
| 536870912 | 390.11 | 390.56 | 389.93 | 390.22 | 0.2% |

## Same-exec null test (config arm executes the same config as default)

- 33554432: -0.21%, P(sup) 0.3951
- 67108864: -0.19%, P(sup) 0.2284
- 134217728: -0.20%, P(sup) 0.3951
- 268435456: -0.05%, P(sup) 0.3704
- 536870912: -0.03%, P(sup) 0.4683

## Notes

- RING/LL was never run above 72 channels at 1M-4M in this search either (72/64 = +1.7% at 2M, below the 2% move threshold) (`all_reduce_1n/live_runs.log`), so the 4M RING/LL win yesterday's full grid found (+4.1%) stays unreachable for the racing search. Parked tool item (REVIEW of check 2).

- Small-size labels moved again (4K ch32, 8K ch64, 16K ch72 vs check 2's 2/48/16) while executing the same 2/4/8: the 0.5% band at <=16K is below print resolution, so the label is a lottery among identical measurements. Conf semantics unaffected (same executed count), but the conf file text differs between checks.

## By hand
See HANDS.md.
