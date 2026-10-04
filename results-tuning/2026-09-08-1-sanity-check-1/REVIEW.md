# Sub-agent review of sanity-check-1 (2026-09-08 16:2xZ), verbatim, plus what was done about it

## Corrections applied to SUMMARY.md / LOG
- 8K "tie-break" explanation was wrong: ch48 won outright on one 0.76 repeat; ch8 (0.73 x3) fell outside the 0.5% band. Fixed.
- "9 reps/arm": medians use cv-core-gated values (7-8 at 7 sizes). Stated now.
- LOG said 65 search runs; live_runs.log has 63. Fixed.

## Accepted, parked for the tool (not changed mid-campaign so checks 1-3 stay comparable)
- rccl_tune should fetch exec truth (merged_exec.csv) for every search run, not only defaults; dbg logs of record live only on node /data.
- rccl_tune should write the RUNLOG row at LAUNCH and quote the exact typed command; today two rows per run.
- PLAN.md should be emitted from POLICY + ab_run's fixed validator flags (--warmup-runs 4, preflight 3, --iters 20 --warmup 5, --retries 4 were run but not listed).
- SUMMARY tables are scriptable from the three CSVs.
- per_size_stats.csv should keep raw repeats plus a gated flag.
- Search default runs run after all cells, not interleaved; search medians are raw-of-3 while A/B medians are gated; different parsers (metrics.csv vs stdout). Same env vars and MSCCL=0 in both stages verified by the reviewer.
- tol 0.5% is below busbw print resolution at <=16K.
- RING/SIMPLE was never evaluated at 16/32/64 ch (racing pruned it): RCCL's own 512K-2M default choice was not on the grid.
- Env path vs plugin path at small sizes (4K: MIN=MAX=4 -> 0.38 vs plugin ch4 executing 2 -> 0.36).
- PLAN default-run flag line changed between attempts 2 and 3 (PRESENT -> REMOVED) without a re-approval; user is away, recorded here.

## Review text
**1. Correctness**
1. Recomputed from per_size_stats.csv: 4K 0.33/0.36/+9.1%, 1M 38.09/44.68/+17.3%, 8M 201.5/147.955/-26.6% — match SUMMARY.
2. Search-default medians over search_raw/d000{0,1,2}/merged_exec.csv: 4K 0.34, 1M 37.90, 512M 389.99 and all 18 diffs — match.
3. Winner busbw = median of 3: 8K TREE/LL/48 = med(0.76,0.72,0.74)=0.74 (r0033-35), 4K TREE/LL/4 = 0.38 (r0048-50) — match optimized.csv.
4. Wrong explanation: SUMMARY said 8K->ch48 is a "tie-break on requested channels". adaptive_search.py:121-133 ties within tol and picks the FEWEST channels. At 8K best=0.74, thr=0.7363, ch8=0.73 is outside the band — ch48 won outright on one 0.76 repeat.
5. "9 reps/arm" — def_runs/cfg_runs hold 7-8 values at 7 sizes. validate_tuner_config.py:903-904 writes the MAD-core-gated values; raw 9 are not on local disk.
6. LOG.md 15:59Z said "65 runs"; live_runs.log has r0000-r0062 = 63.
**2. Methodology**
1. PLAN:12 matches rccl_tune.log:11 exactly.
2. PLAN omits validator params actually run (ab_run.log:1): --warmup-runs 4, preflight 3, --min/max-bytes, --iters 20 --warmup 5, --retries 4 (ab_run.py:32,95-99). Consistent with the search's -n 20 -w 5, but unlisted.
3. PLAN changed between attempt 2 and 3 on a measurement-relevant line (default-run flag PRESENT->REMOVED). No record of re-approval (METHODOLOGY §1).
4. Search default runs executed after all 63 cells, not interleaved; PLAN silent.
5. ETA "A/B ~65 min" vs 9 min actual.
**3. Consistency**
1. Sound on env vars: same 22 yaml vars, MSCCL=0 both, same drop mechanism, same node/alloc, 2 min apart.
2. Receipt gap: A/B leaves no command.txt or dbg log locally; validate.log does not print the docker line. ENV-subsys proof sits on node 8 /data. Executed channel counts do not prove flag-off.
3. Different statistics compared: search medians raw-of-3; A/B medians cv-core-gated. Different parsers (metrics.csv vs parse_busbw stdout).
4. Minor env delta: search has NCCL_DEBUG_SUBSYS=...GRAPH... plus TOPO/GRAPH dump files; A/B drops them. Init-time only.
**4. Hand steps, priority**
1. search_raw/ rsync fetched no merged_exec/dbg for r*; executed channels of the 63 search cells absent locally.
2. RUNLOG: tool row misquotes the command, no launch-time row -> duplicate hand row.
3. PLAN.md: emit from POLICY + ab_run flags.
4. SUMMARY tables scriptable.
5. Log move.
**5. Risks**
1. Env path vs plugin path at 4K (see above); optimized.csv nchannels is A_flag_ceiling.
2. tol 0.5% below print resolution <=16K.
3. RING/SIMPLE never evaluated at 16/32/64 (racing pruned).
4. Cell repeats back-to-back; A/B interleaved. Single node, single day.
5. 1M config arm gated to 7 values.
**6. Other**
1. Dbg logs only on node-local /data.
2. Two RUNLOG rows per run.
3. per_size_stats should carry raw repeats + gated flag.
4. final.conf merges 32K-128K after per-size validation; the merged range itself was not A/B'd.
5. Co-tenants present at launch; irrelevant at 1n.
