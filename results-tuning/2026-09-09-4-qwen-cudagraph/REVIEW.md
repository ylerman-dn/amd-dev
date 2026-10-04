# Sub-agent review of the graphs-on runs (-4-, -5-) and the 4M settle (-2-), 2026-09-09 ~09:1xZ; verbatim below, actions on top

## Applied
- SUMMARY -4-/-5-: spread claims corrected (Qwen rep 2 is the low rep, up to -4.5%; separation is min>max in all 12 reps); caveats added (custom AR off + MSCCL off baseline; CPU-bound mechanism is inference; magnitude check vs rccl-tests).
- SUMMARY -2-: "not trimmed" made precise (capped to 112, not trimmed to 103); the alternative hypothesis kept open.
- RUNLOG rows for -1- and -2-: guessed times replaced by real UTC.
- Launched the reviewer's "single most valuable extra arm": stock SGLang baseline (custom AR on, MSCCL on, flag present) vs none vs sc2, graphs on, gpt-oss -> 2026-09-09-6-gptoss-stock.
- Not done: infer_score's drop-first rule (should detect the cold rep, label mean-of-rep columns) - parked with yesterday's scorer items; run dirs are gitignored by CLAUDE.md, so only RUNLOG rows are committed (reviewer 5.4 conflicts with the repo rule).

## Review text
**1. Recompute (bench_rep2..6, scorer drop-first=1)**
1. Qwen cg1: none med 3371.16, sc2 3970.42 -> +17.78%; TPOT 9.20 vs 7.77 (-15.5%); dummy 1799.65 (-46.6%); dummy2 1035.10 (-69.3%). All match `2026-09-09-4-qwen-cudagraph/node/cg1_score.csv` and SUMMARY.
2. gpt-oss cg1: none 4831.29, sc2 5725.20 -> +18.50%; TPOT 6.38 vs 5.35 (-16.1%); dummy 2119.64 (-56.1%). Match `2026-09-09-5-gptoss-cudagraph/node/cg1_score.csv`.
3. P(sup)=1.00 is trivially saturated: sc2 min > none max over all 12 reps in both runs (Qwen 3790.8 > 3371.9; gpt-oss 5669.6 > 4853.4). The 5 "pairs" are rep indices of different servers, not same-minute pairs.
4. Warm-up drop is wrong-way for Qwen: rep 2 is the low rep in 8/8 servers (none -3.9%, sc2 -4.5% in cg1; -2.7/-3.0% in cg2, with mean TTFT 240-310 vs 152 ms); rep 1 is only -1.2..-1.5%. gpt-oss: rep 1 is the low one (-0.5..-1.0%), rep 2 normal. Medians unaffected (med6 vs med5 differ <0.2 tok/s), but SUMMARY's "rep-to-rep spread <0.2%" is false: Qwen spread over reps 2-6 is 4.8% (sc2 cg1), 4.1% (none cg1).
5. "median TTFT/TPOT" columns are medians of per-rep MEANs (same mislabel flagged in `2026-09-08-4-qwen/REVIEW.md` A4, still open).

**2. Is "+18% decode from the conf" sound?**
1. Direction: yes, robust — two fresh servers per arm, reverse order, full separation, two models, two GPU types.
2. Magnitude is consistent with rccl-tests: 128K default TREE/LL/8 6.56 GB/s = 35 us, sc2 tree/ll/64 10.65 = 21.5 us; x96 calls/step = 1.30 ms predicted vs 1.43 ms observed TPOT drop (`2026-09-08-3-sanity-check-3/ab_out/allreduce_1n/per_size_stats.csv` row 131072; call count from REVIEW.md).
3. Biggest confounder, not in any SUMMARY: both arms run `--disable-custom-all-reduce` and `RCCL_MSCCL_ENABLE=0` (`tools/rccl-sweep/infer_many.sh:41,69`). Stock SGLang routes the 128K decode all_reduce through its custom AR kernel and never consults RCCL/the tuner. "Deployment-realistic" is overstated; the baseline beaten is a crippled one.
4. Plugin-vs-no-plugin: under graphs the tuner runs at capture only, so plugin overhead cannot sit in the decode loop; negligible.
5. Verification gap: no repo log shows what RCCL default actually selected at 128K in-model for the none arm (CLAUDE.md "requested != selected"). The sc2 detect TUNING counts (68k / 24,528 hits) are on the node only; not fetched.
6. Other: conc 32 only (rule size 128K/184K is exactly conc-determined); MI350X for Qwen (mitigated by gpt-oss on MI355X); radix off is fine (makes prefill real).
7. Single most valuable extra arm: `none` with `IM_KEEP_CUSTOM_AR=1` (stock SGLang, graphs on). If stock >= sc2, the conf has no deployment value.

**3. "Graphs-off decode is CPU-bound"**
1. Numbers are consistent: slack 41.0-9.2 = 32 ms/step; dummy2 adds 11.3 ms under graphs (20.5 vs 9.2) -> still hidden graphs-off (41.69 vs 41.03, `qm_d32_score.csv`). Same for dummy (+4.5 ms).
2. Not directly tested: no GPU-timeline/profiler evidence; no graphs-off conf slow enough to exceed the 32 ms slack. Inference, not a finding.
3. Alternatives: (a) RCCL under hipGraph capture may take a different launch/protocol path, so the graphs-on cost of bad confs could be inflated, not merely "unhidden"; (b) 4.5x TPOT gain from graphs alone is large — worth confirming the graphs-off server is not also limited by something else (scheduler, 8-process python).

**4. C: over-request as cause**
1. Supported: 112 requested -> applied 112, executed 103, +0.3% P 0.74 (`2026-09-09-2-ab4m-settle/per_size_stats.csv` row 4194304) falsifies "112 at 4M is inherently bad as a request".
2. Not established: mechanism. Yesterday's row says `ch_trimmed=yes` (144->112), so it WAS trimmed to the comm count; SUMMARY's "not trimmed" is imprecise. Untested alternative: executing 112 channels at 4M is itself pathological, and 144 is just the only request that produced it. Env path executing 128 at 4M gave 131 GB/s (`2026-09-08-3-sanity-check-3/search_raw/r0117..r0122/merged_exec.csv`) — argues against "many channels at 4M is slow", but that comm had 128 channels built.
3. Also assumed: different node (amd-mi355x-8 vs ses2-1), 18-rule conf vs 1-rule conf, one run each; RCCL source not read (SUMMARY admits). Cited dbg lines are on the node only.

**5. Contradictions**
1. RUNLOG rows 449-450 carry guessed times (09:5xZ->10:2xZ) while rows 451-453 use real UTC (07:15Z..); row 450 (ab4m) sorts after row 451 though it ran first. LOG bullets also mix guessed and real times (acknowledged at "07:40Z (real UTC)").
2. SUMMARY "spread <0.2% / <0.3%" vs logs (Qwen 4.8%).
3. LOG "Puzzle closed" / SUMMARY state the CPU-bound mechanism as fact; PLAN calls it a hypothesis. No SUMMARY.md exists for -1- and -3- though LOG cites their numbers.
4. All five 2026-09-09 run dirs are untracked (`git status`), but their RUNLOG rows are committed (7618945) — violates "run + RUNLOG row commit-at-once".
5. Yesterday's SUMMARY still says "sc confs are safe" / decode null "unexplained"; not updated to point at the graphs-on result.

**6. Tell the user first**
1. sc2 beats RCCL default on decode under graphs: +17.8/18.0% Qwen, +18.5/18.0% gpt-oss, min>max separation in all 24 reps — but only with custom AR and MSCCL disabled; stock-SGLang baseline unmeasured.
2. Scorer drops the wrong rep on Qwen (rep 2 low in 8/8 servers, up to -4.5%); results survive, SUMMARY spread claims don't.
3. 4M: the 112 rule is parity (+0.3%); "over-request is the cause" is the best of two live hypotheses, not settled.
