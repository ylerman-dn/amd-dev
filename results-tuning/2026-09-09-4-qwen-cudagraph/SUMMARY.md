# SUMMARY — 2026-09-09-4-qwen-cudagraph: Qwen3-30B-A3B, CUDA graphs ON, one server per arm (amd-mi350x-ses2-1 = MI350X, job 21081)

**sc2 conf: +18% decode throughput vs RCCL default, both passes, P(sup)=1.00. The bad confs cost -46% and -69%.**
This is the deployment-realistic setting (SGLang default = graphs on). Every earlier in-model campaign ran `--disable-cuda-graph` (required by the
paired hot-reload design) and was CPU/launch-bound: TPOT 38-43 ms there vs 7.8-9.2 ms here, so all_reduce time could not show.
Radix cache off, d32 traffic (512/512/conc 32/128 prompts), image NCCL_MIN_NCHANNELS removed, MSCCL=0. Node is MI350X (only free node); the
comparison is within-node; absolute numbers are not comparable to the MI355X runs. Scores: node/cg1_score.csv, node/cg2_score.csv.

| pass | arm | median tok/s | vs none | P(sup) | median TTFT ms | median TPOT ms |
|---|---|---|---|---|---|---|
| 1 (none,sc2,dummy2,dummy) | none (no plugin) | 3371 | 0 | - | 152.5 | 9.20 |
| 1 | sc2 (8 rules 4K-2M) | 3970 | +17.8% | 1.00 | 151.2 | 7.77 |
| 1 | dummy (ring,simple,1) | 1800 | -46.6% | 0.00 | 2098 | 13.70 |
| 1 | dummy2 (tree,ll,1) | 1035 | -69.3% | 0.00 | 5318 | 20.54 |
| 2 (reverse order) | none | 3365 | 0 | - | 152.0 | 9.22 |
| 2 | sc2 | 3971 | +18.0% | 1.00 | 152.0 | 7.77 |
| 2 | dummy | 1802 | -46.5% | 0.00 | 2098 | 13.67 |
| 2 | dummy2 | 1029 | -69.4% | 0.00 | 5190 | 20.98 |

Per rep (node/cg*_*/bench_rep*.log): passes agree within 0.2% on medians. Rep 2, not rep 1, is the low rep on every Qwen server (-2.7..-4.5%, mean TTFT 240-310 ms vs 152): the scorer's drop-first rule removed the wrong rep; medians differ <0.2 tok/s either way. Full separation: sc2's minimum rep beats none's maximum rep in all 12 reps, so P(sup)=1.00 is saturated, not a fine measurement.
Detect (cgdet_sc2_tun / cgdet_dummy2_tun, TUNING logs): tuner consulted at graph capture (128K decode size per captured batch size) and in
prefill (850K-983K chunks, 2M); graph replay runs the captured collectives. sc2 detect: 3906 tok/s, TPOT 7.78; dummy2 detect: 1029, 20.24.

Why yesterday's campaign saw nothing and today's graphs-off runs still see nothing on TPOT (2026-09-09-1-qwen-modes: TPOT 41.0-41.7 for
all four arms while TTFT went 146 -> 5168 ms): with graphs off the decode step is dominated by CPU launch overhead on 8 ranks; the GPU
waits, and a slower all_reduce fills idle time. With graphs on the step is GPU-bound and the all_reduce is on the critical path.
Attempt 1 of this run failed (no Qwen snapshot on the MI350X node); model fast-copied from ses2-1 (61 GB, 50 s), relaunched (HANDS.md).

## Caveats (reviewer, REVIEW.md)
- **Both arms run with SGLang's custom all-reduce DISABLED and MSCCL OFF** (so the all_reduce reaches RCCL/the tuner). Stock SGLang routes the decode all_reduce through its own kernel and keeps MSCCL on. The +18% is vs a crippled baseline until the stock arm is measured (run 2026-09-09-6-gptoss-stock).
- The 'graphs-off decode is CPU-bound' mechanism is an inference from the numbers (32 ms/step slack at graphs off; dummy2 adds 11 ms under graphs and 0 without), not a profiler finding.
- Magnitude check: rccl-tests 128K default TREE/LL/8 35 us vs tree/ll/64 21.5 us, x96 calls/step = 1.30 ms predicted vs 1.43 ms TPOT drop observed.
- TTFT/TPOT columns are medians of per-rep MEANS.
