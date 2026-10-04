# SUMMARY — 2026-09-09-5-gptoss-cudagraph: gpt-oss-120b, CUDA graphs ON, one server per arm (amd-mi355x-ses2-1, job 21078)

**sc2 conf: +18% decode throughput vs RCCL default, reproduced in both passes (P(sup)=1.00, 5 warm reps each).**
Radix cache off, d32 traffic (ISL 512 / OSL 512 / conc 32 / 128 prompts), --attention-backend triton, image NCCL_MIN_NCHANNELS removed, MSCCL=0.
Scores: `node/cg1_score.csv`, `node/cg2_score.csv` (infer_score.py --ref none_def; rep 1 of each server dropped as cold start).

| pass | arm | median tok/s | vs none | P(sup) | median TTFT ms | median TPOT ms |
|---|---|---|---|---|---|---|
| 1 (none,sc2,dummy) | none (no plugin) | 4831 | 0 | - | 123.8 | 6.38 |
| 1 | sc2 | 5725 | +18.5% | 1.00 | 123.5 | 5.35 |
| 1 | dummy (ring,simple,1) | 2120 | -56.1% | 0.00 | 2120 | 10.97 |
| 2 (dummy,sc2,none) | none | 4848 | 0 | - | 124.1 | 6.36 |
| 2 | sc2 | 5720 | +18.0% | 1.00 | 123.8 | 5.35 |
| 2 | dummy | 2119 | -56.3% | 0.00 | 2128 | 10.96 |

Per rep (node/cg*_*/bench_rep*.log): passes agree within 0.4% on medians; rep 1 is the low rep here (-0.5..-1.0%). Full separation: sc2's minimum rep beats none's maximum rep in all 12 reps (P(sup)=1.00 saturated).
Detect (cgdet_sc2_tun, TUNING logs): the tuner is consulted at graph capture and in prefill (24,528 hits: 830K-968K chunk sizes -> tree,ll,64 rule,
plus the 184K decode size per captured batch size); decode replays the captured graph with the rule baked in.

Reading: the conf's tree,ll,64 rule at 128K-256K (decode all_reduce = 184,320 B for gpt-oss at conc 32) cuts TPOT 6.37 -> 5.35 ms.
Yesterday's "everything ties" came from measuring with `--disable-cuda-graph` (CPU-bound decode, TPOT 33 ms). Deployment runs with graphs.

## Caveats (reviewer, REVIEW.md)
- **Both arms run with SGLang's custom all-reduce DISABLED and MSCCL OFF** (so the all_reduce reaches RCCL/the tuner). Stock SGLang routes the decode all_reduce through its own kernel and keeps MSCCL on. The +18% is vs a crippled baseline until the stock arm is measured (run 2026-09-09-6-gptoss-stock).
- The 'graphs-off decode is CPU-bound' mechanism is an inference from the numbers (32 ms/step slack at graphs off; dummy2 adds 11 ms under graphs and 0 without), not a profiler finding.
- Magnitude check: rccl-tests 128K default TREE/LL/8 35 us vs tree/ll/64 21.5 us, x96 calls/step = 1.30 ms predicted vs 1.43 ms TPOT drop observed.
- TTFT/TPOT columns are medians of per-rep MEANS.
