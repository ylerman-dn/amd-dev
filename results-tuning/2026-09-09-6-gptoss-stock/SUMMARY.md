# SUMMARY — 2026-09-09-6-gptoss-stock: against STOCK SGLang the tuned RCCL path loses by 22% (stock is +28% over it)

Node amd-mi355x-ses2-1 (TEST, MI355X), job 21085, gpt-oss-120b, CUDA graphs ON, radix cache OFF, --attention-backend triton, d32
(ISL 512 / OSL 512 / conc 32 / 128 prompts), one server per arm, 6 reps each (rep 1 dropped), two passes in opposite order.
Scores: node/st1_score.csv, node/st2_score.csv (infer_score.py --ref stock_def).

| pass | arm | server env | median tok/s | vs stock | P(sup) | TTFT ms | TPOT ms |
|---|---|---|---|---|---|---|---|
| 1 | **stock** | custom all-reduce ON, RCCL_MSCCL_ENABLE=1, image NCCL_MIN_NCHANNELS=112 present, no plugin | 7309 | 0 | - | 118.7 | 4.14 |
| 1 | sc2 | custom AR off, MSCCL 0, flag removed, plugin + sc2 conf | 5702 | -22.0% | 0.00 | 125.1 | 5.37 |
| 1 | none | custom AR off, MSCCL 0, flag removed, no plugin | 4846 | -33.7% | 0.00 | 123.8 | 6.36 |
| 2 | stock | (same) | 7312 | 0 | - | 119.9 | 4.14 |
| 2 | sc2 | | 5735 | -21.6% | 0.00 | 123.5 | 5.34 |
| 2 | none | | 4831 | -33.9% | 0.00 | 124.3 | 6.38 |

Env receipt (node ses2-1 st1_stock_def/logs/rccl.*.log): `NCCL_MIN_NCHANNELS set by environment to 112`, `RCCL_MSCCL_ENABLE set by environment to 1`,
MSCCL initialised. sc2/none arms as in run 5 (flag absent, MSCCL 0); their numbers reproduce run 5 within 0.5%.

Reading:
- The +18% of runs 4/5 is real but only relative to a baseline that has SGLang's custom all-reduce and MSCCL switched off. Stock SGLang,
  which is what deploys, is 28% faster than the tuned RCCL path (pooled medians 7310 vs 5722 tok/s; TPOT 4.14 vs 5.35 ms) on this model, this traffic (d32), this node.
- The stock arm did not use the tuner plugin, so "does the conf help stock SGLang" is NOT answered here (run 7, 2026-09-09-7-gptoss-attr, adds stock+plugin).
  Whether the custom all-reduce takes the decode all_reduce away from RCCL is a hypothesis until run 7's TUNING counts are read; the custom kernel's size threshold is not in any local file.
  Stock differs from sc2 in three things at once (custom AR, MSCCL, channel flag); run 7 separates them. Open regardless: prefill sizes, other collectives
  (broadcast/reduce_scatter are always tuner-visible, HANDOVER 4), multi-node (custom AR is intra-node), other concurrencies, Qwen stock.
- Consistent with HANDOVER item 3 (2026-09-06: deployment env beats tuner-visible env by +1.9..+6.6% graphs-off); with graphs the gap is 27%.
- The stock arm's better TTFT (119 vs 124 ms) is within run-to-run noise of prefill.

Reviewer notes (REVIEW.md): numbers verified; +27.8% pooled; rep 1 is the low rep on all six servers here (drop-first correct); "no deployment value" was an overreach and is now stated as open.
