# PLAN — 2026-09-09-7-gptoss-attr: which of custom-AR / MSCCL / flag carries stock's +28%, and does the conf add anything on top of stock
Reviewer of run 6 (2026-09-09-6-gptoss-stock/REVIEW.md): arithmetic points at the custom all-reduce (~5-8 us/call); two cheap arms attribute it; a third answers deployment value.
Node amd-mi355x-ses2-1 (TEST, MI355X), job 21086
21086, gpt-oss-120b, CUDA graphs ON, radix OFF, --attention-backend triton, d32, 6 reps/server (rep 1 dropped), one pass:
  at_customar  = custom AR ON, MSCCL 0, image flag REMOVED, no plugin
  at_msccl     = custom AR off, MSCCL ON, flag PRESENT, no plugin      (= the 2026-09-06 "deployment env" arm, now with graphs)
  at_stocksc2  = STOCK (custom AR on, MSCCL on, flag present) + plugin with sc2 conf, TUNING logs (rule hits show whether the decode all_reduce reaches the tuner)
  at_stock     = stock again, same hour, as reference
Chain: /opt/shared/ylerman/GPU-107/gptoss-attr-2026-09-09.chain.sh; outputs node ses2-1 /data/ylerman/gptoss-attr-2026-09-09/. Score: infer_score.py --prefix at --ref stock_def.
Compare with run 6 (stock 7310, sc2 5720, none 4840).
