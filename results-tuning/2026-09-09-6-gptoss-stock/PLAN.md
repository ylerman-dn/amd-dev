# PLAN — 2026-09-09-6-gptoss-stock: is the +18% real against STOCK SGLang? (reviewer's single most valuable extra arm)
All in-model arms so far run with --disable-custom-all-reduce and RCCL_MSCCL_ENABLE=0 so that the all_reduce reaches RCCL and the tuner.
Stock SGLang routes the decode all_reduce through its own custom kernel and keeps MSCCL on, and the image sets NCCL_MIN_NCHANNELS=112.
Node amd-mi355x-ses2-1 (TEST, MI355X), job 21085, gpt-oss-120b, CUDA graphs ON, radix cache OFF, --attention-backend triton, d32, 6 reps/server.
Arms (one server each; pass 1 stock,none,sc2; pass 2 sc2,none,stock):
  stock = custom all-reduce ON, MSCCL ON, image flag present, no plugin      (IM_KEEP_CUSTOM_AR=1 RM_MSCCL=1)
  none  = custom AR off, MSCCL off, flag removed, no plugin                  (as run 5)
  sc2   = none's env + plugin + sc2_grid112.final.conf                       (as run 5)
Chain: /opt/shared/ylerman/GPU-107/gptoss-stock-2026-09-09.chain.sh -> node /data/ylerman/gptoss-stock-2026-09-09/. Score: infer_score.py --prefix st1/st2 --ref stock_def.
Reading: sc2 >= stock -> the conf has deployment value; stock >= sc2 -> the win only exists against a crippled baseline.
