# PLAN — 2026-09-09-1-qwen-modes: why is Qwen decode unaffected by a bad conf? measure the right thing (radix cache OFF), 4 traffic modes
User (09:xxZ, leaving 3-4 h): "proceed, don't wait for approvals; find explanations for the weird behaviour; another version of the config;
make sure you measure the right thing (radix cache); decode-heavy, prefill-heavy and in between; take a TEST node".

Node: amd-mi355x-des2-2 (TEST partition, MI355X, Qwen3-30B-A3B local), job 21076 (-t 300). XAI is fully allocated by another user today.
Server (infer_paired.sh): TP-8, RCCL_MSCCL_ENABLE=0, --disable-custom-all-reduce, image NCCL_MIN_NCHANNELS REMOVED (IM_DROP_FLOOR=1),
  hot-reload plugin, **--disable-radix-cache** (IM_EXTRA) in EVERY mode so every round pays real prefill.
Arms (same server per mode, interleaved per round, 6 rounds):
  sc2    = 2026-09-08-2-sanity-check-2 final.conf (8 rules 4K-2M; 128K -> tree/ll/64)
  dummy  = ring,simple,1 channel all sizes (yesterday's bad conf)
  dummy2 = tree,ll,1 channel all sizes (NEW: worse at 128K per rccl-tests, tree/ll/1 = default's own combo at 1 channel)
  none   = zero rules = RCCL default
Modes (ISL/OSL/conc/prompts, from infer_modes.sh table): d32 512/512/32/128 · p8k 8192/128/32/96 · d128 512/512/128/384 · mix 2048/512/64/192.
Per mode: detect run (1 round, TUNING logs, arms sc2 + dummy2) to read the all_reduce sizes and rule hits under radix-off traffic,
  then the 6-round quiet campaign. One chained srun step: `for m in d32 p8k d128 mix; do infer_paired.sh <qwen> qm_${m}can 1 sc2:.. dummy2:..; infer_paired.sh <qwen> qm_$m 6 sc2:.. dummy:.. dummy2:.. none:NONE; done`
Score: infer_score.py --base node/data/ylerman/qwen-modes-2026-09-09 --prefix qm_<mode> --ref none (also report per-round MEAN and MEDIAN TTFT).
Expected outcomes that would explain yesterday: (a) with radix off, TTFT separates dummy/dummy2 from sc2/none in every mode (prefill is where
  all_reduce matters); (b) decode TPOT still ties even for dummy2 -> all_reduce is off the decode critical path at these sizes (to be stated as
  the finding, with the numbers); (c) d128 (512K decode size) shows a decode effect -> size threshold.
Outputs: node des2-2 /data/ylerman/qwen-modes-2026-09-09/{qm_*,paired_qm_*.log}; driver log /opt/shared/ylerman/GPU-107/qwen-modes-2026-09-09.driver.log;
  fetched (bench logs only) to this dir. ETA ~3.5 h.
