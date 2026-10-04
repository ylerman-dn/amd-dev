# PLAN — sanity-check-1 (2026-09-08): Process A end to end via rccl_tune.py, grid 1..48

Goal: prove `rccl_tune.py run` (restored 61107d0, fixed e46f3b6) produces a validated conf for
all_reduce 1 node with zero hand steps in the measurement chain.

Command (launched detached on the dev VM, log = rccl_tune.log in this dir):
  cd tools/rccl-sweep && setsid nohup /usr/bin/python3 rccl_tune.py run \
      --collectives all_reduce --scales 1 --name sanity-check-1 > .../rccl_tune.log 2>&1 &

What it does (tool eeb5aaa+7512283+c01b7bf+e46f3b6, synced to /opt/shared/ylerman/GPU-107/rccl-sweep-optuna):
  book    salloc -p XAI -N1 -w <idle healthy node> --gres=gpu:8 -t 180 -J ylerman-sanity-check-1
  search  adaptive_search.py live --grid 1,2,4,8,16,24,32,40,48 --anchors 1,8,24,48 --margin 15
          --tol 0.5 --repeat-policy 3 --default-runs 3 --jobid <id>   (each benchmark = srun step,
          rccl_sweep.py --runtime container per config; median of 3 repeats; 3 NCCL-default runs)
  config  generate_tuner_config.py <optimized.csv> --include-algo-proto --no-merge  (one rule per size)
  ab      ab_run.py -> validate_tuner_config.py --runtime container --drop-env NCCL_MIN_NCHANNELS
          --repeats 9 --plugin-must-fire --split-ranges --noise-gate cv-core --preflight-scope rules
          arms: default = image env minus NCCL_MIN_NCHANNELS, MSCCL=0, no plugin;
                config  = same + plugin librccl-tunerv4-dn.so + all_reduce_1n.conf
          gates: P(sup) >= 0.95, gain > 2%, regression < 2%
  record  RUNLOG row, results fetched to this dir, .validated.csv -> all_reduce_1n.final.conf; scancel.

Grid: all_reduce x 1 node x combos {RING/LL, RING/SIMPLE, TREE/LL} (1n honoured set, findings/02)
      x channels from the racing search over 1,2,4,8,16,24,32,40,48; sizes 4K..512M x2.
Env truth: container lmsysorg/sglang:v0.5.17-rocm720-mi35x, stock RCCL 2.27.7; search cells carry
      NCCL_MIN/MAX_NCHANNELS=<cell> NCCL_ALGO NCCL_PROTO RCCL_MSCCL_ENABLE=0 (image flag overridden);
      default runs carry no forcing (image flag 112 PRESENT - note for consistency check vs A/B default arm,
      which has the flag REMOVED). ETA: search ~45 min + A/B ~65 min.
Outputs: this dir: rccl_tune.log, all_reduce_1n/ (search: live_runs.log, adaptive_report.json,
      all_reduce_1n_optimized.csv), all_reduce_1n.conf, all_reduce_1n.validated.csv, ab_out/allreduce_1n/
      (validate.log, per_size_stats.csv), all_reduce_1n.final.conf. Remote raw: /opt/shared/ylerman/GPU-107/
      rccl-tune-2026-09-08-sanity-check-1/.
