# PLAN — 2026-09-09-2-ab4m-settle: is "112 channels executed at 4M" bad, or was it the over-request (144)?
Open point from 2026-09-08-3-sanity-check-3 (rule ring,simple,144 at 4M: plugin executed 112, -73%). Reviewer's settling measurement.
Node: amd-mi355x-ses2-1 (TEST, MI355X), job in HANDS.md. Command (tool copy /opt/shared/ylerman/GPU-107/rccl-sweep-optuna, 3b56455 era):
  validate_tuner_config.py --config settle_4m_ring_simple_112.conf (ONE rule: allreduce,4194304,4194304,ring,simple,112,1,8,-1,-1)
    --binary .../bin/all_reduce_perf --plugin .../ab-tuner-test/librccl-tunerv4-dn.so --plugin-must-fire --runtime container
    --drop-env NCCL_MIN_NCHANNELS --repeats 9 --warmup-runs 4 --min-bytes 4096 --max-bytes 536870912 --iters 20 --warmup 5
    --preflight-scope rules --noise-gate cv-core --split-ranges --logdir /data/ylerman/ab4m-2026-09-09/ab
Same arms/gates as the sanity checks. Reading: cfg_median at 4M ~130 -> the 144 over-request tripped a pathology (clamp conf channels to the
  node max); ~35 -> 112 executed at 4M is itself the problem and RCCL's trim to 103 was protective. Also read cfg_exec (expect RING/SIMPLE/112).
Outputs: node ses2-1 /data/ylerman/ab4m-2026-09-09/{validate.log,ab/per_size_stats.csv}, .validated.csv next to the conf on /opt/shared. ETA 10 min.
