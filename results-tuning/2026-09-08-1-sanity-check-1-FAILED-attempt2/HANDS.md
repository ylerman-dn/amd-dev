# By hand (not by script) — sanity-check-1 (attempt 2, the real one)
- Launch line itself (setsid nohup rccl_tune.py ...), typed on the dev VM at 15:22Z.
- rccl_tune.log started at results-tuning/ and moved into this dir after rccl_tune created it.
- PLAN.md (copied from attempt 1, tool hash updated), HANDS.md, RUNLOG launch row.
- Attempt 1 (15:19-15:21Z) is in ../2026-09-08-1-sanity-check-1-FAILED-attempt1/ - see its HANDS.md.
- Attempt 2 STOPPED by hand at 15:27Z (pkill rccl_tune + adaptive_search on the VM, `scancel 21056`).
  Benchmarks ran (rc=0, ~45s each) but produced NO metrics: output dir was on /opt/shared (NFS,
  root_squash) and the container (root) could not create NCCL_DEBUG_FILE there, so INFO went to
  stdout and interleaved every data row -> parser got nothing, metrics.csv never written
  (evidence: r0000/run_*/outputs/*/output.log has 18 rows all glued to NCCL INFO text, no dbg_*.log,
  summary.csv avg_busbw empty). Verified: `docker run -v /opt/shared/...:/workspace/out ... touch`
  -> Permission denied; same on /data -> OK. Fix: rccl_tune puts benchmark outputs under
  /data/ylerman/rccl-tune-<date>-<name> on the exec node (DATA_ROOT).
