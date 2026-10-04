# By hand (not by script) — sanity-check-1
- Launch line itself (setsid nohup rccl_tune.py ...), typed on the dev VM.
- rccl_tune.log started at results-tuning/ and was moved into this dir after rccl_tune created it
  (rccl_tune creates the dir; a pre-created dir would have shifted the -N- counter).
- PLAN.md, HANDS.md, RUNLOG launch row (rccl_tune writes its own row only at the END).
- Attempt 1 FAILED at 15:21Z: every benchmark srun step rc=1 in 2s. Cause: rccl_tune booked node 5
  (no tabulate/yaml on host python) and picked node 7 as exec node (its old ssh-era fallback);
  with the new `srun --jobid` launch a non-booked exec node is rejected by Slurm. Alloc 21055 released
  by rccl_tune itself. Dir renamed *-FAILED-attempt1 by hand; fix = book only nodes with the deps,
  exec node must be booked (rccl_tune.py pick_nodes/pick_exec_node).
