# Communication
- Be extremely concise. Short answers, no summaries, no re-explaining.
- One step at a time. Propose → wait for my approval → execute. Never chain steps.
- Never declare previous results invalid without showing the log evidence first.

# Prior results
- Everything under `/home/dn/ylerman/tasks/GPU-107/` is **unverified**. Do not cite
  a number from it, do not build on it, do not carry its conclusions forward.
- If one of those results is needed, reproduce it from scratch.
- Archived under `/home/dn/ylerman/tasks/GPU-107/archive/`:
  `2026-07-12_28-pre-reset.tar.gz` (early harnesses and A/B attempts) and
  `2026-07-29_30-abv2-autonomous.tar.gz` (the overnight run and its conclusions).

# Directory discipline
- NEVER create new top-level directories, except `results-tuning/`.
- All sweep outputs go to `results-tuning/<date>-<short-name>/` only.
  (Not `results/` — that holds unrelated upstream content.)
- The tool is one flat directory: `tools/rccl-sweep/`. New scripts go there too —
  no subfolders. Prefer a flag on an existing script over a new script.
- Never commit run output (gitignored, but never `git add -f` it either).

# Evidence
- Every run keeps its raw stdout/stderr in that run's folder. No log, no result.
- Every claim about a result cites the log path it came from — both directions:
  "this is a win" and "this earlier result is wrong".

# Verification (mandatory)
- A run only "counts" if the RCCL (ROCm Collective Communications Library) log
  confirms which algo/proto/nchannels were actually selected. Requested is not
  the same as selected — RCCL substitutes silently.
- Never report busbw numbers from a run whose parameter selection wasn't verified.

# Cluster
- Partition `XAI`, nodes `amd-mi355x-1..9`, 8 GPUs per node. Skip node2
  (orchestrator) and node9 (no ssh key).
- Binaries (`all_reduce_perf` etc.) are at `/opt/shared/ylerman/GPU-107/bin`,
  which the sweep reads as `$MY_PATH`.
- `/opt/shared` is NFS on the cluster nodes and is **not** mounted on this dev VM.
  To see or move a file there, go through a cluster node.
- Check `squeue -p XAI` before booking. A co-tenant sharing the fabric makes
  multi-node numbers meaningless.
- A job that dies mid-run is usually `NODE_FAIL` (transient cluster infra), not
  preemption and not an idle timeout. Check `sacct -j <id>` before theorising.
- Allocate once at the start of the session and hold it for the whole session —
  not per sweep, not per phase. Release once at the end.
- Book the largest node count needed, then run smaller scales on a subset of the
  same nodes. Never release a node to re-book it.
- Pin which nodes each scale uses and keep it fixed across the session. A 2-node
  run on a different pair is not comparable to the previous one.
- Never `scancel -u dn` — `dn` is shared and it kills other people's jobs. Cancel
  by job id.

# results-tuning/RUNLOG.md
Append one row per run, at the time of the run:

  utc_start, duration_s, jobid, nodelist, n_nodes, command, non-default env vars,
  output path, one-line result

`nodelist` is the actual allocated nodes, not the count — it is how we later tell
whether two results are comparable. A run that is not in RUNLOG.md did not happen.
