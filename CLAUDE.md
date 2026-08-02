# Communication
- Be extremely concise. Short answers, no summaries, no re-explaining.
- One step at a time. Propose → wait for my approval → execute. Never chain steps.
- Never declare previous results invalid without showing the log evidence first.

# Prior results
- Everything under `/home/dn/ylerman/tasks/GPU-107/` is **unverified**. Do not cite
  a number from it, do not build on it, do not carry its conclusions forward.
- If one of those results is needed, reproduce it from scratch.
- Archive of the 2026-07-29/30 run:
  `/home/dn/ylerman/tasks/GPU-107/archive/2026-07-29_30-abv2-autonomous.tar.gz`

# Directory discipline
- NEVER create new top-level directories, except `results-tuning/`.
- All sweep outputs go to `results-tuning/<date>-<short-name>/` only.
  (Not `results/` — that holds unrelated upstream content.)
- Driver/wrapper scripts live in `tools/rccl-sweep/scripts/` only. Prefer editing
  an existing script over creating a new one.
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
