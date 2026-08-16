# Communication
- Be extremely concise. Short answers, no summaries, no re-explaining.
- One step at a time. Propose → wait for my approval → execute. Never chain steps.
- Never declare previous results invalid without showing the log evidence first.
- **Never change a measurement parameter on your own.** Iteration counts (`-n`,
  `-w`), benchmark flags, selection tolerances, ranking metrics, sizes, channel
  lists, repeat counts — if it can move a number, you propose it and WAIT, even
  when the current value is obviously wrong and even when you have just proved it
  is wrong. Show me the evidence and the proposed change; do not apply it and do
  not relaunch anything on the back of it.
  (That rule exists because `-n 5`→`-n 20`, `-R`→`-C` and the `optimize_metrics.py`
  ranking metric were all changed unilaterally mid-task on 2026-08-04, each
  silently changing what every downstream number meant. A mid-flight redeploy of
  the tool directory also killed a running sweep with a stale file handle.)
- Assume I am present unless I have said I am leaving. Silence is not approval.
- **Never cite a bare line number.** Always name the file with it, as
  `tools/rccl-sweep/sweep_executor.py:190`, and give the full path when it is
  ambiguous which copy you mean (dev VM vs a frozen copy on `/opt/shared`). Same
  for config keys — say which file they live in.

# Prior results
- Everything under `/home/dn/ylerman/tasks/GPU-107/` is **unverified**. Do not cite
  a number from it, do not build on it, do not carry its conclusions forward.
- If one of those results is needed, reproduce it from scratch.
- Archived under `/home/dn/ylerman/tasks/GPU-107/archive/`:
  `2026-07-12_28-pre-reset.tar.gz` (early harnesses and A/B attempts) and
  `2026-07-29_30-abv2-autonomous.tar.gz` (the overnight run and its conclusions).

# Directory discipline
- NEVER create new top-level directories, except `results-tuning/` and `findings/`.
- All sweep outputs go to `results-tuning/<date>-<N>-<short-name>/` only, where `N`
  starts at 1 each day and increments per run — so the folders sort in run order.
  e.g. `2026-08-03-1-mcheck`, `2026-08-03-2-combomatrix`.
  (Not `results/` — that holds unrelated upstream content.)
- The tool is one flat directory: `tools/rccl-sweep/`. New scripts go there too —
  no subfolders. Prefer a flag on an existing script over a new script.
- Never commit run output (gitignored, but never `git add -f` it either).
- Always use absolute paths for file operations. The shell keeps its working
  directory between commands, so a relative path after an earlier `cd` silently
  writes to the wrong place — this already misfiled a batch of logs once.

# Evidence
- Every run keeps its raw stdout/stderr in that run's folder. No log, no result.
- Every claim about a result cites the log path it came from — both directions:
  "this is a win" and "this earlier result is wrong".

# findings/ — confirmed facts only
`findings/` is the curated set of things we are **sure** of, so a later session can
tell settled fact from working guess. One file per finding,
`findings/<NN>-<slug>.md`, indexed one line each in `findings/README.md`.

Each file states, in order: the claim in one sentence; the evidence — a log path
**and the exact command that reproduces the observation**; the date; the scope and
limits (node count, RCCL version); and what would falsify it.

Keep entries SHORT — the conclusion, a small snippet of the supporting data, and
the path. These are meant to be shown to someone, not read as a report. Show me the
draft and get approval before writing to `findings/`.

Entry criteria — this is the whole point of the folder:
- Goes in ONLY if backed by a log line we captured, or by a primary source
  (vendor docs, or source code we actually read).
- Inference does NOT qualify, however plausible — disassembly reasoning,
  docs-only deduction, and anything from a single unrepeated measurement stay out.
  Those belong in the run's `SUMMARY.md`, labelled as hypotheses.
- A format string found in a binary proves the mechanism EXISTS, not that it fired.
  Never cite one as if it were a log line.
- If a finding is later contradicted, delete or correct it — do not leave it to rot.

# Verification (mandatory)
- A run only "counts" if the RCCL (ROCm Collective Communications Library) log
  confirms which algo/proto/nchannels were actually selected. Requested is not
  the same as selected — RCCL substitutes silently.
- Never report busbw numbers from a run whose parameter selection wasn't verified.
- The flag is `-A 1` (`--output_algo_proto_channels`) on this build, NOT `-M 1`
  (that is `--memory_report` here). Upstream rccl-tests uses `-M`; ours moved it.
- `-A 1` is trustworthy for **algo/proto** but NOT for **nchannels** — its channel
  column reports the planned ceiling, not what ran. Verified 2026-08-03: it says 56
  where the real span was 52/54, and says 56 at >=64M where WarpSpeed actually ran
  222-224 channels. For real channel counts use `NCCL_DEBUG=INFO` and read
  `channel{Lo..Hi}` (count = Hi-Lo+1). Evidence:
  `results-tuning/2026-08-03-4-infocap/info.log`.
- `NCCL_DEBUG=INFO` interleaves with rccl-tests output on the same fd, so no data
  row survives as one intact line. Never parse INFO logs assuming whole-line rows.
  Use `-Z csv -X <file>` (rccl-tests' own CSV reporter) to get results on a
  separate stream, immune to interleaving.
- **Standard measured run**: `NCCL_DEBUG=INFO` with `NCCL_DEBUG_FILE=<path>_%p.log`
  (the `%p` stops 8 ranks sharing one descriptor). This carries per-run selection
  verification at no measurable timing cost — proven: mean -0.06% across 18 sizes,
  ranges overlap 18/18 (`results-tuning/2026-08-03-5-lognoise/`).
  Never INFO on stdout: that costs up to 4% at small sizes.
- Every run that will be compared must use the SAME logging config. Never compare a
  logged run against an unlogged one when busbw matters — the bias is size-dependent
  so it does not cancel, and it is the same order as the 5% tolerance
  `optimize_metrics.py` discriminates at.

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
- **`srun --export` uses commas as its OWN separator.** Any comma-containing value
  passed through it is silently truncated — `OMPI_MCA_btl=self,vader,tcp` becomes
  `btl=self`, `NCCL_IB_HCA=ionic_0:1,ionic_1:1,...` becomes one HCA. Always export
  such vars INSIDE a `bash -c` wrapper instead. This cost 5 failed multi-node
  attempts with misleading errors (btl_tcp connect fail → MPI_Init fail → btl_ofi
  abort) that never pointed at the real cause.
- Multi-node runs need the fabric env or they fail entirely: `NCCL_SOCKET_IFNAME`,
  `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, plus `OMPI_MCA_btl=self,vader,tcp` and
  `OMPI_MCA_btl_tcp_if_include=enp81s0f1np1`. Working reference:
  `results-tuning/2026-08-03-9-multinode/run_mn.sh`.
- Track allocation expiry. A `-t 240` alloc silently killed runs mid-batch once;
  `sacct` said `TIMEOUT`, not `NODE_FAIL`.
- Redirect every `srun`'s stdin from `/dev/null`, or it eats the driving heredoc.

# results-tuning/RUNLOG.md
Append one row per run, at the time of the run:

  utc_start, duration_s, jobid, nodelist, n_nodes, command, non-default env vars,
  output path, one-line result

`nodelist` is the actual allocated nodes, not the count — it is how we later tell
whether two results are comparable. A run that is not in RUNLOG.md did not happen.
