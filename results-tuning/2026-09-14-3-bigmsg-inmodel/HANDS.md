# By hand — 2026-09-14-3-bigmsg-inmodel (campaign B')
- 15:58-16:06Z node 6 went idle; the launcher booked it five times (21248-21252) and released it each time: its salloc parser matched "allocation NNNN" twice, the job id carried a newline,
  every srun --jobid failed with "Requested node configuration is not available" -> "nothing launchable". Fixed (head -1), restaged as launcher2.sh (never overwrite a running script), old
  launcher killed by pid, new one started 16:09Z.
- 16:07Z node 6 booked by hand (21253, 12 h; 360 GB free, image present). Only gpt-oss is on node 6 (/data/ylerman/models/gpt-oss-120b, the 2026-09-10 fast-copy; no Qwen / DeepSeek snapshots
  there) -> gpt-oss B' chain launched there by hand 16:08Z with that path; started_gptoss marker set so the launcher skips it. Pending for other nodes: dsr1, qwen.
- infer_paired.sh prints "mkdir: cannot create directory ..._srv/logs: Permission denied" once per server: docker creates the mount dir as root before the host-side mkdir; the container
  writes /workspace/out/logs itself. Benign (same on every paired run); only gzip of those logs is skipped.
