# By hand — 2026-09-14-3-bigmsg-inmodel (campaign B')
- 15:58-16:06Z node 6 went idle; the launcher booked it five times (21248-21252) and released it each time: its salloc parser matched "allocation NNNN" twice, the job id carried a newline,
  every srun --jobid failed with "Requested node configuration is not available" -> "nothing launchable". Fixed (head -1), restaged as launcher2.sh (never overwrite a running script), old
  launcher killed by pid, new one started 16:09Z.
- 16:07Z node 6 booked by hand (21253, 12 h; 360 GB free, image present). Only gpt-oss is on node 6 (/data/ylerman/models/gpt-oss-120b, the 2026-09-10 fast-copy; no Qwen / DeepSeek snapshots
  there) -> gpt-oss B' chain launched there by hand 16:08Z with that path; started_gptoss marker set so the launcher skips it. Pending for other nodes: dsr1, qwen.
- infer_paired.sh prints "mkdir: cannot create directory ..._srv/logs: Permission denied" once per server: docker creates the mount dir as root before the host-side mkdir; the container
  writes /workspace/out/logs itself. Benign (same on every paired run); only gzip of those logs is skipped.
- 16:2xZ gpt-oss b32768 search round 1 (input tok/s): none 104.6k | rs8 47k | rs16 75k | rs24 92k | rs32 103k | rs48 118k | rs64 127k | rs80 131k | rs112 133.4k | rl32 132.9k | rl64 132.9k |
  rl112 133.2k | tl112 133.0k. Two things to verify before believing the picture: (a) RCCL default (none) = 104k ~ the 32-channel level, far below rccl-tests' ring/simple/112 default ->
  the in-model default may pick fewer channels (read channel{Lo..Hi} of the 188,743,680 B all_reduce from the A/B default arm's INFO log, or a TUNING detect); (b) rl32/rl64/rl112/tl112 all equal
  rs112 to 0.3% although rccl-tests showed LL128 far slower -> suspect RCCL ignores a plugin LL128/TREE request in the model's comm and runs its own config with 112 channels, or the hot-reload
  swap did not take. Check the paired log's "hot-reload events per logs" count (expect 39 x 8 ranks for 13 arms x 3 rounds) and, post-hoc, a TUNING detect with the winning conf.
- 16:3xZ round 2: none = 133.1k (round 1: 104.6k). Reading of the plugin source (/opt/shared/ylerman/GPU-107/dn-tuner-src/plugin.c:317, 182): hot-reload keeps the PREVIOUS table when the new
  file yields no valid configuration -> the NONE arm (header-only file) after any real conf does NOT reset to "no rules"; from round 2 on "none" = the previous arm's conf. Round-1 none (fresh
  server, empty table) 104.6k is therefore the only true no-rule sample in the search, and it is also the first request batch after server start (TTFT 1388 ms vs 837 later): cold, not trustworthy.
  Consequence: the search's PICK compares against a polluted "none"; the real default-vs-conf answer comes from the A/B / stockref phase (fresh servers, no plugin in the default arm).
  Likely reading of rl32/rl64/rl112/tl112 == rs112 (all 133k): RCCL 2.30.4 rejects a plugin LL128/TREE choice the comm has not enabled and runs its own default, which per rccl-tests is
  ring/simple/112 = the same as rs112; then the in-model default is ~133k and rs112 has NO gain. To verify: TUNING detect with rl32 and tl112 confs and a fresh-server none (queued: verify_arms.sh).
