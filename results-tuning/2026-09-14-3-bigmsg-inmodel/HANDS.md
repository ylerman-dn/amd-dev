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
- 16:30Z waiter armed (dev VM background): when the gptoss driver log shows 'b65536 detect', the srun step is cancelled, containers/b65536 dir cleaned, and b65536 relaunched on chain2 (nomatch no-rule arm) into gptoss65536.driver.log.
- 16:30Z gpt-oss b32768 done (16:08-16:30Z, 22 min: 1 detect + 1 search server (13 arms x 3 rounds = 39 bench runs) + 1 stock server (3 reps)). Stock (INT8 quick-reduce) output tok/s 2159-2163
  vs the RCCL arms ~2080 (+4% for stock); input tok/s from the bench logs at fetch time. "[sgptoss] hot-reload events per logs: 0" is a LOGGING artefact: the search server runs with
  NCCL_DEBUG_SUBSYS=INIT,ENV, which filters the plugin's TUNING-class "hot-reloaded" lines (the detect server with TUNING shows 8 = 1 reload x 8 ranks); the swaps did act (47k..133k spread).
- 16:30Z b65536 moved to chain2 (step 21253.1 cancelled at its detect start, containers removed; the root-owned detgptoss_srv/{server.log,logs} of the cancelled server could not be removed
  (docker-created), so the new detect's rank logs share that dir with a few init-phase lines of the killed server: hit counts for b65536 detect may include them; sizes/algos unaffected).
  New step 21253.4, driver log bigmsg-inmodel-2026-09-14.gptoss65536.driver.log, no-rule arm = nomatch.conf.
- 16:52Z gpt-oss ALL_DONE on node 6: 16:08:18Z -> 16:52:09Z = 44 min for both batch sizes (b32768 22 min: detect 2 runs, search 39 runs, stock 3 reps; b65536 22 min: same). Fetched + scored
  (gptoss/b*/score.csv). Both PICKs NO_AB (+0.08% / +0.06%). Stock INT8 quick-reduce vs RCCL default: 138.3k vs 133.1k input tok/s (+3.9%) at 180 MiB, 141.2k vs 134.9k (+4.7%) at 360 MiB.
  verify_arms.sh started in 21253 (step 21253.6) -> verify.driver.log; node 6 to be released after it.
- 16:57Z verify_arms done (node 6, 4 min, gptoss/verify on the node; verify.driver.log). Rank-0 TUNING log, 180 MiB all_reduce, all 7 arms: 10220 x "RING/SIMPLE channel{0..111}" and nothing
  else. Plugin lines: 2920 x "Applied ... ring/simple/112" (warm + rs112 arms) and 1752 x "[ring][ll128] is marked as IGNORE" + 1752 x "[tree][ll128] is marked as IGNORE" (rl32, tl112 arms):
  RCCL passes the plugin a cost table in which LL128 (and tree/ll128) are IGNORE for the SGLang communicator, and plugin.c:463 applies a rule only when its algo/proto is not IGNORE.
  So LL128/tree cannot be selected by a tuner in the model, and RCCL's own default there is ring/simple/112 (nomatch, nomatch2, empty arms all 133k with the same executed line).
  Corrections to earlier notes: (1) the "empty file keeps the previous table" theory is WRONG for this plugin build: the CSV header parses as a dummy rule, the reload succeeds (48 reloads, 0
  failed), so the original chain's NONE arm from round 2 on WAS a valid no-rule measurement; the nomatch.conf change is harmless and stays. (2) the 104k first-batch values are cold start
  (the warm arm, rs112, read 104.4k as first batch too).
- 16:59Z node 6 released (21253) and rebooked (21255, 12 h); Qwen snapshot fast-copied node 2 -> node 6 (/data/ylerman/models/Qwen3-30B-A3B, 57 GB, 54 s over the FE link); Qwen chain2
  launched by hand 16:58:57Z. Launcher had grabbed node 6 in between (21254) and released it: its node check read srun's trailing error line instead of the NOMODEL marker -> fixed
  (launcher4.sh, running). Pending: dsr1 only (needs a node with the DeepSeek snapshot: node 2 after C, or a ~350 GB fast-copy to a free node).
- 17:45Z plan for dsr1: DeepSeek snapshot is 376 GB, node 6 has 302 GB free but 122 GB of it are our own gpt-oss/Qwen copies (no longer needed there). A detached follow-up (dev VM, /home/dn/.claude/jobs/a55495d4/tmp/after_qwen.sh, log after_qwen.log) waits for the Qwen B' ALL_DONE, fetches+scores qwen, removes those two copies, fast-copies DeepSeek from node 2 over the FE link (~6 min; node 2 is running campaign C servers, rsync is disk/CPU only), verifies 83 safetensors + config, then launches the dsr1 chain2 on node 6 (job 21255) and marks started_dsr1.
