# By hand — 2026-09-14-2-sc2plusamd (campaign C)
- 14:27Z node 2 idle after campaign A released it; holds Qwen/gpt-oss/DeepSeek snapshots (same hashes as 2026-09-10), the rocm10 image, 653 GB free. Booked 12 h (21247, ylerman-bigmsgC).
- 14:29Z sequencer started (qwen -> gptoss -> dsr1 chains as srun steps in 21247). No other idle XAI node; cron rebalances gptoss/dsr1 to another node if one frees (touch sc2plus.moved_<tag> before the chain starts).
- 14:33Z RECEIPT_MISMATCH on the first sc2 detect server: receipt column conf=none. Cause: RCCL never prints NCCL_TUNER_CONFIG_FILE as an ENV line (the plugin reads it itself), so the receipt's grep is a
  false negative for BOTH plugin arms; the tuner column (DN-TUNER v4), DDA=0, floor absent, custom AR off are all right. Verified in the rank log: 9 "Loaded config" lines = sc2's 8 rules + the header
  line parsed as a dummy rule "collective_type [0-0] ... nodes=0 ranks=0" (never matches: nodes=0). Not fixing the running chain (editing a script bash is executing shifts its read offsets, see
  2026-09-10 infer_many.sh lesson); the conf per arm is proven post-hoc by the "Loaded config" rule set in each server's logs (sc2plus carries "allreduce [0-4095] tree/ll channels=1") and by the
  hit-maps. Expect one RECEIPT_MISMATCH line per plugin server in this campaign; treat "conf=none" with tuner=DN-TUNER as benign.
- 14:56Z INCIDENT (my mistake, same class as 2026-09-10): I re-synced infer_many.sh to /opt/shared (IM_NO_PLUGIN knob for campaign D) while the Qwen d32 pass2 sc2plus server's infer_many
  instance was running. bash reads scripts incrementally: after its rep loop the instance hit "syntax error / unexpected EOF", skipped its `docker rm -f`, and left ylerman-many-sc2plus-tun
  holding port 8899. The next arm (pass2 sc2) started, its SGLang child died on the port, but infer_many's /health check was answered by the STALE sc2plus server -> reps 1-3 of "sc2" were
  measured against the sc2plus server, reps 4-6 FAIL after I removed the stale container (14:59Z). sc2plus pass2 data itself is complete (6 bench logs) and valid.
  Actions: pass2/sc2_tun renamed CORRUPT_sc2_tun_port_clash (excluded from scoring); fix-up server queued at the end of the sequencer (sc2plus-2026-09-14.fix_qwen_d32_sc2.sh, prints FIXUP
  start/done into the qwen driver log, then FIX_DONE in seq.log); B' launcher blocked from taking node 2 via SEQ_DONE (node2_used marker) so the fix-up and campaign D can use 21247 first.
  Rule, now in memory: never write to a script file on /opt/shared while any srun step may be executing it; stage under a NEW name and switch callers between runs.
- 15:04Z Qwen d32 pass2 nodda: rep1 FAIL (its /health was answered by the stale container, bench ran before the real server was up), reps 2-6 = 3947, 4017, 4017, 4015, 4017 (pass1 4019): valid, rep1 is dropped by scoring anyway. pass2 sc2 = the only lost server (fix-up queued). Chain continues (none pass2, then d128).
- 18:26Z node 8 image pull had DIED at 17:50Z: a docker pull started with setsid inside a short srun step is killed with the step's cgroup when the step ends. Restarted as its own srun step (stays alive for the pull), log /data/ylerman/pull-rocm10-2.log on node 8. C dsr1 launch waiter still armed.
- 18:58Z node 8 UNUSABLE: the DeepSeek server died at once with "The memory capacity is unbalanced. Some GPUs may be occupied by other processes" (server.log). rocm-smi: GPU 0 has 69 GB VRAM held
  by pid 2483436 (python3) of another user's docker container fisher_acc_dosedirprefix128_20260914T171646Z (lighteval, up 2 h, started outside Slurm while the node was draining). Not ours,
  not killable. C's dsr1 chain step cancelled, allocation 21257 released, the moved_dsr1 marker removed BEFORE the node-2 sequencer reached `run dsr1` (it is still in gptoss), so dsr1 runs on
  node 2 after gptoss as originally planned (then fix-up, D, release). Node-8 driver log kept as sc2plus-2026-09-14.dsr1.node8-failed.driver.log. Lesson: check rocm-smi --showpids (foreign
  VRAM) before launching on a node that was just undrained.
