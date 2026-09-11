# By hand — 2026-09-10-2-inmodel-rocm10 (prep, before launch)
- Booked 5 nodes: ses2-1 (21156, 10 h), 4 (21154, 10 h), 7 (21155, 10 h) at ~11:1xZ; 5 (21157, 12 h) and 6 (21158, 12 h) at ~12:0xZ.
- Pulled lmsysorg/sglang:v0.5.19-rocm10-mi35x (44 GB) on 5, 7, ses2-1, 6 (node 4 had it).
- fast-copy (rsync -aL over the FE network): Qwen3-30B-A3B snapshot ses2-1 -> node4:/data/ylerman/models/Qwen3-30B-A3B (61 GB);
  gpt-oss-120b snapshot node7 -> node6:/data/ylerman/models/gpt-oss-120b (65 GB safetensors; the copy stopped at the unneeded original/ subdir when the disk filled).
  Both verified file-by-file (sizes) against the source.
- Nodes 4 and 6 root disks hit 100% (3-6 GB free) after image + model. Removed ONE stale, unused image on each (lmsysorg/sglang:v0.5.11-rocm720-mi35x on 4,
  v0.5.11-rocm700-mi35x on 6; 4 months old, 75 / 60 GB) to make room. Other users can re-pull; nothing running used them.
- infer_many.sh: RM_MSCCL=image option (8d3946b). infer_hitmap/infer_score unchanged. Page generator now reads models.txt/modes.txt/nodes.txt (old page unchanged).
- Verified on node 4 (srun, GPU): SGLang 0.5.19 still has --disable-cuda-graph --disable-custom-all-reduce --disable-radix-cache --attention-backend --trust-remote-code.
- 18:04 Israel (15:04Z): launch via /opt/shared/ylerman/GPU-107/inmodel-rocm10-2026-09-10.launch.sh (5 srun steps on jobs 21156/21154/21155/21158/21157).
- 15:2xZ first detect servers up on all nodes. Two receipt findings, both expected on the new stack, no action on the chain:
  (a) RCCL 2.30.4 has no MSCCL: librccl.so.1 carries 19 "msccl" strings vs 349 in 2.27.7 and no RCCL_MSCCL_ENABLE env string; servers log no "RCCL_MSCCL_ENABLE set by environment" line
      and zero MSCCL lines even though the container gets -e RCCL_MSCCL_ENABLE=0/1. So RECEIPT_MISMATCH "msccl=unset" fires for every arm; the 'msccl' arm on this stack = plain RCCL + NCCL_MIN_NCHANNELS=112 (flag alone).
  (b) receipts.txt custom_AR column is derived from a grep of "--disable-custom-all-reduce" in server.log, but SGLang 0.5.19 prints the parsed args ('disable_custom_all_reduce': True), not the command line,
      so the column reads ON for every arm. Verified from server.log that the sc2/none/dummy servers have disable_custom_all_reduce=True (custom AR OFF) as intended. Re-derived at fetch time from server.log.
- 15:5xZ STOPPED all five chains (scancel the steps, allocations kept) after the first detect phase showed two blockers on the new image:
  (1) every server with the custom all-reduce enabled (stock, stock+plugin) ABORTS: AITER custom_all_reduce.cuh:3448 hipIpcGetMemHandle -> HIP invalid argument,
      preceded by "custom allreduce copy-in during CUDA graph capture because expandable-segments". Cause: our driver sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      (from 2026-08-30); on torch 2.11 / ROCm 10 that breaks AITER's IPC handles. Fix: infer_many.sh IM_NO_EXPANDABLE=1 (fbeb389), used for every server of this campaign.
  (2) the v4 tuner plugin loads on RCCL 2.30.4 (RCCL tries v5 first, falls back to v4) and parses the conf, but matches NOTHING: 0 rule hits, every request logs
      "collType match=0". Also only 64 tuner calls per rank in a whole detect run (vs ~8.5k on 2.27.7) - RCCL 2.30 seems to cache tuner answers per size.
      Diagnosis in progress: a probe conf with one full-range rule per collective name (channels 13..17 encode the name) shows which name the plugin's
      ncclFunc_t mapping assigns to the model's all_reduce under 2.30.4.
- 15:5xZ diagnostics launched: probe server (ses2-1, none env + plugin probe_colltypes.conf, TUNING) and stock test server (node 4, custom AR on, IM_NO_EXPANDABLE=1, 2 reps).
- 16:03Z both diagnostics read (probe: /data/ylerman/inmodel-rocm10-2026-09-10/probe/probe_tun on ses2-1; stock test: .../probe/stocktest_def on node 4; driver logs
  /opt/shared/ylerman/GPU-107/rocm10-probe.driver.log, rocm10-stocktest.driver.log):
  (1) stock test with IM_NO_EXPANDABLE=1 came up: server.log "[AR] Using AiterCustomAllreduce (AMD default)", 2 reps 5008 / 5130 tok/s (d32 params). Fix confirmed.
  (2) the plugin's collType mapping is CORRECT on RCCL 2.30.4: the probe conf's allgather rule (channels=14) was applied to every AllGather ("Applied config for collType=allgather ... channels=14",
      executed channel{0..13}); the allreduce rule (13) was applied to the only AllReduce RCCL saw: one 4-byte call per rank. The model's decode all_reduces (128K..2M) never reached the
      tuner because RCCL 2.30.4 runs them on its new DDA path: the rank log says "ncclDdaIpcCommInit: scratch N bytes, IpcGpuBarrier ...", librccl.so carries dda_all_reduce_ipc/fabric
      kernels and the env knobs RCCL_DDA_ENABLE / RCCL_DDA_THRESHOLD / RCCL_DDA_FABRIC_MAXBLOCKS (strings). DDA is a direct-IPC all-reduce that bypasses algo/proto/channel selection.
  (3) both diagnostic containers were still up because infer_many.sh was re-synced while they ran (bash read the changed file: "line 121: unexpected EOF") - removed by hand. Lesson: never
      re-sync a script a running srun step is executing.
- 16:07Z probe 2 (ses2-1, none env + RCCL_DDA_ENABLE=0 + sc2 conf, TUNING, 2 reps; /opt/shared/ylerman/GPU-107/rocm10-nodda-probe.driver.log, tree .../probe/nodda_tun, hitmap
  .../probe/hitmap_nodda.csv): "RCCL_DDA_ENABLE set by environment to 0", 71392 applied lines, all 8 sc2 rules hit (131072..262144 x11640, 524288..1048576 x39576, 2097152 x8536),
  tok/s 4157 / 4240. For comparison on the same node and params: DDA on (probe 1, effectively plain RCCL) 4861; stock 5008 / 5130.
- 16:1xZ infer_many.sh got IM_EXTRA_ENV="K=V ..." (extra -e); chain_template.sh rewritten: arms stock, none, nodda, sc2, dummy (msccl dropped), IM_NO_EXPANDABLE=1 on every server,
  receipts now record disable_custom_all_reduce (parsed) + "[AR] Using ..." + RCCL_DDA_ENABLE env + ncclDdaIpcCommInit count. Page generator: ARM_ORDER stock,none,nodda,sc2,dummy for this run.
- 16:14:44Z attempt 2 launched (same launch script, same jobs; attempt-1 driver logs kept as *.driver.attempt1.log; partial trees qwen/gptoss/dsr1 removed on all five nodes, probe/ kept).
  Allocations end 00:41Z (ses2-1, 4, 7) and 02:52Z (5, 6); scontrol TimeLimit extension refused (permission). ETA ~8.5 h per node -> ses2-1/4/7 are tight; watch the last mode.
- 16:2xZ attempt 2 detect phase clean on all five nodes: PLUGIN_HITS qwen d32 68288, qwen mix 65184, gptoss d32 24528, gptoss mix 24528 (capture-time counts, deterministic per model);
  STOCK_PLUGIN_HITS 0 everywhere (custom AR takes the all_reduce); no RECEIPT_MISMATCH. Receipt sample (qwen/d32/detect/receipts.txt on ses2-1): sc2/dummy disable_custom_all_reduce=True,
  RCCL_DDA_ENABLE_env=0; stocksc2 False, AR_impl=AiterCustomAllreduce, NCCL_MIN_NCHANNELS_env=112. Note: ncclDdaIpcCommInit still logs with RCCL_DDA_ENABLE=0 (comm setup), the path is
  just not taken - the hit counts are the proof.
- Inspecting a node while its chain step runs needs `srun --overlap --jobid=...`; a plain srun blocks with "step creation temporarily disabled" (one such srun hung 2 min and was killed).
- 19:04Z node 7 ALL_DONE -> fetch_node.sh gptoss amd-mi355x-7 21155 (tar over srun --overlap, excludes rccl logs + server.log; scores pass1/pass2 with infer_score.py --base <pass> --prefix "" --ref stock_def),
  docker ps clean, scancel 21155. Page built once (partial).
- 19:17Z ses2-1 ALL_DONE -> fetch_node.sh qwen amd-mi355x-ses2-1 21156, docker ps clean, scancel 21156. Page rebuilt. Remaining: 4 (qwen d512), 6 (gptoss d512), 5 (dsr1 d128 pass2 -> d512).
- 20:08Z node 6 ALL_DONE -> fetch_node.sh gptoss amd-mi355x-6 21158 (merges into gptoss/ next to node 7's modes), docker ps clean, scancel 21158. Page rebuilt. Remaining: 4 (qwen d512 pass2), 5 (dsr1 d512 pass1).
- 20:15Z node 4 ALL_DONE -> fetch_node.sh qwen amd-mi355x-4 21154, docker ps clean, scancel 21154. Page rebuilt. Remaining: node 5 (dsr1 d512, ETA ~21:30Z).
- 00:17Z node 5 ALL_DONE -> fetch_node.sh dsr1 amd-mi355x-5 21157, docker ps clean. Allocation 21157 KEPT for the miss-penalty diagnostic (rocm10-missdiag.sh: nodda vs empty-conf plugin, DeepSeek d32,
  TUNING logs, 2 reps each; out /data/ylerman/inmodel-rocm10-2026-09-10/missdiag/ on node 5). Release 21157 after it.
- Consistency check over the fetched tree: 30 score.csv, 45 hitmap csv, 150 pass receipts (30 per arm, every arm's columns identical across all nodes: plugin arms disable_custom_all_reduce=True +
  RCCL_DDA_ENABLE_env=0, none same with DDA unset, stock False + AiterCustomAllreduce + floor 112), stocksc2 rule hits 0 in all 15 detects, 0 "tok/s=FAIL" in the five driver logs.
- 00:26Z missdiag done; mechanism = RCCL 2.30.4 built-in CSV tuner (rccl_tuner_gfx950.csv, 4 rules) displaced by our plugin (SUMMARY section 7). docker ps clean, scancel 21157 at 00:30Z.
  Cluster state: none of our jobs/containers left (squeue shows only other users' jobs). Raw trees (rccl logs, server.log) stay on the nodes under /data/ylerman/inmodel-rocm10-2026-09-10/.
