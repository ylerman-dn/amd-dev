# PLAN — 2026-09-14-3-bigmsg-inmodel (campaign B-prime): tune the tuner from the model at >= 128 MiB, where AITER and DDA do not act

Approved by ylerman 2026-09-14 ~14:45Z ("let's try some combos ... find as many value we can give via the tuner for those models x modes ... keep one run of logs ... turn off the quant flag,
not AITER or DDA ... >= 128M ... each model in a different sub-dir ... one model per node, try 3 nodes ... log per model in Israel time how long and how many runs").

- Why here: below 64 MiB AITER custom AR (stock) and RCCL's DDA take every all_reduce without consulting a tuner; above 64 MiB AITER declines, and with ROCM_QUICK_REDUCE_QUANTIZATION=NONE
  quick-reduce declines too, so the prefill all_reduces reach RCCL's classic path. rccl-tests (campaign A) found ring/simple/112 = default optimal there, but rccl-tests has no compute next to
  the collective: CU budget, clocks/power and HBM/L2 contention can move the in-model optimum (e.g. fewer channels).
- Combos (>= 128 MiB, message = prefill batch tokens x hidden x 2 B): Qwen 32768 (128 MiB), 65536 (256 MiB); gpt-oss 32768 (180 MiB), 65536 (360 MiB); DeepSeek 16384 (224 MiB), 32768 (448 MiB).
  Server: --chunked-prefill-size N --max-prefill-tokens N, custom AR ON, graphs ON, radix OFF, IM_NO_EXPANDABLE=1, image v0.5.19-rocm10. Traffic ISL 8192 / OSL 128 / conc 32 / 96 prompts.
- Per combo (chain_template.sh): detect (hot-reload server, TUNING logs, arms NONE + big_ring_simple_112: sizes seen by RCCL, BIG_HITS) -> search (one hot-reload server, 3 rounds
  round-robin over 13 arms: NONE + 12 one-rule confs on [64 MiB+1, 2 GiB]: ring/simple x {8,16,24,32,48,64,80,112}, ring/ll128 x {32,64,112}, tree/ll128 x 112; prefill is not graph-captured so
  each swap acts) -> PICK (median input tok/s) -> if best > NONE by > 2%: A/B one server per arm, 6 reps, pass1 default/conf/stock, pass2 reversed; else NO_AB + a 3-rep stock reference.
- Metrics: input token throughput and mean TTFT (prefill), output tok/s recorded. Gate for a "value" claim: P(sup) >= 0.95 and > 2% on input tok/s in the A/B.
- Nodes: launcher.sh (on amd-mi355x-1) starts one model per node as nodes free: node 2 after campaign C (reuse 21247 if >= 3 h left, else rebook), any idle XAI node (12 h booking).
  Checks image, model snapshot, >= 20 GB free before launching; NEED_COPY logged if a model is absent (fast-copy by hand).
- Outputs: /data/ylerman/bigmsg-inmodel-2026-09-14/<tag>/b<batch>/{detect,search,pass1,pass2|stockref} on the node -> fetched here under <tag>/. Driver logs bigmsg-inmodel-2026-09-14.<tag>.driver.log.
  TIMELINE.md in Israel time with per-model start/end; SUMMARY.md with wall time and run counts per model; page in results-tuning/2026-08-23-1-pages/.
