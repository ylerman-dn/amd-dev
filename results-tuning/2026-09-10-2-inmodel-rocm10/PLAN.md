# PLAN — 2026-09-10-2-inmodel-rocm10: the 2026-09-10 in-model campaign repeated on the NEW stack
Same design, same confs, same modes, same page style; only the SGLang image changes. Separate dirs/pages, nothing of -1- touched.

## Stack under test
- Image lmsysorg/sglang:v0.5.19-rocm10-mi35x, pushed to Docker Hub 2026-09-10T03:00Z (digest sha256:10c46e93...). Inside: SGLang 0.5.19, ROCm 10.0.0,
  torch 2.11.0+rocm10.0.0, RCCL 2.30.4 (HEAD:6b0e43f, not in the public ROCm/rccl repo; NCCL-2.30.4-based -> code no older than 2026-04-22), AITER 4ad9983.
- Reference stack (-1-): v0.5.17-rocm720-mi35x = SGLang 0.5.17, ROCm 7.2.0, RCCL 2.27.7 @0d2c4fd (2025-12-09).
- Image env of note: NCCL_MIN_NCHANNELS=112 (same as old), ROCM_QUICK_REDUCE_QUANTIZATION=INT8 (new: AITER quick-reduce with INT8 quantization may be the stock all-reduce path),
  no RCCL_MSCCL_ENABLE (RCCL default applies; expected OFF in 2.28+).
- RCCL 2.30.4 exports ncclTunerPlugin_v2..v6; our plugin is v4 -> should load. Verified at run start by the detect server's hit-map (chain prints PLUGIN_HITS / PLUGIN_NOT_APPLIED).

## Nodes (5, booked; images pulled; models local or fast-copied)
| node | job | model | modes | model path |
|---|---|---|---|---|
| amd-mi355x-ses2-1 (TEST) | 21156 (10 h) | Qwen3-30B-A3B | d32, p8k, d128 | /huggingface snapshot |
| amd-mi355x-4 (XAI) | 21154 (10 h) | Qwen3-30B-A3B | mix, d256, d512 | /data/ylerman/models/Qwen3-30B-A3B (fast-copied from ses2-1) |
| amd-mi355x-7 (XAI) | 21155 (10 h) | gpt-oss-120b | d32, p8k, d128 | /huggingface snapshot |
| amd-mi355x-6 (XAI) | 21158 (12 h) | gpt-oss-120b | mix, d256, d512 | /data/ylerman/models/gpt-oss-120b (fast-copied from node 7) |
| amd-mi355x-5 (XAI) | 21157 (12 h) | DeepSeek-R1-0528 MXFP4 (amd/, quark, hidden 7168, 61 layers) | d32, d128, d512 | /huggingface snapshot 913fc83 |
Launch: /opt/shared/ylerman/GPU-107/inmodel-rocm10-2026-09-10.launch.sh (5 srun steps). ETA: 3 units x ~13 servers x ~13 min ~ 8.5 h per node (DeepSeek slower to load).

## Design (extends -1-)
- CUDA graphs ON, radix cache OFF, one server per arm, 6 reps (rep 1 dropped), two passes reversed. SGLang 0.5.19 verified to still accept every flag we use.
- FIVE arms (revised 16:1xZ after the first attempt; the planned 'msccl' arm is void on this stack - RCCL 2.30.4 has no MSCCL at all):
  stock = AITER custom all-reduce ON, image env untouched (NCCL_MIN_NCHANNELS=112 present), no plugin
  none  = custom AR OFF, NCCL_MIN_NCHANNELS REMOVED, RCCL as shipped: its new DDA path (RCCL_DDA_ENABLE default on) runs the decode all_reduces and never consults the tuner
  nodda = none + RCCL_DDA_ENABLE=0: the classic ring/tree path (the only path a tuner plugin can steer)
  sc2   = nodda + plugin + sc2_grid112.final.conf          dummy = nodda + plugin + ring,simple,1 (all sizes)
  Every server also gets IM_NO_EXPANDABLE=1 (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True aborts AITER custom AR on torch 2.11 / ROCm 10).
- Pass order: stock, none, nodda, sc2, dummy; pass 2 reversed.
- Detect servers per mode (TUNING logs, 1 rep): sc2, dummy, and STOCK + sc2 plugin -> hit-maps show where each path's all_reduce goes (stock: expected 0 hits).
- Receipts: after EVERY server the chain greps the RCCL ENV lines into <pass>/receipts.txt (parsed disable_custom_all_reduce and '[AR] Using ...' line from server.log, NCCL_MIN_NCHANNELS env value or 'absent',
  RCCL_DDA_ENABLE env value or 'unset', ncclDdaIpcCommInit lines, RCCL version) and prints RECEIPT_MISMATCH if an arm does not match its definition (none: flag absent, DDA unset, custom AR off;
  nodda/sc2/dummy: flag absent, DDA 0, custom AR off; stock: flag present, custom AR on). Pass-server rccl logs are gzipped after the receipt (disk).
- Modes (6): d32 512/512/32 · p8k 8192/128/32 · d128 512/512/128 · mix 2048/512/64 · d256 512/512/256 · d512 512/512/512. Decode all_reduce bytes = conc x hidden x 2:
  Qwen 128K / 128K / 512K / 256K / 1M / 2M · gpt-oss 180K / 180K / 720K / 360K / 1.4M / 2.8M · DeepSeek 448K / 1.75M / 7M. d256/d512 push the decode all_reduce to 1-7 MB,
  where rccl-tests said the tuner rules stop mattering (>= 4M parity) and where AITER/MSCCL relevance is the open question.
- DeepSeek gets 3 modes (d32, d128, d512) on one node: 448K (in the sc2 gap 256K..512K!), 1.75M (2M rule? no: 2097152 exact only -> gap), 7M (no rule). It mostly tests stock vs RCCL paths, not sc2.
- Tree: /data/ylerman/inmodel-rocm10-2026-09-10/<model>/<mode>/{detect,pass1,pass2}/<arm>_<def|tun>/ + receipts.txt per pass dir -> fetched here.
- Page: inmodel_rocm10_2026-09-10.html (generator reads models.txt / modes.txt / nodes.txt here; old page untouched). TIMELINE.md (Israel time), RUNLOG rows, LOG, SUMMARY, sub-agent REVIEW, commit.

## What this can and cannot tell
- CAN: does the ordering stock > sc2 > none >> dummy hold on the current stack; how much stock (AITER) moved; whether the 2.27.7-tuned sc2 rules still beat RCCL 2.30.4's own defaults; whether the plugin still loads.
- CANNOT: attribute a change to RCCL alone (SGLang, AITER, torch, ROCm all move at once). sc2 was derived on 2.27.7; a fair "tuned for this RCCL" conf would need Process A rerun on the new image (rccl-tests binary compatibility with 2.30.4 untested) - proposed as a follow-up, not part of this run.

## Hypotheses
1. stock still best everywhere; plugin 0 hits under stock (AITER path unchanged in kind). 2. sc2 > none in covered modes, but the gap may shrink if RCCL 2.30.4 defaults improved (MSCCL off by default changes "none" too).
3. gpt-oss mix still sc2 == none (368,640 B gap). 4. dummy still catastrophic. 5. Possible new-stack surprise: quick-reduce INT8 in stock (check server.log for the all-reduce backend line).
ETA ~8.5 h per node in parallel (bookings 10-12 h; the 10 h ones end ~21:1xZ: if a node runs late its last mode is re-queued on a fresh allocation); cron heartbeat every 30 min.
