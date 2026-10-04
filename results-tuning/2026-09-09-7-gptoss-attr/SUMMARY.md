# SUMMARY — 2026-09-09-7-gptoss-attr: stock's advantage is the custom all-reduce; in stock SGLang the RCCL tuner never sees the model's all_reduce

Node amd-mi355x-ses2-1 (TEST, MI355X), job 21086, gpt-oss-120b, CUDA graphs ON, radix cache OFF, --attention-backend triton, d32, one server per
arm, 6 reps (rep 1 dropped). Score node/at_score.csv (ref = stock). Env receipts per arm: env_receipts.txt (grepped from the node's rccl logs).

| arm | custom all-reduce | MSCCL | NCCL_MIN_NCHANNELS=112 | plugin | median tok/s | vs stock | TPOT ms |
|---|---|---|---|---|---|---|---|
| stock (reference, rerun) | ON | ON | present | none | 7306 | 0 | 4.14 |
| customar | ON | off | removed | none | 7316 | +0.1% | 4.14 |
| stocksc2 | ON | ON | present | sc2 conf, TUNING logs | 7313 | +0.1% | 4.14 |
| msccl | off | ON | present | none | 6226 | -14.8% | 4.90 |
| (run 6) sc2 | off | off | removed | sc2 conf | 5720 | -21.7% | 5.35 |
| (run 6) none | off | off | removed | none | 4840 | -33.8% | 6.37 |

Findings:
1. **The custom all-reduce alone reproduces stock** (7316 vs 7306). MSCCL and the channel flag add nothing once the custom kernel is on.
2. **With the custom all-reduce on, RCCL is not used for the model's all_reduce at all**: the stocksc2 server loaded the plugin (184 DN-TUNER init
   lines, 8 ranks) and applied ZERO rules; its rccl logs contain exactly one `AllReduce: 4 Bytes` line for the whole run (node ses2-1
   at_stocksc2_tun/logs/). Decode (184K) and prefill chunks (2.9 MB) all went through SGLang's kernel. So on single-node TP-8 stock serving of this
   model the tuner conf has no lever, good or bad.
3. Within the RCCL path (custom all-reduce off): MSCCL + flag (the image's defaults) 6226 > sc2 conf 5720 > plain RCCL 4840. The tuned conf is
   +18% over plain RCCL but -8% under MSCCL+flag, and MSCCL goes default-off only in RCCL 2.28.3 / ROCm 7.11 (HANDOVER 3).
4. Consistent with HANDOVER items 3-4 (deployment env beats tuner-visible env; some paths are not tuner-visible) and explains the 2026-09-06
   verify campaign (deployment env +2.9..+6.6% over the plugin arms, graphs off).

Where a tuner conf could still matter (not measured): multi-node serving (custom all-reduce is intra-node), models/configs where SGLang falls back
to RCCL for large all_reduce (size threshold not in any local file), other collectives (reduce_scatter/broadcast are always tuner-visible), training.
