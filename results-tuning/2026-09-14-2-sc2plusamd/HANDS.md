# By hand — 2026-09-14-2-sc2plusamd (campaign C)
- 14:27Z node 2 idle after campaign A released it; holds Qwen/gpt-oss/DeepSeek snapshots (same hashes as 2026-09-10), the rocm10 image, 653 GB free. Booked 12 h (21247, ylerman-bigmsgC).
- 14:29Z sequencer started (qwen -> gptoss -> dsr1 chains as srun steps in 21247). No other idle XAI node; cron rebalances gptoss/dsr1 to another node if one frees (touch sc2plus.moved_<tag> before the chain starts).
- 14:33Z RECEIPT_MISMATCH on the first sc2 detect server: receipt column conf=none. Cause: RCCL never prints NCCL_TUNER_CONFIG_FILE as an ENV line (the plugin reads it itself), so the receipt's grep is a
  false negative for BOTH plugin arms; the tuner column (DN-TUNER v4), DDA=0, floor absent, custom AR off are all right. Verified in the rank log: 9 "Loaded config" lines = sc2's 8 rules + the header
  line parsed as a dummy rule "collective_type [0-0] ... nodes=0 ranks=0" (never matches: nodes=0). Not fixing the running chain (editing a script bash is executing shifts its read offsets, see
  2026-09-10 infer_many.sh lesson); the conf per arm is proven post-hoc by the "Loaded config" rule set in each server's logs (sc2plus carries "allreduce [0-4095] tree/ll channels=1") and by the
  hit-maps. Expect one RECEIPT_MISMATCH line per plugin server in this campaign; treat "conf=none" with tuner=DN-TUNER as benign.
