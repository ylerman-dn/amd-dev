# PLAN — 2026-09-10-1-inmodel: two models x four traffic modes x four arms, CUDA graphs ON, radix cache OFF, one server per arm, two passes
Approved design (chat 2026-09-09 evening): arms stock / none / sc2 / dummy; 6 reps per server (rep 1 = cold, dropped); pass 2 in reverse order (drift check);
a detect server (TUNING logs, 1 rep) per plugin arm and mode for the rule hit-map. Nothing swaps mid-run. Nodes booked 10 h (jobs 21122-21125).

| node | job | model | modes | driver call |
|---|---|---|---|---|
| amd-mi355x-8 (XAI) | 21122 | Qwen3-30B-A3B | d32, p8k | chain.sh <qwen> qwen "d32 p8k" |
| amd-mi355x-9 (XAI) | 21123 | Qwen3-30B-A3B | d128, mix | chain.sh <qwen> qwen "d128 mix" |
| amd-mi355x-des2-2 (TEST) | 21124 | gpt-oss-120b | d32, p8k | chain.sh <gptoss> gptoss "d32 p8k" "--attention-backend triton" |
| amd-mi355x-ses2-1 (TEST) | 21125 | gpt-oss-120b | d128, mix | chain.sh <gptoss> gptoss "d128 mix" "--attention-backend triton" |

Model paths: /huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/ ; /huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/ (both present on all four nodes).
Modes (ISL/OSL/conc/prompts): d32 512/512/32/128 · p8k 8192/128/32/96 · d128 512/512/128/384 · mix 2048/512/64/192.
Arms: stock = custom all-reduce ON, RCCL_MSCCL_ENABLE=1, image NCCL_MIN_NCHANNELS=112 present, no plugin.
      none  = custom AR off, MSCCL 0, flag removed, no plugin.   sc2 = none + plugin + sc2_grid112.final.conf.   dummy = none + plugin + ring,simple,1 all sizes.
Per node: 2 modes x (2 detect + 8 timing servers) = 20 servers; ~12 min each (p8k ~20) -> ~4.5 h per node, all four in parallel.
Driver: infer_many.sh (573c356) via chain_template.sh (copied to /opt/shared as inmodel-2026-09-10.chain.sh); launched as one srun step per node from amd-mi355x-1.
Output tree on each node: /data/ylerman/inmodel-2026-09-10/<model>/<mode>/{detect,pass1,pass2}/<arm>_<def|tun>/bench_rep*.log (+ detect/hitmap_*.csv).
Fetched to this dir with the same tree (bench logs, hitmaps, score CSVs; server rccl logs stay on the nodes).
Score per model/mode/pass: infer_score.py --base <pass dir> --prefix "" ... (the scorer expects <prefix>_<arm>; run with --prefix set to the arm-dir stem, see HANDS).
Page: a NEW html (results-tuning/2026-08-23-1-pages/inmodel_2026-09-10.html): sc2 conf and how it was derived (sanity-check-2), then per model -> mode -> pass tables, hit-map per model-mode-arm.
