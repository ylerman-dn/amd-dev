# PLAN — 2026-09-09-3-gptoss-modes: second model, gpt-oss-120b, decode-heavy and prefill-heavy, radix cache OFF
Node: amd-mi355x-ses2-1 (TEST, MI355X, openai/gpt-oss-120b local), same allocation as the settling A/B, runs after it.
Server: as the Qwen plan + `--attention-backend triton` (the backend used for gpt-oss on 2026-09-06, RUNLOG). Arms: sc2, dummy, none (6 rounds).
Modes: d32 512/512/32/128 and p8k 8192/128/32/96, each with a 1-round detect (sc2) first. gpt-oss hidden 2880 -> decode all_reduce at conc 32 =
  184,320 B -> hits sc2's 131072-262144 tree,ll,64 rule; prefill chunks larger.
Chained after the A/B in the same srun step: `for m in d32 p8k; do infer_paired.sh <gptoss> gm_${m}can 1 sc2:..; infer_paired.sh <gptoss> gm_$m 6 sc2:.. dummy:.. none:NONE; done`
Score: infer_score.py --prefix gm_<mode> --ref none. Outputs: node ses2-1 /data/ylerman/gptoss-2026-09-09/; driver log /opt/shared/ylerman/GPU-107/gptoss-2026-09-09.driver.log. ETA ~2 h.
