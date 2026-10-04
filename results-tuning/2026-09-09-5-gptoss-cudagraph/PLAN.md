# PLAN — 2026-09-09-5-gptoss-cudagraph: gpt-oss-120b with CUDA graphs ON (deployment-realistic), one server per arm
Same design as 2026-09-09-4-qwen-cudagraph, on amd-mi355x-ses2-1 (TEST, MI355X, job 21078, after the graphs-off gpt-oss modes finished 08:06Z).
infer_many.sh 573c356 with IM_CUDA_GRAPH=1 IM_DROP_FLOOR=1; server args --attention-backend triton --disable-radix-cache; d32 (512/512/conc 32/128 prompts).
Detect: sc2, TUNING, 1 rep. Quiet: pass 1 none/sc2/dummy, pass 2 dummy/sc2/none, 6 reps each (rep 1 = cold, dropped by the scorer).
Chain: /opt/shared/ylerman/GPU-107/gptoss-cg-2026-09-09.chain.sh; driver log ...gptoss-cg-2026-09-09.driver.log; outputs node ses2-1 /data/ylerman/gptoss-cg-2026-09-09/.
Question: does the sc2 conf's decode win under graphs (Qwen: TPOT -15.5%) repeat on a second model (hidden 2880 -> 184K decode all_reduce, same sc2 rule tree,ll,64)?
