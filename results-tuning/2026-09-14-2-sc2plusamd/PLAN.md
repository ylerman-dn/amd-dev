# PLAN — 2026-09-14-2-sc2plusamd (campaign C): does carrying AMD's built-in rules remove the plugin's miss penalty?

Background: on RCCL 2.30.4 loading our tuner plugin replaces the built-in CSV tuner and its gfx950 table; every size our conf does not cover then runs the generic cost model
(SIMPLE instead of AMD's LL) and loses 13-15% (2026-09-10-2 SUMMARY §7, reproduced with an empty conf). Fix under test: sc2_plus_amd.conf = the sc2 rules verbatim + AMD's four rules
cut to the gaps (../2026-09-14-1-bigmsg/sc2_plus_amd.conf, staged in /opt/shared/ylerman/GPU-107/infer-2026-08-30/).

- Arms (custom AR OFF in all, this is about the RCCL path): none (RCCL as shipped = DDA) · nodda (RCCL_DDA_ENABLE=0, AMD's built-in table active) · sc2 (nodda + plugin + sc2) · sc2plus (nodda + plugin + sc2_plus_amd).
- Models x modes: Qwen d32 d128 mix d256; gpt-oss d32 d128 mix d256 (--attention-backend triton); DeepSeek-R1 d32 d128. Same traffic params as 2026-09-10.
- Per mode: detect servers sc2 and sc2plus (TUNING, 1 rep) -> hit-maps; pass 1 none, nodda, sc2, sc2plus; pass 2 reversed; 6 reps, rep 1 dropped. Graphs ON, radix OFF, IM_NO_EXPANDABLE=1, image v0.5.19-rocm10.
- Receipts per server: disable_custom_all_reduce, NCCL_MIN_NCHANNELS env, RCCL_DDA_ENABLE env, which tuner loaded (built-in CSV vs DN-TUNER), which conf. RECEIPT_MISMATCH if an arm deviates.
- Expected: sc2plus >= nodda everywhere (never below AMD's table), sc2plus == sc2 where sc2 has a rule, sc2plus > sc2 where sc2 has none (gpt-oss d256 1.47M, DeepSeek d32 448K, d128 1.75M).
- Node 2 (job 21247, 12 h) runs the three chains sequentially through a sequencer; a chain not yet started can be moved to another node by touching /opt/shared/ylerman/GPU-107/sc2plus.moved_<tag>.
- Scoring: infer_score.py --prefix "" --ref none_def per pass; SUMMARY table sc2plus vs nodda / vs sc2 / vs none.
