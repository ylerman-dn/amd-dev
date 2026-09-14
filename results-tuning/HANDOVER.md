# HANDOVER — GPU-107 tuner value (updated 2026-09-07, end of session)

Branch: `gpu107-value-runs` (base `gpu107-cli`, keep both). Index of every run: `results-tuning/RUNLOG.md`.
Pages (http://10.10.73.168:8410/): `collective_all_reduce_v2.html` (conf of record) · `verify.html` (in-model verdicts) ·
`channels.html` · `modes_matrix.html` · `rulehits.html` · `models_status.html`.

## Settled facts
1. **Conf of record: `gpu107_1n_v4_scripted.validated.csv`** (on /opt/shared/.../infer-2026-08-30/) — derived and judged 100%
   by the tool chain (`rccl_sweep --runtime container` 44-cell grid → `optimize_metrics` → `generate_tuner_config` →
   `validate_tuner_config --runtime container --drop-env NCCL_MIN_NCHANNELS`, 7 reps/arm, BOTH arms flag-off).
   **7 rules KEPT, all P(sup)=1.00**: tree/ll 64@4K, 84@8–32K, 32@64K, 112@128K, 84@256K; ring/ll 64@512K, 96@1–4M
   (+5.9..+75.7% vs flag-off default). 8M/≥16M: default optimal, no rules.
   Raw: `results-tuning/2026-09-07-1-fullgrid/` (grid raw/, ab_v4_noflag/).
2. **In-model (serving, 1-node TP-8): per-size rules are a tie** — 50 rounds/arm × 3 modes × 3 models (qwen30b/gptoss/DeepSeek-R1):
   −0.1..−1.1%, P(sup) 0.21–0.51. Ceiling effect: all_reduce ≈ few % of step time. BUT: that verify used conf v3 (flat 112 channels);
   **the v4 conf (fewer channels = less CU theft) was NEVER tested in-model — genuinely open.**
3. **Deployment env (image 112-flag + MSCCL on) beats the tuner-visible env in-model: +1.9..+6.6% tok/s, TTFT −30% (gptoss)** —
   verified 3 models. MSCCL goes default-OFF in RCCL 2.28.3 / ROCm 7.11 → the tuner path becomes deployment default then.
4. Per-collective map (deployment): alltoall never tunable (p2p); reduce_scatter/broadcast always tuner-visible; all_gather partial,
   its Direct algo unbeatable & inexpressible. Channels: real executed spans ≠ `-A` plan (validator reads truth).
5. Training exists in-house: **Arik Gelman** ran multi-node GROK1 training on this cluster (#ai-xai-poc, 2025-12-18),
   explicitly "without any RCCL optimization" — the partner for the multi-node/training value question.

## Next steps (in value order)
a. In-model A/B of the **v4** conf (window modes d128/d256, paired driver, flag-off) — the untested low-channel hypothesis.
b. Multi-node: rccl-tests probe first (tool bare runtime supports multi-node), then Arik's training harness for the real comm-share case.
c. findings/ entries (need user approval): MSCCL verdict, flag story, v4 conf, executed-channels truth.
d. Housekeeping: old 8807-page edit still uncommitted on main checkout; broadcast rules exist only in v3 (v4 = allreduce-only).

## Process rules (paid for in blood this session)
- Any derivation = the tool CLI end-to-end, never hand steps: sweep → optimize → generate → validate.
- Spell out the exact grid + BASELINE ENV (flag on/off!) and get explicit OK before running.
- RUNLOG row at launch. P(sup)≥0.95 gates. Canary/quiet split. Verify env from ENV log lines per run.
- pkill over ssh needs the [b]racket trick. Container hangs need `docker kill`, not client timeout.

## Future improvements (parked, not blocking; 2026-09-08)
- **Sweep through the tuner plugin instead of NCCL_MIN/MAX_NCHANNELS + NCCL_ALGO/PROTO env vars.**
  One-rule conf per cell, plugin mounted like the validator does; default cell = empty conf.
  Why: measure on the deployed path. At >=64M a plugin rule pins channels and disables WarpSpeed
  (-63..-66%, findings/11) while the env cap is ignored (222-224ch executed, 2026-08-03-4-infocap) -
  an env-var sweep cannot see that collapse, only the A/B catches it. Does NOT fix per-call channel
  trimming: plugin-set 96 runs as 94 at 8M exactly like the env var
  (2026-09-07-1-fullgrid/ab_v4_noflag/per_size_stats.csv cfg_applied_ch vs cfg_exec). ~30 lines in
  sweep_executor.py.


## 2026-09-08 sanity-check campaign (see results-tuning/2026-09-08-LOG.md, top section)
- Process A runs end to end with `tools/rccl-sweep/rccl_tune.py run --collectives all_reduce --scales 1 --name <n> [--grid ...]`
  (restored + container-era fixes today). Three grids done (1..48 / 1..112 / 1..224): 9/10/10 of 18 sizes kept, all P=1.00,
  defaults reproducible across all runs. Confs: results-tuning/2026-09-08-{1,2,3}-sanity-check-*/all_reduce_1n.final.conf.
- In-model (Qwen3-30B-A3B, conc 32): decode ties for every conf incl. a 1-channel dummy; prefill not measured (radix cache);
  dummy TTFT 12x on real prefill. Open: settling A/B for the 4M -73% plugin collapse; prefill-mode run; decode null vs arithmetic.
- Parked tool items: the three REVIEW.md files + 2026-09-08-4-qwen/REVIEW.md.


## 2026-09-09 follow-ups (see results-tuning/2026-09-09-LOG.md, DAY SUMMARY at top)
- Yesterday's in-model tie = radix cache (no prefill after round 1) + `--disable-cuda-graph` (CPU-bound decode). With graphs on the sc2 conf is +18% over the
  plugin-less RCCL path on Qwen and gpt-oss - but **stock SGLang's custom all-reduce bypasses RCCL entirely** (stock 7310 vs tuned 5720 tok/s; stock+plugin = 0 rule hits).
  Single-node TP-8 stock serving: the tuner has no lever. Value can only be in multi-node, training, other collectives, or frameworks that use RCCL for all_reduce.
- 4M -73% settled: over-request above the 112 built channels; a 112 rule is parity. Guard (never emit > node default channels) still to add.
- In-model measurement rules now: CUDA graphs ON (infer_many.sh IM_CUDA_GRAPH=1, one server per arm), radix cache OFF, reference = STOCK SGLang.


## 2026-09-10 in-model campaign (results-tuning/2026-09-10-1-inmodel/, page inmodel_2026-09-10.html)
- 2 models x 4 modes x 4 arms x 2 passes, graphs ON, radix OFF, one server per arm. Same order in all 16 tables: stock > sc2 > none >> dummy.
- Stock SGLang stays +8..+34% over the tuned RCCL path (custom all-reduce; no rule hits). Within the RCCL path the sc2 conf is +5..+28% over plain RCCL
  where a rule covers the mode's decode all_reduce (conc x hidden x 2 B), and 0% where none does (gpt-oss mix, 368,640 B in the 256K..512K gap).
- Design rules for any future in-model test: graphs on, radix cache off, one server per arm, reversed second pass, stock as reference, detect server for hit-maps.


## 2026-09-10 rerun on the NEW stack (results-tuning/2026-09-10-2-inmodel-rocm10/, SUMMARY.md there, page inmodel_rocm10_2026-09-10.html)
- Image lmsysorg/sglang:v0.5.19-rocm10-mi35x = SGLang 0.5.19, ROCm 10.0.0, RCCL 2.30.4 @6b0e43f, torch 2.11, AITER 4ad9983. 3 models (Qwen3-30B-A3B, gpt-oss-120b,
  DeepSeek-R1-0528 MXFP4) x 6 modes (d32 p8k d128 mix d256 d512; DeepSeek d32 d128 d512) x 5 arms x 2 passes on 5 nodes; 0 failures, receipts per server.
- **RCCL 2.30.4 facts** (verified in logs/strings, see SUMMARY section 1): no MSCCL at all; a new DDA direct all-reduce path (`RCCL_DDA_ENABLE`, `RCCL_DDA_THRESHOLD`) takes every
  intra-node all_reduce by default and never consults the tuner (0 rule hits with RCCL as shipped, plugin mapping itself fine). AITER custom AR aborts if
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set (our driver used to; infer_many.sh IM_NO_EXPANDABLE=1 drops it).
- **Result**: stock (AITER custom AR) fastest in all 15 model-mode tables (30 score.csv), but only +3..+13% over RCCL as shipped (DDA) - the old stack had +14..+52%. DDA beats the tuned classic path (sc2)
  in 14/15 tables in pass 1, 15/15 in pass 2 (+3..+28%). Within the classic path sc2 still helps where a rule covers the decode size (+6..+17%), ties where none does at 360K/2.9M/7M, and COSTS 13-15% where none
  does at 1.47M (gpt-oss d256) and 448K/1.75M (DeepSeek d32/d128). Cause (SUMMARY section 7, reproduced with an empty conf): **RCCL 2.30.4 ships a built-in CSV tuner with an MI355X table**
  (`share/rccl/tuner/rccl_tuner_gfx950.csv`: allreduce <16K tree/ll/1, 16K-512K ring/ll, 512K-1M ring/ll/32, 1M-2M ring/ll/56) that an external plugin REPLACES, so uncovered sizes fall to the generic
  cost model (SIMPLE instead of LL). Any conf for RCCL >= 2.30 must carry those rules for what it does not override.
- Source evidence for the RCCL changes (PR numbers, commits, verbatim motivations, code paths): results-tuning/2026-09-10-2-inmodel-rocm10/RCCL_2.27.7_to_2.30.4.md. Monorepo clone for future digging: /home/dn/ylerman/wg/rocm-systems (projects/rccl).
- Consequence for GPU-107: on this stack a single-node tuner conf has no lever at all for SGLang TP-8 all_reduce (custom AR, and behind it DDA). Remaining candidates: multi-node
  (nNodes > 1), sizes above the DDA threshold, other collectives (AllGathers DO go through the tuner: the probe applied a rule to every AllGather), other frameworks.
- Tools: infer_many.sh knobs IM_EXTRA_ENV, IM_NO_EXPANDABLE, IM_PLUGIN, RM_MSCCL=image; fetch_node.sh in the run dir; build_inmodel_page.py reads models/modes/nodes.txt and switches
  to the 5-arm layout (nodda base) when models.txt exists. Inspecting a node while its chain runs: `srun --overlap --jobid=...`.


## 2026-09-14 night campaigns A, B', C, D — is there any place left for a 1-node RCCL tuner conf on the rocm10 stack? (page night_2026-09-14.html)
- **A** (results-tuning/2026-09-14-1-bigmsg): rccl-tests all_reduce 1 node 128M..2G on RCCL 2.30.4: exact parity, the default (RING/SIMPLE/112) is the best cell of {ring,tree}x{LL,LL128,simple}x
  channels. LL128 is honoured on 2.30.4 (substituted on 2.27.7) but 20% behind SIMPLE per channel; 224 requested channels run slower than 112. Tool: rccl_tune.py --combos/--min-size/--max-size/--image/--minutes.
- **B'** (results-tuning/2026-09-14-3-bigmsg-inmodel): tune FROM THE MODEL at >= 128 MiB prefill all_reduces (custom AR declines > 64 MiB, ROCM_QUICK_REDUCE_QUANTIZATION=NONE): six model x size combos,
  13 arms x 3 rounds on one hot-reload server each: no arm beats RCCL default (best +/-0.1%); ring/simple channels below 112 only lose (8 ch -57..-68%); LL128/tree are IGNORE in the SGLang
  communicator (TUNING-log proof), so a tuner can steer only ring/simple channels there and the default already uses 112. Stock INT8 quick-reduce +2.4..+5.8% over any exact RCCL config.
- **C** (results-tuning/2026-09-14-2-sc2plusamd): sc2 + AMD's built-in gfx950 rules in the gaps (sc2_plus_amd.conf) removes the plugin's miss penalty completely (gpt-oss d256 -14.3% -> -0.1%,
  DeepSeek d32 -15.4% -> -0.1%, d128 -13.2% -> -0.1% vs AMD's table) and equals sc2 where sc2 has a rule. Rule for RCCL >= 2.30: a conf must carry the built-in rules for everything it does not override.
- **D** (results-tuning/2026-09-14-4-csvtuner): the same conf through RCCL's built-in CSV tuner (no plugin .so) — see its SUMMARY.
- Bottom line, 1 node, this stack: below 64 MiB AITER (SGLang) and DDA (RCCL) never consult a tuner; above 64 MiB the default is optimal in every space a tuner can reach. Remaining tuner-visible
  domain = multi-node (training-shaped traffic). Ops lessons in the campaign HANDS.md files: never overwrite a running script; detached processes die with their srun step; check rocm-smi
  --showpids for foreign VRAM on a freshly undrained node; fast-copy (rsync over FE) moves 376 GB in 6 min.
