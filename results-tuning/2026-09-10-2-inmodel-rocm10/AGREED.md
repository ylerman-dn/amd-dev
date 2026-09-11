# AGREED — rocm10 rerun (2026-09-10). Inherits every rule of ../2026-09-10-1-inmodel/AGREED.md; differences listed here first, then the inherited text.

## Differences for this run
- Image lmsysorg/sglang:v0.5.19-rocm10-mi35x (RCCL 2.30.4); nodes ses2-1 + 4 (Qwen), 7 + 6 (gpt-oss), 5 (DeepSeek); jobs 21156/21154/21155/21158/21157 (superseded the first draft's 21153-21156 map); tree /data/ylerman/inmodel-rocm10-2026-09-10; repo dir results-tuning/2026-09-10-2-inmodel-rocm10; page inmodel_rocm10_2026-09-10.html.
- FIVE arms: stock (AITER on, image env), none (AITER off, floor removed, RCCL as shipped = DDA path), nodda (none + RCCL_DDA_ENABLE=0), sc2 and dummy (= nodda + plugin).
  Revised 16:1xZ: the planned msccl arm (AITER off, RCCL_MSCCL_ENABLE=1, flag present) is void - RCCL 2.30.4 carries no MSCCL and does not recognise the env var - and the first
  attempt showed that RCCL 2.30.4's DDA path takes every decode all_reduce without consulting the tuner (0 hits), so a plugin arm needs DDA off. Every server: IM_NO_EXPANDABLE=1.
- SIX modes (+ d256, d512); THIRD model DeepSeek-R1-0528 MXFP4 (amd/) on node 5 with d32/d128/d512. Five nodes: ses2-1, 4, 7, 6, 5 (jobs 21156, 21154, 21155, 21158, 21157).
- Receipts per server (receipts.txt) + RECEIPT_MISMATCH lines; detect also runs stock+plugin per mode.
- Chain prints '=== RCCL_IN_SERVER ...' and '=== PLUGIN_HITS ... / PLUGIN_NOT_APPLIED' after each mode's detect: if PLUGIN_NOT_APPLIED appears, stop and diagnose (plugin ABI vs RCCL 2.30.4) before letting the passes run.
  (Happened at 15:2xZ in attempt 1; diagnosed as DDA, not ABI - see HANDS.md. Attempt 2 launched 16:14:44Z.)
- Nothing in ../2026-09-10-1-inmodel/ or its page is modified.

# (inherited) AGREED with ylerman, 2026-09-09 evening

## Status
- 4 nodes booked 10 h from ~17:5xZ 2026-09-09: jobs 21122 (amd-mi355x-8, Qwen d32+p8k), 21123 (amd-mi355x-9, Qwen d128+mix),
  21124 (amd-mi355x-des2-2, gpt-oss d32+p8k), 21125 (amd-mi355x-ses2-1, gpt-oss d128+mix). NOT LAUNCHED. Launch ONLY on explicit user approval.
- Chain script staged: /opt/shared/ylerman/GPU-107/inmodel-2026-09-10.chain.sh (= chain_template.sh here). Launch per node from amd-mi355x-1:
    setsid nohup srun --jobid=<job> -N1 -w <node> bash /opt/shared/ylerman/GPU-107/inmodel-2026-09-10.chain.sh <model_path> <qwen|gptoss> "<mode1 mode2>" ["--attention-backend triton" for gptoss] \
      > /opt/shared/ylerman/GPU-107/inmodel-2026-09-10.<node>.driver.log 2>&1 < /dev/null &
  (capture the jobid with `| head -1`; put the launch in a script file if quoting fights back - it did twice today.)
- Cron 0b65fe69 every 30 min: before approval only checks jobs alive; after launch drives fetch/score/release/page/review and deletes itself.

## Design (fixed)
- CUDA graphs ON (infer_many.sh IM_CUDA_GRAPH=1), radix cache OFF (--disable-radix-cache), one SGLang server per arm, 6 reps per server, rep 1 dropped (cold).
- Two passes per model-mode, pass 2 in reverse arm order = drift check (drift ~1-2%/h). Passes should agree within ~1%.
- Arms: stock (custom all-reduce ON, RCCL_MSCCL_ENABLE=1, image NCCL_MIN_NCHANNELS=112 present, no plugin) · none (custom AR off, MSCCL 0, flag removed, no plugin)
  · sc2 (= none + plugin + sc2_grid112.final.conf) · dummy (= none + plugin + ring,simple,1 all sizes). Confs in /opt/shared/ylerman/GPU-107/infer-2026-08-30/.
- Detect server per plugin arm and mode (TUNING logs, 1 rep) -> infer_hitmap.py -> hitmap_<arm>.csv.
- Modes: d32 512/512/32/128 · p8k 8192/128/32/96 · d128 512/512/128/384 · mix 2048/512/64/192 (ISL/OSL/conc/prompts).
- NO mid-run conf swap (infer_paired.sh) and NO graphs-off runs: both declared irrelevant by the user. Do not show graphs-off data anywhere.

## Output and reporting (fixed)
- Node tree: /data/ylerman/inmodel-2026-09-10/<model>/<mode>/{detect,pass1,pass2}/<arm>_<def|tun>/bench_rep*.log. Same tree fetched into this dir (no rccl logs).
- Score per pass: infer_score.py (--ref stock_def), table = 5 scored runs per arm: median tok/s, vs stock, P(sup), TTFT, TPOT, verdict.
- NEW page results-tuning/2026-08-23-1-pages/inmodel_2026-09-10.html (served on port 8411 = http://10.10.73.168:8411/): 
  1) the sc2 conf file and how it was derived (sanity-check-2: one-command rccl_tune.py, grid 1..112, A/B kept 10/18), 
  2) per model -> per mode -> pass 1 table, pass 2 table (format of today's section A/B tables),
  3) hit-map per model-mode for sc2 and dummy in the simple format: rule | hits | share, one line of uncovered sizes. No empty tables.
- Style rules from today: table titles carry model, graphs ON, radix OFF, mode, pass, arm order, node. Every number cites its CSV.
- After all done: sub-agent review, SUMMARY.md here, RUNLOG row per node (launch row at launch, completed at finish), 2026-09-10-LOG.md, commit RUNLOG/LOG/HANDOVER. Release each node when its chain is done.

## Hypotheses stated before the run
1. stock best on decode in every mode; zero rule hits. 2. sc2 > none on d32/d128/mix (+15..20% or more at d128). 3. sc2 ~ none on p8k (no rule at 47-94 MB prefill chunks).
4. dummy worst everywhere (-40..-70% decode; p8k TTFT 10x). 5. passes agree within ~1%. 6. hit-maps: 128K-256K tree,ll,64 at d32/mix; 512K-1M ring,ll,64 at d128.
Risk: stock may fall back to RCCL for prefill sizes above the custom kernel's limit -> would show as stock hits at p8k.

## Timeline file (user request 2026-09-09 evening)
- ONE file, results-tuning/2026-09-10-1-inmodel/TIMELINE.md, times in ISRAEL time (Asia/Jerusalem = UTC+3 now), only main events:
  campaign launch per node, each model/mode start, each pass start, each arm server start, mode done, node done, fetch, release.
  Built from the chain driver logs ("=== <model> <mode> <pass> arm=<arm> start <UTC>") at each heartbeat; no per-action noise.
