# Sub-agent review of run 6 and the day (2026-09-09 ~10:1xZ); actions on top, text below

## Applied
- SUMMARY -6-: 27% -> 28% (pooled 7310/5722); "no deployment value" softened to open; the three-way confound (custom AR / MSCCL / flag) stated; custom-AR-takes-the-all_reduce stated as hypothesis.
- LOG: "spread <0.1%" fixed; 27 -> 28. RUNLOG: -1- row filled (d32/p8k/d128), 28%.
- Launched run 7 (2026-09-09-7-gptoss-attr): the two attribution arms + stock+sc2 plugin with TUNING logs + stock reference.
- Not done: TTFT statistic naming across files (score.csv = mean-based, SUMMARY -1-/-3- tables = median-based, both labelled); fetching rccl logs for runs 4-6 (multi-GB; env receipt for the stock arm was grepped on the node and quoted in SUMMARY -6-).

## Review text
**Review — 2026-09-09 runs (read-only)**

**1. Run-6 recompute** (`2026-09-09-6-gptoss-stock/node/st{1,2}_*/bench_rep2..6.log`, rep 1 dropped)
- stock 7308.52 / 7312.42; sc2 5702.18 / 5735.41; none 4846.46 / 4831.28 — match `st1_score.csv`/`st2_score.csv` exactly.
- stock vs sc2: +28.2% (st1), +27.5% (st2), pooled medians 7310.07/5721.57 = **+27.8%**. "27%" is correctly derived but under-rounded; 28% is the honest figure. sc2 −22.0/−21.6%, none −33.7/−33.9% correct. TPOT 4.14 vs 5.37/5.34.
- Spread reps 2–6: stock 0.4%/0.8%, sc2 0.35%, none 0.25%; min stock (7294) > max sc2 (5750) — fully separated. Rep 1 is the low rep on all 6 servers here, so drop-first is right for this run (unlike Qwen in -4-).

**2. Which of the three carries it** (inference, not a log finding)
- Per-call arithmetic (73 all_reduce/step from the 149k calls/rank in LOG 10:5xZ, 2048 steps): none→sc2 saves 1.02 ms/step = 14 us/call, matching rccl-tests 35→21.5 us at 128K (REVIEW.md 2.2). sc2→stock saves another 1.21 ms = 16.6 us/call, i.e. stock's all_reduce runs at ~5–8 us. No RCCL algo/proto/channel choice measured in this project gets 184K near that; MSCCL-on was +20% over the best combo at 512K (RUNLOG row 419), the flag ties tree/ll/112 at ≤256K (row 421). So custom all-reduce is the most plausible carrier; flag+MSCCL should explain at most the graphs-off +3..+6.6% of rows 427–428 (which had custom AR OFF).
- Attribution arms (both `infer_many.sh`, graphs on, no plugin): (a) `IM_KEEP_CUSTOM_AR=1 RM_MSCCL=0 IM_DROP_FLOOR=1` — custom AR alone; (b) `RM_MSCCL=1`, flag present, custom AR off — the 2026-09-06 "deployment env" under graphs. Plus the value arm (c) stock + plugin sc2 with TUNING subsys: answers "does the conf add anything to stock" and its `Applied config` count at 184320 proves whether custom AR takes the decode all_reduce at all. Nothing local shows custom AR was active in run 6 (no server.log/rccl logs fetched).

**3. Does the chain support SUMMARY -6-**
- "+18% only exists after disabling custom AR and MSCCL": supported for gpt-oss d32 on ses2-1 — run-5 and run-6 sc2/none reproduce within 0.4% (5725/5720 vs 5702/5735).
- "no deployment value on single-node TP-8": overreach. One model, one mode (conc 32), one collective size; stock+conf never run; Qwen stock never measured; custom AR size threshold not in any local file (a negative claim, unverified). Still open: prefill sizes above that threshold (2.9 MB, 47/94 MB — sc2 has no rules there anyway), broadcast/reduce_scatter (HANDOVER 4: always tuner-visible), multi-node (custom AR is intra-node), conc 128–256. HANDOVER item 3 ("MSCCL default-off → tuner path becomes deployment default") needs a custom-AR caveat if 2(a) confirms.

**4. Graphs-off modes (-1-, -3-)**
- "sc2 ties none": correct. gm_d32 sc2 −0.74% with one dip (round 4: 925.9, TPOT 34.32 — `gm_d32_sc2/bench_round4.log`); qm_d32 sc2 loses 4/5 rounds (−1.4%), same tie-leaning-negative as 2026-09-06 rows 426/428. qm_p8k none has two dips (rounds 3, 6: 557/546).
- "dummies hurt prefill": correct, no overlap in any round.
- d128 dummy2 TPOT +50%: does not contradict CPU-bound for none/sc2 — none TPOT 42.45 at conc 128 vs 41.03 at conc 32 (4x batch, +3.5%) is itself the strongest CPU-bound evidence of the day. But the SUMMARY's "per-call penalty exceeds the slack" is one of two mechanisms: in qm_p8k the decode size stayed 128K and dummy2 TPOT still went 48→340 ms (prefill stalls under continuous batching leak into TPOT). Not separable from bench_serving output; SUMMARY states it as fact.

**5. Contradictions / missing files**
- LOG 08:1xZ still says Qwen graphs-on "spread <0.1%"; REVIEW showed 4.8%, only SUMMARY was fixed.
- LOG mixes guessed (09:5x–11:1xZ) and real times; bullets out of order (08:1xZ after 09:0xZ). PLAN -1- header time also guessed.
- RUNLOG row 449 still "(pending)" while d32/p8k/d128 have scores; row 454 "27%" vs recomputed 27.8%.
- TTFT statistics differ silently: SUMMARY -1-/-3- = median of per-round medians (1883), RUNLOG 451 and score.csv = median of per-round means (1828).
- HANDOVER "+1.9%" lower bound not traceable to rows 427–429 (+2.9 is the smallest there).
- Missing locally (METHODOLOGY §3): no `logs/` for runs 4/5/6 (env receipt for stock, TUNING counts); run-1/-3 `logs/` dirs are empty; chain.sh drivers and driver.log on /opt/shared only; no server.log anywhere. All six run dirs untracked while RUNLOG rows are committed (c51ff14); `.gitignore` only excludes `*.log/*.txt`, so PLAN/SUMMARY/csv could be committed. LOG has 2 uncommitted lines.

**6. Take-aways**
1. Stock SGLang 7310 vs tuned path 5722 tok/s (+27.8%, TPOT 4.14 vs 5.35) — the +18% of -4-/-5- is real only against custom-AR-off/MSCCL-off (`2026-09-09-6-gptoss-stock/node/st*_score.csv`).
2. Arithmetic points at custom all-reduce (~5–8 us/call vs RCCL's best 21.5 us), not MSCCL/flag; two cheap arms settle it — unverified until run.
3. Whether the conf adds anything on top of stock was never measured; that arm decides deployment value.
4. Graphs-off decode is CPU-bound (TPOT 41.0→42.5 for 4x batch, `qm_d32/qm_d128_score.csv`); all graphs-off in-model ties, including 2026-09-06/08, are uninformative for decode.
5. Bad confs cost real prefill: TTFT 12x/35x, −79/−92% tok/s at p8k (`qm_p8k_score.csv`) — the tuner's live risk is downside, on single-node serving.
