# Sub-agent review of sanity-check-3 (2026-09-08 ~18:35Z), verbatim below, plus what was done

## Corrections applied to SUMMARY.md
- 128/144 executed 128 at every size >=1M (not >=2M).
- Added the omitted material fact: env-path exec-128 is worse than default at 8M-64M by 11..31% (only 2M/4M +2%).
- "plugin's 112 is catastrophic" softened: unsettled; the settling A/B (4M rule ring,simple,112) is PROPOSED, not run.
- Why the search stopped at 144: adaptive_search improve_eps 2% (hard-coded, not in POLICY). RING/LL never above 72 (not 80). A/B 9m38s.

## Take-aways of the three checks (reviewer §5, agreed)
- sc2 and sc3 execute identically at every rule -> in the Qwen run they are a built-in null pair.
- Grid cap, not the tool, produced sc1's landmines and sc3's 4M rule; the A/B gate caught every one (P=0.00).
- Conf channel labels above the executed count are noise; conf text is not reproducible, conf behaviour is.
- The plugin cannot exceed the node's default channel count; >112 in a conf is untested territory until the settling A/B runs.

## Parked tool items (after campaign): expose improve_eps in POLICY/PLAN; clamp generate_tuner_config channels to the node max;
## stage-3 plateau escape; tie-break on executed channels; exec-truth fetch inside rccl_tune; RUNLOG row at launch.

## Review text
1. Correctness: all 18 A/B medians/gains match (4M 129.89/34.79/-73.2%, every cfg value below every def value). Search 4M RING/SIMPLE medians and exec counts (48->47, 56->52, 72->64, 80/84->74, 96->86, 104/112->103, 128/144->128) match. Default medians 18/18 match. Wrong: req 128/144 executed 128 at every size >=1M, not >=2M. Omitted: exec-128 collapsed at 8M-64M vs req-112 (8M 179.6 vs 201.4, 16M 215.2 vs 273.2, 32M 234.4 vs 322.4, 64M 247.9 vs 359.2; 128M-512M -2..-4%; only 2M/4M +2%). Minor: A/B 9m38s; RING/LL never above 72 (r0090-92), not 80.
2. Claim (c): supported only as "executed counts differ" (env 144->128, plugin 144->112, ch_trimmed=yes). Not supported: "112 is catastrophic" - nothing else measured 112 executed at 4M (env 112 trimmed to 103 -> 130.1); in check 2 the plugin trimmed like env (72->64). Alternatives: over-request > comm nChannels trips a different RCCL branch (WarpSpeed-style, findings/11); 112 untrimmed at 4M intrinsically bad (unlikely, 103 and 128 both ~130); env creates 128 channels at init while the plugin can only choose among the existing 112 (semantics, predicts capping not collapse). Settling measurement: A/B the conf with 4M rule ring,simple,112, 9 reps, container, flag-off; read that arm's TUNING lines at 4M.
3. Search stop: adaptive_search.py:220-222,296-308 moves the incumbent only on >2% (improve_eps_pct=2.0, hard-coded). 128/112 = 1.0225 at 4M -> incumbent 128 -> 144 tried; 144/128 = 1.0054 -> stop. 160-224 never tested; the excursion hinged on 2.25% vs 2.0% on a 3-repeat cell (133.02/131.03/133.9). Same mechanism kept RING/LL at 72. 4M pick: band floor 133.07, ch128 = 133.02 out by 0.05 -> 144 won outright.
4. Tables verified 18/18; null test P 0.23-0.47 this time (check 2: 0.03-0.10) - penalty smaller, not reproduced as significant. 16M correctly excluded (103 vs 108).
5. Three checks: sc2/sc3 identical executed behaviour at every rule; grid cap decided the landmines, A/B caught all; conf labels above exec are noise; plugin cannot exceed the node default channel count.
6. Qwen: decode all_reduce 131072 B -> sc2/sc3 rule 131072-262144 tree,ll,64 (+62%, 6.56->10.65 GB/s), sc1 32768-131072 tree,ll,40 (exec 32, +47%); 1.59M hits = 2048 steps x 8 ranks x 96. Ceiling ~13 us x 96 = ~1.3 ms/step vs TPOT ~40 ms -> expect ~3%. Ragged decode batches (e.g. 31 tokens = 126976 B) hit no rule: check total all_reduce calls vs hits. Prefill 512 tokens = 2 MB -> sc2/sc3 ring,ll,72 (+15.9%), sc1 none: TTFT may separate sc1 from sc2/sc3.
