# Sub-agent review of sanity-check-2 (2026-09-08 ~17:45Z), verbatim, plus what was done about it

## Corrections applied to SUMMARY.md
- 4M is a real -1.1% loss (P=0.00), not parity; 8M/16M+ are the parity cases. Fixed.
- "search finds RCCL's own choice" only true 16M..512M. Fixed.
- 4M: KEPT yesterday (RING/LL/96->94, +4.1%), dropped today: RING/LL >80 never run by the search. Stated.
- 128K and 256K did change vs check 1. Stated. A/B 9m39s.

## Parked for the tool (after the campaign): stage-3 hill-climb stops on plateaus (RING/LL >=84 at 1M-4M unreachable);
## tie-break on executed channels; tol 0.5% below 3-repeat noise; exec-truth fetch for search runs inside rccl_tune;
## same-exec null test (~0.3% config-arm penalty) to be reported per run.

## Review text
1. Correctness: all 18 A/B medians/gains/P(sup) recomputed and match; search-default medians match; winner medians and executed-in-search match 18/18. Wrong: 4M labelled parity (it is -1.1%, all 9 below all 9, P=0.00); "search finds RCCL's own choice" only holds 16M-512M (4M picked RING/SIMPLE/72->exec 64 vs default 103; 8M 96->94 vs 103); 128K went +47.1 -> +60.9% and 256K switched algo, so "agree within a few points" was overstated; A/B was 9m39s.
2. PLAN vs actual: ETA lines copied by sed (A/B ~65 min vs 9 actual; search 45 vs 53); search_raw/r*/merged_exec.csv built by hand on node 8 (no --exec-from-logs call in rccl_tune/adaptive_search), an undeclared step behind SUMMARY column 4; tool version moved 3b56455 -> efa0117 between checks (dir numbering only); default runs after all 120 cells, not interleaved.
3. Default consistency: identical executed algo/proto/channels at 18/18 in all four sources incl. yesterday; max spread 1.8% today, 3.0% incl. yesterday (4K, one print unit). All four on node 8: "independent" overstates. Config arm same-exec sizes (14/18) yesterday vs today within 1%. Where rules differ (4M) the "matches yesterday" claim breaks.
4. Grid effect: changes at 128K/1M/2M/4M come from a step in RING/LL (exec 52 -> 45.8, exec 64 -> 61.6 at 1M; yesterday RING/LL 84 -> 116, 96 -> 136 at 4M). Check 2 never ran RING/LL above 80: adaptive_search.py:311-345 stage 3 probes only grid neighbours of the incumbent and 72~80 is a plateau at 2M/4M. Tie-break at 4M: band held RING/SIMPLE 96 (130.39, exec 86), 104 (130.38, exec 103), 72 (129.89, exec 64); fewest-requested picked exec 64, rejected by the A/B at P=0.00.
5. Risks: tol 0.5% below 3-repeat noise everywhere (1M 59.91/61.64/61.65; 2M 89.93/91.74/93.07). Null control: same-exec sizes 16M/32M/64M read -0.2..-0.35%, P 0.03/0.05/0.10 (yesterday 16M P=0.00) - systematic ~0.3% config-arm penalty. Requested->executed depends on size: RING/SIMPLE/112 executes 1,1,2,4,...,86,103,108,111,112; "ch112" means "RCCL's cap".
6. For check 3: confirm from live_runs.log whether RING/LL >= 84 was ever run at 1M-4M; list the 0.5% band with executed channels per winner and flag picks whose exec differs from the band leader; check exec counts for 128-224 in merged_exec before crediting any >112 winner, and report the same-exec P(sup) null test.
