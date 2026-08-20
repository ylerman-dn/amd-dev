# racing search replaces the full grid at equal answer quality

An elimination ("racing") search over the sweep grid finds per-size winners never worse
than the full grid's, at fewer benchmark runs. Live on all_reduce (policy anchors
8,24,48): 69/105/105 runs vs the grid's 93/165/165 at 1/2/3 nodes, chosen configs
equal-or-faster than the grid winner at all 54 size-scale points in same-session data.
Replay on fresh broadcast/reduce grids exposed a failure mode — bimodal channel curves
(ch=1 wins 32K–512K multi-node by 15–53%) defeat the hill-climb — fixed by anchoring the
ladder bottom: **anchors 1,8,24,48 recovers 17–18/18 exact winners on all 9 grids owned
(worst gap 1.45%), saving 4–26% by grid size**. That is the recommended policy. Optuna
samplers are not the mechanism: scalarized TPE recovers exact winners in 3–7 of 10 seeds;
single-size TPE needs ~17.6 configs per size; random needs ~the full grid.

**Evidence:** `results-tuning/2026-08-16-13-optuna-replay/`,
`2026-08-17-{1,2,3}-optuna{1,2,3}node/` (live command.txt counts),
`2026-08-18-1-newcolls/` (grids, `anchor_fix_test.py`),
`2026-08-17-4-optuna-1size/`. Reproduce: `adaptive_search.py replay --dataset <grid>
--anchors 1,8,24,48 --margin 15 --repeat-policy 3 --tol 0.5`.

**Date:** 2026-08-16..18. **Scope:** all_reduce live 1/2/3n; broadcast/reduce replay on
fresh grids ({6}, {1,3}, {1,3,6} nodes); RCCL 2.28.3-develop:2e42aa8.

**Falsified by:** any grid where the search's pick measures >5% below the grid winner in
the same session, or a pruned combo winning a size in the full grid.
