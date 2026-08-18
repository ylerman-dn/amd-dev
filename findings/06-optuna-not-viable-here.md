# Optuna (TPE) is not a useful search here — scalarized or single-size

Scalarized (mean busbw/max across all 18 sizes — the natural way to score one trial when
one benchmark run returns 18 numbers): TPE recovers the exact 18-winner set in only 3/10
seeds at 1 node, 7/10 at 2 node, 4/10 at 3 node (10 seeds, tol 0.5%).

Fixed to one size (removes the averaging entirely): reliability jumps to 10/10 on all 15
(node x size) combinations tested — but TPE beats blind random search in only 12/15 of
them, by a modest margin (avg 17.6 vs 22.4 configs needed, out of 30-54 in the full grid).

Either way it loses to the racing/elimination search (`adaptive_search.py`), which needs
no per-size repeat because it exploits the one property Optuna's per-trial scalar throws
away: one benchmark run scores all 18 sizes at once.

**Evidence:**
- `results-tuning/2026-08-16-13-optuna-replay/baseline_{1n,2n,3n}.json` — reproduce:
  `adaptive_search.py baseline --dataset <grid> [--nodes N] --samplers tpe --seeds 10`
- `results-tuning/2026-08-17-4-optuna-1size/aggregate.json` — reproduce:
  `adaptive_search.py baseline --dataset <grid> [--nodes N] --samplers tpe,random --seeds 10 --target-size <bytes>`

**Date:** 2026-08-16 (scalarized), 2026-08-17 (single-size).

**Scope:** replay only, not live-validated. `all_reduce_perf`, 1/2/3 nodes, grids from
2026-08-04 (1n) and 2026-08-16 (2n/3n), tol 0.5%. Single-size result covers 5 of 18 sizes
(4K, 64K, 1M, 16M, 256M).

**Falsified by:** a live run, or a replay on a new grid, where TPE matches the racing
search's exact-winner rate at fewer total runs.
