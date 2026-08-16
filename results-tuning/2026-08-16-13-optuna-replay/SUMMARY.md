# 2026-08-16-13 — Can Optuna cut the sweep run count? (replay phase, no cluster time)

**Question.** The full grid per (collective, node-count) is 30 configs / 90 runs at 1 node,
54 configs / 162 runs at 2-3 nodes (plus default runs). Can an adaptive search find the same
per-size winners with fewer runs?

**Framing.** One benchmark run returns all 18 sizes, so a trial scores 18 objectives — this is
18 argmax problems sharing every evaluation, not one optimization. The tested search is
successive elimination (`tools/rccl-sweep/adaptive_search.py`): anchor channels per algo/proto
combo → drop combos beaten at EVERY size by >margin → per-size channel hill-climb restricted to
sizes where the combo is a contender, plus a downward walk to serve the min-channels tie-break →
select with the existing `optimize_metrics.py` rule (tol 0.5%). The selection reimplementation
reproduces `2026-08-04-1-sweep1node/optimized.csv` 18/18 (`adaptive_search.py selftest`).

**Method.** Replay = simulate the search against already-measured full grids, feeding stored
repeats back one at a time. Datasets: `2026-08-04-1-sweep1node/merged.csv` (1 node) and
`2026-08-16-12-variance/variance.json` nodes 2 and 3. NOTE: `variance.json` node "1" is
byte-identical to the 2026-08-04 merged.csv repeats (verified cell-by-cell, 540/540 identical),
so there are three independent datasets, not four.

## Results (this folder, replay_*.json)

| policy | dataset | runs | full | saving | exact winners | worst busbw gap |
|---|---|---|---|---|---|---|
| anchors 8,24,48 · margin 15 · 3 repeats | 1n 08-04 | 66 | 90 | 27% | 18/18 | 0.00% |
| " | 2n 08-16 | 102 | 162 | 37% | 18/18 | 0.00% |
| " | 3n 08-16 | 102 | 162 | 37% | 18/18 | 0.00% |
| anchors 8,48 · margin 10 · 1 explore + finalists to 3 | 1n 08-04 | 44 | 90 | 51% | 17/18 | 0.17% |
| " | 2n 08-16 | 59 | 162 | 64% | 18/18 | 0.00% |
| " | 3n 08-16 | 65 | 162 | 60% | 17/18 | 0.00% |

The conservative policy recovers the exact 18/18 winners on every dataset at every margin
10–25 tried (`policy_tune.json`); the aggressive one halves-plus the runs and its rare misses
are ties within 0.17% of the grid winner's bandwidth. 1+2 numbers are the worst case over
which stored repeat plays the explore role (rotations 0/1/2).

## Optuna verdict (baseline_*.json)

Optuna's samplers do not fit this objective. TPE on a scalarized objective (mean busbw
normalized per size) reaches within-5%-everywhere quickly (median 12.5 configs at 1n) but
usually NEVER recovers the exact winners (7/10 seeds at 1n, 6/10 at 3n, even after covering
the space) — the per-size argmax + min-channels tie-break is not expressible as one scalar.
Random needs nearly the full grid (median 29/30, 50/54, 51.5/54 configs). The run savings come
from the elimination structure, not from a smarter sampler; Optuna adds bookkeeping, not value,
here.

## Reproduce

```
python3 tools/rccl-sweep/adaptive_search.py selftest --dataset results-tuning/2026-08-04-1-sweep1node/merged.csv
python3 tools/rccl-sweep/adaptive_search.py replay --dataset results-tuning/2026-08-04-1-sweep1node/merged.csv --anchors 8,24,48 --margin 15 --repeat-policy 3
python3 tools/rccl-sweep/adaptive_search.py replay --dataset results-tuning/2026-08-16-12-variance/variance.json --nodes 2 --anchors 8,48 --margin 10 --repeat-policy 1+2
python3 tools/rccl-sweep/adaptive_search.py tune --datasets "..." --anchor-sets "8,48 4,32 8,32 16,48 4,16,48 8,24,48 2,8,32 4,24,56" --margins 10,15,20,25 --repeat-policies 3,1+2
python3 tools/rccl-sweep/adaptive_search.py baseline --dataset <ds> [--nodes N] --seeds 10
```

**Status: replay only — simulation on measured data. Everything here is bounded by what the
three grids contain. Live out-of-sample validation (new cluster runs) is the next folder.**
Chosen for live: anchors 8,24,48, margin 15, 3 repeats, tol 0.5 — the deliverable's criterion
is exact winner recovery, so the conservative policy goes live; the aggressive one is reported
as replay evidence only.
