# 2026-08-19-2 — all_gather + reduce_scatter grids at 1/2/3 nodes (last two tunable collectives)

**What ran (job 20672, published trio: 1n=node7, 2n={5,7}, 3n={5,6,7}).** Full grids,
RING × protos × channels, 3 repeats + defaults = 462 runs, 36/36 driver blocks rc=0
(`remote PROGRESS`; raw runs on `/opt/shared/ylerman/GPU-107/optuna-live-2026-08-19/agrs/`).
Merged datasets, replay JSONs, detector reports in this directory
(`analyze_agrs.py` reproduces).

## Finding 1 — all_gather is only tunable at large sizes on this build

Below 16M (1n) / 8M (2-3n), RCCL routes all_gather to its **Direct** algorithm regardless
of the forced RING request — every such row comes back `substituted=1`
(requested RING → measured Direct). The tunable space is only the top 6 (1n) / 7 (2-3n)
sizes. This matches the Direct-allgather behavior noted in findings/02 and closes it with
grid-scale evidence.

## Finding 2 — adaptive search (anchors 1,8,24,48): exact winners on both collectives

| dataset | tunable sizes | exact | runs vs grid | saving |
|---|---|---|---|---|
| all_gather 1n / 2n / 3n | 6 / 7 / 7 | 6/6 · 7/7 · 7/7 | 30/60 · 51/81 · 51/81 | 50% · 37% · 37% |
| reduce_scatter 1n / 2n / 3n | 18 each | 18/18 all | 54/60 · 72/81 · 69/81 | 10% · 11% · 15% |

Worst gap 0.0% everywhere. With these, the racing policy is now exact-or-tie on **all 15
grids measured in this project** (all_reduce ×3, broadcast ×3, reduce ×3, all_gather ×3,
reduce_scatter ×3).

## Finding 3 — defaults are healthy here (detector)

Hotspot detector on the fresh default curves: **0 dips at the 10% threshold** for both
collectives at all scales; at 5% a single marginal one (all_gather 3n 128M, −6.1%).
Consistent with the known small headroom (+7.5%/+1.7%): these two collectives' defaults
are basically fine — the tuning money stays in broadcast/reduce and small-size all_reduce.

## Reproduce

```
python3 results-tuning/2026-08-19-2-agrs/analyze_agrs.py
python3 tools/rccl-sweep/detect_hotspots.py results-tuning/2026-08-19-2-agrs/agrs_default_curves.csv --threshold 0.10
```
