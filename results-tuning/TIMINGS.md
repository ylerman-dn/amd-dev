# GPU-107 phase timings

Wall-clock cost of each phase, so future runs can be planned rather than guessed.

**Method:** the sweep driver writes one log per phase; that log's mtime is the phase's completion
time, so `mtime[n] - mtime[n-1]` is the phase duration. The driver's own mtime is the launch time.
Durations are therefore real measurements, not estimates — except where marked.

Hardware for everything below: `amd-mi355x-7` (XAI, MI355X), 8 GPUs, Slurm job 14138.
RCCL 2.28.3-develop:2e42aa8.

## Sweep phases (config creation)

Each "ring" phase = 20 runs (RING × {SIMPLE,LL} × 10 channel values).
Each "tree" phase = 10 runs (TREE/LL × 10 channel values).
Each "def" phase = 1 unforced reference run.
Every run covers all 18 message sizes (4K–512M, f=2).

| sweep | phase | duration | runs | per run |
|---|---|---|---|---|
| **`-n 5`** (superseded) | rep1 ring | ~146s* | 20 | ~7s |
| | rep1 tree | 146s | 10 | 15s |
| | rep1 def | 11s | 1 | 11s |
| | rep2 ring | 172s | 20 | 9s |
| | rep2 tree | 102s | 10 | 10s |
| | rep3 ring | 173s | 20 | 9s |
| | rep3 tree | 101s | 10 | 10s |
| | **total** | **~16 min** | **93** | **~10s** |
| **`-n 20`** | rep ring | 230–231s | 20 | ~11.5s |
| | rep tree | 153–154s | 10 | ~15.3s |
| | rep def | 11s | 1 | 11s |
| | **total (3 reps)** | **~20 min** | **93** | **~13s** |

\* the `-n 5` rep1 ring figure is unreliable — the phase-1 timing was taken across a
tool-redeploy boundary. All other figures are clean.

**`-n 20` costs ~25% more wall time than `-n 5`** (13s vs 10s per run) for 4× the iterations —
because per-run fixed cost (process start, RCCL init, 18 size setups) dominates over the
iteration loop. Cheap insurance against the 10–20% spread `-n 5` produced.

## Post-processing (config creation, after the sweep)

| step | script | duration |
|---|---|---|
| merge 9 sessions | `merge_metrics.py` | <5s |
| median per (config,size) | inline python (not in the tool) | <5s |
| pick winner per size | `optimize_metrics.py` | <5s |
| emit config | `generate_tuner_config.py` | <5s |

Negligible — all four together are under 20s. Sweep time dominates entirely.

## Per-scale budget — measured

Durations from `driver.sh`/`run_ab.sh` mtime (launch) to `PROGRESS` mtime (completion).

| scale | sweep | runs | post | A/B | A/B runs | presentation |
|---|---|---|---|---|---|---|
| 1 node | **1186s** (19.8 min) | 93 | <20s | **161s** (2.7 min) | 14 | ~18 min (subagent) |
| 2 nodes | **1889s** (31.5 min) | 165 | <20s | **191s** (3.2 min) | 14 | ~18 min (subagent) |
| 3 nodes | **~2200s** (36.7 min) | 165 | <20s | **~190s** (3.2 min) | 14 | ~17 min (subagent) |

**End-to-end for all three scales: ~1h35m of cluster time** (sweeps 1186+1889+2200 = 5275s,
A/Bs 161+191+190 = 542s), plus ~50 min of subagent time for the three HTML pages (which needs no
cluster). Wall-clock from first sweep launch (12:48Z) to last A/B finish (15:04Z) was 2h16m,
the gap being analysis and page-building between phases.

Per-run cost:

| scale | sweep per run | A/B per run |
|---|---|---|
| 1 node | 12.8s | 11.5s |
| 2 nodes | 11.4s | 13.6s |
| 3 nodes | ~13s (partial) | — |

**The A/B is cheap — 3 minutes.** All the time is in the sweep, and the sweep scales with the
matrix, not the node count: 1 node swept 3 honoured algo/proto combos (93 runs), 2 and 3 nodes
swept all 6 (165 runs). Per-run cost barely moves with node count (11–13s), so multi-node is not
intrinsically slower here — the matrix is just bigger because more combinations are honoured.

Sizing rule for planning: **~13s per sweep run, ~13s per A/B run.** A scale costs
`(combos x channel_values x repeats) x 13s` for the sweep plus `~3 min` for the A/B.

## Overhead not in the table

Worth recording because it dominated the actual elapsed time today:

| what | cost |
|---|---|
| `-n 5` sweep, discarded (wrong iteration count) | 16 min |
| `-n 20` sweep, killed at 70/93 by a mid-flight tool redeploy | 16 min |
| **wasted** | **~32 min** |

Both were self-inflicted. The frozen-tool-copy pattern (`<run>/tool/`) now prevents the second;
the CLAUDE.md rule on not changing measurement parameters prevents the first.

## Allocation

| job | nodes | held | note |
|---|---|---|---|
| 14138 | amd-mi355x-5,6,7 | 12:05Z → 20:05Z (`-t 480`) | 1-node = node7, 2-node = {5,7}, 3-node = {5,6,7} |

Earlier jobs 14048 / 14059 / 14067 all ended in `TIMEOUT` rather than explicit release, holding
nodes idle for hours. Release with `scancel <jobid>` when done.
