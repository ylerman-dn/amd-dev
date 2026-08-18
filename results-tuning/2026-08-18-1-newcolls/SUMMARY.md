# 2026-08-18-1 — broadcast + reduce grids at 1/2/3 nodes, and what they taught the adaptive search

One directory for the whole effort. Grids are the ground truth; the adaptive search was
replayed against them (never run live here).

## What ran

- **1n (node 6, job 14617, 13:02–13:24Z):** RING × {SIMPLE,LL} × 10 channels × 3 reps
  + 3 defaults, per collective = 126 runs, 126/126 rc=0.
- **2n ({1,3}, job 14606, 17:07–18:35Z):** RING × {SIMPLE,LL,LL128} × 9 channels × 3 reps
  + defaults = 84 runs per collective, all rc=0.
- **3n ({1,3,6}, jobs 14606+14814, 18:35–19:05Z):** same shape, all rc=0.
  **Node 4 is unusable**: every 3n test on {1,3,4} hung to the 600s timeout and a 2n
  {1,4} probe hung too, while {1,3} ran at ~10s/run — fault isolated to node 4's fabric
  path (`broadcast_3n_grid_rep1.log`, `remote/probe_14/`). Driver killed, node 6
  substituted (same 2×L1+1×L2 shape). **Node 4 needs cluster-admin attention.**
- Raw runs in `remote/` (462 command.txt + per-rank INFO logs); merged per-dataset CSVs
  and replay JSONs in this directory. Nodes differ from the published {5,6,7} set —
  the effort is self-contained (grid and replay share every measurement).

## Finding 1 — a real failure mode of the adaptive search, and its fix

At 2n/3n broadcast/reduce, the true winner at 32K–512K is **RING/LL128 or RING/LL with
ONE channel**, beating 8+ channels by 15–53%. The channel curve there is **bimodal**
(fast at ch=1, slow at 4–16, recovering toward 32+), which breaks the hill-climb's
rising-curve assumption: with anchors {8,24,48} the search never crosses the valley and
missed those winners by up to 52.7% (`replay_broadcast_3n.json`, `replay_reduce_2n.json`).
all_reduce curves are unimodal, which is why three live runs never hit this.

**Fix, validated on all 9 grids we own** (`anchor_fix_test.py`): add the ladder bottom to
the anchors — **anchors 1,8,24,48** ⇒ 17–18/18 exact winners everywhere, worst gap 1.45%
(one near-tie, reduce 1n), savings 4–26% by grid size:

| dataset | runs vs grid | exact |
|---|---|---|
| all_reduce 1n / 2n / 3n | 78/90 · 120/162 · 120/162 | 18/18 each |
| broadcast 1n / 2n / 3n | 60/60 · 78/81 · 72/81 | 18/18 each |
| reduce 1n / 2n / 3n | 54/60 · 69/81 · 75/81 | 17–18/18 |

The safe default policy is therefore **anchors 1,8,24,48 · margin 15 · 3 repeats ·
tol 0.5**: less spectacular than the all_reduce-only 26–38%, but correct on every
collective tested. Savings scale with grid size — small RING-only grids leave little to
prune.

## Finding 2 — RCCL lead: 1 channel is ~2× at bcast/reduce multi-node mid-sizes

RING/LL128/1 at 32K–512K (2n and 3n, both collectives) outperforms every higher channel
count by 15–53% in these grids. Untuned, un-A/B'd — a candidate for the next tuner
config round (backlog, needs `validate_tuner_config.py` pass).

## Finding 3 — Halperin detector at 3n (closes the earlier gap)

On the fresh default curves: **broadcast 3n fires at 64K (−20.5% below running max)**;
reduce 3n and both 2n curves stay clean at 10% (`hotspots_mn_th10.csv`). Combined with
2026-08-17-4: the 512K 1-node dip does not recur multi-node, but 3n broadcast has its own
distinct dip at 64K. (3n curves measured on {1,3,6}, exporter active, medians of 3.)

## Reproduce

```
python3 results-tuning/2026-08-18-1-newcolls/analyze_1n.py
python3 results-tuning/2026-08-18-1-newcolls/analyze_mn.py
python3 results-tuning/2026-08-18-1-newcolls/anchor_fix_test.py
```
