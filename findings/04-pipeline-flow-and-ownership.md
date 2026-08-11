# The sweep→config pipeline: what each step does, and who wrote it

Seven steps. Step 4 is not in the tool.

| # | step | in → out | origin |
|---|---|---|---|
| 1 | `rccl_sweep.py` + `sweep_executor.py` | runs the benchmark per (algo, proto, channels), ×3 repeats | Halperin 2026-01-09, **patched** |
| 2 | `sweep_db.py` | SQLite store + CSV export | Halperin 2026-01-09, **patched** |
| 3 | `merge_metrics.py` | all runs → `merged.csv` (**2970 rows** = 990 configs × 3 repeats) | Halperin 2026-01-09, untouched |
| 4 | *(nothing)* | `merged.csv` → `merged_median.csv` (**990 rows**) | 🔴 **inline shell python, never saved** |
| 5 | `optimize_metrics.py` | one winner per size → `optimized.csv` (**19 rows**) | Halperin 2026-01-09, **patched** |
| 6 | `generate_tuner_config.py` | winners → plugin rules, merging adjacent sizes → `.conf` | Halperin 2026-01-09, untouched |
| 7 | `validate_tuner_config.py` | A/B config vs default, 7 repeats/arm, keeps proven rules | ylerman 2026-08-02, **patched** |

Origins from `git log --diff-filter=A`. "Patched" = modified 2026-08-04, uncommitted.

## Step 4 is a gap, not a step

`merged.csv` holds each config three times, one row per repeat:

```
131072,TREE,LL128,16,2.87    rep1
131072,TREE,LL128,16,2.89    rep2
131072,TREE,LL128,16,2.89    rep3   ->  median 2.89, one row in merged_median.csv
```

The collapse was done by python typed into a shell: group by
(collective, num_nodes, num_gpus, size_bytes, algo, proto, nchannels), take the median of the
numeric columns, add `n_repeats` and `spread_pct`. **Only the output survives** — `find
results-tuning -name '*.py'` returns nothing.

It is load-bearing: without it step 5 ranks across un-aggregated repeat rows and selects whichever
config got lucky once, out of 90 rows per size.

## Step 5 does not know which file it got

Its input is a bare positional argument (`tools/rccl-sweep/optimize_metrics.py:129`):

```python
parser.add_argument('input_file', type=str, help='Path to the input metrics.csv file')
```

Feeding it medians instead of raw repeats was a change to the **command typed**, not to the tool.
Nothing enforces it, and nothing warns if step 4 is skipped.

## Where the tunables live

- Tolerance: `optimize_metrics.py:141-146`, `-t/--tolerance`, **default 5.0** — a CLI flag, not
  hardcoded.
- Ranking: `busbw_ip` (max), changed 2026-08-04 from `time_ip_us` (min). The module docstring at
  `optimize_metrics.py:127` still describes the old behaviour.

Row counts verified against `results-tuning/2026-08-04-6-sweep3node/` · 3 nodes, 24 ranks.
