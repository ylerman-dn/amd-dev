# Which algo/proto/nchannels requests does RCCL actually honour?

Job 14048, `amd-mi355x-4`, 1 node, 8 ranks × 1 GPU, all_reduce, sizes 256M + 512M.
`all_reduce_perf -b 256M -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1`, launched with
`srun --mpi=pmix --ntasks-per-node=8`. Forcing via `NCCL_ALGO` / `NCCL_PROTO` /
`NCCL_MIN_NCHANNELS`+`NCCL_MAX_NCHANNELS`, i.e. the same mechanism as
`sweep_executor.py:119-129`.

All 19 runs: rc=0, 2 data rows each, `#wrong`=0, 16 fields per row. Selection was identical
at both sizes in every run, so one column suffices below.

## Result

| requested algo/proto | reported | verdict |
|---|---|---|
| RING / SIMPLE | `RING*` / SIMPLE | **honoured** |
| RING / LL | `RING*` / LL | **honoured** |
| TREE / LL | TREE / LL | **honoured** |
| RING / LL128 | `RING*` / **SIMPLE** | substituted — proto ignored |
| TREE / SIMPLE | **`RING*`** / SIMPLE | substituted — algo ignored |
| TREE / LL128 | **`RING*`** / **SIMPLE** | substituted — both ignored |

**nchannels is always honoured exactly**, including the deliberately awkward 19
(non-power-of-2, not a divisor of the default 56): 19→19, 32→32, unset→56. It stays honoured
even when algo/proto are being substituted, so the channel knob is independent of the
algo/proto knob.

`LL128` is never honoured for all_reduce on this stack — it always becomes SIMPLE.
`TREE` only survives with `LL`.

## Consequence for the sweep

Six requested algo/proto pairs collapse to **three** distinct real configurations, because
RING/SIMPLE, RING/LL128, TREE/SIMPLE and TREE/LL128 all end up as `RING*`/SIMPLE.

Across the 18 forced runs that means:
- **9 rows would carry a correct label** (RING/SIMPLE, RING/LL, TREE/LL × 3 channel levels).
- **9 rows would be mislabelled** (RING/LL128, TREE/SIMPLE, TREE/LL128 × 3) — the sweep records
  what it asked for, so exactly half the matrix is fiction.
- Duplicate work: 12 of the 18 runs are re-measuring `RING*`/SIMPLE at 3 channel counts.

Evidence that these are the *same* run, not merely similar: at 32 channels
`RING_SIMPLE_ch32` and `RING_LL128_ch32` report 326.56 and 326.60 GB/s; `TREE_SIMPLE_ch32`
and `TREE_LL128_ch32` report 325.99 and 326.28.

## Cross-check against `unsupported_combos.yaml`

| combo | file says | measured | |
|---|---|---|---|
| TREE + SIMPLE @ 1 node | unsupported (`max_nodes: 1`) | substituted | file **correct** |
| RING + LL128 | nothing | substituted | rule **missing** |
| TREE + LL128 | nothing | substituted | rule **missing** |

The one rule the file has for this case is right; it has no LL128 rule at all. Since
`autotune_config.yaml:29-32` sweeps `SIMPLE, LL, LL128` by default, the whole LL128 arm is
wasted and mislabelled today, and `combo_validator.py` does not catch it.

## Parser/mapping issue

The algo string is `RING*` with a trailing asterisk in every RING run (TREE reports plain
`TREE`). `config_generator.py`'s `ALGO_MAP` keys on `'RING'`, so `RING*` misses and maps to
`-1` — the generated tuner config would silently say "use RCCL default" wherever the real
answer was ring. Whatever comparison we add has to normalise the suffix.

## Single-shot throughput — NOT a claim

Recorded for orientation only; one run per point, no repeats, no statistics. Do not cite.

| config | 256M | 512M |
|---|---|---|
| default (`RING*`/SIMPLE/56) | 390.25 | 397.12 |
| `RING*`/SIMPLE/32 | 326.60 | 331.98 |
| `RING*`/SIMPLE/19 | 210.68 | 210.85 |
| `RING*`/LL/56 | 129.14 | 129.43 |
| TREE/LL/56 | 108.85 | 109.42 |
| TREE/LL/19 | 39.36 | 39.60 |

Direction of travel worth noting for later: at these two large sizes nothing beat the default,
and reducing channels cost throughput monotonically. That is a hypothesis to test properly with
repeats, not a result.
