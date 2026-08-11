# Defaults and accepted combinations, per collective

8 GPUs per node, 4K–512M, at **1, 2 and 3 nodes**. Collectives: all_reduce, all_gather,
reduce_scatter, broadcast, reduce, alltoall.

**Multi-node behaves differently from single-node — conclusions do not transfer between them.**

## What you can force

`✓` = honoured. Anything else is **silently substituted** — it runs the shown config and returns
success, with no error.

### 1 node

| requested | all_reduce | all_gather* | reduce_scatter | broadcast | reduce | alltoall |
|---|---|---|---|---|---|---|
| RING/SIMPLE | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| RING/LL | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| RING/LL128 | → SIMPLE | → SIMPLE | → SIMPLE | → SIMPLE | → SIMPLE | n/a |
| TREE/SIMPLE | → RING | → RING | → RING | → RING | → RING | n/a |
| TREE/LL | **✓** | → RING | → RING | → RING | → RING | n/a |
| TREE/LL128 | → RING/SIMPLE | → RING | → RING | → RING | → RING | n/a |

**11 usable combinations.**

### 2 and 3 nodes (identical)

| requested | all_reduce | all_gather* | reduce_scatter | broadcast | reduce | alltoall |
|---|---|---|---|---|---|---|
| RING/SIMPLE | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| RING/LL | ✓ | ✓ | ✓ | ✓ | ✓ | n/a |
| **RING/LL128** | **✓** | **✓** | **✓** | **✓** | **✓** | n/a |
| **TREE/SIMPLE** | **✓** | → RING | → RING | → RING | → RING | n/a |
| **TREE/LL** | **✓** | → RING | → RING | → RING | → RING | n/a |
| **TREE/LL128** | **✓** | → RING | → RING | → RING | → RING | n/a |

**18 usable combinations** — LL128 becomes available everywhere, and TREE becomes fully available
for all_reduce.

\* **all_gather only above its `Direct` threshold** (~8M at 1 node, ~4M at 2/3 nodes). Below it,
every request is ignored and the `Direct` addon runs regardless. This is the one place acceptance
varies with message size — everywhere else it is size-invariant.

Also, at every node count: **`NVLS` and `PAT` are silently substituted**, and
**`COLLNET_DIRECT` / `COLLNET_CHAIN` hard-error** with exit 3 `invalid usage`.

### The short version

- **TREE only ever works for all_reduce**, at any node count.
- **LL128 never works at 1 node; it works everywhere at 2+ nodes** — and is the common default there.
- **alltoall has no algo/proto at all.**
- 36 combinations are nominally sweepable per node count; **11 (1 node) / 18 (2+) are real.**

## Defaults — 1 node

`algo/proto`, channels in brackets.

| size | all_reduce | all_gather | reduce_scatter | broadcast | reduce |
|---|---|---|---|---|---|
| 4K–256K | TREE/LL (1→16) | `Direct`/SIMPLE (—) | RING/LL (1→16) | RING/LL (1→16) | RING/LL (1→16) |
| 512K–1M | RING/SIMPLE (16→32) | `Direct`/SIMPLE (—) | RING/LL (32→52) | RING/SIMPLE (16→32) | RING/SIMPLE (16→32) |
| 2M–8M | RING/SIMPLE (52→54) | `Direct`/SIMPLE (—) | RING/SIMPLE (43→52) | RING/SIMPLE (64→103) | RING/SIMPLE (64→103) |
| 16M–32M | RING/SIMPLE (56) | RING/SIMPLE (54→56) | RING/SIMPLE (54→56) | RING/SIMPLE (103→108) | RING/SIMPLE (103→108) |
| 64M–512M | `RING*`/SIMPLE (222→224) | `RING*`/SIMPLE (56→223) | RING/SIMPLE (56) | RING/SIMPLE (111→112) | RING/SIMPLE (111→112) |

Channel ceilings differ per collective: 56 all_reduce/reduce_scatter, **112** broadcast/reduce,
222–224 under WarpSpeed.

## Defaults — 2 nodes (3 nodes near-identical)

| size | all_reduce | all_gather | reduce_scatter | broadcast | reduce |
|---|---|---|---|---|---|
| 4K | TREE/LL (2) | `Direct`/SIMPLE (32) | RING/LL (1) | RING/LL (1) | RING/LL (1) |
| 8K–32K | TREE/LL (4→16) | `Direct`/SIMPLE (32) | RING/LL (1→4) | RING/LL (1) | RING/LL (1) |
| 64K–128K | TREE/LL (32→48) | `Direct`/SIMPLE (32) | RING/LL (8→16) | RING/LL128 (16→32) | RING/LL (2→4) |
| 256K | TREE/LL (48) | `Direct`/SIMPLE (32) | RING/LL (16) | RING/LL128 (48) | RING/LL128 (48) |
| 512K–1M | **TREE/LL128** (48) | `Direct`/SIMPLE (32) | RING/LL (16) | RING/LL128 (48) | RING/LL128 (48) |
| 2M–4M | **TREE/LL128** (48) | `Direct`/SIMPLE (32) | RING/LL128 (32) | RING/LL128 (48) | RING/LL128 (48) |
| 8M–32M | **TREE/LL128** (48) | RING/LL128 (32→48) | RING/LL128 (32→48) | RING/LL128 (48) | RING/LL128 (48) |
| 64M | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) |
| 128M–256M | RING/SIMPLE (48) | RING/LL128 → SIMPLE (48) | RING/LL128 → SIMPLE (48) | RING/LL128 (48) | RING/LL128 (48) |
| 512M | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) |

3-node differences: all_gather `Direct` uses **64** channels not 32; all_reduce holds RING/LL128
through 256M; reduce's LL→LL128 crossover is one step later; reduce_scatter mid-range channels
differ (21/42 at 16M/32M).

**Two structural differences from single-node:** LL128 dominates (never selected at one node), and
all_reduce defaults to **TREE/LL128** from 512K–32M where one node used RING/SIMPLE. `RING*`
(WarpSpeed) never appears multi-node at any size.

## Channels are capped at 48 from 2 nodes up

Every collective, every size, 2 and 3 nodes. Against 56 / 112 / 222–224 at one node. RCCL says so
directly:

```
RCCL MaxChannels: default capping to 48
```

14,400 occurrences in one 2-node run, **zero** in any single-node run. Real counts from INFO are
47–48. Sole exception: all_gather's `Direct` path is not capped (32 ch at 2n, 64 at 3n).

## alltoall

No algo/proto concept at any node count — `-A 1` returns `N/A`, INFO logs no channel line.
**Intermittently unstable multi-node**: 5 of 18 runs died (`rc=137`, or a genuine
`alltoall.cu.cpp:380` test failure); retries sometimes pass.

## Confidence

- Unforced defaults: **3 repeats, 0 selection mismatches**, at all three node counts.
- Accepted combinations: one run per combination per scale. Selection is deterministic, so one run
  suffices for requested-vs-selected.
- Channels: 1-node from INFO; 2/3-node from `-A 1` with the 48 cap cross-checked against INFO.
  all_gather below ~8M and all alltoall channels are **not obtainable** from either source.

`results-tuning/2026-08-03-{2,4,6,7,9}-*` — 1-node acceptance in `-7-collmap/`, 2/3-node in `-9-multinode/`
