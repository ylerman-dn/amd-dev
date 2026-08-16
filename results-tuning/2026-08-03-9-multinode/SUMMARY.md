# Multi-node (2 and 3 nodes): defaults and accepted combinations

Job 14067, nodes `amd-mi355x-3,5,6`. 2-node = nodes 3,5; 3-node = 3,5,6 (subset of the same set,
so the scales are comparable). 8 ranks/node. 4K–512M f=2, `-n 20 -w 5 -c 1 -A 1 -Z csv -X`,
`NCCL_DEBUG=INFO` → per-rank file. Runner: `run_mn.sh` (kept alongside the data).

Per scale: 6 collectives × 6 algo/proto pairs, 6 collectives × 3 unforced repeats, 4 exotic probes
on all_reduce. **111 of 116 runs succeeded**; all 5 failures were alltoall (see below).

Unforced selection was **identical across all 3 repeats** for every collective at both scales.

## Headline: multi-node inverts the single-node acceptance result

| requested | 1 node | 2 nodes | 3 nodes |
|---|---|---|---|
| RING/SIMPLE | honoured | honoured | honoured |
| RING/LL | honoured | honoured | honoured |
| TREE/LL | honoured | honoured | honoured |
| **RING/LL128** | substituted → SIMPLE | **honoured** | **honoured** |
| **TREE/SIMPLE** | substituted → RING | **honoured** | **honoured** |
| **TREE/LL128** | substituted → RING/SIMPLE | **honoured** | **honoured** |

For all_reduce, **all six combinations work at 2+ nodes** — versus three at one node.

Consequences for our own earlier conclusions:
- `findings/02`'s "LL128 never works, for any collective, at any size" is **single-node only**.
  A global LL128 rule in `unsupported_combos.yaml` would have been wrong.
- `unsupported_combos.yaml`'s `TREE + SIMPLE, max_nodes: 1` rule is now confirmed in **both**
  directions: blocked at 1 node, available at 2+.
- The other collectives are unchanged: TREE is still never honoured for reduce_scatter, broadcast
  or reduce, at any node count. LL128 *is* now honoured for them.
- COLLNET_DIRECT / COLLNET_CHAIN still hard-error (exit 3). NVLS and PAT still substitute silently.

## The 48-channel cap: confirmed, with a log line

Previously an open question resting on disassembly. Now settled:

```
RCCL MaxChannels: default capping to 48
```

Fires **14,400 times** in a single 2-node run and **zero** times in any single-node run. Real
channel counts from INFO cap at 47–48 across all sizes, against 56 / 112 / 222–224 single-node.

The gate is node count, as the disassembly suggested. One exception: all_gather's `Direct` path is
**not** capped — it reports 32 channels at 2 nodes and **64** at 3 nodes, both outside the 48 limit.

## Defaults per size — 2 nodes

`algo/proto (channels from -A 1)`. Determinism: 3/3 repeats identical for every collective.

| size | all_reduce | all_gather | reduce_scatter | broadcast | reduce |
|---|---|---|---|---|---|
| 4K | TREE/LL (2) | `Direct`/SIMPLE (32) | RING/LL (1) | RING/LL (1) | RING/LL (1) |
| 8K | TREE/LL (4) | `Direct`/SIMPLE (32) | RING/LL (1) | RING/LL (1) | RING/LL (1) |
| 16K | TREE/LL (8) | `Direct`/SIMPLE (32) | RING/LL (2) | RING/LL (1) | RING/LL (1) |
| 32K | TREE/LL (16) | `Direct`/SIMPLE (32) | RING/LL (4) | RING/LL (1) | RING/LL (1) |
| 64K | TREE/LL (32) | `Direct`/SIMPLE (32) | RING/LL (8) | RING/LL128 (16) | RING/LL (2) |
| 128K | TREE/LL (48) | `Direct`/SIMPLE (32) | RING/LL (16) | RING/LL128 (32) | RING/LL (4) |
| 256K | TREE/LL (48) | `Direct`/SIMPLE (32) | RING/LL (16) | RING/LL128 (48) | RING/LL128 (48) |
| 512K–1M | TREE/LL128 (48) | `Direct`/SIMPLE (32) | RING/LL (16) | RING/LL128 (48) | RING/LL128 (48) |
| 2M–4M | TREE/LL128 (48) | `Direct`/SIMPLE (32) | RING/LL128 (32) | RING/LL128 (48) | RING/LL128 (48) |
| 8M | TREE/LL128 (48) | RING/LL128 (32) | RING/LL128 (40) | RING/LL128 (48) | RING/LL128 (48) |
| 16M | TREE/LL128 (48) | RING/LL128 (40) | RING/LL128 (32) | RING/LL128 (48) | RING/LL128 (48) |
| 32M | TREE/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) |
| 64M | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) |
| 128M | RING/SIMPLE (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) | RING/LL128 (48) |
| 256M | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/LL128 (48) | RING/LL128 (48) |
| 512M | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) | RING/SIMPLE (48) |

## Defaults per size — 3 nodes

Nearly identical to 2 nodes. The only differences: all_gather's `Direct` uses **64** channels
instead of 32; all_reduce stays on RING/LL128 at 128M–256M where 2 nodes had already moved to
RING/SIMPLE; reduce's LL→LL128 crossover shifts one step later; and reduce_scatter's mid-range
channel counts differ (21/42 vs 32/48 at 16M/32M). Full per-size data in the CSVs and logs.

## Multi-node defaults look nothing like single-node

Two structural differences worth stating plainly:

1. **LL128 dominates.** Single-node never selected LL128 at any size, for any collective.
   Multi-node it is the default across most of the range.
2. **all_reduce's default is TREE/LL128** from 512K to 32M, where single-node used RING/SIMPLE.
3. **WarpSpeed never appears.** No `RING*` at any size, either scale — consistent with the
   thresholds (AR 64 MiB) being evaluated against something that does not trigger here, or
   WarpSpeed being single-node only. Not investigated.

Any tuning conclusion drawn at one node therefore does not transfer to two.

## busbw from this batch is NOT citable at small/mid sizes

Measured 2026-08-04 from the 3 unforced repeats. Worst spread across repeats:

| scale | worst spread | where |
|---|---|---|
| 1 node | 2.9% | 16K |
| 2 nodes | **101%** | 4K, 8K, 16K |
| 3 nodes | **46.6%** | 512K (also 8M 9.1%) |

**Cause identified — a startup stall, not contention.** The 2-node "101%" is not jitter: rep2 took
**18,889 µs** at 4K where rep1 took **37.8 µs** — a ~500× stall on the first three sizes of that
one run, recovering from 32K onward. (The `0.00` busbw in the table is display rounding of
0.0004, not a missing value.) rep1 and rep3 agree to <1% at those sizes.

A stall confined to the opening sizes of a single run points at **inter-node connection wireup**,
since each run is a fresh process doing its own setup. An earlier version of this section blamed
co-tenant fabric contention (jobs 14028/14029 on nodes 1/7/8, which do share our leaf switches and
spine uplinks). That was speculation and does not fit the evidence — a steady neighbour load would
not confine itself to the first three sizes of one repeat out of three.

Where all three repeats agree they agree tightly (0.1–3%), and the medians are physically sensible
(1n > 2n > 3n). The problem is purely that 1 run in 3 contains an unexplained stall, and 3 repeats
is too few to tell how often it occurs or which points it touched.

Selection data is unaffected — it is deterministic and identical across all repeats.
To get citable multi-node busbw: 7+ interleaved repeats on an idle partition.

## alltoall is unstable multi-node

5 failures, all alltoall: 1 at 2 nodes, 4 at 3 nodes, all `rc=137` (SIGKILL) or a genuine
`Test failure .../alltoall.cu.cpp:380`. Successful runs report `N/A` for algo/proto, same as
single-node. Retries sometimes pass, so it is intermittent rather than a hard failure. Recorded
rather than chased.

One extra note: the 2-node `alltoall__REF_rep1` CSV is absent because my first rerun attempt had a
shell-quoting error that lost the output path. rep2 and rep3 are complete and agree, and alltoall
has no algo/proto data to lose, so it was not re-run again.

## Harness lesson — cost 5 failed attempts

`srun --export` uses **commas as its own separator**, so any comma-containing value passed through
it is silently truncated: `OMPI_MCA_btl=self,vader,tcp` became `btl=self`, and
`NCCL_IB_HCA=ionic_0:1,ionic_1:1,…` became a single HCA. That produced a cascade of misleading
errors (btl_tcp connect failure → MPI_Init failure → btl_ofi abort → "processes unable to reach
each other"), none of which pointed at the real cause. Fix: export everything inside a
`bash -c` wrapper. `run_mn.sh` does this and works. Added to `CLAUDE.md`.

Multi-node also needs the fabric env at all (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`,
`NCCL_IB_GID_INDEX`, …) plus `OMPI_MCA_btl=self,vader,tcp` and
`OMPI_MCA_btl_tcp_if_include=enp81s0f1np1` — without these, 2+ node runs fail entirely.

## Files

`2n/` and `3n/`: 55 and 56 CSVs, 58 stdout logs each, `debug_files.tgz` (per-rank INFO logs).
`smoke_*.log`: the 7 smoke attempts, kept as the record of the wireup diagnosis. `run_mn.sh`.

---

# Findings update — APPROVED AND WRITTEN 2026-08-03

`findings/02` currently states single-node results as though general. It needs a node-count axis:

1. Retitle the accepted-combinations table to **single node**, and add the multi-node column set
   above showing all six all_reduce pairs honoured at 2+ nodes.
2. Correct "LL128 never works" → **"LL128 never works at one node; it is the common default at
   2+ nodes."**
3. Add the 48-channel cap as a confirmed fact: ≤48 at 2+ nodes, 56/112/222–224 at one node,
   evidenced by the `RCCL MaxChannels: default capping to 48` message (14,400 hits at 2n, 0 at 1n).
   This also **removes** the corresponding entry from `findings/README.md`'s open questions.
4. Note that multi-node defaults differ structurally from single-node, so single-node tuning
   conclusions do not transfer.
5. Add alltoall's multi-node instability.
