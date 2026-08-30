# alltoall, 2 nodes — re-test with the correct channel knob

Run 2026-08-30, allocation 20856, nodes `amd-mi355x-3,amd-mi355x-5` (both L1, same leaf).
`alltoall_perf -b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1`, 5 repeats per arm, 8 arms.
Raw: this directory (`*_rep*.csv`, `*.log`, `*_dbg_*.log`); remote copy
`/opt/shared/ylerman/GPU-107/a2a-2026-08-30`.

## Why this run happened

The 2026-08-26 alltoall work (`results-tuning/2026-08-26-1-a2a123`, `2026-08-26-1-a2a45`)
drove `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`. Those bound `comm->nChannels`, which is
the **collective** path. alltoall runs on the **p2p** path, whose channel count comes from
`ncclTopoComputeP2pChannels` and is controlled by `NCCL_MIN_P2P_NCHANNELS` /
`NCCL_MAX_P2P_NCHANNELS` (`src/graph/paths.cc:982-983`, RCCL 2e42aa8). The old runs
therefore changed nothing, and the "+0.00% median" they reported was a null edit, not a
measurement of alltoall.

That also explains the 40-requested / 64-executed observation recorded at the time: with
`comm->nChannels` bounded to 40, `p2pnChannels` is rounded by `pow2Up` (`paths.cc:1015`)
to 64 — the same value the default arm used. The earlier "WarpSpeed x4 cap" explanation
was wrong; the WarpSpeed multiplier is on the collective path only.

## Result 1 — the correct knob works, and default already wins

Median busbw (GB/s) over 5 repeats, percent versus the default arm:

| size | default | p2p8 | p2p16 | p2p32 | p2p64 | pivot |
|---|---|---|---|---|---|---|
| 64K | 1.58 | -12.4% | +2.4% | -0.5% | +0.8% | +0.4% |
| 256K | 6.19 | -14.0% | +0.2% | +0.0% | +0.7% | +0.1% |
| 1M | 17.95 | -13.9% | +0.3% | +0.7% | +0.5% | +0.1% |
| 4M | 42.68 | -20.2% | -3.2% | +0.2% | -0.6% | +0.3% |
| 16M | 67.23 | -21.6% | +2.1% | +0.5% | +0.2% | +0.8% |
| 64M | 82.00 | -24.8% | +0.5% | -0.0% | -0.0% | -0.0% |
| 256M | 88.79 | -26.6% | -0.4% | -0.0% | +0.0% | -0.0% |
| 512M | 88.04 | -25.1% | +2.0% | +0.3% | +0.1% | +1.6% |

Forcing 8 p2p channels costs 11-27%, so the knob demonstrably moves the number. Every
other setting sits within a few percent of the default at every size: RCCL's default
channel count is already at or above the plateau, and the only way to move busbw is down.

**No tuning gain is available for alltoall on the channel axis at 2 nodes.** Same verdict
as before, but now resting on a measurement that changes something.

## Result 2 — the tuner is never consulted for alltoall (log proof)

Arms `plugdef` (plugin + conf, default path) and `plugpivot` (plugin + conf + pivot) load
the DN tuner plugin. Across every repeat:

- `DN-TUNER/Plugin: Loaded ... tuning configurations` — present on all 16 ranks
- `TUNER/Plugin: Applied config` — **0 occurrences**
- `DN-TUNER/Plugin: Config does not match` — **0 occurrences**

The plugin was alive and holding a config, and RCCL asked it nothing. This confirms from
our own log what the source says: `src/enqueue.cc:3026-3030` expands `ncclFuncAlltoAll`
into 2 x nRanks p2p tasks, and the only tuner call site (`src/enqueue.cc:2335`) is on the
collective path.

Busbw for these arms is within +-1% of default at every size above 32K, as expected —
loading a plugin that is never called costs nothing.

## Result 3 — the pivot path did not engage

`RCCL_ALL_TO_ALL_PIVOT_ENABLE=1` changed nothing (within +-1.8% at every size, and no
`Applied config` lines in `plugpivot`). The gate at `src/collectives.cc:180-181` requires
`comm->topo->pivotA2AEnabled` and `comm->nChannels >= pivotA2ANumBiRings * 2` in addition
to the env var and the >=744KB per-rank size; one of the topology conditions evidently
does not hold here. Not chased further.

## Incidental — alltoall multi-node instability is intermittent, not a config bug

A first smoke run died with `ionic_comp_msn:1277: cqe with error 12` (rc=137) while
`all_reduce_perf` on the same pair minutes later was clean, which suggested
`NCCL_IB_QPS_PER_CONNECTION=2` (our standard, `tools/rccl-sweep/sweep_config.yaml:92`) as
the cause. A controlled probe rejected that: 3 repeats at QPS=2 and 3 at QPS=1, all six
rc=0 with zero `ionic_comp` errors (`qps/` subdirectory). The full run then lost 2 of 40
runs to the same rc=137, about 5%, matching the older "5 of 18 died" note. The failure is
intermittent and unexplained; it is not the QPS setting.

## What would falsify this

A run showing `Applied config` lines for a plain alltoall, or a p2p channel count above 64
that beats the default by more than the repeat spread.
