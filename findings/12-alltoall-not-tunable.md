# 12 — alltoall cannot be tuned: no algo, no proto, tuner never consulted

**Claim.** RCCL decomposes alltoall into p2p send/recv tasks, so it has no algorithm and no
protocol to choose, and the tuner plugin is never consulted for it; its only knob is the p2p
channel count, where RCCL's default is already at the plateau.

**Evidence.**

Source, at the commit our binaries are built from (2e42aa8): `src/enqueue.cc:3026-3030` expands
`ncclFuncAlltoAll` into `2 x nRanks` `p2pTaskAppend` calls, and the only tuner call site in the
file is `src/enqueue.cc:2335`, on the collective path.

Log, from a run with the plugin loaded and holding a config — across every repeat:

```
DN-TUNER/Plugin: Loaded 1 tuning configurations from .../probe.conf   (present, all 16 ranks)
TUNER/Plugin: Applied config                                          (0 occurrences)
DN-TUNER/Plugin: Config does not match                                (0 occurrences)
```

Measurement, 5 repeats per arm, median busbw versus default:

| size | p2p8 | p2p16 | p2p32 | p2p64 |
|---|---|---|---|---|
| 256K | -14.0% | +0.2% | +0.0% | +0.7% |
| 16M | -21.6% | +2.1% | +0.5% | +0.2% |
| 256M | -26.6% | -0.4% | -0.0% | +0.0% |

Reproduce: `/opt/shared/ylerman/GPU-107/a2a-2026-08-30/run_a2a.sh <jobid>`
(copy in `results-tuning/2026-08-30-1-value/alltoall-2n/run_a2a.sh`).

**Date.** 2026-08-30.

**Scope and limits.** 2 nodes, 16 ranks, `amd-mi355x-3,amd-mi355x-5` (same leaf),
RCCL 2.17.9-develop:2e42aa8. Sizes 4K-512M. The pivot path
(`RCCL_ALL_TO_ALL_PIVOT_ENABLE=1`) would route alltoall through the collective path where the
tuner does apply, but it did not engage on this topology, so it remains untested.

**Corrects.** Earlier alltoall work (`results-tuning/2026-08-26-1-a2a123`,
`2026-08-26-1-a2a45`) drove `NCCL_MIN/MAX_NCHANNELS`, which bounds the collective path and does
not control p2p channels (`src/graph/paths.cc:982-983`). Those runs changed nothing, so their
"+0.00% median" measured nothing. The verdict was right; the evidence was not.

**Falsified by.** An `Applied config` line for a plain alltoall, or a p2p channel count that
beats the default by more than the repeat spread.
