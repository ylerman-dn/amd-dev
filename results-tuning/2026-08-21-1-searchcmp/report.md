# Search-method comparison — 15 grids, 5 collectives × 1/2/3 nodes

All methods replayed **offline against the same measured grid data** (no new cluster runs). Sources: all_reduce `2026-08-04-{1,4,6}`, broadcast/reduce `2026-08-18-1-newcolls`, all_gather/reduce_scatter `2026-08-19-2-agrs`.

**Cost unit = measurements. 1 measurement = median of 3 benchmark runs** (the project's repeat policy; raw run counts = x3, kept in searchcmp_summary.json).

**Scoring.** Grid = ground truth (its winners define 'exact'). Adaptive = racing search (anchors 1,8,24,48, margin 15, 3 repeats, tol 0.5). Optuna/random = runs until their winner set matches the grid's, median of 10 seeds ('found' = seeds that ever matched). Triage = 3 default runs + adaptive on dip-flagged sizes only (10% detector); 'missed' = biggest headroom it leaves at unflagged sizes.

| grid | grid meas | adaptive | exact | gap% | optuna med | found | random med | found | triage | flagged | missed% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_reduce 1n | 30 | 26 | 18/18 | 0.0 | 28 | 3/10 | 29 | 10/10 | 1 | none | 40.1 |
| all_reduce 2n | 54 | 40 | 18/18 | 0.0 | 45 | 7/10 | 50 | 10/10 | 1 | none | 10.2 |
| all_reduce 3n | 54 | 40 | 18/18 | 0.0 | 45 | 4/10 | 52 | 10/10 | 1 | none | 14.1 |
| broadcast 1n | 20 | 20 | 18/18 | 0.0 | 15 | 10/10 | 20 | 10/10 | 11 | 512K | 53.6 |
| broadcast 2n | 27 | 26 | 18/18 | 0.0 | 24 | 6/10 | 26 | 10/10 | 1 | none | 52.1 |
| broadcast 3n | 27 | 24 | 18/18 | 0.0 | 20 | 8/10 | 24 | 10/10 | 14 | 64K | 59.0 |
| reduce 1n | 20 | 18 | 17/18 | 1.449 | 18 | 7/10 | 20 | 10/10 | 11 | 512K | 54.5 |
| reduce 2n | 27 | 23 | 18/18 | 0.0 | 22 | 9/10 | 25 | 10/10 | 1 | none | 58.3 |
| reduce 3n | 27 | 25 | 18/18 | 0.0 | 21 | 10/10 | 22 | 10/10 | 1 | none | 45.4 |
| all_gather 1n | 20 | 10 | 6/6 | 0.0 | 8 | 10/10 | 6 | 10/10 | 1 | none | 0.1 |
| all_gather 2n | 27 | 17 | 7/7 | 0.0 | 17 | 9/10 | 22 | 10/10 | 1 | none | 16.2 |
| all_gather 3n | 27 | 17 | 7/7 | 0.0 | 18 | 10/10 | 21 | 10/10 | 1 | none | 29.3 |
| reduce_scatter 1n | 20 | 18 | 18/18 | 0.0 | 19 | 6/10 | 20 | 10/10 | 1 | none | 6.6 |
| reduce_scatter 2n | 27 | 24 | 18/18 | 0.0 | 25 | 6/10 | 26 | 10/10 | 1 | none | 24.9 |
| reduce_scatter 3n | 27 | 23 | 18/18 | 0.0 | 24 | 5/10 | 26 | 10/10 | 1 | none | 33.7 |
| **total** | **434** | **351** | | | | | | | **48** | | |

## Configs (grid-truth pipeline: median → optimize -t 0.5 → tuner conf)

### all_reduce.conf
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allreduce,4096,4096,tree,ll,24,1,8,-1,-1
allreduce,8192,8192,tree,ll,4,1,8,-1,-1
allreduce,16384,65536,tree,ll,40,1,8,-1,-1
allreduce,131072,131072,tree,ll,32,1,8,-1,-1
allreduce,262144,262144,ring,ll,32,1,8,-1,-1
allreduce,524288,524288,ring,ll,48,1,8,-1,-1
allreduce,1048576,1048576,ring,ll,56,1,8,-1,-1
allreduce,2097152,2097152,ring,simple,48,1,8,-1,-1
allreduce,4194304,536870912,ring,simple,56,1,8,-1,-1
allreduce,4096,4096,tree,ll,2,2,16,-1,-1
allreduce,8192,8192,tree,ll,4,2,16,-1,-1
allreduce,16384,16384,tree,ll,8,2,16,-1,-1
allreduce,32768,32768,tree,ll,16,2,16,-1,-1
allreduce,65536,65536,tree,ll,48,2,16,-1,-1
allreduce,131072,131072,tree,ll,40,2,16,-1,-1
allreduce,262144,524288,tree,ll128,32,2,16,-1,-1
allreduce,1048576,16777216,tree,ll128,48,2,16,-1,-1
allreduce,33554432,33554432,ring,ll128,40,2,16,-1,-1
allreduce,67108864,67108864,ring,ll128,48,2,16,-1,-1
allreduce,134217728,134217728,ring,simple,40,2,16,-1,-1
allreduce,268435456,268435456,ring,simple,48,2,16,-1,-1
allreduce,536870912,536870912,ring,simple,40,2,16,-1,-1
allreduce,4096,4096,tree,ll,2,3,24,-1,-1
allreduce,8192,8192,tree,ll,4,3,24,-1,-1
allreduce,16384,16384,tree,ll,8,3,24,-1,-1
allreduce,32768,32768,tree,ll,16,3,24,-1,-1
allreduce,65536,65536,tree,ll,32,3,24,-1,-1
allreduce,131072,131072,tree,ll128,16,3,24,-1,-1
allreduce,262144,262144,tree,ll128,48,3,24,-1,-1
allreduce,524288,524288,tree,ll128,32,3,24,-1,-1
allreduce,1048576,16777216,tree,ll128,48,3,24,-1,-1
allreduce,33554432,33554432,ring,ll128,32,3,24,-1,-1
allreduce,67108864,134217728,ring,ll128,40,3,24,-1,-1
allreduce,268435456,536870912,ring,simple,48,3,24,-1,-1
```

### broadcast.conf
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
broadcast,4096,4096,ring,ll,40,1,8,-1,-1
broadcast,8192,8192,ring,ll,56,1,8,-1,-1
broadcast,16384,16384,ring,ll,16,1,8,-1,-1
broadcast,32768,32768,ring,ll,2,1,8,-1,-1
broadcast,65536,65536,ring,ll,24,1,8,-1,-1
broadcast,131072,262144,ring,ll,40,1,8,-1,-1
broadcast,524288,524288,ring,ll,48,1,8,-1,-1
broadcast,1048576,8388608,ring,ll,56,1,8,-1,-1
broadcast,16777216,536870912,ring,simple,56,1,8,-1,-1
broadcast,4096,4096,ring,ll128,4,2,16,-1,-1
broadcast,8192,8192,ring,ll,1,2,16,-1,-1
broadcast,16384,16384,ring,simple,16,2,16,-1,-1
broadcast,32768,262144,ring,ll128,1,2,16,-1,-1
broadcast,524288,524288,ring,ll128,40,2,16,-1,-1
broadcast,1048576,1048576,ring,ll128,32,2,16,-1,-1
broadcast,2097152,2097152,ring,ll128,48,2,16,-1,-1
broadcast,4194304,4194304,ring,ll128,24,2,16,-1,-1
broadcast,8388608,8388608,ring,ll128,48,2,16,-1,-1
broadcast,16777216,67108864,ring,ll128,40,2,16,-1,-1
broadcast,134217728,134217728,ring,ll128,48,2,16,-1,-1
broadcast,268435456,268435456,ring,simple,40,2,16,-1,-1
broadcast,536870912,536870912,ring,simple,32,2,16,-1,-1
broadcast,4096,4096,ring,ll,24,3,24,-1,-1
broadcast,8192,8192,ring,ll128,16,3,24,-1,-1
broadcast,16384,16384,ring,ll128,24,3,24,-1,-1
broadcast,32768,524288,ring,ll128,1,3,24,-1,-1
broadcast,1048576,16777216,ring,ll128,32,3,24,-1,-1
broadcast,33554432,33554432,ring,ll128,40,3,24,-1,-1
broadcast,67108864,536870912,ring,ll128,48,3,24,-1,-1
```

### reduce.conf
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
reduce,4096,4096,ring,ll,16,1,8,-1,-1
reduce,8192,8192,ring,ll,4,1,8,-1,-1
reduce,16384,16384,ring,simple,24,1,8,-1,-1
reduce,32768,32768,ring,simple,32,1,8,-1,-1
reduce,65536,65536,ring,ll,16,1,8,-1,-1
reduce,131072,131072,ring,ll,56,1,8,-1,-1
reduce,262144,262144,ring,ll,40,1,8,-1,-1
reduce,524288,524288,ring,ll,48,1,8,-1,-1
reduce,1048576,4194304,ring,ll,56,1,8,-1,-1
reduce,8388608,536870912,ring,simple,56,1,8,-1,-1
reduce,4096,8192,ring,ll,1,2,16,-1,-1
reduce,16384,16384,ring,ll128,24,2,16,-1,-1
reduce,32768,262144,ring,ll128,1,2,16,-1,-1
reduce,524288,524288,ring,ll128,40,2,16,-1,-1
reduce,1048576,1048576,ring,ll128,32,2,16,-1,-1
reduce,2097152,2097152,ring,ll128,48,2,16,-1,-1
reduce,4194304,16777216,ring,ll128,40,2,16,-1,-1
reduce,33554432,134217728,ring,ll128,48,2,16,-1,-1
reduce,268435456,268435456,ring,simple,40,2,16,-1,-1
reduce,536870912,536870912,ring,simple,32,2,16,-1,-1
reduce,4096,8192,ring,ll,1,3,24,-1,-1
reduce,16384,16384,ring,ll128,24,3,24,-1,-1
reduce,32768,524288,ring,ll128,1,3,24,-1,-1
reduce,1048576,2097152,ring,ll128,40,3,24,-1,-1
reduce,4194304,8388608,ring,ll128,32,3,24,-1,-1
reduce,16777216,536870912,ring,ll128,48,3,24,-1,-1
```

### all_gather.conf
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allgather,16777216,536870912,ring,simple,56,1,8,-1,-1
allgather,8388608,16777216,ring,ll128,24,2,16,-1,-1
allgather,33554432,33554432,ring,ll128,40,2,16,-1,-1
allgather,67108864,67108864,ring,ll128,48,2,16,-1,-1
allgather,134217728,134217728,ring,simple,40,2,16,-1,-1
allgather,268435456,536870912,ring,simple,48,2,16,-1,-1
allgather,8388480,8388480,ring,ll128,16,3,24,-1,-1
allgather,16776960,16776960,ring,ll128,24,3,24,-1,-1
allgather,33554304,33554304,ring,ll128,32,3,24,-1,-1
allgather,67108608,134217600,ring,ll128,40,3,24,-1,-1
allgather,268435200,536870784,ring,simple,48,3,24,-1,-1
```

### reduce_scatter.conf
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
reducescatter,4096,4096,ring,ll,2,1,8,-1,-1
reducescatter,8192,8192,ring,ll,1,1,8,-1,-1
reducescatter,16384,32768,ring,ll,2,1,8,-1,-1
reducescatter,65536,65536,ring,ll,40,1,8,-1,-1
reducescatter,131072,131072,ring,ll,32,1,8,-1,-1
reducescatter,262144,1048576,ring,ll,40,1,8,-1,-1
reducescatter,2097152,4194304,ring,simple,32,1,8,-1,-1
reducescatter,8388608,8388608,ring,simple,48,1,8,-1,-1
reducescatter,16777216,536870912,ring,simple,56,1,8,-1,-1
reducescatter,4096,16384,ring,ll,1,2,16,-1,-1
reducescatter,32768,65536,ring,ll,2,2,16,-1,-1
reducescatter,131072,131072,ring,simple,24,2,16,-1,-1
reducescatter,262144,262144,ring,ll,16,2,16,-1,-1
reducescatter,524288,524288,ring,ll,32,2,16,-1,-1
reducescatter,1048576,1048576,ring,simple,32,2,16,-1,-1
reducescatter,2097152,2097152,ring,ll,16,2,16,-1,-1
reducescatter,4194304,4194304,ring,ll128,16,2,16,-1,-1
reducescatter,8388608,8388608,ring,ll128,24,2,16,-1,-1
reducescatter,16777216,16777216,ring,ll128,32,2,16,-1,-1
reducescatter,33554432,33554432,ring,ll128,40,2,16,-1,-1
reducescatter,67108864,67108864,ring,ll128,48,2,16,-1,-1
reducescatter,134217728,268435456,ring,simple,40,2,16,-1,-1
reducescatter,536870912,536870912,ring,simple,48,2,16,-1,-1
reducescatter,3840,32640,ring,ll,1,3,24,-1,-1
reducescatter,65280,65280,ring,ll,2,3,24,-1,-1
reducescatter,130944,130944,ring,ll,4,3,24,-1,-1
reducescatter,261888,261888,ring,ll,8,3,24,-1,-1
reducescatter,524160,1048320,ring,ll128,4,3,24,-1,-1
reducescatter,2097024,2097024,ring,ll128,8,3,24,-1,-1
reducescatter,4194048,8388480,ring,ll128,16,3,24,-1,-1
reducescatter,16776960,16776960,ring,ll128,24,3,24,-1,-1
reducescatter,33554304,33554304,ring,ll128,32,3,24,-1,-1
reducescatter,67108608,134217600,ring,ll128,40,3,24,-1,-1
reducescatter,268435200,268435200,ring,simple,40,3,24,-1,-1
reducescatter,536870784,536870784,ring,simple,48,3,24,-1,-1
```

### alltoall — no config
RCCL forces RING/SIMPLE for alltoall at source (`enqueue.cc:2095`); algo/proto are not tunable and no grid was measured.

## Caveats
- broadcast/reduce 1n triage uses the only existing 1n default curve (2026-08-17-4-hotspot-mn, node 8) while their grids ran on node 6 - cross-node.
- Each grid's node set differs (0804 vs {1,3}/{1,3,6} vs {5,7}/{5,6,7}); comparisons are within-grid only, never busbw across grids.
- all_gather rows below 16M (1n) / 8M (2-3n) are Direct-substituted and excluded; the conf only covers the RING-honoured sizes, defaults rule below.
- These configs are grid winners; only broadcast/reduce 2n/3n winners have been A/B-validated live (2026-08-19). The rest are unvalidated.
- Optuna numbers use the findings/06 protocol; a real deployment cannot know when to stop — the medians shown are its *oracle-best* stopping point.
- Measurement counts are runs/3, exact under the current repeat policy; if the 1+2 policy is ever used live, that table must show runs.
