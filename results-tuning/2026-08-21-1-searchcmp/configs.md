# 15 tuner configs — grid-truth winners

## all_reduce
### all_reduce 1n — **NOT validated live**
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
```

### all_reduce 2n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
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
```

### all_reduce 3n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
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

## broadcast
### broadcast 1n — **NOT validated live**
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
```

### broadcast 2n — **A/B attempted 15x on {5,7}, unmeasurable at small sizes - not validated**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
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
```

### broadcast 3n — **VALIDATED live (2026-08-19)**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
broadcast,4096,4096,ring,ll,24,3,24,-1,-1
broadcast,8192,8192,ring,ll128,16,3,24,-1,-1
broadcast,16384,16384,ring,ll128,24,3,24,-1,-1
broadcast,32768,524288,ring,ll128,1,3,24,-1,-1
broadcast,1048576,16777216,ring,ll128,32,3,24,-1,-1
broadcast,33554432,33554432,ring,ll128,40,3,24,-1,-1
broadcast,67108864,536870912,ring,ll128,48,3,24,-1,-1
```

## reduce
### reduce 1n — **NOT validated live**
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
```

### reduce 2n — **VALIDATED live (2026-08-19)**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
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
```

### reduce 3n — **VALIDATED live (2026-08-19)**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
reduce,4096,8192,ring,ll,1,3,24,-1,-1
reduce,16384,16384,ring,ll128,24,3,24,-1,-1
reduce,32768,524288,ring,ll128,1,3,24,-1,-1
reduce,1048576,2097152,ring,ll128,40,3,24,-1,-1
reduce,4194304,8388608,ring,ll128,32,3,24,-1,-1
reduce,16777216,536870912,ring,ll128,48,3,24,-1,-1
```

## all_gather
### all_gather 1n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allgather,16777216,536870912,ring,simple,56,1,8,-1,-1
```

### all_gather 2n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allgather,8388608,16777216,ring,ll128,24,2,16,-1,-1
allgather,33554432,33554432,ring,ll128,40,2,16,-1,-1
allgather,67108864,67108864,ring,ll128,48,2,16,-1,-1
allgather,134217728,134217728,ring,simple,40,2,16,-1,-1
allgather,268435456,536870912,ring,simple,48,2,16,-1,-1
```

### all_gather 3n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allgather,8388480,8388480,ring,ll128,16,3,24,-1,-1
allgather,16776960,16776960,ring,ll128,24,3,24,-1,-1
allgather,33554304,33554304,ring,ll128,32,3,24,-1,-1
allgather,67108608,134217600,ring,ll128,40,3,24,-1,-1
allgather,268435200,536870784,ring,simple,48,3,24,-1,-1
```

## reduce_scatter
### reduce_scatter 1n — **NOT validated live**
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
```

### reduce_scatter 2n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
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
```

### reduce_scatter 3n — **NOT validated live**
```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
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

---
all_gather: rules only where RING honoured (>=16M 1n, >=8M 2-3n). alltoall: no config, forced RING/SIMPLE at source.
