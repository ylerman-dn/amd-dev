#!/bin/bash
# Did the SWEEP actually run LL128 when it forced NCCL_PROTO=LL128?
# The sweep stamped rows with the REQUESTED algo/proto (binary doesn't report actual).
# RCCL's own "AllReduce: N Bytes -> Algo X proto Y" line is authoritative.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/ab-v2/verify; mkdir -p $OUT
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 8388608 -e 33554432 -f 2 -n 2 -w 1 -c 0 -M 1 -R 1"

for N in 1 2; do
  O=$(salloc --no-shell -p XAI -N $N --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 25 -J ylerman 2>&1)
  JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n"; continue; }
  echo "=== ${N}n JID=$JID $(squeue -j $JID -h -o %N) ==="
  srun --jobid=$JID --overlap --gres=none -N$N --ntasks-per-node=1 sleep 1800 >/dev/null 2>&1 & KA=$!

  # (a) 8 separate procs, -g 1  == how the SWEEP ran it
  timeout 200 srun --jobid=$JID -N$N --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "export $BASE $NET NCCL_ALGO=TREE NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64; $BIN/all_reduce_perf $SZ -g 1" > $OUT/${N}n_sweepstyle_g1_TREE_LL128.log 2>&1
  # (b) 1 proc, -g 8  == how the A/B ran it
  timeout 200 srun --jobid=$JID -N$N --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "export $BASE $NET NCCL_ALGO=TREE NCCL_PROTO=LL128 NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64; $BIN/all_reduce_perf $SZ -g 8" > $OUT/${N}n_abstyle_g8_TREE_LL128.log 2>&1

  for f in $OUT/${N}n_*_TREE_LL128.log; do
    echo "--- $(basename $f): rows=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' $f)"
    grep -hoE "AllReduce: [0-9]+ Bytes -> Algo [A-Z]+ proto [A-Za-z0-9]+" $f | sort -u | head -4 | sed 's/^/      /'
  done
  kill $KA 2>/dev/null; scancel $JID; echo "=== ${N}n released ==="
done
echo VERIFY_DONE
