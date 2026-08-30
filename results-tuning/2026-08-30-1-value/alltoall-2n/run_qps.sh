#!/bin/bash
# GPU-107: does NCCL_IB_QPS_PER_CONNECTION cause the alltoall 2-node crash?
# 3 repeats at QPS=2 (our standard, sweep_config.yaml:92) and 3 at QPS=1, alternating.
# Usage: run_qps.sh <jobid>
set -u
JID=$1
MY_PATH=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/a2a-2026-08-30/qps
NODES=amd-mi355x-3,amd-mi355x-5
mkdir -p "$OUT"
ARGS="-b 4K -e 64K -f 2 -g 1 -n 20 -w 5 -c 1 -A 1"

run() {  # $1=tag  $2=qps
  local tag=$1 qps=$2 t0=$(date +%s)
  timeout 600 srun --jobid=$JID -N2 --nodelist=$NODES --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "
export LD_LIBRARY_PATH=/usr/local/lib:$MY_PATH:/opt/rocm/bin
export LD_PRELOAD=$MY_PATH/librccl.so
export NCCL_SOCKET_IFNAME=enp81s0f1np1
export NCCL_IB_HCA='ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1'
export NCCL_IB_GID_INDEX=1 NCCL_IB_TIMEOUT=5 NCCL_IB_QPS_PER_CONNECTION=$qps NCCL_IB_TC=104 NCCL_IB_FIFO_TC=192 NCCL_IB_USE_INLINE=1
export NCCL_GDR_FLUSH_DISABLE=1 NCCL_GDRCOPY_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export OMPI_MCA_btl='self,vader,tcp' OMPI_MCA_pml=ob1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH,ENV NCCL_DEBUG_FILE=$OUT/${tag}_dbg_%p.log
exec $MY_PATH/alltoall_perf $ARGS -Z csv -X $OUT/$tag.csv
" > "$OUT/$tag.log" 2>&1 < /dev/null
  local rc=$? rows errs
  rows=$(grep -c "^0," "$OUT/$tag.csv" 2>/dev/null || echo 0)
  errs=$(grep -c "ionic_comp" "$OUT/$tag.log" 2>/dev/null || echo 0)
  printf "%-16s qps=%s rc=%-4s %3ss csvrows=%-3s ionic_errs=%s\n" \
    "$tag" "$qps" "$rc" "$(( $(date +%s) - t0 ))" "$rows" "$errs"
}

for r in 1 2 3; do
  run "qps2_rep${r}" 2
  run "qps1_rep${r}" 1
done
echo "QPS PROBE DONE"
