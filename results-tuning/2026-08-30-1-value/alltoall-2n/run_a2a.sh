#!/bin/bash
# GPU-107 alltoall 2-node re-test: correct p2p channel knob + pivot path + tuner-consult log proof.
# Usage: run_a2a.sh <jobid>
# All env exported INSIDE bash -c: srun --export uses commas as its own separator.
set -u
JID=$1
MY_PATH=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/a2a-2026-08-30
CONF=$OUT/probe.conf
PLUGIN=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
NODES=amd-mi355x-3,amd-mi355x-5
mkdir -p "$OUT"
ARGS="-b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1"

run() {  # $1=tag  $2=extra exports
  local tag=$1 extra=$2 t0=$(date +%s)
  timeout 600 srun --jobid=$JID -N2 --nodelist=$NODES --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "
export LD_LIBRARY_PATH=/usr/local/lib:$MY_PATH:/opt/rocm/bin
export LD_PRELOAD=$MY_PATH/librccl.so
export NCCL_SOCKET_IFNAME=enp81s0f1np1
export NCCL_IB_HCA='ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1'
export NCCL_IB_GID_INDEX=1 NCCL_IB_TIMEOUT=5 NCCL_IB_QPS_PER_CONNECTION=2 NCCL_IB_TC=104 NCCL_IB_FIFO_TC=192 NCCL_IB_USE_INLINE=1
export NCCL_GDR_FLUSH_DISABLE=1 NCCL_GDRCOPY_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export OMPI_MCA_btl='self,vader,tcp' OMPI_MCA_pml=ob1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH,ENV NCCL_DEBUG_FILE=$OUT/${tag}_dbg_%p.log
$extra
exec $MY_PATH/alltoall_perf $ARGS -Z csv -X $OUT/$tag.csv
" > "$OUT/$tag.log" 2>&1 < /dev/null
  local rc=$? rows
  rows=$(grep -c "^0," "$OUT/$tag.csv" 2>/dev/null || echo 0)
  printf "%-28s rc=%-3s %3ss csvrows=%s\n" "$tag" "$rc" "$(( $(date +%s) - t0 ))" "$rows"
}

for r in 1 2 3 4 5; do
  run "def_rep${r}"        ""
  for n in 8 16 32 64; do
    run "p2p${n}_rep${r}"  "export NCCL_MIN_P2P_NCHANNELS=$n NCCL_MAX_P2P_NCHANNELS=$n"
  done
  run "pivot_rep${r}"      "export RCCL_ALL_TO_ALL_PIVOT_ENABLE=1"
  run "plugdef_rep${r}"    "export NCCL_TUNER_PLUGIN=$PLUGIN NCCL_TUNER_CONFIG_FILE=$CONF"
  run "plugpivot_rep${r}"  "export RCCL_ALL_TO_ALL_PIVOT_ENABLE=1 NCCL_TUNER_PLUGIN=$PLUGIN NCCL_TUNER_CONFIG_FILE=$CONF"
done
echo "A2A 2n DONE"
