#!/bin/bash
# GPU-107 multi-node acceptance/defaults runner.
# Usage: run_mn.sh <jobid> <2|3> <group>
#   group: fwd1 (all_reduce,all_gather,reduce_scatter) | fwd2 (alltoall,broadcast,reduce)
#          | refs (3 unforced repeats x 6 colls) | probes (4 exotic algos on all_reduce)
# All env is exported INSIDE bash -c: srun --export uses commas as its own separator,
# so any comma-containing value (NCCL_IB_HCA, OMPI_MCA_btl) is mangled if passed via --export.
set -u
JID=$1; NN=$2; GROUP=$3
MY_PATH=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/mn-2026-08-03/${NN}n
mkdir -p "$OUT"
case $NN in
  2) NODES=amd-mi355x-3,amd-mi355x-5 ;;
  3) NODES=amd-mi355x-3,amd-mi355x-5,amd-mi355x-6 ;;
esac
ARGS="-b 4K -e 512M -f 2 -g 1 -n 20 -w 5 -c 1 -A 1"

run() {  # $1=tag  $2=extra exports (space separated, may be empty)
  local tag=$1 extra=$2 t0=$(date +%s)
  timeout 400 srun --jobid=$JID -N$NN --nodelist=$NODES --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
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
exec $MY_PATH/\$COLL $ARGS -Z csv -X $OUT/$tag.csv
" > "$OUT/$tag.log" 2>&1 < /dev/null
  local rc=$? rows
  rows=$(grep -c "^0," "$OUT/$tag.csv" 2>/dev/null || echo 0)
  printf "%-34s rc=%-3s %3ss csvrows=%s\n" "$tag" "$rc" "$(( $(date +%s) - t0 ))" "$rows"
}

case $GROUP in
  fwd1) COLLS="all_reduce all_gather reduce_scatter" ;;
  fwd2) COLLS="alltoall broadcast reduce" ;;
  refs) COLLS="all_reduce all_gather reduce_scatter alltoall broadcast reduce" ;;
  probes) COLLS="all_reduce" ;;
esac

for c in $COLLS; do
  export COLL=${c}_perf
  case $GROUP in
    fwd1|fwd2)
      for a in RING TREE; do for p in SIMPLE LL128 LL; do
        run "${c}__${a}_${p}" "export COLL=${c}_perf NCCL_ALGO=$a NCCL_PROTO=$p" ; done; done ;;
    refs)
      for r in 1 2 3; do run "${c}__REF_rep${r}" "export COLL=${c}_perf" ; done ;;
    probes)
      for a in NVLS PAT COLLNET_DIRECT COLLNET_CHAIN; do
        run "${c}__probe_${a}" "export COLL=${c}_perf NCCL_ALGO=$a NCCL_PROTO=SIMPLE" ; done ;;
  esac
done
echo "GROUP $GROUP @ ${NN}n DONE"
