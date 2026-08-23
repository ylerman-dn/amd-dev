#!/bin/bash
D=/opt/shared/ylerman/GPU-107/ab-2026-08-16
MY_PATH=/opt/shared/ylerman/GPU-107/bin
# exact env from results-tuning/2026-08-03-9-multinode/run_mn.sh (the known-good multi-node run)
export LD_LIBRARY_PATH=/usr/local/lib:$MY_PATH:/opt/rocm/bin
export NCCL_SOCKET_IFNAME=enp81s0f1np1
export NCCL_IB_HCA="ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1"
export NCCL_IB_GID_INDEX=1 NCCL_IB_TIMEOUT=5 NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_TC=104 NCCL_IB_FIFO_TC=192 NCCL_IB_USE_INLINE=1
export NCCL_GDR_FLUSH_DISABLE=1 NCCL_GDRCOPY_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1
export OMPI_MCA_btl="self,vader,tcp" OMPI_MCA_pml=ob1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1
run() {
  n=$1; nodelist=$2
  rm -rf $D/v3_${n}n; mkdir -p $D/v3_${n}n/logs
  python3 $D/rccl-sweep/validate_tuner_config.py \
    --config $D/gen_${n}n_t1.conf \
    --binary $MY_PATH/all_reduce_perf \
    --plugin /opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so \
    --jobid 14457 --nodelist $nodelist --ranks-per-node 8 --gpus-per-rank 1 \
    --repeats 7 --warmup-runs 4 --min-bytes 4096 --max-bytes 536870912 \
    --iters 20 --warmup 5 --plugin-must-fire --split-ranges \
    --logdir $D/v3_${n}n/logs --times-csv $D/v3_${n}n/times.csv \
    > $D/v3_${n}n/validate.log 2>&1
  echo "${n}n rc=$? $(date -u +%H:%M:%S)" >> $D/MN_PROGRESS
}
: > $D/MN_PROGRESS
run 2 amd-mi355x-5,amd-mi355x-7
run 3 amd-mi355x-3,amd-mi355x-5,amd-mi355x-7
echo "MN DONE $(date -u +%FT%TZ)" >> $D/MN_PROGRESS
