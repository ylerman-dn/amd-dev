#!/bin/bash
set -u
B=/opt/shared/ylerman/GPU-107; AB=$B/ab3node-2026-08-04
cd $AB/tool
python3 validate_tuner_config.py \
  --config $AB/candidate.conf \
  --binary $B/bin/all_reduce_perf \
  --plugin $B/ab-tuner-test/librccl-tunerv4-dn.so \
  --jobid 14138 --nodelist amd-mi355x-5,amd-mi355x-6,amd-mi355x-7 \
  --ranks-per-node 8 --gpus-per-rank 1 \
  --repeats 7 --min-bytes 4096 --max-bytes 536870912 \
  --iters 20 --warmup 5 --plugin-must-fire \
  --logdir $AB/logs --times-csv $AB/times.csv \
  --env NCCL_SOCKET_IFNAME=enp81s0f1np1 \
  --env NCCL_IB_HCA=ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1 \
  --env NCCL_IB_GID_INDEX=1 --env NCCL_IB_TIMEOUT=5 --env NCCL_IB_QPS_PER_CONNECTION=2 \
  --env NCCL_IB_TC=104 --env NCCL_IB_FIFO_TC=192 --env NCCL_IB_USE_INLINE=1 \
  --env NCCL_GDR_FLUSH_DISABLE=1 --env NCCL_GDRCOPY_ENABLE=0 --env NCCL_IGNORE_CPU_AFFINITY=1 \
  --env HSA_NO_SCRATCH_RECLAIM=1 \
  --env OMPI_MCA_btl=self,vader,tcp --env OMPI_MCA_pml=ob1 \
  --env OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 \
  > $AB/validate_report.txt 2>&1
echo "AB DONE rc=$? $(date -u +%FT%TZ)" >> $AB/PROGRESS
