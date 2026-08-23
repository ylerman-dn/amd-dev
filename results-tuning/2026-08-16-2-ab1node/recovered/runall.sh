#!/bin/bash
D=/opt/shared/ylerman/GPU-107/ab-2026-08-16
IB="ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_6:1,ionic_7:1"
run() {
  n=$1; nodelist=$2; shift 2
  mkdir -p $D/v2_${n}n/logs
  python3 $D/rccl-sweep/validate_tuner_config.py \
    --config $D/gen_${n}n_t1.conf \
    --binary /opt/shared/ylerman/GPU-107/bin/all_reduce_perf \
    --plugin /opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so \
    --jobid 14457 --nodelist $nodelist --ranks-per-node 8 --gpus-per-rank 1 \
    --repeats 7 --warmup-runs 4 --min-bytes 4096 --max-bytes 536870912 \
    --iters 20 --warmup 5 --plugin-must-fire --split-ranges \
    --logdir $D/v2_${n}n/logs --times-csv $D/v2_${n}n/times.csv "$@" \
    > $D/v2_${n}n/validate.log 2>&1
  echo "${n}n rc=$? $(date -u +%H:%M:%S)" >> $D/RUNALL_PROGRESS
}
: > $D/RUNALL_PROGRESS
run 1 amd-mi355x-7
run 2 amd-mi355x-5,amd-mi355x-7 --env NCCL_SOCKET_IFNAME=enp81s0f1np1 --env NCCL_IB_HCA=$IB --env NCCL_IB_GID_INDEX=3
run 3 amd-mi355x-3,amd-mi355x-5,amd-mi355x-7 --env NCCL_SOCKET_IFNAME=enp81s0f1np1 --env NCCL_IB_HCA=$IB --env NCCL_IB_GID_INDEX=3
echo "ALL DONE $(date -u +%FT%TZ)" >> $D/RUNALL_PROGRESS
