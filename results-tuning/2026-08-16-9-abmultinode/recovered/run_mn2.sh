#!/bin/bash
D=/opt/shared/ylerman/GPU-107/ab-2026-08-16
MY_PATH=/opt/shared/ylerman/GPU-107/bin
run() {
  n=$1; nodelist=$2
  rm -rf $D/v5_${n}n; mkdir -p $D/v5_${n}n/logs
  python3 $D/rccl-sweep/validate_tuner_config.py \
    --config $D/gen_${n}n_t1.conf \
    --binary $MY_PATH/all_reduce_perf \
    --plugin /opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so \
    --jobid 14466 --nodelist $nodelist --ranks-per-node 8 --gpus-per-rank 1 \
    --repeats 7 --warmup-runs 4 --preflight 5 --min-bytes 4096 --max-bytes 536870912 \
    --iters 20 --warmup 5 --plugin-must-fire --split-ranges \
    --logdir $D/v5_${n}n/logs --times-csv $D/v5_${n}n/times.csv \
    > $D/v5_${n}n/validate.log 2>&1
  echo "${n}n rc=$? $(date -u +%H:%M:%S)" >> $D/MN2_PROGRESS
}
: > $D/MN2_PROGRESS
run 2 amd-mi355x-5,amd-mi355x-7
run 3 amd-mi355x-5,amd-mi355x-6,amd-mi355x-7
echo "MN2 DONE $(date -u +%FT%TZ)" >> $D/MN2_PROGRESS
