#!/bin/bash
D=/opt/shared/ylerman/GPU-107/bcast-2026-08-16
T=/opt/shared/ylerman/GPU-107/ab-2026-08-16/rccl-sweep
MY=/opt/shared/ylerman/GPU-107/bin
: > $D/PROGRESS
for c in broadcast reduce; do
  bin=$MY/${c}_perf
  rm -rf $D/${c}; mkdir -p $D/${c}/logs
  python3 $T/validate_tuner_config.py \
    --config $D/${c}.conf --binary $bin \
    --plugin /opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so \
    --jobid 14469 --nodelist amd-mi355x-8 --ranks-per-node 8 --gpus-per-rank 1 \
    --repeats 7 --warmup-runs 4 --preflight 4 \
    --min-bytes 4096 --max-bytes 536870912 --iters 20 --warmup 5 \
    --plugin-must-fire --split-ranges \
    --logdir $D/${c}/logs --times-csv $D/${c}/times.csv > $D/${c}/validate.log 2>&1
  echo "$c rc=$? $(date -u +%H:%M:%S)" >> $D/PROGRESS
done
echo "BCAST DONE" >> $D/PROGRESS
