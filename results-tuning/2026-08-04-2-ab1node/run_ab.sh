#!/bin/bash
set -u
B=/opt/shared/ylerman/GPU-107; AB=$B/ab1node-2026-08-04
cd $AB/tool
python3 validate_tuner_config.py \
  --config $AB/candidate.conf \
  --binary $B/bin/all_reduce_perf \
  --plugin $B/ab-tuner-test/librccl-tunerv4-dn.so \
  --jobid 14138 --nodelist amd-mi355x-7 \
  --ranks-per-node 8 --gpus-per-rank 1 \
  --repeats 7 --min-bytes 4096 --max-bytes 536870912 \
  --iters 20 --warmup 5 --plugin-must-fire \
  --logdir $AB/logs --times-csv $AB/times.csv \
  > $AB/validate_report.txt 2>&1
echo "AB DONE rc=$? $(date -u +%FT%TZ)" >> $AB/PROGRESS
