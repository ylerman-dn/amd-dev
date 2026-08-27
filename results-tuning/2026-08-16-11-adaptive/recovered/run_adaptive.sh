#!/bin/bash
D=/opt/shared/ylerman/GPU-107/adaptive-2026-08-16
T=/opt/shared/ylerman/GPU-107/ab-2026-08-16/rccl-sweep
export MY_PATH=/opt/shared/ylerman/GPU-107/bin
cd $T
: > $D/PROGRESS
for n in 1 2 3; do
  start=$(date +%s)
  timeout 5400 python3 rccl_autotune.py \
    -n $n -c 1,2,4,8,16,24,32,40,48,56 \
    --collective all_reduce --algo RING,TREE --proto SIMPLE,LL,LL128 \
    --min-size 4K --max-size 512M \
    --servers $D/servers${n}.txt \
    -o $D/out${n}n \
    --tuner-output $D/adaptive_${n}n.conf \
    --report-output $D/report_${n}n.csv > $D/log_${n}n.txt 2>&1
  rc=$?; end=$(date +%s)
  runs=$(find $D/out${n}n -name command.txt 2>/dev/null | wc -l)
  echo "${n}n rc=$rc wall=$((end-start))s runs=$runs" >> $D/PROGRESS
done
echo "ADAPTIVE DONE" >> $D/PROGRESS
