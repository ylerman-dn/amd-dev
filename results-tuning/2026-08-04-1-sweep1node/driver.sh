#!/bin/bash
set -u
NEW=/opt/shared/ylerman/GPU-107/sweep-clean-2026-08-04
cd $NEW/tool
export MY_PATH=/opt/shared/ylerman/GPU-107/bin
CH=1,2,4,8,16,24,32,40,48,56
run() { local r=$1 tag=$2; shift 2
  python3 rccl_sweep.py --servers servers.txt --output-dir "$NEW/rep$r" \
    --nodes 1 --collective all_reduce --min-size 4K --max-size 512M "$@" \
    > "$NEW/rep${r}_${tag}.log" 2>&1
  echo "rep$r/$tag rc=$? runs=$(grep -cE '^  Avg BW' "$NEW/rep${r}_${tag}.log") subst=$(grep -c 'substituted:' "$NEW/rep${r}_${tag}.log")" >> "$NEW/PROGRESS"
}
: > $NEW/PROGRESS
for r in 1 2 3; do
  run $r ring --channels $CH --algo RING --proto SIMPLE,LL
  run $r tree --channels $CH --algo TREE --proto LL
  run $r def
done
echo "ALL DONE $(date -u +%FT%TZ)" >> $NEW/PROGRESS
