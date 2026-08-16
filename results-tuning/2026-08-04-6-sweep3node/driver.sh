#!/bin/bash
set -u
N=/opt/shared/ylerman/GPU-107/sweep3n-2026-08-04
cd $N/tool
export MY_PATH=/opt/shared/ylerman/GPU-107/bin
CH=1,2,4,8,16,24,32,40,48
run() { local r=$1 tag=$2; shift 2
  python3 rccl_sweep.py --servers servers.txt --output-dir "$N/rep$r" \
    --nodes 3 --collective all_reduce --min-size 4K --max-size 512M "$@" \
    > "$N/rep${r}_${tag}.log" 2>&1
  echo "rep$r/$tag rc=$? runs=$(grep -cE '^  Avg BW' "$N/rep${r}_${tag}.log") subst=$(grep -c 'substituted:' "$N/rep${r}_${tag}.log")" >> "$N/PROGRESS"
}
: > $N/PROGRESS
for r in 1 2 3; do
  run $r ring --channels $CH --algo RING --proto SIMPLE,LL,LL128
  run $r tree --channels $CH --algo TREE --proto SIMPLE,LL,LL128
  run $r def
done
echo "ALL DONE $(date -u +%FT%TZ)" >> $N/PROGRESS
