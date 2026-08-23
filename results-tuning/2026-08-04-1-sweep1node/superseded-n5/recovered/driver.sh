#!/bin/bash
# GPU-107 fresh sweep driver. all_reduce, 1 node, honoured combos only, 3 repeats.
set -u
cd /opt/shared/ylerman/GPU-107/rccl-sweep
export MY_PATH=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/sweep-2026-08-04
CH=1,2,4,8,16,24,32,40,48,56
run() { # $1=rep $2=tag $3..=args
  local r=$1 tag=$2; shift 2
  python3 rccl_sweep.py --servers servers.txt --output-dir "$OUT/rep$r" \
    --nodes 1 --collective all_reduce --min-size 4K --max-size 512M "$@" \
    > "$OUT/rep${r}_${tag}.log" 2>&1
  echo "rep$r/$tag rc=$? runs=$(grep -cE '^  Avg BW' "$OUT/rep${r}_${tag}.log") subst=$(grep -c 'substituted:' "$OUT/rep${r}_${tag}.log")" >> "$OUT/PROGRESS"
}
: > "$OUT/PROGRESS"
echo "rep1/ring already done: 20 runs, 0 subst" >> "$OUT/PROGRESS"
run 1 tree --channels $CH --algo TREE --proto LL
run 1 def
for r in 2 3; do
  run $r ring --channels $CH --algo RING --proto SIMPLE,LL
  run $r tree --channels $CH --algo TREE --proto LL
  run $r def
done
echo "ALL DONE $(date -u +%FT%TZ)" >> "$OUT/PROGRESS"
