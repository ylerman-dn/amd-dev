#!/bin/bash
# GPU-107 new-collectives grids: broadcast+reduce, 1/2/3 nodes, 3 repeats.
# Nodes pinned: 1n=node1, 2n={1,3}, 3n={1,3,4}. RING only (TREE unsupported for both).
set -u
SW=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna
OUT=/opt/shared/ylerman/GPU-107/optuna-live-2026-08-18/newcolls
export MY_PATH=/opt/shared/ylerman/GPU-107/bin
cd $SW
printf "172.30.160.204 # amd-mi355x-1\n172.30.160.119 # amd-mi355x-3\n" > servers_2n_13.txt
printf "172.30.160.204 # amd-mi355x-1\n172.30.160.119 # amd-mi355x-3\n172.30.160.111 # amd-mi355x-4\n" > servers_3n_134.txt
mkdir -p $OUT
run() { local rep=$1 coll=$2 n=$3 srv=$4 tag=$5; shift 5
  python3 rccl_sweep.py --servers $srv --output-dir $OUT/${coll}_${n}n_${tag}_rep$rep \
    --nodes $n --collective $coll --min-size 4K --max-size 512M "$@" \
    > $OUT/${coll}_${n}n_${tag}_rep$rep.log 2>&1
  echo "$(date -u +%H:%M:%SZ) rep$rep $coll ${n}n $tag rc=$? runs=$(grep -cE '^  Avg BW' $OUT/${coll}_${n}n_${tag}_rep$rep.log)" >> $OUT/PROGRESS
}
: > $OUT/PROGRESS
for rep in 1 2 3; do
  for coll in broadcast reduce; do
    run $rep $coll 2 servers_2n_13.txt grid --channels 1,2,4,8,16,24,32,40,48 --algo RING --proto SIMPLE,LL,LL128
    run $rep $coll 2 servers_2n_13.txt def
    run $rep $coll 3 servers_3n_134.txt grid --channels 1,2,4,8,16,24,32,40,48 --algo RING --proto SIMPLE,LL,LL128
    run $rep $coll 3 servers_3n_134.txt def
  done
done
echo "ALL DONE $(date -u +%FT%TZ)" >> $OUT/PROGRESS
