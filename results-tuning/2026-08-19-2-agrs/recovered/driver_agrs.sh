#!/bin/bash
# all_gather + reduce_scatter grids, 1/2/3 nodes, 3 repeats + defaults.
# Nodes: 1n=node7, 2n={5,7}, 3n={5,6,7} (published trio).
set -u
SW=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna
OUT=/opt/shared/ylerman/GPU-107/optuna-live-2026-08-19/agrs
export MY_PATH=/opt/shared/ylerman/GPU-107/bin
cd $SW
printf "172.30.160.165 # amd-mi355x-7\n" > servers_1n7.txt
printf "172.30.160.131 # amd-mi355x-5\n172.30.160.165 # amd-mi355x-7\n" > servers_2n57.txt
printf "172.30.160.131 # amd-mi355x-5\n172.30.160.201 # amd-mi355x-6\n172.30.160.165 # amd-mi355x-7\n" > servers_3n567.txt
mkdir -p $OUT
run() { local rep=$1 coll=$2 n=$3 srv=$4 tag=$5; shift 5
  python3 rccl_sweep.py --servers $srv --output-dir $OUT/${coll}_${n}n_${tag}_rep$rep \
    --nodes $n --collective $coll --min-size 4K --max-size 512M "$@" \
    > $OUT/${coll}_${n}n_${tag}_rep$rep.log 2>&1
  echo "$(date -u +%H:%M:%SZ) rep$rep $coll ${n}n $tag rc=$? runs=$(grep -cE '^  Avg BW' $OUT/${coll}_${n}n_${tag}_rep$rep.log)" >> $OUT/PROGRESS
}
: > $OUT/PROGRESS
for rep in 1 2 3; do
  for coll in all_gather reduce_scatter; do
    run $rep $coll 1 servers_1n7.txt grid --channels 1,2,4,8,16,24,32,40,48,56 --algo RING --proto SIMPLE,LL
    run $rep $coll 1 servers_1n7.txt def
    run $rep $coll 2 servers_2n57.txt grid --channels 1,2,4,8,16,24,32,40,48 --algo RING --proto SIMPLE,LL,LL128
    run $rep $coll 2 servers_2n57.txt def
    run $rep $coll 3 servers_3n567.txt grid --channels 1,2,4,8,16,24,32,40,48 --algo RING --proto SIMPLE,LL,LL128
    run $rep $coll 3 servers_3n567.txt def
  done
done
echo "ALL DONE $(date -u +%FT%TZ)" >> $OUT/PROGRESS
