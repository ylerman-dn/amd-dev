#!/bin/bash
# Controls for the 1n small-size v4>v5 gap:
#   revorder/  — v5 runs FIRST in each pair (if gap flips to favor v5, it's an ordering artifact)
#   v4v4/      — v4 in BOTH slots (any slot1-vs-slot2 gap with identical plugins = pure artifact)
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
MY_PATH=/opt/shared/ylerman/GPU-107/bin
eval $(python3 -c "
import yaml
env=yaml.safe_load(open('$D/sweep_config.yaml'))['env_vars']
drop={'NCCL_DEBUG','NCCL_DEBUG_FILE','NCCL_DEBUG_SUBSYS','PATH','NCCL_TOPO_DUMP_FILE','NCCL_GRAPH_DUMP_FILE'}
[print(f'export {k}=\"{v}\"') for k,v in env.items() if k not in drop]
")
export LD_LIBRARY_PATH=$MY_PATH:$LD_LIBRARY_PATH
export LD_PRELOAD=$MY_PATH/librccl.so
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV
export NCCL_TUNER_CONFIG_FILE=$D/gen_3n_t1.validated.csv
one() { # $1 outdir  $2 tag  $3 plugin-so
  export NCCL_TUNER_PLUGIN=$D/dn-tuner/$3
  export NCCL_DEBUG_FILE=$1/$2_dbg_%p.log
  srun --jobid=14470 --nodelist=amd-mi355x-5 -N1 --ntasks-per-node=8 --gres=gpu:8 \
       --mpi=pmix --export=ALL \
       $MY_PATH/all_reduce_perf -b 4096 -e 536870912 -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 \
       < /dev/null > $1/$2.log 2>&1
  echo "$2 rc=$? $(date -u +%FT%TZ)" >> $1/PROGRESS
}
mkdir -p $D/revorder $D/v4v4
for rep in 1 2 3 4 5 6 7; do
  one $D/revorder v5_r${rep} librccl-tunerv5-dn.so
  one $D/revorder v4_r${rep} librccl-tunerv4-dn.so
done
echo "REVORDER DONE $(date -u +%FT%TZ)" >> $D/revorder/PROGRESS
for rep in 1 2 3 4 5 6 7; do
  one $D/v4v4 slot1_r${rep} librccl-tunerv4-dn.so
  one $D/v4v4 slot2_r${rep} librccl-tunerv4-dn.so
done
echo "V4V4 DONE $(date -u +%FT%TZ)" >> $D/v4v4/PROGRESS
