#!/bin/bash
# GPU-107 v4-vs-v5 parity at 1 node with the 3-node rule file: NO rule matches, so this
# covers the plugin-loaded-but-never-fires path. 7 interleaved repeats per arm, same
# benchmark flags as the validator. Analysis is done offline from the logs.
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
MY_PATH=/opt/shared/ylerman/GPU-107/bin
mkdir -p $D/parity_1n
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
# 4 discarded warm-up runs (clocks), then 7 interleaved repeats per arm.
for w in 1 2 3 4; do
  unset NCCL_TUNER_PLUGIN
  export NCCL_DEBUG_FILE=$D/parity_1n/warmup_r${w}_dbg_%p.log
  srun --jobid=14470 --nodelist=amd-mi355x-5 -N1 --ntasks-per-node=8 --gres=gpu:8 \
       --mpi=pmix --export=ALL \
       $MY_PATH/all_reduce_perf -b 4096 -e 536870912 -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 \
       < /dev/null > $D/parity_1n/warmup_r${w}.log 2>&1
  echo "warmup_r${w} rc=$? $(date -u +%FT%TZ)" >> $D/parity_1n/PROGRESS
done
for rep in 1 2 3 4 5 6 7; do
  for arm in v4 v5; do
    export NCCL_TUNER_PLUGIN=$D/dn-tuner/librccl-tuner${arm}-dn.so
    export NCCL_DEBUG_FILE=$D/parity_1n/${arm}_r${rep}_dbg_%p.log
    srun --jobid=14470 --nodelist=amd-mi355x-5 -N1 --ntasks-per-node=8 --gres=gpu:8 \
         --mpi=pmix --export=ALL \
         $MY_PATH/all_reduce_perf -b 4096 -e 536870912 -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 \
         < /dev/null > $D/parity_1n/${arm}_r${rep}.log 2>&1
    echo "${arm}_r${rep} rc=$? $(date -u +%FT%TZ)" >> $D/parity_1n/PROGRESS
  done
done
echo "PARITY1N DONE $(date -u +%FT%TZ)" >> $D/parity_1n/PROGRESS
