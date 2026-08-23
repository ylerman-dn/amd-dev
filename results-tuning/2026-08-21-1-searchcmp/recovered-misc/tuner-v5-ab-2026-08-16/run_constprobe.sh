#!/bin/bash
# GPU-107 constants live/dead probe. v5 plugin + empty rules file, one run per arm:
#   noconst — constants logged, untouched
#   extreme — constants_extreme.txt applied (absurd values)
# If selections (-A 1 algo/proto columns) differ anywhere, constants are LIVE on this
# runtime; identical selections everywhere = DEAD. Benchmark flags identical to the
# standard measured run. Selection probe: 1 run per arm per scale, no timing claims.
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
MY_PATH=/opt/shared/ylerman/GPU-107/bin
mkdir -p $D/constprobe
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
export NCCL_TUNER_PLUGIN=$D/dn-tuner/librccl-tunerv5-dn.so
export NCCL_TUNER_CONFIG_FILE=$D/empty_rules.csv
run_one() {
  scale=$1; nodelist=$2; arm=$3
  export NCCL_DEBUG_FILE=$D/constprobe/${scale}n_${arm}_dbg_%p.log
  if [ "$arm" = extreme ]; then export NCCL_TUNER_CONSTANTS_FILE=$D/constants_extreme.txt
  else unset NCCL_TUNER_CONSTANTS_FILE; fi
  srun --jobid=14470 --nodelist=$nodelist -N$scale --ntasks-per-node=8 --gres=gpu:8 \
       --mpi=pmix --export=ALL \
       $MY_PATH/all_reduce_perf -b 4096 -e 536870912 -f 2 -g 1 -n 20 -w 5 -c 1 -A 1 \
       < /dev/null > $D/constprobe/${scale}n_${arm}.log 2>&1
  echo "${scale}n_${arm} rc=$? $(date -u +%FT%TZ)" >> $D/constprobe/PROGRESS
}
run_one 1 amd-mi355x-5 noconst
run_one 1 amd-mi355x-5 extreme
run_one 3 amd-mi355x-5,amd-mi355x-6,amd-mi355x-7 noconst
run_one 3 amd-mi355x-5,amd-mi355x-6,amd-mi355x-7 extreme
echo "CONSTPROBE DONE $(date -u +%FT%TZ)" >> $D/constprobe/PROGRESS
