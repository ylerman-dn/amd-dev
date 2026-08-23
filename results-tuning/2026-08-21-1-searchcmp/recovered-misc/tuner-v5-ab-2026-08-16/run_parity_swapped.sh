#!/bin/bash
# GPU-107 v5-vs-v4 SWAPPED-ARMS tuner plugin PARITY A/B, 3 nodes.
# baseline arm = v4 plugin, config arm = v5 plugin, SAME rule file, constants untouched.
# Parameters copied from the reference run ab-2026-08-16/run_mn2.sh, except
# --warmup-runs 8 (user's instruction for 3-node runs, 2026-08-16 brief).
D=/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16
MY_PATH=/opt/shared/ylerman/GPU-107/bin
rm -rf $D/parity_3n_swapped; mkdir -p $D/parity_3n_swapped/logs
python3 $D/validate_tuner_config.py \
  --config $D/gen_3n_t1.validated.csv \
  --baseline-config $D/gen_3n_t1.validated.csv \
  --baseline-plugin $D/dn-tuner/librccl-tunerv5-dn.so \
  --plugin $D/dn-tuner/librccl-tunerv4-dn.so \
  --binary $MY_PATH/all_reduce_perf \
  --jobid 14470 --nodelist amd-mi355x-5,amd-mi355x-6,amd-mi355x-7 \
  --ranks-per-node 8 --gpus-per-rank 1 \
  --repeats 7 --warmup-runs 8 --preflight 5 \
  --min-bytes 4096 --max-bytes 536870912 \
  --iters 20 --warmup 5 --plugin-must-fire --split-ranges \
  --logdir $D/parity_3n_swapped/logs --times-csv $D/parity_3n_swapped/times.csv \
  > $D/parity_3n_swapped/validate.log 2>&1
echo "parity_3n_swapped rc=$? $(date -u +%FT%TZ)" >> $D/PARITY_PROGRESS
