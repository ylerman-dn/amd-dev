#!/bin/bash
# End-to-end proof: run the new validator against the ORIGINAL 17-rule conf.
# Expected: hang-rate gate trips, and most/all rules get pruned (11 are ll128 = inert, one caused -54%).
set -u
BASE=/opt/shared/ylerman/GPU-107
V=$BASE/opt2-sweep/amd-dev/tools/rccl-sweep/validate_tuner_config.py
CONF=$BASE/ab-v2/validator_demo2/orig_17rule.csv
R=$BASE/ab-v2/validator_demo2; mkdir -p $R
cp $BASE/ab-v2/conf/new_tuner.csv $CONF
O=$(salloc --no-shell -p XAI -N3 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 110 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== HELD JID=$JID nodes=$(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N3 --ntasks-per-node=1 sleep 6600 >/dev/null 2>&1 & KA=$!
python3 -u "$V" \
  --config "$CONF" \
  --binary $BASE/bin/all_reduce_perf \
  --plugin $BASE/ab-tuner-test/librccl-tunerv4-dn.so \
  --launcher "srun --jobid=$JID" \
  --repeats 5 --logdir $R/logs --times-csv $BASE/ab-v2/TIMES.csv \
  --env NCCL_IB_GID_INDEX=1 --env NCCL_SOCKET_IFNAME=enp81s0f1np1 \
  --env OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 --env OMPI_MCA_oob_tcp_if_include=enp81s0f1np1
rc=$?
kill $KA 2>/dev/null; scancel $JID
echo "=== released $JID (validator exit=$rc) ==="
echo VALIDATOR_DEMO_DONE
