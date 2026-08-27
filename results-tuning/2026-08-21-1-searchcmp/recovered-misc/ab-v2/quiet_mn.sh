#!/bin/bash
# QUIET-WINDOW multi-node measurement: cluster was empty of other tenants.
# One 3-node allocation held for the whole phase (no realloc churn).
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
R=/opt/shared/ylerman/GPU-107/ab-v2/quiet_mn; mkdir -p $R/logs $R/conf
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 1048576 -e 536870912 -f 2 -n 20 -w 10 -c 1 -M 1 -R 1 -g 8"
O=$(salloc --no-shell -p XAI -N3 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 110 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && { echo NOALLOC; exit 1; }
NODES=$(squeue -j $JID -h -o %N)
echo "=== HELD JID=$JID nodes=$NODES for the whole phase ==="
srun --jobid=$JID --overlap --gres=none -N3 --ntasks-per-node=1 sleep 6600 >/dev/null 2>&1 & KA=$!
for N in 2 3; do
  RK=$((N*8))
  for CH in 32 64 128 256; do
    printf "collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff\nallreduce,0,17179869184,-1,-1,%s,%s,%s,-1,-1\n" $CH $N $RK > $R/conf/ch${CH}_${N}n.csv
  done
  echo "--- ${N}n : 7 reps x 5 variants ---"
  for rep in 1 2 3 4 5 6 7; do
    for v in default ch32 ch64 ch128 ch256; do
      others=$(squeue -p XAI -h -t RUNNING -o "%u" | grep -vc "^dn$" || true)
      EX=""; [ "$v" != default ] && EX="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$R/conf/${v}_${N}n.csv"
      for try in 1 2; do
        f=$R/logs/${N}n_${v}_r${rep}.log; [ $try -gt 1 ] && f=$R/logs/${N}n_${v}_r${rep}_try2.log
        s=$(date +%s)
        timeout 220 srun --jobid=$JID -N$N --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
          bash -c "export $BASE $NET $EX; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
        rc=$?; e=$(date +%s); nr=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$f")
        printf "%s,quiet_mn,%sn,%s_r%s,%s,%s,rows=%s;cotenant=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$v" "$rep" "$((e-s))" "$rc" "$nr" "$others" >> "$T"
        [ "$nr" -gt 0 ] && break
      done
    done
    echo "  ${N}n rep$rep done $(date -u +%T)"
  done
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo QUIET_MN_DONE
