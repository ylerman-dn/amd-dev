#!/bin/bash
# Verify candidate 1-node conf (only the measured wins) vs default. 7 repeats, round-robin.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
R=/opt/shared/ylerman/GPU-107/ab-v2/cand1n; mkdir -p $R/logs
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
CF=$R/cand_1n.csv
cat > $CF <<'CONF'
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
allreduce,131072,131072,-1,-1,32,1,8,-1,-1
allreduce,262144,262144,-1,-1,96,1,8,-1,-1
allreduce,2097152,4194304,-1,-1,32,1,8,-1,-1
allreduce,8388608,8388608,-1,-1,64,1,8,-1,-1
CONF
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
SZ="-b 131072 -e 536870912 -f 2 -n 5 -w 5 -c 1 -M 1 -R 1 -g 8"
O=$(salloc --no-shell -p XAI -N1 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 45 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
[ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== 1n JID=$JID $(squeue -j $JID -h -o %N) : cand vs default, 7 reps ==="
srun --jobid=$JID --overlap --gres=none -N1 --ntasks-per-node=1 sleep 3000 >/dev/null 2>&1 & KA=$!
for rep in 1 2 3 4 5 6 7; do
  for v in default cand; do
    EX=""; [ "$v" = cand ] && EX="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CF"
    for try in 1 2 3; do
      f=$R/logs/${v}_rep${rep}.log; [ $try -gt 1 ] && f=$R/logs/${v}_rep${rep}_try$try.log
      s=$(date +%s)
      timeout 200 srun --jobid=$JID -N1 --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
        bash -c "export $BASE $EX; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
      rc=$?; e=$(date +%s); nr=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$f")
      printf "%s,cand1n,1n,%s_rep%s,%s,%s,rows=%s\n" "$(date -u -d @$s +%FT%TZ)" "$v" "$rep" "$((e-s))" "$rc" "$nr" >> "$T"
      [ "$nr" -gt 0 ] && { echo "$v rep$rep ok ($((e-s))s)"; break; }
    done
  done
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo CAND1N_DONE
