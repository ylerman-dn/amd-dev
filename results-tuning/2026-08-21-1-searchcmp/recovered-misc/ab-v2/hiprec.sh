#!/bin/bash
# High-precision 2M/4M: -n 200 -w 50 (long sampling shrinks per-run variance) x 20 repeats.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
R=/opt/shared/ylerman/GPU-107/ab-v2/hiprec; mkdir -p $R/logs $R/conf
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
printf "collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff\nallreduce,2097152,4194304,-1,-1,32,1,8,-1,-1\n" > $R/conf/ch32.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
SZ="-b 2097152 -e 4194304 -f 2 -n 200 -w 50 -c 1 -M 1 -R 1 -g 8"
O=$(salloc --no-shell -p XAI -N1 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 55 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== 1n JID=$JID $(squeue -j $JID -h -o %N): 20 reps x 2 variants, n=200 iters ==="
srun --jobid=$JID --overlap --gres=none -N1 --ntasks-per-node=1 sleep 3300 >/dev/null 2>&1 & KA=$!
for rep in $(seq 1 20); do
  for v in default ch32; do
    EX=""; [ "$v" = ch32 ] && EX="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$R/conf/ch32.csv"
    f=$R/logs/${v}_r${rep}.log; s=$(date +%s)
    timeout 200 srun --jobid=$JID -N1 --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export $BASE $EX; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
    rc=$?; e=$(date +%s)
    printf "%s,hiprec,1n,%s_r%s,%s,%s,rows=%s\n" "$(date -u -d @$s +%FT%TZ)" "$v" "$rep" "$((e-s))" "$rc" "$(grep -cE '^ +[0-9]+ +[0-9]+ +float' $f)" >> "$T"
  done
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo HIPREC_DONE
