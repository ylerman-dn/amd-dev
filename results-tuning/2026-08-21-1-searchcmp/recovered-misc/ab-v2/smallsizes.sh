#!/bin/bash
# Does the global env recipe HURT small collectives? env vars apply to the whole job.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
R=/opt/shared/ylerman/GPU-107/ab-v2/smallsizes; mkdir -p $R/logs
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 65536 -e 4194304 -f 2 -n 50 -w 20 -c 1 -M 1 -R 1 -g 8"
O=$(salloc --no-shell -p XAI -N3 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 100 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== HELD JID=$JID nodes=$(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N3 --ntasks-per-node=1 sleep 6000 >/dev/null 2>&1 & KA=$!
for N in 2 3; do
  echo "--- ${N}n: 64KB-4MB, 10 reps, default vs MIN=MAX=32 ---"
  for rep in $(seq 1 10); do
    for v in default env32; do
      ct=$(squeue -p XAI -h -t RUNNING -o "%u" | grep -vc "^dn$" || true)
      EX=""; [ "$v" = env32 ] && EX="NCCL_MIN_NCHANNELS=32 NCCL_MAX_NCHANNELS=32"
      f=$R/logs/${N}n_${v}_r${rep}.log; s=$(date +%s)
      timeout 200 srun --jobid=$JID -N$N --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
        bash -c "export $BASE $NET $EX; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
      rc=$?; e=$(date +%s); nr=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$f")
      printf "%s,smallsizes,%sn,%s_r%s,%s,%s,rows=%s;cotenant=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$v" "$rep" "$((e-s))" "$rc" "$nr" "$ct" >> "$T"
    done
  done
  echo "  ${N}n done $(date -u +%T)"
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo SMALL_DONE
