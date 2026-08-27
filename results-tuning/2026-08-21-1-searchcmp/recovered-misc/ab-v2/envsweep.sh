#!/bin/bash
# Is 32 actually the optimum via the ENV route? Sweep channel counts the way we deploy.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
R=/opt/shared/ylerman/GPU-107/ab-v2/envsweep; mkdir -p $R/logs
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 8388608 -e 536870912 -f 2 -n 20 -w 10 -c 1 -M 1 -R 1 -g 8"
O=$(salloc --no-shell -p XAI -N2 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 100 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== HELD JID=$JID nodes=$(squeue -j $JID -h -o %N) : 2n env channel sweep ==="
srun --jobid=$JID --overlap --gres=none -N2 --ntasks-per-node=1 sleep 6000 >/dev/null 2>&1 & KA=$!
for rep in 1 2 3 4 5; do
  for CH in default 8 16 24 32 48 64; do
    EX=""; [ "$CH" != default ] && EX="NCCL_MIN_NCHANNELS=$CH NCCL_MAX_NCHANNELS=$CH"
    f=$R/logs/2n_${CH}_r${rep}.log; s=$(date +%s)
    timeout 200 srun --jobid=$JID -N2 --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export $BASE $NET $EX; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
    rc=$?; e=$(date +%s); nr=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$f")
    printf "%s,envsweep,2n,ch%s_r%s,%s,%s,rows=%s\n" "$(date -u -d @$s +%FT%TZ)" "$CH" "$rep" "$((e-s))" "$rc" "$nr" >> "$T"
  done
  echo "  rep$rep done $(date -u +%T)"
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo ENVSWEEP_DONE
