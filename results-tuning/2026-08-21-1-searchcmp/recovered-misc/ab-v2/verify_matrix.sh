#!/bin/bash
# Which forced algo/proto combos does RCCL actually honour? Decides sweep data validity.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/ab-v2/verify; mkdir -p $OUT
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
[ -f "$T" ] || printf "utc_start,phase,scale,detail,duration_s,rc,notes\n" > "$T"
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 8388608 -e 16777216 -f 2 -n 2 -w 1 -c 0 -M 1 -R 1 -g 1"

for N in 1 2; do
  s0=$(date +%s)
  O=$(salloc --no-shell -p XAI -N $N --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 30 -J ylerman 2>&1)
  JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n"; continue; }
  echo "=== ${N}n JID=$JID $(squeue -j $JID -h -o %N) ==="
  srun --jobid=$JID --overlap --gres=none -N$N --ntasks-per-node=1 sleep 2400 >/dev/null 2>&1 & KA=$!
  printf "%-16s %-24s %s\n" "FORCED" "RCCL ACTUALLY RAN" "honoured?"
  for A in RING TREE; do for P in SIMPLE LL LL128; do
    f=$OUT/${N}n_force_${A}_${P}.log; s=$(date +%s)
    timeout 200 srun --jobid=$JID -N$N --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export $BASE $NET NCCL_ALGO=$A NCCL_PROTO=$P NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64; $BIN/all_reduce_perf $SZ" > $f 2>&1
    rc=$?; e=$(date +%s)
    got=$(grep -hoE "Algo [A-Z]+ proto [A-Za-z0-9]+" $f | sort -u | head -1 | sed 's/Algo //;s/ proto /\//')
    [ -z "$got" ] && got="(no data rc=$rc)"
    ok=$([ "$got" = "$A/$P" ] && echo YES || echo "NO -> fell back")
    printf "%-16s %-24s %s\n" "$A/$P" "$got" "$ok"
    printf "%s,verify_matrix,%sn,%s,%s,%s,got=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$A/$P" "$((e-s))" "$rc" "$got" >> "$T"
  done; done
  kill $KA 2>/dev/null; scancel $JID
  printf "%s,verify_matrix_scale,%sn,all6combos,%s,0,jid=%s\n" "$(date -u -d @$s0 +%FT%TZ)" "$N" "$(( $(date +%s)-s0 ))" "$JID" >> "$T"
  echo "=== ${N}n released ==="
done
echo MATRIX_DONE
