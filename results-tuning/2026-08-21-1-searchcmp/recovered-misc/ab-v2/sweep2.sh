#!/bin/bash
# Corrected sweep: honoured combos only, MIN-only channels (avoids 1n hang), split size range.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
R=/opt/shared/ylerman/GPU-107/ab-v2/sweep2; mkdir -p $R/logs
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
ARGS="-f 2 -n 5 -w 5 -c 1 -M 1 -R 1 -g 1"
rows(){ local c; c=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$1" 2>/dev/null); echo "${c:-0}"; }

for N in 1 2 3; do
  COMBOS="RING/SIMPLE RING/LL TREE/LL"
  [ $N -gt 1 ] && COMBOS="$COMBOS TREE/SIMPLE"
  s0=$(date +%s)
  O=$(salloc --no-shell -p XAI -N $N --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 90 -J ylerman 2>&1)
  JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n"; continue; }
  NODES=$(squeue -j $JID -h -o %N); echo "=== ${N}n JID=$JID $NODES ==="
  srun --jobid=$JID --overlap --gres=none -N$N --ntasks-per-node=1 sleep 6000 >/dev/null 2>&1 & KA=$!
  for c in $COMBOS; do A=${c%/*}; P=${c#*/}
    for CH in 32 64 96 128 160 192 224 256; do
      for sp in A B; do
        [ $sp = A ] && SZ="-b 65536 -e 524288" || SZ="-b 1048576 -e 536870912"
        for try in 1 2; do
          id="${N}n_${A}_${P}_${CH}ch_${sp}"; [ $try -gt 1 ] && id="${id}_try2"
          f=$R/logs/$id.log; s=$(date +%s)
          timeout 120 srun --jobid=$JID -N$N --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
            bash -c "export $BASE $NET NCCL_ALGO=$A NCCL_PROTO=$P NCCL_MIN_NCHANNELS=$CH; $BIN/all_reduce_perf $SZ $ARGS" > "$f" 2>&1
          rc=$?; e=$(date +%s); nr=$(rows "$f")
          printf "%s,sweep2,%sn,%s/%s_%sch_%s,%s,%s,rows=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$A" "$P" "$CH" "$sp" "$((e-s))" "$rc" "$nr" >> "$T"
          [ "$nr" -gt 0 ] && break
        done
      done
    done
    echo "  ${N}n $c done ($(date -u +%T))"
  done
  kill $KA 2>/dev/null; scancel $JID
  printf "%s,sweep2_scale,%sn,all,%s,0,jid=%s\n" "$(date -u -d @$s0 +%FT%TZ)" "$N" "$(( $(date +%s)-s0 ))" "$JID" >> "$T"
  echo "=== ${N}n released after $(( ($(date +%s)-s0)/60 ))min ==="
done
echo SWEEP2_DONE
