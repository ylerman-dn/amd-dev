#!/bin/bash
# 1n high-channel probe: which env style lets us reach >=96 channels without hanging?
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
OUT=/opt/shared/ylerman/GPU-107/ab-v2/verify; mkdir -p $OUT
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING"
SZ="-b 67108864 -e 67108864 -f 2 -n 2 -w 1 -c 0 -M 1 -R 1 -g 1"
O=$(salloc --no-shell -p XAI -N1 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 25 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
[ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== 1n JID=$JID $(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N1 --ntasks-per-node=1 sleep 1500 >/dev/null 2>&1 & KA=$!
probe(){ # $1 label  $2 env
  local f=$OUT/probe_$1.log s e rc nr act
  s=$(date +%s)
  timeout 60 srun --jobid=$JID -N1 --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "export $BASE NCCL_ALGO=RING NCCL_PROTO=SIMPLE $2; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
  rc=$?; e=$(date +%s); nr=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$f")
  act=$(grep -hoE "channel\{Lo\.\.Hi\}=\{[0-9]+\.\.[0-9]+\}" "$f" | head -1)
  cap=$(grep -hoE "MaxChannels[^,]*" "$f" | head -1)
  printf "%-28s rc=%-4s rows=%-3s dur=%3ss  actual=%-24s %s\n" "$1" "$rc" "$nr" "$((e-s))" "${act:-none}" "$cap"
  printf "%s,probe_channels,1n,%s,%s,%s,rows=%s;%s\n" "$(date -u -d @$s +%FT%TZ)" "$1" "$((e-s))" "$rc" "$nr" "${act:-none}" >> "$T"
}
probe "baseline_nochanenv"   ""
probe "MINMAX_64"            "NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64"
probe "MINMAX_96"            "NCCL_MIN_NCHANNELS=96 NCCL_MAX_NCHANNELS=96"
probe "MAXonly_96"           "NCCL_MAX_NCHANNELS=96"
probe "MINonly_96"           "NCCL_MIN_NCHANNELS=96"
probe "MINMAX_224"           "NCCL_MIN_NCHANNELS=224 NCCL_MAX_NCHANNELS=224"
probe "MAXonly_224"          "NCCL_MAX_NCHANNELS=224"
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo PROBE_DONE
