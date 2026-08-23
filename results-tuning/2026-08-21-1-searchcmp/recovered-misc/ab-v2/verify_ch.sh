#!/bin/bash
# Does NCCL_MIN_NCHANNELS actually pin the channel count? Read NChannels: from INFO.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
V=/opt/shared/ylerman/GPU-107/ab-v2/verify; mkdir -p $V
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH"
SZ="-b 536870912 -e 536870912 -f 2 -n 2 -w 1 -c 0 -M 1 -R 1 -g 1"
O=$(salloc --no-shell -p XAI -N1 --exclude=amd-mi355x-2,amd-mi355x-9,amd-mi355x-5,amd-mi355x-7 --gres=gpu:8 -t 30 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
[ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== 1n JID=$JID $(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N1 --ntasks-per-node=1 sleep 1500 >/dev/null 2>&1 & KA=$!
printf "%-16s %-14s %-26s %s\n" "MIN setting" "busbw_ip" "NChannels: seen" "rings/graph"
for CH in "" 32 64 128 224; do
  lbl=${CH:-none}; f=$V/chk_MIN${lbl}.log; s=$(date +%s)
  env_s=""; [ -n "$CH" ] && env_s="NCCL_MIN_NCHANNELS=$CH"
  timeout 150 srun --jobid=$JID -N1 --ntasks-per-node=8 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "export $BASE $env_s NCCL_ALGO=RING NCCL_PROTO=SIMPLE; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
  rc=$?; e=$(date +%s)
  v=$(awk '/^ +536870912 /{print $12}' "$f"|head -1)
  nc=$(grep -hoE "NChannels:[0-9]+" "$f"|sort -u|tr '\n' ' ')
  rg=$(grep -hoE "Nchannels [0-9]+" "$f"|sort -u|tr '\n' ' ')
  printf "%-16s %-14s %-26s %s\n" "$lbl" "${v:-NONE}" "${nc:-(none logged)}" "${rg:-}"
  printf "%s,verify_ch,1n,MIN=%s,%s,%s,busbw=%s;NChannels=%s\n" "$(date -u -d @$s +%FT%TZ)" "$lbl" "$((e-s))" "$rc" "${v:-none}" "${nc// /|}" >> "$T"
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo CHK_DONE
