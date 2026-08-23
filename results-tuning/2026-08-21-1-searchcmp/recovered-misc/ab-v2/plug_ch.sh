#!/bin/bash
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
CN=/opt/shared/ylerman/GPU-107/ab-v2/conf/new_tuner.csv
V=/opt/shared/ylerman/GPU-107/ab-v2/verify; T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH"
SZ="-b 536870912 -e 536870912 -f 2 -n 5 -w 5 -c 1 -M 1 -R 1"
O=$(salloc --no-shell -p XAI -N1 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 25 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && exit 1
echo "=== JID=$JID $(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N1 --ntasks-per-node=1 sleep 1200 >/dev/null 2>&1 & KA=$!
go(){ # label ntasks g env
  local f=$V/pch_$1.log s e v nc
  s=$(date +%s)
  timeout 180 srun --jobid=$JID -N1 --ntasks-per-node=$2 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "export $BASE $4; $BIN/all_reduce_perf $SZ -g $3" > "$f" 2>&1
  e=$(date +%s); v=$(awk '/^ +536870912 /{print $12}' "$f"|head -1)
  nc=$(grep -hoE "NChannels:[0-9]+" "$f"|sort -u|tr '\n' ' ')
  rg=$(grep -hoE "Nchannels [0-9]+" "$f"|sort -u|tr '\n' ' ')
  printf "  %-22s busbw=%-8s NChannels:{%s} graph:{%s}\n" "$1" "${v:-NONE}" "${nc:-none}" "${rg:-none}"
  printf "%s,plug_ch,1n,%s,%s,0,busbw=%s;NCh=%s\n" "$(date -u -d @$s +%FT%TZ)" "$1" "$((e-s))" "${v:-none}" "${nc// /|}" >> "$T"
}
go env_MIN64   8 1 "NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_MIN_NCHANNELS=64"
go plugin_g1   8 1 "NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CN"
go plugin_g8   1 8 "NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CN"
kill $KA 2>/dev/null; scancel $JID; echo "=== released ==="; echo PCH_DONE
