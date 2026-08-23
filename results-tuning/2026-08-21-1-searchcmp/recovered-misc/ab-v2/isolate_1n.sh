#!/bin/bash
# Why did 1n/512MB drop to ~180 in the A/B when env-forced 64ch gives ~395?
# Isolate: -g 1 vs -g 8, env-forced vs tuner-plugin.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
CN=/opt/shared/ylerman/GPU-107/ab-v2/conf/new_tuner.csv
O_=/opt/shared/ylerman/GPU-107/ab-v2/verify; mkdir -p $O_
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
SZ="-b 536870912 -e 536870912 -f 2 -n 5 -w 5 -c 1 -M 1 -R 1"
O=$(salloc --no-shell -p XAI -N1 --exclude=amd-mi355x-2,amd-mi355x-9,amd-mi355x-5,amd-mi355x-7 --gres=gpu:8 -t 30 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
[ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== 1n JID=$JID $(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N1 --ntasks-per-node=1 sleep 1800 >/dev/null 2>&1 & KA=$!
t(){ # $1 label  $2 ntasks  $3 -g  $4 env
  local f=$O_/iso_$1.log s e rc v
  s=$(date +%s)
  timeout 200 srun --jobid=$JID -N1 --ntasks-per-node=$2 --gres=gpu:8 --mpi=pmix --export=ALL \
    bash -c "export $BASE $4; $BIN/all_reduce_perf $SZ -g $3" > "$f" 2>&1
  rc=$?; e=$(date +%s); v=$(awk '/^ +536870912 /{print $12}' "$f"|head -1)
  printf "  %-34s busbw_ip=%-9s rc=%s dur=%ss\n" "$1" "${v:-NONE}" "$rc" "$((e-s))"
  printf "%s,isolate_1n,1n,%s,%s,%s,busbw=%s\n" "$(date -u -d @$s +%FT%TZ)" "$1" "$((e-s))" "$rc" "${v:-none}" >> "$T"
}
t "g1_8procs_default"    8 1 ""
t "g1_8procs_MIN64"      8 1 "NCCL_MIN_NCHANNELS=64"
t "g8_1proc_default"     1 8 ""
t "g8_1proc_MIN64"       1 8 "NCCL_MIN_NCHANNELS=64"
t "g8_1proc_MINMAX64"    1 8 "NCCL_MIN_NCHANNELS=64 NCCL_MAX_NCHANNELS=64"
t "g8_1proc_PLUGIN_new"  1 8 "NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CN"
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo ISO_DONE
