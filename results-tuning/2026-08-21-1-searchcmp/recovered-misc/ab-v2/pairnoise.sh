#!/bin/bash
# Is 2-node noise pair-dependent (bad NIC/link) or intrinsic? 5 reps of the SAME config per pair.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
R=/opt/shared/ylerman/GPU-107/ab-v2/pairnoise; mkdir -p $R/logs
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 536870912 -e 536870912 -f 2 -n 5 -w 5 -c 1 -M 1 -R 1 -g 8"
for PAIR in "amd-mi355x-1,amd-mi355x-3" "amd-mi355x-4,amd-mi355x-6" "amd-mi355x-5,amd-mi355x-7" "amd-mi355x-3,amd-mi355x-4"; do
  s0=$(date +%s)
  O=$(salloc --no-shell -p XAI -w "$PAIR" --gres=gpu:8 -t 30 -J ylerman 2>&1)
  JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
  [ -z "$JID" ] && { echo "NOALLOC $PAIR : $(echo $O|head -c 90)"; continue; }
  others=$(squeue -h -o "%u %N" | grep -v ylerman | wc -l)
  srun --jobid=$JID --overlap --gres=none -N2 --ntasks-per-node=1 sleep 2000 >/dev/null 2>&1 & KA=$!
  vals=""
  for rep in 1 2 3 4 5; do
    f=$R/logs/$(echo $PAIR|tr ',' '_')_rep${rep}.log; s=$(date +%s)
    timeout 200 srun --jobid=$JID -N2 --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export $BASE $NET; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
    rc=$?; e=$(date +%s); v=$(awk '/^ +536870912 .*float/{print $12}' "$f"|head -1)
    vals="$vals ${v:-NONE}"
    printf "%s,pairnoise,2n,%s_rep%s,%s,%s,busbw=%s;otherjobs=%s\n" "$(date -u -d @$s +%FT%TZ)" "$(echo $PAIR|tr ',' '_')" "$rep" "$((e-s))" "$rc" "${v:-none}" "$others" >> "$T"
  done
  kill $KA 2>/dev/null; scancel $JID
  echo "$PAIR (other jobs on cluster: $others):$vals"
done
echo PAIRNOISE_DONE
