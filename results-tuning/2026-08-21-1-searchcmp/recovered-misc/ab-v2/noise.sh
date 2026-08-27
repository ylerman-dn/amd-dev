#!/bin/bash
# Item 4: quantify run-to-run noise. 7 repeats, round-robin, per scale.
# Waits for plugsweep to finish so allocations never overlap.
set -u
while pgrep -f "[p]lugsweep.sh" >/dev/null; do sleep 30; done
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
R=/opt/shared/ylerman/GPU-107/ab-v2/noise; mkdir -p $R/logs $R/conf
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 536870912 -e 536870912 -f 2 -n 5 -w 5 -c 1 -M 1 -R 1"
REPS=7
for N in 2 3 1; do
  RK=$((N*8)); s0=$(date +%s)
  O=$(salloc --no-shell -p XAI -N $N --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 60 -J ylerman 2>&1)
  JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n"; continue; }
  echo "=== ${N}n JID=$JID $(squeue -j $JID -h -o %N) : $REPS reps x 4 configs ==="
  srun --jobid=$JID --overlap --gres=none -N$N --ntasks-per-node=1 sleep 4000 >/dev/null 2>&1 & KA=$!
  for CH in 64 256; do
    cf=$R/conf/ch${CH}_${N}n.csv
    printf "collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff\n" > $cf
    printf "allreduce,0,17179869184,-1,-1,%s,%s,%s,-1,-1\n" "$CH" "$N" "$RK" >> $cf
  done
  for rep in $(seq 1 $REPS); do
    for cfg in default plug64 plug256 g1default; do
      case $cfg in
        default)   EX=""; NT=1; G=8;;
        plug64)    EX="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$R/conf/ch64_${N}n.csv"; NT=1; G=8;;
        plug256)   EX="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$R/conf/ch256_${N}n.csv"; NT=1; G=8;;
        g1default) EX=""; NT=8; G=1;;
      esac
      f=$R/logs/${N}n_${cfg}_rep${rep}.log; s=$(date +%s)
      timeout 200 srun --jobid=$JID -N$N --ntasks-per-node=$NT --gres=gpu:8 --mpi=pmix --export=ALL \
        bash -c "export $BASE $NET $EX; $BIN/all_reduce_perf $SZ -g $G" > "$f" 2>&1
      rc=$?; e=$(date +%s); v=$(awk '/^ +536870912 .*float/{print $12}' "$f"|head -1)
      printf "%s,noise,%sn,%s_rep%s,%s,%s,busbw=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$cfg" "$rep" "$((e-s))" "$rc" "${v:-none}" >> "$T"
    done
  done
  kill $KA 2>/dev/null; scancel $JID
  echo "--- ${N}n results (7 reps each) ---"
  for cfg in default plug64 plug256 g1default; do
    vals=$(grep ",noise,${N}n,${cfg}_rep" $T | awk -F'busbw=' '{print $2}' | grep -v none | sort -n | tr '\n' ' ')
    echo "  $cfg: $vals"
  done
  printf "%s,noise_scale,%sn,all,%s,0,jid=%s\n" "$(date -u -d @$s0 +%FT%TZ)" "$N" "$(( $(date +%s)-s0 ))" "$JID" >> "$T"
done
echo NOISE_DONE
