#!/bin/bash
# PLUGIN-PATH channel sweep: measure through the deployment mechanism (channels-only confs).
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
R=/opt/shared/ylerman/GPU-107/ab-v2/plugsweep; mkdir -p $R/logs $R/conf
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
ARGS="-f 2 -n 5 -w 5 -c 1 -M 1 -R 1 -g 8"
rows(){ local c; c=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$1" 2>/dev/null); echo "${c:-0}"; }

for N in 1 2 3; do
  RK=$((N*8)); s0=$(date +%s)
  O=$(salloc --no-shell -p XAI -N $N --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 60 -J ylerman 2>&1)
  JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n"; continue; }
  echo "=== ${N}n JID=$JID $(squeue -j $JID -h -o %N) ==="
  srun --jobid=$JID --overlap --gres=none -N$N --ntasks-per-node=1 sleep 4000 >/dev/null 2>&1 & KA=$!
  for CH in none 32 64 96 128 160 192 224 256; do
    if [ "$CH" = none ]; then EXTRA=""
    else
      cf=$R/conf/ch${CH}_${N}n.csv
      printf "collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff\n" > $cf
      printf "allreduce,0,17179869184,-1,-1,%s,%s,%s,-1,-1\n" "$CH" "$N" "$RK" >> $cf
      EXTRA="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$cf"
    fi
    for sp in A B; do
      [ $sp = A ] && SZ="-b 65536 -e 524288" || SZ="-b 1048576 -e 536870912"
      for try in 1 2 3; do
        id="${N}n_ch${CH}_${sp}"; [ $try -gt 1 ] && id="${id}_try$try"
        f=$R/logs/$id.log; s=$(date +%s)
        timeout 200 srun --jobid=$JID -N$N --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
          bash -c "export $BASE $NET $EXTRA; $BIN/all_reduce_perf $SZ $ARGS" > "$f" 2>&1
        rc=$?; e=$(date +%s); nr=$(rows "$f")
        printf "%s,plugsweep,%sn,ch%s_%s,%s,%s,rows=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$CH" "$sp" "$((e-s))" "$rc" "$nr" >> "$T"
        [ "$nr" -gt 0 ] && break
      done
    done
    v=$(awk '/^ +536870912 .*float/{print $12}' $R/logs/${N}n_ch${CH}_B.log 2>/dev/null|head -1)
    printf "  %sn ch=%-5s 512MB busbw_ip=%s\n" "$N" "$CH" "${v:-NONE}"
  done
  kill $KA 2>/dev/null; scancel $JID
  printf "%s,plugsweep_scale,%sn,all,%s,0,jid=%s\n" "$(date -u -d @$s0 +%FT%TZ)" "$N" "$(( $(date +%s)-s0 ))" "$JID" >> "$T"
  echo "=== ${N}n released after $(( ($(date +%s)-s0)/60 ))min ==="
done
echo PLUGSWEEP_DONE
