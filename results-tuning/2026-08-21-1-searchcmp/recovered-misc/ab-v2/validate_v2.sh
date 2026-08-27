#!/bin/bash
# End-to-end validation of the DELIVERED conf file vs default. 7 reps, 1n/2n/3n, quiet window.
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
CF=/opt/shared/ylerman/GPU-107/ab-v2/final/rccl_tuner_verified_v2.conf
R=/opt/shared/ylerman/GPU-107/ab-v2/validate; mkdir -p $R/logs
T=/opt/shared/ylerman/GPU-107/ab-v2/TIMES.csv
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib LD_PRELOAD=$BIN/librccl.so NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
SZ="-b 1048576 -e 536870912 -f 2 -n 20 -w 10 -c 1 -M 1 -R 1 -g 8"
O=$(salloc --no-shell -p XAI -N3 --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 110 -J ylerman 2>&1)
JID=$(echo "$O"|grep -oE 'job allocation [0-9]+'|grep -oE '[0-9]+'|tail -1); [ -z "$JID" ] && { echo NOALLOC; exit 1; }
echo "=== HELD JID=$JID nodes=$(squeue -j $JID -h -o %N) ==="
srun --jobid=$JID --overlap --gres=none -N3 --ntasks-per-node=1 sleep 6600 >/dev/null 2>&1 & KA=$!
for N in 1 2 3; do
  echo "--- ${N}n: conf-file vs default, 7 reps ---"
  for rep in 1 2 3 4 5 6 7; do
    for v in default confv2; do
      ct=$(squeue -p XAI -h -t RUNNING -o "%u" | grep -vc "^dn$" || true)
      EX=""; [ "$v" = confv2 ] && EX="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CF"
      for try in 1 2; do
        f=$R/logs/${N}n_${v}_r${rep}.log; [ $try -gt 1 ] && f=$R/logs/${N}n_${v}_r${rep}_try2.log
        s=$(date +%s)
        timeout 220 srun --jobid=$JID -N$N --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
          bash -c "export $BASE $NET $EX; $BIN/all_reduce_perf $SZ" > "$f" 2>&1
        rc=$?; e=$(date +%s); nr=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$f")
        printf "%s,validate,%sn,%s_r%s,%s,%s,rows=%s;cotenant=%s\n" "$(date -u -d @$s +%FT%TZ)" "$N" "$v" "$rep" "$((e-s))" "$rc" "$nr" "$ct" >> "$T"
        [ "$nr" -gt 0 ] && break
      done
    done
  done
  echo "  ${N}n done $(date -u +%T)"
done
kill $KA 2>/dev/null; scancel $JID; echo "=== released $JID ==="; echo VALIDATE_DONE
