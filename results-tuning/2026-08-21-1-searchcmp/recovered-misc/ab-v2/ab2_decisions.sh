#!/bin/bash
# Decision capture: NCCL_DEBUG=INFO, ONE run per (scale,variant). NOT for timing.
set -u
ROOT=/opt/shared/ylerman/GPU-107/ab-v2
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
C3=$ROOT/conf/amd_blog_3rule.conf; CN=$ROOT/conf/new_tuner.csv
D=$ROOT/decisions; mkdir -p "$D"
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
ARGS="-b 65536 -e 536870912 -f 2 -g 8 -n 2 -w 1 -c 0 -M 1 -R 1"
rows(){ local c; c=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$1" 2>/dev/null); echo "${c:-0}"; }

for N in 1 2 3; do
  O=$(salloc --no-shell -p XAI -N "$N" --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 45 -J ylerman 2>&1)
  JID=$(echo "$O" | grep -oE 'job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n"; continue; }
  echo "=== ${N}n JID=$JID nodes=$(squeue -j "$JID" -h -o %N) ==="
  srun --jobid="$JID" --overlap --gres=none -N"$N" --ntasks-per-node=1 sleep 3000 >/dev/null 2>&1 & KA=$!
  for v in default 3rule new; do
    extra=""
    [ "$v" = 3rule ] && extra="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$C3"
    [ "$v" = new   ] && extra="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CN"
    for try in 1 2 3; do
      f="$D/${N}n_${v}_INFO.log"; [ $try -gt 1 ] && f="$D/${N}n_${v}_INFO_try${try}.log"
      timeout 300 srun --jobid=$JID -N"$N" --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
        bash -c "export $BASE $NET $extra; $BIN/all_reduce_perf $ARGS" > "$f" 2>&1
      rc=$?; nr=$(rows "$f")
      echo "  [${N}n $v try$try] rc=$rc rows=$nr size=$(stat -c%s "$f")"
      [ "$nr" -gt 0 ] && { ln -sf "$(basename $f)" "$D/${N}n_${v}_USE.log" 2>/dev/null; break; }
    done
  done
  kill $KA 2>/dev/null; scancel "$JID"; echo "=== ${N}n released ==="
done
echo "DECISIONS_DONE"
