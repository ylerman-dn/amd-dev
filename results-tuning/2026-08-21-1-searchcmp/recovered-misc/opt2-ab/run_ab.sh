#!/bin/bash
# GPU-107 Option-2 A/B (current RCCL). Per scale N nodes, compare all_reduce busbw for 4 variants,
# applying each conf via the DN tuner plugin (one run each). Clean busbw (no NCCL_DEBUG). Times logged.
# Run ON a cluster node.  Usage:  bash run_ab.sh <N>     (N = 1, 2, or 3)
set -u
N="${1:?usage: run_ab.sh N}"
ROOT=/opt/shared/ylerman/GPU-107/opt2-ab
OUT="$ROOT/runs/${N}n"; mkdir -p "$OUT"
TIMES="$ROOT/AB_TIMES.csv"
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
NEWCONF=/opt/shared/ylerman/GPU-107/opt2-sweep/amd-dev/tools/rccl-sweep/generated_tuner.csv
OLDCONF=/opt/shared/ylerman/GPU-107/ab-tuner-test/confs/amddev_gfx950.conf
RULE3=/opt/shared/ylerman/GPU-107/ab-tuner-test/confs/amd_blog_3rule.conf
SZ="-b 8 -e 536870912 -f 2 -g 8"
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
mkdir -p "$(dirname "$TIMES")"
[ -f "$TIMES" ] || printf "utc_time,phase,scale_nodes,variant,duration_s,status,notes\n" > "$TIMES"

O=$(salloc --no-shell -p XAI -N "$N" --gres=gpu:8 -t 60 -J ylerman 2>&1)
JID=$(echo "$O" | grep -oE "job allocation [0-9]+" | grep -oE "[0-9]+" | tail -1)
[ -z "$JID" ] && { echo "NOALLOC: $O"; exit 1; }
echo "JID=$JID nodes=$(squeue -j "$JID" -h -o %N)"

run(){ # $1 variant  $2 extra-env
  local v=$1 extra=$2 s e rc st
  s=$(date +%s)
  if [ "$N" = 1 ]; then
    timeout 300 srun --jobid=$JID -N1 -n1 --gres=gpu:8 \
      bash -c "export $BASE $extra; $BIN/all_reduce_perf $SZ" > "$OUT/$v.log" 2>&1
  else
    timeout 400 srun --jobid=$JID -N"$N" --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export $BASE $NET $extra; $BIN/all_reduce_perf $SZ" > "$OUT/$v.log" 2>&1
  fi
  rc=$?; e=$(date +%s)
  st=$([ "$rc" -eq 0 ] && echo ok || echo "rc$rc")
  printf "%s,ab,%s,%s,%s,%s,jid=%s\n" "$(date -u +%FT%TZ)" "$N" "$v" "$((e-s))" "$st" "$JID" >> "$TIMES"
  echo "[${N}n $v] rc=$rc dur=$((e-s))s rows=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$OUT/$v.log")"
}

run default ""
run newconf "NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$NEWCONF"
run oldconf "NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$OLDCONF"
run rule3   "NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$RULE3"

scancel "$JID" && echo "released $JID"
echo "AB_DONE_${N}n"
