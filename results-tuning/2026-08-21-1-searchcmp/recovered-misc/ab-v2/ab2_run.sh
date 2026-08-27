#!/bin/bash
# GPU-107 ab-v2 A/B: default vs 3rule vs new. Split size range (flaky-hang workaround) + retries.
set -u
ROOT=/opt/shared/ylerman/GPU-107/ab-v2
BIN=/opt/shared/ylerman/GPU-107/bin
PLUG=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
C3=$ROOT/conf/amd_blog_3rule.conf
CN=$ROOT/conf/new_tuner.csv
MD3=a126eb32e224be9ddeec393346664672
MDN=f17fbca0832f05f8a9e956118d0075d9
LOGD=$ROOT/logs; RUNLOG=$ROOT/RUNLOG.csv; mkdir -p "$LOGD"
BASE="LD_LIBRARY_PATH=$BIN:/opt/rocm/lib NCCL_DEBUG=VERSION"
NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
ARGS="-f 2 -g 8 -n 5 -w 5 -c 1 -M 1 -R 1"

rows(){ local c; c=$(grep -cE '^ +[0-9]+ +[0-9]+ +float' "$1" 2>/dev/null); echo "${c:-0}"; }

one(){ # $1 scale $2 variant $3 rep $4 split $5 jid $6 nodes
  local n=$1 v=$2 rep=$3 sp=$4 jid=$5 nodes=$6 extra="" md="-" sz try rc s e f id nr
  case "$v" in
    3rule) extra="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$C3"; md=$MD3;;
    new)   extra="NCCL_TUNER_PLUGIN=$PLUG NCCL_TUNER_CONFIG_FILE=$CN"; md=$MDN;;
  esac
  [ "$sp" = A ] && sz="-b 65536 -e 524288" || sz="-b 1048576 -e 536870912"
  for try in 1 2 3; do
    id="ab2_${n}n_${v}_rep${rep}_${sp}"; [ $try -gt 1 ] && id="${id}_try${try}"
    f="$LOGD/$id.log"; s=$(date +%s)
    timeout 240 srun --jobid=$jid -N"$n" --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export $BASE $NET $extra; $BIN/all_reduce_perf $sz $ARGS" > "$f" 2>&1
    rc=$?; e=$(date +%s); nr=$(rows "$f")
    if [ "$v" = default ]; then pl=none; else grep -q "Using DN-TUNER" "$f" && pl=yes || pl="na(VERSION)"; fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,"%s",%s,%s,%s,%s\n' \
      "$id" "$(date -u -d @$s +%FT%TZ)" "$(date -u -d @$e +%FT%TZ)" "$((e-s))" \
      "$n" "$v" "$rep" "$jid" "$nodes" "$md" "$pl" "$rc" "$f" >> "$RUNLOG"
    echo "  [${n}n $v rep$rep $sp try$try] rc=$rc rows=$nr dur=$((e-s))s plug=$pl"
    [ "$nr" -gt 0 ] && return 0
  done
  echo "  !! ${n}n $v rep$rep $sp FAILED after 3 tries"; return 1
}

for N in 1 2 3; do
  echo "=== scale ${N}n : allocating ==="
  O=$(salloc --no-shell -p XAI -N "$N" --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 60 -J ylerman 2>&1)
  JID=$(echo "$O" | grep -oE 'job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)
  [ -z "$JID" ] && { echo "NOALLOC ${N}n: $O"; continue; }
  NODES=$(squeue -j "$JID" -h -o %N)
  echo "=== scale ${N}n JID=$JID nodes=$NODES ==="
  srun --jobid="$JID" --overlap --gres=none -N"$N" --ntasks-per-node=1 sleep 3600 >/dev/null 2>&1 &
  KA=$!
  for rep in 1 2 3; do
    for v in default 3rule new; do      # round-robin: variants inside each rep
      for sp in A B; do one "$N" "$v" "$rep" "$sp" "$JID" "$NODES"; done
    done
  done
  kill $KA 2>/dev/null; scancel "$JID"; echo "=== scale ${N}n done, released $JID ==="
done
echo "ALL_SCALES_DONE"
