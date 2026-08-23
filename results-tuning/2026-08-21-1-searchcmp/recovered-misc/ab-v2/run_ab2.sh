#!/bin/bash
# GPU-107 ab-v2: bench-runner. 3-way A/B (default / 3rule / new) x 1/2/3 nodes.
# Splits size range in two (A: 64K-512K, B: 1M-512M) to dodge the known intermittent
# tuner-plugin comm-init hang; retries up to 3x per invocation with a 240s timeout.
# Round-robin reps: rep(default,3rule,new) x3.
set -u

BASE=/opt/shared/ylerman/GPU-107
AB=$BASE/ab-v2
LOGDIR=$AB/logs
CSV=$AB/RUNLOG.csv
PLUGIN=$BASE/ab-tuner-test/librccl-tunerv4-dn.so
CONF_3RULE=$AB/conf/amd_blog_3rule.conf
CONF_NEW=$AB/conf/new_tuner.csv
BIN=$BASE/bin/all_reduce_perf
LIBDIR=$BASE/bin
PRELOAD=$BASE/bin/librccl.so

mkdir -p "$LOGDIR"
[ -f "$CSV" ] || echo "run_id,utc_start,utc_end,duration_s,scale,variant,rep,slurm_jobid,nodelist,conf_md5,plugin_loaded,rc,logfile" > "$CSV"

MD5_3RULE=$(md5sum "$CONF_3RULE" | awk '{print $1}')
MD5_NEW=$(md5sum "$CONF_NEW" | awk '{print $1}')

md5_for() { case "$1" in default) echo "-";; 3rule) echo "$MD5_3RULE";; new) echo "$MD5_NEW";; esac; }
range_for() { case "$1" in A) echo "-b 64K -e 512K";; B) echo "-b 1M -e 512M";; esac; }
env_for() {
  case "$1" in
    default) echo "" ;;
    3rule) echo "NCCL_TUNER_PLUGIN=$PLUGIN NCCL_TUNER_CONFIG_FILE=$CONF_3RULE" ;;
    new) echo "NCCL_TUNER_PLUGIN=$PLUGIN NCCL_TUNER_CONFIG_FILE=$CONF_NEW" ;;
  esac
}
epoch() { date -u -d "$1" +%s 2>/dev/null; }

append_csv() {
  # run_id,utc_start,utc_end,duration_s,scale,variant,rep,slurm_jobid,nodelist,conf_md5,plugin_loaded,rc,logfile
  echo "$1,$2,$3,$4,$5,$6,$7,$8,\"$9\",${10},${11},${12},${13}" >> "$CSV"
}

# one timed invocation w/ retry. Args: scale variant rep split jid nodelist
run_one() {
  local scale=$1 variant=$2 rep=$3 split=$4 jid=$5 nodelist=$6
  local range envs md5 attempt=1 max_attempts=3 success=0
  range=$(range_for "$split"); envs=$(env_for "$variant"); md5=$(md5_for "$variant")
  while [ $attempt -le $max_attempts ]; do
    local suffix=""; [ $attempt -gt 1 ] && suffix="_try${attempt}"
    local run_id="ab2_${scale}n_${variant}_rep${rep}_${split}${suffix}"
    local logfile="$LOGDIR/${run_id}.log"
    local start end dur rc rows plugin_loaded
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    timeout 240 srun --jobid=$jid -N${scale} --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export LD_LIBRARY_PATH=$LIBDIR LD_PRELOAD=$PRELOAD NCCL_DEBUG=VERSION $envs; $BIN -g 8 -f 2 -n 5 -w 5 -c 1 -M 1 -R 1 $range" \
      > "$logfile" 2>&1
    rc=$?
    end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur=$(( $(epoch "$end") - $(epoch "$start") ))
    rows=$(grep -cE '^ *[0-9]+ +[0-9]+ +float' "$logfile" 2>/dev/null)
    if grep -q "Using DN-TUNER\|Successfully loaded external tuner" "$logfile" 2>/dev/null; then
      plugin_loaded="yes"
    else
      plugin_loaded="no"
    fi
    append_csv "$run_id" "$start" "$end" "$dur" "${scale}n" "$variant" "$rep" "$jid" "$nodelist" "$md5" "$plugin_loaded" "$rc" "$logfile"
    if [ "$rc" -eq 0 ] && [ "$rows" -gt 0 ]; then
      success=1
      echo "OK   $run_id rows=$rows dur=${dur}s plugin_loaded=$plugin_loaded"
      break
    else
      echo "FAIL $run_id rc=$rc rows=$rows attempt=$attempt/$max_attempts"
    fi
    attempt=$((attempt+1))
  done
  [ $success -eq 0 ] && echo "GIVEUP $run_id after $max_attempts attempts"
}

# one-off, non-timed diagnostic to confirm plugin_loaded (NCCL_DEBUG=INFO here is
# deliberately NOT used on the timed reps above -- see measurement hygiene rule).
# Args: scale variant jid nodelist
run_pluginchk() {
  local scale=$1 variant=$2 jid=$3 nodelist=$4
  local envs md5 attempt=1 max_attempts=3 success=0
  envs=$(env_for "$variant"); md5=$(md5_for "$variant")
  while [ $attempt -le $max_attempts ]; do
    local suffix=""; [ $attempt -gt 1 ] && suffix="_try${attempt}"
    local run_id="ab2_${scale}n_pluginchk_${variant}${suffix}"
    local logfile="$LOGDIR/${run_id}.log"
    local start end dur rc rows plugin_loaded
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    timeout 120 srun --jobid=$jid -N${scale} --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL \
      bash -c "export LD_LIBRARY_PATH=$LIBDIR LD_PRELOAD=$PRELOAD NCCL_DEBUG=INFO $envs; $BIN -g 8 -f 2 -n 1 -w 0 -c 1 -M 1 -R 1 -b 64K -e 128K" \
      > "$logfile" 2>&1
    rc=$?
    end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    dur=$(( $(epoch "$end") - $(epoch "$start") ))
    rows=$(grep -cE '^ *[0-9]+ +[0-9]+ +float' "$logfile" 2>/dev/null)
    if grep -q "Using DN-TUNER\|Successfully loaded external tuner" "$logfile" 2>/dev/null; then
      plugin_loaded="yes"
    else
      plugin_loaded="no"
    fi
    append_csv "$run_id" "$start" "$end" "$dur" "${scale}n" "$variant" "0" "$jid" "$nodelist" "$md5" "$plugin_loaded" "$rc" "$logfile"
    if [ "$rc" -eq 0 ] && [ "$rows" -gt 0 ]; then
      success=1
      echo "OK   $run_id plugin_loaded=$plugin_loaded"
      break
    else
      echo "FAIL $run_id rc=$rc rows=$rows attempt=$attempt/$max_attempts"
    fi
    attempt=$((attempt+1))
  done
  [ $success -eq 0 ] && echo "GIVEUP $run_id after $max_attempts attempts"
}

SCALES_TO_RUN="${SCALES_TO_RUN:-1 2 3}"

for scale in $SCALES_TO_RUN; do
  echo "==== SCALE ${scale}n: requesting allocation ===="
  ALLOC_OUT=$(salloc --no-shell -p XAI -N${scale} --exclude=amd-mi355x-2,amd-mi355x-9 --gres=gpu:8 -t 60 -J ylerman 2>&1)
  echo "$ALLOC_OUT"
  JID=$(echo "$ALLOC_OUT" | grep -oE "Granted job allocation [0-9]+" | grep -oE "[0-9]+")
  if [ -z "$JID" ]; then
    echo "ALLOC FAILED for scale=${scale}n"
    continue
  fi
  sleep 3
  NODELIST=$(squeue -j $JID -h -o "%N")
  echo "ALLOC OK scale=${scale}n JID=$JID nodelist=$NODELIST"

  run_pluginchk $scale 3rule $JID "$NODELIST"
  run_pluginchk $scale new $JID "$NODELIST"

  for rep in 1 2 3; do
    for variant in default 3rule new; do
      for split in A B; do
        run_one $scale $variant $rep $split $JID "$NODELIST"
      done
    done
  done

  scancel $JID
  echo "==== SCALE ${scale}n: released JID=$JID ===="
done
echo "ALL_SCALES_DONE"
