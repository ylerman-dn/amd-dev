#!/bin/bash
# Paired in-model A/B on ONE live SGLang server, using the hot-reload tuner plugin
# (dn/dn-tuner, DN_TUNER_HOT_RELOAD=1): the driver swaps the rule file between benchmark
# reps, round-robin across arms, so every arm samples the same minutes — time drift cancels
# by construction and no server restarts are paid.
#
# Usage: infer_paired.sh <model-path> <prefix> <rounds> <arm1:conf1> <arm2:conf2> ...
#   arm spec  name:<conf filename in IM_CONF_DIR>   (tuner rules for that arm)
#             name:NONE                             (empty rule set — "no rule" reference*)
#   Each round runs ONE rep per arm; total reps per arm = <rounds>.
#
# *Note: with a shared live server the plugin stays loaded for every arm, so "NONE" means
#  "plugin loaded, zero rules" — not "no plugin". The quiet campaigns showed plugin presence
#  without matching rules costs nothing measurable on covered models, but state this in any
#  writeup. For an absolute no-plugin reference, run a separate def arm via infer_many.sh.
#
# Env overrides: IM_BASE, IM_CONF_DIR, IM_IMAGE, RM_SUBSYS (default INIT,ENV — quiet; use
# INIT,TUNING,ENV only for a 1-round detect/canary run), IM_PLUGIN (default the hot-reload
# build), ISL/OSL via IM_ISL/IM_OSL (default 512/512 — the frozen campaign params).
set -u
MODEL=$1
PFX=$2
ROUNDS=$3
shift 3
ARMS="$@"

CONF_DIR=${IM_CONF_DIR:-/opt/shared/ylerman/GPU-107/infer-2026-08-30}
B=${IM_BASE:-/data/ylerman/models-2026-08-30}
IMAGE=${IM_IMAGE:-lmsysorg/sglang:v0.5.17-rocm720-mi35x}
PLUGIN=${IM_PLUGIN:-librccl-tunerv4-dn-hotreload.so}
PORT=8899
CNAME=ylerman-paired-${PFX}
LIVE=live_${PFX}.conf
L=$B/paired_${PFX}.log
ISL=${IM_ISL:-512}
OSL=${IM_OSL:-512}
CONC=32
NUM_PROMPTS=128
HDR="collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff"

mkdir -p "$B"
for spec in $ARMS; do mkdir -p "$B/${PFX}_${spec%%:*}/logs"; done
echo "$HDR" > "$CONF_DIR/$LIVE"   # start empty; first swap sets the real rules

docker rm -f "$CNAME" >/dev/null 2>&1
docker run -d --ipc=host --shm-size=16g --network=host --name="$CNAME" \
  --privileged --ulimit memlock=-1 \
  --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  -v /data:/data:ro \
  -v /huggingface:/huggingface:ro \
  -v "$CONF_DIR":/opt/rccl/tuner:ro \
  -v "$B/${PFX}_srv":/workspace/out -w /workspace \
  -e RCCL_MSCCL_ENABLE=0 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=${RM_SUBSYS:-INIT,ENV} \
  -e NCCL_DEBUG_FILE=/workspace/out/logs/rccl.%h.%p.log \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_TUNER_PLUGIN=/opt/rccl/tuner/$PLUGIN \
  -e NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/$LIVE \
  -e DN_TUNER_HOT_RELOAD=1 \
  --entrypoint=/bin/bash \
  "$IMAGE" -c "
mkdir -p /workspace/out/logs
python3 -m sglang.launch_server --model-path $MODEL --tp 8 \
  --host 0.0.0.0 --port $PORT --disable-cuda-graph --disable-custom-all-reduce \
  --trust-remote-code ${IM_EXTRA:-} > /workspace/out/server.log 2>&1
" > /dev/null

mkdir -p "$B/${PFX}_srv/logs"
echo "=== paired $PFX: server starting $(date -u +%F' '%H:%M), arms: $ARMS ===" | tee -a "$L"
up=0
for i in $(seq 1 180); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
  sleep 10
done
[ "$up" = 1 ] || { echo "SERVER FAILED" | tee -a "$L"; docker logs "$CNAME" 2>&1 | tail -20 >> "$L"; docker rm -f "$CNAME"; exit 1; }
echo "server up; $ROUNDS rounds round-robin" | tee -a "$L"

for r in $(seq 1 "$ROUNDS"); do
  for spec in $ARMS; do
    name=${spec%%:*}; conf=${spec#*:}
    if [ -e "$B/${PFX}_${name}/bench_round${r}.log" ]; then continue; fi   # restart-safe
    if [ "$conf" = "NONE" ]; then echo "$HDR" > "$CONF_DIR/$LIVE"
    else cat "$CONF_DIR/$conf" > "$CONF_DIR/$LIVE"; fi
    sleep 2   # mtime tick + plugin's 1s reload throttle
    docker exec "$CNAME" bash -c "
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --random-range-ratio 1 --max-concurrency $CONC --num-prompts $NUM_PROMPTS \
" > "$B/${PFX}_${name}/bench_round${r}.log" 2>&1
    tps=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' "$B/${PFX}_${name}/bench_round${r}.log")
    echo "[$PFX/$name] round${r} tok/s=${tps:-FAIL}" | tee -a "$L"
  done
done

hits=$(grep -h -c "hot-reloaded" "$B/${PFX}_srv/logs/"*.log 2>/dev/null | paste -sd+ | bc)
echo "[$PFX] hot-reload events per logs: ${hits:-0}" | tee -a "$L"
docker rm -f "$CNAME" >/dev/null 2>&1
echo "=== paired $PFX DONE $(date -u +%F' '%H:%M) ===" | tee -a "$L"
