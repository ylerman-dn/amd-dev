#!/bin/bash
# N-repeat inference A/B for one model. Starts the SGLang server ONCE per arm and loops the
# benchmark against it, so server startup is paid twice rather than 2N times.
#
# MSCCL is disabled (RCCL_MSCCL_ENABLE=0): with it on, MSCCL executes the model's all_reduces
# itself and never consults the tuner, so no rule can fire. See SUMMARY.md section 5.
# Custom all-reduce is disabled so the all_reduce reaches RCCL at all (section 1).
# Uses the container's stock RCCL: our build hangs under SGLang (section 2).
#
# Optional env overrides (2026-08-31, ablation/quiet experiments — defaults keep old behavior):
#   RM_SUBSYS  NCCL_DEBUG_SUBSYS value        (default INIT,TUNING,ENV)
#   RM_CONF    tuner conf filename in CONF_DIR (default all_reduce_1n.final.conf)
#   RM_MSCCL   RCCL_MSCCL_ENABLE value         (default 0; set 1 for deployment-default MSCCL)
#
# Usage: run_many.sh <model-path> <short-name> <arm: def|tun> <reps> [extra sglang args...]
set -u
MODEL=$1
NAME=$2
ARM=$3
REPS=$4
shift 4
EXTRA="$*"

CONF_DIR=${IM_CONF_DIR:-/opt/shared/ylerman/GPU-107/infer-2026-08-30}
OUT=${IM_BASE:-/data/ylerman/models-2026-08-30}/${NAME}_${ARM}
IMAGE=${IM_IMAGE:-lmsysorg/sglang:v0.5.17-rocm720-mi35x}
PORT=8899
CNAME=ylerman-many-${NAME}-${ARM}

ISL=${IM_ISL:-512}
OSL=${IM_OSL:-512}
CONC=32
NUM_PROMPTS=128

mkdir -p "$OUT/logs"
rm -rf "${OUT:?}/logs"/* "$OUT"/bench_rep*.log

# IM_KEEP_CUSTOM_AR=1 keeps SGLang's custom all-reduce kernel (RCCL then carries ~0 of the
# TP all_reduce - stock behavior). Default: disabled, so collectives reach RCCL/the tuner.
DISABLE_AR_FLAG="--disable-custom-all-reduce"
[ -n "${IM_KEEP_CUSTOM_AR:-}" ] && DISABLE_AR_FLAG=""

TUNER_ENV=""
if [ "$ARM" = "tun" ]; then
  TUNER_ENV="-e NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -e NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/${RM_CONF:-all_reduce_1n.final.conf}"
fi

docker rm -f "$CNAME" >/dev/null 2>&1

docker run -d --ipc=host --shm-size=16g --network=host --name="$CNAME" \
  --privileged --ulimit memlock=-1 \
  --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  -v /data:/data:ro \
  -v /huggingface:/huggingface:ro \
  -v "$CONF_DIR":/opt/rccl/tuner:ro \
  -v "$OUT":/workspace/out -w /workspace \
  -e RCCL_MSCCL_ENABLE=${RM_MSCCL:-0} \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=${RM_SUBSYS:-INIT,TUNING,ENV} \
  -e NCCL_DEBUG_FILE=/workspace/out/logs/rccl.%h.%p.log \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $TUNER_ENV \
  --entrypoint=/bin/bash \
  "$IMAGE" -c "
python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tp 8 \
  --host 0.0.0.0 --port $PORT \
  --disable-cuda-graph \
  $DISABLE_AR_FLAG \
  --trust-remote-code $EXTRA \
  > /workspace/out/server.log 2>&1
" > /dev/null

echo "[$NAME/$ARM] server starting..."
up=0
for i in $(seq 1 180); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  sleep 10
done
if [ "$up" != 1 ]; then
  echo "[$NAME/$ARM] SERVER FAILED"
  docker logs "$CNAME" 2>&1 | tail -25
  docker rm -f "$CNAME" >/dev/null 2>&1
  exit 1
fi
echo "[$NAME/$ARM] server up after ${i}0s; running $REPS repeats"

for r in $(seq 1 "$REPS"); do
  docker exec "$CNAME" bash -c "
python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port $PORT --dataset-name random \
  --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1 \
  --max-concurrency $CONC --num-prompts $NUM_PROMPTS \
" > "$OUT/bench_rep${r}.log" 2>&1
  tps=$(grep -oP 'Output token throughput \(tok/s\):\s+\K[0-9.]+' "$OUT/bench_rep${r}.log")
  tpot=$(grep -oP 'Mean TPOT \(ms\):\s+\K[0-9.]+' "$OUT/bench_rep${r}.log")
  echo "[$NAME/$ARM] rep${r} tok/s=${tps:-FAIL} TPOT=${tpot:-—}"
done

hits=$(grep -h -c "Applied config" "$OUT"/logs/*.log 2>/dev/null | paste -sd+ | bc)
echo "[$NAME/$ARM] tuner_hits=${hits:-0}"
docker rm -f "$CNAME" >/dev/null 2>&1
echo "[$NAME/$ARM] done -> $OUT"
