#!/bin/bash
# GPU-107 single-node inference value test: Llama-3.1-8B, TP=8, node 9.
# Arm "def"  = custom RCCL, no tuner plugin.
# Arm "tun"  = same + DN tuner plugin with the shipped 1-node all_reduce rules.
# 3rd arg (optional) "rccl-ar" adds --disable-custom-all-reduce so the TP all_reduce
# goes through RCCL instead of SGLang's own kernel. Without it SGLang bypasses RCCL
# entirely for TP all_reduce (disable_custom_all_reduce defaults to False) and no
# tuner rule can ever fire.
# Usage: run_infer.sh <arm: def|tun> <tag> [rccl-ar]
set -u
ARM=$1
TAG=$2
AR_MODE=${3:-custom}
# 4th arg "stock" leaves the container's own RCCL in place instead of mounting ours,
# to tell a bug in our build apart from one in the serving stack.
RCCL_MODE=${4:-custom}
MODEL=/data/mlperf_llama31_8b/model
RCCL_DIR=/opt/shared/ylerman/GPU-107/bin
CONF_DIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30
# Output must be node-local: the NFS share is not writable from inside the container.
OUT=/data/ylerman/infer-2026-08-30/$TAG
IMAGE=lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260817
PORT=8899
CNAME=ylerman-sgl-$TAG

# Benchmark shape: decode-heavy so all_reduce lands at hidden(4096) x conc x 2B = 256KB at conc=32.
ISL=512
OSL=512
CONC=32
NUM_PROMPTS=128

mkdir -p "$OUT/logs"
rm -rf "$OUT/logs"/*

RCCL_ENV="-e LD_LIBRARY_PATH=/opt/rccl/lib:/opt/rocm/lib -e SGLANG_NCCL_SO_PATH=/opt/rccl/lib/librccl.so"
if [ "$RCCL_MODE" = "stock" ]; then
  RCCL_ENV=""
fi

AR_FLAG=""
if [ "$AR_MODE" = "rccl-ar" ]; then
  AR_FLAG="--disable-custom-all-reduce"
fi

TUNER_ENV=""
if [ "$ARM" = "tun" ]; then
  TUNER_ENV="-e NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -e NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/all_reduce_1n.final.conf"
fi

docker rm -f "$CNAME" >/dev/null 2>&1

docker run -d --ipc=host --shm-size=16g --network=host --name="$CNAME" \
  --privileged --ulimit memlock=-1 \
  --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  -v "$MODEL":"$MODEL":ro \
  -v "$RCCL_DIR":/opt/rccl/lib:ro \
  -v "$CONF_DIR":/opt/rccl/tuner:ro \
  -v "$OUT":/workspace/out -w /workspace \
  $RCCL_ENV \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV,COLL \
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
  --trust-remote-code \
  $AR_FLAG \
  > /workspace/out/server.log 2>&1
" > /dev/null

echo "[$TAG] server starting, waiting for health..."
for i in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[$TAG] server up after ${i}0s"; break
  fi
  sleep 10
done

if ! curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[$TAG] SERVER FAILED TO START"
  docker logs "$CNAME" 2>&1 | tail -40
  docker rm -f "$CNAME" >/dev/null 2>&1
  exit 1
fi

docker exec "$CNAME" bash -c "
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port $PORT \
  --dataset-name random \
  --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1 \
  --max-concurrency $CONC --num-prompts $NUM_PROMPTS \
  --output-file /workspace/out/bench.jsonl \
" > "$OUT/bench.log" 2>&1
echo "[$TAG] bench rc=$?"

docker rm -f "$CNAME" >/dev/null 2>&1
echo "[$TAG] done. logs in $OUT"
