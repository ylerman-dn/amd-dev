#!/bin/bash
# env224 probe — node 8, 2026-09-06. Does a comm BUILT with more channels beat the image's
# 112 floor anywhere on 4K-2G? Env levels {112 (image default), 168, 224} x MSCCL {0, on},
# 8 MPI procs in container (deployment-faithful), 5 interleaved reps, no tuner plugin.
# Restart-safe: rep skipped if stdout already has "Avg bus bandwidth".
set -u
OUT=/data/ylerman/env224-2026-09-06
SHARED_BIN=/opt/shared/ylerman/GPU-107/bin
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
ARGS="-b 4K -e 2G -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"

run() { # $1=tag $2=nch(112|168|224) $3=msccl(0|1)
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  x="-x RCCL_MSCCL_ENABLE=$3"
  if [ "$2" != 112 ]; then x="$x -x NCCL_MIN_NCHANNELS=$2 -x NCCL_MAX_NCHANNELS=$2"; fi
  # nch=112: rely on the image's own NCCL_MIN_NCHANNELS=112 (true baseline, no extra env)
  start=$(date +%s)
  timeout 900 docker run --rm --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri \
    -v "$SHARED_BIN":/opt/rccl-tests:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      $x /opt/rccl-tests/all_reduce_perf $ARGS -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}

echo "$(date -u +%FT%TZ) ENV224 START" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  for nch in 112 168 224; do
    for ms in 0 1; do
      run "e${nch}_m${ms}_r${rep}" "$nch" "$ms"
    done
  done
  echo "$(date -u +%FT%TZ) REP $rep COMPLETE" >> "$OUT/progress.log"
done
echo "$(date -u +%FT%TZ) ENV224 DONE" >> "$OUT/progress.log"
