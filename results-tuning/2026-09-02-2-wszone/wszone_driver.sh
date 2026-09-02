#!/bin/bash
# wszone sweep driver — 1-node all_reduce >=64MiB (WarpSpeed zone) candidate sweep
# 31 arms (default + 3 combos x 10 channel values) x 5 reps, interleaved round-robin.
# Restart-safe: a rep whose stdout already shows "Avg bus bandwidth" is skipped.
# Touch $OUT/STOP_AT_3REPS to end the sweep after round 3.
set -u
OUT=/data/ylerman/wszone-2026-09-02
SHARED_BIN=/opt/shared/ylerman/GPU-107/bin
SHARED_TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x

ARMS="default"
for combo in ring_simple ring_ll tree_ll; do
  for ch in 8 16 32 56 112 168 224 256 280 320; do
    ARMS="$ARMS ${combo}_${ch}"
  done
done

mkdir -p "$OUT"

# wait for image (pull may still be in flight)
until docker image inspect "$IMG" >/dev/null 2>&1; do
  echo "$(date -u +%FT%TZ) waiting for image pull" >> "$OUT/progress.log"
  sleep 30
done

echo "$(date -u +%FT%TZ) SWEEP START arms=$(echo $ARMS | wc -w) reps=5" >> "$OUT/progress.log"

for rep in 1 2 3 4 5; do
  if [ -f "$OUT/STOP_AT_3REPS" ] && [ "$rep" -gt 3 ]; then
    echo "$(date -u +%FT%TZ) STOP_AT_3REPS present, ending after 3 reps" >> "$OUT/progress.log"
    break
  fi
  for arm in $ARMS; do
    tag="${arm}_r${rep}"
    if [ -f "$OUT/${tag}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${tag}_stdout.log"; then
      continue
    fi
    rm -f "$OUT/${tag}_"*.log "$OUT/${tag}_"*.log.gz
    envargs=""
    if [ "$arm" != "default" ]; then
      envargs="-e NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -e NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/sw_ws_${arm}.conf"
    fi
    start=$(date +%s)
    timeout 900 docker run --rm --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
      --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --device=/dev/kfd --device=/dev/dri \
      -v "$SHARED_BIN":/opt/rccl-tests:ro -v "$SHARED_TUN":/opt/rccl/tuner:ro -v "$OUT":/workspace/out -w /workspace \
      -e RCCL_MSCCL_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV \
      -e NCCL_DEBUG_FILE=/workspace/out/${tag}_%p.log \
      $envargs \
      --entrypoint=/bin/bash "$IMG" -c "/opt/rccl-tests/all_reduce_perf -b 64M -e 512M -f 2 -g 8 -n 20 -w 5 -c 1 -A 1" \
      > "$OUT/${tag}_stdout.log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) $tag rc=$rc dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
  done
  echo "$(date -u +%FT%TZ) REP $rep COMPLETE" >> "$OUT/progress.log"
done
echo "$(date -u +%FT%TZ) SWEEP DONE" >> "$OUT/progress.log"
