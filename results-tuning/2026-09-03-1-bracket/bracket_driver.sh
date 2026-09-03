#!/bin/bash
# bracket sweep driver — 1-node all_reduce, bracket the default channel count per size.
# Arms: default (no plugin), arm1..arm5 (per-size x{0.5,0.75,1,1.25,1.5} ladder around each
# size's old bare-metal default channels; >=64M sizes carry the around-112 ladder), and
# big1..big5 (>=64M only, around-224 ladder {192,208,224,240,252}; other sizes no rule).
# 11 arms x 5 reps, interleaved round-robin. Restart-safe: a rep whose stdout already
# shows "Avg bus bandwidth" is skipped. Slurm hold: jobid 20983 (ylerman-bracket).
set -u
OUT=/data/ylerman/bracket-2026-09-03
SHARED_BIN=/opt/shared/ylerman/GPU-107/bin
SHARED_TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x

ARMS="default arm1 arm2 arm3 arm4 arm5 big1 big2 big3 big4 big5"
mkdir -p "$OUT"
echo "$(date -u +%FT%TZ) BRACKET SWEEP START arms=$(echo $ARMS | wc -w) reps=5" >> "$OUT/progress.log"

for rep in 1 2 3 4 5; do
  for arm in $ARMS; do
    tag="${arm}_r${rep}"
    if [ -f "$OUT/${tag}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${tag}_stdout.log"; then
      continue
    fi
    rm -f "$OUT/${tag}_"*.log "$OUT/${tag}_"*.log.gz
    envargs=""
    if [ "$arm" != "default" ]; then
      envargs="-e NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -e NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/sw_bk_${arm}.conf"
    fi
    start=$(date +%s)
    timeout 1200 docker run --rm --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
      --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      --device=/dev/kfd --device=/dev/dri \
      -v "$SHARED_BIN":/opt/rccl-tests:ro -v "$SHARED_TUN":/opt/rccl/tuner:ro -v "$OUT":/workspace/out -w /workspace \
      -e RCCL_MSCCL_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV \
      -e NCCL_DEBUG_FILE=/workspace/out/${tag}_%p.log \
      $envargs \
      --entrypoint=/bin/bash "$IMG" -c "/opt/rccl-tests/all_reduce_perf -b 4K -e 512M -f 2 -g 8 -n 20 -w 5 -c 1 -A 1" \
      > "$OUT/${tag}_stdout.log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) $tag rc=$rc dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
  done
  echo "$(date -u +%FT%TZ) REP $rep COMPLETE" >> "$OUT/progress.log"
done
echo "$(date -u +%FT%TZ) BRACKET SWEEP DONE" >> "$OUT/progress.log"
