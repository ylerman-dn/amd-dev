#!/bin/bash
# mid-ladder + MSCCL crossover, des2-2, 8-proc container harness (2026-09-03 evening).
# [d] does any channel value 32..128 beat the 112 floor at 64K-8M? 8 arms x 5 reps, MSCCL=0.
# [e] MSCCL on/off crossover, 8M-32M in 4M steps, 5 reps each.
set -u
OUT=/data/ylerman/regime-2026-09-03
SHARED_BIN=/opt/shared/ylerman/GPU-107/bin
SHARED_TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
run() { # tag msccl conf-or-NONE args
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  x="-x RCCL_MSCCL_ENABLE=$2"
  [ "$3" != NONE ] && x="$x -x NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -x NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/$3"
  start=$(date +%s)
  timeout 900 docker run --rm --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri \
    -v "$SHARED_BIN":/opt/rccl-tests:ro -v "$SHARED_TUN":/opt/rccl/tuner:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      $x /opt/rccl-tests/all_reduce_perf $4 -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}
ML="-b 64K -e 8M -f 2 -n 20 -w 5 -c 1 -A 1"
XS="-b 8M -e 32M -i 4194304 -n 20 -w 5 -c 1 -A 1"
echo "$(date -u +%FT%TZ) MIDLADDER START" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  run "ml_default_r${rep}" 0 NONE "$ML"
  for ch in 32 48 64 84 96 112 128; do run "ml_${ch}_r${rep}" 0 "sw_ml_${ch}.conf" "$ML"; done
done
echo "$(date -u +%FT%TZ) MIDLADDER DONE" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  run "xs_m0_r${rep}" 0 NONE "$XS"
  run "xs_mon_r${rep}" 1 NONE "$XS"
done
echo "$(date -u +%FT%TZ) CROSSOVER DONE" >> "$OUT/progress.log"
