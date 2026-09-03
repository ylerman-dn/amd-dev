#!/bin/bash
# regime program — des2-2 (2026-09-03 evening). rccl-tests in the DEPLOYMENT-FAITHFUL harness:
# container stock RCCL + 8 MPI processes (mpirun inside the container), matching SGLang's
# 8-worker topology. Phases:
#  A sanity (5 reps x 2 cells, 4K-32M): r8_m0 (MSCCL=0) and r8_mon (MSCCL on) — compare vs
#    bare-metal 8-proc (stackcmp cell A), old site rows, and 1-proc container cells.
#  B bracket rerun (11 arms x 5 reps, 4K-512M, MSCCL=0): default + arm1..5 + big1..5
#    (sw_bk_*.conf) — re-ask "does anything beat 112" in the correct regime.
#  C canary (1 run, MSCCL on + arm3 conf, TUNING logs): which sizes still consult the tuner
#    with MSCCL enabled.
set -u
OUT=/data/ylerman/regime-2026-09-03
SHARED_BIN=/opt/shared/ylerman/GPU-107/bin
SHARED_TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
mkdir -p "$OUT"

run() { # $1=tag $2=msccl(0|1) $3=conf-or-NONE $4=args
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

SMALL="-b 4K -e 32M -f 2 -n 20 -w 5 -c 1 -A 1"
FULL="-b 4K -e 512M -f 2 -n 20 -w 5 -c 1 -A 1"

echo "$(date -u +%FT%TZ) REGIME START" >> "$OUT/progress.log"
# Phase A
for rep in 1 2 3 4 5; do
  run "r8m0_r${rep}"  0 NONE "$SMALL"
  run "r8mon_r${rep}" 1 NONE "$SMALL"
done
echo "$(date -u +%FT%TZ) PHASE A COMPLETE" >> "$OUT/progress.log"
# Phase B
ARMS="default arm1 arm2 arm3 arm4 arm5 big1 big2 big3 big4 big5"
for rep in 1 2 3 4 5; do
  for arm in $ARMS; do
    conf=NONE; [ "$arm" != default ] && conf="sw_bk_${arm}.conf"
    run "bk8_${arm}_r${rep}" 0 "$conf" "$FULL"
  done
  echo "$(date -u +%FT%TZ) PHASE B REP $rep COMPLETE" >> "$OUT/progress.log"
done
# Phase C
run "canary_mon_arm3" 1 "sw_bk_arm3.conf" "$FULL"
echo "$(date -u +%FT%TZ) REGIME DONE" >> "$OUT/progress.log"
