#!/bin/bash
# collmap — node 8, 2026-09-06. Extend the MSCCL/tuner boundary map beyond all_reduce.
# Phase 1 canaries: MSCCL ON + tuner plugin with empty conf (consultation counter),
#   1 rep per collective (reduce_scatter, all_gather, broadcast, alltoall), 4K-512M.
#   Read-out: consultations per size ("No matching config found" in TUNING logs).
# Phase 2 perf A/B: MSCCL {0, on} x 5 interleaved reps, NO plugin,
#   reduce_scatter_perf + all_gather_perf, 4K-512M.
set -u
OUT=/data/ylerman/collmap-2026-09-06
BIN=/opt/shared/ylerman/GPU-107/bin
TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
ARGS="-b 4K -e 512M -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"

run() { # $1=tag $2=binary $3=msccl(0|1) $4=plugin(0|1)
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  x="-x RCCL_MSCCL_ENABLE=$3"
  [ "$4" = 1 ] && x="$x -x NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -x NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/sw_empty.conf"
  start=$(date +%s)
  timeout 900 docker run --rm --name="collmap_$1" --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri \
    -v "$BIN":/opt/rccl-tests:ro -v "$TUN":/opt/rccl/tuner:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      $x /opt/rccl-tests/$2 $ARGS -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}

echo "$(date -u +%FT%TZ) COLLMAP START" >> "$OUT/progress.log"
# phase 1: boundary canaries (MSCCL on + empty-conf plugin)
for c in reduce_scatter all_gather broadcast alltoall; do
  run "can_${c}" "${c}_perf" 1 1
done
echo "$(date -u +%FT%TZ) CANARIES DONE" >> "$OUT/progress.log"
# phase 2: MSCCL on/off perf, no plugin
for rep in 1 2 3 4 5; do
  for c in reduce_scatter all_gather; do
    for m in 0 1; do run "${c}_m${m}_r${rep}" "${c}_perf" "$m" 0; done
  done
done
echo "$(date -u +%FT%TZ) COLLMAP DONE" >> "$OUT/progress.log"
