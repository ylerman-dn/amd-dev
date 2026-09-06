#!/bin/bash
# collsweep — node 4, 2026-09-06. Fresh per-collective candidate sweep in the FINAL env
# (container stock RCCL, 8-proc, MSCCL=0, NCCL_MIN_NCHANNELS removed): reduce_scatter,
# all_gather, broadcast. Arms x 5 interleaved reps, 4K-512M:
#   van    : vanilla defaults (flag off)          — baseline
#   ch112  : env NCCL_MIN/MAX_NCHANNELS=112       — "112 flat" candidate
#   r_ll, r_simple, t_ll : forced valid combos (vanilla channels)
set -u
OUT=/data/ylerman/collsweep-2026-09-06
BIN=/opt/shared/ylerman/GPU-107/bin
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
ARGS="-b 4K -e 512M -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"
unset NCCL_MIN_NCHANNELS
run() { # $1=tag $2=binary $3=extra -x
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  start=$(date +%s)
  timeout 900 docker run --rm --name="collsweep_$1" --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri -e NCCL_MIN_NCHANNELS \
    -v "$BIN":/opt/rccl-tests:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      -x RCCL_MSCCL_ENABLE=0 $3 /opt/rccl-tests/$2 $ARGS -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}
echo "$(date -u +%FT%TZ) COLLSWEEP START" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  for c in reduce_scatter all_gather broadcast; do
    run "${c}_van_r${rep}"      "${c}_perf" ""
    run "${c}_ch112_r${rep}"    "${c}_perf" "-x NCCL_MIN_NCHANNELS=112 -x NCCL_MAX_NCHANNELS=112"
    run "${c}_r_ll_r${rep}"     "${c}_perf" "-x NCCL_ALGO=Ring -x NCCL_PROTO=LL"
    run "${c}_r_simple_r${rep}" "${c}_perf" "-x NCCL_ALGO=Ring -x NCCL_PROTO=Simple"
    run "${c}_t_ll_r${rep}"     "${c}_perf" "-x NCCL_ALGO=Tree -x NCCL_PROTO=LL"
  done
  echo "$(date -u +%FT%TZ) REP $rep COMPLETE" >> "$OUT/progress.log"
done
echo "$(date -u +%FT%TZ) COLLSWEEP DONE" >> "$OUT/progress.log"
