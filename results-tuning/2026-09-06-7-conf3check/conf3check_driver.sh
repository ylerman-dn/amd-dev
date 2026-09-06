#!/bin/bash
# conf3check — node 4: acceptance for gpu107_1n_v3.conf across 4 collectives.
# 2 arms (v_none / v_conf3) x 4 collectives x 5 interleaved reps, flag off, MSCCL=0.
# all_gather is intentionally uncovered by the conf (its Direct algo is not expressible
# by the tuner) — its arm-pair checks fallthrough does no harm.
set -u
OUT=/data/ylerman/conf3check-2026-09-06
BIN=/opt/shared/ylerman/GPU-107/bin
TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
ARGS="-b 4K -e 512M -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"
unset NCCL_MIN_NCHANNELS
run() { # tag binary plugin(0|1)
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  x="-x RCCL_MSCCL_ENABLE=0"
  [ "$3" = 1 ] && x="$x -x NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -x NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/gpu107_1n_v3.conf"
  start=$(date +%s)
  timeout 900 docker run --rm --name="conf3_$1" --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri -e NCCL_MIN_NCHANNELS \
    -v "$BIN":/opt/rccl-tests:ro -v "$TUN":/opt/rccl/tuner:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      $x /opt/rccl-tests/$2 $ARGS -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}
echo "$(date -u +%FT%TZ) CONF3CHECK START" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  for c in all_reduce broadcast reduce_scatter all_gather; do
    run "${c}_none_r${rep}" "${c}_perf" 0
    run "${c}_conf_r${rep}" "${c}_perf" 1
  done
done
echo "$(date -u +%FT%TZ) CONF3CHECK DONE" >> "$OUT/progress.log"
