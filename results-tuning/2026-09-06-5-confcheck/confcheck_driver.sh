#!/bin/bash
# confcheck — node 4, 2026-09-06: acceptance for gpu107_1n_v2.conf (flag-off world).
# 3 arms x 5 reps, all_reduce 4K-512M, 8-proc container, MSCCL=0:
#   v_none : flag removed, no plugin       (the new baseline)
#   f112   : image as-is                    (what SGLang ships)
#   v_conf : flag removed + plugin + gpu107_1n_v2.conf (our full conf)
set -u
OUT=/data/ylerman/confcheck-2026-09-06
BIN=/opt/shared/ylerman/GPU-107/bin
TUN=/opt/shared/ylerman/GPU-107/infer-2026-08-30
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
ARGS="-b 4K -e 512M -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"
unset NCCL_MIN_NCHANNELS
run() { # tag dropfloor plugin
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  d=""; [ "$2" = 1 ] && d="-e NCCL_MIN_NCHANNELS"
  x="-x RCCL_MSCCL_ENABLE=0"
  [ "$3" = 1 ] && x="$x -x NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so -x NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/gpu107_1n_v2.conf"
  start=$(date +%s)
  timeout 900 docker run --rm --name="confcheck_$1" --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri $d \
    -v "$BIN":/opt/rccl-tests:ro -v "$TUN":/opt/rccl/tuner:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      $x /opt/rccl-tests/all_reduce_perf $ARGS -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}
echo "$(date -u +%FT%TZ) CONFCHECK START" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  run "v_none_r${rep}" 1 0
  run "f112_r${rep}"   0 0
  run "v_conf_r${rep}" 1 1
done
echo "$(date -u +%FT%TZ) CONFCHECK DONE" >> "$OUT/progress.log"
