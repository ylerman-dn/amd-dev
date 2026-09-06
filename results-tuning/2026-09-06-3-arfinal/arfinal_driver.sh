#!/bin/bash
# arfinal — node 8, 2026-09-06. Close the all_reduce 1-node direction on the deployment
# stack: (a) full algo/proto matrix via env (never done on container stock at all sizes),
# (b) MSCCL-on with a LOWER channel floor (env224 only tested >=112).
# Arms x 5 interleaved reps, all_reduce 4K-512M, 8-proc container:
#   ref_m0, ref_m1                      : stock defaults, MSCCL off/on
#   {ring,tree}x{ll,simple,ll128}, m0   : NCCL_ALGO/NCCL_PROTO forced, channels default
#   f28_m1, f56_m1                      : MSCCL on + NCCL_MIN/MAX_NCHANNELS {28,56}
set -u
OUT=/data/ylerman/arfinal-2026-09-06
BIN=/opt/shared/ylerman/GPU-107/bin
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
ARGS="-b 4K -e 512M -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"

run() { # $1=tag $2=extra -x args $3=msccl
  if [ -f "$OUT/${1}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${1}_stdout.log"; then return; fi
  rm -f "$OUT/${1}_"*.log
  start=$(date +%s)
  timeout 900 docker run --rm --name="arfinal_$1" --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
    --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri \
    -v "$BIN":/opt/rccl-tests:ro -v "$OUT":/workspace/out -w /workspace \
    --entrypoint=/bin/bash "$IMG" -c "mpirun --allow-run-as-root -np 8 \
      -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV -x NCCL_DEBUG_FILE=/workspace/out/${1}_%p.log \
      -x RCCL_MSCCL_ENABLE=$3 $2 /opt/rccl-tests/all_reduce_perf $ARGS -g 1" \
    > "$OUT/${1}_stdout.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}

echo "$(date -u +%FT%TZ) ARFINAL START" >> "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  run "ref_m0_r${rep}" "" 0
  run "ref_m1_r${rep}" "" 1
  for a in Ring Tree; do
    for p in LL Simple LL128; do
      run "$(echo ${a}_${p} | tr 'A-Z' 'a-z')_m0_r${rep}" "-x NCCL_ALGO=$a -x NCCL_PROTO=$p" 0
    done
  done
  run "f28_m1_r${rep}" "-x NCCL_MIN_NCHANNELS=28 -x NCCL_MAX_NCHANNELS=28" 1
  run "f56_m1_r${rep}" "-x NCCL_MIN_NCHANNELS=56 -x NCCL_MAX_NCHANNELS=56" 1
  echo "$(date -u +%FT%TZ) REP $rep COMPLETE" >> "$OUT/progress.log"
done
echo "$(date -u +%FT%TZ) ARFINAL DONE" >> "$OUT/progress.log"
