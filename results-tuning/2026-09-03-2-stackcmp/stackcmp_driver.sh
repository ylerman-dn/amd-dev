#!/bin/bash
# stackcmp: isolate the old-site vs container gap at 4K-32M. 4 cells x 5 reps, interleaved.
#  a_mpi_fork : bare metal, fork librccl (LD_PRELOAD), 8 MPI ranks x -g 1 via srun  = old-site replica
#  b_sp_fork  : bare metal, fork librccl, 1 process x -g 8                          = fork + new launch
#  c_ct_m0    : container stock RCCL, -g 8, RCCL_MSCCL_ENABLE=0                     = bracket-sweep replica
#  d_ct_mon   : container stock RCCL, -g 8, MSCCL left enabled                      = SGLang default env
# Restart-safe: rep skipped if its stdout already has "Avg bus bandwidth".
set -u
OUT=/data/ylerman/stackcmp-2026-09-03
BIN=/opt/shared/ylerman/GPU-107/bin
IMG=lmsysorg/sglang:v0.5.17-rocm720-mi35x
JOBID=${STACKCMP_JOBID:?need STACKCMP_JOBID}
ARGS="-b 4K -e 32M -f 2 -n 20 -w 5 -c 1 -A 1"
mkdir -p "$OUT"
echo "$(date -u +%FT%TZ) STACKCMP START jobid=$JOBID" >> "$OUT/progress.log"

run_cell() { # $1=cell $2=rep
  tag="${1}_r${2}"
  if [ -f "$OUT/${tag}_stdout.log" ] && grep -q "Avg bus bandwidth" "$OUT/${tag}_stdout.log"; then return; fi
  rm -f "$OUT/${tag}_"*.log
  start=$(date +%s)
  case $1 in
    a_mpi_fork)
      # library env copied from tools/rccl-sweep/sweep_executor.py:163-165 (old-site launcher)
      timeout 600 srun --jobid="$JOBID" --mpi=pmix --ntasks=8 bash -c "export LD_LIBRARY_PATH=/usr/local/lib:$BIN/:/opt/rocm/bin LD_PRELOAD=$BIN/librccl.so HSA_NO_SCRATCH_RECLAIM=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV NCCL_DEBUG_FILE=$OUT/${tag}_%p.log; exec $BIN/all_reduce_perf $ARGS -g 1" \
        < /dev/null > "$OUT/${tag}_stdout.log" 2>&1 ;;
    b_sp_fork)
      timeout 600 env LD_LIBRARY_PATH=/usr/local/lib:$BIN/:/opt/rocm/bin LD_PRELOAD=$BIN/librccl.so HSA_NO_SCRATCH_RECLAIM=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV NCCL_DEBUG_FILE=$OUT/${tag}_%p.log \
        $BIN/all_reduce_perf $ARGS -g 8 < /dev/null > "$OUT/${tag}_stdout.log" 2>&1 ;;
    c_ct_m0|d_ct_mon)
      mscclargs=""; [ "$1" = c_ct_m0 ] && mscclargs="-e RCCL_MSCCL_ENABLE=0"
      timeout 600 docker run --rm --ipc=host --shm-size=16g --network=host --privileged --ulimit memlock=-1 \
        --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --device=/dev/kfd --device=/dev/dri \
        -v "$BIN":/opt/rccl-tests:ro -v "$OUT":/workspace/out -w /workspace \
        $mscclargs -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV \
        -e NCCL_DEBUG_FILE=/workspace/out/${tag}_%p.log \
        --entrypoint=/bin/bash "$IMG" -c "/opt/rccl-tests/all_reduce_perf $ARGS -g 8" \
        > "$OUT/${tag}_stdout.log" 2>&1 ;;
  esac
  echo "$(date -u +%FT%TZ) $tag rc=$? dur=$(( $(date +%s) - start ))s" >> "$OUT/progress.log"
}

for rep in 1 2 3 4 5; do
  for cell in a_mpi_fork b_sp_fork c_ct_m0 d_ct_mon; do run_cell "$cell" "$rep"; done
  echo "$(date -u +%FT%TZ) REP $rep COMPLETE" >> "$OUT/progress.log"
done
echo "$(date -u +%FT%TZ) STACKCMP DONE" >> "$OUT/progress.log"
