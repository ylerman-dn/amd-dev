#!/bin/bash
# Does the CONTAINER'S STOCK RCCL consult the tuner per collective, or only at init?
# Runs our rccl-tests binary inside the SGLang container with no LD_PRELOAD, so it links
# the container's own RCCL, with the DN tuner plugin and the shipped 1-node conf.
#
# If `Applied config` appears once per size, the init-only behaviour seen under SGLang is
# SGLang's calling pattern, not the library — and the inference demo is reachable.
# If it appears only at init, stock RCCL genuinely never consults per op.
set -u
CONF_DIR=/opt/shared/ylerman/GPU-107/infer-2026-08-30
BIN=/opt/shared/ylerman/GPU-107/bin
OUT=/data/ylerman/infer-2026-08-30/tunerprobe
IMAGE=lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260817
CNAME=ylerman-tunerprobe

mkdir -p "$OUT"
rm -f "$OUT"/*.log "$OUT"/*.csv
docker rm -f "$CNAME" >/dev/null 2>&1

docker run --rm --ipc=host --shm-size=16g --network=host --name="$CNAME" \
  --privileged --ulimit memlock=-1 \
  --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  -v "$BIN":/opt/rccl-tests:ro \
  -v "$CONF_DIR":/opt/rccl/tuner:ro \
  -v "$OUT":/workspace/out -w /workspace \
  -e NCCL_TUNER_PLUGIN=/opt/rccl/tuner/librccl-tunerv4-dn.so \
  -e NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/all_reduce_1n.final.conf \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV \
  -e NCCL_DEBUG_FILE=/workspace/out/probe_%p.log \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  --entrypoint=/bin/bash \
  "$IMAGE" -c "
ldd /opt/rccl-tests/all_reduce_perf 2>/dev/null | grep -i rccl
/opt/rccl-tests/all_reduce_perf -b 4K -e 1M -f 2 -g 8 -n 20 -w 5 -c 1 -A 1
" > "$OUT/probe.log" 2>&1

echo "rc=$?"
echo "-- which librccl was linked:"
grep -i "librccl" "$OUT/probe.log" | head -2
echo "-- tuner lines:"
echo "   Loaded:  $(grep -h -c 'Loaded .* tuning configurations' $OUT/probe_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "   Applied: $(grep -h -c 'Applied config' $OUT/probe_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "   distinct sizes applied:"
grep -h -o "Applied config for collType=allreduce, bytes=[0-9]*" $OUT/probe_*.log 2>/dev/null | sort -u
