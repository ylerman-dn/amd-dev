#!/bin/bash
# A/B driver for GPU-107 — run all_reduce_perf against a chosen RCCL build, with/without the amd-dev tuner plugin.
# Records the FULL command + env into the log header (fixes the "commands not saved" gap), plus captures
# RCCL's chosen algo/proto/channels via NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING.
#
# Run ON a cluster node (has /opt/shared NFS + srun). Requires an existing salloc allocation ($JID).
#
# Params via env:
#   JID       (required) jobid of an salloc --no-shell allocation (1 node for NODES=1, 2 nodes for NODES=2)
#   NODES     1 | 2                              (default 1)
#   RCCL_LIB  dir containing librccl.so.1        (default: our 2e42aa8 build in .../bin)
#   PLUGIN    path to librccl-tunerv4-dn.so      (default: empty = default RCCL, no tuner)
#   CONF      path to tuner .conf                (default: empty)
#   TAG       output basename                    (default: run)
#   OUTDIR    where logs go                      (default: /opt/shared/ylerman/GPU-107/ab-oldrccl)
#   SIZES     rccl-tests size args               (default: -b 8 -e 536870912 -f 2 -g 8)
set -u
BIN=/opt/shared/ylerman/GPU-107/bin
RCCL_LIB=${RCCL_LIB:-$BIN}
PLUGIN=${PLUGIN:-}
CONF=${CONF:-}
TAG=${TAG:-run}
NODES=${NODES:-1}
OUTDIR=${OUTDIR:-/opt/shared/ylerman/GPU-107/ab-oldrccl}
SIZES=${SIZES:--b 8 -e 536870912 -f 2 -g 8}
: "${JID:?set JID to your salloc allocation id}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/${TAG}.log"

ENV="LD_LIBRARY_PATH=$RCCL_LIB:/opt/rocm/lib NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,ENV"
[ -n "$PLUGIN" ] && ENV="$ENV NCCL_TUNER_PLUGIN=$PLUGIN"
[ -n "$CONF" ]   && ENV="$ENV NCCL_TUNER_CONFIG_FILE=$CONF"

if [ "$NODES" = 2 ]; then
  NET="NCCL_IB_GID_INDEX=1 NCCL_SOCKET_IFNAME=enp81s0f1np1 OMPI_MCA_btl_tcp_if_include=enp81s0f1np1 OMPI_MCA_oob_tcp_if_include=enp81s0f1np1"
  LAUNCH="srun --jobid=$JID -N2 --ntasks-per-node=1 --gres=gpu:8 --mpi=pmix --export=ALL"
  FULLENV="$ENV $NET"
else
  LAUNCH="srun --jobid=$JID -N1 -n1 --gres=gpu:8"
  FULLENV="$ENV"
fi

# Self-describing header so the log records exactly what ran.
{
  echo "# GPU-107 A/B run  $(date -u)"
  echo "# TAG=$TAG NODES=$NODES"
  echo "# RCCL_LIB=$RCCL_LIB"
  echo "# PLUGIN=${PLUGIN:-<none>}"
  echo "# CONF=${CONF:-<none>}"
  echo "# LAUNCH=$LAUNCH"
  echo "# ENV=$FULLENV"
  echo "# CMD=$BIN/all_reduce_perf $SIZES"
  echo "# ----------------------------------------------------------------"
} > "$LOG"

$LAUNCH bash -c "export $FULLENV; $BIN/all_reduce_perf $SIZES" >> "$LOG" 2>&1
rc=$?
echo "# srun rc=$rc" >> "$LOG"

echo "wrote $LOG (rc=$rc)"
echo "--- chosen algo/proto/ch (unique) ---"
grep -oE "AllReduce: [0-9]+ Bytes -> Algo [A-Za-z0-9]+ proto [A-Za-z0-9]+ channel\{Lo..Hi\}=\{[0-9]+..[0-9]+\}" "$LOG" | sort -k2 -n -u
echo "--- tuner plugin load lines (if any) ---"
grep -iE "tuner|DN-TUNER|TUNER_PLUGIN|Using.*plugin" "$LOG" | head
