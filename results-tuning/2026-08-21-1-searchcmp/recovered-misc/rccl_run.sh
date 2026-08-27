#!/usr/bin/env bash
# rccl_run.sh — run ONE rccl-tests collective on the current host(s), capture a full log.
# Node-side helper used by the GPU-107 spike (single-node: run directly; multi-node: via mpirun).
#
# Usage:
#   rccl_run.sh <collective_binary> <out_log> [extra rccl-tests args...]
# Env knobs (optional), exported by caller to sweep parameters:
#   NCCL_PROTO, NCCL_ALGO, NCCL_DEBUG, NCCL_IB_HCA, RCCL_MSCCL_ENABLE, ...
set -uo pipefail

BINDIR="${RCCL_BINDIR:-/opt/shared/ylerman/GPU-107/bin}"
COLL="${1:?need collective binary name, e.g. all_reduce_perf}"; shift
OUT="${1:?need output log path}"; shift

export LD_LIBRARY_PATH="${BINDIR}:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$(dirname "$OUT")"

{
  echo "# ===== rccl_run.sh ====="
  echo "# host           : $(hostname)"
  echo "# date_utc       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# collective     : ${COLL}"
  echo "# args           : $*"
  echo "# bindir         : ${BINDIR}"
  echo "# LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
  echo "# librccl        : $(readlink -f "${BINDIR}/librccl.so.1" 2>/dev/null) ($(md5sum "${BINDIR}/librccl.so.1.0" 2>/dev/null | awk '{print $1}'))"
  echo "# nccl/rccl env  :"
  env | grep -E '^(NCCL_|RCCL_|HIP_|ROCR_|HSA_)' | sort | sed 's/^/#   /' || true
  echo "# ============================================================"
  "${BINDIR}/${COLL}" "$@"
  rc=$?
  echo "# ============================================================"
  echo "# EXIT_CODE=${rc}"
} > "$OUT" 2>&1
