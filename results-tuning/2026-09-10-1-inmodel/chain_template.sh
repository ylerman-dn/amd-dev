#!/bin/bash
# In-model campaign 2026-09-10 (one node = one model, two modes). Args: MODEL_PATH MODEL_TAG "MODE1 MODE2" EXTRA_SERVER_ARGS
# Per mode: detect servers (TUNING logs, 1 rep) for sc2 and dummy -> hit-maps; then pass 1 (stock,none,sc2,dummy) and pass 2 (reverse),
# one server per arm, 6 reps each. CUDA graphs ON, radix cache OFF everywhere. Output tree: $ROOT/<model>/<mode>/{detect,pass1,pass2}/<arm>_<def|tun>/
set -u
M=$1; TAG=$2; MODES=$3; XTRA=${4:-}
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_many.sh
H=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna/infer_hitmap.py
C=/opt/shared/ylerman/GPU-107/infer-2026-08-30
ROOT=/data/ylerman/inmodel-2026-09-10
export IM_CUDA_GRAPH=1
mode_params() {  # ISL OSL CONC NP (from infer_modes.sh table)
  case $1 in
    d32)  IM_ISL=512;  IM_OSL=512; IM_CONC=32;  IM_NPROMPTS=128;;
    p8k)  IM_ISL=8192; IM_OSL=128; IM_CONC=32;  IM_NPROMPTS=96;;
    d128) IM_ISL=512;  IM_OSL=512; IM_CONC=128; IM_NPROMPTS=384;;
    mix)  IM_ISL=2048; IM_OSL=512; IM_CONC=64;  IM_NPROMPTS=192;;
  esac
  export IM_ISL IM_OSL IM_CONC IM_NPROMPTS
}
run_arm() {  # $1 base dir, $2 arm, $3 reps, $4 subsys
  export IM_BASE=$1
  echo "=== $TAG $(basename $(dirname $1)) $(basename $1) arm=$2 start $(date -u +%FT%TZ)"
  case $2 in
    stock) IM_KEEP_CUSTOM_AR=1 RM_MSCCL=1 RM_SUBSYS=$4 bash $S $M stock def $3 --disable-radix-cache $XTRA ;;
    none)  IM_DROP_FLOOR=1 RM_MSCCL=0 RM_SUBSYS=$4 bash $S $M none def $3 --disable-radix-cache $XTRA ;;
    sc2)   IM_DROP_FLOOR=1 RM_MSCCL=0 RM_SUBSYS=$4 RM_CONF=sc2_grid112.final.conf bash $S $M sc2 tun $3 --disable-radix-cache $XTRA ;;
    dummy) IM_DROP_FLOOR=1 RM_MSCCL=0 RM_SUBSYS=$4 RM_CONF=sc_dummy_ring_simple_1ch.conf bash $S $M dummy tun $3 --disable-radix-cache $XTRA ;;
  esac
}
for m in $MODES; do
  mode_params $m
  D=$ROOT/$TAG/$m; mkdir -p $D/detect $D/pass1 $D/pass2
  echo "=== $TAG $m detect $(date -u +%FT%TZ)"
  for a in sc2 dummy; do run_arm $D/detect $a 1 INIT,TUNING,ENV; done
  python3 $H --logs $D/detect/sc2_tun/logs   --conf $C/sc2_grid112.final.conf       -o $D/detect/hitmap_sc2.csv   > /dev/null
  python3 $H --logs $D/detect/dummy_tun/logs --conf $C/sc_dummy_ring_simple_1ch.conf -o $D/detect/hitmap_dummy.csv > /dev/null
  echo "=== $TAG $m pass1 $(date -u +%FT%TZ)"
  for a in stock none sc2 dummy; do run_arm $D/pass1 $a 6 INIT,ENV; done
  echo "=== $TAG $m pass2 $(date -u +%FT%TZ)"
  for a in dummy sc2 none stock; do run_arm $D/pass2 $a 6 INIT,ENV; done
done
echo "ALL_DONE $TAG $(date -u +%FT%TZ)"
