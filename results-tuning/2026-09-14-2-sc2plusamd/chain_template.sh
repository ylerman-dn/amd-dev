#!/bin/bash
# Campaign C (2026-09-14): does carrying AMD's built-in gfx950 rules for the sizes sc2 does not cover remove the miss penalty?
# Args: MODEL_PATH MODEL_TAG "MODE1 MODE2 ..." EXTRA_SERVER_ARGS
# Per mode: 2 detect servers (TUNING logs, 1 rep): sc2, sc2plus -> hit-maps; pass 1 (none, nodda, sc2, sc2plus) and pass 2 (reverse), one server per arm,
# 6 reps. CUDA graphs ON, radix cache OFF, custom AR OFF in all arms (this campaign is about the RCCL path), IM_NO_EXPANDABLE=1, image v0.5.19-rocm10.
# Arms:  none    = RCCL as shipped (DDA path, floor removed)          nodda = none + RCCL_DDA_ENABLE=0 (classic path, AMD's built-in gfx950 table active)
#        sc2     = nodda + plugin + sc2_grid112.final.conf (displaces AMD's table: uncovered sizes fall to the generic cost model)
#        sc2plus = nodda + plugin + sc2_plus_amd.conf (sc2 rules + AMD's rules in every gap)
set -u
M=$1; TAG=$2; MODES=$3; XTRA=${4:-}
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_many.sh
H=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna/infer_hitmap.py
C=/opt/shared/ylerman/GPU-107/infer-2026-08-30
ROOT=/data/ylerman/inmodel-sc2plus-2026-09-14
export IM_CUDA_GRAPH=1 IM_NO_EXPANDABLE=1 IM_IMAGE=lmsysorg/sglang:v0.5.19-rocm10-mi35x
mode_params() {  # ISL OSL CONC NP (same as the 2026-09-10 campaigns)
  case $1 in
    d32)  IM_ISL=512;  IM_OSL=512; IM_CONC=32;  IM_NPROMPTS=128;;
    d128) IM_ISL=512;  IM_OSL=512; IM_CONC=128; IM_NPROMPTS=384;;
    mix)  IM_ISL=2048; IM_OSL=512; IM_CONC=64;  IM_NPROMPTS=192;;
    d256) IM_ISL=512;  IM_OSL=512; IM_CONC=256; IM_NPROMPTS=768;;
  esac
  export IM_ISL IM_OSL IM_CONC IM_NPROMPTS
}
receipt() {  # $1 base dir, $2 arm, $3 kind(def|tun)
  local L=$1/$2_$3/logs
  local minch=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "NCCL_MIN_NCHANNELS set by environment to [0-9]+" | grep -oE "[0-9]+$")
  local dda=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "RCCL_DDA_ENABLE set by environment to [0-9]+" | grep -oE "[0-9]+$")
  local tuner=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "Using built-in CSV tuner|TUNER/Plugin: Using [A-Z-]+ \(v[0-9]\)")
  local conf=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "NCCL_TUNER_CONFIG_FILE set by environment to [^ ]+" | sed 's|.*/||')
  local car=$(grep -m1 -oE "disable_custom_all_reduce.: (True|False)" $1/$2_$3/server.log 2>/dev/null | grep -oE "(True|False)$"); car=${car:-unknown}
  echo "$2 | disable_custom_all_reduce=$car | NCCL_MIN_NCHANNELS_env=${minch:-absent} | RCCL_DDA_ENABLE_env=${dda:-unset} | tuner=${tuner:-none} | conf=${conf:-none}" >> $1/receipts.txt
  local ok=1
  case $2 in
    none)    [ -z "$minch" ] && [ -z "$dda" ] && [ "$car" = True ] || ok=0;;
    nodda)   [ -z "$minch" ] && [ "${dda:-x}" = 0 ] && [ "$car" = True ] && [[ "$tuner" == *CSV* ]] || ok=0;;
    sc2)     [ -z "$minch" ] && [ "${dda:-x}" = 0 ] && [ "$car" = True ] && [[ "$tuner" == *DN-TUNER* ]] && [ "$conf" = sc2_grid112.final.conf ] || ok=0;;
    sc2plus) [ -z "$minch" ] && [ "${dda:-x}" = 0 ] && [ "$car" = True ] && [[ "$tuner" == *DN-TUNER* ]] && [ "$conf" = sc2_plus_amd.conf ] || ok=0;;
  esac
  [ $ok = 1 ] || echo "=== RECEIPT_MISMATCH $TAG $(basename $(dirname $1)) $(basename $1) arm=$2 minch=${minch:-absent} dda=${dda:-unset} car=$car tuner=${tuner:-none} conf=${conf:-none}"
}
run_arm() {  # $1 base dir, $2 arm, $3 reps, $4 subsys
  export IM_BASE=$1
  echo "=== $TAG $(basename $(dirname $1)) $(basename $1) arm=$2 start $(date -u +%FT%TZ)"
  case $2 in
    none)    IM_DROP_FLOOR=1 RM_MSCCL=0 RM_SUBSYS=$4 bash $S $M none def $3 --disable-radix-cache $XTRA; receipt $1 none def ;;
    nodda)   IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=$4 bash $S $M nodda def $3 --disable-radix-cache $XTRA; receipt $1 nodda def ;;
    sc2)     IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=$4 RM_CONF=sc2_grid112.final.conf bash $S $M sc2 tun $3 --disable-radix-cache $XTRA; receipt $1 sc2 tun ;;
    sc2plus) IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=$4 RM_CONF=sc2_plus_amd.conf bash $S $M sc2plus tun $3 --disable-radix-cache $XTRA; receipt $1 sc2plus tun ;;
  esac
}
for m in $MODES; do
  mode_params $m
  D=$ROOT/$TAG/$m; mkdir -p $D/detect $D/pass1 $D/pass2
  echo "=== $TAG $m detect $(date -u +%FT%TZ)"
  for a in sc2 sc2plus; do run_arm $D/detect $a 1 INIT,TUNING,ENV; done
  python3 $H --logs $D/detect/sc2_tun/logs     --conf $C/sc2_grid112.final.conf -o $D/detect/hitmap_sc2.csv     > /dev/null
  python3 $H --logs $D/detect/sc2plus_tun/logs --conf $C/sc2_plus_amd.conf      -o $D/detect/hitmap_sc2plus.csv > /dev/null
  grep -m1 -hoE "RCCL version[^,]*" $D/detect/sc2_tun/logs/*.log | head -1 | sed "s/^/=== RCCL_IN_SERVER /"
  for a in sc2 sc2plus; do hits=$(awk -F, -v a=$a 'NR>1 && $1=="rule"{h+=$7} END{print h+0}' $D/detect/hitmap_$a.csv); echo "=== PLUGIN_HITS $TAG $m $a $hits"; [ "$hits" = 0 ] && echo "=== PLUGIN_NOT_APPLIED $TAG $m $a"; done
  for a in sc2 sc2plus; do gzip -q $D/detect/${a}_tun/logs/*.log 2>/dev/null; done
  echo "=== $TAG $m pass1 $(date -u +%FT%TZ)"
  for a in none nodda sc2 sc2plus; do run_arm $D/pass1 $a 6 INIT,ENV; gzip -q $D/pass1/${a}_*/logs/*.log 2>/dev/null; done
  echo "=== $TAG $m pass2 $(date -u +%FT%TZ)"
  for a in sc2plus sc2 nodda none; do run_arm $D/pass2 $a 6 INIT,ENV; gzip -q $D/pass2/${a}_*/logs/*.log 2>/dev/null; done
done
echo "ALL_DONE $TAG $(date -u +%FT%TZ)"
