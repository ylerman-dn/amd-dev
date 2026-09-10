#!/bin/bash
# In-model campaign 2026-09-10 rerun on the NEW stack (image v0.5.19-rocm10-mi35x: SGLang 0.5.19, ROCm 10.0, RCCL 2.30.4).
# Args: MODEL_PATH MODEL_TAG "MODE1 MODE2 ..." EXTRA_SERVER_ARGS
# Per mode: 3 detect servers (TUNING logs, 1 rep): sc2, dummy, stocksc2 (stock + plugin -> proves where the all_reduce goes) -> hit-maps;
#           pass 1 (stock, none, nodda, sc2, dummy) and pass 2 (reverse), one server per arm, 6 reps. CUDA graphs ON, radix cache OFF.
# Arms:  stock = AITER custom all-reduce ON, image env untouched (NCCL_MIN_NCHANNELS=112 present), no plugin
#        none  = custom AR off, NCCL_MIN_NCHANNELS removed, RCCL as shipped: its DDA path (RCCL_DDA_ENABLE default) takes the decode all_reduces and never consults the tuner
#        nodda = none + RCCL_DDA_ENABLE=0 (classic ring/tree path = the only path a tuner plugin can steer)     sc2/dummy = nodda + plugin + conf
#        (the planned 'msccl' arm was dropped 16:1xZ: RCCL 2.30.4 carries no MSCCL, RCCL_MSCCL_ENABLE is not even recognised; RM_MSCCL=0 is still passed, harmless)
# Every server: IM_NO_EXPANDABLE=1 (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True breaks AITER custom AR on torch 2.11 / ROCm 10).
# Receipts: after every server the ENV lines (NCCL_MIN_NCHANNELS / RCCL_DDA_ENABLE) and the parsed disable_custom_all_reduce are extracted to receipts.txt
#           and checked against the arm; pass-server rccl logs are gzipped afterwards (disk).
set -u
M=$1; TAG=$2; MODES=$3; XTRA=${4:-}
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_many.sh
H=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna/infer_hitmap.py
C=/opt/shared/ylerman/GPU-107/infer-2026-08-30
ROOT=/data/ylerman/inmodel-rocm10-2026-09-10
export IM_CUDA_GRAPH=1 IM_NO_EXPANDABLE=1 IM_IMAGE=lmsysorg/sglang:v0.5.19-rocm10-mi35x
mode_params() {  # ISL OSL CONC NP
  case $1 in
    d32)  IM_ISL=512;  IM_OSL=512; IM_CONC=32;  IM_NPROMPTS=128;;
    p8k)  IM_ISL=8192; IM_OSL=128; IM_CONC=32;  IM_NPROMPTS=96;;
    d128) IM_ISL=512;  IM_OSL=512; IM_CONC=128; IM_NPROMPTS=384;;
    mix)  IM_ISL=2048; IM_OSL=512; IM_CONC=64;  IM_NPROMPTS=192;;
    d256) IM_ISL=512;  IM_OSL=512; IM_CONC=256; IM_NPROMPTS=768;;
    d512) IM_ISL=512;  IM_OSL=512; IM_CONC=512; IM_NPROMPTS=1536;;
  esac
  export IM_ISL IM_OSL IM_CONC IM_NPROMPTS
}
receipt() {  # $1 base dir, $2 arm, $3 kind(def|tun) -> one line in $1/receipts.txt + mismatch check
  local L=$1/$2_$3/logs
  local minch=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "NCCL_MIN_NCHANNELS set by environment to [0-9]+" | grep -oE "[0-9]+$")
  local dda=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "RCCL_DDA_ENABLE set by environment to [0-9]+" | grep -oE "[0-9]+$")
  local ddainit=$(cat $L/*.log 2>/dev/null | grep -c "ncclDdaIpcCommInit")
  local rccl=$(cat $L/*.log 2>/dev/null | grep -m1 -oE "RCCL version[^,]*")
  local car=$(grep -m1 -oE "disable_custom_all_reduce.: (True|False)" $1/$2_$3/server.log 2>/dev/null | grep -oE "(True|False)$"); car=${car:-unknown}
  local arpath=$(grep -m1 -oE "\[AR\] Using [A-Za-z]+" $1/$2_$3/server.log 2>/dev/null | cut -d" " -f3); arpath=${arpath:-none}
  echo "$2 | disable_custom_all_reduce=$car | AR_impl=$arpath | NCCL_MIN_NCHANNELS_env=${minch:-absent} | RCCL_DDA_ENABLE_env=${dda:-unset} | DdaIpcCommInit_lines=$ddainit | $rccl" >> $1/receipts.txt
  local ok=1
  case $2 in
    none)            [ -z "$minch" ] && [ -z "$dda" ] && [ "$car" = True ] || ok=0;;
    nodda|sc2|dummy) [ -z "$minch" ] && [ "${dda:-x}" = "0" ] && [ "$car" = True ] || ok=0;;
    stock*)          [ -n "$minch" ] && [ "$car" = False ] || ok=0;;
  esac
  [ $ok = 1 ] || echo "=== RECEIPT_MISMATCH $TAG $(basename $(dirname $1)) $(basename $1) arm=$2 minch=${minch:-absent} dda=${dda:-unset} disable_custom_AR=$car"
}
run_arm() {  # $1 base dir, $2 arm, $3 reps, $4 subsys
  export IM_BASE=$1
  echo "=== $TAG $(basename $(dirname $1)) $(basename $1) arm=$2 start $(date -u +%FT%TZ)"
  case $2 in
    stock)    IM_KEEP_CUSTOM_AR=1 RM_MSCCL=image RM_SUBSYS=$4 bash $S $M stock def $3 --disable-radix-cache $XTRA; receipt $1 stock def ;;
    stocksc2) IM_KEEP_CUSTOM_AR=1 RM_MSCCL=image RM_SUBSYS=$4 RM_CONF=sc2_grid112.final.conf bash $S $M stocksc2 tun $3 --disable-radix-cache $XTRA; receipt $1 stocksc2 tun ;;
    none)     IM_DROP_FLOOR=1 RM_MSCCL=0 RM_SUBSYS=$4 bash $S $M none def $3 --disable-radix-cache $XTRA; receipt $1 none def ;;
    nodda)    IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=$4 bash $S $M nodda def $3 --disable-radix-cache $XTRA; receipt $1 nodda def ;;
    sc2)      IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=$4 RM_CONF=sc2_grid112.final.conf bash $S $M sc2 tun $3 --disable-radix-cache $XTRA; receipt $1 sc2 tun ;;
    dummy)    IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=$4 RM_CONF=sc_dummy_ring_simple_1ch.conf bash $S $M dummy tun $3 --disable-radix-cache $XTRA; receipt $1 dummy tun ;;
  esac
}
for m in $MODES; do
  mode_params $m
  D=$ROOT/$TAG/$m; mkdir -p $D/detect $D/pass1 $D/pass2
  echo "=== $TAG $m detect $(date -u +%FT%TZ)"
  for a in sc2 dummy stocksc2; do run_arm $D/detect $a 1 INIT,TUNING,ENV; done
  python3 $H --logs $D/detect/sc2_tun/logs      --conf $C/sc2_grid112.final.conf        -o $D/detect/hitmap_sc2.csv      > /dev/null
  python3 $H --logs $D/detect/dummy_tun/logs    --conf $C/sc_dummy_ring_simple_1ch.conf -o $D/detect/hitmap_dummy.csv    > /dev/null
  python3 $H --logs $D/detect/stocksc2_tun/logs --conf $C/sc2_grid112.final.conf        -o $D/detect/hitmap_stocksc2.csv > /dev/null
  grep -m1 -hoE "RCCL version[^,]*" $D/detect/sc2_tun/logs/*.log | head -1 | sed "s/^/=== RCCL_IN_SERVER /"
  hits=$(awk -F, 'NR>1 && $1=="rule"{h+=$7} END{print h+0}' $D/detect/hitmap_sc2.csv); echo "=== PLUGIN_HITS $TAG $m sc2 $hits"; [ "$hits" = 0 ] && echo "=== PLUGIN_NOT_APPLIED $TAG $m"
  shits=$(awk -F, 'NR>1 && $1=="rule"{h+=$7} END{print h+0}' $D/detect/hitmap_stocksc2.csv); echo "=== STOCK_PLUGIN_HITS $TAG $m $shits"
  for a in sc2 dummy stocksc2; do gzip -q $D/detect/${a}_tun/logs/*.log 2>/dev/null; done
  echo "=== $TAG $m pass1 $(date -u +%FT%TZ)"
  for a in stock none nodda sc2 dummy; do run_arm $D/pass1 $a 6 INIT,ENV; gzip -q $D/pass1/${a}_*/logs/*.log 2>/dev/null; done
  echo "=== $TAG $m pass2 $(date -u +%FT%TZ)"
  for a in dummy sc2 nodda none stock; do run_arm $D/pass2 $a 6 INIT,ENV; gzip -q $D/pass2/${a}_*/logs/*.log 2>/dev/null; done
done
echo "ALL_DONE $TAG $(date -u +%FT%TZ)"
