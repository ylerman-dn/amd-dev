#!/bin/bash
# Campaign D (2026-09-14): delivery check - the same conf through RCCL 2.30.4's BUILT-IN CSV tuner (no plugin .so) vs through our plugin.
# One server per model, d32 params, DDA off (so the classic path runs), TUNING logs, 3 reps: arm csv (IM_NO_PLUGIN=1: only NCCL_TUNER_CONFIG_FILE=sc2_plus_amd.conf,
# RCCL's built-in tuner reads it) and arm plug (our plugin + the same conf) back to back on the same node/minute. Hit-maps + tok/s compared in SUMMARY.md.
# Run inside job 21247 on node 2 after campaign C's FIX_DONE: srun --jobid=21247 -N1 -w amd-mi355x-2 bash <this>
set -u
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_many.sh
H=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna/infer_hitmap.py
C=/opt/shared/ylerman/GPU-107/infer-2026-08-30
ROOT=/data/ylerman/csvtuner-2026-09-14
export IM_CUDA_GRAPH=1 IM_NO_EXPANDABLE=1 IM_IMAGE=lmsysorg/sglang:v0.5.19-rocm10-mi35x IM_ISL=512 IM_OSL=512 IM_CONC=32 IM_NPROMPTS=128
Q=/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/
G=/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/
D=/huggingface/hub/models--amd--DeepSeek-R1-0528-MXFP4/snapshots/913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a/
one() {  # tag model xtra
  local tag=$1 m=$2 x=${3:-}; local B=$ROOT/$tag; mkdir -p $B
  echo "=== $tag csv start $(date -u +%FT%TZ)"
  IM_BASE=$B IM_NO_PLUGIN=1 IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=INIT,TUNING,ENV RM_CONF=sc2_plus_amd.conf bash $S $m csv tun 3 --disable-radix-cache $x
  echo "=== $tag csv tuner_line: $(cat $B/csv_tun/logs/*.log | grep -m1 -oE 'Using built-in CSV tuner, config: [^ ]+|TUNER/Plugin: Using [A-Z-]+ \(v[0-9]\)|NCCL_TUNER_PLUGIN set by environment to [^ ]+')"
  python3 $H --logs $B/csv_tun/logs --conf $C/sc2_plus_amd.conf -o $B/hitmap_csv.csv > /dev/null 2>&1
  local ap=$(cat $B/csv_tun/logs/*.log | grep -c 'Applied config'); local cs=$(cat $B/csv_tun/logs/*.log | grep -c 'CsvTuner'); local hc=$(awk -F, 'NR>1 && $1=="rule"{h+=$7} END{print h+0}' $B/hitmap_csv.csv 2>/dev/null)
  echo "=== $tag csv applied_lines=$ap csvtuner_lines=$cs hits=$hc"
  gzip -q $B/csv_tun/logs/*.log 2>/dev/null
  echo "=== $tag plug start $(date -u +%FT%TZ)"
  IM_BASE=$B IM_DROP_FLOOR=1 RM_MSCCL=0 IM_EXTRA_ENV="RCCL_DDA_ENABLE=0" RM_SUBSYS=INIT,TUNING,ENV RM_CONF=sc2_plus_amd.conf bash $S $m plug tun 3 --disable-radix-cache $x
  python3 $H --logs $B/plug_tun/logs --conf $C/sc2_plus_amd.conf -o $B/hitmap_plug.csv > /dev/null 2>&1
  local hp=$(awk -F, 'NR>1 && $1=="rule"{h+=$7} END{print h+0}' $B/hitmap_plug.csv 2>/dev/null)
  echo "=== $tag plug hits=$hp"
  gzip -q $B/plug_tun/logs/*.log 2>/dev/null
  echo "=== $tag done $(date -u +%FT%TZ)"
}
one qwen "$Q" ""
one gptoss "$G" "--attention-backend triton"
one dsr1 "$D" ""
echo "ALL_DONE csvtuner $(date -u +%FT%TZ)"
