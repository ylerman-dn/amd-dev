#!/bin/bash
# Campaign B' (2026-09-14): tune the tuner FROM THE MODEL at >= 128 MiB all_reduces (prefill), where AITER (<= 64 MiB) and DDA (<= 64 MiB) do not act.
# Args: MODEL_PATH MODEL_TAG "BATCH1 BATCH2" EXTRA_SERVER_ARGS
#   BATCH = prefill batch tokens (--chunked-prefill-size = --max-prefill-tokens); message = BATCH x hidden x 2 B.
# Env of every server: custom AR ON (image default), ROCM_QUICK_REDUCE_QUANTIZATION=NONE on the RCCL arms (so > 64 MiB falls through to RCCL), CUDA graphs ON,
#   radix cache OFF, IM_NO_EXPANDABLE=1, image v0.5.19-rocm10, floor NCCL_MIN_NCHANNELS removed on the RCCL arms.
# Per batch size:
#   detect : one hot-reload server, TUNING logs, 1 round of arms NONE + big_ring_simple_112 -> which all_reduce sizes reach RCCL and that the big rule fires (hit-map).
#   search : one hot-reload server (INIT,ENV), ROUNDS rounds round-robin over 13 arms: NONE (RCCL default) + 12 one-rule confs over [64 MiB+1, 2 GiB]:
#            ring/simple x {8,16,24,32,48,64,80,112}, ring/ll128 x {32,64,112}, tree/ll128 x 112. Prefill is not graph-captured, so each swap acts on the next rep.
#            Metric: prefill = Input token throughput and Mean TTFT (bench_serving), per round.
#   pick   : best arm by median input tok/s over rounds; run the A/B only if it beats NONE by > 2%.
#   ab     : one server per arm, 6 reps, pass1 default(QR NONE, no plugin) -> conf -> stock(image INT8 quick reduce); pass2 reversed.
# The hot-reload live conf (live_<prefix>.conf) lives in the shared CONF_DIR on NFS: prefixes carry the model tag so three nodes never share one file.
# Traffic: ISL 8192 / OSL 128 / conc 32 / 96 prompts (prefill-heavy). Every timestamp UTC; make_timeline.py renders Israel time.
set -u
M=$1; TAG=$2; BATCHES=$3; XTRA=${4:-}
S=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_many.sh
P=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh
H=/opt/shared/ylerman/GPU-107/rccl-sweep-optuna/infer_hitmap.py
C=/opt/shared/ylerman/GPU-107/infer-2026-08-30
ROOT=/data/ylerman/bigmsg-inmodel-2026-09-14
ROUNDS=${BM_ROUNDS:-3}
ARMS="none:nomatch.conf rs8:big_ring_simple_8.conf rs16:big_ring_simple_16.conf rs24:big_ring_simple_24.conf rs32:big_ring_simple_32.conf rs48:big_ring_simple_48.conf rs64:big_ring_simple_64.conf rs80:big_ring_simple_80.conf rs112:big_ring_simple_112.conf rl32:big_ring_ll128_32.conf rl64:big_ring_ll128_64.conf rl112:big_ring_ll128_112.conf tl112:big_tree_ll128_112.conf"
export IM_CUDA_GRAPH=1 IM_NO_EXPANDABLE=1 IM_KEEP_CUSTOM_AR=1 IM_IMAGE=lmsysorg/sglang:v0.5.19-rocm10-mi35x
export IM_ISL=8192 IM_OSL=128 IM_CONC=32 IM_NPROMPTS=96
QR="ROCM_QUICK_REDUCE_QUANTIZATION=NONE"
med() { sort -n | awk '{a[NR]=$1} END{if(NR==0){print 0}else if(NR%2){print a[(NR+1)/2]}else{print (a[NR/2]+a[NR/2+1])/2}}'; }
for B in $BATCHES; do
  D=$ROOT/$TAG/b$B; mkdir -p $D
  PF="--chunked-prefill-size $B --max-prefill-tokens $B $XTRA"
  echo "=== $TAG b$B detect $(date -u +%FT%TZ)"
  IM_BASE=$D/detect IM_DROP_FLOOR=1 IM_EXTRA_ENV="$QR" RM_SUBSYS=INIT,TUNING,ENV IM_EXTRA="$PF --disable-radix-cache" bash $P $M det$TAG 1 none:nomatch.conf rs112:big_ring_simple_112.conf
  python3 $H --logs $D/detect/det${TAG}_srv/logs --conf $C/big_ring_simple_112.conf -o $D/detect/hitmap_big.csv > /dev/null 2>&1
  big=$(awk -F, 'NR>1 && $1=="rule"{h+=$7} END{print h+0}' $D/detect/hitmap_big.csv); echo "=== BIG_HITS $TAG b$B $big"
  echo "=== AR_SIZES $TAG b$B $(zcat -f $D/detect/det${TAG}_srv/logs/*.log* | grep -oE 'AllReduce: [0-9]+ Bytes' | awk '{print $2}' | sort -n | uniq -c | awk '$2>67108864{printf "%s x%s; ", $2, $1}')"
  gzip -q $D/detect/det${TAG}_srv/logs/*.log 2>/dev/null
  echo "=== $TAG b$B search $(date -u +%FT%TZ)"
  IM_BASE=$D/search IM_DROP_FLOOR=1 IM_EXTRA_ENV="$QR" RM_SUBSYS=INIT,ENV IM_EXTRA="$PF --disable-radix-cache" bash $P $M s$TAG $ROUNDS $ARMS
  gzip -q $D/search/s${TAG}_srv/logs/*.log 2>/dev/null
  # pick: median input tok/s per arm from the bench logs
  : > $D/search/pick.txt
  for spec in $ARMS; do a=${spec%%:*}; v=$(grep -hoP 'Input token throughput \(tok/s\):\s+\K[0-9.]+' $D/search/s${TAG}_$a/bench_round*.log 2>/dev/null | med); t=$(grep -hoP 'Mean TTFT \(ms\):\s+\K[0-9.]+' $D/search/s${TAG}_$a/bench_round*.log 2>/dev/null | med); echo "$a $v $t" >> $D/search/pick.txt; done
  base=$(awk '$1=="none"{print $2}' $D/search/pick.txt); best=$(awk '$1!="none"' $D/search/pick.txt | sort -k2 -gr | head -1)
  bname=${best%% *}; bval=$(echo $best | awk '{print $2}')
  gain=$(awk -v b=$bval -v n=$base 'BEGIN{if(n>0) printf "%.2f", (b/n-1)*100; else print 0}')
  echo "=== PICK $TAG b$B best=$bname in_tok/s=$bval none=$base gain=${gain}%"
  if awk -v g=$gain 'BEGIN{exit !(g>2)}'; then
    conf=$(echo "$ARMS" | tr ' ' '\n' | grep "^$bname:" | cut -d: -f2)
    echo "=== $TAG b$B ab conf=$conf $(date -u +%FT%TZ)"
    for pass in pass1 pass2; do
      if [ $pass = pass1 ]; then order="default conf stock"; else order="stock conf default"; fi
      for arm in $order; do
        echo "=== $TAG b$B $pass arm=$arm start $(date -u +%FT%TZ)"
        case $arm in
          default) IM_BASE=$D/$pass IM_DROP_FLOOR=1 IM_EXTRA_ENV="$QR" RM_SUBSYS=INIT,ENV bash $S $M default def 6 --disable-radix-cache $PF ;;
          conf)    IM_BASE=$D/$pass IM_DROP_FLOOR=1 IM_EXTRA_ENV="$QR" RM_SUBSYS=INIT,ENV RM_CONF=$conf bash $S $M conf tun 6 --disable-radix-cache $PF ;;
          stock)   IM_BASE=$D/$pass RM_MSCCL=image RM_SUBSYS=INIT,ENV bash $S $M stock def 6 --disable-radix-cache $PF ;;
        esac
        gzip -q $D/$pass/${arm}_*/logs/*.log 2>/dev/null
      done
    done
  else
    echo "=== NO_AB $TAG b$B (best $bname only ${gain}% over RCCL default)"
    echo "=== $TAG b$B stockref $(date -u +%FT%TZ)"
    IM_BASE=$D/stockref RM_MSCCL=image RM_SUBSYS=INIT,ENV bash $S $M stock def 3 --disable-radix-cache $PF; gzip -q $D/stockref/stock_def/logs/*.log 2>/dev/null
  fi
  echo "=== $TAG b$B done $(date -u +%FT%TZ)"
done
echo "ALL_DONE $TAG $(date -u +%FT%TZ)"
