#!/bin/bash
# B' verification (post-hoc, TUNING logs): what does RCCL execute for the 180 MiB gpt-oss prefill all_reduce under
#   (1) a fresh server with an EMPTY plugin conf (true "no rule"), (2) rl32 = ring/ll128/32, (3) tl112 = tree/ll128/112, (4) rs112, (5) nomatch.conf (valid file, rule never matches)?
# One hot-reload server, 1 round per arm, TUNING logs; afterwards the rank logs give "AllReduce: 188743680 Bytes -> Algo X proto Y channel{Lo..Hi}" per phase.
# Usage (inside job 21253 on node 6 after the gpt-oss chain): srun --jobid=21253 -N1 -w amd-mi355x-6 bash <this>
set -u
P=/opt/shared/ylerman/GPU-107/infer-2026-08-30/infer_paired.sh
export IM_CUDA_GRAPH=1 IM_NO_EXPANDABLE=1 IM_KEEP_CUSTOM_AR=1 IM_IMAGE=lmsysorg/sglang:v0.5.19-rocm10-mi35x
export IM_ISL=8192 IM_OSL=128 IM_CONC=32 IM_NPROMPTS=96
M=/data/ylerman/models/gpt-oss-120b
D=/data/ylerman/bigmsg-inmodel-2026-09-14/gptoss/verify; mkdir -p $D
echo "=== verify start $(date -u +%FT%TZ)"
IM_BASE=$D IM_DROP_FLOOR=1 IM_EXTRA_ENV="ROCM_QUICK_REDUCE_QUANTIZATION=NONE" RM_SUBSYS=INIT,TUNING,ENV IM_EXTRA="--chunked-prefill-size 32768 --max-prefill-tokens 32768 --attention-backend triton --disable-radix-cache" \
  bash $P $M vgptoss 1 none:NONE rl32:big_ring_ll128_32.conf tl112:big_tree_ll128_112.conf rs112:big_ring_simple_112.conf nomatch:nomatch.conf none2:NONE
L=$D/vgptoss_srv/logs
echo "=== hot-reload lines: $(cat $L/*.log 2>/dev/null | grep -c 'hot-reloaded') failed: $(cat $L/*.log 2>/dev/null | grep -c 'hot-reload of .* FAILED')"
echo "=== executed configs for 188743680 B, in log order (rank 0):"
f=$(ls $L/*.log | head -1); grep -oE "AllReduce: 188743680 Bytes -> Algo [A-Z]+ proto [A-Z0-9]+ channel\{Lo\.\.Hi\}=\{[0-9]+\.\.[0-9]+\}" $f | uniq -c
echo "=== applied lines per rule (rank 0):"; grep -oE "Applied config for collType=allreduce, bytes=188743680.*channels=[0-9]+" $f | sed -E 's/.*algo=/algo=/' | uniq -c
gzip -q $L/*.log 2>/dev/null
echo "=== verify done $(date -u +%FT%TZ)"
