Q=$(python3 -c "import sglang.srt.distributed.device_communicators.quick_all_reduce as q;print(q.__file__)")
echo "== quick_all_reduce.py: $Q"
grep -n -E "_QR_MIN_SIZE|ROCM_QUICK_REDUCE|_SUPPORTED_WORLD_SIZES|supported_archs|def should_quick_allreduce|qr_max_size|self.disabled|regime|class QuickReduceRegime|FP =|INT8 =|INT6 =|INT4 =|NONE =" $Q | head -50
echo "== _QR_MIN_SIZE table =="; awk '/_QR_MIN_SIZE = \{/{p=1} p{print NR": "$0} /^\}/{if(p){p=0}}' $Q | head -20
echo "== should_quick_allreduce body =="; awk '/def should_quick_allreduce/{p=1;c=0} p{print NR": "$0; c++} c>20{p=0}' $Q
echo "== __init__ regime/cast lines =="; sed -n '160,215p' $Q
