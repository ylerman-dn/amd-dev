set -u
echo "== aiter git =="; (cd /sgl-workspace/aiter 2>/dev/null && git rev-parse HEAD && git log -1 --format='%cd %s' ) || echo "no git at /sgl-workspace/aiter"
pip show aiter 2>/dev/null | head -3
F=$(python3 -c "import importlib.util,sys;s=importlib.util.find_spec('aiter');print(s.submodule_search_locations[0])" 2>/dev/null)/dist/device_communicators/custom_all_reduce.py
echo "== aiter file: $F"; ls -la "$F"
echo "== aiter size logic (grep) =="
grep -n -E "_DEFAULT_CAR_MAX_SIZE|AITER_CUSTOM_AR_MAX_SIZE|AITER_CUSTOM_AR_MIN_SIZE|max_size\s*=|def _fits_custom_ar_size|_car_max_size|def should_custom_ar|_SUPPORTED_WORLD_SIZES|fully_connected\s*=|def custom_all_reduce|def all_reduce\b" "$F"
echo "== _fits_custom_ar_size / should_custom_ar bodies =="
awk '/def _fits_custom_ar_size|def should_custom_ar/{p=1;c=0} p{print NR": "$0; c++} c>28{p=0}' "$F"
echo "== sglang custom_all_reduce.py lines 38-60 =="
G=$(python3 -c "import sglang.srt.distributed.device_communicators.custom_all_reduce as c;print(c.__file__)")
sed -n '38,60p' "$G"
echo "== sglang: where AiterCustomAllreduce is constructed (max_size arg?) =="
grep -n -E "AiterCustomAllreduce\(|max_size" "$G" | head -20
echo "== gpu import test =="
python3 - <<'PY'
import aiter.dist.device_communicators.custom_all_reduce as m
print("import ok; _DEFAULT_CAR_MAX_SIZE =", getattr(m,"_DEFAULT_CAR_MAX_SIZE","ABSENT"))
PY
