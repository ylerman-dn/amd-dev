import inspect, re, subprocess, sys
print("== aiter ==")
try:
    import aiter
    print("aiter file:", aiter.__file__, "version:", getattr(aiter, "__version__", None))
except Exception as e:
    print("import aiter failed:", e)
try:
    print(subprocess.run(["pip", "show", "aiter"], capture_output=True, text=True).stdout.strip())
except Exception as e:
    print("pip show failed:", e)
try:
    import aiter.dist.device_communicators.custom_all_reduce as m
    print("module:", m.__file__)
    print("_DEFAULT_CAR_MAX_SIZE =", getattr(m, "_DEFAULT_CAR_MAX_SIZE", "ABSENT"))
    src = inspect.getsource(m)
    pat = r"_DEFAULT_CAR_MAX_SIZE|AITER_CUSTOM_AR_MAX_SIZE|AITER_CUSTOM_AR_MIN_SIZE|max_size\s*=|def _fits_custom_ar_size|_car_max_size|def should_custom_ar|_SUPPORTED_WORLD_SIZES|fully_connected\s*="
    for i, l in enumerate(src.splitlines(), 1):
        if re.search(pat, l):
            print(f"  {i}: {l.rstrip()}")
    i0 = src.find("def _fits_custom_ar_size")
    if i0 >= 0:
        print("--- _fits_custom_ar_size ---"); print("\n".join(src[i0:].splitlines()[:30]))
    i1 = src.find("def should_custom_ar")
    if i1 >= 0:
        print("--- should_custom_ar ---"); print("\n".join(src[i1:].splitlines()[:25]))
except Exception as e:
    print("aiter custom_all_reduce inspect failed:", e)
print("== sglang ==")
try:
    import sglang
    print("sglang version:", sglang.__version__)
    import sglang.srt.distributed.device_communicators.custom_all_reduce as c
    for i, l in enumerate(inspect.getsource(c).splitlines(), 1):
        if re.search(r"SGLANG_USE_AITER_AR|_MAX_CAR_SIZE|Adapted from|AiterCustomAllreduce", l):
            print(f"  {i}: {l.rstrip()}")
except Exception as e:
    print("sglang inspect failed:", e)
