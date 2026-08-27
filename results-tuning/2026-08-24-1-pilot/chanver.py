import re, glob, collections
BASE = "/opt/shared/ylerman/GPU-107/rccl-tune-2026-08-24-pilot/ab_out3/allreduce_1n"

def parse(pattern):
    execd = collections.defaultdict(list)
    applied = {}
    for f in glob.glob(pattern):
        for line in open(f, errors="ignore"):
            m = re.search(r"AllReduce: (\d+) Bytes -> Algo (\S+) proto (\S+) "
                          r"channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}", line)
            if m:
                execd[int(m.group(1))].append(
                    (m.group(2), m.group(3),
                     int(m.group(5)) - int(m.group(4)) + 1))
            m = re.search(r"Applied config for collType=allreduce, "
                          r"bytes=(\d+).*channels=(\d+)", line)
            if m:
                applied[int(m.group(1))] = int(m.group(2))
    out = {}
    for size, v in execd.items():
        mode = collections.Counter(v).most_common(1)[0][0]
        out[size] = (mode, applied.get(size))
    return out

hdr = "%10s %18s %20s %12s"
print("=== CONFIG ARM (r1, all 8 ranks) - plugin-applied vs executed ===")
print(hdr % ("size", "applied ch", "executed algo/proto", "executed ch"))
for size, ((algo, proto, ch), app) in sorted(parse(BASE + "/1n_config_r1_dbg_*.log").items()):
    flag = "" if app is None or app == ch else "   <-- differs"
    print((hdr % (size, app, f"{algo}/{proto}", ch)) + flag)
print()
print("=== DEFAULT ARM (r1) - RCCL's own choice, executed ===")
print(hdr % ("size", "-", "executed algo/proto", "executed ch"))
for size, ((algo, proto, ch), _) in sorted(parse(BASE + "/1n_default_r1_dbg_*.log").items()):
    print(hdr % (size, "-", f"{algo}/{proto}", ch))
