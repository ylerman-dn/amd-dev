#!/usr/bin/python3
"""rccl-tune: one command for the full tuning loop on the AMD MI355X cluster.

    rccl_tune.py run --collectives all_reduce --scales 1,2 --name pilot [--dry-run]
    rccl_tune.py status
    rccl_tune.py report

`run` executes, per (collective, scale):

  book    reserve nodes via salloc (blacklist enforced, co-tenants reported,
          allocation ALWAYS released on exit, even on crash/Ctrl-C)
  search  adaptive racing search, live (anchors 1,8,24,48, margin 15, tol 0.5,
          median-of-3) -> per-size winners + 3 default runs for the gains page
  config  winners -> tuner conf (lowercase-string format, plugin-ready)
  ab      validate the conf through the tuner plugin vs default
          (interleaved pairs, P(sup) >= 0.95, landmine check)
  record  RUNLOG.md row + results fetched under results-tuning/<date>-N-<name>/

The dev VM has no Slurm client: Slurm calls go through ssh to --slurm-host.
Benchmarks themselves run mpirun-over-ssh from the first booked node, driven
by adaptive_search.py (search) and validate_tuner_config.py (A/B).

Node policy: node 2 (orchestrator) is never used (node 9 verified usable 2026-08-24); the
broken list starts with node 4 (fabric fault 2026-08-18) and can be edited in
BROKEN_NODES below. Booking prefers previously-published node sets.
"""

import argparse
import csv
import datetime
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

TOOL = Path(__file__).resolve().parent
REPO = TOOL.parent.parent
RESULTS = REPO / "results-tuning"  # override with --results-root
SLURM_HOST = "amd-mi355x-1"
SHARED = "/opt/shared/ylerman/GPU-107"
REMOTE_TOOL = f"{SHARED}/rccl-sweep-optuna"
PLUGIN = f"{SHARED}/ab-tuner-test/librccl-tunerv4-dn.so"
MY_PATH = f"{SHARED}/bin"
# 2026-09-08: benchmark OUTPUT dirs live on the exec node's LOCAL disk. The
# container runs as root and /opt/shared (NFS, root_squash) refuses root writes:
# NCCL_DEBUG_FILE could not be opened, INFO spilled onto stdout, the parser lost
# every data row and metrics.csv was never written (sanity-check-1 attempt 2).
# Confs and the A/B launch log stay on SHARED (written by dn on the host side).
DATA_ROOT = "/data/ylerman"

FORBIDDEN_NODES = set()           # none (user decision 2026-08-27); node 2's
                                  # old "orchestrator" label was never traced
BROKEN_NODES = set()              # node 4's 2026-08-18 "fabric fault" note had
                                  # no locatable log; treat as healthy until a
                                  # failure points at it
NODE_PREFERENCE = [5, 8, 3, 7, 6, 1]

# the validated search policy (findings/07); changing any of these is a
# measurement-parameter change and needs explicit user approval
# 2026-09-14 (bigmsg campaign): optional overrides threaded to search (adaptive_search live) and A/B (ab_run ->
# validate): combos "ALGO:PROTO,...", size window, container image. Empty = the validated defaults.
EXTRA = dict(combos=None, min_size=None, max_size=None, image=None)


def size_bytes(s):
    s = str(s).strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(s[:-1]) * mult[s[-1]]) if s and s[-1] in mult else int(s)


def ab_extra_args():
    out = []
    if EXTRA.get("min_size"): out += ["--min-bytes", str(size_bytes(EXTRA["min_size"]))]
    if EXTRA.get("max_size"): out += ["--max-bytes", str(size_bytes(EXTRA["max_size"]))]
    if EXTRA.get("image"): out += ["--image", EXTRA["image"]]
    return " ".join(out)


POLICY = dict(anchors="1,8,24,48", margin="15", tol="0.5",
              grid="1,2,4,8,16,24,32,40,48")

COLLECTIVES = ["all_reduce", "broadcast", "reduce", "all_gather",
               "reduce_scatter", "alltoall"]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{utc()}] {msg}", flush=True)


class Cmd:
    """Runs commands, or just prints them under --dry-run."""

    def __init__(self, dry):
        self.dry = dry

    def local(self, args_, timeout=7200, check=True, quiet=False):
        if not quiet:
            log("local: " + " ".join(str(a) for a in args_))
        if self.dry:
            return subprocess.CompletedProcess(args_, 0, "", "")
        p = subprocess.run([str(a) for a in args_], capture_output=True,
                           text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        if check and p.returncode != 0:
            raise RuntimeError(f"command failed rc={p.returncode}: "
                               f"{args_}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
        return p

    def remote(self, cmd, host=SLURM_HOST, timeout=600, check=True,
               quiet=False):
        if not quiet:
            log(f"ssh {host}: {cmd}")
        if self.dry:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        p = subprocess.run(["ssh", "-o", "BatchMode=yes",
                            "-o", "ConnectTimeout=15", host, cmd],
                           capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        if check and p.returncode != 0:
            raise RuntimeError(f"ssh {host} failed rc={p.returncode}: "
                               f"{cmd}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
        return p


# ------------------------------------------------------------------ booking

def node_num(name):
    return int(name.rsplit("-", 1)[1])


def idle_nodes(c):
    p = c.remote("sinfo -p XAI -N -h -o '%N %T'", quiet=True)
    if c.dry:
        return []
    usable = []
    for line in p.stdout.splitlines():
        name, state = line.split()
        n = node_num(name)
        if state == "idle" and n not in FORBIDDEN_NODES and n not in BROKEN_NODES:
            usable.append(n)
    return usable


def has_sweep_python(c, n):
    """rccl_sweep.py runs on the booked node's HOST python and needs
    tabulate+yaml (+colorama); nodes 4 and 5 lack them (2026-09-08)."""
    p = c.remote("python3 -c 'import tabulate, yaml, colorama'",
                 host=f"amd-mi355x-{n}", check=False, quiet=True)
    return c.dry or p.returncode == 0


def pick_nodes(c, k):
    # 2026-09-08: benchmarks run as srun steps INSIDE the allocation, so the
    # exec node must be a booked node - book only nodes whose host python can
    # run rccl_sweep.py (was: book anything idle, drive from a fallback node)
    avail = [n for n in idle_nodes(c) if has_sweep_python(c, n)]
    ranked = [n for n in NODE_PREFERENCE if n in avail] + \
             sorted(n for n in avail if n not in NODE_PREFERENCE)
    if c.dry:
        ranked = NODE_PREFERENCE[:k]  # illustrative
    if len(ranked) < k:
        raise RuntimeError(f"need {k} healthy idle nodes, have {ranked} "
                           f"(forbidden {sorted(FORBIDDEN_NODES)}, "
                           f"broken {sorted(BROKEN_NODES)})")
    return sorted(ranked[:k])


def co_tenant_report(c):
    p = c.remote("squeue -p XAI -h -o '%i %u %j %T %N'", quiet=True)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    if lines:
        log("co-tenants on the fabric (multi-node numbers may be disturbed):")
        for l in lines:
            log("   " + l)
    return lines


def book(c, nodes, minutes, jobname):
    nodelist = ",".join(f"amd-mi355x-{n}" for n in nodes)
    cmd = (f"salloc --no-shell -p XAI -N{len(nodes)} -w {nodelist} "
           f"--gres=gpu:8 -t {minutes} -J {jobname} 2>&1")
    p = c.remote(cmd)
    if c.dry:
        return "DRYRUN-JOBID", nodelist
    m = re.search(r"Granted job allocation (\d+)", p.stdout)
    if not m:
        raise RuntimeError(f"salloc gave no jobid:\n{p.stdout}")
    return m.group(1), nodelist


def release(c, jobid):
    if jobid and jobid != "DRYRUN-JOBID":
        c.remote(f"scancel {jobid}", check=False)
        log(f"released allocation {jobid}")
    elif c.dry:
        log("dry-run: would scancel the allocation")


# ------------------------------------------------------- remote preparation

def node_ip_map(c):
    """servers.txt on shared: lines '<ip> #<node>' (possibly commented)."""
    p = c.remote(f"cat {REMOTE_TOOL}/servers.txt", quiet=True)
    if c.dry:
        return {n: f"172.30.160.{n}" for n in range(1, 10)}
    out = {}
    for line in p.stdout.splitlines():
        m = re.match(r"#?\s*([\d.]+)\s*#(\d+)", line.strip())
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def write_servers_file(c, nodes, remote_dir, host=SLURM_HOST):
    ips = node_ip_map(c)
    missing = [n for n in nodes if n not in ips]
    if missing and not c.dry:
        raise RuntimeError(f"no FE IP known for nodes {missing} in "
                           f"{REMOTE_TOOL}/servers.txt")
    content = "\\n".join(f"{ips.get(n, '?')} #{n}" for n in nodes)
    path = f"{remote_dir}/servers.txt"
    c.remote(f"mkdir -p {remote_dir} && printf '{content}\\n' > {path}", host=host)
    return path


def pick_exec_node(c, booked):
    """The search driver (rccl_sweep.py) runs on one node over ssh and needs
    python3 with tabulate+yaml - not every node has them (node 5 does not,
    2026-08-24 pilot). Probe the booked nodes first, then healthy fallbacks;
    the exec node only drives, so it need not be part of the allocation."""
    # 2026-09-08: booked nodes ONLY - the benchmark is an srun step inside the
    # allocation and a non-booked exec node makes every step fail in 2s
    # (sanity-check-1 attempt 1: node 5 lacked tabulate, fell back to node 7)
    for n in booked:
        if has_sweep_python(c, n):
            log(f"exec node: amd-mi355x-{n} (python env OK)")
            return n
    raise RuntimeError(f"no booked node {booked} has a python3 with "
                       "tabulate+yaml+colorama - pass --nodes with one that "
                       "does (1,3,8,9 verified 2026-09-08)")


def sync_tool(c):
    """Push the tool files a live run executes remotely. Refuses while any
    sweep/validate process is running (a mid-flight redeploy once killed a
    sweep via a stale file handle)."""
    p = c.remote("pgrep -af 'rccl_sweep.py|validate_tuner_config.py|ab_run.py'"
                 " | grep -v grep | wc -l", check=False, quiet=True)
    if not c.dry and p.stdout.strip() not in ("", "0"):
        raise RuntimeError("remote sweep/validate processes are running - "
                           "refusing to sync the tool over them")
    # the full remote-executed stack, not only the orchestrators: rccl_sweep
    # and its parser/db/executor run ON the exec node, so an unsynced parser
    # silently serves stale logic there (the 2026-08-26 alltoall N/A fix was
    # live locally but every remote run still produced zero metrics)
    # 2026-09-08: + the container runtime pieces added 2026-09-06 (container_run,
    # sweep_config.yaml with runtime.mode) and the conf generator's package
    for f in ("adaptive_search.py", "validate_tuner_config.py", "ab_run.py",
              "merge_metrics.py", "optimize_metrics.py",
              "generate_tuner_config.py", "rccl_sweep.py", "sweep_parser.py",
              "sweep_executor.py", "sweep_db.py", "container_run.py",
              "sweep_config.yaml", "autotune/__init__.py",
              "autotune/config_generator.py"):
        c.local(["scp", "-o", "BatchMode=yes", str(TOOL / f),
                 f"{SLURM_HOST}:{REMOTE_TOOL}/{f}"], quiet=True)
    log("tool synced to shared")


# ------------------------------------------------------------------- phases

def results_dir(name):
    """results-tuning/<date>-<N>-<name>, N = next run number of the DAY
    (CLAUDE.md: N starts at 1 each day and increments per run, so the folders
    sort in run order). Was: N counted only same-name dirs (2026-09-08)."""
    date = datetime.date.today().isoformat()
    taken = [int(m.group(1)) for d in RESULTS.glob(f"{date}-*")
             for m in [re.match(rf"{date}-(\d+)-", d.name)] if m]
    n = max(taken, default=0) + 1
    d = RESULTS / f"{date}-{n}-{name}"
    d.mkdir(parents=True)
    return d


def phase_search(c, coll, scale, nodes, outdir, remote_base, exec_node,
                 jobid=None, data_base=None):
    """Adaptive live search -> emitted optimized csv -> tuner conf."""
    tag = f"{coll}_{scale}n"
    # unique per attempt: a reused remote dir leaves stale run_* dirs whose
    # metrics get concatenated into the parse (stage3a incident, 2026-08-24)
    # outputs on the exec node's local disk (see DATA_ROOT); remote_base (shared)
    # is kept only as the fallback for callers that pass no data_base
    remote_out = f"{data_base or remote_base}/{tag}-{int(time.time())}"
    servers = write_servers_file(c, nodes[:scale], remote_out,
                                 host=f"amd-mi355x-{exec_node}")
    local_out = outdir / tag
    opt_csv = local_out / f"{tag}_optimized.csv"
    conf = outdir / f"{tag}.conf"
    args_ = [sys.executable, TOOL / "adaptive_search.py", "live",
             "--nodes", scale, "--grid", POLICY["grid"],
             "--anchors", POLICY["anchors"], "--margin", POLICY["margin"],
             "--tol", POLICY["tol"], "--repeat-policy", "3",
             "--collective", coll,
             "--exec-node", f"amd-mi355x-{exec_node}",
             "--remote-tool", REMOTE_TOOL, "--remote-out", remote_out,
             "--servers-file", servers, "--my-path", MY_PATH,
             "--local-out", local_out, "--default-runs", "3",
             "--emit-optimized", opt_csv]
    for k, flag in (("combos", "--combos"), ("min_size", "--min-size"), ("max_size", "--max-size"), ("image", "--image")):
        if EXTRA.get(k):
            args_ += [flag, str(EXTRA[k])]
    if jobid:
        # 2026-09-08: benchmarks run as srun steps INSIDE our allocation
        # (killed with it, never orphaned on a released node), not bare ssh
        args_ += ["--jobid", jobid, "--slurm-host", SLURM_HOST]
    c.local(args_, timeout=4 * 3600)
    if not c.dry:
        with open(opt_csv) as f:
            n_rows = sum(1 for _ in f) - 1
        if n_rows <= 0:
            raise RuntimeError(
                f"search produced ZERO winners ({opt_csv}) - every live run "
                f"failed; see {local_out}/live_runs.log. Refusing to continue "
                f"to an empty config.")
        log(f"search {tag}: {n_rows} per-size winners")
    # alltoall has no tuner-plugin deployment (p2p path, plugin never
    # consulted): the shippable artifact is one global channel env setting,
    # picked from the search's own data and A/B'd as an env arm.
    if coll == "alltoall":
        if c.dry:
            log(f"dry-run: would pick a global channel for {tag} from "
                f"{local_out}/adaptive_report.json")
            return conf
        return alltoall_env_conf(local_out / "adaptive_report.json",
                                 conf, scale)
    # one rule PER SIZE into the A/B: every size is judged alone, nothing
    # rides in on a neighbour's P(sup). Merging happens only after validation.
    c.local([sys.executable, TOOL / "generate_tuner_config.py", opt_csv,
             "-o", conf, "--include-algo-proto", "--no-merge"])
    return conf


def alltoall_env_conf(report_path, conf_path, scale):
    """Pick ONE channel count for alltoall from the search's own data and
    write it as a single-rule conf for the env-arm A/B.

    Env vars are one value per launch, so unlike the plugin collectives there
    is no per-size deployment: the candidate that maximises the median
    per-size gain over the same-session default runs wins. Candidates whose
    search medians already show a >2% regression at any size are considered
    only if no clean candidate exists (the A/B landmine gate still judges
    whatever is sent). Returns None - and leaves a SEARCH-VERDICT marker -
    when no candidate beats the default at all, which is a legitimate
    'defaults win' outcome, not a failure."""
    rep = json.loads(Path(report_path).read_text())
    defaults = rep.get("default_busbw") or []
    if not defaults:
        raise RuntimeError(f"{report_path}: no default runs recorded - "
                           f"cannot rank alltoall candidates")
    sizes = sorted({int(s) for d in defaults for s in d})
    def_med = {s: statistics.median([d[str(s)] for d in defaults
                                     if str(s) in d]) for s in sizes}
    scored = []          # (median_gain_pct, landmined?, channels, n_sizes)
    for cfg_s, per_size in rep["evals"].items():
        ch = int(cfg_s.split("/")[-1])
        gains = []
        for s_str, vals in per_size.items():
            vals = [v for v in vals if v is not None]
            s = int(s_str)
            if not vals or def_med.get(s, 0) <= 0:
                continue
            gains.append((statistics.median(vals) / def_med[s] - 1) * 100)
        if gains:
            scored.append((statistics.median(gains),
                           min(gains) < -2.0, ch, len(gains)))
    if not scored:
        raise RuntimeError(f"{report_path}: no evaluated candidate has data")
    clean = [x for x in scored if not x[1]]
    score, landmined, ch, n = max(clean or scored)
    if score <= 0:
        marker = conf_path.parent / (conf_path.stem + ".SEARCH-VERDICT.txt")
        marker.write_text(
            f"defaults win at the search stage: best candidate ch={ch} "
            f"scored {score:+.2f}% median gain over {n} sizes vs the "
            f"same-session default runs ({report_path}). No config to ship, "
            f"nothing to A/B.\n")
        log(f"alltoall {scale}n: defaults win in search data "
            f"(best ch={ch} at {score:+.2f}%) - no conf, skipping A/B")
        return None
    header = ("collective_type,min_bytes,max_bytes,algorithm,protocol,"
              "channels,nNodes,nRanks,numPipeOps,regBuff")
    rule = f"alltoall,4096,536870912,none,none,{ch},{scale},{scale * 8},-1,-1"
    conf_path.write_text(
        f"# env-arm conf: deployed as NCCL_MIN_NCHANNELS/NCCL_MAX_NCHANNELS="
        f"{ch}, not via the tuner plugin (alltoall runs on the p2p path).\n"
        f"# picked from {report_path.name}: median gain {score:+.2f}% over "
        f"{n} sizes vs same-session defaults"
        f"{' (landmined in search data, A/B decides)' if landmined else ''}.\n"
        f"{header}\n{rule}\n")
    log(f"alltoall {scale}n: candidate ch={ch} ({score:+.2f}% median gain) "
        f"-> env-arm conf {conf_path.name}")
    return conf_path


def phase_ab(c, confs, jobid, nodes, repeats, remote_base, outdir,
             ab_retries=3, data_base=None, exec_node=None):
    """Run the A/B batch remotely via ab_run.py and wait for its summary."""
    remote_confs = f"{remote_base}/confs"
    c.remote(f"mkdir -p {remote_confs}")
    for conf in confs:
        c.local(["scp", "-o", "BatchMode=yes", str(conf),
                 f"{SLURM_HOST}:{remote_confs}/"], quiet=True)
    nodelist = ",".join(f"amd-mi355x-{n}" for n in nodes)
    conf_args = " ".join(f"{remote_confs}/{Path(cf).name}" for cf in confs)
    ab_log = f"{remote_base}/ab_run.log"   # written by dn on the Slurm host: shared is fine
    ab_out = f"{data_base or remote_base}/ab_out"   # validator dbg logs: local disk (DATA_ROOT)
    data_host = f"amd-mi355x-{exec_node}" if exec_node else SLURM_HOST
    c.remote(f"mkdir -p {ab_out}", host=data_host)
    # the launch ssh can hang even though the remote process detaches fine
    # (observed 2026-08-24: 600s TimeoutExpired killed the whole run and the
    # finally-release orphaned a live ab_run against a dead allocation).
    # Short client-side timeout, no check - then verify the launch by the log.
    try:
        # 2026-09-08: the validator's container runtime runs docker on the
        # host it is started from, so ab_run must execute ON the booked node:
        # launched as an srun step inside the allocation (was: bare on the
        # Slurm login node, which is only right for the srun-based bare path)
        inner = (f"cd {SHARED} && python3 {REMOTE_TOOL}/ab_run.py "
                 f"--jobid {jobid} --nodelist {nodelist} --repeats {repeats} "
                 f"--retries {ab_retries} "
                 f"{ab_extra_args()} "
                 f"--outdir {ab_out} {conf_args}")
        c.remote(f"setsid nohup srun --jobid={jobid} -N1 -w {nodelist.split(',')[0]} "
                 f"bash -c {shlex.quote(inner)} "
                 f"> {ab_log} 2>&1 < /dev/null & echo started",
                 timeout=30, check=False)
    except subprocess.TimeoutExpired:
        pass
    if not c.dry:
        time.sleep(5)
        p = c.remote(f"pgrep -c -f 'ab_run.py --jobid {jobid}'", check=False,
                     quiet=True)
        if p.stdout.strip() in ("", "0"):
            raise RuntimeError("A/B batch failed to start - see " + ab_log)
        log("A/B batch confirmed running")
    if c.dry:
        log("dry-run: would poll the A/B log until its summary appears")
        return
    deadline = time.time() + 3 * 3600
    seen = ""
    while time.time() < deadline:
        time.sleep(60)
        p = c.remote(f"tail -40 {ab_log}", check=False, quiet=True)
        for line in p.stdout.splitlines():
            if line.startswith("[") and "rc=" in line and line not in seen:
                log("ab: " + line)
                seen += line + "\n"
        if "\nsummary:" in "\n" + p.stdout:
            log("ab batch finished")
            break
    else:
        raise RuntimeError("A/B batch did not finish within 3h - "
                           "allocation will be released; inspect " + ab_log)
    c.local(["bash", "-c",
             f"scp -o BatchMode=yes '{SLURM_HOST}:{remote_confs}/"
             f"*.validated.csv' {outdir}/ 2>/dev/null; "
             f"scp -o BatchMode=yes {SLURM_HOST}:{ab_log} {outdir}/ 2>/dev/null; "
             f"ssh -o BatchMode=yes {data_host} 'cd {ab_out}/.. && "
             f"tar czf /tmp/rt_ab.tgz ab_out/*/validate.log ab_out/*/per_size_stats.csv' && "
             f"scp -o BatchMode=yes {data_host}:/tmp/rt_ab.tgz {outdir}/ && "
             f"cd {outdir} && tar xzf rt_ab.tgz && rm rt_ab.tgz"],
            check=False)
    stats = list(outdir.glob("**/per_size_stats.csv"))
    for st in stats:
        with open(st) as f:
            hdr = f.readline()
        if "def_exec" not in hdr:
            raise RuntimeError(f"{st}: no exec-truth columns - channel values "
                               f"would fall back to the -A 1 plan; the remote "
                               f"tool copy is stale, re-run sync")
    if stats:
        log(f"exec truth verified from debug logs in {len(stats)} stats files "
            f"(channels sourced from channel{{Lo..Hi}}, not -A 1)")
    for v in outdir.glob("*.validated.csv"):
        n_rules, n_merged = merge_validated(
            v, outdir / (v.name.split(".")[0] + ".final.conf"))
        log(f"final config: {v.name.split('.')[0]}.final.conf "
            f"({n_rules} validated sizes -> {n_merged} rules)")


def merge_validated(validated_csv, out_conf):
    """Merge adjacent surviving per-size rules with identical settings into
    ranges - the FINAL config. Purely cosmetic: same match behaviour, fewer
    lines. Adjacency = next size is exactly double (the sweep's size ladder)."""
    header, rows = None, []
    for line in Path(validated_csv).read_text().splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if s.startswith("collective_type"):
            header = s
            continue
        p = s.split(",")
        rows.append((p[0], int(p[1]), int(p[2]), tuple(p[3:])))
    rows.sort(key=lambda r: (r[0], r[3], r[1]))
    merged = []
    for coll, lo, hi, rest in rows:
        if merged and merged[-1][0] == coll and merged[-1][3] == rest \
                and lo == merged[-1][2] * 2:
            merged[-1] = (coll, merged[-1][1], hi, rest)
        else:
            merged.append((coll, lo, hi, rest))
    merged.sort(key=lambda r: (r[0], r[1]))  # human-readable: by size
    out = [header or "collective_type,min_bytes,max_bytes,algorithm,protocol,"
                     "channels,nNodes,nRanks,numPipeOps,regBuff"]
    out += [f"{c},{lo},{hi},{','.join(r)}" for c, lo, hi, r in merged]
    Path(out_conf).write_text("\n".join(out) + "\n")
    return len(rows), len(merged)


def runlog_append(text):
    with open(RESULTS / "RUNLOG.md", "a") as f:
        f.write(text)


# -------------------------------------------------------------- subcommands

def cmd_run(args):
    global RESULTS
    if args.results_root:
        RESULTS = Path(args.results_root)
        if not RESULTS.is_dir():
            sys.exit(f"--results-root {RESULTS} does not exist")
    c = Cmd(args.dry_run)
    POLICY["grid"] = args.grid
    for k in ("combos", "min_size", "max_size", "image"):
        EXTRA[k] = getattr(args, k, None)
    colls = args.collectives.split(",")
    scales = [int(s) for s in args.scales.split(",")]
    for coll in colls:
        if coll not in COLLECTIVES:
            sys.exit(f"unsupported collective '{coll}'; supported: {COLLECTIVES}")
    pairs = [(coll, s) for coll in colls for s in scales]
    need = max(scales)
    outdir = RESULTS / "DRY-RUN" if args.dry_run else results_dir(args.name)
    remote_base = f"{SHARED}/rccl-tune-{datetime.date.today().isoformat()}-{args.name}"
    data_base = f"{DATA_ROOT}/rccl-tune-{datetime.date.today().isoformat()}-{args.name}"
    log(f"plan: {pairs}; {need} nodes; results -> {outdir}")

    co_tenant_report(c)
    if args.nodes:
        nodes = sorted(int(n) for n in args.nodes.split(","))
        bad = [n for n in nodes if n in FORBIDDEN_NODES or n in BROKEN_NODES]
        if bad:
            sys.exit(f"nodes {bad} are forbidden/broken - refusing")
        if len(nodes) < need:
            sys.exit(f"--nodes gives {len(nodes)} nodes, plan needs {need}")
    else:
        nodes = pick_nodes(c, need)
    # 2026-09-08: container runtime is slower per run (docker start) and the
    # A/B is 9 reps x 2 arms x every rule; 85 min timed out in planning.
    minutes = min(480, 120 + 60 * len(pairs))
    if getattr(args, "minutes", None):
        minutes = args.minutes
    jobid = None
    t0 = time.time()
    try:
        jobid, nodelist = book(c, nodes, minutes, f"ylerman-{args.name}")
        log(f"allocation {jobid} on {nodelist} for {minutes}m")
        sync_tool(c)
        exec_node = pick_exec_node(c, nodes)
        confs = []
        for coll, scale in pairs:
            log(f"=== search {coll} {scale}n ===")
            cf = phase_search(c, coll, scale, nodes, outdir,
                              remote_base, exec_node, jobid=jobid,
                              data_base=data_base)
            if cf is not None:   # None = defaults won at the search stage
                confs.append(cf)
        if confs:
            log(f"=== A/B: {len(confs)} confs, {args.repeats} repeats ===")
            phase_ab(c, confs, jobid, nodes, args.repeats, remote_base,
                     outdir, args.ab_retries, data_base=data_base,
                     exec_node=exec_node)
        else:
            log("no confs to A/B - every pair resolved to 'defaults win' "
                "at the search stage")
    finally:
        release(c, jobid)
    dur = int(time.time() - t0)
    if not args.dry_run:
        runlog_append(
            f"\n| {utc()} | {dur}s wall | {jobid} (own alloc, released) | "
            f"{nodelist} | {args.scales} | `rccl_tune.py run --collectives "
            f"{args.collectives} --scales {args.scales} --repeats "
            f"{args.repeats} --grid {POLICY['grid']}` (policy: tol {POLICY['tol']}, anchors "
            f"{POLICY['anchors']}, margin {POLICY['margin']}, median-of-3) | "
            f"sweep_config.yaml env | `results-tuning/{outdir.name}` (remote "
            f"`{remote_base}`, raw on node `{data_base}`) | see per-scale validate.log + "
            f".validated.csv |\n")
        log(f"done in {dur}s; results in {outdir}")
    return 0


def cmd_status(args):
    c = Cmd(False)
    p = c.remote("squeue -u dn -h -o '%i %j %T %M %N'", check=False,
                 quiet=True)
    print("our slurm jobs:")
    print(p.stdout or "  none")
    p = c.remote("pgrep -af 'ab_run.py|adaptive_search|rccl_sweep' "
                 "| grep -v grep", check=False, quiet=True)
    print("remote tool processes:")
    print(p.stdout or "  none")
    return 0


def cmd_report(args):
    pages = sorted(RESULTS.glob("*-pages/build_pages.py"))
    if not pages:
        sys.exit("no pages builder found under results-tuning/*-pages/")
    subprocess.run([sys.executable, str(pages[-1])], check=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("run", help="full loop: book, search, config, A/B")
    pr.add_argument("--collectives", required=True,
                    help=f"comma list from {COLLECTIVES}")
    pr.add_argument("--scales", required=True, help="comma list from 1,2,3")
    pr.add_argument("--name", default="tune",
                    help="short results-dir suffix (default: tune)")
    pr.add_argument("--repeats", type=int, default=9,
                    help="A/B repeats per arm (user-approved default 9)")
    pr.add_argument("--ab-retries", type=int, default=4,
                    help="A/B attempts per conf before giving up (preflight "
                         "refusals and noisy runs both consume attempts)")
    pr.add_argument("--grid", default=POLICY["grid"],
                    help="channel grid for the search (measurement parameter; "
                         f"default = validated policy {POLICY['grid']})")
    pr.add_argument("--combos", default=None, help="search combos ALGO:PROTO,... (default: tool policy per node count)")
    pr.add_argument("--min-size", default=None, help="size window start for search AND A/B, e.g. 128M (default 4K)")
    pr.add_argument("--max-size", default=None, help="size window end, e.g. 2G (default 512M)")
    pr.add_argument("--image", default=None, help="container image for search AND A/B (default: sweep_config / validator default)")
    pr.add_argument("--minutes", type=int, default=None, help="booking length in minutes (default: 120 + 60 per collective/scale pair, max 480)")
    pr.add_argument("--nodes", default=None,
                    help="override node pick, e.g. 5,8 (still blacklisted-checked)")
    pr.add_argument("--dry-run", action="store_true",
                    help="print every command; touch nothing")
    pr.add_argument("--results-root", default=None,
                    help="results-tuning dir to write into (default: this "
                         "repo's; pass the main worktree's when the tool "
                         "branch differs from the results branch)")
    sub.add_parser("status", help="allocations + remote tool processes")
    sub.add_parser("report", help="rebuild the results site")
    args = ap.parse_args()
    return {"run": cmd_run, "status": cmd_status,
            "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
