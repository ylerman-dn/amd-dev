#!/usr/bin/env python3
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

Node policy: nodes 2 (orchestrator) and 9 (no ssh key) are never used; the
broken list starts with node 4 (fabric fault 2026-08-18) and can be edited in
BROKEN_NODES below. Booking prefers previously-published node sets.
"""

import argparse
import csv
import datetime
import re
import subprocess
import sys
import time
from pathlib import Path

TOOL = Path(__file__).resolve().parent
REPO = TOOL.parent.parent
RESULTS = REPO / "results-tuning"
SLURM_HOST = "amd-mi355x-1"
SHARED = "/opt/shared/ylerman/GPU-107"
REMOTE_TOOL = f"{SHARED}/rccl-sweep-optuna"
PLUGIN = f"{SHARED}/ab-tuner-test/librccl-tunerv4-dn.so"
MY_PATH = f"{SHARED}/bin"

FORBIDDEN_NODES = {2, 9}          # orchestrator / no ssh key
BROKEN_NODES = {4}                # fabric fault 2026-08-18; edit when repaired
NODE_PREFERENCE = [5, 8, 3, 7, 6, 1]

# the validated search policy (findings/07); changing any of these is a
# measurement-parameter change and needs explicit user approval
POLICY = dict(anchors="1,8,24,48", margin="15", tol="0.5",
              grid="1,2,4,8,16,24,32,40,48")

COLLECTIVES = ["all_reduce", "broadcast", "reduce", "all_gather",
               "reduce_scatter"]


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


def pick_nodes(c, k):
    avail = idle_nodes(c)
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


def write_servers_file(c, nodes, remote_dir):
    ips = node_ip_map(c)
    missing = [n for n in nodes if n not in ips]
    if missing and not c.dry:
        raise RuntimeError(f"no FE IP known for nodes {missing} in "
                           f"{REMOTE_TOOL}/servers.txt")
    content = "\\n".join(f"{ips.get(n, '?')} #{n}" for n in nodes)
    path = f"{remote_dir}/servers.txt"
    c.remote(f"mkdir -p {remote_dir} && printf '{content}\\n' > {path}")
    return path


def sync_tool(c):
    """Push the tool files a live run executes remotely. Refuses while any
    sweep/validate process is running (a mid-flight redeploy once killed a
    sweep via a stale file handle)."""
    p = c.remote("pgrep -af 'rccl_sweep.py|validate_tuner_config.py|ab_run.py'"
                 " | grep -v grep | wc -l", check=False, quiet=True)
    if not c.dry and p.stdout.strip() not in ("", "0"):
        raise RuntimeError("remote sweep/validate processes are running - "
                           "refusing to sync the tool over them")
    for f in ("adaptive_search.py", "validate_tuner_config.py", "ab_run.py",
              "merge_metrics.py", "optimize_metrics.py",
              "generate_tuner_config.py"):
        c.local(["scp", "-o", "BatchMode=yes", str(TOOL / f),
                 f"{SLURM_HOST}:{REMOTE_TOOL}/{f}"], quiet=True)
    log("tool synced to shared")


# ------------------------------------------------------------------- phases

def results_dir(name):
    date = datetime.date.today().isoformat()
    n = 1
    while (RESULTS / f"{date}-{n}-{name}").exists():
        n += 1
    d = RESULTS / f"{date}-{n}-{name}"
    d.mkdir(parents=True)
    return d


def phase_search(c, coll, scale, nodes, outdir, remote_base):
    """Adaptive live search -> emitted optimized csv -> tuner conf."""
    tag = f"{coll}_{scale}n"
    remote_out = f"{remote_base}/{tag}"
    servers = write_servers_file(c, nodes[:scale], remote_out)
    local_out = outdir / tag
    opt_csv = local_out / f"{tag}_optimized.csv"
    conf = outdir / f"{tag}.conf"
    args_ = [sys.executable, TOOL / "adaptive_search.py", "live",
             "--nodes", scale, "--grid", POLICY["grid"],
             "--anchors", POLICY["anchors"], "--margin", POLICY["margin"],
             "--tol", POLICY["tol"], "--repeat-policy", "3",
             "--collective", coll,
             "--exec-node", f"amd-mi355x-{nodes[0]}",
             "--remote-tool", REMOTE_TOOL, "--remote-out", remote_out,
             "--servers-file", servers, "--my-path", MY_PATH,
             "--local-out", local_out, "--default-runs", "3",
             "--emit-optimized", opt_csv]
    c.local(args_, timeout=4 * 3600)
    c.local([sys.executable, TOOL / "generate_tuner_config.py", opt_csv,
             "-o", conf, "--include-algo-proto"])
    return conf


def phase_ab(c, confs, jobid, nodes, repeats, remote_base, outdir):
    """Run the A/B batch remotely via ab_run.py and wait for its summary."""
    remote_confs = f"{remote_base}/confs"
    c.remote(f"mkdir -p {remote_confs}")
    for conf in confs:
        c.local(["scp", "-o", "BatchMode=yes", str(conf),
                 f"{SLURM_HOST}:{remote_confs}/"], quiet=True)
    nodelist = ",".join(f"amd-mi355x-{n}" for n in nodes)
    conf_args = " ".join(f"{remote_confs}/{Path(cf).name}" for cf in confs)
    ab_log = f"{remote_base}/ab_run.log"
    c.remote(f"cd {SHARED} && setsid nohup python3 {REMOTE_TOOL}/ab_run.py "
             f"--jobid {jobid} --nodelist {nodelist} --repeats {repeats} "
             f"--outdir {remote_base}/ab_out {conf_args} "
             f"> {ab_log} 2>&1 < /dev/null & echo started")
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
             f"ssh -o BatchMode=yes {SLURM_HOST} 'cd {remote_base} && "
             f"tar czf /tmp/rt_ab.tgz ab_run.log ab_out/*/validate.log ab_out/*/per_size_stats.csv' && "
             f"scp -o BatchMode=yes {SLURM_HOST}:/tmp/rt_ab.tgz {outdir}/ && "
             f"cd {outdir} && tar xzf rt_ab.tgz && rm rt_ab.tgz"],
            check=False)


def runlog_append(text):
    with open(RESULTS / "RUNLOG.md", "a") as f:
        f.write(text)


# -------------------------------------------------------------- subcommands

def cmd_run(args):
    c = Cmd(args.dry_run)
    colls = args.collectives.split(",")
    scales = [int(s) for s in args.scales.split(",")]
    for coll in colls:
        if coll not in COLLECTIVES:
            sys.exit(f"unsupported collective '{coll}' "
                     f"(alltoall: channels-only sweep, not wired yet; "
                     f"supported: {COLLECTIVES})")
    pairs = [(coll, s) for coll in colls for s in scales]
    need = max(scales)
    outdir = RESULTS / "DRY-RUN" if args.dry_run else results_dir(args.name)
    remote_base = f"{SHARED}/rccl-tune-{datetime.date.today().isoformat()}-{args.name}"
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
    minutes = min(480, 45 + 40 * len(pairs))
    jobid = None
    t0 = time.time()
    try:
        jobid, nodelist = book(c, nodes, minutes, f"ylerman-{args.name}")
        log(f"allocation {jobid} on {nodelist} for {minutes}m")
        sync_tool(c)
        confs = []
        for coll, scale in pairs:
            log(f"=== search {coll} {scale}n ===")
            confs.append(phase_search(c, coll, scale, nodes, outdir,
                                      remote_base))
        log(f"=== A/B: {len(confs)} confs, {args.repeats} repeats ===")
        phase_ab(c, confs, jobid, nodes, args.repeats, remote_base, outdir)
    finally:
        release(c, jobid)
    dur = int(time.time() - t0)
    if not args.dry_run:
        runlog_append(
            f"\n| {utc()} | {dur}s wall | {jobid} (own alloc, released) | "
            f"{nodelist} | {args.scales} | `rccl_tune.py run --collectives "
            f"{args.collectives} --scales {args.scales} --repeats "
            f"{args.repeats}` (policy: tol {POLICY['tol']}, anchors "
            f"{POLICY['anchors']}, margin {POLICY['margin']}, median-of-3) | "
            f"sweep_config.yaml env | `{outdir.relative_to(REPO)}` (remote "
            f"`{remote_base}`) | see per-scale validate.log + "
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
    pr.add_argument("--nodes", default=None,
                    help="override node pick, e.g. 5,8 (still blacklisted-checked)")
    pr.add_argument("--dry-run", action="store_true",
                    help="print every command; touch nothing")
    sub.add_parser("status", help="allocations + remote tool processes")
    sub.add_parser("report", help="rebuild the results site")
    args = ap.parse_args()
    return {"run": cmd_run, "status": cmd_status,
            "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
