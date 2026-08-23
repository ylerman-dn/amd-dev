#!/usr/bin/env python3
"""Orchestrate live A/B validation of tuner configs - the launcher that was
previously composed by hand per scale (2026-08-19 session).

For each .conf given: infer collective and node count from its rules, build the
validate_tuner_config.py command with the project-standard parameters (the
2026-08-19 published ones), run it, and apply the retry policy on the two
non-verdict exits:

  exit 2 (run completed but too noisy)   -> retry, counts toward --retries
  exit 3 (preflight refused the machine) -> retry, counts toward --retries
  exit 0/1 (verdict reached)             -> done (1 = "no rule survived",
                                            a legitimate verdict, not a failure)

--dry-run prints every command without launching anything.

Booking stays human: pass --jobid of a held allocation (salloc --no-shell).
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

TOOL = Path(__file__).parent
# tuner-config collective names -> benchmark binaries
BINARY = {"allreduce": "all_reduce_perf", "broadcast": "broadcast_perf",
          "reduce": "reduce_perf", "allgather": "all_gather_perf",
          "reducescatter": "reduce_scatter_perf", "alltoall": "alltoall_perf"}
# 2026-08-19 published parameters (results-tuning/2026-08-19-1-bcmn-ab/SUMMARY.md)
WARMUP_RUNS = {1: 4, 2: 4, 3: 8}


def conf_target(path):
    """(collective, nNodes) from the conf's rules; refuses a mixed file."""
    seen = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("collective_type"):
            continue
        p = line.split(",")
        seen.add((p[0], int(p[6])))
    if len(seen) != 1:
        raise SystemExit(f"{path}: expected one (collective, nNodes), got {sorted(seen)}")
    return seen.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("confs", nargs="+", help="tuner .conf files to validate")
    ap.add_argument("--jobid", required=True, help="held Slurm allocation")
    ap.add_argument("--nodelist", required=True,
                    help="nodes inside the allocation, comma separated; scales "
                         "use a prefix of this list (pin them for the session)")
    ap.add_argument("--my-path", default="/opt/shared/ylerman/GPU-107/bin")
    ap.add_argument("--plugin", default="librccl-tunerv4-dn.so")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--retries", type=int, default=3,
                    help="max attempts per conf on exit 2/3")
    ap.add_argument("--outdir", required=True,
                    help="results dir for logs (results-tuning/<date>-<N>-<name>/)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nodes_all = args.nodelist.split(",")
    outdir = Path(args.outdir)
    results = []
    for conf in args.confs:
        coll, n = conf_target(conf)
        if n > len(nodes_all):
            raise SystemExit(f"{conf}: needs {n} nodes, only {len(nodes_all)} given")
        stem = f"{coll}_{n}n"
        binary = f"{args.my_path}/{BINARY[coll]}"
        for attempt in range(1, args.retries + 1):
            logdir = outdir / (stem if attempt == 1 else f"{stem}_retry{attempt}")
            cmd = [sys.executable, str(TOOL / "validate_tuner_config.py"),
                   "--config", conf, "--binary", binary,
                   "--plugin", args.plugin, "--plugin-must-fire",
                   "--jobid", args.jobid,
                   "--nodelist", ",".join(nodes_all[:n]),
                   "--repeats", str(args.repeats),
                   "--warmup-runs", str(WARMUP_RUNS[n]),
                   "--split-ranges",
                   "--min-bytes", "4096", "--max-bytes", "536870912",
                   "--iters", "20", "--warmup", "5",
                   "--logdir", str(logdir)]
            print(f"[{stem} attempt {attempt}] {' '.join(cmd)}", flush=True)
            if args.dry_run:
                results.append((stem, "dry-run", 0))
                break
            logdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(logdir / "validate.log", "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
            print(f"[{stem} attempt {attempt}] rc={rc} in {time.time()-t0:.0f}s",
                  flush=True)
            if rc in (0, 1):  # verdict reached
                results.append((stem, "verdict", rc))
                break
            results.append((stem, "noisy" if rc == 2 else "preflight-fail", rc))
        else:
            print(f"[{stem}] gave up after {args.retries} attempts", flush=True)

    print("\nsummary:")
    for stem, kind, rc in results:
        print(f"  {stem:22s} {kind} (rc={rc})")
    bad = [r for r in results if r[1] not in ("verdict", "dry-run")]
    return 1 if bad and all(k != "verdict" for _, k, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
