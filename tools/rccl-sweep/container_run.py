"""Container launch helper for rccl-tests runs (GPU-107, 2026-09-06).

Why this exists: rules derived from rccl-tests are deployed under SGLang's container
(stock RCCL + the image's env, NCCL_MIN_NCHANNELS=112), launched as 8 processes.
The bare-metal path (srun + LD_PRELOAD of the team fork) measures a stack nothing
deploys on — the channel-unit trap and the 2026-09-03 regime findings both came from
that mismatch (results-tuning/2026-09-03-2-stackcmp, -3-regime). So the deployment-
faithful harness is: docker run <image> mpirun -np <ranks> ... <binary> -g 1.

Single-node only: mpirun inside one container cannot span nodes. Multi-node container
sweeps need per-node containers + a cross-node launcher — not implemented; callers
must reject num_nodes > 1 in container mode.
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

DEFAULT_IMAGE = "lmsysorg/sglang:v0.5.17-rocm720-mi35x"

# Mount points inside the container (fixed; callers translate paths with these).
BIN_MOUNT = "/opt/rccl-tests"
OUT_MOUNT = "/workspace/out"
TUNER_MOUNT = "/opt/rccl/tuner"
CONF_MOUNT = "/opt/rccl/conf"


def build_docker_mpirun(name: str,
                        image: str,
                        num_ranks: int,
                        bin_dir: str,
                        out_dir: str,
                        test_argv: List[str],
                        env_vars: Dict[str, str],
                        tuner_dir: Optional[str] = None,
                        conf_dir: Optional[str] = None,
                        bind_to: str = "numa") -> List[str]:
    """Build the full `docker run ... mpirun ...` argv for one rccl-tests run.

    test_argv: the binary basename + its flags, with paths already expressed in
    container terms (BIN_MOUNT/..., OUT_MOUNT/...). env_vars go through mpirun -x,
    which unlike `srun --export` does NOT split on commas, so comma-valued vars
    (NCCL_DEBUG_SUBSYS=INIT,TUNING,ENV) survive intact.
    """
    inner = ["mpirun", "--allow-run-as-root", "-np", str(num_ranks),
             "--bind-to", bind_to]
    for key, value in env_vars.items():
        inner += ["-x", f"{key}={value}"]
    inner += test_argv
    inner_str = " ".join(inner)

    cmd = ["docker", "run", "--rm", f"--name={name}",
           "--ipc=host", "--shm-size=16g", "--network=host", "--privileged",
           "--ulimit", "memlock=-1",
           "--cap-add=CAP_SYS_ADMIN", "--cap-add=IPC_LOCK", "--cap-add=SYS_PTRACE",
           "--security-opt", "seccomp=unconfined",
           "--device=/dev/kfd", "--device=/dev/dri",
           "-v", f"{os.path.abspath(bin_dir)}:{BIN_MOUNT}:ro",
           "-v", f"{os.path.abspath(out_dir)}:{OUT_MOUNT}",
           "-w", "/workspace"]
    if tuner_dir:
        cmd += ["-v", f"{os.path.abspath(tuner_dir)}:{TUNER_MOUNT}:ro"]
    if conf_dir and os.path.abspath(conf_dir) != os.path.abspath(tuner_dir or ""):
        cmd += ["-v", f"{os.path.abspath(conf_dir)}:{CONF_MOUNT}:ro"]
    cmd += ["--entrypoint=/bin/bash", image, "-c", inner_str]
    return cmd


def kill_container(name: str) -> None:
    """Kill a run's container after a timeout. `timeout`/proc.kill() only kills the
    docker CLIENT; a GPU-hung container survives it and blocks the device (observed
    2026-09-03: arm4_r1 hung 30 min past its timeout until docker kill)."""
    subprocess.run(["docker", "kill", name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
