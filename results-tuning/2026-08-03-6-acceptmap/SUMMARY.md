# Acceptance map per size + exotic-algo probes

Job 14048 (runs 1–11) and 14059 (REF rep2/rep3 rerun), `amd-mi355x-4`, 1 node, 8 ranks × 1 GPU,
all_reduce, 4K–512M f=2, channels unset. `-n 20 -w 5 -c 1 -A 1 -Z csv -X <file>`,
`NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE=..._%p.log` — the new standard measured-run config.

## Acceptance is size-invariant

The same 3 of 6 algo/proto pairs are honoured at **every one of the 18 sizes**. Requested → got:

| requested | 4K – 32M | 64M – 512M | verdict |
|---|---|---|---|
| RING / SIMPLE | RING/SIMPLE | `RING*`/SIMPLE | honoured |
| RING / LL | RING/LL | `RING*`/LL | honoured |
| TREE / LL | TREE/LL | TREE/LL | honoured |
| RING / LL128 | RING/**SIMPLE** | `RING*`/**SIMPLE** | proto dropped |
| TREE / SIMPLE | **RING**/SIMPLE | **`RING*`**/SIMPLE | algo dropped |
| TREE / LL128 | **RING**/**SIMPLE** | **`RING*`**/**SIMPLE** | both dropped |

**This refutes the hypothesis that motivated the run.** I expected acceptance to vary with size,
reasoning that RCCL's own default flips from TREE/LL to RING/SIMPLE at 512K. It does not — the
accepted set is constant. The earlier 256M/512M-only result generalised correctly after all, and
finding 03's scope widens from 2 sizes to all 18.

Two incidental confirmations: WarpSpeed applies with **LL** as well as SIMPLE (`RING*`/LL at
≥64M), and never with TREE — consistent with it being ring-only.

## Unavailable algorithms fail in two different ways

| requested | result |
|---|---|
| `NVLS` | silently substituted → RING/SIMPLE, exit 0 |
| `PAT` | silently substituted → RING/SIMPLE, exit 0 |
| `COLLNET_DIRECT` | **hard error**, exit 3, `invalid usage` |
| `COLLNET_CHAIN` | **hard error**, exit 3, `invalid usage` |

So a sweep hitting COLLNET **crashes**, while NVLS/PAT/LL128/TREE+SIMPLE **silently lie**. Both
matter and they need different handling: the first is loud and safe, the second is the dangerous
one. `tools/rccl-sweep/unsupported_combos.yaml` covers neither case.

## New reference baseline, verified

3 repeats, nothing forced, under the standard logging config:

- **0 selection mismatches** across the 3 repeats at all 18 sizes.
- **0 differences** vs the old `NCCL_DEBUG=VERSION` baseline (`2026-08-03-3-defaults`) — so
  switching to INFO→file did not perturb selection.
- Real channel counts from INFO, identical to the earlier diagnostic: 1, 1, 1, 2, 4, 8, 16, 16,
  32, 52, 52, 54, 56, 56, 222, 222, 223, 224.

This run supersedes `2026-08-03-3-defaults` as the comparable baseline, because it satisfies the
identical-logging-config rule that future A/B runs will use.

## Run losses — my error, not the cluster's

Job 14048 hit its **4-hour time limit** at 13:08:14Z (`sacct` reports `TIMEOUT`, exit 0:0 — not
`NODE_FAIL`). I allocated `-t 240` at 09:08 and did not track the expiry. Consequences:

- `REF_rep2` was killed mid-run (rc=143 = SIGTERM); it had completed all 18 sizes but was
  terminated before the CSV flushed.
- `REF_rep3` never started (`Slurm job 14048 has expired`).

Both were rerun on a fresh allocation of the **same node** (job 14059, `-t 480`), so all three
repeats are comparable. The COLLNET failures happened at 09:xx while the job was healthy and are
genuine results, not timeout artefacts.

## Files

10 CSVs + 13 stdout logs, `debug_files.tgz` and `ref_rerun_dbg.tgz` (per-rank INFO logs).
`REF_rep2.csv`/`REF_rep3.csv` are from the rerun; the timed-out originals were overwritten.
