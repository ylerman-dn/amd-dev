# What changed in RCCL between the two campaigns — evidence from the source history

Source: ROCm monorepo `github.com/ROCm/rocm-systems`, clone `/home/dn/ylerman/wg/rocm-systems` (fetched 2026-09-14, `origin/develop` at b70c089619 of 2026-09-13), path `projects/rccl/`.
RCCL moved into this monorepo in January 2026 ("Migrating rccl", standalone clone `/home/dn/ccl-gaps-24-05/repos/rccl` ends at 57e58688, 2026-01-22).
Old campaign RCCL: 2.27.7 at commit 0d2c4fd = "Fix rcclNetP2pPolicy issue (#2072) (#2094)", 2025-12-09 (standalone repo). New campaign RCCL: 2.30.4, library built 2026-09-10, 4-rule gfx950 table loaded (rank logs).
Everything below happened between those two points. PR links: `https://github.com/ROCm/rocm-systems/pull/<n>`.

## 1. MSCCL removed — PR #4340, commit 5ffe6a3dfb, Wenkai Du, 2026-04-13

- Commit body (verbatim): "MSCCL has been obsoleted by MSFT. Existing MSCCLPP integration was built on top of MSCCLPP and code path is stale due to lack of update and testing. ... Do not expect any impact as MSCCL and MSCCLPP have not been built for sometime."
- 80 files changed, 130,724 lines deleted: `src/device/msccl_kernel_impl.h`, `src/include/msccl/*`, `ext-src/mscclpp` submodule, `cmake/Findmscclpp_nccl_static.cmake`, ...
- CHANGELOG.md line 84 (section "RCCL 2.28.3 for ROCm 7.13"): "Removed MSCCL and MSCCL++ collective integration; legacy `mscclLoadAlgo`, `mscclRunAlgo`, and `mscclUnloadAlgo` APIs remain as no-ops for link compatibility."
- What we saw: `librccl.so` in the new image has 19 "msccl" strings (the no-op API names) vs 349 in 2.27.7, and no server log line for `RCCL_MSCCL_ENABLE` (HANDS.md 15:2xZ).
- Consequence for us: the 2026-09-03 finding "MSCCL on beats MSCCL off by 4.5% for gpt-oss decode" (RUNLOG 2026-09-03) describes a code path that no longer exists.

## 2. DDA = "Direct Data Access" all-reduce — PR #4958, commit c2c4b25f6d, Nusrat Islam, 2026-05-17

- Commit body (verbatim): "This PR adds direct algorithms for scaleup allreduce collective in RCCL. The work is inspired by the DDA allreduce algorithms in ctran/rcclx. The PR adds both one-shot and two-shot direct allreduce algorithms and brings significant speedup over existing RCCL allreduce. ... infrastructure for scaleup direct collectives using IPC with the user buffer being copied to a staging buffer allocated during init. ... tested using rccl test on MI350. ... up to 3x speedup over RCCL default allreduce algorithm." 20 files, +1360 lines.
- What it is (code): `src/include/algorithms/all_reduce/all_reduce_dda.h` (header: "Derived from Meta torchcomms comms/common/algorithms/all_reduce/all_reduce_dda.cuh"). Every rank copies its buffer into an IPC-shared staging buffer; after a GPU barrier each rank reads all peers' buffers directly over XGMI and reduces (one-shot "flat" kernel `ddaAllReduceFlatIpc` up to 256 KB, two-shot "tree" = reduce-scatter + all-gather above, `kDdaFlatTreeThresholdBytes = 1<<18` in `src/dda_all_reduce_ipc.cu`). No ring/tree pipeline, no channels, no protocol choice: the classic algo/proto/channel machinery, and therefore any tuner, is bypassed.
- Gating (`src/collectives.cc` 127-146, 440-442): `RCCL_DDA_ENABLE` (default 1), `RCCL_DDA_THRESHOLD` (default 64 MiB on gfx950; gfx942 fixed 8 MiB for all_reduce), only gfx942/gfx950, `nRanks >= 8`, not inside a group, not with `NCCL_LAUNCH_ORDER_IMPLICIT`, not when symmetric memory is active. Eligibility (`src/dda_all_reduce_ipc.cu` 116-160): single node, exactly 8 ranks, `ncclSum`, fp32/fp16/bf16, 16-byte aligned sizes. Every SGLang TP-8 decode all_reduce meets all of this.
- Follow-ups in the same window: #6663 (2026-06-03) gate on NCCL_LAUNCH_ORDER_IMPLICIT; #6783 (2026-06-05) skip DDA init for directMode/MNNVL; #6549 (2026-06-10) "Add optimized scaleup RS, AG, and A2A" (DDA for reduce-scatter, all-gather, all-to-all: `src/dda_*_ipc.cu`); arch guard (2026-06-16). CHANGELOG 2.30.4 lines 20 and 57 mention DDA only through these fixes; the feature itself has no CHANGELOG entry.
- What we saw: `ncclDdaIpcCommInit: scratch N bytes, IpcGpuBarrier ...` in every rank log; 0 tuner hits with RCCL as shipped; all 8 sc2 rules hit with `RCCL_DDA_ENABLE=0` (HANDS.md 16:03Z / 16:07Z; SUMMARY.md section 1).

## 3. Built-in CSV tuner with an MI355X table — PR #4503, commit 2234545028, Mustafa Abduljabbar, 2026-04-15

- Commit body (verbatim, excerpt): "Currently, customizing RCCL's algorithm selection ... requires either modifying RCCL source code (e.g., topoGetAlgoInfo) and rebuilding the library, or developing and deploying a separate tuner plugin .so file. ... This PR adds a built-in CSV tuner compiled directly into librccl.so that enables runtime algorithm tuning through a simple CSV config file. ... activated when no external tuner plugin is present. Config files are auto-discovered ... (env var, library directory, installed share path, ROCM_PATH)." 9 files, +790 lines: `src/plugin/tuner/csv_tuner.cc`, `tuner/rccl_tuner_gfx950.csv`, `tuner/README.md`.
- What it means: the same mechanism and the same CSV format as the NCCL example tuner plugin our `librccl-tunerv4-dn.so` is built from (`colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks[,numPipeOps][,regBuff]`, `tuner/README.md`), but compiled into librccl and shipped with an AMD-authored table per GPU architecture. Selection order (`src/plugin/tuner.cc` 63-84): if `NCCL_TUNER_PLUGIN` is set, load that plugin (or `none` = no tuner); otherwise try `libnccl-tuner.so`; otherwise the built-in CSV tuner with `rccl_tuner_<arch>.csv`. So an external plugin replaces AMD's table completely.
- The table in the PR (and in our image, "Allocated memory for 4 configurations"):
  ```
  allreduce,0,16383,tree,ll,1,1,8,-1,-1
  allreduce,16384,524287,ring,ll,-1,1,8,-1,-1
  allreduce,524288,1048575,ring,ll,32,1,8,-1,-1
  allreduce,1048576,2097000,ring,ll,56,1,8,-1,-1
  ```
  PR #9442 (Pedram Alizadeh, 2026-08-11, "Fixing MI350X 2-node allreduce regression", JIRA AICOMRCCL-1642) added three 2-node/16-rank rules for 32 MB-224 MB (ring/ll128/64, tree/simple/64, ring/simple/48; "up to 1.23x"). Our image does not have them (4 rules loaded), so its RCCL branched before 2026-08-11.
- CHANGELOG.md line 69 (2.28.3 for ROCm 7.13): "Added built-in CSV tuner for runtime algorithm/protocol/channel selection without rebuilds."
- What we saw: `Using built-in CSV tuner, config: .../share/rccl/tuner/rccl_tuner_gfx950.csv` in the no-plugin server; with our plugin loaded the 448K all_reduce switches from ring/LL to ring/SIMPLE and DeepSeek d32 loses 15% (SUMMARY.md section 7).

## 4. Order of events

| date | event | commit / PR |
|---|---|---|
| 2025-12-09 | RCCL 2.27.7 commit used by the old campaign's image | 0d2c4fd (standalone repo) |
| 2026-01-22 | RCCL migrated into rocm-systems | 57e58688 |
| 2026-04-13 | MSCCL / MSCCL++ removed | #4340 5ffe6a3dfb |
| 2026-04-15 | built-in CSV tuner + gfx950 table | #4503 2234545028 |
| 2026-05-17 | DDA direct all-reduce (single node, 8 ranks) | #4958 c2c4b25f6d |
| 2026-06-10 | DDA reduce-scatter / all-gather / all-to-all | #6549 3dc4fb138e |
| 2026-06-16 | NCCL 2.30.4 sync | #6837 b40d918f58 |
| 2026-08-11 | 2-node rules added to the gfx950 table (not in our image) | #9442 d5c6c83045 |
| 2026-09-10 | image `v0.5.19-rocm10-mi35x` built with RCCL 2.30.4 | docker inspect |

Note on the CHANGELOG version labels: the file says "RCCL 2.30.4 for ROCm 7.14.0" and "2.28.3 for ROCm 7.13"; the image reports ROCm "10.0.0.0-9999-6b0e43f3" (TheRock numbering, development build). The RCCL version string in the rank logs, 2.30.4, is the anchor.
