# 09 — Which algorithms RCCL honours, per collective

**Claim.** On the deployed build (2.28.3-develop:2e42aa8): TREE is honoured
only for all_reduce (TREE/SIMPLE additionally requires >= 2 nodes on
gfx942/gfx950); broadcast, reduce, all_gather and reduce_scatter accept only
RING through `NCCL_ALGO`. Unsupported requests are silently substituted to
RING. all_gather additionally runs the AMD-specific Direct kernel below its
size threshold regardless of any request.

**Evidence.** Source, exact deployed commit:
`git -C /home/dn/ylerman/wg/gitea/rocm-systems show 2e42aa8:projects/rccl/src/graph/tuning.cc`
lines ~653 (broadcast/reduce RING-only), ~654-656 (ag/rs allowed set), ~657
(all_reduce excludes PAT), ~660 (TREE/SIMPLE single-node skip). Live logs:
`results-tuning/2026-08-03-6-acceptmap/TREE_LL.log` (TREE honoured, all_reduce),
`results-tuning/2026-08-03-9-multinode/{2n,3n}/all_reduce__TREE_*.log`;
Direct executing 37,632x in `results-tuning/2026-08-03-7-collmap/debug_files.tgz`
(INFO "RCCL DIRECT ALLGATHER count").

**Date.** 2026-08-24. **Scope.** This build, MI355X, 8 GPU/node.
**Falsified by.** An RCCL upgrade changing tuning.cc's per-collective skips.
