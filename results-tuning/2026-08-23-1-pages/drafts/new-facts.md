# New verified facts — one line each (draft, not approved)

All verified against RCCL source at the deployed commit (`git show 2e42aa8:...`, rocm-systems clone) or captured logs.

1. TREE is honoured **only for all_reduce** (TREE/SIMPLE needs >=2 nodes) — `tuning.cc:653,657,660`; live logs `2026-08-03-6-acceptmap/TREE_LL.log`.
2. broadcast, reduce, all_gather, reduce_scatter: **RING is the only algo reachable via NCCL_ALGO** — `tuning.cc:653-656`.
3. An unsupported algo request is **silently replaced by RING** (INFO "Optimal algorithm is not found...") — `enqueue.cc:2152-2155`.
4. Our COLLNET probes failed on **spelling** (`COLLNET_DIRECT` vs RCCL's `CollNetDirect`, parser rejects at init) — proves nothing about CollNet — `tuning.cc:100-108`.
5. all_gather below ~8 MiB runs the AMD **"Direct" kernel** regardless of request; observed executing 37,632x — `rccl_wrap.cc:445-484`, `collectives.cc:124`.
6. `-A 1` **cannot see Direct reduce_scatter** (execution path has it, reporting path doesn't); it can only fire at 2 nodes and >=128K — `collectives.cc:424` vs `rccl_wrap.cc:371`, gate `rccl_wrap.cc:522-524`.
7. RCCL's init-time perf table prints **shifted row labels** (row printed "AllGather" is really Broadcast, etc.) — `tuning.cc:928` vs `init.cc:98`; captured `2026-08-03-4-infocap/info.log:7448`.
8. Finding 04 is stale: the median step **exists** since 2026-08-16 (`merge_metrics.py --median`, commit `7fd439b`); the racing-search-to-config gap is closed by `--emit-optimized` (still uncommitted).
9. The workflow doc teaches **numeric algo codes; the plugin only accepts lowercase strings** and silently coerces anything else to ring/simple — `rccl-sweep-user-workflow.md:227-236` vs `plugin.c:82-116`.
10. Every generated config ships an **inert header rule** (plugin loads the CSV header as config #0; it can never match) — `plugin.c:130-154`, `config_generator.py:266-279`.

## Added 2026-08-26 (awaiting approval, drafts for findings/)

11. **alltoall never reaches the collective tuner.** enqueue.cc (deployed
    2e42aa8) decomposes ncclAlltoAll into per-rank p2p send/recv tasks;
    getAlgoInfo (and with it the tuner plugin) runs only for collective
    tasks. Consequences, all observed live: NCCL_ALGO/NCCL_PROTO are no-ops
    (RING/LL == RING/SIMPLE at every channel count), `-A 1` prints N/A in
    all selection columns, a tuner.conf cannot apply to alltoall. Evidence:
    `git -C ~/wg/gitea/rocm-systems show 2e42aa8:projects/rccl/src/enqueue.cc`
    (taskAppend), runs in results-tuning/2026-08-26-1-a2a123/.

12. **alltoall executed p2p channels vs the NCCL_MIN/MAX_NCHANNELS request**
    (from the runs' debug "p2p channels:" lines): at 1 node the request is
    multiplied x4 and capped at 64 (request 1 -> 4, 8 -> 32, >=24 -> 64:
    default 64); at 2 nodes honoured up to a cap of 32; at 5 nodes honoured
    at 1/8/16, 24-32 -> 32, 40-48 -> 64 (default 64). Evidence: dbg logs in
    /opt/shared/.../rccl-tune-2026-08-26-a2a123 and -a2a45.

13. **Chronic vs transient noise are different failure modes and need
    different gates.** Chronic: specific small sizes scatter >25% in
    back-to-back same-config runs on an otherwise clean machine (broadcast
    2n at 4-16K, 22 refused attempts; alltoall 1n at 32K, 690%) - solved by
    the rule-scoped preflight (exclude the size, judge the rest; broadcast
    2n then verified 4 rules incl. +104.7% at 64K). Transient: random sizes
    collapse up to 97% for one run at >=4 nodes with no wall-clock 30s/60s
    phase alignment (stage5n/NOISE-ANALYSIS.md) - not solved; rc=2 void
    gate stays the backstop.
