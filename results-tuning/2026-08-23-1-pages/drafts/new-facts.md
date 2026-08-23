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
