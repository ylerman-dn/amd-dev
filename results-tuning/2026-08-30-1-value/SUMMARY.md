# 2026-08-30 — proving value: inference, alltoall, and leaf placement

Allocation 20856 (`ylerman-gpu107-value`), nodes `amd-mi355x-[2-3,5,9]`.
Three independent questions, one session. Per-task detail in the subdirectory SUMMARYs.

| | question | answer |
|---|---|---|
| [leaf-ab](leaf-ab/SUMMARY.md) | does a tuning gain survive crossing the spine? | **yes** — 32M rule gives +9.5% same-leaf, +7.7% cross-leaf, no overlap between arms |
| [alltoall-2n](alltoall-2n/SUMMARY.md) | can alltoall be tuned once the right knob is used? | **no** — but the earlier "no" was measured with a knob that did nothing |
| [inference](inference/SUMMARY.md) | can we show value on a real model? | **blocked** — for a concrete, fixable reason |

## Leaf placement — the clearest result of the day

Same-leaf `{3,5}` versus cross-leaf `{3,9}`, node 3 anchored in both, 5 interleaved
repeats, tuned and default arms, every run verified by `Applied config` in 16/16 rank logs.

Placement moves the **baseline** by up to 17% at small sizes (-9.2% @4K, -17.4% @16K,
-8.8% @128K, -2.9% @1M) and by nothing at all above 8M — the signature of per-hop latency.

Placement does **not** take the gain away. At 32M every tuned repeat beat every default
repeat in both placements: +9.5% same-leaf, +7.7% cross-leaf. That independently
replicates the shipped 32M verdict (+11%) on different nodes.

The 256K rule could not be resolved: repeat spread is 7-12% at that size and a no-rule
control swings +6% on noise alone, so 5 repeats cannot see a 5% effect.

**Consequence for the project:** the tuner does not need a topology input. But because
placement shifts the baseline so much at small sizes, any comparison between runs on
different node sets carries a placement component, and `RUNLOG.md` nodelists are the only
record of which placement a result came from.

## alltoall — the previous measurement was invalid, the conclusion survives

The 2026-08-26 alltoall work drove `NCCL_MIN/MAX_NCHANNELS`. Those bound the **collective**
path. alltoall runs on the **p2p** path, controlled by `NCCL_MIN/MAX_P2P_NCHANNELS`
(`src/graph/paths.cc:982-983`). The old runs changed nothing, so "+0.00% median" was a null
edit rather than a measurement.

Redone with the correct knob: forcing 8 p2p channels moves busbw by -11% to -27%, proving
the knob works. 16/32/64 all sit within a few percent of default at every size. RCCL's
default is already at or above the plateau; the only direction available is down. Verdict
unchanged, evidence now real.

Also closed by log rather than by reading source: with the plugin loaded and holding a
config, an entire alltoall run produced **zero** `Applied config` and **zero**
`Config does not match` lines. RCCL never consults the tuner for alltoall. The pivot path
(`RCCL_ALL_TO_ALL_PIVOT_ENABLE=1`), which would route alltoall through the collective path,
did not engage on this topology.

## Inference — blocked by a pincer, both halves identified

Llama-3.1-8B, TP=8, SGLang v0.5.17. Chosen because it is dense (all_reduce is the only
collective, so attribution is clean) and `hidden_size` 4096 in bf16 puts the per-layer
all_reduce at 8KB, inside the shipped 1-node rule band.

1. **SGLang bypasses RCCL by default.** `disable_custom_all_reduce=False`; the whole RCCL
   log holds 6 AllReduce entries, all `count 1`. No rule can fire in a default deployment.
2. **Our RCCL hangs when it does carry the traffic.** With `--disable-custom-all-reduce`,
   both attempts hung at the identical point (p2p `Send`/`Recv`, `opCount 989`). Stock
   container RCCL in the same configuration completes fine, so it is our build or its
   ABI/version fit with ROCm 7.2.
3. **Under SGLang, stock RCCL consults the tuner only at init.** 72 calls, all at
   `opCount 0`; zero across the 12488 steady-state all_reduces at 8192B and 4160 at
   16384B — both of which have matching rules.
4. **But that is SGLang's doing, not the library's.** rccl-tests run inside the same
   container against the same stock RCCL logs `Applied config` for all 8 rule sizes,
   8192 and 16384 included. So the rules can reach this library; something in SGLang's
   calling pattern skips the tuner for steady-state collectives.

The demo is therefore reachable, blocked on one unexplained SGLang behaviour rather than
on an incompatibility.

One measurement worth keeping: routing all_reduce through RCCL instead of SGLang's own
kernel costs **2.8%** throughput (687.35 -> 668.22 tok/s). Any RCCL-side tuning has to
make that up before it breaks even with a stock deployment.

## Cluster facts learned

- **Node 2 cannot run MPI benchmarks.** Every `{3,2}` arm died with
  `mpi/pmix_v4: pmixp_server.c:1582: Cannot send message`, twice, while `{3,5}` ran clean
  with identical env. A bare `srun --mpi=pmix hostname` on `{3,2}` succeeds, so it is not
  connectivity. Node 9, never previously exercised, worked perfectly as the L2 partner.
- **alltoall multi-node instability is intermittent, not a config bug.** A first crash
  (`ionic_comp_msn:1277: cqe with error 12`) suggested `NCCL_IB_QPS_PER_CONNECTION=2`; a
  controlled probe rejected that (3 repeats each at QPS=2 and QPS=1, all six clean). The
  full run then lost 2 of 40 to the same error, ~5%, matching the older "5 of 18" note.
- **rccl-tests' CSV writer is off by one** — 13 header names over 14-field rows, so any
  `DictReader` on it silently misaligns. `analyze.py` here parses positionally.

## Open, in priority order

1. Find why SGLang's steady-state all_reduces skip the tuner while rccl-tests' do not, on
   the same library in the same container. Candidates: a graph-capture path still live
   despite `--disable-cuda-graph`, a cached plan, or a separate communicator.
2. Rebuild our RCCL against the container's ROCm and retry the hang — that would let the
   demo run on the library the rules were measured against.
3. Failing 1, use a small PyTorch `dist.all_reduce` harness at the model's sizes (TP=8) to
   demonstrate the rules on a real framework path.
4. Re-run the 256K leaf comparison at 9 repeats if that rule's portability matters.
