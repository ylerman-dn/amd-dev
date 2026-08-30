# Inference value test — Llama-3.1-8B, TP=8, single node

Run 2026-08-30, allocation 20856, node `amd-mi355x-9`.
SGLang `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260817`, model
`/data/mlperf_llama31_8b/model` (Llama-3.1-8B, dense, `hidden_size` 4096, 32 layers,
30GB, copied from node 5 over the FE network with the `fast-copy` skill).
Benchmark: `sglang.bench_serving`, random dataset, ISL 512 / OSL 512, concurrency 32,
128 prompts. Raw logs: node-local `/data/ylerman/infer-2026-08-30/<tag>/`.

Model chosen because it is dense — all_reduce is then effectively the only collective, so
any throughput change is attributable — and because `hidden_size` 4096 in bf16 puts the
per-layer all_reduce at 8KB, inside the shipped 1-node rule band (4K-1M, +12% to +64%).

## Result — the demo is blocked, and the reason is specific

| arm | RCCL | all_reduce path | tok/s | TPOT (ms) | median ITL (ms) |
|---|---|---|---|---|---|
| A | ours (mounted) | SGLang custom kernel (default) | 687.35 | 14.09 | 13.97 |
| B | ours (mounted) | RCCL (`--disable-custom-all-reduce`) | **hang** | - | - |
| C | container stock | RCCL | 668.22 | 15.69 | 15.61 |
| D | container stock | RCCL + DN tuner, 9 rules | tuner never fires | - | - |

Three findings, in order of importance.

### 1. SGLang bypasses RCCL for TP all_reduce by default

`disable_custom_all_reduce=False` is the default. In arm A the entire RCCL debug log holds
**6** AllReduce entries, all `count 1` — init-time barriers. The per-layer TP all_reduce
never reaches RCCL, so no tuner rule can fire in a default SGLang deployment.

With `--disable-custom-all-reduce` the picture inverts: **4747** AllReduce calls per rank.

### 2. Our RCCL build hangs when it actually carries the traffic

Arms B and B-retry both hung at the identical point — p2p `Send`/`Recv`, `opCount 989`,
`count 513024` — during benchmark warmup. Health endpoint dead, GPUs pinned at 100% in
RCCL busy-wait. The same configuration with the container's stock RCCL (arm C) completes
normally, so the hang is specific to our library (or to an ABI/version mismatch: ours is
`2.17.9-develop:2e42aa8`, the container is ROCm 7.2).

Note the shape: this is a **p2p-path** hang, the same path alltoall uses, and the same day
2 of 40 alltoall runs died with `ionic_comp` errors. Possibly one underlying issue; not
established.

### 3. Stock RCCL consults the tuner only at init, so the rules never fire

Arm D loads the plugin correctly — `DN-TUNER/Plugin: Loaded 9 tuning configurations` — and
RCCL does call it, proven by `Config does not match` lines. But there are only **72** such
calls, all immediately after `AllReduce: opCount 0 ... count 1`, i.e. the init-time 4-byte
all_reduce. Across the **12488** steady-state all_reduces at 8192 bytes and **4160** at
16384 bytes, the tuner is consulted **zero** times.

Both of those sizes have matching rules in `all_reduce_1n.final.conf`
(`allreduce,8192,8192,tree,ll,4` and `allreduce,16384,16384,tree,ll,16`). The rules are
aimed at the right sizes; they are simply never asked for.

Contrast with our own library under rccl-tests, which consults per collective:
`results-tuning/2026-08-04-2-ab1node/logs/1n_config_r6_dbg_3993445.log` contains **900**
`Applied config` lines spanning all 18 sizes.

So the two halves needed for a value demo sit in different libraries:

- our RCCL consults the tuner per op — but hangs under SGLang
- stock RCCL runs under SGLang — but consults the tuner only at init

## Incidental measurement worth keeping

Routing TP all_reduce through RCCL instead of SGLang's own kernel costs **2.8%**
throughput (687.35 -> 668.22 tok/s) and **1.6ms** of TPOT. That is the deficit any RCCL
tuning would have to make up before it is even even with the default deployment.

## 4. Follow-up probe — the library is not the problem, SGLang is

`run_tunerprobe.sh` runs our rccl-tests binary **inside the same container**, with no
`LD_PRELOAD`, so it links the container's own RCCL (`ldd` confirms
`/opt/rocm/lib/librccl.so.1`), with the same plugin and the same conf.

Result: **9 `Applied config` lines covering all 8 rule sizes**, including 8192 and 16384 —
the two the model actually uses.

```
Applied config for collType=allreduce, bytes=4096
Applied config for collType=allreduce, bytes=8192
Applied config for collType=allreduce, bytes=16384
Applied config for collType=allreduce, bytes=32768
Applied config for collType=allreduce, bytes=65536
Applied config for collType=allreduce, bytes=131072
Applied config for collType=allreduce, bytes=262144
Applied config for collType=allreduce, bytes=524288
Applied config for collType=allreduce, bytes=1048576
```

So stock RCCL **does** consult the tuner per collective. Finding 3 above is therefore not a
property of the library — it is something in SGLang's calling pattern that stops the tuner
being consulted for the steady-state all_reduces, despite `--disable-cuda-graph`.

That is good news: the rules can reach this library, and the demo is reachable once the
SGLang-side cause is found. It also means the honest blocker is narrower than it looked —
one unexplained behaviour in the serving stack, not an incompatibility.

## What to try next

1. Find why SGLang's steady-state all_reduces skip the tuner while rccl-tests' do not, on
   the same library in the same container. Candidates: a graph-capture path still active
   despite `--disable-cuda-graph`, a cached plan, or a separate communicator.
2. Build our RCCL against the ROCm version the container ships, and retry arm B — that
   would let the demo run on the library the rules were actually measured against.
3. If the SGLang cause proves intractable, a smaller harness (a PyTorch script doing
   `dist.all_reduce` at the model's sizes, TP=8) would still demonstrate the rules on a
   real framework path.
