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

## Result — the rules do reach the model, after two obstacles were cleared

**Read sections 5 and 6 first.** Sections 1-4 are the investigation in the order it
happened, and section 3's conclusion ("the tuner never fires") was superseded once the
real cause was found.

| arm | RCCL | all_reduce path | tok/s | TPOT (ms) | median ITL (ms) |
|---|---|---|---|---|---|
| A | ours (mounted) | SGLang custom kernel (default) | 687.35 | 14.09 | 13.97 |
| B | ours (mounted) | RCCL (`--disable-custom-all-reduce`) | **hang** | - | - |
| C | container stock | RCCL | 668.22 | 15.69 | 15.61 |
| D | container stock | RCCL + DN tuner, 9 rules | 0 tuner hits — MSCCL was running the collectives | - | - |
| E | container stock | RCCL + tuner, `RCCL_MSCCL_ENABLE=0` | **rules fire** — 13000 hits at 8192B | - | - |

Two obstacles, both cleared: SGLang bypasses RCCL by default (section 1), and MSCCL
intercepts the collectives before the tuner sees them (section 5). One obstacle remains
open: our own RCCL build hangs under SGLang (section 2), so the demo currently runs on the
container's stock RCCL rather than the library the rules were measured against.

Findings in the order they were established.

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

### 3. Under SGLang the tuner was consulted only at init (superseded by section 5)

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

At this point the conclusion drawn was that the two halves needed for a value demo sat in
different libraries. That was wrong — see section 5. The collectives were being executed by
MSCCL, which never asks the tuner, so the library was never the issue.

## Incidental measurement — but see section 6 before quoting it

Routing TP all_reduce through RCCL instead of SGLang's own kernel appeared to cost **2.8%**
throughput (687.35 -> 668.22 tok/s). Section 6 shows the benchmark's no-op noise is 3.5%,
so this figure is inside the noise band and needs repeats before it means anything.

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

That narrowed the blocker to one unexplained behaviour in the serving stack rather than an
incompatibility. Section 5 identifies it.

## 5. Root cause found — MSCCL was executing the collectives

The reason the tuner saw nothing during serving is **MSCCL**, a separate algorithm engine
inside RCCL that selects from pre-generated plans and never consults an external tuner. It
is on by default (`RCCL_MSCCL_ENABLE`, default 1, `src/misc/msccl/msccl_lifecycle.cc:30`).

Evidence: after warmup the RCCL debug log is dominated by

```
133315  mscclFuncAllReduce
```

With `RCCL_MSCCL_ENABLE=0`, `mscclFuncAllReduce` drops to **0** and the tuner is consulted
at exactly the sizes the model uses:

| size asked about | tuner hits | matching shipped rule |
|---|---|---|
| 8192 B | 13000 | `allreduce,8192,8192,tree,ll,4` |
| 16384 B | 4160 | `allreduce,16384,16384,tree,ll,16` |
| 262144 B | 1064960 | `allreduce,262144,262144,ring,ll,32` |

So the shipped 1-node rules **do** reach a real model workload. Findings 1-3 above were
about a path the collectives were never taking.

An intermediate probe explains the earlier confusion: with the rules-plus-catch-all conf and
MSCCL still enabled, the tuner was asked 65 times per rank, but only about 58-130MB
collectives — the ones too large for MSCCL's plans. The small ones, which are the model's
actual traffic, went to MSCCL.

## 6. Benchmark noise — read before trusting any single number

A conf that changes nothing (wildcard rule, `-1` for algo, proto and channels) produced
644.52 tok/s against 668.22 for the identical setup without it. That is a **3.5% swing from
a no-op**, so single runs cannot resolve small effects, and the 2.8% RCCL-versus-custom-kernel
figure in section 3 is inside the noise band and should not be quoted without repeats.

The A/B in `run_ab.sh` therefore alternates arms across repeats.

## 7. The A/B — rules fire, model gets slightly slower

MSCCL off, stock RCCL, 3 repeats per arm, arms alternating. Output token throughput:

| | run 1 | run 2 | run 3 | median | tuner hits |
|---|---|---|---|---|---|
| default | 626.10 | 660.92 | 652.25 | **652.25** | 0 |
| tuned | 612.58 | 631.36 | 630.56 | **630.56** | 1,082,120 |

**-3.3% at the median.** The tuner arm lost all three paired runs (-2.2%, -4.5%, -3.3%) and
TPOT rose in all three. The rules unambiguously applied — over a million hits per run against
zero in the default arm.

Caveat that keeps this from being a firm negative: the default arm's own three runs span
626-661, a 5.6% spread, so a 3.3% gap sits inside the noise. Three-for-three in the same
direction is suggestive, not conclusive; a sign test on 3 pairs gives p=0.125.

So: **no gain demonstrated on a real model, with a hint of a small regression.**

Hypothesis for why, untested. The rules were selected on rccl-tests busbw with the collective
running alone on the GPU, where tree/LL at 2-16 channels wins. In a real model the all_reduce
overlaps with compute; a config that maximises isolated bandwidth may be the wrong choice when
it has to share compute units with matmuls. Finding 11 already showed a related effect from the
other direction — a pinned-channel rule disabling WarpSpeed.

## What to try next

1. Finish the A/B (`run_ab.sh`, MSCCL off, tuner vs no tuner, alternating repeats) and see
   whether the shipped rules move throughput beyond the 3.5% noise band.
2. Build our RCCL against the ROCm version the container ships, and retry arm B — the demo
   currently runs on stock RCCL, not the library the rules were measured against.
3. Decide whether `RCCL_MSCCL_ENABLE=0` is an acceptable deployment condition. If it is
   not, the rules cannot reach a stock SGLang deployment no matter how good they are, and
   the honest framing is that MSCCL owns these collectives.
