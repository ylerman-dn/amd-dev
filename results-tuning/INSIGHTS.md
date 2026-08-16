# What we learned — RCCL tuning on MI355X

Everything below comes from measurements in `results-tuning/`. Plain language, one idea per section.

---

## 1. There are two completely different problems, and people confuse them

**On one node, RCCL uses too few channels.**
For small messages the default runs 1, 1, 1, 2, 4, 8, 16 channels. The best available is 24–56.
That's it. That's the whole 1-node story. Turn the channel count up and you gain 19–67%.

**On two or three nodes, the channels are already fine.**
The default sits at 48 (the cap) nearly everywhere. There is almost nothing to gain by changing them.
What's wrong instead is *when* RCCL switches algorithm or protocol.

So "tune RCCL" means two different jobs depending on scale. A method that works for one may find
nothing at the other.

---

## 2. Where the money actually is

Best config we measured, versus RCCL's own default, same sweep:

| | 1 node | 2 nodes | 3 nodes |
|---|---|---|---|
| average headroom | **+22.7%** | +1.3% | +3.2% |
| sizes with ≥10% headroom | **9 of 18** | 1 of 18 | 3 of 18 |
| where it lives | 4K–1M | 256K, 32M | 256K, 32M, 128M, 256M |

At 1 node, **everything above 2 MB is already perfect** — within 0.2% at every size. Don't spend
effort there.

At 2 and 3 nodes the headroom is concentrated in a handful of specific sizes, not spread out.

---

## 2b. The biggest win in the project is on a collective nobody tuned

`broadcast` and `reduce` at 1 node beat their default by **more than double** in the 512K–2M band.
Verified by A/B, 7 repeats per arm, probability-of-superiority 1.00:

| size | default | forced `RING/LL` | change |
|---|---|---|---|
| 256K | 13.08 | 12.93 | −1.1% |
| **512K** | **10.18** | **25.28** | **+148.3%** |
| **1M** | 19.38 | 37.59 | **+93.9%** |
| **2M** | 37.89 | 50.29 | **+32.7%** |
| 4M | 65.92 | 60.47 | −8.3% |
| 8M | 107.86 | 66.95 | −37.9% |

`reduce` is the same shape: +143.5% / +99% / +28.3%.

**The tell is that the default goes backwards.** 13.08 GB/s at 256K falls to 10.18 at 512K —
bandwidth should never drop as the message grows. RCCL switches away from LL around 512K and the
replacement is half as fast for three consecutive sizes, then becomes correct again by 4M.

Same class of error as everything in section 3 — a switch at the wrong size — but two to three times
larger, on collectives that have never been swept.

Evidence: `results-tuning/2026-08-16-15-bcast/`. Found by mining `2026-08-03-7-collmap/`, which had
sat unanalysed since it was collected.

---

## 3. RCCL switches algorithms too late — every single time

This is the most useful thing we found.

When RCCL changes algorithm or protocol as messages get bigger, it consistently waits too long:

| scale | the switch | should happen at | RCCL waits until | |
|---|---|---|---|---|
| 2 nodes | LL → LL128 | 256K | 512K | 1 step late |
| 2 nodes | TREE → RING | 32M | 64M | 1 step late |
| 2 nodes | LL128 → SIMPLE | 128M | 128M | correct |
| 3 nodes | LL → LL128 | 128K | 512K | **2 steps late** |
| 3 nodes | TREE → RING | 32M | 64M | 1 step late |
| 3 nodes | LL128 → SIMPLE | 256M | 512M | 1 step late |

**Five of six are late. None are early.** That is a systematic bias, not random error.

Every multi-node win we verified is one of these. We have been fixing them one size at a time by
writing rules. The cause is upstream — RCCL computes those switch points from internal constants.

**At 1 node there's a different error:** RCCL goes straight from `TREE/LL` to `RING/SIMPLE` and never
uses `RING/LL` at all. But `RING/LL` is the best choice from 256K to 1M. It's skipping a whole regime.

---

## 4. More channels is not always better — sometimes it's catastrophic

Everyone assumes bandwidth rises with channel count. It does for TREE. For RING at multi-node small
and mid sizes it collapses:

```
2 nodes, 256K, RING/LL:    8 channels -> 5.96 GB/s
                          48 channels -> 2.13 GB/s      -64%
```

38 config-curves at 2 nodes and 35 at 3 nodes lose more than 3% by going to maximum channels. Some
lose more than half.

**This is also what causes the famous 128M dip at 3 nodes:**

```
3 nodes, 128M, RING/LL128:  32ch -> 256.4   40ch -> 271.4 (peak)   48ch -> 243.9
```

The default uses 48 because that's the cap. The peak is at 40. That's the entire anomaly — not a
fabric problem, not a bug, just over-provisioning past the optimum.

**Consequence for any search method:** you cannot assume the channel curve is monotonic and stop
early. It has a peak, and the peak moves with size and algorithm.

---

## 5. Where the channel curve flattens

How many channels you actually need before it stops helping:

| message size | 1 node | 3 nodes |
|---|---|---|
| 4K | 24 | 2 |
| 32K | 16 | 16 |
| 256K | 24 | 16 |
| 2M | 32 | 48 |
| 16M | 56 | 48 |
| 128M | 48 | 40 |

Small messages saturate almost immediately. Large messages need everything you have. The knee moves
right as size grows — which is exactly why a single channel count can't be right for all sizes, and
why the tuner exists.

---

## 6. The measurement was wrong five different ways before it was right

None of these were visible in the output. Each would have produced confident nonsense.

| what was wrong | what it did |
|---|---|
| asked for the wrong report flag (`-M` not `-A`) | the sweep never saw what RCCL actually chose |
| `-R` was buffer registration, not CPU-time reporting | silently changed the thing being measured |
| recorded what we *asked for*, not what RCCL *did* | 37% of rules named combinations RCCL can't deliver |
| ranked on a time column that had become CPU time | picked 1-channel configs that were 99% slower |
| A/B used 1 rank × 8 GPUs, sweep used 8 × 1 | compared two different machine shapes |

**The lesson isn't the individual bugs — it's that the tool reported success throughout.** Every gate
we added afterwards exists because something passed silently.

---

## 7. Three environmental effects that will bite anyone measuring here

**GPUs start cold.** They idle at 195 MHz against a 2400 MHz peak. The first few runs after an
allocation measure the ramp, not the hardware. One run kept 0 of 11 rules; the identical config kept
7 of 11 once warm. Two nodes need ~4 warm-up runs, three nodes need ~8.

**A monitoring daemon disturbs multi-node runs.** `amd-nic-metrics` spawns 9 `nicctl` processes
polling RDMA queue-pair statistics every ~30 s for ~9 s. Runs take 11–15 s, so roughly a third land
in a poll and come back 10× slow. It is the best explanation for multi-node noise — better than the
co-tenant theory we initially believed and later disproved.

**Co-tenants matter less than expected.** Nodes 5 and 7 share a leaf switch, so traffic between them
never reaches the spine another job is using. We wrongly blamed a neighbour job for hours.

---

## 8. How noisy the measurement actually is

Same configuration, measured three times, across 2538 config-size points:

| | median spread | 90th percentile | worst |
|---|---|---|---|
| 1 node | 0.62% | 2.3% | 172% |
| 2 nodes | 0.80% | 4.5% | 233% |
| 3 nodes | 0.65% | 4.5% | 200% |

**The median is tiny; the tail is huge.** Most points repeat within 1%. A few stall completely.

Two consequences:
- The old 5% "these are equivalent" tolerance was ~7× the real noise, so it declared genuinely
  different configs tied. Now 0.5%.
- A single stalled run can sink a good rule. One did: a 3-node rule measured +7.6% but scored
  P(sup) 0.86 because one repeat came back 16× slow. It passed at +7.2% on a rerun.

---

## 9. How many repeats do you need?

We rebuilt the config from each single repeat and compared.

| | configs differ at | typical cost | worst case |
|---|---|---|---|
| 1 node | 7 of 18 sizes | 0.6% | 2.7% |
| 2 nodes | 10 of 18 sizes | 1.1% | **10.2%** |
| 3 nodes | 6 of 18 sizes | 0.4% | 0.7% |

**One repeat is usually fine and occasionally costs you a win.** The configs that swap places are
genuinely close, so the choice rarely matters — but sometimes it does.

The A/B is what protects you: a weak candidate still has to beat the default over 7 repeats. So a
1-repeat sweep risks *missing* a win, not *shipping* a bad rule.

**Practical rule: 1 repeat to find where the wins are, 3 repeats to measure them properly.**

---

## 10. The "smart search" in the tool does nothing here

`rccl_autotune.py` has a hotspot-refinement loop meant to re-sweep interesting regions. We ran it at
1, 2 and 3 nodes:

```
Hotspots detected: 0      Iterations: 1      (all three scales)
```

It looks for bandwidth *dropping* as size grows. Bandwidth rises monotonically, so on a healthy curve
there is nothing to find. And the one real dip we have — 3 nodes at 128M — is 8.4%, below its 10%
trigger.

Its apparent 3× speed advantage was entirely from doing 1 repeat instead of 3.

**The orchestration part of that script is genuinely useful.** The hotspot part is not, as configured.

---

## 11. The trap that cost 65%

Where WarpSpeed is active (≥64 MB, 1 node), RCCL multiplies the channel count by 4.

- Set channels **by environment variable** → that's the **base** count
- Set channels **in a plugin rule** → that's the **post-multiplier total**

Same integer, four times the meaning. We swept `48` via env var, wrote `48` into the config, and
deployed 12 base / 48 actual against the default's 56 base / **224** actual. Measured −60.3%.

Writing `224` restores parity exactly. So: **multiply by 4 before putting a swept channel count into
a plugin rule, wherever WarpSpeed applies.**

This is the clearest example of a general rule: *the path you measure on must be the path you deploy
on, or you are not measuring what you ship.*

---

## 12. What to do next, in order of value

**1. Fix the cause, not the symptoms (highest value).**
RCCL supports a v5 tuner plugin that lets you rewrite its internal cost-model constants — latencies
and per-channel bandwidths per algorithm and protocol. Those constants are what produce the switch
points in section 3. We are on v4, which only lets us override outcomes size by size. Correcting the
constants could fix all six late transitions at once, and would apply to collectives we never swept.

**2. Sweep the other five collectives.**
Everything we know is all_reduce. `reduce_scatter`, `all_gather`, `broadcast`, `reduce` and
`alltoall` are untouched. Section 1's split (channels at 1 node, thresholds at multi-node) is a
prediction to test, not a fact, for those.

**3. Find out which message sizes your workload actually uses.**
We can prove each rule helps. We cannot say what the config is worth to you, because that depends
entirely on your size distribution. A model doing mostly 4 MB all_reduces gains nothing from our
1-node config; one doing 512 KB gains 65%. This is the missing denominator on every number we
report.

**4. Reduce runs, if the space grows.**
54 configs per collective-scale is small enough that exhaustive search takes ~35 minutes. Smarter
search only pays once the space is much larger — more parameters, more scales. Prune combos that
lose at *every* size, and sample the channel curve sparsely rather than all 9 values; expect roughly
half the runs, not a tenth.

---

## 13. Numbers worth remembering

- **+148.3%** — best verified gain anywhere: broadcast, 1 node, 512K
- **+65.3%** — best verified all_reduce gain (1 node, 512K)
- **+17.3%** — best verified 3-node gain (256M–512M)
- **0.65%** — median run-to-run spread; the noise floor everything must clear
- **5 of 6** — multi-node algorithm/protocol switches that happen too late
- **48 vs 224** — the WarpSpeed channel-unit trap, in one comparison
- **0** — hotspots found by the "smart" search, across three scales
- **~30%** — fraction of multi-node runs disturbed by the NIC metrics poller

---

*Sources: `results-tuning/2026-08-03-*`, `2026-08-04-*`, `2026-08-16-*`, and `findings/`.
Everything measured on MI355X (gfx950), RCCL 2.28.3-develop:2e42aa8, 8 ranks × 1 GPU per node,
all_reduce, float/sum, 4K–512M. A different RCCL version may behave differently — the acceptance
map and the defaults being beaten are both version-specific.*
