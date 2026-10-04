# SUMMARY — 2026-09-09-1-qwen-modes: Qwen3-30B-A3B, radix cache OFF, graphs off (paired driver), four traffic modes

Node amd-mi355x-des2-2 (TEST, MI355X), job 21076. One server per mode, arms sc2 / dummy / dummy2 / none interleaved per round, 6 rounds (round 1 dropped), --disable-radix-cache, image NCCL_MIN_NCHANNELS removed, MSCCL=0, custom all-reduce off. Scores node/qm_*_score.csv.

## d32 (ISL 512 / OSL 512 / conc 32 / 128 prompts)

| arm | median tok/s | vs none | P(sup) | per-round MEDIAN TTFT ms (median over rounds 2-6) | TPOT ms |
|---|---|---|---|---|---|
| dummy | 716.82 | -7.6% | 0.0 | 1775 | 41.33 |
| dummy2 | 622.14 | -19.8% | 0.0 | 5168 | 41.69 |
| none | 775.71 | +0.0% | 0.5 | 146 | 41.03 |
| sc2 | 765.19 | -1.4% | 0.4 | 147 | 41.38 |

## p8k (ISL 8192 / OSL 128 / conc 32 / 96 prompts)

| arm | median tok/s | vs none | P(sup) | per-round MEDIAN TTFT ms (median over rounds 2-6) | TPOT ms |
|---|---|---|---|---|---|
| dummy | 121.46 | -78.9% | 0.0 | 15215 | 146.06 |
| dummy2 | 47.39 | -91.8% | 0.0 | 43306 | 340.42 |
| none | 574.57 | +0.0% | 0.5 | 1003 | 48.22 |
| sc2 | 570.99 | -0.6% | 0.4 | 1001 | 48.55 |

## d128 (ISL 512 / OSL 512 / conc 128 / 384 prompts; decode all_reduce = 512K -> sc2 rule 512K-1M ring,ll,64)

| arm | median tok/s | vs none | P(sup) | per-round MEDIAN TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| dummy | 2289.45 | -22.8% | 0.0 | 3704 | 47.25 |
| dummy2 | 1454.84 | -50.9% | 0.0 | 12850 | 63.7 |
| none | 2964.69 | +0.0% | 0.5 | 332 | 42.45 |
| sc2 | 2988.34 | +0.8% | 0.8 | 340 | 42.24 |

At 512K the bad confs hurt DECODE even with graphs off: dummy2 TPOT 63.7 vs 42.5 ms (+50%), dummy 47.3 (+11%) - the per-call penalty at 512K exceeds the CPU-side slack. sc2 +0.8% (P 0.80): the 512K-1M ring,ll,64 rule (+74% in rccl-tests) buys nothing measurable here while graphs are off.

## mix (ISL 2048 / OSL 512 / conc 64 / 192 prompts; decode all_reduce 256K -> sc2 rule 128K-256K tree,ll,64; prefill 8 MB chunks)

| arm | median tok/s | vs none | P(sup) | per-round MEDIAN TTFT ms | TPOT ms |
|---|---|---|---|---|---|
| dummy | 919.22 | -36.9% | 0.0 | 7374 | 54.12 |
| dummy2 | 519.53 | -64.3% | 0.0 | 21493 | 77.75 |
| none | 1455.78 | +0.0% | 0.5 | 656 | 42.7 |
| sc2 | 1468.84 | +0.9% | 0.4 | 507 | 42.62 |

Sizes (detect runs, node des2-2 qm_*can_srv/logs): d32 decode 128K (1.59M calls/arm) + 4K/8K + 2M prefill chunks; p8k 128K decode + 64 MB prefill chunks (8730) -> no sc2 rule.

Reading: the radix cache hid prefill yesterday. With it off, dummy (ring,simple,1) costs -8% (d32) / -79% (p8k) and 12x TTFT; dummy2 (tree,ll,1) -20% / -92% and 35x TTFT. sc2 ties none on tok/s in both modes: graphs off, decode is CPU-bound (TPOT 41-48 ms identical across arms), and sc2 has no rule at the prefill sizes. The decode effect appears only with CUDA graphs (-4-: sc2 +18% vs none, dummy2 -69%).
