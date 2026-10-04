# Validated tuner configurations (GPU-107)

Each file passed the A/B gate on the MI355X cluster, 8 GPUs per node: default vs rule, interleaved,
7 repeats, P(sup) >= 0.95, gain > 2%, no size inside the rule's range regressing > 2%.

| file | scope | RCCL | note |
|---|---|---|---|
| `all_reduce_1n.conf` | 1 node, all_reduce, 4K..2M | 2.27.7, SGLang container | 8 rules, +6..+76% busbw over default in rccl-tests |
| `all_reduce_1n_plus_amd_gfx950.conf` | same + AMD's built-in gfx950 rules in the gaps | 2.30.4 | required on RCCL >= 2.30: a loaded conf replaces AMD's table, uncovered sizes lost 13-15% without this. Loads through the built-in CSV tuner (`NCCL_TUNER_CONFIG_FILE`), no plugin needed |
| `<collective>_<N>n.conf` | 1..5 nodes, 5 collectives | 2.28.3-develop, bare-metal | 120 rules total; not re-verified on 2.30.4 |

Verdict of the study (Sept 2026): none of the single-node all_reduce rules reaches a SGLang server.
SGLang's custom all-reduce (AITER) and RCCL 2.30's direct all-reduce (DDA) take every all_reduce
below 64 MiB before any tuner is consulted and are faster than the tuned path; above 64 MiB RCCL's
default is already optimal. Multi-node is the one place rules still apply, measured in rccl-tests only.

Evidence, pages and findings: branch `gpu107-clean`, start at
`results-tuning/2026-08-23-1-pages/final.html`; one row per run in `results-tuning/RUNLOG.md`.
