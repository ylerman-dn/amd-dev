# Verifying what RCCL actually ran

`-A 1` (`--output_algo_proto_channels`) prints what was selected. **Not `-M 1`** — on this build
that is `--memory_report`, and `-R` is `--local_register`, not `--report_cputime` (`-C`). Upstream
rccl-tests uses `-M`; ours moved it. `tools/rccl-sweep/sweep_executor.py:186,189` passes both
wrong flags.

| column | vs INFO | trust it? |
|---|---|---|
| algo | base algo agrees 18/18 | yes |
| proto | agrees 18/18 | yes |
| nchannels | said 56 where real was 52 / 54 / 222 / 224 | **no** |

The two report **different layers**, so they are not always string-identical (14/18 for algo):
`-A 1` shows the RCCL addon (`RING*` = WarpSpeed, `Direct`), INFO's per-call line shows the NCCL
enum underneath. `-A 1` is the more informative of the two — `Direct` is visible only there.

Real channel counts come from `NCCL_DEBUG=INFO` → `channel{Lo..Hi}`, count = Hi−Lo+1. `-A 1`'s
channel column is a planned ceiling: trimmed down at 2M–8M, and multiplied ~4× by WarpSpeed
above 64M.

Requested is not selected — RCCL substitutes silently and returns success.

`results-tuning/2026-08-03-1-mcheck/` (flags) · `-4-infocap/info.log` (channels)
