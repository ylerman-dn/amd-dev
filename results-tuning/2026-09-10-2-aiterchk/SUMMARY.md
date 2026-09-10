# SUMMARY — 2026-09-10-2-aiterchk: what carries the stock all_reduce above the custom-AR cap

Question: in stock SGLang, the gpt-oss prefill all_reduce chunks are 91-94 MB, above aiter's 64 MiB
custom-AR cap, yet RCCL saw a single 4-byte call in the whole stocksc2 run
(`results-tuning/2026-09-09-7-gptoss-attr/SUMMARY.md`). Who took them?

Method: read the code and env of the actual container `lmsysorg/sglang:v0.5.17-rocm720-mi35x` on
amd-mi355x-5 (one-shot `srun`, jobs 21147-21150, `docker run --entrypoint bash`, no model, no
measurement), plus grep of the stock/customar server logs on ses2-1. Logs: `aiterchk.log`,
`aiterchk2.log`, `aiterchk3.log`, `qrchk.log` here.

Confirmed in the container (source):
- aiter at `/sgl-workspace/aiter` is commit d9e5ef7ce08e (2026-07-29), the Dockerfile pin (`aiterchk2.log`).
- `_DEFAULT_CAR_MAX_SIZE = 8192 * 8192` (64 MiB), env `AITER_CUSTOM_AR_MAX_SIZE` unset in the image, so
  custom AR runs for 16 B-aligned inputs up to 64 MiB and declines above (`_fits_custom_ar_size`,
  `aiterchk2.log`).
- SGLang's own class caps at 16 MiB on HIP but is not the one used: server logs of the stock and customar
  arms print `[AR] Using AiterCustomAllreduce (AMD default)` on all 8 ranks
  (`/data/ylerman/gptoss-attr-2026-09-09/at_stock_def/server.log` on ses2-1).
- **The image sets `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`** (`aiterchk3.log`, container env). SGLang's
  default is NONE (quick reduce off); the AMD image turns it on. With
  `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1` (default) the size table used is fp16/TP-8:
  INT8 min 4 MB, max 2 GB (`quick_all_reduce.py:61, 199-209`, `qrchk.log`).
- Dispatch order (`parallel_state.py:880-914`): custom AR if <= 64 MiB, else quick reduce if >= 4 MB,
  else mscclpp (off), else RCCL via pynccl.

Conclusion (inferred from the above, not observed as a log line: quick reduce logs nothing on success):
- decode 184 KB and prefill 2.9 MB / 47 MB: aiter custom AR (two-shot, exact bf16).
- prefill 91-94 MB: above the custom-AR cap, taken by **quick reduce with INT8 quantisation**.
- Qwen p8k 61.9 MiB chunks: custom AR (just under the cap).
- RCCL sees nothing in either case, which matches the stocksc2 observation. The contradiction is resolved.
- Stock therefore quantises its largest prefill all_reduce to INT8. Whether that affects output quality
  was not measured.

Not measured: throughput of any path; whether quick reduce actually fired (needs a probe with
ROCM_QUICK_REDUCE_QUANTIZATION=NONE, or a RCCL log showing the 94 MB call appear when it is unset).
