# 11 — A pinned-channels tuner rule disables WarpSpeed at >= 64M (1 node)

**Claim.** At >= 64M single-node, RCCL's default executes WarpSpeed
(~222-224 channels). A tuner-plugin rule pinning ring/simple with a fixed
channel count executes exactly that count (48), WarpSpeed does not engage,
and throughput collapses ~ -63..-66% vs default. This answers the previously
open question "does a tuner config saying ring disable WarpSpeed above 64M".

**Evidence.** Same A/B, same sizes, executed channels from the runs' own
debug logs: default arm 222-224ch vs config arm 48ch, gains -63.5..-65.8%
(P(sup) 0.00) — `results-tuning/2026-08-24-1-pilot/ab_out3/allreduce_1n/`
(`per_size_stats.csv` def_exec/cfg_exec columns, `validate.log` verdicts);
requested-vs-executed table `results-tuning/2026-08-24-1-pilot/channel_verification.md`.
Reproduce: `grep "channel{Lo..Hi}" <dbg log>` at 67108864 bytes, both arms.

**Date.** 2026-08-24. **Scope.** all_reduce measured; mechanism
(warpSpeedChannelMultiplier 4) build-wide. The A/B gate drops such rules, so
shipped configs are unaffected — the danger is only in trusting env-var
search results above 64M without an A/B.

**Falsified by.** A plugin/RCCL change letting WarpSpeed coexist with pinned
channel rules.
