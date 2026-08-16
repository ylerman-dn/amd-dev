# Validated tuner config — the shippable output

`validated_all_scales.conf` (md5 `d34745da827b517b236be539d9d963e1`) — **11 rules**, every one of
which beat RCCL's default in an interleaved A/B: 7 repeats per arm, probability-of-superiority
**1.00** (the config won every head-to-head pair), median gain > 2%, and no size inside the rule's
range regressing.

Deploy:
```
export NCCL_TUNER_PLUGIN=/opt/shared/ylerman/GPU-107/ab-tuner-test/librccl-tunerv4-dn.so
export NCCL_TUNER_CONFIG_FILE=<path>/validated_all_scales.conf
```

## Measured gains

| scale | rules | gains |
|---|---|---|
| 1 node | 8 | +12.5% @4K, +29.1% @8K, +45.3% @16K, +56.1% @32K, +60.2% @64K, +32.3% @128K, +39.5% @256K, +63.0% @512K–1M |
| 2 nodes | 1 | +11.1% @32M |
| 3 nodes | 2 | +8.7% @32M, +10.2% @64M–128M |

Per-scale evidence: `../2026-08-04-2-ab1node/`, `../2026-08-04-3-ab2node/`, `../2026-08-04-5-ab3node/`
(each has `SUMMARY.md`, `presentation.html`, `validate_report.txt`, and the raw arm logs).

## Verified end-to-end, not just per-scale

The combined file was run at 1 node after merging, to confirm the merge did not break scoping:

- 4K–1M: the 8 tuned rules fire (`TREE/LL/2,4,8,16,32` then `RING/LL/8,16,32`)
- 2M–512M: falls through to RCCL default (`RING/SIMPLE/56`, and `RING*/SIMPLE/56` from 64M)
- the 2-node and 3-node rules did **not** fire — `nNodes`/`nRanks` scoping works
- ≥64M retains `RING*` at 56 channels, i.e. WarpSpeed intact

Evidence: `/opt/shared/ylerman/GPU-107/verifycombined-2026-08-04/`.

## What is deliberately NOT in here

- **≥2M at 1 node.** No rule beat default. ≥64M is actively dangerous — see the warning below.
- **Most multi-node sizes.** RCCL's multi-node defaults already sit at the 48-channel cap and are
  hard to improve on. That is a finding, not a gap in effort.
- **Collectives other than all_reduce.** Not swept.
- **Two genuine 3-node wins** (+8.9% @256K, +12.1% @256M) that were rejected because
  `generate_tuner_config.py` merged them into ranges also containing a regression. Recoverable —
  see `../2026-08-04-5-ab3node/SUMMARY.md`.

## The one trap that cost 65.7%

Where WarpSpeed is active (≥64M, 1 node) the plugin's `channels` field is the **post-multiplier
total**, while `NCCL_MIN/MAX_NCHANNELS` sets the **base**. Same integer, 4× different meaning.

A channel value swept via env var and written unchanged into a plugin rule under-provisions by 4×:
a `48` request became 12 base / 48 actual against the default's 56 base / **224** actual — measured
at **−65.7%**. This is why no 1-node rule above 1M appears here, and why any future config must not
copy env-var channel numbers into plugin rules wherever WarpSpeed applies.

## Scope and limits

- MI355X (gfx950), **RCCL 2.28.3-develop:2e42aa8** with rccl-tests from the same commit. A
  different RCCL may select differently — the whole acceptance map is version-specific.
- 8 ranks × 1 GPU per node. Nodes `amd-mi355x-5,6,7`.
- all_reduce, float/sum, 4K–512M.
- A/B measured with `srun`; the sweep that generated candidates used mpirun-outside-Slurm and
  reads 1–10% lower. The A/B is self-contained (both arms `srun`, interleaved) — but sweep and A/B
  busbw must not be compared to each other.
- Co-tenants shared the fabric leaf switches during the multi-node runs. That affects absolute
  busbw, not the A/B contrast, since arms were interleaved under identical conditions.

## Format note

Both tools disagree about the header line, and the file keeps it deliberately:
`dn/dn-tuner/plugin.c:199` skips only `#`/blank lines, so it parses the header as one extra
harmless non-matching rule (hence the plugin logging one more "configuration" than there are
rules). `validate_tuner_config.py`'s `parse_rules()` treats the first non-comment line as a header,
so removing it would make the validator swallow the first real rule.
