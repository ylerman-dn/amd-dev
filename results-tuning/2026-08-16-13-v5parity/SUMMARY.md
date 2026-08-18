# v5 tuner port — parity & constants results

Worktree `amd-dev-v5`, branch `gpu107-tuner-v5`. Cluster workdir (raw logs also
copied here): `/opt/shared/ylerman/GPU-107/tuner-v5-ab-2026-08-16/`.
Runtime under test: deployed librccl 2.28.3-develop:2e42aa8 (see EVIDENCE.md).
Plugins: `librccl-tunerv4-dn.so` / `librccl-tunerv5-dn.so`, both built on
amd-mi355x-1 (gcc 11.4.0) from the same `dn/dn-tuner/plugin.c` at commit 2f44a5e.
Rule file both arms: `gen_3n_t1.validated.csv` (the shipped 3-node config).

## Verdict so far

1. **1-node parity: PASS.** (details below)
2. **v5 constants: DEAD on this runtime — proven empirically.** 144 absurd
   overrides (every bw table -> 0.0001 GB/s, TREE latencies -> 10000 us) changed
   NOTHING in RCCL's selected algo/proto/nchannels at any of 18 sizes, at 1 node
   and at 3 nodes. `constprobe/*_extreme_dbg_*.log` prove all ranks applied all
   144 overrides ("applied 144 constants override(s)", llMaxBws[0][0] 39 -> 0.0001)
   and loaded v5 ("TUNER/Plugin: Using DN-TUNER (v5)"). Cause (EVIDENCE.md §2-3):
   this build predates ROCm/rccl 212d82678; on its AMD path the tuner model reads
   `rcclTuningModel` + static tables, never `comm->tunerConstants`.
   Consequence: **the "fix the constants, not the symptoms" plan needs a newer
   RCCL.** In the 2.30.4 tree (`/opt/shared/ylerman/rccl-src`), `hwLatencies` and
   `bwRatio` ARE consumed on AMD (tuning.cc:694,865-904) — the latency side that
   drives the 5-of-6 late switch points — while llMaxBws/perChMax* remain
   non-AMD-only even there.
3. **3-node parity: PASS.** Forward pass (v4 baseline / v5 config) 09:36Z rc=1;
   swapped pass (arms reversed) 10:21Z rc=1. Selections identical 18/18 in every
   run of both passes; rule firings exactly equal (50400 "Applied config" lines
   per arm per pass). Order-balanced pool (14 runs/arm): 17/18 sizes within
   ±1.2%; 8K −4.5% is one 0.01 GB/s print-quantum at 0.22 GB/s, psup 0.31; zero
   sizes both ≥2% and P(sup)-significant. `parity_3n/`, `parity_3n_swapped/`.

4. **The 3n preflight blocker was `amd-nic-metrics-exporter` — causal evidence.**
   With it running: 5 consecutive gate failures (single-run dips 27–73% at one
   random size each; pollers active during every run, so correlation alone could
   not convict — `sampler_node{5,6,7}.log`). With it stopped on nodes 5,6,7
   (user-approved, 10:15–10:21Z, restart verified): first-try pass at 8.7%
   worst spread, the cleanest 3n gate of the day. Overnight's separate blocker
   was real fabric saturation from a cross-leaf 2-node co-tenant ([1-2]); the
   leaf-local [4,9] co-tenant did NOT prevent clean gates.

## 1-node parity (node amd-mi355x-5, jobid 14470, 2026-08-17 07:12–07:25Z)

Three interleaved 7-repeat pair sets, benchmark flags identical to the validator
(`-b 4096 -e 536870912 -f 2 -g 1 -n 20 -w 5 -c 1 -A 1`, 8 ranks x 1 GPU):

| set | slot1 | slot2 | small-size result (4K–32K medians) |
|---|---|---|---|
| `parity_1n/` | v4 | v5 | v5 lower by 3.2–6.3% |
| `revorder/`  | v5 | v4 | v4 lower by 3.3–6.4% (mirror image) |
| `v4v4/`      | v4 | v4 | slot2 lower by up to 2.6% |

The deficit follows the SLOT, not the plugin — a run-order artifact, present even
with identical binaries. **Order-balanced pool (14 runs/arm, slots counter-
balanced): 16 of 18 sizes within ±0.81%, all P(sup) in 0.23–0.67.** The two
residuals: 4K −3.2% is exactly one 0.01 GB/s print-quantum at 0.30 GB/s with
psup 0.37 (no dominance); 128K −1.7% is inside the 2% tolerance.

Selection identity: algo/proto/nchannels IDENTICAL v4-vs-v5 at every size in all
three sets (54 size-points). Load proof: every rank log shows `Using DN-TUNER (v4)`
resp. `(v5)`; the 3n rule file correctly fires zero rules at 1 node in both arms
(`applied-config lines=0`).

**Methodology finding (affects the validator too):** `validate_tuner_config.py`
always runs default before config inside a repeat; at 1 node the second slot of a
back-to-back pair measures up to ~3–6% low at 4K–32K. Validated small-size gains
near the 2% threshold may be slot artifacts; the big 2026-08-04 1-node wins
(+12…+63%) are far above it. Fix candidates: alternate arm order per repeat, or
judge small sizes on order-balanced pairs.

## Raw data map

- `parity_1n/`, `revorder/`, `v4v4/` — stdout + per-rank NCCL INFO logs, PROGRESS
- `constprobe/` — {1n,3n}_{noconst,extreme}.log + dbg logs
- `parity_3n_attempt{1,2}_preflightfail/` — aborted 3n attempts (see RUNLOG)
- analysis: `analyze_parity.py` (cluster workdir copy); pooled numbers in this
  file were produced by the inline script recorded in the session, from
  `parity_1n/` + `revorder/` stdout logs
