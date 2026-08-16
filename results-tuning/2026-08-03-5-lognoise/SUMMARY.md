# Does NCCL_DEBUG=INFO distort busbw? Mostly no — and the cost is stdout, not INFO

Job 14048, `amd-mi355x-4`, 1 node, 8 ranks × 1 GPU, all_reduce, 4K–512M f=2, nothing forced.
`-n 20 -w 5 -c 1 -A 1 -Z csv -X <file>`. Three arms × 5 repeats, **interleaved round-robin**
(A,B,C,A,B,C,…) so drift cannot favour one arm. 15/15 rc=0, 36 CSV rows each, `#wrong`=0.

| arm | env | INFO lines produced |
|---|---|---|
| A | `NCCL_DEBUG=VERSION` | 0 |
| B | `NCCL_DEBUG=INFO`, to **stdout** | 51,551 |
| C | `NCCL_DEBUG=INFO`, to **`NCCL_DEBUG_FILE`** (per-rank, `%p`) | 51,551 |

B and C produce **identical INFO volume** (51,551 lines each, verified); only the destination
differs. Arm A produces none. Results were captured with rccl-tests' CSV reporter (`-Z csv -X`)
rather than stdout, so arm B's interleaving cannot corrupt its own measurements.

## Result

In-place busbw, median of 5. `ovlp` = do the arm's min–max range and arm A's overlap?

| size | A med | A range | B med | B vs A | ovlp | C med | C vs A | ovlp |
|---|---|---|---|---|---|---|---|---|
| 4K | 0.32 | 0.32–0.32 | 0.31 | −2.8% | yes | 0.33 | +1.4% | yes |
| 8K | 0.55 | 0.54–0.56 | 0.52 | **−4.0%** | **NO** | 0.55 | +1.1% | yes |
| 16K | 0.86 | 0.85–0.88 | 0.85 | −1.9% | yes | 0.86 | −0.3% | yes |
| 32K | 1.70 | 1.70–1.72 | 1.67 | −1.9% | **NO** | 1.70 | −0.2% | yes |
| 64K | 3.27 | 3.07–3.30 | 3.20 | −2.3% | yes | 3.28 | +0.1% | yes |
| 128K | 6.36 | 5.85–6.40 | 6.15 | −3.3% | yes | 6.32 | −0.7% | yes |
| 256K | 12.66 | 12.59–12.71 | 12.35 | −2.4% | yes | 12.60 | −0.4% | yes |
| 512K | 21.35 | 21.28–21.45 | 20.86 | −2.3% | **NO** | 21.23 | −0.5% | yes |
| 1M | 41.75 | 41.56–42.03 | 40.87 | −2.1% | **NO** | 41.56 | −0.4% | yes |
| 2M | 79.55 | 72.04–79.78 | 78.06 | −1.9% | yes | 79.47 | −0.1% | yes |
| 4M | 128.62 | 128.33–129.06 | 126.71 | −1.5% | **NO** | 128.10 | −0.4% | yes |
| 8M | 197.90 | 197.55–198.19 | 195.73 | −1.1% | **NO** | 197.54 | −0.2% | yes |
| 16M | 268.76 | 268.30–269.59 | 266.97 | −0.7% | **NO** | 268.48 | −0.1% | yes |
| 32M | 317.79 | 317.23–318.10 | 316.63 | −0.4% | **NO** | 317.20 | −0.2% | yes |
| 64M | 355.33 | 355.12–356.62 | 354.66 | −0.2% | **NO** | 355.27 | −0.0% | yes |
| 128M | 379.72 | 379.65–380.03 | 378.96 | −0.2% | yes | 379.89 | +0.0% | yes |
| 256M | 390.69 | 390.41–391.98 | 390.57 | −0.0% | yes | 390.68 | −0.0% | yes |
| 512M | 397.15 | 396.63–397.85 | 396.69 | −0.1% | yes | 397.24 | +0.0% | yes |

## Two conclusions

**1. Producing INFO messages costs nothing measurable.** Arm C is within ±1.4% of arm A at every
size, and its range overlaps arm A's at **all 18 of 18 sizes**. The formatting and writing of
51,551 log lines is not detectable in busbw.

**2. Sending INFO to stdout does cost, and the cost is real but small.** Arm B is slower than
arm A at **18 of 18 sizes** — never faster. Magnitude scales inversely with message size: −1.9%
to −4.0% below 1M, −1.1% to −1.9% at 2M–8M, and ≤0.7% from 16M up, vanishing to −0.0% at 256M.
Ranges are non-overlapping at 9 of 18 sizes, concentrated in the small-to-mid region.

A uniformly-signed 18/18 result is not what random noise produces, so the effect is real even
where individual ranges overlap. But 5 repeats supports "consistent direction, few-percent
magnitude" — **no significance claim is made**.

Since B and C differ *only* in `NCCL_DEBUG_FILE`, the cost is **stdout contention** between the
log stream and the benchmark's own output, not the cost of INFO itself.

### Why the three-arm design was necessary

Arms B and C both run through a `bash -c` wrapper (needed because `NCCL_DEBUG_SUBSYS`'s commas
collide with `srun --export`'s separator); arm A does not. Arm C matching arm A to within ±1.4%
shows the wrapper itself costs nothing, which is what licenses reading A-vs-B as the INFO-to-stdout
effect rather than a wrapper artefact. A two-arm A-vs-B test could not have separated these.

## My original claim was overstated

I asserted "INFO logging distorts timing" and used it to justify treating the INFO capture's
timings as unusable. Partly right, mostly overstated:

- **Right about the direction** — INFO to stdout is consistently slower.
- **Wrong about the magnitude** — at most 4%, and under 1% above 16M. Not "unusable".
- **Wrong about the cause** — I implied INFO itself was expensive. It is not; the contention with
  stdout is.

Practical consequence: the decisions read out of `results-tuning/2026-08-03-4-infocap/` (selection
map, WarpSpeed threshold, channel counts) were never at risk — a 4% timing shift cannot change
which algorithm RCCL picks. Keeping that run's *numbers* out of the baseline was still the right
call, but out of caution rather than necessity.

**The rule this establishes:** when you need INFO *and* trustworthy timings, set
`NCCL_DEBUG_FILE` (with `%p`, so 8 ranks do not share one descriptor) rather than letting INFO
share stdout.

## Files

15 CSVs (`{A,B,C}_rep{1..5}.csv`), 15 stdout logs, and `armC_debug_files.tgz` (arm C's 40
per-rank debug files, archived — 24 MB uncompressed). Arm C's raw debug files were archived
rather than stored loose purely for size; they are the evidence that arm C really produced INFO.
