# 5-node A/B noise: exporter-correlation screening (2026-08-26)

Question: are the per-node NIC-metrics exporters (30 s scrape cadence, 10-11 s
busy) the cause of the same-config spreads that refused/voided every 5-node
A/B attempt on 2026-08-25?

Method: `exporter_correlation.py` (this directory) parses every raw
preflight/default/config benchmark log under the 2026-08-25 stage5n
`ab_out/` tree (222 runs), flags each (run, size) sample whose busbw_ip is
>25% below the same-attempt same-size median (the preflight's own noise
definition, one-sided), reconstructs the approximate wall-clock moment each
size executed from the log mtime and the per-size iteration times, and tests
whether outlier moments cluster on a 30 s or 60 s wall grid (Rayleigh
resultant vs. the alpha=0.01 uniformity threshold).

Results (182 outlier samples, 3688 clean):

- mod 30 s: R = 0.131, reject threshold 0.159 - **no clustering**
- mod 60 s: R = 0.153, reject threshold 0.159 - no clustering
- outliers hit **every** size class, 4K through 512M, including collapses of
  97% at 32M (137.5 -> 3.2 GB/s, reducescatter config r9) and 35% at 512M
- outliers per run: 71 runs had exactly one hit size, 24 had two, 13 had
  three or more (worst: 11 of 18 sizes in one run)

Reading:

1. A wall-aligned periodic disturbance (systemd-timer exporters) is
   **unsupported** - outlier phases are statistically uniform.
2. The signature is transient fabric-level stalls: mostly a single size per
   run (sub-second events), a tail of multi-second events wiping much of a
   run, amplitude far beyond scrape-jitter scale at large sizes.
3. This does NOT fully acquit the exporters: per-node timers that are not
   wall-aligned and drift independently would also produce a flat histogram.
   The definitive test is an A/B window with the exporters stopped
   (needs admin access to the nodes) - still open.

Status of the "quiet cluster fixes 5n" claim: remains retracted; these
attempts ran in an empty-queue window and still failed.
