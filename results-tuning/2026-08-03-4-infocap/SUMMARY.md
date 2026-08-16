# NCCL_DEBUG=INFO diagnostic capture

Job 14048, `amd-mi355x-4`, 1 node, 8 ranks × 1 GPU, all_reduce, 4K–512M f=2, nothing forced.
`-n 5 -w 1 -c 1 -A 1`, `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH,ENV`.
Log: `info.log` (2,175,542 bytes, 21,636 lines, 21,563 `NCCL INFO` lines).

**Diagnostic run — produces no throughput numbers.** INFO logging distorts timing; the baseline
in `results-tuning/2026-08-03-3-defaults/` is the measurement of record.

Run validity: 18 data rows, all 36 `#wrong` fields 0, `# Out of bounds values : 0 OK`.
Stack: RCCL 2.28.3-develop:2e42aa8, ROCm 7.2.0.0-43, rccl-tests 2.17.9-develop:2e42aa8.

## Headline finding: `-A 1`'s channel column is not what runs

`-A 1` is correct for **algo/proto** at all 18 sizes — it matches INFO's
`AllReduce: N Bytes -> Algo X proto Y` exactly. It is **wrong for nchannels**:

| size | `-A 1` says | INFO `channel{Lo..Hi}` | real |
|---|---|---|---|
| 2M | 56 | `{0..51}` | **52** |
| 4M | 56 | `{0..51}` | **52** |
| 8M | 56 | `{0..53}` | **54** |
| 16M, 32M | 56 | `{0..55}` | 56 |
| 64M | 56 | `{0..221}` | **222** |
| 128M | 56 | `{0..221}` | **222** |
| 256M | 56 | `{0..222}` | **223** |
| 512M | 56 | `{0..223}` | **224** |

4K–1M agree (1,1,1,2,4,8,16,16,32). So the column reports a planned ceiling; the real count is
trimmed below it at 2M–8M and multiplied ~4× above the WarpSpeed threshold. The named mechanism
for the trimming is in the binary: `RCCL tuning model overrides nchannels to %i, channels may be
decreased further due to MinTrafficPerchannel thresholds`.

Consequence: any channel number recorded from `-A 1` is unreliable, most severely under
WarpSpeed where it understates by ~4×. Channel verification needs the INFO span. Recorded in
the project CLAUDE.md verification rules.

## WarpSpeed threshold: 67108864 bytes (64 MiB)

The log does not print the value, but brackets it precisely. Sizes 4K–32M each log:

```
RCCL WarpSpeed not enabled for AllReduce at 33554432 bytes as it below the warpSpeed threshold
```

(14 distinct sizes, 4096 → 33554432; nothing at 67108864 or above). From 64M up instead:

```
RCCL Warp Speed Channels set to 224. Warps per block is set to 4
```

The byte value comes from the binary default of `rcclParamWarpSpeedARThreshold()`
(`mov $0x4000000` = 67108864), env name `RCCL_WARP_SPEED_AR_THRESHOLD`, unset in this run.
Engages at 4 of 18 sizes: 64M, 128M, 256M, 512M. Matches AMD's documented "from 64MB" and our
measured RING→`RING*` transition.

Once per rank at init (8 times), a capability line: `WarpSpeed enabled:
warpSpeedChannelMultiplier 4, maxNchannels 448, nc 224` — note 224 = 56 × 4. This is not
per-call engagement; the per-call signal is the `RING*` column and the "not enabled" message.

## 48-vs-56: answered — no 48 cap is in effect

The observed channel counts settle it. From the same `channel{Lo..Hi}` lines used for the table
above: **52, 54, 56, 222, 223, 224** — all above 48. A run cannot exceed a cap that is being
applied, so the cap is not applied here. Reproduce with:

```
grep -oE "AllReduce: [0-9]+ Bytes -> Algo [A-Z]+ proto [A-Z]+ channel\{Lo\.\.Hi\}=\{0\.\.[0-9]+\}" info.log | sort -u
```

An earlier version of this file said "NOT answered by the log", reasoning that the cap *message*
never printed (zero matches for `MaxChannels` / `capping` / `rcclRestrictMaxChannels` in 21,636
lines). That was a mistake: the absence of the message plus channel counts above 48 is the
confirmation, not a gap in it. Two separate questions were conflated.

**Still open, narrower:** under what conditions the cap *would* fire. Disassembly of
`_Z23rcclRestrictMaxChannelsP8ncclCommRi` shows two early returns — one on a struct field being
`< 2`, one on `IsArchMatch(..., "gfx950")` — with the field identified as `nNodes` by calibrating
against an adjacent log call. Our run is `nRanks 8 nNodes 1`. That is **inference from
disassembly, not a verified fact**, and it does not belong in `findings/`. A 2-node run decides
it: if channels drop to 48, confirmed.

Same function, if `NCCL_MAX_NCHANNELS` is set: cap is `min(value, 64)`; unset → 48;
set-but-unparseable → 48 plus the "ignoring invalid" message.

## Does WarpSpeed force RING? Not demonstrated here

`Overriding.*algorithm` — **no matches**. The message exists in the binary
(`Overriding %s algorithm with RING for nccl%s at %zu bytes as WarpSpeed is requested and only
supports RING`) but never fired, because the tuner had already chosen RING from 512K up, so
there was nothing to override. To observe the override path you would need WarpSpeed active
where the tuner wants TREE — e.g. `RCCL_WARP_SPEED_AR_THRESHOLD` below ~256K, or
`RCCL_WARP_SPEED_FORCE_ENABLE`.

## Other absences worth recording

- `tuning model cannot override nchannels` — **no matches**; no `tuning model` line at all.
  Every rank instead logs `RCCL Channel Tuning not applied`. The tuning model did not engage.
- No `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` / `RCCL_WARP_SPEED_*` echoes and no
  `set by environment` lines — none were set, as intended for a defaults run.
- `minNChannels:-2` logged once per collective by every rank, identical at all 18 sizes. A
  negative channel count looks like a sentinel or uninitialised value; unexplained, flagged.

## Method caveat

With `NCCL_DEBUG=INFO` on the same fd, INFO lines interleave mid-row and **no rccl-tests data
row survives as a single intact line**. Field-position parsing of such a log returns garbage.
The table above was built by extracting row fragments separately. Any future parser must not
assume whole-line rows in an INFO log.
