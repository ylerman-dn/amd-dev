# SUMMARY — 2026-09-14-4-csvtuner (campaign D): the same conf through RCCL 2.30.4's built-in CSV tuner, no plugin

Question (ylerman): RCCL 2.30.4 ships a built-in CSV tuner (PR #4503) that reads the same conf format our plugin reads. Can our conf be delivered as a plain file, without the plugin .so, and does it
behave the same?

Answer: **Yes. With `NCCL_TUNER_CONFIG_FILE=/opt/rccl/tuner/sc2_plus_amd.conf` and no `NCCL_TUNER_PLUGIN`, RCCL logs "Using built-in CSV tuner, config: .../sc2_plus_amd.conf", applies exactly the same number of rules as our plugin (133,480 / 74,760 / 66,920 applied lines for Qwen / gpt-oss / DeepSeek in both arms), executes the same algo/proto/channels (e.g. Qwen 128 KB: TREE/LL channel{0..63} in both), and delivers the same throughput within 0.5%. The plugin .so is not needed on RCCL >= 2.30; the deliverable is the conf file.**

## Setup

Node 2 (job 21247, after campaign C), 22:05-22:27Z (01:05-01:27 Israel), image `lmsysorg/sglang:v0.5.19-rocm10-mi35x`. Per model one server per arm, d32 traffic (ISL 512 / OSL 512 / conc 32 / 128 prompts),
custom AR off, `RCCL_DDA_ENABLE=0` (so the classic path runs and the tuner is consulted), floor removed, CUDA graphs on, radix off, TUNING logs, 3 reps (rep 1 dropped below).
- arm csv  = `infer_many.sh ... IM_NO_PLUGIN=1 RM_CONF=sc2_plus_amd.conf`: only `NCCL_TUNER_CONFIG_FILE` is set; RCCL's built-in CSV tuner reads it.
- arm plug = our plugin `librccl-tunerv4-dn.so` + the same conf.
Script `run_csvtuner.sh` (staged as `/opt/shared/ylerman/GPU-107/csvtuner-2026-09-14.sh`), driver log `/opt/shared/ylerman/GPU-107/csvtuner-2026-09-14.driver.log`, raw trees `/data/ylerman/csvtuner-2026-09-14/<tag>/` on node 2, fetched here (bench logs, hit-maps).

## Results (`summary.csv`)

| model | arm | tuner line in the rank log | rules loaded | applied lines (8 ranks) | executed for the decode all_reduce | tok/s reps 2-3 |
|---|---|---|---|---|---|---|
| Qwen | csv | `Using built-in CSV tuner, config: /opt/rccl/tuner/sc2_plus_amd.conf` | 16 (+ header skipped: "Skipping line 1 with unknown colltype") | 133,480 | 131072 B: TREE/LL channel{0..63} | 4255, 4254 |
| Qwen | plug | `TUNER/Plugin: Using DN-TUNER (v4)` | 16 (+ header parsed as a dummy rule) | 133,480 | TREE/LL channel{0..63} | 4256, 4257 |
| gpt-oss | csv | built-in CSV tuner, sc2_plus_amd.conf | 16 | 74,760 | (184320 B) | 5939, 5951 |
| gpt-oss | plug | DN-TUNER (v4) | 16 | 74,760 | | 5939, 5925 |
| DeepSeek | csv | built-in CSV tuner, sc2_plus_amd.conf | 16 | 66,920 | (458752 B) | 2118, 2118 |
| DeepSeek | plug | DN-TUNER (v4) | 16 | 66,920 | | 2105, 2112 |

The built-in tuner's log line is `TUNER/CsvTuner: Applied config for allreduce, bytes=N: algo=..., proto=..., channels=...` (no `collType=`), which is why `infer_hitmap.py` reports 0 hits for the csv arm
(`hitmap_csv.csv`); the applied-line counts above were taken with a plain grep and match the plugin's hit-map exactly (`hitmap_plug.csv`).

## Reading

1. Delivery form: a conf file dropped where RCCL looks (`NCCL_TUNER_CONFIG_FILE`, or `share/rccl/tuner/rccl_tuner_gfx950.csv` to replace AMD's table) is a complete substitute for the plugin on this
   stack. No .so, no ABI version to track (RCCL 2.30.4 asks for tuner v5 and falls back to our v4).
2. Semantics identical to the plugin: the built-in tuner also replaces AMD's own table wholesale when a config file is given, so the conf must carry AMD's rules for the sizes it does not
   override (campaign C's sc2_plus_amd.conf does).
3. Differences: the built-in tuner rejects the CSV header line with a warning (harmless) and has no hot-reload; the plugin's hot-reload remains useful for in-model searches (campaign B').
4. What this does not change: the tuner is only consulted on the classic path (DDA off or > 64 MiB) and stock SGLang never reaches it for its all_reduces (AITER), so the delivery question is
   answered but the value question (2026-09-10-2 and tonight's A, B', C) is unchanged.

## Bookkeeping

6 servers, 18 bench reps, 22 min. Allocation 21247 released by the sequencer at 22:27:13Z; no jobs or containers of ours remain on the cluster.
