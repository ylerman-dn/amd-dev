# Rule hit-maps (2026-09-09), made with tools/rccl-sweep/infer_hitmap.py from each server's TUNING logs (on the nodes)
File name = <model>_<setting>_<mode>_<arm>.csv. "hits" = plugin 'Applied config' lines matching that rule (size range + algo/proto/channels),
summed over 8 rank logs. "miss" rows = all_reduce sizes RCCL executed that no rule of that conf covers. executed_channels = channel count RCCL
actually ran at those sizes (count x occurrences). Graphs ON servers: counts are graph-capture + prefill counts (decode is replayed, not re-tuned).
Graphs-off servers: every call counts.
