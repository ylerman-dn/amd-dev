#!/usr/bin/env python3
"""Test anchors 1,8,24,48 (adds ch=1) on every dataset: the 6 new-collective
grids plus the 3 all_reduce grids, conservative policy."""
import json
import subprocess
import sys

BASE = 'results-tuning/2026-08-18-1-newcolls'
DATASETS = [
    (f'{BASE}/broadcast_1n_merged.csv', None, 'broadcast 1n'),
    (f'{BASE}/reduce_1n_merged.csv', None, 'reduce 1n'),
    (f'{BASE}/broadcast_2n_merged.csv', None, 'broadcast 2n'),
    (f'{BASE}/reduce_2n_merged.csv', None, 'reduce 2n'),
    (f'{BASE}/broadcast_3n_merged.csv', None, 'broadcast 3n'),
    (f'{BASE}/reduce_3n_merged.csv', None, 'reduce 3n'),
    ('results-tuning/2026-08-04-1-sweep1node/merged.csv', None, 'all_reduce 1n'),
    ('results-tuning/2026-08-16-12-variance/variance.json', 2, 'all_reduce 2n'),
    ('results-tuning/2026-08-16-12-variance/variance.json', 3, 'all_reduce 3n'),
]
for path, nodes, name in DATASETS:
    cmd = [sys.executable, 'tools/rccl-sweep/adaptive_search.py', 'replay',
           '--dataset', path, '--anchors', '1,8,24,48', '--margin', '15',
           '--repeat-policy', '3', '--tol', '0.5']
    if nodes:
        cmd += ['--nodes', str(nodes)]
    d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    print(f"{name:15} runs {d['runs']:>3}/{d['full_runs']:>3} "
          f"saving {d['saving_pct']:>5}% exact {d['exact']}/18 "
          f"worst_gap {d['worst_gap_pct']}%")
