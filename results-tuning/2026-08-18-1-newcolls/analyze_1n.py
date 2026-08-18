#!/usr/bin/env python3
"""1n analysis for the newcolls effort: replay summaries + 1+2 rotations."""
import json
import subprocess
import sys

BASE = 'results-tuning/2026-08-18-1-newcolls'

r = json.load(open(f'{BASE}/replay_reduce_1n.json'))
print('reduce 1n rep3:', {k: r[k] for k in
      ('runs', 'full_runs', 'saving_pct', 'exact', 'worst_gap_pct')})
for row in r['rows']:
    if not row['exact']:
        print('  miss', row['size'], 'grid', row['grid_cfg'],
              '| adaptive', row['adaptive_cfg'], 'gap%', row['gap_pct'])

for coll in ('broadcast', 'reduce'):
    for rot in (0, 1, 2):
        p = subprocess.run(
            [sys.executable, 'tools/rccl-sweep/adaptive_search.py', 'replay',
             '--dataset', f'{BASE}/{coll}_1n_merged.csv',
             '--anchors', '8,24,48', '--margin', '15',
             '--repeat-policy', '1+2', '--rotation', str(rot), '--tol', '0.5'],
            capture_output=True, text=True)
        d = json.loads(p.stdout)
        print(f'{coll} 1n 1+2 rot{rot}:',
              {k: d[k] for k in ('runs', 'saving_pct', 'exact', 'worst_gap_pct')})
