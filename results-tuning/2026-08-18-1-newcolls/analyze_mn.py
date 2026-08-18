#!/usr/bin/env python3
"""newcolls 2n/3n analysis: merged datasets, adaptive replays, hotspot detector."""
import csv
import io
import json
import statistics
import subprocess
import sys

BASE = 'results-tuning/2026-08-18-1-newcolls'


def parse_raw(path):
    """'== path' separated metrics.csv concatenation -> list of csv rows."""
    txt = open(path).read()
    header, rows = None, []
    for chunk in txt.split('== ')[1:]:
        lines = chunk.splitlines()
        r = list(csv.reader(io.StringIO('\n'.join(lines[1:]))))
        if not r:
            continue
        if header is None:
            header = r[0]
        rows += r[1:]
    return header, rows


# 1. merged grid datasets + replay
results = {}
for coll in ('broadcast', 'reduce'):
    for n in (2, 3):
        header, rows = parse_raw(f'{BASE}/{coll}_{n}n_grid_raw.txt')
        out = f'{BASE}/{coll}_{n}n_merged.csv'
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        p = subprocess.run(
            [sys.executable, 'tools/rccl-sweep/adaptive_search.py', 'replay',
             '--dataset', out, '--anchors', '8,24,48', '--margin', '15',
             '--repeat-policy', '3', '--tol', '0.5'],
            capture_output=True, text=True)
        d = json.loads(p.stdout)
        with open(f'{BASE}/replay_{coll}_{n}n.json', 'w') as fh:
            json.dump(d, fh, indent=2)
        key = f'{coll} {n}n'
        results[key] = {k: d[k] for k in
                        ('runs', 'full_runs', 'saving_pct', 'exact',
                         'n_sizes', 'worst_gap_pct', 'alive_combos')}
        print(key, results[key])
        for row in d['rows']:
            if not row['exact']:
                print('   miss', row['size'], 'grid', row['grid_cfg'],
                      '| adaptive', row['adaptive_cfg'], 'gap%', row['gap_pct'])

# 2. default curves -> hotspot detector input (the 3n Halperin gap)
det_in = f'{BASE}/newcolls_default_curves.csv'
with open(det_in, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['collective', 'num_nodes', 'num_gpus', 'size_bytes',
                'busbw_ip', 'algo', 'proto', 'nchannels'])
    for coll in ('broadcast', 'reduce'):
        for n in (2, 3):
            header, rows = parse_raw(f'{BASE}/{coll}_{n}n_def_raw.txt')
            idx = {c: i for i, c in enumerate(header)}
            by_size = {}
            for r in rows:
                by_size.setdefault(int(r[idx['size_bytes']]), []).append(
                    float(r[idx['busbw_ip']]))
            for size, vals in sorted(by_size.items()):
                w.writerow([coll + '_perf', n, n * 8, size,
                            statistics.median(vals), 'DEFAULT', 'DEFAULT', 0])
for th in ('0.10', '0.05'):
    p = subprocess.run(
        [sys.executable, 'tools/rccl-sweep/detect_hotspots.py', det_in,
         '-o', f'{BASE}/hotspots_mn_th{th[2:]}.csv', '--threshold', th],
        capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if 'Detected' in line:
            print(f'detector th={th}:', line.strip())
    with open(f'{BASE}/hotspots_mn_th{th[2:]}.csv') as fh:
        for i, line in enumerate(fh):
            if i > 0:
                f = line.split(',')
                print(f'   {f[0]} {f[1]}n {f[5]}: drop {f[11]}%')
