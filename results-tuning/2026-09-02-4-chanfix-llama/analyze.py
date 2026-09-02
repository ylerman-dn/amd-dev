#!/usr/bin/env python3
"""Analysis for 2026-09-02-4-chanfix-llama: unit-corrected llama prefill channel retest.

Parses paired finals bench logs (5 arms x 30 rounds) pulled from node amd-mi355x-5,
computes per-arm median tok/s, mean TTFT, mean TPOT, MWU vs none, and paired
per-round diffs vs none.

Usage: python3 analyze.py <benchdir>   # benchdir holds <arm>/bench_roundN.log trees
"""
import re
import sys
import statistics as st
from pathlib import Path

from scipy.stats import mannwhitneyu, wilcoxon

ARMS = ["none", "current", "rs112", "rs168", "rs224"]
ROUNDS = 30

PATTERNS = {
    "tps": re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
    "ttft": re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)"),
    "tpot": re.compile(r"Mean TPOT \(ms\):\s+([0-9.]+)"),
}


def parse(benchdir: Path):
    data = {}
    for arm in ARMS:
        rows = {}
        for r in range(1, ROUNDS + 1):
            f = benchdir / f"lcf_{arm}" / f"bench_round{r}.log"
            if not f.exists():
                continue
            txt = f.read_text(errors="replace")
            row = {}
            for k, pat in PATTERNS.items():
                m = pat.search(txt)
                if m:
                    row[k] = float(m.group(1))
            if "tps" in row:
                rows[r] = row
        data[arm] = rows
    return data


def main():
    benchdir = Path(sys.argv[1])
    data = parse(benchdir)
    none_rows = data["none"]
    none_tps = [none_rows[r]["tps"] for r in sorted(none_rows)]
    med_none = st.median(none_tps)

    print(f"{'arm':<9}{'n':>3}{'med tok/s':>11}{'vs none':>9}{'mTTFT ms':>10}"
          f"{'mTPOT ms':>10}{'MWU p':>12}{'paired med diff':>17}{'Wilcoxon p':>12}")
    for arm in ARMS:
        rows = data[arm]
        tps = [rows[r]["tps"] for r in sorted(rows)]
        ttft = [rows[r]["ttft"] for r in sorted(rows) if "ttft" in rows[r]]
        tpot = [rows[r]["tpot"] for r in sorted(rows) if "tpot" in rows[r]]
        med = st.median(tps)
        rel = 100.0 * (med - med_none) / med_none
        if arm == "none":
            p_s = wp_s = "-"
            pdiff_s = "-"
        else:
            p = mannwhitneyu(tps, none_tps, alternative="two-sided").pvalue
            common = sorted(set(rows) & set(none_rows))
            diffs = [rows[r]["tps"] - none_rows[r]["tps"] for r in common]
            pdiff = st.median(diffs)
            try:
                wp = wilcoxon(diffs).pvalue
                wp_s = f"{wp:.3g}"
            except ValueError:
                wp_s = "n/a"
            p_s = f"{p:.3g}"
            pdiff_s = f"{pdiff:+.1f}"
        print(f"{arm:<9}{len(tps):>3}{med:>11.1f}{rel:>+8.2f}%{st.mean(ttft):>10.1f}"
              f"{st.mean(tpot):>10.2f}{p_s:>12}{pdiff_s:>17}{wp_s:>12}")


if __name__ == "__main__":
    main()
