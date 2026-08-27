#!/usr/bin/env python3
"""Noise dig page: every repeat of every (size, arm) point of the 2026-08-27
L1-only 4n A/B attempts (nodes 1,3,5,7), with spread / stddev / variance and
why each attempt passed or was voided. Reads the per_size_stats.csv sidecars
fetched from the remote ab_out tree. Standalone - not part of the main site."""
import csv
import glob
import html
import statistics
from pathlib import Path

HERE = Path(__file__).parent
SPREAD_LIMIT = 25.0   # % (max-min)/min per (size,arm) point
VOID_FRACTION = 0.20  # >20% noisy points voids the attempt (rc=2)
RES = 0.005           # print resolution floor (values rounded to 0.01)
MAD_Z = 3.5           # Iglewicz-Hoaglin robust z cut
CV_LIMIT = 8.0        # % stddev/mean per point (proposed gate)
CORE_MIN = 7          # a point is usable if >=7 of 9 repeats form a clean core


def mad_core(vals):
    """Repeats within MAD_Z robust-z of the median. MAD=0 falls back to the
    mean absolute deviation (Iglewicz-Hoaglin); scale floored at RES."""
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals])
    scale = 1.4826 * mad if mad > 0 else \
        1.2533 * statistics.mean([abs(v - med) for v in vals])
    scale = max(scale, RES)
    return [v for v in vals if abs(v - med) / scale <= MAD_Z], med, scale

CSS = """body{font-family:-apple-system,system-ui,sans-serif;margin:12px;max-width:1400px;
background:#14161a;color:#d8dbe0}
h1{font-size:20px;color:#f0f2f5}h2{font-size:16px;color:#e6e9ee;margin-top:28px}
.tw{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;font-size:12.5px}
th{border:1px solid #34383f;padding:3px 7px;text-align:center;color:#e6e9ee}
td{border:1px solid #34383f;padding:3px 7px;text-align:right;white-space:nowrap}
td:first-child{text-align:left}
td.runs{font-family:ui-monospace,monospace;font-size:11px}
tr.noisy{background:#331a1a}
tr.noisy td.sp{color:#ff6b6b;font-weight:600}
.g{color:#4ec96a}.b{color:#ff6b6b}.n{color:#8a8f98}.warn{color:#e0c05a}
.note{color:#9aa0a8;font-size:13px}
.dip{color:#ff6b6b;font-weight:600}
summary{cursor:pointer;color:#6ea8ff;font-size:14px;margin:6px 0}
code{background:#1b1e24;padding:1px 5px;border-radius:4px}"""


def fmt(v, nd=3):
    return f"{v:.{nd}g}" if v is not None else ""


def excluded_sizes(d):
    """Sizes the scoped preflight excluded, from the attempt's validate.log."""
    import re
    log = d / "validate.log"
    if not log.exists():
        return set()
    m = re.search(r"EXCLUDED \(\[([\d, ]*)\]\)", log.read_text())
    return {int(x) for x in m.group(1).split(",")} if m and m.group(1).strip() else set()


def attempt_block(d):
    stats = d / "per_size_stats.csv"
    if not stats.exists():
        return None
    rows = list(csv.DictReader(open(stats)))
    if not rows:
        return None
    excl = excluded_sizes(d)
    # attempt-level accounting, same math as the rc=2 gate: points on sizes
    # the scoped preflight excluded do NOT count toward the void fraction
    points = noisy = mad_noisy = 0
    out_rows = []
    for r in rows:
        for arm, runs_key in (("default", "def_runs"), ("config", "cfg_runs")):
            vals = [float(x) for x in r[runs_key].split(";") if x]
            pf_excl = int(r["size_bytes"]) in excl
            if len(vals) < 2 or min(vals) <= 0:
                flag, sp = "no data", None
            else:
                sp = (max(vals) - min(vals)) / min(vals) * 100
                over = sp > SPREAD_LIMIT
                if pf_excl:
                    flag = "preflight-excluded"
                else:
                    flag = "NOISY" if over else "ok"
                    points += 1
                    noisy += over
            if len(vals) >= 2:
                core, c_med, c_scale = mad_core(vals)
            else:
                core, c_med, c_scale = vals, None, None
            core_sp = ((max(core) - min(core)) / min(core) * 100
                       if core and min(core) > 0 else None)
            core_cv = (statistics.stdev(core) / statistics.mean(core) * 100
                       if len(core) > 1 and statistics.mean(core) else None)
            def _cv(vv):
                mm = statistics.mean(vv)
                return (statistics.stdev(vv) / mm * 100
                        if len(vv) > 1 and mm else 0.0)
            # proposed gate: noisy when the point's CV exceeds CV_LIMIT AND
            # its MAD core (>= CORE_MIN repeats) does not come in under it
            core_bad = _cv(vals) > CV_LIMIT and not (
                len(core) >= CORE_MIN and _cv(core) <= CV_LIMIT)
            if flag in ("NOISY", "ok"):
                mad_noisy += core_bad
                mflag = "NOISY" if core_bad else "ok"
            else:
                mflag = flag
            med = statistics.median(vals) if vals else None
            mean = statistics.mean(vals) if vals else None
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            # wobble = stddev relative to the mean (coefficient of variation)
            var = sd / mean * 100 if sd is not None and mean else None
            # bold red = repeats the MAD rule rejects (outside 3.5 robust-z)
            def rejected(v):
                return (c_scale is not None
                        and abs(v - c_med) / c_scale > MAD_Z)
            shown = " ".join(
                f"<span class=dip>{v:g}</span>" if rejected(v)
                else f"{v:g}" for v in vals)
            out_rows.append((int(r["size_bytes"]), arm, len(vals), shown,
                             med, mean, sd, var, sp, flag,
                             len(core), core_sp, core_cv, mflag))
    void = points and noisy > points * VOID_FRACTION
    mvoid = points and mad_noisy > points * VOID_FRACTION
    verdictline = (f"<span class=b>VOIDED (rc=2): {noisy} of {points} points "
                   f"over {SPREAD_LIMIT:.0f}% - above the {VOID_FRACTION:.0%} "
                   f"limit</span>" if void else
                   f"<span class=g>valid: {noisy} of {points} points over "
                   f"{SPREAD_LIMIT:.0f}%</span>")
    mline = (f"<span class=b>still VOID: {mad_noisy} of {points}</span>"
             if mvoid else
             f"<span class=g>valid: {mad_noisy} of {points}</span>")
    h = [f"<h2>{d.name}</h2><p class=note>current rule: {verdictline} "
         f"&nbsp;·&nbsp; proposed CV+core rule: {mline}</p>"]
    h.append("<details><summary>per-point numbers ({} rows)</summary>"
             .format(len(out_rows)))
    h.append("<div class=tw><table><tr><th>size</th><th>arm</th><th>n</th>"
             "<th>repeats (GB/s; red = dip &lt;75% of median)</th>"
             "<th>median</th><th>mean</th><th>stddev</th><th>CV %</th>"
             "<th>spread %</th><th>flag (current)</th>"
             "<th>core n</th><th>core spread %</th><th>core CV %</th>"
             "<th>flag (CV+core)</th></tr>")
    for size, arm, n, shown, med, mean, sd, var, sp, flag, ncore, \
            core_sp, core_cv, mflag in out_rows:
        cls = " class=noisy" if flag == "NOISY" else ""
        sz = (f"{size//1048576}M" if size >= 1048576 and size % 1048576 == 0
              else f"{size//1024}K" if size >= 1024 and size % 1024 == 0
              else str(size))
        h.append(f"<tr{cls}><td>{sz}</td><td>{arm}</td><td>{n}</td>"
                 f"<td class=runs>{shown}</td><td>{fmt(med)}</td>"
                 f"<td>{fmt(mean)}</td><td>{fmt(sd)}</td><td>{fmt(var, 3)}</td>"
                 f"<td class=sp>{fmt(sp, 4) if sp is not None else ''}</td>"
                 f"<td>{flag}</td><td>{ncore}</td>"
                 f"<td>{fmt(core_sp, 4) if core_sp is not None else ''}</td>"
                 f"<td>{fmt(core_cv, 3) if core_cv is not None else ''}</td>"
                 f"<td class={'b' if mflag == 'NOISY' else 'n'}>{mflag}"
                 f"</td></tr>")
    h.append("</table></div></details>")
    return "\n".join(h)


page = [f"<!doctype html><meta charset=utf-8><meta name=viewport "
        f"content='width=device-width,initial-scale=1'>"
        f"<title>4n noise dig</title><style>{CSS}</style>"]
page.append("<h1>Noise dig - L1-only 4n A/B attempts (nodes 1,3,5,7, "
            "2026-08-27)</h1>")
page.append(f"<p class=note>Rule under test: a (size, arm) point is noisy "
            f"when (max-min)/min of its repeats exceeds {SPREAD_LIMIT:.0f}%; "
            f"an attempt is voided (rc=2) when more than "
            f"{VOID_FRACTION:.0%} of points are noisy. Proposed gate (shown for comparison, not deployed): a point is noisy when its CV exceeds 8% AND its MAD core (>=7 repeats within 3.5 robust-z, bold red = rejected) is not itself under 8% CV; void stays at >20% noisy points. stddev is per point over its repeats; CV (coefficient of variation) = stddev/mean (under ~5% = healthy, tens of %% = a broken repeat).</p>")
for d in sorted(HERE.glob("ab_out/*/")):
    b = attempt_block(Path(d))
    if b:
        page.append(b)
(HERE / "noise.html").write_text("\n".join(page))
print("built noise.html")
