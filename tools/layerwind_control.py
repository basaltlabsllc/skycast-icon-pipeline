#!/usr/bin/env python3
"""Control for the layer wind: download ONE real step and print, per deck, at the
grid node nearest (lat, lon): the hand-picked cloudy run, the levels the deck
holds there, and the deck wind (what the reader returns), plus the step's
download MB and seconds.

    python tools/layerwind_control.py gfs  <YYYYMMDDHH> <fh> <lat> <lon>
    python tools/layerwind_control.py hrrr <YYYYMMDDHH> <fh> <lat> <lon>
    python tools/layerwind_control.py hrrr-threshold <YYYYMMDDHH> <fh>

gfs / hrrr print the one node asked for plus, for HRRR, a count-only positive
control (how many cells have a cloudy level, no coordinates). hrrr-threshold
compares, per point, HRRR's lowest cloudy level with its own PRES:cloud base at
CONDENSATE_CLOUDY candidates and prints counts and shares only.

Downloads with curl (byte ranges, the same ranges the build asks for); needs
noaa-gfs-bdp-pds.s3.amazonaws.com (gfs) or noaa-hrrr-bdp-pds.s3.amazonaws.com (hrrr)."""
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eccodes  # noqa: E402
import hrrr.grib as hgrib  # noqa: E402
from globalfeed import grib  # noqa: E402
from globalfeed.models import GFS  # noqa: E402
from pipeline import layerwind  # noqa: E402


def curl_get(stats):
    def get(url, rng=None):
        cmd = ["curl", "-sS", "--fail", "--retry", "3", "-o", "-"]
        if rng is not None:
            cmd += ["-r", f"{rng[0]}-{'' if rng[1] is None else rng[1]}"]
        data = subprocess.run(cmd + [url], check=True, capture_output=True).stdout
        stats["bytes"] += len(data)
        return data
    return get


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def nearest(lat, lon, lat0, lon0):
    d = (lat - lat0) ** 2 + ((lon - lon0) * np.cos(np.radians(lat0))) ** 2
    return int(np.argmin(d))


def hrrr_node(argv):
    from pipeline import wind
    run = datetime.strptime(argv[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    fh, lat0, lon0 = int(argv[3]), float(argv[4]), float(argv[5])
    stats = {"bytes": 0}
    get = curl_get(stats)
    t0 = time.monotonic()
    f = hgrib.read_layer_fields(run, fh, get)
    t1 = time.monotonic()
    ps = f["ps"]
    bounds = layerwind.deck_bounds("ncep", ps)
    decks = layerwind.deck_winds(f["levels"], f["u"], f["v"], f["cloudy"], ps, bounds)
    t2 = time.monotonic()
    six = {k: wind.lattice_from_points(f["lat"], f["lon"], a).astype(np.float32) for k, a in decks.items()}
    t3 = time.monotonic()
    i = nearest(f["lat"], f["lon"], lat0, lon0)
    print(f"HRRR {run:%Y-%m-%d %HZ} f{fh:03d}, node {f['lat'][i]:.2f},{f['lon'][i]:.2f}")
    print(f"surface pressure {ps[i]:.1f} hPa; cells {ps.size}")
    for k, lev in enumerate(f["levels"]):
        print(f"  {lev:5d} hPa  u {f['u'][k][i]:7.2f}  v {f['v'][k][i]:7.2f}  "
              f"{'cloudy' if f['cloudy'][k][i] else 'clear '}{'  underground' if lev > ps[i] else ''}")
    for d, (key, name) in enumerate(zip(layerwind.DECK_KEYS, ("low", "mid", "high"))):
        lo, hi = bounds[d][0][i], bounds[d][1][i]
        members = [lev for k, lev in enumerate(f["levels"]) if lo <= lev <= min(hi, ps[i]) and (lev < hi or d == 0)]
        cl = [lev for lev in members if f["cloudy"][f["levels"].index(lev)][i]]
        runs, cur = [], []
        for lev in members:
            if lev in cl:
                cur.append(lev)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        best = max(runs, key=lambda r: abs(r[0] - r[-1]), default=[])
        print(f"{name:4s} deck {hi:.0f}..{lo:.0f} hPa: levels {members or 'none'}; cloudy {cl or 'none'}; "
              f"run {best or 'none (mean of every level)'}; "
              f"wind u {decks[(key, 'u')][i]:.2f} v {decks[(key, 'v')][i]:.2f} m/s")
    # Positive control, no coordinates: where a deck holds cloud, its cloud-layer wind must differ
    # from the plain mean over its levels at some cells (and at none where it is clear).
    plain = layerwind.deck_winds(f["levels"], f["u"], f["v"], [np.zeros(ps.shape, bool)] * len(f["levels"]), ps, bounds)
    for d, (key, name) in enumerate(zip(layerwind.DECK_KEYS, ("low", "mid", "high"))):
        lo, hi = bounds[d]
        any_cloud = np.zeros(ps.shape, bool)
        for k, lev in enumerate(f["levels"]):
            inside = (lev >= lo) & (lev <= ps) & ((lev < hi) | (d == 0) & (lev <= hi))
            any_cloud |= inside & f["cloudy"][k]
        diff = np.hypot(decks[(key, "u")] - plain[(key, "u")], decks[(key, "v")] - plain[(key, "v")])
        moved = diff > 0.5
        print(f"control {name:4s}: {int(any_cloud.sum())} cells hold a cloudy level; "
              f"cloud-layer wind differs from the plain deck mean by > 0.5 m/s at {int((moved & any_cloud).sum())} of them, "
              f"and at {int((moved & ~any_cloud).sum())} clear cells (must be 0)")
    mb = stats["bytes"] / 1e6
    print(f"download {mb:.1f} MB (2 idx + messages); download+decode+rotate {t1 - t0:.1f} s, layer rule {t2 - t1:.1f} s, "
          f"lattice x6 {t3 - t2:.1f} s, step total {t3 - t0:.1f} s; peak RSS {peak_rss_mb():.0f} MB")
    step = t3 - t0
    print(f"projection: 49 steps / 8 workers x {step:.1f} s = {49 * step / 8:.0f} s of wind (budget 600 s); "
          f"download-bound lower bound at the same MB per step is the same figure only if bandwidth scales with workers")


def cloud_base_pressure(run, fh, get):
    """PRES:cloud base (surface file) in hPa, NaN where HRRR reports none."""
    url = hgrib.file_url(run, fh)
    rows = hgrib._rows(get(url + ".idx").decode())
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    k = next(i for i, r in enumerate(rows) if len(r) > 5 and r[3] == "PRES" and r[4] == "cloud base" and r[5] == want)
    g = eccodes.codes_new_from_message(get(url, hgrib._range_of(rows, k)))
    try:
        kind = str(eccodes.codes_get(g, "typeOfLevel"))
        assert kind == "cloudBase", kind
        vals = eccodes.codes_get_values(g)
        miss = eccodes.codes_get(g, "missingValue")
    finally:
        eccodes.codes_release(g)
    return np.where(np.isclose(vals, miss) | ~np.isfinite(vals) | (vals <= 0) | (vals > 2e5), np.nan, vals) / 100.0


def hrrr_threshold(argv):
    run = datetime.strptime(argv[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    fh = int(argv[3])
    get = curl_get({"bytes": 0})
    f = hgrib.read_layer_fields(run, fh, get, keep_condensate=True)
    base = cloud_base_pressure(run, fh, get)
    levels = np.array(f["levels"], float)               # 1000, 950 .. 150
    ps = f["ps"]
    has_base = np.isfinite(base)
    # level index of the base: nearest of the 18 levels on the 50 mb list
    bidx = np.clip(np.rint((1000.0 - base) / 50.0), 0, len(levels) - 1)
    print(f"HRRR {run:%Y-%m-%d %HZ} f{fh:03d}: {ps.size} cells, {int(has_base.sum())} with a PRES:cloud base "
          f"({100.0 * has_base.mean():.1f} %)")
    nb = int(has_base.sum())
    print("share within one level / exact level, two denominators: BOTH = cells with a base and a cloudy level "
          "(shrinks as the threshold rises, so it flatters a strict one); ALL = every cell with a base (fair)")
    print("threshold kg/kg  | BOTH n  within1  exact | ALL n  within1  exact | base, no cloudy level | cloudy level, no base")
    shares = {}
    for thr in (1e-6, 1e-5, 1e-4):
        low = np.full(ps.shape, -1.0)               # index of the lowest cloudy level above ground, -1 = none
        for k in range(len(levels) - 1, -1, -1):    # from the top down so the lowest (highest pressure) wins last
            hit = (f["condensate"][k] >= thr) & (levels[k] <= ps)
            low = np.where(hit, k, low)
        has_cloudy = low >= 0
        both = has_base & has_cloudy
        d = np.abs(low[both] - bidx[both])
        n1, n0, n = int((d <= 1).sum()), int((d == 0).sum()), int(both.sum())
        print(f"{thr:10.0e}       | {n:8d} {100 * n1 / n:6.1f} % {100 * n0 / n:6.1f} % | {nb:8d} {100 * n1 / nb:6.1f} % "
              f"{100 * n0 / nb:6.1f} % | {int((has_base & ~has_cloudy).sum()):14d} | {int((~has_base & has_cloudy).sum()):14d}")
        print(f"                   counts: both within1 {n1}, exact {n0}")
        shares[thr] = n1 / nb
    top = max(shares, key=shares.get)
    pick = 1e-5 if shares[1e-5] >= shares[top] - 0.02 else top      # a tie within 2 points keeps 1e-5
    print(f"best ALL share {100 * shares[top]:.1f} % at {top:.0e}; chosen {pick:.0e} "
          f"({100 * shares[pick]:.1f} %; a tie within 2 points keeps 1e-5)")
    print(f"peak RSS {peak_rss_mb():.0f} MB")


def main(argv):
    if len(argv) == 6 and argv[1] == "hrrr":
        return hrrr_node(argv)
    if len(argv) == 4 and argv[1] == "hrrr-threshold":
        return hrrr_threshold(argv)
    if len(argv) != 6 or argv[1] != "gfs":
        sys.exit(__doc__)
    run = datetime.strptime(argv[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    fh, lat0, lon0 = int(argv[3]), float(argv[4]), float(argv[5])
    stats = {"bytes": 0}
    get = curl_get(stats)
    t0 = time.monotonic()
    raw = grib.read_wind_fields(GFS, run, fh, get)
    t1 = time.monotonic()
    result = grib.gfs_deck_winds(raw)
    t2 = time.monotonic()
    ranges = [k for k in raw if isinstance(k, tuple)]
    i = int(np.argmin(np.abs(result["lat"] - lat0)))
    j = int(np.argmin(np.abs(result["lon"] - lon0)))
    print(f"GFS {run:%Y-%m-%d %HZ} f{fh:03d}, node {result['lat'][i]:.2f},{result['lon'][j]:.2f}")
    ps = float(raw[("sfc", "pres")][i, j]) / 100.0
    print(f"surface pressure {ps:.1f} hPa; messages {len(ranges)}")
    levels = sorted({k[0] for k in raw if isinstance(k, tuple) and k[0] != "sfc"}, reverse=True)
    for lev in levels:
        print(f"  {lev:5d} hPa  u {raw[(lev, 'u')][i, j]:7.2f}  v {raw[(lev, 'v')][i, j]:7.2f}  "
              f"tcdc {raw[(lev, 'tcdc')][i, j]:6.1f}{'  underground' if lev > ps else ''}")
    names = ("low", "mid", "high")
    for d, (key, name) in enumerate(zip(layerwind.DECK_KEYS, names)):
        lo, hi = layerwind.deck_bounds("ncep", np.array([ps]))[d]
        members = [l for l in levels if lo[0] <= l <= min(hi[0], ps) and (l < hi[0] or d == 0)]
        cloudy = [l for l in members if raw[(l, "tcdc")][i, j] >= layerwind.TCDC_CLOUDY]
        # thickest unbroken run by count of consecutive members, for display only
        runs, cur = [], []
        for l in members:
            if l in cloudy:
                cur.append(l)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        best = max(runs, key=lambda r: abs(r[0] - r[-1]), default=[])
        u, v = result[(key, "u")][i, j], result[(key, "v")][i, j]
        print(f"{name:4s} deck {hi[0]:.0f}..{lo[0]:.0f} hPa: levels {members or 'none'}; "
              f"cloudy {cloudy or 'none'}; run {best or 'none (mean of every level)'}; "
              f"wind u {u:.2f} v {v:.2f} m/s")
    mb = stats["bytes"] / 1e6
    print(f"download {mb:.1f} MB (index + {len(ranges)} messages); download+decode {t1 - t0:.1f} s, "
          f"layer rule {t2 - t1:.1f} s, step total {t2 - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv)
