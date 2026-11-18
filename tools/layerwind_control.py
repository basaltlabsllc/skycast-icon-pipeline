#!/usr/bin/env python3
"""Control for the layer wind: download ONE real step and print, per deck, at the
grid node nearest (lat, lon): the hand-picked cloudy run, the levels the deck
holds there, and the deck wind (what the reader returns), plus the step's
download MB and seconds.

    python tools/layerwind_control.py gfs  <YYYYMMDDHH> <fh> <lat> <lon>
    python tools/layerwind_control.py hrrr <YYYYMMDDHH> <fh> <lat> <lon>
    python tools/layerwind_control.py hrrr-threshold <YYYYMMDDHH> <fh>
    python tools/layerwind_control.py rrfs <YYYYMMDDHH> <fh> <lat> <lon>
    python tools/layerwind_control.py rrfs-threshold <YYYYMMDDHH> <fh>
    python tools/layerwind_control.py aifs <YYYYMMDDHH> <fh> <lat> <lon> [<lat> <lon> ...]
    python tools/layerwind_control.py ifs  <YYYYMMDDHH> <fh> <lat> <lon> [<lat> <lon> ...]

gfs / hrrr / rrfs print the one node asked for plus, for HRRR and RRFS, a count-only positive
control (how many cells have a cloudy level, no coordinates). hrrr-threshold
(and rrfs-threshold) compares, per point, the model's lowest cloudy level with its own
PRES:cloud base at CONDENSATE_CLOUDY candidates and prints counts and shares only.
rrfs goes through the build's own NOMADS rate limiter (90 requests a minute).
aifs / ifs read one step through the build's own reader from the S3 mirror of ECMWF
open data, print each node with a hand computation of every deck (written here, not
pipeline.layerwind) beside the reader's wind, a count-only positive control, the
message params read from the bytes, and requests, MB, seconds, peak RSS and the
projection over the model's wind steps.

Downloads with curl (byte ranges, the same ranges the build asks for); needs
noaa-gfs-bdp-pds.s3.amazonaws.com (gfs), noaa-hrrr-bdp-pds.s3.amazonaws.com (hrrr)
nomads.ncep.noaa.gov (rrfs) or ecmwf-forecasts.s3.eu-central-1.amazonaws.com (aifs, ifs)."""
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
import rrfs.build as rbuild  # noqa: E402
import rrfs.grib as rgrib  # noqa: E402
from globalfeed import grib  # noqa: E402
from globalfeed.models import AIFS, GFS, IFS  # noqa: E402
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


def rrfs_get(stats):
    """curl_get behind the build's own NOMADS rate limiter, counting requests and the seconds the limiter held us."""
    inner = curl_get(stats)
    lim = rbuild.RateLimiter(rbuild.REQUESTS_PER_MINUTE)
    stats.setdefault("waited", 0.0)

    def get(url, rng=None):
        t = time.monotonic()
        lim.acquire()
        stats["waited"] += time.monotonic() - t
        stats["requests"] += 1
        return inner(url, rng)
    return get


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def nearest(lat, lon, lat0, lon0):
    d = (lat - lat0) ** 2 + ((lon - lon0) * np.cos(np.radians(lat0))) ** 2
    return int(np.argmin(d))


def hrrr_node(argv, mod=hgrib, label="HRRR", get_factory=None, steps=49, workers=8, requests_per_step=None):
    from pipeline import wind
    run = datetime.strptime(argv[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    fh, lat0, lon0 = int(argv[3]), float(argv[4]), float(argv[5])
    stats = {"bytes": 0, "requests": 0}
    get = (get_factory or curl_get)(stats)
    t0 = time.monotonic()
    f = mod.read_layer_fields(run, fh, get)
    t1 = time.monotonic()
    ps = f["ps"]
    bounds = layerwind.deck_bounds("ncep", ps)
    decks = layerwind.deck_winds(f["levels"], f["u"], f["v"], f["cloudy"], ps, bounds)
    t2 = time.monotonic()
    six = {k: wind.lattice_from_points(f["lat"], f["lon"], a).astype(np.float32) for k, a in decks.items()}
    t3 = time.monotonic()
    i = nearest(f["lat"], f["lon"], lat0, lon0)
    print(f"{label} {run:%Y-%m-%d %HZ} f{fh:03d}, node {f['lat'][i]:.2f},{f['lon'][i]:.2f}")
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
    if label == "RRFS":
        n = stats["requests"]
        floor = steps * n / rbuild.REQUESTS_PER_MINUTE * 60.0
        print(f"requests per step: {n} (2 idx + {n - 2} ranged); wait the limiter added this step: {stats['waited']:.1f} s")
        print(f"projection: {steps} steps x {step:.1f} s / {workers} workers = {steps * step / workers:.0f} s of work; "
              f"rate-limit floor {steps} x {n} = {steps * n} requests at {rbuild.REQUESTS_PER_MINUTE} a minute = {floor:.0f} s "
              f"(the limiter is shared by all workers); budget {rbuild.WIND_BUDGET_S} s, so the projection is "
              f"{max(steps * step / workers, floor):.0f} s")
        return
    print(f"projection: 49 steps / 8 workers x {step:.1f} s = {49 * step / 8:.0f} s of wind (budget 600 s); "
          f"download-bound lower bound at the same MB per step is the same figure only if bandwidth scales with workers")


def cloud_base_pressure(run, fh, get, mod=hgrib):
    """PRES:cloud base (surface / 2dfld file) in hPa, NaN where the model reports none."""
    url = mod.file_url(run, fh)
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


def hrrr_threshold(argv, mod=hgrib, label="HRRR", get_factory=None):
    run = datetime.strptime(argv[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    fh = int(argv[3])
    get = (get_factory or curl_get)({"bytes": 0, "requests": 0})
    f = mod.read_layer_fields(run, fh, get, keep_condensate=True)
    base = cloud_base_pressure(run, fh, get, mod)
    levels = np.array(f["levels"], float)               # 1000, 950 .. 150
    ps = f["ps"]
    has_base = np.isfinite(base)
    # level index of the base: nearest of the 18 levels on the 50 mb list
    bidx = np.clip(np.rint((1000.0 - base) / 50.0), 0, len(levels) - 1)
    print(f"{label} {run:%Y-%m-%d %HZ} f{fh:03d}: {ps.size} cells, {int(has_base.sum())} with a PRES:cloud base "
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
    print(f"best ALL share {100 * shares[top]:.1f} % at {top:.0e} (informational: CONDENSATE_CLOUDY is "
          f"{layerwind.CONDENSATE_CLOUDY:.0e}, settled by the HRRR control)")
    print(f"peak RSS {peak_rss_mb():.0f} MB")


EC_MIRROR = ("https://data.ecmwf.int/forecasts/", "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/")


def _hand_rh(hum, val, t, p_hpa):
    """Effective RH (%), written out again here from the spec (not pipeline.layerwind):
    q -> e = q p / (0.622 + 0.378 q); RH over water = e / esw; below 0 C x esw / esi."""
    esw = 611.2 * np.exp(17.62 * (t - 273.15) / (t - 30.03))
    esi = np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t) - 0.00728332 * t)
    rh = 100.0 * (val * p_hpa * 100.0 / (0.622 + 0.378 * val)) / esw if hum == "q" else val
    return rh * esw / esi if t < 273.15 else rh


def _hand_deck(levels, u, v, moist, lo, hi, ps, low_deck):
    """One node's deck by hand: members, weights, thickest moist run, weighted mean."""
    mem = [k for k, p in enumerate(levels) if lo <= p <= ps and (p < hi or (low_deck and p <= hi))]
    if not mem:
        return [], [], None, (float("nan"), float("nan"))
    w = {}
    for j, k in enumerate(mem):                     # levels descend: mem[j - 1] is the level below
        below = 0.5 * (levels[mem[j - 1]] + levels[k]) if j > 0 else min(hi, ps)
        above = 0.5 * (levels[k] + levels[mem[j + 1]]) if j + 1 < len(mem) else lo
        w[k] = below - above
    runs, cur = [], []
    for k in mem:
        if moist[k]:
            cur.append(k)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    best = None
    for r in runs:                                  # ground up, so a tie keeps the lower run
        if best is None or sum(w[k] for k in r) > sum(w[k] for k in best):
            best = r
    use = best or mem
    sw = sum(w[k] for k in use)
    return mem, w, best, (sum(w[k] * u[k] for k in use) / sw, sum(w[k] * v[k] for k in use) / sw)


def ecmwf_node(argv):
    import dataclasses
    model = {"aifs": AIFS, "ifs": IFS}[argv[1]]
    model = dataclasses.replace(model, _url=model._url.replace(*EC_MIRROR))
    run = datetime.strptime(argv[2], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    fh = int(argv[3])
    nodes = [(float(argv[k]), float(argv[k + 1])) for k in range(4, len(argv) - 1, 2)]
    stats = {"bytes": 0, "requests": 0}
    inner = curl_get(stats)
    raw_msgs = []

    stats["retries"], stats["waited"] = 0, 0.0

    def get(url, rng=None):
        # the S3 mirror answers 503 SlowDown under bursts: back off and retry, and keep the
        # seconds spent waiting out of the step time (the build reads data.ecmwf.int)
        stats["requests"] += 1
        for attempt in range(8):
            try:
                b = inner(url, rng)
                break
            except subprocess.CalledProcessError:
                if attempt == 7:
                    raise
                stats["retries"] += 1
                t = time.monotonic()
                time.sleep(2.0 * (attempt + 1))
                stats["waited"] += time.monotonic() - t
        if rng is not None:
            raw_msgs.extend(grib.split_messages(b))
        return b

    t0 = time.monotonic()
    raw = grib.read_wind_fields(model, run, fh, get)
    t1 = time.monotonic()
    result = grib.ecmwf_deck_winds(model, raw)
    t2 = time.monotonic()
    levels, hum = grib.ECMWF_WIND[model.name]
    seen = {}
    for m in raw_msgs:
        g = eccodes.codes_new_from_message(m)
        try:
            k = tuple(str(eccodes.codes_get(g, x)) for x in ("shortName", "typeOfLevel", "units"))
            seen.setdefault(k, []).append(int(eccodes.codes_get(g, "level")))
        finally:
            eccodes.codes_release(g)
    print(f"{model.name.upper()} {run:%Y-%m-%d %HZ} f{fh:03d}: {len(raw_msgs)} messages read; params from the bytes:")
    for (short, tol, units), levs in sorted(seen.items()):
        print(f"  {short:3s} {tol:14s} {units:10s} levels {sorted(levs, reverse=True)}")
    lat, lon = result["lat"], result["lon"]
    ps_all = raw[("sfc", "sp")] / 100.0
    for lat0, lon0 in nodes:
        i = int(np.argmin(np.abs(lat - lat0)))
        j = int(np.argmin(np.abs(lon - lon0)))
        ps = float(ps_all[i, j])
        u = [float(raw[(l, "u")][i, j]) for l in levels]
        v = [float(raw[(l, "v")][i, j]) for l in levels]
        rh = [float(_hand_rh(hum, float(raw[(l, hum)][i, j]), float(raw[(l, "t")][i, j]), l)) for l in levels]
        moist = [x >= 90.0 for x in rh]
        print(f"node {lat[i]:.2f},{lon[j]:.2f}: surface pressure {ps:.1f} hPa")
        for k, l in enumerate(levels):
            print(f"  {l:5d} hPa  u {u[k]:7.2f}  v {v[k]:7.2f}  {hum} {float(raw[(l, hum)][i, j]):.3g}  "
                  f"t {float(raw[(l, 't')][i, j]):6.1f}  RH_eff {rh[k]:5.1f} {'moist' if moist[k] else 'dry  '}"
                  f"{'  underground' if l > ps else ''}")
        for d, (key, name) in enumerate(zip(layerwind.DECK_KEYS, ("low", "mid", "high"))):
            a, b = max(0.8 * ps, 150.0), max(0.45 * ps, 150.0)
            lo, hi = ((a, ps), (b, a), (150.0, b))[d]
            mem, w, best, (hu, hv) = _hand_deck(levels, u, v, moist, lo, hi, ps, d == 0)
            ru, rv = float(result[(key, "u")][i, j]), float(result[(key, "v")][i, j])
            ok = (np.isnan(hu) and np.isnan(ru)) or (abs(hu - ru) < 1e-3 and abs(hv - rv) < 1e-3)
            print(f"  {name:4s} deck {hi:.0f}..{lo:.0f}: levels {[levels[k] for k in mem] or 'none'} "
                  f"weights {[round(w[k], 1) for k in mem]}; run {[levels[k] for k in best] if best else 'none (mean of every level)'}; "
                  f"hand u {hu:.3f} v {hv:.3f}; reader u {ru:.3f} v {rv:.3f}; {'MATCH' if ok else 'MISMATCH'}")
    # Positive control, no coordinates printed
    shape = ps_all.shape
    psf = ps_all.reshape(-1).astype(np.float64)
    bounds = layerwind.deck_bounds("ecmwf", psf)
    uu = [raw[(l, "u")].reshape(-1) for l in levels]
    vv = [raw[(l, "v")].reshape(-1) for l in levels]
    mm = [layerwind.moist_from_q(raw[(l, "q")].reshape(-1), raw[(l, "t")].reshape(-1), l) for l in levels]
    plain = layerwind.deck_winds(levels, uu, vv, [np.zeros(psf.shape, bool)] * len(levels), psf, bounds)
    for k, l in enumerate(levels):
        above = l <= psf
        print(f"moist share {l:5d} hPa: {100.0 * mm[k][above].mean():5.1f} % of above-ground cells")
    for d, (key, name) in enumerate(zip(layerwind.DECK_KEYS, ("low", "mid", "high"))):
        lo, hi = bounds[d]
        any_m = np.zeros(psf.shape, bool)
        for k, l in enumerate(levels):
            any_m |= (l >= lo) & (l <= psf) & ((l < hi) | (d == 0) & (l <= hi)) & mm[k]
        diff = np.hypot(result[(key, "u")].reshape(-1) - plain[(key, "u")],
                        result[(key, "v")].reshape(-1) - plain[(key, "v")])
        moved = diff > 0.5
        nan = int(np.isnan(result[(key, "u")]).sum())
        print(f"control {name:4s}: {int(any_m.sum())} of {psf.size} cells hold a moist level; humidity-layer wind "
              f"differs from the whole-deck mean by > 0.5 m/s at {int((moved & any_m).sum())} of them, at "
              f"{int((moved & ~any_m).sum())} clear cells (must be 0), max clear-cell diff "
              f"{float(np.nanmax(np.where(any_m, 0.0, diff))):.2g}; NaN cells {nan}")
    t3 = time.monotonic()
    mb = stats["bytes"] / 1e6
    step = t2 - t0 - stats["waited"]
    steps = len(model.main_steps) + len(model.tail_steps)
    print(f"mirror retries {stats['retries']}, {stats['waited']:.1f} s of backoff (left out of the step time below)")
    print(f"requests {stats['requests']} (1 index + {stats['requests'] - 1} ranged), download {mb:.1f} MB; "
          f"download+decode {t1 - t0:.1f} s, layer rule {t2 - t1:.1f} s, step total {step:.1f} s "
          f"(control extras {t3 - t2:.1f} s, not counted); peak RSS {peak_rss_mb():.0f} MB")
    print(f"projection: {steps} steps x {step:.1f} s / {model.workers} workers = {steps * step / model.workers:.0f} s "
          f"of wind against WIND_BUDGET_S 600 (if the mirror's bandwidth scales with workers)")


def main(argv):
    if len(argv) >= 6 and len(argv) % 2 == 0 and argv[1] in ("aifs", "ifs"):
        return ecmwf_node(argv)
    if len(argv) == 6 and argv[1] == "hrrr":
        return hrrr_node(argv)
    if len(argv) == 4 and argv[1] == "hrrr-threshold":
        return hrrr_threshold(argv)
    if len(argv) == 6 and argv[1] == "rrfs":
        return hrrr_node(argv, rgrib, "RRFS", rrfs_get, steps=len(rbuild.wind_hours(rbuild.STEPS)), workers=rbuild.workers())
    if len(argv) == 4 and argv[1] == "rrfs-threshold":
        return hrrr_threshold(argv, rgrib, "RRFS", rrfs_get)
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
