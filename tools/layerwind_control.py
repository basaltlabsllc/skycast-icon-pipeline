#!/usr/bin/env python3
"""Control for the layer wind: download ONE real GFS step and print, per deck, at
the grid node nearest (lat, lon): the hand-picked cloudy run, the levels the deck
holds there, and the deck wind (what read_wind_step returns), plus the step's
download MB and seconds.

    python tools/layerwind_control.py gfs <YYYYMMDDHH> <fh> <lat> <lon>

Downloads with curl (byte ranges, the same ranges the build asks for); needs
noaa-gfs-bdp-pds.s3.amazonaws.com. Prints the one node asked for, nothing else."""
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def main(argv):
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
