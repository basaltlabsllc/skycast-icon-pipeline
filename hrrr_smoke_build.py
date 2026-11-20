#!/usr/bin/env python3
# hrrr_smoke_build.py — NOAA HRRR near-ground smoke -> site-hrrr-smoke.
# Builds the newest short (18 h) and long (48 h) runs the live smoke index does
# not carry yet, then writes smoke/latest.json. Exit codes: 0 built, 3 nothing to
# do, 1 failed (the live site stays).
import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from hrrr import smokebuild as sb
from hrrr_build import _exists, _get

BUILT, NOTHING_TO_DO, FAILED = 0, 3, 1


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _fetch_live(url):
    """The live smoke index, {} when there is none."""
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return {}
        raise


def main(argv=None, fetch_live=_fetch_live, get=_get, exists=_exists, now=_now,
         sleep=time.sleep, clock=time.monotonic, build=sb.build_run):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="site-hrrr-smoke")
    p.add_argument("--live-index", default="")
    a = p.parse_args(argv)
    try:
        t0 = clock()
        t = now()
        live = fetch_live(a.live_index)
        want = sb.wanted_runs(exists, t)
        todo = []
        for kind, last in (("short", sb.SHORT_LAST), ("long", sb.LONG_LAST)):
            run = want[kind]
            if run is not None and (live.get(kind) or {}).get("runMs") != sb.run_ms(run):
                todo.append((kind, run, last))
        if not todo:
            print("live smoke index already carries the wanted runs")
            return NOTHING_TO_DO
        out = Path(a.out)
        built = {}
        for kind, run, last in todo:
            left = sb.SMOKE_BUDGET_S - (clock() - t0)
            pack = build(run, last, out, get, sleep, left, clock) if left > 0 else None
            if pack:
                built[kind] = {"run": run, "last": last, "pack": pack}
            else:
                print(f"::warning::smoke {kind} run {run:%Y-%m-%dT%HZ} not built")
        if not built:
            print("FAILED: no smoke run could be built")
            return FAILED
        idx = sb.next_index(live, built, t)
        (out / "smoke").mkdir(parents=True, exist_ok=True)
        (out / "smoke" / "latest.json").write_text(json.dumps(idx, separators=(",", ":")))
        print(f"built smoke {sorted(built)}; {len(idx['past'])} past runs, {clock() - t0:.0f} s")
        return BUILT
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return FAILED


if __name__ == "__main__":
    sys.exit(main())
