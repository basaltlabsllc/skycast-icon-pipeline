# rrfs/build.py — one RRFS run -> ICL1 squares + manifest.json.
#
# The same file layout and manifest shape as hrrr/build.py (tiles/<YYYYMMDDHH>/
# <square>.icl, "model": "rrfs", no blocks, thick, columns or tail keys): the
# app reads it with the decoder it has. RRFS runs at 00/06/12/18Z and is read
# f000..f084 hourly, no history. NOMADS only, under its request limit: a shared
# RateLimiter (rrfs_build.py wraps every request in one) and two workers.
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from pipeline import wind
from pipeline.encode import encode_square
from pipeline.site import build_manifest, tiles_path, wind_path

from .grib import FIELDS, file_url, read_fields, read_layer_wind
from .grid import SQUARES, bin_index, coverage, cut_squares, node_average
from pipeline.squares import square_name

RETRIES = 2          # extra attempts per forecast hour
ATTRIBUTION = "Cloud data: NOAA RRFS"
RUN_HOURS = (0, 6, 12, 18)
LAST_HOUR = 84
STEPS = tuple(range(LAST_HOUR + 1))
WORKFLOW_TIMEOUT_S = 2400  # must match `timeout -k 10 2400` on rrfs_build.py in .github/workflows/rrfs.yml
WIND_MARGIN_S = 120        # kept free after wind for the manifest write and publish
WIND_BUDGET_S = 600        # build_wind gives up after this long: the cover must still go out inside the workflow timeout
# Wind is kept for the groups the cloud squares cover (lat 0-80, lon -180 to -50;
# group corners fall on -10, 10 .. 70 N and -180, -160 .. -60 E, 35 groups);
# the 13 km grid reaches over the pole and across to Asia, which the app never asks of this feed.
WIND_GROUPS = [wind.group_name(la, lo) for la in range(-10, 80, wind.GROUP_DEG) for lo in range(-180, -50, wind.GROUP_DEG)]
# Wind is read every WIND_STEP_H hours (f000, f003 .. f084: 29 steps), not hourly like the cover: a wind step
# costs 23 NOMADS requests on the live idx (21 where every level is one range) and the limit is shared, so
# hourly wind (85 x 23 = 1,955 requests) could never fit WIND_BUDGET_S. Erik chose this 2026-11-18; the ICW1 file lists only the hours it holds and the app bridges the gaps.
WIND_STEP_H = 3
REQUESTS_PER_MINUTE = 90      # NOMADS's own limit could not be read; stay well under 100


class RateLimiter:
    """At most `limit` acquisitions in any `window` seconds, across threads."""

    def __init__(self, limit, window=60.0, clock=time.monotonic, sleep=time.sleep):
        self.limit, self.window, self.clock, self.sleep = limit, window, clock, sleep
        self._stamps = deque()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            while True:
                now = self.clock()
                while self._stamps and self._stamps[0] <= now - self.window:
                    self._stamps.popleft()
                if len(self._stamps) < self.limit:
                    self._stamps.append(now)
                    return
                self.sleep(self._stamps[0] + self.window - now)


def limited(get, limiter):
    """get(url, rng=None) -> bytes, counted against the limiter first."""
    def wrapped(url, rng=None):
        limiter.acquire()
        return get(url, rng)
    return wrapped


def wind_hours(fhs):
    """The forecast hours of fhs that get wind: every WIND_STEP_H-th, f000 included."""
    return [fh for fh in fhs if fh % WIND_STEP_H == 0]


def workers():
    return int(os.environ.get("RRFS_WORKERS", "2"))


def newest_complete_run(exists, now):
    """The newest of the last four 00/06/12/18Z cycles whose f084 idx exists, or
    None. now is timezone-aware UTC."""
    top = now.replace(hour=now.hour - now.hour % 6, minute=0, second=0, microsecond=0)
    for back in range(4):
        run = top - dt.timedelta(hours=6 * back)
        if exists(file_url(run, LAST_HOUR) + ".idx"):
            return run
    return None


def _read_step(run, fh, get, sleep):
    for attempt in range(RETRIES + 1):
        try:
            return read_fields(run, fh, get)
        except Exception:
            if attempt == RETRIES:
                raise
            sleep(2 * (attempt + 1))


def _read_wind_step(run, fh, get, sleep):
    for attempt in range(RETRIES + 1):
        try:
            return read_layer_wind(run, fh, get)
        except Exception:
            if attempt == RETRIES:
                raise
            sleep(2 * (attempt + 1))


class _OverBudget(Exception):
    pass


def build_wind(run, out_dir, get, sleep, fhs, workers, budget_s=WIND_BUDGET_S, clock=time.monotonic):
    """wind/<run>/<group>.icw over the hours fhs; returns the manifest's `wind`
    entry, or None (any failure, or more than budget_s seconds: the files are
    removed and the cover goes out without wind). get is the limiter-wrapped
    get the cover uses: per wind hour the two idx files, then one request per level
    (its U, V, ABSV, CLMR and ICMR rows as one range; the live 500 and 250 mb levels carry an
    extra STRM row in between, so those two are two ranges) and one for PRES:surface:
    23 requests an hour on the live idx, every one counted by the limiter. Groups outside WIND_GROUPS are not written."""
    t0 = clock()
    deadline = t0 + budget_s

    def one(fh):
        if clock() > deadline:
            raise _OverBudget()
        d = _read_wind_step(run, fh, get, sleep)
        return {k: wind.lattice_from_points(d["lat"], d["lon"], d[k]).astype(np.float32)
                for k in d if k not in ("lat", "lon")}

    ex = cf.ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        try:
            futures = [ex.submit(one, fh) for fh in fhs]
            per = []
            for f in futures:
                try:
                    per.append(f.result(timeout=max(0.0, deadline - clock())))
                except cf.TimeoutError:
                    raise _OverBudget()
                if clock() > deadline:
                    raise _OverBudget()
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
        base_hour = int(run.timestamp() // 3600)
        total = wind.write_wind(out_dir, run, [base_hour + fh for fh in fhs], per, names=WIND_GROUPS)
        groups = len(list((Path(out_dir) / wind_path(run)).glob("*.icw")))
    except Exception as e:
        if isinstance(e, _OverBudget):
            print(f"::warning::wind: over budget ({clock() - t0:.0f} s), dropped")
        else:
            print(f"::warning::wind failed: {type(e).__name__}: {e}")
        shutil.rmtree(Path(out_dir) / "wind", ignore_errors=True)
        return None
    print(f"wind: {groups} groups, {len(fhs)} steps, {total / 1e6:.1f} MB, {clock() - t0:.0f} s")
    return wind.manifest_entry(run, layer="cloud")


def build(run, out_dir, get, now, sleep=time.sleep, fhs=None, clock=time.monotonic):
    """Write out_dir/tiles/<YYYYMMDDHH>/<name>.icl for every SQUARES entry and
    out_dir/manifest.json. Returns the square names. Any hour that still fails
    after its retries raises: a build is all or nothing. fhs limits the forecast
    hours (a sample run); such a manifest says complete: false."""
    started = clock()
    out_dir = Path(out_dir)
    full = list(STEPS)
    fhs = full if fhs is None else sorted(set(int(f) for f in fhs))
    # The first hour fixes the bins for the whole run (the grid is fixed); each
    # worker then bins its own hour and hands back only the small node grids.
    first = _read_step(run, fhs[0], get, sleep)
    bins = bin_index(first["lat"], first["lon"])

    def grids(fields):
        return np.stack([node_average(fields[name], bins) for name, _ in FIELDS])

    per_step = [grids(first)]
    have, empty = coverage(per_step[0][0])
    print(f"f{fhs[0]:03d} low cloud: {have} of {have + empty} domain nodes carry a value, {empty} are 255")
    del first
    with cf.ThreadPoolExecutor(max_workers=max(1, workers())) as ex:
        per_step += list(ex.map(lambda fh: grids(_read_step(run, fh, get, sleep)), fhs[1:]))
    base_hour = int(run.timestamp() // 3600)
    hours = [base_hour + fh for fh in fhs]
    tiles = out_dir / tiles_path(run)
    tiles.mkdir(parents=True, exist_ok=True)
    cut = cut_squares(np.stack(per_step, axis=1))        # [fields][steps] -> square [steps][fields]
    names = []
    for sl, so in SQUARES:
        name = square_name(sl, so)
        cube = np.ascontiguousarray(cut[name].transpose(1, 0, 2, 3))
        (tiles / f"{name}.icl").write_bytes(encode_square(sl, so, hours, cube))
        names.append(name)
    # The whole build runs under the workflow's timeout: wind gets what is left
    # of it minus a margin, so a slow cover can never lose the manifest to wind.
    wind_budget = min(WIND_BUDGET_S, WORKFLOW_TIMEOUT_S - (clock() - started) - WIND_MARGIN_S)
    if wind_budget <= 0:
        print("::warning::wind: no time left, skipped")
        wind_entry = None
    else:
        wind_entry = build_wind(run, out_dir, get, sleep, wind_hours(fhs), workers(), budget_s=wind_budget) \
            if wind_hours(fhs) else None
    m = json.loads(build_manifest(run, now, hours, names, 0, complete=fhs == full))
    m["model"] = "rrfs"
    m["attribution"] = ATTRIBUTION
    for k in ("blocks", "thick", "columns", "tail"):
        m.pop(k, None)
    if wind_entry:
        m["wind"] = wind_entry
    (out_dir / "manifest.json").write_text(json.dumps(m, separators=(",", ":")))
    return names
