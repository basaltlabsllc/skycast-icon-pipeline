#!/usr/bin/env python3
"""tools/check_heights.py — the control for the cloud-height feed (ICH1).

Given one built heights file (a path, or the URL of a square's .ich), a run,
a step and a place, it reads DWD's RAW icosahedral cloud cover on model levels
(CLC) and half-level heights (HHL) of that run and step itself — no regrid,
no kit, none of the build's remapping — takes the raw cell nearest the feed's
node (by CLAT/CLON), and prints side by side

  * what the file publishes at that node and step: the bytes, decoded to m and
    ft above sea level and ft above ground, and
  * the base and top of each deck read straight from the raw column, with the
    difference in metres and in model levels.

The deck rule itself (which levels are low and mid, the 5 % threshold) is
pipeline/heights.py's, applied here to the raw column; what this checks is
everything between DWD's files and the published bytes.

READ THIS BEFORE JUDGING A DIFFERENCE. The feed's node is a point of the
regridded 0.25 degree lattice: its cover and heights are DWD's weighted blend
of the icosahedral cells around it, not one cell. So at some nodes a deck's
base or top lands one model level away from the nearest raw cell's (a cell
with 4 % cover beside one with 9 % blends to either side of the threshold;
the ground under the node is a blend too). A difference of one model level is
therefore expected at some nodes. It is reported, not hidden: every line
carries its difference, and the last line says SAME or DIFFERENT and by how
many levels. More than one level, at more than the odd node, is a fault.

It only reads: it publishes nothing and writes nothing but the DWD files it
downloads (about 115 MB: CLAT, CLON, 36 HHL and 35 CLC files), into a
temporary folder it deletes, or into --cache DIR, where they are kept and
reused by the next check of the same run.

  python tools/check_heights.py <file-or-url.ich> <run YYYYMMDDHH> <step> <lat> <lon> [--cache DIR]

Needs eccodes (requirements.txt) to read the raw GRIB cells; cdo is not used.
"""
import argparse
import bz2
import datetime as dt
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import dwd, heights                              # noqa: E402
from pipeline.history import hour_of                           # noqa: E402
from pipeline.squares import square_name                       # noqa: E402

FT = 3.28084
KM_PER_DEG = 111.2
LABELS = ("low base", "low top", "mid base", "mid top")


# ---- the pure parts -----------------------------------------------------------------

def node_of(d, lat, lon):
    """(row, column, node lat, node lon): the square's lattice node nearest
    the place. d: a decoded ICH1 (sw_lat, sw_lon, spacing_mdeg, rows, cols)."""
    sp = d["spacing_mdeg"] / 1000.0
    r = int(np.floor((lat - d["sw_lat"]) / sp + 0.5))
    c = int(np.floor((lon - d["sw_lon"]) / sp + 0.5))
    if not (0 <= r < d["rows"] and 0 <= c < d["cols"]):
        raise ValueError(f"{lat}, {lon} is not in this square ({square_name(d['sw_lat'], d['sw_lon'])}: "
                         f"{d['sw_lat']}..{d['sw_lat'] + sp * (d['rows'] - 1):g} N, "
                         f"{d['sw_lon']}..{d['sw_lon'] + sp * (d['cols'] - 1):g} E)")
    return r, c, d["sw_lat"] + r * sp, d["sw_lon"] + c * sp


def nearest_cell(clat, clon, lat, lon):
    """(index, km): the raw cell whose centre is nearest (lat, lon), degrees."""
    dlon = (np.asarray(clon, dtype=float) - lon + 180.0) % 360.0 - 180.0
    d2 = (np.asarray(clat, dtype=float) - lat) ** 2 + (dlon * np.cos(np.radians(lat))) ** 2
    i = int(np.argmin(d2))
    return i, float(np.sqrt(d2[i]) * KM_PER_DEG)


def step_index(step_hours, run, step):
    """Where the run's step sits among the file's hours."""
    want = hour_of(run) + int(step)
    if want not in step_hours:
        raise ValueError(f"run {dwd.run_id(run)} +{step} h (unix hour {want}) is not in the file: it holds "
                         f"{len(step_hours)} hours, {step_hours[0]}..{step_hours[-1]}")
    return list(step_hours).index(want)


def levels_apart(a_m, b_m, hhl_col):
    """How many model levels lie between two heights: the distance between
    the half levels of the column nearest each."""
    hhl_col = np.asarray(hhl_col, dtype=float)
    return abs(int(np.argmin(np.abs(hhl_col - a_m))) - int(np.argmin(np.abs(hhl_col - b_m))))


def _degrees(a):
    a = np.asarray(a, dtype=float)
    return np.degrees(a) if np.nanmax(np.abs(a)) <= np.pi + 1e-6 else a


def _plural(n):
    return f"{n} level" if n == 1 else f"{n} levels"


def _side(b, ground):
    """One published byte in words."""
    b = int(b)
    if b == heights.NO_LEVEL:
        return f"byte {b:3d} = no cloud level in the deck"
    if b == heights.UNKNOWN:
        return f"byte {b:3d} = not read"
    m = heights.byte_metres(b)
    if m is None:
        return f"byte {b:3d} = not a value the feed writes"
    return f"byte {b:3d} = {m:.0f} m = {m * FT:.0f} ft above sea level, {(m - ground) * FT:.0f} ft above ground"


def report(blob, run, step, lat, lon, read):
    """The lines of one check. read(field, name) -> the raw cells of one DWD
    file (1-D, DWD's cell order)."""
    d = heights.decode_height_square(blob)
    r, c, nlat, nlon = node_of(d, lat, lon)
    k = step_index(d["step_hours"], run, step)
    rec = d["heights"][k, r, c]
    lines = [f"file: {d['format']}, square {square_name(d['sw_lat'], d['sw_lon'])}, {d['rows']} x {d['cols']} nodes "
             f"every {d['spacing_mdeg'] / 1000:g} degrees, {len(d['step_hours'])} hours",
             f"node: row {r}, column {c} = {nlat:.3f}, {nlon:.3f} (asked for {lat:.3f}, {lon:.3f})",
             f"step: run {dwd.run_id(run)} +{int(step)} h = unix hour {d['step_hours'][k]}, the file's hour {k}"]

    clat = _degrees(read("CLAT", dwd.invariant_file_name(run, "CLAT")))
    clon = _degrees(read("CLON", dwd.invariant_file_name(run, "CLON")))
    i, km = nearest_cell(clat, clon, nlat, nlon)
    hhl = np.array([float(read("HHL", dwd.invariant_file_name(run, "HHL", lv))[i]) for lv in dwd.HEIGHT_HHL_LEVELS])
    clc = np.array([float(read("CLC", dwd.model_file_name(run, step, lv, "CLC"))[i]) for lv in dwd.HEIGHT_MLEVELS])
    ground = float(hhl[-1])
    lines.append(f"raw cell: number {i} at {float(clat[i]):.3f}, {float(clon[i]):.3f}, {km:.1f} km from the node, "
                 f"surface {ground:.0f} m = {ground * FT:.0f} ft above sea level (the ground used on both sides)")

    low, mid = (m[:, 0] for m in heights.deck_masks(hhl[:, None]))
    bounds = heights.deck_bounds(clc[:, None], hhl[:, None], heights.CLC_THRESHOLD_PCT)[:, 0]
    unknown = heights.deck_unknown(clc[:, None], hhl[:, None])[:, 0]
    raw_flag = bool(heights.top_at_reach(clc[:, None], hhl[:, None], heights.CLC_THRESHOLD_PCT)[0])

    lines.append(f"decks at this cell: low = levels {dwd.HEIGHT_MLEVELS[int(np.argmax(low))]}-{dwd.HEIGHT_MLEVELS[-1]}, "
                 + (f"mid = levels {dwd.HEIGHT_MLEVELS[int(np.argmax(mid))]}-"
                    f"{dwd.HEIGHT_MLEVELS[len(mid) - 1 - int(np.argmax(mid[::-1]))]}" if mid.any() else "mid = none")
                 + f"; cloud = cover >= {heights.CLC_THRESHOLD_PCT:g} %")
    lines.append("")
    worst, differs, blind = 0, False, False
    for n, label in enumerate(LABELS):
        b = int(rec[n])
        pub = heights.byte_metres(b)
        raw = None if np.isnan(bounds[n]) else float(bounds[n])
        if unknown[n // 2]:
            raw_txt, verdict = "raw not readable (a level's cover or a height is missing)", "cannot be compared"
            blind = True
        elif raw is None:
            raw_txt = "raw: no cloud level in the deck"
            if b == heights.NO_LEVEL:
                verdict = "same"
            elif pub is None:
                verdict, blind = "cannot be compared", True
            else:
                verdict, differs = "DIFFERENT: published has a cloud level, the raw cell none", True
        else:
            raw_txt = f"raw {raw:.0f} m = {raw * FT:.0f} ft above sea level, {(raw - ground) * FT:.0f} ft above ground"
            if pub is None and b == heights.NO_LEVEL:
                verdict, differs = "DIFFERENT: the raw cell has a cloud level, published none", True
            elif pub is None:
                verdict, blind = "cannot be compared", True
            else:
                apart = levels_apart(pub, raw, hhl)
                worst = max(worst, apart)
                differs = differs or abs(pub - raw) > heights.HEIGHT_STEP_M / 2
                verdict = f"difference {pub - raw:+.0f} m, {_plural(apart)}"
        lines.append(f"{label}: {_side(b, ground)} | {raw_txt} | {verdict}")
    pub_flag = bool(int(rec[heights.FLAGS]) & heights.FLAG_TOP_AT_REACH)
    lines.append(f"top above reach (the mid top is the top of the highest level read, {dwd.HEIGHT_MLEVELS[0]}): "
                 f"published {'yes' if pub_flag else 'no'}, raw {'yes' if raw_flag else 'no'}")
    lines.append("")
    lines.append(f"raw column, levels with cover >= 1 % (level, bottom-top ft above ground, m above sea level, cover):")
    shown = 0
    for j in range(len(clc) - 1, -1, -1):
        if not clc[j] >= 1.0:
            continue
        shown += 1
        deck = "low" if low[j] else ("mid" if mid[j] else "above the mid deck")
        mark = "" if clc[j] >= heights.CLC_THRESHOLD_PCT else " (under the threshold)"
        lines.append(f"  L{dwd.HEIGHT_MLEVELS[j]}  {(hhl[j + 1] - ground) * FT:6.0f}-{(hhl[j] - ground) * FT:6.0f} ft  "
                     f"{hhl[j + 1]:6.0f}-{hhl[j]:6.0f} m  {clc[j]:5.1f} %  {deck}{mark}")
    if not shown:
        lines.append("  none")
    lines.append("")
    if differs or pub_flag != raw_flag:
        how = ("by less than one model level" if worst == 0 else
               "by one model level" if worst == 1 else f"by {worst} model levels")
        lines.append(f"DIFFERENT {how}" + (" (the flag differs)" if pub_flag != raw_flag else "")
                     + ": the node is a blend of the cells around it, so up to one level is expected at some nodes; "
                       "more is not.")
    elif blind:
        lines.append("NOT COMPARED in full: a value above was not read on one side; the rest are the SAME.")
    else:
        lines.append(f"SAME: every published value is within its {heights.HEIGHT_STEP_M:g} m rounding of the raw read.")
    return lines


# ---- reading the file and DWD ----------------------------------------------------------

def _fetch(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def load_file(src, fetch=_fetch):
    """The .ich bytes from a path or an http(s) URL."""
    if src.startswith(("http://", "https://")):
        return fetch(src)
    return Path(src).read_bytes()


def dwd_reader(run, folder, download):
    """read(field, name): the raw cells of one DWD file, fetched into
    `folder` unless it is already there."""
    import eccodes                                             # only the real read needs it

    def read(field, name):
        path = Path(folder) / name
        if not path.exists():
            download(dwd.dir_url(run, field, name), path)
        h = eccodes.codes_new_from_message(bz2.decompress(path.read_bytes()))
        try:
            return eccodes.codes_get_values(h)
        finally:
            eccodes.codes_release(h)
    return read


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", help="a built heights file: a path, or the URL of a square's .ich")
    ap.add_argument("run", help="the ICON run, YYYYMMDDHH (UTC)")
    ap.add_argument("step", type=int, help="the forecast hour of that run")
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--cache", default=None, help="keep DWD's files here and reuse them (default: a temporary "
                                                  "folder, deleted afterwards)")
    a = ap.parse_args(argv)
    run = dt.datetime.strptime(a.run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    import run_build                                           # its whole-file download, with retries
    folder = Path(a.cache) if a.cache else Path(tempfile.mkdtemp(prefix="check-heights-"))
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for line in report(load_file(a.file), run, a.step, a.lat, a.lon, dwd_reader(run, folder, run_build._download)):
            print(line)
    except ValueError as e:
        print(f"check_heights: {e}", file=sys.stderr)
        return 2
    finally:
        if not a.cache:
            shutil.rmtree(folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
