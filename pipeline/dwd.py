# pipeline/dwd.py — where a DWD ICON global run lives and whether it is whole.
#
# DWD publishes each run under grib/<HH>/<field>/, one directory per run hour
# of the day; the run DATE is only in the file names. A run uploads over ~20
# minutes, so "newest" is not enough: a build must only take a run whose
# every file we need is listed, for all four fields.
import datetime as dt
import re

BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"
RUN_HOURS = ["00", "06", "12", "18"]
FIELDS = ["CLCL", "CLCM", "CLCH", "CLCT"]          # low, mid, high, total — the file's field order
# Hourly to 78 h, then every 3 h to 120 h: enough to reach the far end of the
# app's window (today+2, 23:59 local, any zone) from a run up to 9 h old.
STEPS = list(range(0, 79)) + list(range(81, 121, 3))

# The tail, for days 4-5 in the app (Premium's 5-day forecast): hours 123-144
# of the newest whole 00Z or 12Z run. Only those two runs go past 120 h
# (verified on DWD's listing 2026-09-30: 00/12 reach 180, 06/18 stop at 120).
TAIL_RUN_HOURS = ["00", "12"]
TAIL_STEPS = list(range(123, 145, 3))

# Cloud thickness: vertically integrated cloud ice and cloud water, kg/m²
# (DWD single-level TQI_DIA, TQC_DIA — measured 2026-10-20; the ice, water
# order is the thickness file's field order). They ride alongside the cover
# fields but never decide whether a run is complete: DWD may lag on them, and
# a missing thickness must not hold back the cover.
THICK_FIELDS = ["TQI_DIA", "TQC_DIA"]

# The field may hold an underscore (TQI_DIA), which a listing must still match.
_NAME = re.compile(r"icon_global_icosahedral_single-level_(\d{10})_(\d{3})_([A-Z_]+)\.grib2\.bz2")


def run_id(run):
    return run.strftime("%Y%m%d%H")


def run_iso(run):
    return run.strftime("%Y-%m-%dT%HZ")


def file_name(run, step, field):
    return f"icon_global_icosahedral_single-level_{run_id(run)}_{step:03d}_{field}.grib2.bz2"


def file_url(run, step, field):
    return f"{BASE}/{run:%H}/{field.lower()}/{file_name(run, step, field)}"



def listing_url(hh, field):
    return f"{BASE}/{hh}/{field.lower()}/"


def run_from_listing(html):
    """The run time the listed files belong to, or None for an empty listing."""
    m = _NAME.search(html or "")
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)


def missing_files(html, run, field, steps=None):
    steps = STEPS if steps is None else steps
    present = set(_NAME.findall(html or ""))
    return [file_name(run, s, field) for s in steps
            if (run_id(run), f"{s:03d}", field) not in present]


def newest_complete_run(listing, run_hours=None, steps=None):
    """`listing(hh, field) -> html`. The newest run (by time) whose every
    needed file is listed for every field, else None. Defaults: the near
    product's run hours and steps."""
    candidates = []
    for hh in (RUN_HOURS if run_hours is None else run_hours):
        html = listing(hh, FIELDS[0])
        run = run_from_listing(html)
        if run is None:
            continue
        if all(not missing_files(html if f == FIELDS[0] else listing(hh, f), run, f, steps) for f in FIELDS):
            candidates.append(run)
    return max(candidates) if candidates else None


def missing_thick_files(listing, run, steps):
    """`listing(hh, field) -> html`. The thickness file names of `run` not
    listed yet, across both fields."""
    hh = f"{run:%H}"
    return [n for f in THICK_FIELDS for n in missing_files(listing(hh, f), run, f, steps)]


# ---- The column feed (docs/column-feed-design.md) ---------------------------
# Pressure levels DWD publishes for ICON global, of the ones the app's fog
# profile reads (SkyCast INVERSION_LEVELS_HPA): 975 hPa is NOT published
# (verified on DWD's listing 2026-10-02: 1000, 950, 925, 900, 850, 800, 700,
# 600 ... 30 — no 975), so the column carries seven levels.
COLUMN_PLEVELS = [1000, 950, 925, 900, 850, 800, 700]
COLUMN_PFIELDS = ["T", "RELHUM", "FI"]              # per pressure level
COLUMN_SFIELDS = ["T_2M", "RELHUM_2M", "WW"]        # single level, per step
# WW is ICON's present weather, a WMO 4677 code (0-3 cloud states, 45/48 fog,
# 51-57 drizzle, 61-67 rain, 71-77 snow, 80-86 showers, 95/96 thunder); DWD
# publishes it under grib/<HH>/ww/ at the same steps as T_2M (listing checked
# 2026-11-07). It is categorical: remapped nearest-link, never blended.
# No 80 m field exists in ICON open data; T at 80 m above ground is
# interpolated in height between the two of these model levels (full levels;
# 120 is the lowest) that bracket 80 m at each node, with the time-invariant
# HHL (half-level heights). Measured on DWD's HHL (design doc): full level
# 118 sits 50-105 m above ground (median 96), 119 at 30-46 m, 117 at 74-185 m
# — so 118/119 bracket 80 m almost everywhere and 117/118 where 118 is lower.
COLUMN_MLEVELS = [117, 118, 119]
COLUMN_HHL_LEVELS = [117, 118, 119, 120]            # bounds of those full levels
# Cloud water per pressure level (the column's CLMR fields, kind 12). ICON
# publishes cloud water (QC, kg/kg) ONLY on model levels (grib/<HH>/qc/,
# levels 1-120, the same steps as T; listing checked 2026-11-19), so CLMR at a
# pressure level is QC interpolated in HEIGHT to that level's GH, between the
# two of these full levels that bracket it (run_build.clmr_at_heights).
#
# Levels 99-120: the fewest whose full levels reach, at every node, above the
# highest 700 hPa height (measured 2026-11-19/20 on DWD's HHL, all 2,949,120
# cells and the 0.25 deg column lattice; GH from the 2026-11-20 00Z run, all
# 93 steps). Full-level height, sea level min / median / max (m):
#   120:  -336 /   10 / 6232   (always 10-11 m above the model ground)
#   101:  3003 / 3038 / 7448
#   100:  3278 / 3310 / 7570
#    99:  3563 / 3592 / 7697
#    98:  3859 / 3885 / 7830
# GH 700 hPa peaked at 3286 m; at the worst node and step it sat 323 m under
# level 99, 41 m under level 100 (too thin a margin for a deeper low), and
# above level 101 at up to 397,281 nodes. So 99 is the top: 22 QC files a
# step, HHL 99-121.
QC_TOP = 99
COLUMN_QC_MLEVELS = list(range(QC_TOP, 121))
COLUMN_QC_HHL_LEVELS = list(range(QC_TOP, 122))     # bounds of those full levels (121 = the ground)


def pressure_file_name(run, step, level, field):
    return f"icon_global_icosahedral_pressure-level_{run_id(run)}_{step:03d}_{level}_{field}.grib2.bz2"


def model_file_name(run, step, level, field):
    return f"icon_global_icosahedral_model-level_{run_id(run)}_{step:03d}_{level}_{field}.grib2.bz2"


def invariant_file_name(run, field, level=None):
    mid = f"_{level}" if level is not None else ""
    return f"icon_global_icosahedral_time-invariant_{run_id(run)}{mid}_{field}.grib2.bz2"


def dir_url(run, field, name):
    """Every DWD file lives under grib/<run hour>/<field, lower case>/."""
    return f"{BASE}/{run:%H}/{field.lower()}/{name}"


def column_files(run, steps):
    """{directory field: [file names]} — every file one column build reads."""
    out = {f: [] for f in COLUMN_PFIELDS + COLUMN_SFIELDS + ["HHL", "HSURF"]}
    out["HSURF"].append(invariant_file_name(run, "HSURF"))
    out["HHL"] += [invariant_file_name(run, "HHL", lv) for lv in COLUMN_HHL_LEVELS]
    for s in steps:
        for f in COLUMN_PFIELDS:
            out[f] += [pressure_file_name(run, s, lv, f) for lv in COLUMN_PLEVELS]
        out["T"] += [model_file_name(run, s, lv, "T") for lv in COLUMN_MLEVELS]
        for f in COLUMN_SFIELDS:
            out[f].append(file_name(run, s, f))
    return out


def column_qc_files(run, steps):
    """{directory field: [file names]} — the cloud-water files: QC on each of
    COLUMN_QC_MLEVELS per step, and their HHL half levels once per run. Kept
    apart from column_files on purpose: they never decide whether the columns
    are built (a late or missing QC file leaves CLMR missing, nothing else)."""
    return {"QC": [model_file_name(run, s, lv, "QC") for s in steps for lv in COLUMN_QC_MLEVELS],
            "HHL": [invariant_file_name(run, "HHL", lv) for lv in COLUMN_QC_HHL_LEVELS]}


def listed_qc_steps(listing, run, steps):
    """`listing(hh, field) -> html`. The steps whose every QC file is listed."""
    html = listing(f"{run:%H}", "QC") or ""
    return [s for s in steps if all(model_file_name(run, s, lv, "QC") in html for lv in COLUMN_QC_MLEVELS)]


def missing_column_files(listing, run, steps):
    """`listing(hh, field) -> html`. The column files of `run` not listed yet."""
    hh = f"{run:%H}"
    missing = []
    for field, names in column_files(run, steps).items():
        html = listing(hh, field) or ""
        missing += [n for n in names if n not in html]
    return missing


# ---- The wind (ICW1, pipeline/wind.py) --------------------------------------
# u and v at 850, 500 and 250 hPa (the app's low, mid and high deck), from the
# pressure-level files under grib/<HH>/u/ and grib/<HH>/v/. Like the thickness
# they never decide whether a run is complete: a missing wind must not hold
# back the cover.
WIND_LEVELS = (850, 500, 250)


def wind_files(run, steps):
    """{"U": [names], "V": [names]} — every file one wind build reads, step by
    step, the three levels of each."""
    return {f: [pressure_file_name(run, s, lv, f) for s in steps for lv in WIND_LEVELS] for f in ("U", "V")}


def missing_wind_files(listing, run, steps):
    """`listing(hh, field) -> html`. The wind files of `run` not listed yet."""
    hh = f"{run:%H}"
    missing = []
    for field, names in wind_files(run, steps).items():
        html = listing(hh, field) or ""
        missing += [n for n in names if n not in html]
    return missing
