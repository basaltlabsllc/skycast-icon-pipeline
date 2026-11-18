# hrrr/grib.py — fetch HRRR's cloud-cover fields from NOAA's open S3 bucket.
#
# Layer heights, which differ from ICON's and must be remembered wherever the
# two feeds are compared: HRRR low = ground to 642 hPa, mid 642-350, high
# 350-150 (NOAA UPP INITPOST.F); ICON low = ground to 800 hPa, mid 800-400,
# high above 400.
#
# Each field is one GRIB2 message; the .idx sidecar lists every message's byte
# offset, so we ask S3 for just the four we want instead of the whole file.
import numpy as np
import eccodes

from globalfeed.grib import _groups, split_messages
from pipeline import layerwind

# (short name, level text in the idx) in the feed's low / mid / high / total order.
FIELDS = (
    ("LCDC", "low cloud layer"),
    ("MCDC", "middle cloud layer"),
    ("HCDC", "high cloud layer"),
    ("TCDC", "entire atmosphere"),
)
_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{d}/conus/hrrr.t{h:02d}z.wrfsfcf{f:02d}.grib2"


def file_url(run, fh):
    return _BASE.format(d=run.strftime("%Y%m%d"), h=run.hour, f=fh)


def field_ranges(idx_text):
    """Byte range of each wanted field: {short name: (start, end)}.

    end is inclusive (the next message's start - 1), or None for the last
    message in the file, which means an open range."""
    rows = [ln.split(":") for ln in idx_text.strip().split("\n") if ln.strip()]
    out = {}
    for name, level in FIELDS:
        k = next((i for i, r in enumerate(rows) if len(r) > 4 and r[3] == name and r[4] == level), None)
        if k is None:
            raise KeyError(f"{name} ({level}) not in idx")
        start = int(rows[k][1])
        end = int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None
        out[name] = (start, end)
    return out


# ---- Thickness (ICT1) ---------------------------------------------------------
# Ice is TCOLI plus the snow column: HRRR's microphysics keeps most of its ice
# mass in the snow category, and TCOLI alone reads about a third of ICON's on
# the same decks (plan 2026-11-09, measured against GOES CODC). Snow is SNMR
# (kg/kg) from the pressure-level file at every other level from 100 mb,
# integrated with dp from numpy.gradient of those levels; every other level
# matches the full integral for RRFS's 25 mb spacing (ice 1.000, water 1.005).
G = 9.80665
SNOW_LEVELS = tuple(range(100, 1001, 50))
_PRS_BASE = _BASE.replace("wrfsfc", "wrfprs")
_SFC_THICK = ("TCOLI", "TCOLW")


def prs_url(run, fh):
    return _PRS_BASE.format(d=run.strftime("%Y%m%d"), h=run.hour, f=fh)


def layer_weights(levels_hpa):
    """kg/m2 of air per unit mixing ratio, one per level: dp / g with dp from
    numpy.gradient of the levels in Pa (centred inside, one-sided at the ends)."""
    return np.gradient(np.asarray(levels_hpa, dtype=float) * 100.0) / G


def _rows(idx_text):
    return [ln.split(":") for ln in idx_text.strip().split("\n") if ln.strip()]


def _range_of(rows, k):
    start = int(rows[k][1])
    return start, (int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None)


def thick_ranges(sfc_idx, prs_idx, fh):
    """{"TCOLI": (s, e), "TCOLW": (s, e), "SNMR": [(level hPa, (s, e)), ...]}
    for forecast hour fh (f00 reads "anl"). KeyError if anything is missing."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    out = {}
    rows = _rows(sfc_idx)
    for name in _SFC_THICK:
        k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
                  and r[4].startswith("entire atmosphere") and r[5] == want), None)
        if k is None:
            raise KeyError(f"{name} (entire atmosphere, {want}) not in idx")
        out[name] = _range_of(rows, k)
    rows = _rows(prs_idx)
    snow = []
    for lev in SNOW_LEVELS:
        k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == "SNMR"
                  and r[4] == f"{lev} mb" and r[5] == want), None)
        if k is None:
            raise KeyError(f"SNMR ({lev} mb, {want}) not in idx")
        snow.append((lev, _range_of(rows, k)))
    out["SNMR"] = snow
    return out


def _decode(raw):
    g = eccodes.codes_new_from_message(raw)
    try:
        vals = eccodes.codes_get_values(g)
        miss = eccodes.codes_get(g, "missingValue")
    finally:
        eccodes.codes_release(g)
    return np.where(np.isclose(vals, miss) | (vals < 0), np.nan, vals)


def read_thick(run, fh, get):
    """Column cloud ice and water for one forecast hour, in kg/m2, on the same
    grid and cell order as read_fields: {"ice": TCOLI + snow, "water": TCOLW}.

    get(url, rng=None) -> bytes is injected. NaN where TCOLI or TCOLW is missing;
    a snow level that is missing at a cell adds nothing there. The snow column
    is accumulated one level at a time, so a worker holds a few arrays, not 19."""
    sfc, prs = file_url(run, fh), prs_url(run, fh)
    r = thick_ranges(get(sfc + ".idx").decode(), get(prs + ".idx").decode(), fh)
    ice = _decode(get(sfc, r["TCOLI"]))
    water = _decode(get(sfc, r["TCOLW"]))
    levels = [lev for lev, _ in r["SNMR"]]
    w = layer_weights(levels)
    snow = np.zeros_like(ice)
    for wt, (_, rng) in zip(w, r["SNMR"]):
        v = _decode(get(prs, rng))
        snow += np.where(np.isnan(v), 0.0, v) * wt
    return {"ice": ice + snow, "water": water}


def read_fields(run, fh, get):
    """Decode the four cloud fields for one forecast hour.

    get(url, rng=None) -> bytes is injected (rng = (start, end), end None for
    an open range). Returns {"lat", "lon", "LCDC", ...}: lon in -180..180, one
    float array per field, NaN where missing or negative."""
    url = file_url(run, fh)
    ranges = field_ranges(get(url + ".idx").decode())
    out = {}
    for name, _ in FIELDS:
        g = eccodes.codes_new_from_message(get(url, ranges[name]))
        try:
            vals = eccodes.codes_get_values(g)
            if "lat" not in out:
                lat = eccodes.codes_get_array(g, "latitudes")
                lon = eccodes.codes_get_array(g, "longitudes")
                out["lat"] = lat
                out["lon"] = np.where(lon > 180, lon - 360, lon)
            miss = eccodes.codes_get(g, "missingValue")
        finally:
            eccodes.codes_release(g)
        out[name] = np.where(np.isclose(vals, miss) | (vals < 0), np.nan, vals)
    return out


# ---- Wind (ICW1) --------------------------------------------------------------
# u/v at 850 / 500 / 250 mb (the low, mid and high deck's steering levels) from
# the surface file. HRRR's components follow its Lambert grid (uvRelativeToGrid
# = 1), not east and north, so each point is turned by the grid's convergence
# angle before it goes on the lattice: sin(Latin1) x (lon - LoV).
WIND_LEVELS = (850, 500, 250)


def wind_ranges(sfc_idx, fh):
    """{(level hPa, 'u'|'v'): (start, end)} for forecast hour fh (f00 reads
    "anl") from the surface file's idx. KeyError if any of the six is missing."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = _rows(sfc_idx)
    out = {}
    for lev in WIND_LEVELS:
        for comp, name in (("u", "UGRD"), ("v", "VGRD")):
            k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
                      and r[4] == f"{lev} mb" and r[5] == want), None)
            if k is None:
                raise KeyError(f"{name} ({lev} mb, {want}) not in idx")
            out[(lev, comp)] = _range_of(rows, k)
    return out


def _turn(lon, lov_deg, latin1_deg):
    """(cos a, sin a) of the grid's turn at each lon: a = sin(Latin1) x (lon - LoV)."""
    dlon = (np.asarray(lon, dtype=np.float64) - lov_deg + 180.0) % 360.0 - 180.0
    a = np.sin(np.radians(latin1_deg)) * np.radians(dlon)
    return np.cos(a), np.sin(a)


def rotate_to_earth(u, v, lon, lov_deg, latin1_deg):
    """Grid-relative (u, v) on a Lambert conformal grid -> east/north (ue, ve).
    The grid's x axis lies a = sin(Latin1) x (lon - LoV) counter-clockwise of
    east, so ue = cos a u + sin a v and ve = -sin a u + cos a v. lon and LoV in
    degrees east, either -180..180 or 0..360."""
    c, s = _turn(lon, lov_deg, latin1_deg)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return c * u + s * v, -s * u + c * v


def _wind_geometry(g):
    """(lat, lon, rotation) of one message: rotation is None when its components
    already point east/north, else (LoV, Latin1) in degrees. A grid type other
    than the two HRRR files and the tests use is an error, not a guess."""
    kind = str(eccodes.codes_get(g, "gridType"))
    lat = eccodes.codes_get_array(g, "latitudes")
    lon = eccodes.codes_get_array(g, "longitudes")
    lon = np.where(lon > 180, lon - 360, lon)
    if kind == "regular_ll":
        return lat, lon, None
    if kind != "lambert":
        raise ValueError(f"wind: unsupported grid {kind}")
    if not int(eccodes.codes_get(g, "uvRelativeToGrid")):
        return lat, lon, None
    return lat, lon, (float(eccodes.codes_get(g, "LoVInDegrees")), float(eccodes.codes_get(g, "Latin1InDegrees")))


def read_wind(run, fh, get):
    """The six wind fields of one forecast hour, turned to east/north, in m/s:
    {"lat", "lon" (-180..180), (level, 'u'|'v'): array}, flat over HRRR's cells,
    NaN where missing, from the surface file at 850 / 500 / 250 mb. RRFS still
    reads its wind this way; HRRR's own feed uses read_layer_wind. get(url,
    rng=None) -> bytes is injected. Each message's own shortName and level must
    match the idx row that asked for it."""
    url = file_url(run, fh)
    ranges = wind_ranges(get(url + ".idx").decode(), fh)
    out, raw, rot = {}, {}, None
    for key, rng in ranges.items():
        g = eccodes.codes_new_from_message(get(url, rng))
        try:
            got = (str(eccodes.codes_get(g, "shortName")), int(round(float(eccodes.codes_get(g, "level")))))
            if got != (key[1], key[0]):
                raise ValueError(f"wind: expected {key[1]} {key[0]} hPa, message is {got[0]} {got[1]}")
            if "lat" not in out:
                out["lat"], out["lon"], rot = _wind_geometry(g)
            vals = eccodes.codes_get_values(g)
            miss = eccodes.codes_get(g, "missingValue")
        finally:
            eccodes.codes_release(g)
        raw[key] = np.where(np.isclose(vals, miss), np.nan, vals)
    c, s = _turn(out["lon"], rot[0], rot[1]) if rot is not None else (None, None)
    for lev in WIND_LEVELS:
        u, v = raw.pop((lev, "u")), raw.pop((lev, "v"))
        if rot is not None:
            u, v = c * u + s * v, -s * u + c * v
        out[(lev, "u")], out[(lev, "v")] = u, v
    return out


# ---- Wind on HRRR's cloud layer (ICW1) ------------------------------------------
# Each deck's wind is the mean over HRRR's own cloud layer there (pipeline.layerwind,
# HRRR's deck bounds), built from u / v / cloud water / cloud ice at every 50 mb
# from 1000 to 150 mb in the pressure file, and the surface pressure from the
# surface file. A level where cloud water + cloud ice >= CONDENSATE_CLOUDY is cloudy.
# The wrfprs idx holds these rows at every 25 mb, 1000 to 50 mb (checked 2026-11-18,
# f06); a missing row at any named level fails the step, so no level quietly drops
# out and shifts its deck's weights.
HRRR_WIND_LEVELS = tuple(range(1000, 149, -50))
LAYER_CHUNK = 1 << 18          # cells per deck_winds call
_PARAMS = {"u": ("UGRD", (0, 2, 2)), "v": ("VGRD", (0, 2, 3)),
           "clmr": ("CLMR", (0, 1, 22)), "cimixr": ("CIMIXR", (0, 1, 82))}


def layer_wind_ranges(prs_idx, sfc_idx, fh):
    """{(level hPa, 'u'|'v'|'clmr'|'cimixr'): (start, end), ('sfc', 'pres'): (start, end)}
    for forecast hour fh ("anl" at f00). The level keys index the pressure file
    (prs_idx), ('sfc', 'pres') the surface file (sfc_idx). Rows are picked by exact
    name, level text and forecast text; end is inclusive or None for the file's
    last message. KeyError if any row is missing."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"

    def find(rows, name, level):
        k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
                  and r[4] == level and r[5] == want), None)
        if k is None:
            raise KeyError(f"{name} ({level}, {want}) not in idx")
        return _range_of(rows, k)

    prs, sfc = _rows(prs_idx), _rows(sfc_idx)
    out = {}
    for lev in HRRR_WIND_LEVELS:
        for comp, (name, _) in _PARAMS.items():
            out[(lev, comp)] = find(prs, name, f"{lev} mb")
    out[("sfc", "pres")] = find(sfc, "PRES", "surface")
    return out


def _check_layer_message(g, lev, comp):
    """The idx said which message this is; the bytes must agree, by GRIB2 parameter
    (not shortName: CIMIXR has none in ecCodes' tables) and by level."""
    short = str(eccodes.codes_get(g, "shortName"))
    level = int(round(float(eccodes.codes_get(g, "level"))))
    triple = tuple(int(eccodes.codes_get(g, k)) for k in ("discipline", "parameterCategory", "parameterNumber"))
    kind = str(eccodes.codes_get(g, "typeOfLevel"))
    if lev == "sfc":
        # PRES also exists at cloud base / top, the tropopause and others, all level 0
        if triple != (0, 3, 0) or kind != "surface":
            raise ValueError(f"wind: expected surface pressure, message is {short} {kind} {level} {triple}")
    elif triple != _PARAMS[comp][1] or kind != "isobaricInhPa" or level != lev:
        raise ValueError(f"wind: expected {comp} {lev} hPa, message is {short} {kind} {level} {triple}")


def read_layer_fields(run, fh, get, keep_condensate=False):
    """The decoded inputs of one forecast hour's layer wind: {"lat", "lon"
    (-180..180), "levels": HRRR_WIND_LEVELS, "ps": surface pressure in hPa, "u",
    "v": per-level float32 lists turned to east/north, "cloudy": per-level bool
    lists (cloud water + ice >= CONDENSATE_CLOUDY; with keep_condensate the
    float32 sums themselves, "condensate", are kept too, for the threshold
    control)}. get(url, rng=None) -> bytes is injected. Every message is checked
    against its idx row by parameter and level; a missing row raises KeyError."""
    prs, sfc = prs_url(run, fh), file_url(run, fh)
    ranges = layer_wind_ranges(get(prs + ".idx").decode(), get(sfc + ".idx").decode(), fh)
    keys = list(ranges)
    out, rot = {}, None
    uv, cond, ps = {}, {}, None
    for url, ks in ((prs, [k for k in keys if k[0] != "sfc"]), (sfc, [("sfc", "pres")])):
        for rng, positions in _groups([(k, ranges[k]) for k in ks]):
            msgs = split_messages(get(url, rng))
            if len(msgs) != len(positions):
                raise ValueError(f"wind: asked for {len(positions)} messages, got {len(msgs)}")
            for pos, raw in zip(positions, msgs):
                lev, comp = ks[pos]
                g = eccodes.codes_new_from_message(raw)
                try:
                    _check_layer_message(g, lev, comp)
                    if "lat" not in out:
                        out["lat"], out["lon"], rot = _wind_geometry(g)
                    vals = eccodes.codes_get_values(g)
                    miss = eccodes.codes_get(g, "missingValue")
                finally:
                    eccodes.codes_release(g)
                if vals.shape != out["lat"].shape:
                    raise ValueError(f"wind: {comp} {lev} has {vals.size} cells, the grid has {out['lat'].size}")
                vals = np.where(np.isclose(vals, miss), np.nan, vals)
                if lev == "sfc":
                    ps = vals / 100.0
                elif comp in ("u", "v"):
                    uv.setdefault(lev, {})[comp] = vals.astype(np.float32)
                else:
                    # cloud water + ice; a missing part counts as none, both missing is clear
                    part = np.where(np.isnan(vals), 0.0, vals).astype(np.float32)
                    have, acc = cond.get(lev, (0, None))
                    acc = part if acc is None else acc + part
                    cond[lev] = (have + 1, acc)
                    if have + 1 == 2 and not keep_condensate:
                        cond[lev] = (2, acc >= layerwind.CONDENSATE_CLOUDY)
    levels = list(HRRR_WIND_LEVELS)      # descending: the order deck_winds is happiest with
    if rot is not None:
        c, s = _turn(out["lon"], rot[0], rot[1])
    u, v = [], []
    for lev in levels:
        pair = uv.pop(lev)
        ul, vl = pair["u"], pair["v"]
        if rot is not None:
            ul, vl = (c * ul + s * vl).astype(np.float32), (-s * ul + c * vl).astype(np.float32)
        u.append(ul); v.append(vl)
    out.update(levels=levels, ps=ps, u=u, v=v, cloudy=[cond[lev][1] for lev in levels])
    if keep_condensate:
        out["condensate"], out["cloudy"] = out["cloudy"], [a >= layerwind.CONDENSATE_CLOUDY for a in out["cloudy"]]
    return out


def read_layer_wind(run, fh, get):
    """The wind of one forecast hour on HRRR's cloud layer: {"lat", "lon"
    (-180..180), (850|500|250, 'u'|'v'): float32 array} (low / mid / high deck),
    flat over HRRR's cells, east/north m/s, NaN where a deck has no level with a
    wind. Inputs as read_layer_fields; each level is turned to east/north before
    the layer mean."""
    f = read_layer_fields(run, fh, get)
    out = {"lat": f["lat"], "lon": f["lon"]}
    ps = f["ps"]
    # deck_winds works on every point at once with ~30 float64 temporaries; HRRR has 1.9 M points,
    # so run it in slices (each point is independent) to keep a worker's peak memory down
    parts = []
    for a in range(0, ps.size, LAYER_CHUNK):
        sl = slice(a, a + LAYER_CHUNK)
        parts.append(layerwind.deck_winds(f["levels"], [x[sl] for x in f["u"]], [x[sl] for x in f["v"]],
                                          [x[sl] for x in f["cloudy"]], ps[sl],
                                          layerwind.deck_bounds("ncep", ps[sl])))
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    return out
