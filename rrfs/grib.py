# rrfs/grib.py — fetch NOAA RRFS's cloud-cover fields from NOMADS.
#
# File: the 13 km "2dfld" product on its North America grid (one rotated
# lat/lon grid, valid 3.8-90 N). The 3 km CONUS subset is HRRR's own grid and
# would not reach Seattle or Miami. Against the 3 km mean at the feed's nodes
# the 13 km file has correlation 0.998, mean difference 0.8 points (plan
# 2026-11-09). Cover costs 1.54 MB a forecast hour.
#
# Layer heights are NOAA UPP's, as HRRR's and GFS's: low = ground to 642 hPa,
# mid 642-350, high 350-150.
#
# Each field is one GRIB2 message; the .idx sidecar lists every message's byte
# offset, so we ask for just the four we want. The file also carries averaged
# rows ("5-6 hour ave fcst") for the same field names: only the instantaneous
# row for the hour is wanted.
import numpy as np
import eccodes

from globalfeed.grib import _groups, split_messages
from hrrr.grib import _range_of, _rows, rotate_to_earth
from pipeline import layerwind

# (short name, level text in the idx) in the feed's low / mid / high / total order.
FIELDS = (
    ("LCDC", "low cloud layer"),
    ("MCDC", "middle cloud layer"),
    ("HCDC", "high cloud layer"),
    ("TCDC", "entire atmosphere (considered as a single layer)"),
)
_BASE = ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.{d}/{h:02d}/"
         "rrfs.t{h:02d}z.2dfld.13km.f{f:03d}.na.grib2")


def file_url(run, fh):
    return _BASE.format(d=run.strftime("%Y%m%d"), h=run.hour, f=fh)


def field_ranges(idx_text, fh):
    """Byte range of each wanted field for forecast hour fh: {short name: (start, end)}.

    The row must say "<fh> hour fcst" ("anl" at f000). end is inclusive (the
    next message's start - 1), or None for the last message in the file."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [ln.split(":") for ln in idx_text.strip().split("\n") if ln.strip()]
    out = {}
    for name, level in FIELDS:
        k = next((i for i, r in enumerate(rows)
                  if len(r) > 5 and r[3] == name and r[4] == level and r[5] == want), None)
        if k is None:
            raise KeyError(f"{name} ({level}, {want}) not in idx")
        start = int(rows[k][1])
        end = int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None
        out[name] = (start, end)
    return out


def read_fields(run, fh, get):
    """Decode the four cloud fields for one forecast hour.

    get(url, rng=None) -> bytes is injected (rng = (start, end), end None for an
    open range). Returns {"lat", "lon", "LCDC", ...}: lon in -180..180, one float
    array per field, NaN where missing or negative."""
    url = file_url(run, fh)
    ranges = field_ranges(get(url + ".idx").decode(), fh)
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


# ---- Wind on RRFS's cloud layer (ICW1) -------------------------------------------
# Each deck's wind is the mean over RRFS's own cloud layer there (pipeline.layerwind,
# NCEP deck bounds: low p >= 642, mid 642-350, high 350-150 hPa), built from u / v /
# cloud water (CLMR) / cloud ice (ICMR) at every 50 mb from 1000 to 150 mb in the 13 km
# "prslev" file, and the surface pressure (PRES:surface) from the "2dfld" file the cover
# reads: both on the same rotated lat/lon NA grid. A level where cloud water + cloud ice
# >= CONDENSATE_CLOUDY is cloudy. The prslev idx lists 25 mb levels and 14 fields a
# level (checked 2026-11-18, f06); a row missing at any named level fails the step, so
# no level quietly drops out and shifts its deck's weights.
#
# The components follow the grid (uvRelativeToGrid = 1), and RRFS's 13 km NA grid is a
# ROTATED lat/lon grid (gridType rotated_ll), not Lambert: each level is turned to
# east/north with the rotated pole (not hrrr.grib.rotate_to_earth's convergence) BEFORE
# the layer mean.
PRS_BASE = ("https://nomads.ncep.noaa.gov/pub/data/nccf/com/rrfs/prod/rrfs.{d}/{h:02d}/"
             "rrfs.t{h:02d}z.prslev.13km.f{f:03d}.na.grib2")


def prs_url(run, fh):
    return PRS_BASE.format(d=run.strftime("%Y%m%d"), h=run.hour, f=fh)


def rotate_rotated_pole(u, v, lat, lon, south_lat_deg, south_lon_deg):
    """Grid-relative (u, v) on a rotated lat/lon grid -> east/north (ue, ve).

    The grid's north is the direction toward its rotated north pole, at
    (-south_lat, south_lon + 180). Its bearing a from geographic north
    (clockwise) at (lat, lon) is
        a = atan2(sin(dl) cos(pl), cos(lat) sin(pl) - sin(lat) cos(pl) cos(dl)),
    pl, dl = the pole's latitude and (pole lon - lon). The grid's north is then
    (sin a, cos a) in (east, north) and its east is (cos a, -sin a), so
        ue = cos a u + sin a v,   ve = -sin a u + cos a v.
    Degrees in; a = 0 on the meridian through the south pole's longitude (the
    grid's central meridian), whatever the latitude."""
    pl = np.radians(-float(south_lat_deg))
    lp = np.radians(float(south_lon_deg) + 180.0)
    la = np.radians(np.asarray(lat, dtype=np.float64))
    dl = lp - np.radians(np.asarray(lon, dtype=np.float64))
    a = np.arctan2(np.sin(dl) * np.cos(pl), np.cos(la) * np.sin(pl) - np.sin(la) * np.cos(pl) * np.cos(dl))
    c, s = np.cos(a), np.sin(a)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return c * u + s * v, -s * u + c * v


def _wind_geometry(g):
    """(lat, lon, turn) of one message: turn(u, v) -> (east, north) on the
    message's own points (None when the components already point east/north).
    A grid type other than the two this feed's files use is an error."""
    kind = str(eccodes.codes_get(g, "gridType"))
    if kind not in ("rotated_ll", "lambert"):
        raise ValueError(f"wind: unsupported grid {kind}")
    lat = eccodes.codes_get_array(g, "latitudes")
    lon = eccodes.codes_get_array(g, "longitudes")
    lon = np.where(lon > 180, lon - 360, lon)
    if not int(eccodes.codes_get(g, "uvRelativeToGrid")):
        return lat, lon, None
    if kind == "lambert":
        lov = float(eccodes.codes_get(g, "LoVInDegrees"))
        lat1 = float(eccodes.codes_get(g, "Latin1InDegrees"))
        return lat, lon, lambda u, v: rotate_to_earth(u, v, lon, lov, lat1)
    if float(eccodes.codes_get(g, "angleOfRotationInDegrees")) != 0.0:
        raise ValueError("wind: rotated grid with a non-zero angle of rotation is not handled")
    s_lat = float(eccodes.codes_get(g, "latitudeOfSouthernPoleInDegrees"))
    s_lon = float(eccodes.codes_get(g, "longitudeOfSouthernPoleInDegrees"))
    return lat, lon, lambda u, v: rotate_rotated_pole(u, v, lat, lon, s_lat, s_lon)


RRFS_WIND_LEVELS = tuple(range(1000, 149, -50))
LAYER_CHUNK = 1 << 18          # cells per deck_winds call
# (shortName in the idx, GRIB2 discipline / category / number), read off real f06 messages
# 2026-11-18: ICMR is 0/1/23 (shortName "icmr"), CLMR 0/1/22 ("clwmr"), surface PRES 0/3/0 ("sp").
_PARAMS = {"u": ("UGRD", (0, 2, 2)), "v": ("VGRD", (0, 2, 3)),
           "clmr": ("CLMR", (0, 1, 22)), "icmr": ("ICMR", (0, 1, 23))}


def _locate(rows, name, level, want):
    k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
              and r[4] == level and r[5] == want), None)
    if k is None:
        raise KeyError(f"{name} ({level}, {want}) not in idx")
    return k


def layer_wind_ranges(prs_idx, sfc_idx, fh):
    """{(level hPa, 'u'|'v'|'clmr'|'icmr'): (start, end), ('sfc', 'pres'): (start, end)}
    for forecast hour fh ("anl" at f00). The level keys index the prslev file
    (prs_idx), ('sfc', 'pres') the 2dfld file (sfc_idx). Rows are picked by exact
    name, level text and forecast text; end is inclusive or None for the file's
    last message. KeyError if any row is missing."""
    return layer_wind_plan(prs_idx, sfc_idx, fh)[0]


def layer_wind_plan(prs_idx, sfc_idx, fh):
    """(ranges, requests): ranges as layer_wind_ranges; requests = [(which, (start, end),
    [key or None, ...])], which = 'prs' | 'sfc', one entry per ranged request, its keys in
    file order. A level whose rows run UGRD, VGRD, ABSV, CLMR, ICMR back to back in the idx
    (NOAA's order, checked 2026-11-18) is ONE request carrying five messages: the ABSV one is
    downloaded and thrown away (key None). Any other layout at a level falls back to the
    rows that touch merged, never a wider span."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    prs, sfc = _rows(prs_idx), _rows(sfc_idx)
    ranges, requests = {}, []
    for lev in RRFS_WIND_LEVELS:
        at = {comp: _locate(prs, name, f"{lev} mb", want) for comp, (name, _) in _PARAMS.items()}
        for comp, k in at.items():
            ranges[(lev, comp)] = _range_of(prs, k)
        k0 = at["u"]
        last = _range_of(prs, at["icmr"])
        if (at["v"] == k0 + 1 and at["clmr"] == k0 + 3 and at["icmr"] == k0 + 4
                and prs[k0 + 2][3] == "ABSV" and prs[k0 + 2][4] == f"{lev} mb" and prs[k0 + 2][5] == want
                and last[1] is not None):
            requests.append(("prs", (ranges[(lev, "u")][0], last[1]),
                             [(lev, "u"), (lev, "v"), None, (lev, "clmr"), (lev, "icmr")]))
        else:
            ks = [(lev, c) for c in _PARAMS]
            for rng, positions in _groups([(k, ranges[k]) for k in ks]):
                requests.append(("prs", rng, [ks[i] for i in positions]))
    ranges[("sfc", "pres")] = _range_of(sfc, _locate(sfc, "PRES", "surface", want))
    requests.append(("sfc", ranges[("sfc", "pres")], [("sfc", "pres")]))
    return ranges, requests


def _check_absv(g):
    triple = tuple(int(eccodes.codes_get(g, k)) for k in ("discipline", "parameterCategory", "parameterNumber"))
    if triple != (0, 2, 10):
        raise ValueError(f"wind: expected the absolute vorticity message between v and clmr, message is {triple}")


def _check_layer_message(g, lev, comp):
    """The idx said which message this is; the bytes must agree, by GRIB2 parameter
    and level (not shortName alone)."""
    short = str(eccodes.codes_get(g, "shortName"))
    level = int(round(float(eccodes.codes_get(g, "level"))))
    triple = tuple(int(eccodes.codes_get(g, k)) for k in ("discipline", "parameterCategory", "parameterNumber"))
    kind = str(eccodes.codes_get(g, "typeOfLevel"))
    if lev == "sfc":
        # PRES also exists at cloud base / top, the tropopause and others
        if triple != (0, 3, 0) or kind != "surface":
            raise ValueError(f"wind: expected surface pressure, message is {short} {kind} {level} {triple}")
    elif triple != _PARAMS[comp][1] or kind != "isobaricInhPa" or level != lev:
        raise ValueError(f"wind: expected {comp} {lev} hPa, message is {short} {kind} {level} {triple}")


def read_layer_fields(run, fh, get, keep_condensate=False):
    """The decoded inputs of one forecast hour's layer wind: {"lat", "lon"
    (-180..180), "levels": RRFS_WIND_LEVELS, "ps": surface pressure in hPa, "u",
    "v": per-level float32 lists turned to east/north, "cloudy": per-level bool
    lists (cloud water + ice >= CONDENSATE_CLOUDY; with keep_condensate the
    float32 sums themselves, "condensate", are kept too, for the threshold
    control)}. get(url, rng=None) -> bytes is injected. Rows that touch are one
    ranged request. Every message is checked against its idx row by parameter and
    level; a missing row raises KeyError."""
    prs, sfc = prs_url(run, fh), file_url(run, fh)
    _, requests = layer_wind_plan(get(prs + ".idx").decode(), get(sfc + ".idx").decode(), fh)
    out, turn = {}, None
    uv, cond, ps = {}, {}, None
    for which, rng, ks in requests:
        msgs = split_messages(get(prs if which == "prs" else sfc, rng))
        if len(msgs) != len(ks):
            raise ValueError(f"wind: asked for {len(ks)} messages, got {len(msgs)}")
        for key, raw in zip(ks, msgs):
            g = eccodes.codes_new_from_message(raw)
            try:
                if key is None:                  # ABSV: identified by parameter, never decoded
                    _check_absv(g)
                    continue
                lev, comp = key
                _check_layer_message(g, lev, comp)
                if "lat" not in out:
                    out["lat"], out["lon"], turn = _wind_geometry(g)
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
    levels = list(RRFS_WIND_LEVELS)      # descending: the order deck_winds is happiest with
    u, v = [], []
    for lev in levels:
        pair = uv.pop(lev)
        ul, vl = pair["u"], pair["v"]
        if turn is not None:
            ul, vl = turn(ul, vl)
            ul, vl = ul.astype(np.float32), vl.astype(np.float32)
        u.append(ul); v.append(vl)
    out.update(levels=levels, ps=ps, u=u, v=v, cloudy=[cond[lev][1] for lev in levels])
    if keep_condensate:
        out["condensate"], out["cloudy"] = out["cloudy"], [a >= layerwind.CONDENSATE_CLOUDY for a in out["cloudy"]]
    return out


def read_layer_wind(run, fh, get):
    """The wind of one forecast hour on RRFS's cloud layer: {"lat", "lon"
    (-180..180), (850|500|250, 'u'|'v'): float32 array} (low / mid / high deck),
    flat over RRFS's cells, east/north m/s, NaN where a deck has no level with a
    wind. Inputs as read_layer_fields; each level is turned to east/north before
    the layer mean."""
    f = read_layer_fields(run, fh, get)
    out = {"lat": f["lat"], "lon": f["lon"]}
    ps = f["ps"]
    # deck_winds works on every point at once with many float64 temporaries: run it in slices
    # (each point is independent) to keep a worker's peak memory down
    parts = []
    for a in range(0, ps.size, LAYER_CHUNK):
        sl = slice(a, a + LAYER_CHUNK)
        parts.append(layerwind.deck_winds(f["levels"], [x[sl] for x in f["u"]], [x[sl] for x in f["v"]],
                                          [x[sl] for x in f["cloudy"]], ps[sl],
                                          layerwind.deck_bounds("ncep", ps[sl])))
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    return out
