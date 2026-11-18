# globalfeed/grib.py — byte-range readers for GFS (NCEP .idx) and AIFS / IFS
# (ECMWF .index JSON) cloud-cover fields. Same approach as hrrr/grib.py: read the
# sidecar index, then ask the server for just the messages we want.
import json

import numpy as np
import eccodes

from pipeline import layerwind

G = 9.80665
_MASS = (("ICMR", "ice"), ("CLMR", "water"))


def _key(f):
    return f[0] if isinstance(f, tuple) else f


def _ncep_ranges(model, index_text, fh):
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [ln.split(":") for ln in index_text.strip().split("\n") if ln.strip()]
    out = {}
    for name, level in model.fields:
        k = next((i for i, r in enumerate(rows)
                  if len(r) > 5 and r[3] == name and r[4] == level and r[5] == want), None)
        if k is None:
            raise KeyError(f"{name} ({level}, {want}) not in idx")
        start = int(rows[k][1])
        end = int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None
        out[name] = (start, end)
    return out


def _ecmwf_ranges(model, index_text, keys=None):
    rows = [json.loads(ln) for ln in index_text.strip().split("\n") if ln.strip()]
    out = {}
    for key in (model.fields if keys is None else keys):
        r = next((r for r in rows if r.get("param") == key and r.get("levtype") == "sfc"), None)
        if r is None:
            raise KeyError(f"{key} (sfc) not in index")
        out[key] = (r["_offset"], r["_offset"] + r["_length"] - 1)
    return out


def field_ranges(model, index_text, fh):
    """Byte range of each wanted field: {key: (start, end)}; end is inclusive,
    or None for an open range (last message of an NCEP file)."""
    if model.index_kind == "ncep":
        return _ncep_ranges(model, index_text, fh)
    return _ecmwf_ranges(model, index_text)


def read_step(model, run, fh, get):
    """Decode the four cloud fields for one forecast hour.

    get(url, rng=None) -> bytes is injected. Returns {"lat", "lon", <key>...}:
    lon in -180..180, float arrays, NaN where missing or negative."""
    ranges = field_ranges(model, get(model.index_url(run, fh)).decode(), fh)
    url = model.url(run, fh)
    out = {}
    for f in model.fields:
        key = _key(f)
        g = eccodes.codes_new_from_message(get(url, ranges[key]))
        try:
            vals = eccodes.codes_get_values(g)
            if "lat" not in out:
                lat = eccodes.codes_get_array(g, "latitudes")
                lon = eccodes.codes_get_array(g, "longitudes")
                out["lat"] = lat
                out["lon"] = np.where(lon > 180, lon - 360, lon)
            miss = eccodes.codes_get(g, "missingValue")
            scale = _cover_scale(g, key) if model.check_cover_units else 1.0
        finally:
            eccodes.codes_release(g)
        out[key] = np.where(np.isclose(vals, miss) | (vals < 0), np.nan, vals * scale)
    return out


# GRIB units of a cover field -> factor to percent.
_PERCENT = {"%": 1.0, "(0 - 1)": 100.0}


def _cover_scale(g, key):
    units = eccodes.codes_get(g, "units")
    if units not in _PERCENT:
        raise ValueError(f"{key}: cover in unexpected units {units!r}")
    return _PERCENT[units]


# ---- Thickness (ICT1) ---------------------------------------------------------
# GFS's ice (ICMR) and cloud water (CLMR) mixing ratios, kg/kg, on the pressure
# levels in model.thick_levels, integrated to kg/m2 with dp / g, dp from
# numpy.gradient of the levels in Pa (centred inside, one-sided at the ends).

def thick_ranges(model, index_text, fh):
    """{"ICMR": [(level hPa, (start, end)), ...], "CLMR": [...]}, levels
    ascending, rows picked by exact name, level and forecast text ("anl" at
    f000). end is inclusive or None for the file's last message. KeyError if a
    row is missing."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [ln.split(":") for ln in index_text.strip().split("\n") if ln.strip()]
    out = {}
    for name, _ in _MASS:
        picked = []
        for lev in sorted(model.thick_levels):
            k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
                      and r[4] == f"{lev} mb" and r[5] == want), None)
            if k is None:
                raise KeyError(f"{name} ({lev} mb, {want}) not in idx")
            start = int(rows[k][1])
            end = int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None
            picked.append((lev, (start, end)))
        out[name] = picked
    return out


def split_messages(buf):
    """Whole GRIB2 messages laid end to end -> [message bytes]; each message
    states its own length in bytes 8..15."""
    out, pos = [], 0
    while pos < len(buf):
        if buf[pos:pos + 4] != b"GRIB":
            raise ValueError(f"not a GRIB message at byte {pos}")
        n = int.from_bytes(buf[pos + 8:pos + 16], "big")
        out.append(buf[pos:pos + n])
        pos += n
    return out


def _groups(ranges):
    """Merge ranges that touch into one request: [((start, end), [positions])],
    positions in file order (NCEP lists 1000 mb first, the ranges here ascend by level)."""
    groups = []
    for i, (_, (s, e)) in sorted(enumerate(ranges), key=lambda t: t[1][1][0]):
        if groups and groups[-1][0][1] is not None and groups[-1][0][1] + 1 == s:
            groups[-1] = ((groups[-1][0][0], e), groups[-1][1] + [i])
        else:
            groups.append(((s, e), [i]))
    return groups


def _decode(g):
    vals = eccodes.codes_get_values(g)
    miss = eccodes.codes_get(g, "missingValue")
    return np.where(np.isclose(vals, miss) | (vals < 0), np.nan, vals)


def read_thick_step(model, run, fh, get):
    """Column cloud ice and water for one forecast hour, kg/m2, on the source
    grid: {"lat", "lon", "ice", "water"} (lon in -180..180). get(url, rng=None)
    -> bytes is injected. A model with thick_fields reads its column totals
    (read_column_thick); one with thick_levels integrates them here: messages
    whose idx rows touch are fetched as one range, a level missing at a cell
    adds nothing there, and a cell missing at every level is NaN."""
    if model.thick_fields:
        return read_column_thick(model, run, fh, get)
    if not model.thick_levels:
        raise ValueError(f"{model.name} has no thickness levels")
    ranges = thick_ranges(model, get(model.index_url(run, fh)).decode(), fh)
    url = model.url(run, fh)
    weights = np.gradient(np.asarray(sorted(model.thick_levels), dtype=float) * 100.0) / G
    out = {}
    for name, key in _MASS:
        acc, seen = None, None
        for rng, positions in _groups(ranges[name]):
            msgs = split_messages(get(url, rng))
            if len(msgs) != len(positions):
                raise ValueError(f"{name}: asked for {len(positions)} messages, got {len(msgs)}")
            for pos, raw in zip(positions, msgs):
                g = eccodes.codes_new_from_message(raw)
                try:
                    if "lat" not in out:
                        lat = eccodes.codes_get_array(g, "latitudes")
                        lon = eccodes.codes_get_array(g, "longitudes")
                        out["lat"] = lat
                        out["lon"] = np.where(lon > 180, lon - 360, lon)
                    v = _decode(g)
                finally:
                    eccodes.codes_release(g)
                good = ~np.isnan(v)
                if acc is None:
                    acc, seen = np.zeros_like(v), np.zeros(v.shape, dtype=bool)
                acc += np.where(good, v, 0.0) * weights[pos]
                seen |= good
        out[key] = np.where(seen, acc, np.nan)
    return out


# ---- Thickness from column fields (IFS) ---------------------------------------
# ECMWF's tciw / tclw are already the column's cloud ice and liquid water in
# kg/m2: one sfc message each, read from the same .index as the cover.
_KG_M2 = "kg m**-2"


def read_column_thick(model, run, fh, get):
    """{"lat", "lon", "ice", "water"} from model.thick_fields, kg/m2 as
    published (negative or missing values NaN). One range request per field.
    ValueError if a message is not in kg m**-2; KeyError if the index lacks a field."""
    keys = [k for k, _ in model.thick_fields]
    ranges = _ecmwf_ranges(model, get(model.index_url(run, fh)).decode(), keys)
    url = model.url(run, fh)
    out = {}
    for key, name in model.thick_fields:
        g = eccodes.codes_new_from_message(get(url, ranges[key]))
        try:
            units = eccodes.codes_get(g, "units")
            if units != _KG_M2:
                raise ValueError(f"{key}: expected {_KG_M2}, got {units!r}")
            if "lat" not in out:
                lat = eccodes.codes_get_array(g, "latitudes")
                lon = eccodes.codes_get_array(g, "longitudes")
                out["lat"] = lat
                out["lon"] = np.where(lon > 180, lon - 360, lon)
            out[name] = _decode(g)
        finally:
            eccodes.codes_release(g)
    return out


# ---- Wind (ICW1) --------------------------------------------------------------
# u and v for the three decks (ICW1 keys 850 / 500 / 250 = low / mid / high, see
# pipeline.wind.LEVELS): each deck's wind over the model's own cloud layer
# (pipeline.layerwind). GFS: from UGRD / VGRD / TCDC at every isobaric level
# 1000-150 mb and the surface pressure. AIFS / IFS (no cloud on levels in their
# open data): from u / v and the humidity run (RH >= 90 %, over ice below 0 C) at
# every pressure level 1000-150 hPa, from q and t (both models), with sp.

# GFS's isobaric levels 1000-150 hPa (all present in the 0.25 degree idx at f000 and
# f120, checked 2026-11-18). A missing row at any of them fails the step: a level
# that quietly dropped out would shift every weight in its deck.
GFS_WIND_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
                   600, 550, 500, 450, 400, 350, 300, 250, 200, 150)


def _ncep_wind_ranges(index_text, fh):
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [ln.split(":") for ln in index_text.strip().split("\n") if ln.strip()]

    def find(name, level):
        k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
                  and r[4] == level and r[5] == want), None)
        if k is None:
            raise KeyError(f"{name} ({level}, {want}) not in idx")
        return int(rows[k][1]), int(rows[k + 1][1]) - 1 if k + 1 < len(rows) else None

    out = {}
    for lev in GFS_WIND_LEVELS:
        for comp, name in (("u", "UGRD"), ("v", "VGRD"), ("tcdc", "TCDC")):
            out[(lev, comp)] = find(name, f"{lev} mb")
    out[("sfc", "pres")] = find("PRES", "surface")
    return out


# The pressure levels 1000-150 hPa each ECMWF open-data index lists for u, v, t and
# q (AIFS 2026-11-18 00Z and IFS 2026-11-18 12Z, step 6, S3 mirror), and the
# humidity param each model's run is found from. Both read specific humidity q:
# AIFS publishes no r, and IFS's r is NOT RH over water below 0 C (measured
# 2026-11-18 00Z f006 against RH from IFS's own q and t: r / RH_ice(q) = 1.00 at
# 200-500 hPa below -23 C, r / RH_water(q) = 1.00 above 0 C; ECMWF's mixed-phase
# saturation), so ICON's water-to-ice step on r would count the ice twice. Levels
# above 150 hPa are never read; a missing row at any listed level fails the step,
# as for GFS.
AIFS_WIND_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150)
IFS_WIND_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150)
ECMWF_WIND = {"aifs": (AIFS_WIND_LEVELS, "q"), "ifs": (IFS_WIND_LEVELS, "q")}


def _ecmwf_wind_ranges(model, index_text):
    levels, hum = ECMWF_WIND[model.name]
    rows = [json.loads(ln) for ln in index_text.strip().split("\n") if ln.strip()]

    def find(param, levtype, lev=None):
        r = next((r for r in rows if r.get("param") == param and r.get("levtype") == levtype
                  and (lev is None or str(r.get("levelist")) == str(lev))), None)
        if r is None:
            raise KeyError(f"{param} ({levtype}{'' if lev is None else f' {lev}'}) not in index")
        return r["_offset"], r["_offset"] + r["_length"] - 1

    out = {}
    for lev in levels:
        for comp in ("u", "v", hum, "t"):
            out[(lev, comp)] = find(comp, "pl", lev)
    out[("sfc", "sp")] = find("sp", "sfc")
    return out


def wind_ranges(model, index_text, fh):
    """{key: (start, end)} for the wind messages of one forecast hour; end is
    inclusive or None for the file's last message. NCEP (GFS): keys
    (level hPa, 'u'|'v'|'tcdc') for each of GFS_WIND_LEVELS, plus ('sfc', 'pres'); rows picked by exact name, level and
    forecast text ("anl" at f000). ECMWF: (level, 'u'|'v'|'q'|'t') for each of
    the model's ECMWF_WIND levels, plus ('sfc', 'sp'); rows by param, levtype pl
    (sfc for sp) and levelist. KeyError if a row is missing."""
    if model.index_kind == "ncep":
        return _ncep_wind_ranges(index_text, fh)
    return _ecmwf_wind_ranges(model, index_text)


def _check_message(g, want_lev, want_comp):
    """The index said which message this is; the bytes must agree."""
    level = int(round(float(eccodes.codes_get(g, "level"))))
    short = str(eccodes.codes_get(g, "shortName"))
    if want_lev == "sfc":
        # PRES also exists at the tropopause, max wind and cloud base/top, all level 0
        if short not in ("sp", "pres") or level != 0 or str(eccodes.codes_get(g, "typeOfLevel")) != "surface":
            raise ValueError(f"wind: expected surface pressure, message is {short} {level}")
    elif want_comp == "tcdc":
        # the shortName of a pressure-level cloud cover is not stable across ecCodes
        # tables: check the GRIB2 parameter (discipline 0, category 6, number 1) instead
        triple = tuple(int(eccodes.codes_get(g, k)) for k in ("discipline", "parameterCategory", "parameterNumber"))
        if level != want_lev or triple != (0, 6, 1):
            raise ValueError(f"wind: expected tcdc {want_lev} hPa, message is {short} {level} {triple}")
    elif (short, level) != (want_comp, want_lev) or str(eccodes.codes_get(g, "typeOfLevel")) != "isobaricInhPa":
        raise ValueError(f"wind: expected {want_comp} {want_lev} hPa, message is {short} {level} "
                         f"{eccodes.codes_get(g, 'typeOfLevel')}")


def read_wind_step(model, run, fh, get):
    """The wind of one forecast hour: {"lat": 1-D, "lon": 1-D (-180..180),
    (key, 'u'|'v'): float [nlat][nlon] m/s}, the source's regular grid. get(url,
    rng=None) -> bytes is injected; messages whose rows touch are fetched as one
    range. Every model reads its levels 1000-150 hPa and returns the three decks'
    cloud-layer winds (float32) under keys 850 / 500 / 250 (low / mid / high):
    GFS from TCDC, AIFS / IFS from the humidity run."""
    out = read_wind_fields(model, run, fh, get)
    if model.index_kind != "ncep":
        return ecmwf_deck_winds(model, out)
    return gfs_deck_winds(out)


def read_wind_fields(model, run, fh, get):
    """The messages wind_ranges asks for, decoded as {"lat", "lon", key: [nlat][nlon]}
    (float32 for GFS, float64 for ECMWF; NaN where missing), each checked against
    its index row."""
    ranges = wind_ranges(model, get(model.index_url(run, fh)).decode(), fh)
    keys = list(ranges)
    url = model.url(run, fh)
    out = {}
    for rng, positions in _groups([(k, ranges[k]) for k in keys]):
        msgs = split_messages(get(url, rng))
        if len(msgs) != len(positions):
            raise ValueError(f"wind: asked for {len(positions)} messages, got {len(msgs)}")
        for pos, raw in zip(positions, msgs):
            g = eccodes.codes_new_from_message(raw)
            try:
                ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
                if "lat" not in out:
                    lat = eccodes.codes_get_array(g, "latitudes").reshape(nj, ni)
                    lon = eccodes.codes_get_array(g, "longitudes").reshape(nj, ni)
                    out["lat"] = lat[:, 0]
                    out["lon"] = np.where(lon[0] > 180, lon[0] - 360, lon[0])
                _check_message(g, *keys[pos])
                vals = eccodes.codes_get_values(g).reshape(nj, ni)
                miss = eccodes.codes_get(g, "missingValue")
            finally:
                eccodes.codes_release(g)
            vals = np.where(np.isclose(vals, miss), np.nan, vals)
            # GFS fields are cast to float32 to hold 61 per step; ECMWF stays float64 as before
            out[keys[pos]] = vals.astype(np.float32) if model.index_kind == "ncep" else vals
    return out


def gfs_deck_winds(fields):
    """read_wind_fields' GFS result -> {"lat", "lon", (850|500|250, 'u'|'v'): float32
    [nlat][nlon]}: each deck's wind over its cloud layer (pipeline.layerwind)."""
    out = {"lat": fields["lat"], "lon": fields["lon"]}
    levels = sorted({k[0] for k in fields if isinstance(k, tuple) and k[0] != "sfc"}, reverse=True)
    ps = fields[("sfc", "pres")]
    shape = ps.shape
    flat = lambda a: a.reshape(-1)
    ps_hpa = flat(ps).astype(np.float64) / 100.0
    u = [flat(fields[(lev, "u")]) for lev in levels]
    v = [flat(fields[(lev, "v")]) for lev in levels]
    cloudy = [flat(fields[(lev, "tcdc")]) >= layerwind.TCDC_CLOUDY for lev in levels]   # NaN is never cloudy
    decks = layerwind.deck_winds(levels, u, v, cloudy, ps_hpa, layerwind.deck_bounds("ncep", ps_hpa))
    for k, a in decks.items():
        out[k] = a.reshape(shape)
    return out


def ecmwf_deck_winds(model, fields):
    """read_wind_fields' AIFS / IFS result -> {"lat", "lon", (850|500|250, 'u'|'v'):
    float32 [nlat][nlon]}: each deck's wind over its humidity run (RH >= 90 %, over
    ice below 0 C, from q, t and the level's pressure), deck bounds 0.8 / 0.45 x
    surface pressure (sp, Pa -> hPa)."""
    levels, hum = ECMWF_WIND[model.name]
    if hum != "q":
        raise ValueError(f"{model.name}: wind humidity must be q, not {hum!r}")
    out = {"lat": fields["lat"], "lon": fields["lon"]}
    ps = fields[("sfc", "sp")]
    shape = ps.shape
    flat = lambda a: a.reshape(-1)
    ps_hpa = flat(ps).astype(np.float64) / 100.0
    u = [flat(fields[(lev, "u")]) for lev in levels]
    v = [flat(fields[(lev, "v")]) for lev in levels]
    moist = [layerwind.moist_from_q(flat(fields[(lev, "q")]), flat(fields[(lev, "t")]), lev) for lev in levels]
    decks = layerwind.deck_winds(levels, u, v, moist, ps_hpa, layerwind.deck_bounds("ecmwf", ps_hpa))
    for k, a in decks.items():
        out[k] = a.reshape(shape)
    return out
