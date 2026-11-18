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

from hrrr.grib import WIND_LEVELS, rotate_to_earth, wind_ranges

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


# ---- Wind (ICW1) --------------------------------------------------------------
# u/v at 850 / 500 / 250 mb (the low, mid and high deck's steering levels) from
# the 13 km "prslev" file, same NA grid. The idx rows have the surface file's
# layout ("250 mb", "anl" / "N hour fcst"), so hrrr.grib.wind_ranges reads them.
# The file lists each level's U then V back to back, so the two are one ranged
# request: three requests a forecast hour plus the idx.
#
# The components follow the grid (uvRelativeToGrid = 1), and RRFS's 13 km NA
# grid is a ROTATED lat/lon grid (gridType rotated_ll), not Lambert: turned to
# east/north with the rotated pole, not hrrr.grib.rotate_to_earth's convergence.
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


def _requests(ranges):
    """[(start, end, [keys...])]: a level's U and V rows that touch are one request."""
    out = []
    for lev in WIND_LEVELS:
        (us, ue), (vs, ve) = ranges[(lev, "u")], ranges[(lev, "v")]
        if ue is not None and ue + 1 == vs:
            out.append((us, ve, [(lev, "u"), (lev, "v")]))
        else:
            out += [(us, ue, [(lev, "u")]), (vs, ve, [(lev, "v")])]
    return out


def _messages(blob, ranges, keys):
    """Split one ranged response into its messages, by the idx's byte lengths."""
    if len(keys) == 1:
        return [blob]
    cut = ranges[keys[0]][1] - ranges[keys[0]][0] + 1
    return [blob[:cut], blob[cut:]]


def read_wind(run, fh, get):
    """The six wind fields of one forecast hour, turned to east/north, in m/s:
    {"lat", "lon" (-180..180), (level, 'u'|'v'): array}, flat over RRFS's cells,
    NaN where missing. get(url, rng=None) -> bytes is injected. Each message's
    own shortName and level must match the idx row that asked for it."""
    url = prs_url(run, fh)
    ranges = wind_ranges(get(url + ".idx").decode(), fh)
    out, raw, turn = {}, {}, None
    for start, end, keys in _requests(ranges):
        blob = get(url, (start, end))
        if len(keys) == 2 and end is not None and len(blob) != end - start + 1:
            raise ValueError(f"wind: asked {end - start + 1} bytes, got {len(blob)}")
        for key, data in zip(keys, _messages(blob, ranges, keys)):
            g = eccodes.codes_new_from_message(data)
            try:
                got = (str(eccodes.codes_get(g, "shortName")), int(round(float(eccodes.codes_get(g, "level")))))
                if got != (key[1], key[0]):
                    raise ValueError(f"wind: expected {key[1]} {key[0]} hPa, message is {got[0]} {got[1]}")
                if "lat" not in out:
                    out["lat"], out["lon"], turn = _wind_geometry(g)
                vals = eccodes.codes_get_values(g)
                miss = eccodes.codes_get(g, "missingValue")
            finally:
                eccodes.codes_release(g)
            raw[key] = np.where(np.isclose(vals, miss), np.nan, vals)
    for lev in WIND_LEVELS:
        u, v = raw.pop((lev, "u")), raw.pop((lev, "v"))
        if turn is not None:
            u, v = turn(u, v)
        out[(lev, "u")], out[(lev, "v")] = u, v
    return out
