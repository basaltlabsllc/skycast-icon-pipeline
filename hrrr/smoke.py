# hrrr/smoke.py — HRRR near-ground smoke (MASSDEN at 8 m) and smoke optical depth
# (AOTK, whole column) from the wrfsfc file, put on a fixed 0.125 degree lattice
# and packed as one byte per cell (SMK1).
#
# pm25 bytes are piecewise linear inside each EPA band, so every category edge
# is a byte edge and a category survives encode -> decode. AOD bytes are log
# spaced from 0.001 to 5.
import math
import struct
import zlib

import numpy as np
import eccodes

from hrrr.grib import _decode, _range_of, _rows, file_url

SMOKE_FIELDS = (
    ("MASSDEN", "8 m above ground"),
    ("AOTK", "entire atmosphere (considered as a single layer)"),
)
GRID = {"lat0": 20.0, "lon0": -135.0, "step": 0.125, "nlat": 273, "nlon": 617}

_E = np.array([0, 9.0, 35.4, 55.4, 125.4, 225.4, 325.4, 1000.0])
_B = np.array([0, 90, 142, 167, 202, 227, 240, 254])
_STEP = np.diff(_E) / np.diff(_B)
AOD_FLOOR = 0.001
_AOD_RATIO = 5000 ** (1 / 253)


def smoke_ranges(idx_text, fh):
    """{"MASSDEN": (start, end), "AOTK": (start, end)} for forecast hour fh
    (f00 reads "anl"); end is inclusive, None for the file's last message.
    KeyError if a field is missing."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = _rows(idx_text)
    out = {}
    for name, level in SMOKE_FIELDS:
        k = next((i for i, r in enumerate(rows) if len(r) > 5 and r[3] == name
                  and r[4] == level and r[5] == want), None)
        if k is None:
            raise KeyError(f"{name} ({level}, {want}) not in idx")
        out[name] = _range_of(rows, k)
    return out


def _lat_lon(raw):
    g = eccodes.codes_new_from_message(raw)
    try:
        lat = eccodes.codes_get_array(g, "latitudes")
        lon = eccodes.codes_get_array(g, "longitudes")
    finally:
        eccodes.codes_release(g)
    return lat, np.where(lon > 180, lon - 360, lon)


def read_smoke(run, fh, get):
    """{"pm25": µg/m3, "aod", "lat", "lon" (-180..180)}: one flat float array per
    field over HRRR's cells, NaN where missing or negative. get(url, rng=None)
    -> bytes is injected."""
    url = file_url(run, fh)
    r = smoke_ranges(get(url + ".idx").decode(), fh)
    raw = get(url, r["MASSDEN"])
    lat, lon = _lat_lon(raw)
    pm25 = _decode(raw) * 1e9            # kg/m3 -> µg/m3
    aod = _decode(get(url, r["AOTK"]))
    return {"pm25": pm25, "aod": aod, "lat": lat, "lon": lon}


def node_index(lat, lon):
    """Flat lattice index (row * nlon + col) of the nearest GRID node of each
    point; -1 outside the lattice."""
    g = GRID
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        i = np.floor((lat - g["lat0"]) / g["step"] + 0.5)
        j = np.floor((lon - g["lon0"]) / g["step"] + 0.5)
        ok = (i >= 0) & (i < g["nlat"]) & (j >= 0) & (j < g["nlon"])
    return np.where(ok, i * g["nlon"] + j, -1).astype(np.int64)


def lattice(values, idx):
    """Mean of the finite values of the cells that share a node, float32
    (nlat, nlon); NaN where no finite cell falls."""
    n = GRID["nlat"] * GRID["nlon"]
    values = np.asarray(values, dtype=np.float64)
    idx = np.asarray(idx)
    use = (idx >= 0) & np.isfinite(values)
    s = np.bincount(idx[use], weights=values[use], minlength=n)
    c = np.bincount(idx[use], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(c > 0, s / c, np.nan)
    return m.astype(np.float32).reshape(GRID["nlat"], GRID["nlon"])


def encode_pm25(a):
    """µg/m3 -> uint8. 0 = none, 255 = missing, 254 = 1000 or more."""
    a = np.asarray(a, dtype=np.float64)
    nan = np.isnan(a)
    v = np.where(nan, 0.0, a)
    i = np.clip(np.searchsorted(_E, v, side="left") - 1, 0, len(_STEP) - 1)
    b = _B[i] + np.ceil((v - _E[i]) / _STEP[i])
    b = np.where(v <= 0, 0, np.minimum(b, 254))
    return np.where(nan, 255, b).astype(np.uint8)


def decode_pm25(b):
    """uint8 -> µg/m3 float32; 255 is NaN."""
    b = np.asarray(b, dtype=np.int64)
    i = np.clip(np.searchsorted(_B, b, side="left") - 1, 0, len(_STEP) - 1)
    v = _E[i] + (b - _B[i]) * _STEP[i]
    v = np.where(b == 0, 0.0, v)
    return np.where(b == 255, np.nan, v).astype(np.float32)


def encode_aod(a):
    """AOD -> uint8: 0 below 0.001, 1..254 log spaced to 5, 255 missing."""
    a = np.asarray(a, dtype=np.float64)
    nan = np.isnan(a)
    v = np.where(nan | (a < AOD_FLOOR), AOD_FLOOR, a)
    b = np.clip(1 + np.round(np.log(v / AOD_FLOOR) / math.log(_AOD_RATIO)), 1, 254)
    b = np.where(nan, 255, np.where(a < AOD_FLOOR, 0, b))
    return b.astype(np.uint8)


def decode_aod(b):
    b = np.asarray(b, dtype=np.int64)
    v = AOD_FLOOR * _AOD_RATIO ** (np.clip(b, 1, 254) - 1)
    return np.where(b == 255, np.nan, np.where(b == 0, 0.0, v)).astype(np.float32)


def smk_bytes(valid_hour, pm25_u8, aod_u8):
    """zlib(b"SMK1" + u32 LE valid_hour + u16 LE nlat + u16 LE nlon + pm25 + aod)."""
    pm = np.ascontiguousarray(pm25_u8, dtype=np.uint8)
    ao = np.ascontiguousarray(aod_u8, dtype=np.uint8)
    if pm.shape != ao.shape or pm.ndim != 2:
        raise ValueError("smk: pm25 and aod must be 2-D arrays of one shape")
    nlat, nlon = pm.shape
    return zlib.compress(b"SMK1" + struct.pack("<IHH", int(valid_hour), nlat, nlon) + pm.tobytes() + ao.tobytes(), 9)


def parse_smk(raw):
    """(valid_hour, pm25_u8, aod_u8) from smk_bytes' output."""
    d = zlib.decompress(raw)
    if d[:4] != b"SMK1":
        raise ValueError("smk: bad magic")
    hour, nlat, nlon = struct.unpack("<IHH", d[4:12])
    n = nlat * nlon
    if len(d) != 12 + 2 * n:
        raise ValueError("smk: bad length")
    pm = np.frombuffer(d, np.uint8, n, 12).reshape(nlat, nlon)
    ao = np.frombuffer(d, np.uint8, n, 12 + n).reshape(nlat, nlon)
    return hour, pm, ao
