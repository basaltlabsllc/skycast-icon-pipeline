# pipeline/clearnight.py — SkyCast's Clear Night map: total cloud cover over
# each night's astronomically dark hours (Sun below −18°), per 0.125° node.
#
# Design: SkyCast docs/superpowers/specs/2026-11-17-clear-night-map-design.md.
# The phone combines these cloud layers with its own Moon and light-pollution
# terms, so the server ships the parts, never a finished score.
#
# Night D at a node is the 24 hours from local SOLAR noon of D (12:00 UTC −
# lon/15 h): the night is centred on solar midnight whatever the time zone.
# Its dark hours are the whole UTC hours inside that window whose Sun is below
# −18°. Cover at each dark hour comes from the run's own hours (history
# included), linear between two that bracket it — the 3-hourly steps past
# +78 h are bridged that way. A dark hour no two hours bracket, or a MISSING
# sample, makes the node no data for that night: a night half covered would
# read as a better or worse night than the model said.
#
# CLN1, one file per 20° tile per night: a 12-byte header (magic, swLat, swLon,
# side, layers, clearMax), the night as 10 ASCII bytes, then a zlib stream of
# u8 [4][161][161], south row first, west column first, edges included like
# the 5° squares'. Layers: mean cover %, cloudiest hour %, clear hours (cover
# ≤ CLEAR_MAX), dark hours. 255 is no data; dark hours 0 marks a night with no
# astronomical darkness (the other three are 255 there).
import datetime as dt
import struct
import zlib
from pathlib import Path
import numpy as np

from .squares import DEG, NX, NY

MAGIC = b"CLN1"
TILE_DEG = 20
TILE_SIDE = TILE_DEG * 8 + 1      # 161
LAYERS = 4
CLEAR_MAX = 20                    # a dark hour with total cover at or under this is clear
DARK_ALT_DEG = -18.0
MISSING = 255
_HEADER = struct.Struct("<4shhHBB")


def sun_altitude_deg(lat, lon, unix_hour):
    """The Sun's altitude in degrees (the Astronomical Almanac's low-precision
    formulae, good to ~0.01° in declination). Broadcasts over numpy arrays."""
    n = np.asarray(unix_hour, dtype=np.float64) / 24.0 + 2440587.5 - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    gmst_deg = (280.46061837 + 360.98564736629 * n) % 360.0
    ha = np.radians(gmst_deg + np.asarray(lon, dtype=np.float64)) - ra
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    return np.degrees(np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)))


def _night_base_hour(night):
    d = dt.datetime.strptime(night, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return d.timestamp() / 3600.0 + 12.0


def night_stats(cover_at, hours, lats, lons, night):
    """u8 [4][R][C] for night `night`: cover_at u8 [len(hours)][R][C] total
    cover, hours ascending unix hours, lats [R], lons [C] in degrees."""
    cover_at = np.asarray(cover_at)
    hours = np.asarray(hours, dtype=np.int64)
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    R, C = len(lats), len(lons)
    start = _night_base_hour(night) - lons / 15.0          # [C] window start, fractional hours
    h0 = int(np.floor(start.min()))
    h1 = int(np.ceil(start.max() + 24.0))
    total = np.zeros((R, C), np.float64)
    worst = np.zeros((R, C), np.int16)
    clear = np.zeros((R, C), np.int16)
    dark = np.zeros((R, C), np.int16)
    missing = np.zeros((R, C), bool)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    for h in range(h0, h1 + 1):
        inwin = (h >= start) & (h < start + 24.0)
        if not inwin.any():
            continue
        d = (sun_altitude_deg(la, lo, h) < DARK_ALT_DEG) & inwin[None, :]
        if not d.any():
            continue
        i = int(np.searchsorted(hours, h))
        if i < len(hours) and hours[i] == h:
            raw = cover_at[i].astype(np.int16)
            bad = raw == MISSING
            cov = raw.astype(np.float64)
        elif 0 < i < len(hours):
            a, b = cover_at[i - 1].astype(np.int16), cover_at[i].astype(np.int16)
            bad = (a == MISSING) | (b == MISSING)
            t = (h - hours[i - 1]) / float(hours[i] - hours[i - 1])
            cov = a + (b - a) * t
        else:
            missing |= d
            continue
        missing |= d & bad
        ok = d & ~bad
        total[ok] += cov[ok]
        r = np.rint(cov).astype(np.int16)
        worst[ok] = np.maximum(worst[ok], r[ok])
        clear[ok] += (cov[ok] <= CLEAR_MAX)
        dark[d] += 1
    out = np.full((LAYERS, R, C), MISSING, np.uint8)
    has = (dark > 0) & ~missing
    out[0][has] = np.rint(total[has] / dark[has]).astype(np.uint8)
    out[1][has] = worst[has]
    out[2][has] = clear[has]
    out[3][~missing] = dark[~missing]
    return out


def tile_name(sw_lat, sw_lon):
    return f"{'N' if sw_lat >= 0 else 'S'}{abs(sw_lat):02d}{'E' if sw_lon >= 0 else 'W'}{abs(sw_lon):03d}"


def tile_keys():
    return [tile_name(la, lo) for la in range(-90, 90, TILE_DEG) for lo in range(-180, 180, TILE_DEG)]


def tiles_for_squares(names):
    """The 20° tiles holding these 5° squares, in tile_keys order (a test or
    partial build makes only those; a full build's squares give all 162)."""
    from .squares import parse_name
    want = set()
    for n in names:
        la, lo = parse_name(n)
        want.add(tile_name(la - (la + 90) % TILE_DEG, lo - (lo + 180) % TILE_DEG))
    return [k for k in tile_keys() if k in want]


def _parse(key):
    lat = int(key[1:3]) * (1 if key[0] == "N" else -1)
    lon = int(key[4:7]) * (1 if key[3] == "E" else -1)
    return lat, lon


def _cols(sw_lon):
    c0 = round((sw_lon + 180) / DEG)
    out = []
    for c in range(c0, c0 + TILE_SIDE):
        if c >= NX:                     # the seam, as squares.col_indices
            c -= 1
        out.append(c % NX)
    return out


def slice_tile(world2d, sw_lat, sw_lon):
    """world [NY][NX] → [161][161], south row first, west column first."""
    r0 = round((sw_lat + 90) / DEG)
    return np.ascontiguousarray(np.asarray(world2d)[r0:r0 + TILE_SIDE][:, _cols(sw_lon)])


def tile_axes(sw_lat, sw_lon):
    """(lats [161], lons [161]) of a tile's nodes."""
    i = np.arange(TILE_SIDE) * DEG
    return sw_lat + i, sw_lon + i


def encode_tile(sw_lat, sw_lon, night, layers):
    layers = np.asarray(layers, dtype=np.uint8)
    if layers.shape != (LAYERS, TILE_SIDE, TILE_SIDE):
        raise ValueError(f"layers must be [{LAYERS}][{TILE_SIDE}][{TILE_SIDE}], got {layers.shape}")
    nb = night.encode("ascii")
    if len(nb) != 10:
        raise ValueError(f"bad night {night!r}")
    head = _HEADER.pack(MAGIC, int(sw_lat), int(sw_lon), TILE_SIDE, LAYERS, CLEAR_MAX)
    return head + nb + zlib.compress(layers.tobytes(), 9)


def decode_tile(blob):
    magic, sw_lat, sw_lon, side, n, clear_max = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("not a CLN1 file")
    off = _HEADER.size
    night = blob[off:off + 10].decode("ascii")
    body = zlib.decompress(blob[off + 10:])
    return {"sw_lat": sw_lat, "sw_lon": sw_lon, "side": side, "night": night, "clear_max": clear_max,
            "layers": np.frombuffer(body, np.uint8).reshape(n, side, side)}


def nights_for(hours):
    """Candidate nights: the UTC date before the first hour's through the last hour's."""
    first = dt.datetime.fromtimestamp(int(hours[0]) * 3600, dt.timezone.utc).date() - dt.timedelta(days=1)
    last = dt.datetime.fromtimestamp(int(hours[-1]) * 3600, dt.timezone.utc).date()
    out, d = [], first
    while d <= last:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def path(run_id):
    return f"clearnight/{run_id}"


def build_clearnight(out_dir, run_id, hours, cover_world, keys=None, first_night=None):
    """Every listed night's tiles under out_dir/clearnight/<run_id>/<night>/;
    cover_world(i) → the CLCT world [NY][NX] (u8) of hours[i]. A night is
    listed when at least one node has data; nights before first_night
    ('YYYY-MM-DD', the run's UTC date − 1: the oldest any zone can still call
    tonight) are left out. Returns the manifest entry."""
    keys = list(keys) if keys is not None else tile_keys()
    hours = [int(h) for h in hours]
    root = Path(out_dir) / path(run_id)
    # Each tile's cover over every hour, cut once: [hours][161][161] per tile.
    cubes = {k: np.empty((len(hours), TILE_SIDE, TILE_SIDE), np.uint8) for k in keys}
    for i in range(len(hours)):
        w = cover_world(i)
        for k in keys:
            cubes[k][i] = slice_tile(w, *_parse(k))
    nights, sizes = [], {}
    for night in nights_for(hours):
        if first_night and night < first_night:
            continue
        files, any_data = {}, False
        for k in keys:
            la, lo = _parse(k)
            lats, lons = tile_axes(la, lo)
            layers = night_stats(cubes[k], hours, lats, lons, night)
            any_data |= bool((layers[0] != MISSING).any())
            files[k] = encode_tile(la, lo, night, layers)
        if not any_data:
            continue
        d = root / night
        d.mkdir(parents=True, exist_ok=True)
        for k, b in files.items():
            (d / f"{k}.cln").write_bytes(b)
        nights.append(night)
        sizes[night] = sum(len(b) for b in files.values())
    if not nights:
        return None
    return {"format": "CLN1", "tiles": path(run_id), "nights": nights, "deg": TILE_DEG, "side": TILE_SIDE,
            "clearMax": CLEAR_MAX, "darkAltDeg": DARK_ALT_DEG, "keys": keys, "bytes": sizes}
