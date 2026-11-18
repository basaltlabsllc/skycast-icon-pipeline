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
# CLN2, one file per 20° tile per night: a 16-byte header (magic, swLat, swLon,
# side, layers, clearMax, anchor), the night as 10 ASCII bytes, then a zlib
# stream of u8 [8][161][161], south row first, west column first, edges
# included like the 5° squares'. Layers 0-3: mean cover %, cloudiest hour %,
# clear hours (cover ≤ CLEAR_MAX), dark hours. 255 is no data; dark hours 0
# marks a night with no astronomical darkness (the other three are 255 there).
# Layers 4-7: WHICH dark hours are clear, a 32-bit mask little-endian across
# the four (layer 4 the low byte): bit j set = the UTC hour anchor + j (unix
# hours, int32 in the header: the tile's earliest window hour) is dark and
# clear by the same CLEAR_MAX. 0 where the node has no data. CLN1 (12-byte
# header, the first four layers only) is what was published before
# 2026-11-18; decode_tile still reads it (anchor None).
import datetime as dt
import struct
import zlib
from pathlib import Path
import numpy as np

from .squares import DEG, NX, NY

MAGIC = b"CLN2"
FORMAT = "CLN2"
TILE_DEG = 20
TILE_SIDE = TILE_DEG * 8 + 1      # 161
STAT_LAYERS = 4
MASK_BYTES = 4                    # 32 hours from the anchor; a tile's windows span < 27
LAYERS = STAT_LAYERS + MASK_BYTES
CLEAR_MAX = 20                    # a dark hour with total cover at or under this is clear
DARK_ALT_DEG = -18.0
MISSING = 255
_HEADER = struct.Struct("<4shhHBBi")
_HEADER_V1 = struct.Struct("<4shhHBB")


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
    return night_layers(cover_at, hours, lats, lons, night)[0][:STAT_LAYERS]


def clear_hours(layers, anchor, r, c):
    """The unix hours marked dark and clear at node (r, c) of night_layers' output."""
    m = 0
    for b in range(MASK_BYTES):
        m |= int(layers[STAT_LAYERS + b][r][c]) << (8 * b)
    return [anchor + j for j in range(8 * MASK_BYTES) if m >> j & 1]


def night_layers(cover_at, hours, lats, lons, night):
    """(u8 [LAYERS][R][C], anchor): night_stats' four layers plus the clear-hour
    mask (see the header comment), anchor the first whole hour of the
    earliest window among the columns."""
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
    mask = np.zeros((R, C), np.uint32)
    if h1 - h0 >= 8 * MASK_BYTES:
        raise ValueError(f"windows span {h1 - h0} h, more than the {8 * MASK_BYTES}-hour mask")
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
        is_clear = ok & (cov <= CLEAR_MAX)
        clear[is_clear] += 1
        mask[is_clear] |= np.uint32(1 << (h - h0))
        dark[d] += 1
    out = np.full((LAYERS, R, C), MISSING, np.uint8)
    has = (dark > 0) & ~missing
    out[0][has] = np.rint(total[has] / dark[has]).astype(np.uint8)
    out[1][has] = worst[has]
    out[2][has] = clear[has]
    out[3][~missing] = dark[~missing]
    mask[~has] = 0
    for b in range(MASK_BYTES):
        out[STAT_LAYERS + b] = (mask >> (8 * b)) & 0xFF
    return out, h0


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


def encode_tile(sw_lat, sw_lon, night, layers, anchor):
    layers = np.asarray(layers, dtype=np.uint8)
    if layers.shape != (LAYERS, TILE_SIDE, TILE_SIDE):
        raise ValueError(f"layers must be [{LAYERS}][{TILE_SIDE}][{TILE_SIDE}], got {layers.shape}")
    nb = night.encode("ascii")
    if len(nb) != 10:
        raise ValueError(f"bad night {night!r}")
    head = _HEADER.pack(MAGIC, int(sw_lat), int(sw_lon), TILE_SIDE, LAYERS, CLEAR_MAX, int(anchor))
    return head + nb + zlib.compress(layers.tobytes(), 9)


def decode_tile(blob):
    """A CLN2 tile, or a CLN1 one (anchor None, four layers)."""
    if blob[:4] == MAGIC:
        magic, sw_lat, sw_lon, side, n, clear_max, anchor = _HEADER.unpack_from(blob, 0)
        off = _HEADER.size
    elif blob[:4] == b"CLN1":
        magic, sw_lat, sw_lon, side, n, clear_max = _HEADER_V1.unpack_from(blob, 0)
        anchor, off = None, _HEADER_V1.size
    else:
        raise ValueError("not a CLN1 or CLN2 file")
    night = blob[off:off + 10].decode("ascii")
    body = zlib.decompress(blob[off + 10:])
    return {"sw_lat": sw_lat, "sw_lon": sw_lon, "side": side, "night": night, "clear_max": clear_max,
            "anchor": anchor, "layers": np.frombuffer(body, np.uint8).reshape(n, side, side)}


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


def file_name(night, key):
    """A tile's file in its run folder: flat, so pipeline/pack.py's one pack
    (the files directly in the folder) holds every night. SkyCast's
    src/clearnightload.js reads the same name."""
    return f"{night}_{key}.cln"


def build_clearnight(out_dir, run_id, hours, cover_world, keys=None, first_night=None):
    """Every listed night's tiles under out_dir/clearnight/<run_id>/ (file_name);
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
            layers, anchor = night_layers(cubes[k], hours, lats, lons, night)
            any_data |= bool((layers[0] != MISSING).any())
            files[k] = encode_tile(la, lo, night, layers, anchor)
        if not any_data:
            continue
        root.mkdir(parents=True, exist_ok=True)
        for k, b in files.items():
            (root / file_name(night, k)).write_bytes(b)
        nights.append(night)
        sizes[night] = sum(len(b) for b in files.values())
    if not nights:
        return None
    return {"format": FORMAT, "tiles": path(run_id), "nights": nights, "deg": TILE_DEG, "side": TILE_SIDE,
            "clearMax": CLEAR_MAX, "darkAltDeg": DARK_ALT_DEG, "keys": keys, "bytes": sizes}
