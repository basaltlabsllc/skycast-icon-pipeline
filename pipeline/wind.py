# pipeline/wind.py — the wind lattice (ICW1): u/v at 850, 500 and 250 hPa on a
# 1° lattice, cut into 20° groups, for the app's cloud drift.
#
# Every feed reduces its model to one world lattice per step and level:
# float [181][360], node (i, j) = lat -90+i, lon -180+j, the mean of the
# model's points within ±0.5° of the node (NaN where it has none). A group is
# 20° on a side with 21 nodes a side (the north and east edges belong to the
# neighbour too, like the 41-sample cloud squares), 162 groups in all.
#
# ICW1, little-endian: "ICW1", swLat i16, swLon i16, cols u16 = 21, rows u16 =
# 21, steps u16; steps × u32 unix HOURS; 3 × u16 levels (850, 500, 250); then
# ONE zlib stream of int8 [step][deck 3][comp 2 (u, v)][row][col] in m/s,
# rounded to 1 m/s, clipped to ±127, -128 = missing. Row 0 is SOUTH, col 0 WEST.
import struct
import zlib
from pathlib import Path
import numpy as np
from .dwd import run_id, run_iso
from .site import wind_path

MAGIC = b"ICW1"
LEVELS = (850, 500, 250)            # low, mid, high deck
GROUP_DEG = 20
SIDE = 21
MISSING = -128
NLAT, NLON = 181, 360
_HEADER = struct.Struct("<4shhHHH")   # magic, swLat, swLon, cols, rows, steps
_EPS = 1e-6


def group_name(sw_lat, sw_lon):
    if sw_lon == 180:
        sw_lon = -180
    if (sw_lat + 90) % GROUP_DEG or (sw_lon + 180) % GROUP_DEG or not (-90 <= sw_lat <= 70) or not (-180 <= sw_lon <= 160):
        raise ValueError(f"not a group corner: {sw_lat}, {sw_lon}")
    return f"{'N' if sw_lat >= 0 else 'S'}{abs(sw_lat):02d}{'E' if sw_lon >= 0 else 'W'}{abs(sw_lon):03d}"


def group_of(lat, lon):
    """The group whose cell holds the whole-degree node (lat, lon)."""
    la = min(70, -90 + GROUP_DEG * int(np.floor((lat + 90) / GROUP_DEG)))
    lo = ((int(np.floor(lon)) + 180) % 360) // GROUP_DEG * GROUP_DEG - 180
    return group_name(la, lo)


def parse_group(name):
    if len(name) != 7 or name[0] not in "NS" or name[3] not in "EW":
        raise ValueError(f"bad group name {name!r}")
    lat = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "E" else -1)
    return lat, lon


def all_groups():
    return [group_name(la, lo) for la in range(-90, 90, GROUP_DEG) for lo in range(-180, 180, GROUP_DEG)]


def _window_sums(cum, coords, nodes):
    """Sum over coords within ±0.5 of each node, from a cumulative sum along axis 0."""
    lo = np.searchsorted(coords, nodes - 0.5 - _EPS, side="left")
    hi = np.searchsorted(coords, nodes + 0.5 + _EPS, side="right")
    return cum[hi] - cum[lo]


def lattice_from_regular(lat1d, lon1d, field2d):
    """Box mean of a regular lat/lon field onto the 1° lattice (see above)."""
    lat = np.asarray(lat1d, dtype=np.float64)
    lon = (np.asarray(lon1d, dtype=np.float64) + 180.0) % 360.0 - 180.0
    f = np.asarray(field2d, dtype=np.float64)
    if lat.size > 1 and lat[0] > lat[-1]:
        lat, f = lat[::-1], f[::-1]
    order = np.argsort(lon, kind="stable")
    lon, f = lon[order], f[:, order]
    ok = ~np.isnan(f)
    vals = np.where(ok, f, 0.0)
    # latitude first: [181][nlon] sums and counts
    z = np.zeros((1, f.shape[1]))
    nodes_lat = np.arange(-90, 91, dtype=np.float64)
    cs = _window_sums(np.concatenate([z, np.cumsum(vals, axis=0)]), lat, nodes_lat)
    cn = _window_sums(np.concatenate([z, np.cumsum(ok, axis=0, dtype=np.int64)]), lat, nodes_lat)
    # longitude: tile three times so the window wraps through ±180
    lon3 = np.concatenate([lon - 360.0, lon, lon + 360.0])
    cs3 = np.concatenate([cs, cs, cs], axis=1).T      # [3*nlon][181]
    cn3 = np.concatenate([cn, cn, cn], axis=1).T
    zc = np.zeros((1, nodes_lat.size))
    nodes_lon = np.arange(-180, 180, dtype=np.float64)
    s = _window_sums(np.concatenate([zc, np.cumsum(cs3, axis=0)]), lon3, nodes_lon)
    n = _window_sums(np.concatenate([zc, np.cumsum(cn3, axis=0)]), lon3, nodes_lon)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out.T


def lattice_from_points(lat, lon, values):
    """Mean of scattered points, each binned to its nearest node when within ±0.5°."""
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = (np.asarray(lon, dtype=np.float64).ravel() + 180.0) % 360.0 - 180.0
    v = np.asarray(values, dtype=np.float64).ravel()
    i = np.rint(lat)
    j = np.rint(lon)
    keep = (np.abs(lat - i) <= 0.5 + _EPS) & (np.abs(lon - j) <= 0.5 + _EPS) & ~np.isnan(v) & (i >= -90) & (i <= 90)
    i = (i[keep] + 90).astype(np.int64)
    j = (j[keep].astype(np.int64) + 180) % NLON
    flat = i * NLON + j
    s = np.bincount(flat, weights=v[keep], minlength=NLAT * NLON)
    n = np.bincount(flat, minlength=NLAT * NLON)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out.reshape(NLAT, NLON)


def group_cube(lattices_per_step, name):
    """int8 [steps][3][2][21][21] for one group; lattices_per_step[s] = {(level, 'u'|'v'): lattice}."""
    lat0, lon0 = parse_group(name)
    rows = slice(lat0 + 90, lat0 + 90 + SIDE)
    cols = (lon0 + 180 + np.arange(SIDE)) % NLON      # lon 180 reads lon -180
    cube = np.full((len(lattices_per_step), len(LEVELS), 2, SIDE, SIDE), MISSING, dtype=np.int8)
    for s, step in enumerate(lattices_per_step):
        for d, level in enumerate(LEVELS):
            for c, comp in enumerate("uv"):
                sub = np.asarray(step[(level, comp)], dtype=np.float64)[rows][:, cols]
                q = np.clip(np.rint(np.where(np.isnan(sub), 0.0, sub)), -127, 127)
                cube[s, d, c] = np.where(np.isnan(sub), MISSING, q).astype(np.int8)
    return cube


def encode_icw1(sw_lat, sw_lon, step_hours, cube):
    """cube: int8 [steps][3][2][21][21]. step_hours: ascending unix hours."""
    cube = np.asarray(cube)
    if cube.dtype != np.int8:
        raise ValueError(f"cube must be int8, got {cube.dtype}")
    if cube.ndim != 5 or cube.shape[1:] != (len(LEVELS), 2, SIDE, SIDE):
        raise ValueError(f"cube must be [steps][3][2][{SIDE}][{SIDE}], got {cube.shape}")
    hours = [int(h) for h in step_hours]
    if len(hours) != cube.shape[0]:
        raise ValueError(f"{len(hours)} step hours for {cube.shape[0]} steps")
    if any(b <= a for a, b in zip(hours, hours[1:])):
        raise ValueError("step hours must be strictly ascending")
    head = _HEADER.pack(MAGIC, int(sw_lat), int(sw_lon), SIDE, SIDE, len(hours))
    return (head + struct.pack(f"<{len(hours)}I", *hours) + struct.pack(f"<{len(LEVELS)}H", *LEVELS)
            + zlib.compress(np.ascontiguousarray(cube).tobytes(), 9))


def decode_icw1(buf):
    magic, sw_lat, sw_lon, cols, rows, steps = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("not an ICW1 file")
    off = _HEADER.size
    hours = list(struct.unpack_from(f"<{steps}I", buf, off))
    off += 4 * steps
    levels = list(struct.unpack_from(f"<{len(LEVELS)}H", buf, off))
    off += 2 * len(LEVELS)
    body = zlib.decompress(buf[off:])
    cube = np.frombuffer(body, dtype=np.int8).reshape(steps, len(LEVELS), 2, rows, cols)
    return {"sw_lat": sw_lat, "sw_lon": sw_lon, "cols": cols, "rows": rows,
            "step_hours": hours, "levels": levels, "cube": cube}


def write_wind(out_dir, run, step_hours, lattices_per_step, names=None):
    """One .icw per group under wind/<run>/ for every group with a non-missing
    value (or only `names`, when given); returns the bytes written."""
    out = Path(out_dir) / wind_path(run)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in (names if names is not None else all_groups()):
        cube = group_cube(lattices_per_step, name)
        if not (cube != MISSING).any():
            continue
        lat, lon = parse_group(name)
        data = encode_icw1(lat, lon, step_hours, cube)
        (out / f"{name}.icw").write_bytes(data)
        total += len(data)
    return total


def manifest_entry(run):
    return {"format": "ICW1", "run": run_iso(run), "runMs": int(run.timestamp() * 1000),
            "tiles": wind_path(run), "levels": list(LEVELS)}
