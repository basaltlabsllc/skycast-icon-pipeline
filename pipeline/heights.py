# pipeline/heights.py — real cloud heights per deck, from ICON's cloud cover on
# MODEL LEVELS (CLC, %) and the half-level heights (HHL, m above sea level),
# and ICH1, the file that carries them for one 5° square.
#
# WHICH LEVELS ARE A DECK. The low and mid deck are the model's own: the same
# levels ICON sums into its CLCL and CLCM cover figures, so a deck's heights
# dress the cover it belongs to. ICON (mo_nwp_phy_init.f90, "initialization of
# k800 and k400, based on reference pressure"; mo_nwp_diagnosis.f90, the
# generalized-overlap branch) splits each column at two level indices found
# from its REFERENCE pressure — a fixed function of height alone
# (mo_vertical_grid.f90), so HHL gives it and no pressure field is needed:
#   p_ref(z) at each full level's centre z (the mean of its two half levels)
#   p0   = p_ref of the LOWEST full level
#   k800 = the highest level with p_ref / p0 >= 800 / 1013.25
#   k400 = the highest level with p_ref / p0 >= (400 / 1013.25) * sqrt(1e5 / p0)
#   low  = levels k800 .. the lowest          (CLCL)
#   mid  = levels k400 .. k800 - 1            (CLCM)
#   high = every level above k400             (CLCH; not carried here)
# Over the sea that puts the low/mid boundary about 1.9 km above the ground
# and the top of the mid deck about 7.3 km above it; both come down over high
# ground. Checked against DWD's own CLCL and CLCM on all 2,949,120 cells of
# run 2026-11-16 18Z +5 h: one cell disagreed for each; with the boundary one
# level higher or lower, 14,052 to 123,845 did.
#
# THE FIVE BYTES of a node at one step (N_BYTES):
#   0 low base, 1 low top, 2 mid base, 3 mid top — 50 m steps above SEA LEVEL
#     (HEIGHT_STEP_M): 0..252 a height (byte × 50 m), 255 (NO_LEVEL) no level
#     of that deck has cover at or over the threshold, 253 (UNKNOWN) the deck
#     was not read (a level's cover or a height was missing, or an hour no
#     build read), 254 is kept free and never written. base = the bottom of
#     the deck's lowest cloudy level, top = the top of its highest.
#   4 flags — bit 0 (FLAG_TOP_AT_REACH): the mid top is the top of the HIGHEST
#     level read, so the cloud may go on above what the feed read.
#
# ICH1, little-endian (the clue file's layout, pipeline/cluefile.py, with its
# own magic — a reader of one never takes the other for its own):
#   18-byte header  magic "ICH1", swLat i16, swLon i16, spacing u16 (milli-
#                   degrees), cols u16, rows u16, steps u16, bytes u16 (= 5)
#   steps × u32     the steps' times as unix HOURS, ascending
#   zlib stream     of rows·cols·steps·5 u8: node by node (row 0 the SOUTH
#                   edge, column 0 the WEST edge), each node's steps in time
#                   order, each step's five bytes.
import struct
import zlib
import numpy as np

# A model level counts as cloud at this cover (%) or more.
CLC_THRESHOLD_PCT = 5.0
# How far above the ground the levels are read, and the share of the feed's
# nodes the level set must reach it at (tools/measure_heights.py derives the
# set from HHL; dwd.HEIGHT_MLEVELS is what it found).
REACH_M = 5000.0
REACH_SHARE = 0.99

HEIGHT_STEP_M = 50.0
MAX_HEIGHT_BYTE = 252            # 12,600 m
UNKNOWN = 253                    # the deck was not read
NO_LEVEL = 255                   # read, and no level of the deck is cloudy
LOW_BASE, LOW_TOP, MID_BASE, MID_TOP, FLAGS = range(5)
N_HEIGHT_BYTES = 4
N_BYTES = 5
FLAG_TOP_AT_REACH = 0x01

# ICON's reference ("background") atmosphere, mo_vertical_grid.f90 and
# mo_physical_constants.f90, and the two pressure ratios of mo_nwp_phy_init.f90.
P0SL_BG = 101325.0               # Pa, sea-level pressure
T0SL_BG = 288.15                 # K, sea-level temperature
DEL_T_BG = 75.0                  # K, sea level minus the asymptotic stratosphere
H_SCAL_BG = 10000.0              # m, scale height
GRAV = 9.80665
RD = 287.04
P0REF = 100000.0                 # Pa, the Exner function's reference pressure
PR800 = 800.0 / 1013.25
PR400 = 400.0 / 1013.25

MAGIC = b"ICH1"
FORMAT = MAGIC.decode()                       # the manifest entry's "format"
ZLIB_LEVEL = 6                                # as pipeline/cluefile.py
_HEADER = struct.Struct("<4shhHHHHH")         # magic, swLat, swLon, spacing mdeg, cols, rows, steps, bytes


# ---- the level set, from HHL (tools/measure_heights.py) ---------------------------

def reach_share(agl, reach_m):
    """The share of nodes WITH a height where it is at least reach_m."""
    agl = np.asarray(agl, dtype=float)
    ok = np.isfinite(agl)
    return float((agl[ok] >= reach_m).sum() / ok.sum()) if ok.any() else 0.0


def choose_top_level(agl_by_half_level, reach_m, share):
    """agl_by_half_level: {half level: heights above ground, m}; ICON counts
    levels from the top, so the largest is the ground. The top half level of
    the smallest set of lowest levels that is at least reach_m above ground
    at `share` of the nodes — the first such going up — or None when none of
    the given levels is. Full levels top..ground-1 then span ground to reach."""
    ground = max(agl_by_half_level)
    for lv in sorted((k for k in agl_by_half_level if k != ground), reverse=True):
        if reach_share(agl_by_half_level[lv], reach_m) >= share:
            return lv
    return None


def full_levels(top, ground):
    """The full levels between half level `top` and the ground half level."""
    return list(range(top, ground))


# ---- the model's decks ------------------------------------------------------------

def ref_pressure(z):
    """ICON's reference pressure (Pa) at height z (m above sea level)."""
    z = np.asarray(z, dtype=np.float64)
    return P0SL_BG * np.exp(-GRAV / RD * H_SCAL_BG / (T0SL_BG - DEL_T_BG)
                            * np.log((np.exp(z / H_SCAL_BG) * (T0SL_BG - DEL_T_BG) + DEL_T_BG) / T0SL_BG))


def _check(clc, hhl):
    if hhl.shape[0] != clc.shape[0] + 1 or hhl.shape[1:] != clc.shape[1:]:
        raise ValueError(f"{clc.shape[0]} levels need {clc.shape[0] + 1} half levels of the same nodes, "
                         f"got clc {clc.shape}, hhl {hhl.shape}")


def deck_masks(hhl):
    """hhl: [L+1][..] half-level heights (m above sea level) of L full levels,
    the HIGHEST first (ICON's order; level i lies between hhl[i+1] and
    hhl[i]), the last full level the model's LOWEST. → (low, mid): bool
    [L][..], the levels ICON counts into CLCL and into CLCM (the header has
    the rule). Levels above the mid deck are in neither. A node with a
    missing height has only its lowest level low; deck_unknown reports it."""
    hhl = np.asarray(hhl, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = ref_pressure((hhl[:-1] + hhl[1:]) / 2)
        p0 = p[-1]
        ratio = p / p0
        # ICON walks up from the level above the lowest and stops at the first
        # level under the 400 hPa ratio; k800 and k400 are the last levels it
        # set on the way.
        c400 = ratio[:-1] >= PR400 * np.sqrt(P0REF / p0)
        in400 = np.logical_and.accumulate(c400[::-1], axis=0)[::-1]
        in800 = in400 & (ratio[:-1] >= PR800)
    at_or_below = lambda m: np.logical_or.accumulate(m, axis=0)       # level >= the highest level set
    bottom = np.ones((1,) + hhl.shape[1:], dtype=bool)                 # no level set: k = the lowest level
    low = np.concatenate([at_or_below(in800), bottom])
    ge400 = np.concatenate([at_or_below(in400), bottom])
    return low, ge400 & ~low


def deck_bounds(clc, hhl, threshold=CLC_THRESHOLD_PCT, masks=None):
    """clc: [L][..] cover % on the L full levels of deck_masks' hhl. → float32
    [4][..]: low base, low top, mid base, mid top, m above SEA level — the
    bottom of the deck's lowest level with cover >= threshold and the top of
    its highest; NaN where the deck has no such level. A missing cover is
    not cloud here; deck_unknown says where one makes the deck unreadable.
    masks: deck_masks(hhl) when the caller already has it (the heights are
    the same at every step)."""
    clc = np.asarray(clc, dtype=np.float32)
    hhl = np.asarray(hhl, dtype=np.float32)
    _check(clc, hhl)
    with np.errstate(invalid="ignore"):
        cloudy = clc >= threshold                        # NaN compares False
    res = []
    for m in (deck_masks(hhl) if masks is None else masks):
        m = m & cloudy
        base = np.where(m, hhl[1:], np.inf).min(axis=0)
        top = np.where(m, hhl[:-1], -np.inf).max(axis=0)
        none = ~m.any(axis=0)
        res += [np.where(none, np.nan, base), np.where(none, np.nan, top)]
    return np.stack(res).astype(np.float32)


def deck_unknown(clc, hhl, masks=None):
    """bool [2][..] (low, mid): the deck cannot be read at this node — the
    cover of one of ITS levels is missing, or any height is."""
    clc = np.asarray(clc, dtype=np.float32)
    hhl = np.asarray(hhl, dtype=np.float32)
    _check(clc, hhl)
    bad_height = ~np.isfinite(hhl).all(axis=0)
    gone = ~np.isfinite(clc)
    return np.stack([(m & gone).any(axis=0) | bad_height
                     for m in (deck_masks(hhl) if masks is None else masks)])


def top_at_reach(clc, hhl, threshold=CLC_THRESHOLD_PCT, masks=None):
    """bool [..]: the highest level READ is a mid-deck level with cover >=
    threshold, so the mid top is the top of what was read."""
    clc = np.asarray(clc, dtype=np.float32)
    hhl = np.asarray(hhl, dtype=np.float32)
    _check(clc, hhl)
    with np.errstate(invalid="ignore"):
        return (deck_masks(hhl) if masks is None else masks)[1][0] & (clc[0] >= threshold)


def height_bytes(metres):
    """Heights in m above sea level → u8: floor(m / 50 + 0.5) clamped to
    0..252 (half UP, as pipeline/clues.py rounds); NaN → 255."""
    m = np.asarray(metres, dtype=np.float64)
    ok = np.isfinite(m)
    b = np.clip(np.floor(np.where(ok, m, 0.0) / HEIGHT_STEP_M + 0.5), 0, MAX_HEIGHT_BYTE)
    return np.where(ok, b, NO_LEVEL).astype(np.uint8)


def byte_metres(b):
    """A height byte → m above sea level, or None for 255 / 253 / 254."""
    b = int(b)
    return b * HEIGHT_STEP_M if 0 <= b <= MAX_HEIGHT_BYTE else None


def height_record(clc, hhl, threshold=CLC_THRESHOLD_PCT, masks=None):
    """u8 [5][..]: the five bytes of each node (the header has them)."""
    if masks is None:
        masks = deck_masks(np.asarray(hhl, dtype=np.float32))
    out = np.zeros((N_BYTES,) + np.shape(clc)[1:], dtype=np.uint8)
    out[:N_HEIGHT_BYTES] = height_bytes(deck_bounds(clc, hhl, threshold, masks))
    unknown = deck_unknown(clc, hhl, masks)
    out[:N_HEIGHT_BYTES][np.repeat(unknown, 2, axis=0)] = UNKNOWN
    out[FLAGS] = np.where(top_at_reach(clc, hhl, threshold, masks) & ~unknown[1], FLAG_TOP_AT_REACH, 0)
    return out


def unknown_record(shape):
    """u8 [5][shape]: an hour nothing was read for — every height UNKNOWN."""
    out = np.full((N_BYTES,) + tuple(shape), UNKNOWN, dtype=np.uint8)
    out[FLAGS] = 0
    return out


# ---- ICH1 -------------------------------------------------------------------------

def pack_square(cube):
    """u8 [steps][rows][cols][5] → the file's zlib stream."""
    cube = np.asarray(cube)
    if cube.dtype != np.uint8:
        raise ValueError(f"heights must be uint8, got {cube.dtype}")
    if cube.ndim != 4 or cube.shape[3] != N_BYTES:
        raise ValueError(f"heights must be [steps][rows][cols][{N_BYTES}], got {cube.shape}")
    return zlib.compress(np.ascontiguousarray(np.moveaxis(cube, 0, 2)).tobytes(), ZLIB_LEVEL)


def encode_height_square(sw_lat, sw_lon, spacing_mdeg, step_hours, cube):
    """cube: u8 [steps][rows][cols][5]. step_hours: ascending unix hours."""
    body = pack_square(cube)
    steps, rows, cols, _ = np.shape(cube)
    hours = [int(h) for h in step_hours]
    if len(hours) != steps:
        raise ValueError(f"{len(hours)} step hours for {steps} steps")
    if any(b <= a for a, b in zip(hours, hours[1:])):
        raise ValueError("step hours must be strictly ascending")
    head = _HEADER.pack(MAGIC, int(sw_lat), int(sw_lon), int(spacing_mdeg), cols, rows, steps, N_BYTES)
    return head + struct.pack(f"<{steps}I", *hours) + body


def decode_height_square(blob):
    """The inverse: {format, sw_lat, sw_lon, spacing_mdeg, cols, rows,
    step_hours, bytes, heights} with heights u8 [steps][rows][cols][5]."""
    magic, sw_lat, sw_lon, sp, cols, rows, steps, nb = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("not an ICH1 file")
    if nb != N_BYTES:
        raise ValueError(f"ICH1 with {nb} bytes a node, expected {N_BYTES}")
    off = _HEADER.size
    hours = list(struct.unpack_from(f"<{steps}I", blob, off)); off += 4 * steps
    raw = zlib.decompress(blob[off:])
    if len(raw) != rows * cols * steps * nb:
        raise ValueError(f"heights body is {len(raw)} bytes, expected {rows * cols * steps * nb}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, steps, nb)
    return {"format": FORMAT, "sw_lat": sw_lat, "sw_lon": sw_lon, "spacing_mdeg": sp, "cols": cols, "rows": rows,
            "step_hours": hours, "bytes": nb, "heights": np.moveaxis(arr, 2, 0)}
