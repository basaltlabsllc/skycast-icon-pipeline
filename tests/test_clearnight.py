import calendar
import datetime as dt
import numpy as np
import pytest
from pipeline.clearnight import (sun_altitude_deg, night_stats, tile_keys, tile_name, slice_tile, encode_tile,
                                 decode_tile, nights_for, build_clearnight, TILE_SIDE, CLEAR_MAX, MISSING,
                                 night_layers, clear_hours, LAYERS, MASK_BYTES)
from pipeline.squares import NX, NY

BEND = (44.058, -121.315)


def uh(y, mo, d, h, mi=0, s=0):
    """Unix hours (float) of a UTC instant."""
    return calendar.timegm((y, mo, d, h, mi, s)) / 3600.0


def test_sun_altitude_bend_astro_dusk_and_dawn():
    # src/sunsettime.js sunCrossingUtcMs at -18 deg, measured 2026-11-17.
    assert sun_altitude_deg(*BEND, uh(2026, 11, 18, 2, 18, 11)) == pytest.approx(-18, abs=0.3)
    assert sun_altitude_deg(*BEND, uh(2026, 11, 18, 13, 23, 28)) == pytest.approx(-18, abs=0.3)


def bend_hours(end_h=(2026, 11, 19, 0), step=1):
    a, b = int(uh(2026, 11, 17, 12)), int(uh(*end_h))
    return list(range(a, b + 1, step))


def one_node(hours, cover_of):
    cov = np.array([[[cover_of(h)]] for h in hours], dtype=np.uint8)
    return night_stats(cov, hours, np.array([BEND[0]]), np.array([BEND[1]]), "2026-11-17")[:, 0, 0]


def test_night_stats_counts_dark_hours_and_clear():
    six = int(uh(2026, 11, 18, 6))
    out = one_node(bend_hours(), lambda h: 10 if h <= six else 90)
    mean, mx, clear, dark = (int(v) for v in out)
    assert dark == 11                       # 03Z..13Z
    assert clear == 4                       # 03, 04, 05, 06Z
    assert mx == 90
    assert mean == round((4 * 10 + 7 * 90) / 11)


def one_node_hours(hours, cover_of):
    cov = np.array([[[cover_of(h)]] for h in hours], dtype=np.uint8)
    layers, anchor = night_layers(cov, hours, np.array([BEND[0]]), np.array([BEND[1]]), "2026-11-17")
    return layers, anchor


def test_night_layers_mark_which_dark_hours_are_clear():
    # The same night as above: dark 03Z..13Z, clear (cover <= CLEAR_MAX) at 03, 04, 05, 06Z and 10Z.
    six, ten = int(uh(2026, 11, 18, 6)), int(uh(2026, 11, 18, 10))
    layers, anchor = one_node_hours(bend_hours(), lambda h: 10 if h <= six or h == ten else 90)
    assert layers.shape == (LAYERS, 1, 1) and LAYERS == 4 + MASK_BYTES
    assert clear_hours(layers, anchor, 0, 0) == [int(uh(2026, 11, 18, k)) for k in (3, 4, 5, 6, 10)]
    # The first four layers are night_stats' exactly.
    assert list(layers[:4, 0, 0]) == list(one_node(bend_hours(), lambda h: 10 if h <= six or h == ten else 90))


def test_night_layers_threshold_is_clear_max_and_only_dark_hours_count():
    # Cover exactly CLEAR_MAX is clear, one more is not; daylight hours are never marked, however clear.
    seven = int(uh(2026, 11, 18, 7))
    layers, anchor = one_node_hours(bend_hours(), lambda h: CLEAR_MAX if h == seven else (0 if h < int(uh(2026, 11, 18, 2)) else CLEAR_MAX + 1))
    assert clear_hours(layers, anchor, 0, 0) == [seven]


def test_night_layers_no_mask_where_no_data():
    seven = int(uh(2026, 11, 18, 7))
    layers, anchor = one_node_hours(bend_hours(), lambda h: MISSING if h == seven else 0)
    assert clear_hours(layers, anchor, 0, 0) == []
    assert list(layers[:4, 0, 0]) == [MISSING] * 4


def test_night_layers_anchor_covers_a_whole_tile():
    # A tile's 161 columns span 20 deg of longitude: every window hour fits in the 32-bit mask.
    hours = list(range(int(uh(2026, 11, 17, 0)), int(uh(2026, 11, 19, 12))))
    lats, lons = np.array([44.0, 49.9]), np.linspace(-140, -120, TILE_SIDE)
    cov = np.zeros((len(hours), 2, TILE_SIDE), dtype=np.uint8)
    layers, anchor = night_layers(cov, hours, lats, lons, "2026-11-17")
    for c in (0, TILE_SIDE // 2, TILE_SIDE - 1):
        hs = clear_hours(layers, anchor, 0, c)
        assert hs and len(hs) == int(layers[3, 0, c])       # all clear: every dark hour is listed
        assert all(anchor <= h < anchor + 8 * MASK_BYTES for h in hs)


def test_night_stats_no_data_when_hours_missing():
    out = one_node(bend_hours(end_h=(2026, 11, 18, 8)), lambda h: 0)
    assert list(out) == [MISSING] * 4


def test_night_stats_missing_cover_value_is_no_data():
    seven = int(uh(2026, 11, 18, 7))
    out = one_node(bend_hours(), lambda h: MISSING if h == seven else 0)
    assert list(out) == [MISSING] * 4


def test_night_stats_no_darkness_at_60n_midsummer():
    hours = list(range(int(uh(2026, 6, 21, 0)), int(uh(2026, 6, 23, 0))))
    cov = np.zeros((len(hours), 1, 1), dtype=np.uint8)
    out = night_stats(cov, hours, np.array([60.0]), np.array([10.0]), "2026-06-21")[:, 0, 0]
    assert list(out) == [MISSING, MISSING, MISSING, 0]


def test_night_stats_interpolates_3_hourly():
    hours = bend_hours(step=3)
    h0 = hours[0]
    out = one_node(hours, lambda h: min(100, (h - h0) * 2))
    # Dark hours 03Z..13Z Nov 18: cover at h = 2*(h - h0), interpolation exact on a line.
    dark = [int(uh(2026, 11, 18, k)) for k in range(3, 14)]
    want = [2 * (h - h0) for h in dark]
    assert int(out[0]) == round(sum(want) / len(want))
    assert int(out[1]) == max(want)
    assert int(out[2]) == sum(1 for v in want if v <= CLEAR_MAX)


def test_tile_keys_and_names():
    keys = tile_keys()
    assert len(keys) == 162 and len(set(keys)) == 162
    assert keys[0] == "S90W180" and "N70E160" in keys and "N30W140" in keys and "N40W140" not in keys
    assert tile_name(-10, 0) == "S10E000"


def test_slice_tile_is_161_and_wraps_the_dateline():
    world = np.tile(np.arange(NX, dtype=np.int32) % 251, (NY, 1)).astype(np.uint8)
    t = slice_tile(world, 70, 160)
    assert t.shape == (TILE_SIDE, TILE_SIDE)
    # The east edge (180 E) is column 0 again, like squares.col_indices.
    assert t[0, -1] == world[0, 0]
    assert slice_tile(world, -90, -180)[0, 0] == world[0, 0]


def test_build_clearnight_skips_nights_before_first_night(tmp_path):
    hours = list(range(int(uh(2026, 11, 15, 0)), int(uh(2026, 11, 19, 0)) + 1))
    world = np.full((NY, NX), 30, dtype=np.uint8)
    entry = build_clearnight(tmp_path, "2026111712", hours, lambda i: world, keys=["N30W140"],
                             first_night="2026-11-16")
    assert entry["nights"][0] == "2026-11-16"
    assert not (tmp_path / "clearnight" / "2026111712" / "2026-11-15").exists()


def test_tiles_for_squares_are_the_covering_20_degree_tiles():
    from pipeline.clearnight import tiles_for_squares
    assert tiles_for_squares(["N40W125", "N45W120", "N25W100"]) == ["N10W100", "N30W140", "N30W120"]


def test_tile_roundtrip():
    layers = np.random.default_rng(1).integers(0, 256, (LAYERS, TILE_SIDE, TILE_SIDE), dtype=np.uint8)
    blob = encode_tile(40, -140, "2026-11-17", layers, anchor=493000)
    assert blob[:4] == b"CLN2"
    d = decode_tile(blob)
    assert (d["sw_lat"], d["sw_lon"], d["night"], d["clear_max"], d["anchor"]) == (40, -140, "2026-11-17", CLEAR_MAX, 493000)
    assert np.array_equal(d["layers"], layers)


def test_decode_tile_still_reads_cln1():
    import struct, zlib
    layers = np.random.default_rng(2).integers(0, 256, (4, TILE_SIDE, TILE_SIDE), dtype=np.uint8)
    blob = struct.pack("<4shhHBB", b"CLN1", 40, -140, TILE_SIDE, 4, CLEAR_MAX) + b"2026-11-17" + zlib.compress(layers.tobytes())
    d = decode_tile(blob)
    assert d["anchor"] is None and np.array_equal(d["layers"], layers)


def test_nights_for_spans_the_hours():
    hours = list(range(int(uh(2026, 11, 17, 6)), int(uh(2026, 11, 19, 6))))
    assert nights_for(hours) == ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"]


def test_build_clearnight_writes_listed_nights(tmp_path):
    hours = list(range(int(uh(2026, 11, 17, 12)), int(uh(2026, 11, 19, 0)) + 1))
    world = np.full((NY, NX), 30, dtype=np.uint8)
    entry = build_clearnight(tmp_path, "2026111712", hours, lambda i: world)
    assert entry["format"] == "CLN2" and entry["tiles"] == "clearnight/2026111712"
    assert "2026-11-17" in entry["nights"]
    # Flat, <night>_<key>.cln, so pipeline/pack.py (which packs the files
    # directly in a folder) puts every night in the one pack.
    run_dir = tmp_path / "clearnight" / "2026111712"
    assert not [p for p in run_dir.iterdir() if p.is_dir()]
    f = run_dir / "2026-11-17_N30W140.cln"
    d = decode_tile(f.read_bytes())
    # Bend sits at row (44.058-30)/0.125 ≈ 112, col (−121.315+140)/0.125 ≈ 149.
    mean, mx, clear, dark = d["layers"][:4, 112, 149]
    assert (mean, mx, clear) == (30, 30, 0) and dark >= 10
    assert clear_hours(d["layers"], d["anchor"], 112, 149) == []      # 30 % is never clear
    assert entry["bytes"]["2026-11-17"] > 0
