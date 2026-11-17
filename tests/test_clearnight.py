import calendar
import datetime as dt
import numpy as np
import pytest
from pipeline.clearnight import (sun_altitude_deg, night_stats, tile_keys, tile_name, slice_tile, encode_tile,
                                 decode_tile, nights_for, build_clearnight, TILE_SIDE, CLEAR_MAX, MISSING)
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
    layers = np.random.default_rng(1).integers(0, 256, (4, TILE_SIDE, TILE_SIDE), dtype=np.uint8)
    blob = encode_tile(40, -140, "2026-11-17", layers)
    d = decode_tile(blob)
    assert (d["sw_lat"], d["sw_lon"], d["night"], d["clear_max"]) == (40, -140, "2026-11-17", CLEAR_MAX)
    assert np.array_equal(d["layers"], layers)


def test_nights_for_spans_the_hours():
    hours = list(range(int(uh(2026, 11, 17, 6)), int(uh(2026, 11, 19, 6))))
    assert nights_for(hours) == ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"]


def test_build_clearnight_writes_listed_nights(tmp_path):
    hours = list(range(int(uh(2026, 11, 17, 12)), int(uh(2026, 11, 19, 0)) + 1))
    world = np.full((NY, NX), 30, dtype=np.uint8)
    entry = build_clearnight(tmp_path, "2026111712", hours, lambda i: world)
    assert entry["format"] == "CLN1" and entry["tiles"] == "clearnight/2026111712"
    assert "2026-11-17" in entry["nights"]
    # Flat, <night>_<key>.cln, so pipeline/pack.py (which packs the files
    # directly in a folder) puts every night in the one pack.
    run_dir = tmp_path / "clearnight" / "2026111712"
    assert not [p for p in run_dir.iterdir() if p.is_dir()]
    f = run_dir / "2026-11-17_N30W140.cln"
    d = decode_tile(f.read_bytes())
    # Bend sits at row (44.058-30)/0.125 ≈ 112, col (−121.315+140)/0.125 ≈ 149.
    mean, mx, clear, dark = d["layers"][:, 112, 149]
    assert (mean, mx, clear) == (30, 30, 0) and dark >= 10
    assert entry["bytes"]["2026-11-17"] > 0
