# pipeline/heights.py — cloud heights per deck from ICON's cover on model
# levels: which levels are the model's own low and mid deck (its CLCL / CLCM
# boundary), each deck's base and top, the five bytes of a node-step, and the
# ICH1 file. Nothing here touches the network.
import struct
import zlib

import numpy as np
import pytest

from pipeline import cluefile, heights as H


# ---- the constants the app decodes with -------------------------------------------

def test_the_named_constants():
    assert H.CLC_THRESHOLD_PCT == 5.0
    assert H.HEIGHT_STEP_M == 50.0
    assert (H.NO_LEVEL, H.UNKNOWN, H.MAX_HEIGHT_BYTE) == (255, 253, 252)
    assert H.N_BYTES == 5 and H.FLAG_TOP_AT_REACH == 1
    assert (H.LOW_BASE, H.LOW_TOP, H.MID_BASE, H.MID_TOP, H.FLAGS) == (0, 1, 2, 3, 4)
    assert H.MAGIC == b"ICH1" and H.FORMAT == "ICH1"


# ---- ICON's reference atmosphere and its deck boundary ----------------------------

def test_reference_pressure_is_icons_background_atmosphere():
    # mo_vertical_grid.f90: p0sl_bg 101325 Pa, t0sl_bg 288.15 K, del_t_bg 75 K,
    # h_scal_bg 10 km. Values computed from that formula by hand (2026-11-17).
    assert H.ref_pressure(0.0) == pytest.approx(101325.0)
    assert H.ref_pressure(1000.0) == pytest.approx(89859.66, abs=0.01)
    assert H.ref_pressure(5000.0) == pytest.approx(54059.63, abs=0.01)


# Eight full levels over ground at sea level; half-level heights, the highest
# first. The lowest level's centre is 10 m, where the reference pressure is
# 101,204.9 Pa; relative to it 800/1013.25 falls at 1,954.9 m and
# (400/1013.25)·sqrt(1e5/p0) at 7,290.7 m (computed from the formula):
#   level 0: 8,000-7,000 (centre 7,500: above the mid deck, HIGH)
#   level 1: 7,000-5,000 (centre 6,000: mid)
#   level 2: 5,000-3,000 (centre 4,000: mid)
#   level 3: 3,000-2,100 (centre 2,550: mid)
#   level 4: 2,100-1,800 (centre 1,950: low, 5 m under the boundary)
#   level 5: 1,800-1,000 (low)   level 6: 1,000-20 (low)   level 7: 20-0 (low)
HHL = np.array([8000.0, 7000.0, 5000.0, 3000.0, 2100.0, 1800.0, 1000.0, 20.0, 0.0])
LOW = [False, False, False, False, True, True, True, True]
MID = [False, True, True, True, False, False, False, False]


def _masks(hhl):
    low, mid = H.deck_masks(np.asarray(hhl, dtype=float)[:, None])
    return low[:, 0].tolist(), mid[:, 0].tolist()


def test_decks_follow_the_models_800_and_400_hpa_reference_levels():
    assert _masks(HHL) == (LOW, MID)


def test_the_boundary_is_relative_to_the_lowest_levels_pressure_not_a_height_above_ground():
    # The same column on ground at 1,500 m: the lowest centre is 1,510 m, and
    # 800/1013.25 of ITS reference pressure falls 1,880.6 m above it, not
    # 1,944.9 m: level 4 (centre 1,950 m above ground) is now in the MID deck.
    low, mid = _masks(HHL + 1500.0)
    assert low == [False, False, False, False, False, True, True, True]
    assert mid == [False, True, True, True, True, False, False, False]


def test_the_lowest_level_is_always_low_and_a_missing_height_is_no_deck_above_it():
    low, mid = _masks([2000.0, 1000.0, np.nan, 0.0])
    assert low[-1] is True and not any(mid)


def test_masks_are_per_node():
    hhl = np.stack([HHL, HHL + 1500.0], axis=1)                # [9][2]
    low, mid = H.deck_masks(hhl)
    assert low.shape == mid.shape == (8, 2)
    assert low[4].tolist() == [True, False] and mid[4].tolist() == [False, True]


# ---- bases and tops ---------------------------------------------------------------

def _bounds(clc, hhl=HHL):
    clc = np.asarray(clc, dtype=float)[:, None]
    b = H.deck_bounds(clc, np.asarray(hhl)[:, None])
    assert b.shape == (4, 1) and b.dtype == np.float32
    return [None if np.isnan(x) else float(x) for x in b[:, 0]]


def test_no_level_over_the_threshold_is_no_deck():
    assert _bounds([0, 4.9, 0, 4.9, 0, 4.9, 0, 0]) == [None, None, None, None]


def test_the_threshold_itself_counts():
    assert _bounds([0, 0, 0, 0, 0, 5.0, 0, 0]) == [1000.0, 1800.0, None, None]


def test_a_deck_runs_from_its_lowest_cloudy_level_to_its_highest():
    # low: levels 6 and 4 cloudy (5 clear between them); mid: levels 3 and 1.
    assert _bounds([0, 80, 0, 30, 50, 0, 20, 0]) == [20.0, 2100.0, 2100.0, 7000.0]


def test_cloud_on_the_ground_level_is_low():
    assert _bounds([0, 0, 0, 0, 0, 0, 0, 100]) == [0.0, 20.0, None, None]


def test_cloud_above_the_mid_deck_is_in_neither():
    assert _bounds([90, 0, 0, 0, 0, 0, 0, 0]) == [None, None, None, None]


def test_a_lower_threshold_can_be_passed():
    assert _bounds([0, 0, 0, 0, 0, 0, 0, 0]) == [None] * 4
    b = H.deck_bounds(np.array([0, 0, 0, 2.0, 0, 0, 0, 0.0])[:, None], HHL[:, None], threshold=1.0)
    assert b[:, 0].tolist()[2:] == [2100.0, 3000.0]


def test_levels_and_heights_must_agree():
    with pytest.raises(ValueError):
        H.deck_bounds(np.zeros((8, 1)), HHL[:8, None])


# ---- "not read" is not "no cloud" -------------------------------------------------

def _unknown(clc, hhl=HHL):
    u = H.deck_unknown(np.asarray(clc, dtype=float)[:, None], np.asarray(hhl, dtype=float)[:, None])
    assert u.shape == (2, 1)
    return u[:, 0].tolist()


def test_a_missing_cover_makes_its_own_deck_unknown_only():
    nan = np.nan
    assert _unknown([0, 0, 0, 0, 0, 0, 0, 0]) == [False, False]
    assert _unknown([0, 0, 0, 0, 0, nan, 0, 0]) == [True, False]       # a low level
    assert _unknown([0, 0, nan, 0, 0, 0, 0, 0]) == [False, True]       # a mid level
    assert _unknown([nan, 0, 0, 0, 0, 0, 0, 0]) == [False, False]      # above both decks


def test_a_missing_height_makes_both_decks_unknown():
    hhl = HHL.copy()
    hhl[3] = np.nan
    assert _unknown([0] * 8, hhl) == [True, True]


# ---- the height byte --------------------------------------------------------------

def test_height_byte_is_50_m_steps_above_sea_level_half_up():
    b = H.height_bytes(np.array([0.0, 24.9, 25.0, 50.0, 1400.0, 1425.0, 12600.0]))
    assert b.dtype == np.uint8
    assert b.tolist() == [0, 0, 1, 1, 28, 29, 252]


def test_height_byte_clips_below_the_reserved_values_and_keeps_255_for_none():
    # 253 (unknown) and 254 (kept free) are never a height.
    assert H.height_bytes(np.array([-86.0, 12625.0, 12700.0, 20000.0, np.nan])).tolist() == [0, 252, 252, 252, 255]


def test_bytes_decode_back_to_metres():
    assert H.byte_metres(28) == 1400.0 and H.byte_metres(0) == 0.0 and H.byte_metres(252) == 12600.0
    assert H.byte_metres(255) is None and H.byte_metres(253) is None and H.byte_metres(254) is None


# ---- the five bytes of a node-step ------------------------------------------------

def _record(clc, hhl=HHL):
    r = H.height_record(np.asarray(clc, dtype=float)[:, None], np.asarray(hhl, dtype=float)[:, None])
    assert r.shape == (5, 1) and r.dtype == np.uint8
    return r[:, 0].tolist()


def test_record_is_low_base_low_top_mid_base_mid_top_flags():
    assert _record([0, 80, 0, 30, 50, 0, 20, 0]) == [0, 42, 42, 140, 0]       # 20 m -> 0, 2,100 -> 42, 7,000 -> 140
    assert _record([0] * 8) == [255, 255, 255, 255, 0]


def test_record_marks_a_deck_with_a_missing_level_unknown_never_a_height():
    nan = np.nan
    # A cloudy low level beside a missing one: still unknown, not that level's height.
    assert _record([0, 0, 0, 30, 0, nan, 50, 0]) == [253, 253, 42, 60, 0]
    assert _record([0, nan, 0, 0, 0, 0, 50, 0]) == [0, 20, 253, 253, 0]
    hhl = HHL.copy()
    hhl[0] = np.nan
    assert _record([50] * 8, hhl) == [253, 253, 253, 253, 0]
    assert H.unknown_record((2, 3)).shape == (5, 2, 3)
    assert H.unknown_record((1, 1))[:, 0, 0].tolist() == [253, 253, 253, 253, 0]


def test_top_at_reach_flag_is_set_when_the_highest_level_read_is_a_cloudy_mid_level():
    # Without level 0 the highest level read (7,000-5,000 m) is in the mid deck.
    hhl = HHL[1:]
    assert _record([80, 0, 0, 0, 0, 0, 0], hhl) == [255, 255, 100, 140, 1]    # cloudy to the top of what was read
    assert _record([0, 80, 0, 0, 0, 0, 0], hhl) == [255, 255, 60, 100, 0]     # the top is inside
    assert _record([4.9, 80, 0, 0, 0, 0, 0], hhl) == [255, 255, 60, 100, 0]   # under the threshold
    assert _record([np.nan, 80, 0, 0, 0, 0, 0], hhl) == [255, 255, 253, 253, 0]


def test_top_at_reach_flag_is_not_set_when_the_highest_level_read_is_above_the_mid_deck():
    # With level 0 (a HIGH level) read and cloudy, the mid deck's top is its own.
    assert _record([90, 80, 0, 0, 0, 0, 0, 0]) == [255, 255, 100, 140, 0]


def test_records_are_per_node():
    clc = np.zeros((8, 2, 3))
    clc[5, 0, 0] = 50
    clc[2, 1, 2] = 50
    hhl = np.broadcast_to(HHL[:, None, None], (9, 2, 3))
    r = H.height_record(clc, hhl)
    assert r.shape == (5, 2, 3)
    assert r[:, 0, 0].tolist() == [20, 36, 255, 255, 0]
    assert r[:, 1, 2].tolist() == [255, 255, 60, 100, 0]
    assert (r[:4] == 255).sum() == 4 * 6 - 4


# ---- the level set, from HHL (the measurement's rule) -----------------------------

def _agl(tops):
    return {lv: np.asarray(v, dtype=float) for lv, v in tops.items()}


def test_level_set_is_the_fewest_lowest_levels_whose_top_reaches():
    h7 = np.full(100, 5200.0); h7[:2] = 4000.0
    agl = _agl({10: np.zeros(100), 9: np.full(100, 1000.0), 8: np.full(100, 3000.0),
                7: h7, 6: np.full(100, 6500.0), 5: np.full(100, 8000.0)})
    assert H.choose_top_level(agl, 5000.0, 0.99) == 6
    assert H.choose_top_level(agl, 5000.0, 0.98) == 7
    assert H.full_levels(6, 10) == [6, 7, 8, 9]


def test_level_set_is_none_when_no_given_level_reaches():
    agl = _agl({3: np.zeros(4), 2: np.full(4, 900.0), 1: np.full(4, 4000.0)})
    assert H.choose_top_level(agl, 5000.0, 0.99) is None


def test_level_set_ignores_nodes_without_a_height():
    h = np.array([6000.0, 6000.0, np.nan, np.nan])
    agl = _agl({2: np.array([0.0, 0.0, np.nan, np.nan]), 1: h})
    assert H.choose_top_level(agl, 5000.0, 0.99) == 1


# ---- ICH1 -------------------------------------------------------------------------

def _cube(steps=3, rows=2, cols=4):
    return (np.arange(steps * rows * cols * 5) % 251).astype(np.uint8).reshape(steps, rows, cols, 5)


def test_ich1_layout_is_the_clue_files_with_its_own_magic():
    cube = _cube()
    blob = H.encode_height_square(40, -125, 250, [494000, 494001, 494003], cube)
    assert blob[:4] == b"ICH1"
    magic, sw_lat, sw_lon, sp, cols, rows, steps, nb = struct.unpack_from("<4shhHHHHH", blob, 0)
    assert (sw_lat, sw_lon, sp, cols, rows, steps, nb) == (40, -125, 250, 4, 2, 3, 5)
    assert struct.calcsize("<4shhHHHHH") == 18 == cluefile._HEADER.size
    assert list(struct.unpack_from("<3I", blob, 18)) == [494000, 494001, 494003]
    raw = zlib.decompress(blob[18 + 12:])
    # node by node (south row, west column first), each node's steps in time order, 5 bytes each
    assert raw == np.ascontiguousarray(np.moveaxis(cube, 0, 2)).tobytes()
    assert raw[:5] == cube[0, 0, 0].tobytes() and raw[5:10] == cube[1, 0, 0].tobytes()


def test_ich1_round_trip():
    cube = _cube()
    d = H.decode_height_square(H.encode_height_square(-5, 175, 250, [10, 11, 12], cube))
    assert (d["format"], d["sw_lat"], d["sw_lon"], d["spacing_mdeg"], d["cols"], d["rows"]) == ("ICH1", -5, 175, 250, 4, 2)
    assert d["step_hours"] == [10, 11, 12] and d["bytes"] == 5
    assert np.array_equal(d["heights"], cube)


def test_ich1_refuses_bad_input():
    cube = _cube()
    with pytest.raises(ValueError):
        H.encode_height_square(0, 0, 250, [1, 2, 3], cube.astype(np.uint16))
    with pytest.raises(ValueError):
        H.encode_height_square(0, 0, 250, [1, 2, 3], cube[..., :4])
    with pytest.raises(ValueError):
        H.encode_height_square(0, 0, 250, [1, 2], cube)
    with pytest.raises(ValueError):
        H.encode_height_square(0, 0, 250, [1, 3, 2], cube)
    good = H.encode_height_square(0, 0, 250, [1, 2, 3], cube)
    with pytest.raises(ValueError):
        H.decode_height_square(b"ICX2" + good[4:])
    with pytest.raises(ValueError):
        H.decode_height_square(good[:30] + zlib.compress(b"\0" * 7))


def test_a_clue_reader_refuses_a_heights_file_and_the_other_way_round():
    # The two never share a magic: an app that reads clue files cannot take a
    # heights file for one.
    blob = H.encode_height_square(0, 0, 250, [1, 2, 3], _cube())
    with pytest.raises(ValueError):
        cluefile.decode_clue_square(blob)
    assert H.MAGIC not in cluefile.MAGIC_BYTES
