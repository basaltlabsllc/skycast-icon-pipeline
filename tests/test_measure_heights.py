# tools/measure_heights.py — the pure parts of the cloud-height measurement:
# which model levels reach 5 km above ground (from HHL), a column's deck bases
# and tops, the height byte, the square's packing, and DWD's listing sizes.
# Nothing here touches the network.
import importlib.util
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("measure_heights", ROOT / "tools/measure_heights.py")
mh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mh)

BANDS = {"low": (300.0, 2500.0), "mid": (2500.0, 5000.0)}


# ---- the stand-in constants ---------------------------------------------------

def test_the_stand_in_bands_are_the_clue_bands_capped_at_the_reach():
    from pipeline import clues
    assert mh.CLC_THRESHOLD_PCT == 5.0
    assert mh.REACH_M == 5000.0
    assert mh.STANDIN_DECK_BANDS_M == {
        "low": (clues.DECK_BANDS_KM["low"][0] * 1000, clues.DECK_BANDS_KM["low"][1] * 1000),
        "mid": (clues.DECK_BANDS_KM["mid"][0] * 1000, 5000.0)}


# ---- the level set, from HHL ----------------------------------------------------

def _agl(tops):
    """{half level: [nodes] m above ground}; the largest level is the ground."""
    return {lv: np.asarray(v, dtype=float) for lv, v in tops.items()}


def test_level_set_is_the_fewest_lowest_levels_whose_top_reaches():
    # 100 nodes: half level 7 is 5 km up at 98 of them, half level 6 at all.
    h7 = np.full(100, 5200.0); h7[:2] = 4000.0
    agl = _agl({10: np.zeros(100), 9: np.full(100, 1000.0), 8: np.full(100, 3000.0),
                7: h7, 6: np.full(100, 6500.0), 5: np.full(100, 8000.0)})
    assert mh.choose_top_level(agl, 5000.0, 0.99) == 6        # full levels 6..9
    assert mh.choose_top_level(agl, 5000.0, 0.98) == 7        # 98 % is enough: one level fewer
    assert mh.full_levels(6, 10) == [6, 7, 8, 9]


def test_level_set_is_none_when_no_given_level_reaches():
    agl = _agl({3: np.zeros(4), 2: np.full(4, 900.0), 1: np.full(4, 4000.0)})
    assert mh.choose_top_level(agl, 5000.0, 0.99) is None


def test_level_set_ignores_nodes_without_a_height():
    h = np.array([6000.0, 6000.0, np.nan, np.nan])
    agl = _agl({2: np.array([0.0, 0.0, np.nan, np.nan]), 1: h})
    assert mh.choose_top_level(agl, 5000.0, 0.99) == 1


# ---- bases and tops from one column ---------------------------------------------

# Five full levels over ground at 1,000 m MSL; half-level heights above ground:
#   level 1: 5,400-4,600 (centre 5,000: above the reach, in no deck)
#   level 2: 4,600-3,000 (centre 3,800: mid)
#   level 3: 3,000-2,200 (centre 2,600: mid)
#   level 4: 2,200-  400 (centre 1,300: low)
#   level 5:   400-    0 (centre   200: below the low band)
GROUND = 1000.0
HHL = GROUND + np.array([5400.0, 4600.0, 3000.0, 2200.0, 400.0, 0.0])


def _bounds(clc):
    clc = np.asarray(clc, dtype=float)[:, None]
    b = mh.deck_bounds(clc, HHL[:, None], np.array([GROUND]), 5.0, BANDS)
    assert b.shape == (4, 1)
    return [None if np.isnan(x) else float(x) for x in b[:, 0]]


def test_no_level_over_the_threshold_is_no_deck():
    assert _bounds([0, 4.9, 0, 4.9, 0]) == [None, None, None, None]


def test_the_threshold_itself_counts():
    assert _bounds([0, 0, 0, 5.0, 0]) == [GROUND + 400, GROUND + 2200, None, None]


def test_cloud_only_in_the_mid_band():
    assert _bounds([0, 60, 0, 0, 0]) == [None, None, GROUND + 3000, GROUND + 4600]


def test_a_deck_runs_from_its_lowest_cloudy_level_to_its_highest():
    assert _bounds([0, 80, 30, 50, 0]) == [GROUND + 400, GROUND + 2200, GROUND + 2200, GROUND + 4600]


def test_top_above_the_reach_stops_at_the_highest_level_read():
    # Cloud on level 1 too (centre at the 5 km reach): it is in no deck, so the
    # mid top is level 2's own top, never level 1's.
    assert _bounds([90, 90, 0, 0, 0]) == [None, None, GROUND + 3000, GROUND + 4600]


def test_cloud_below_the_low_band_is_in_no_deck():
    assert _bounds([0, 0, 0, 0, 100]) == [None, None, None, None]


def test_a_missing_cover_is_not_cloud():
    assert _bounds([np.nan, np.nan, 40, np.nan, np.nan]) == [None, None, GROUND + 2200, GROUND + 3000]


def test_bounds_are_per_node():
    clc = np.zeros((5, 2, 3))
    clc[3, 0, 0] = 50                     # low cloud at one node only
    clc[1, 1, 2] = 50                     # mid cloud at another
    hhl = np.broadcast_to(HHL[:, None, None], (6, 2, 3))
    b = mh.deck_bounds(clc, hhl, np.full((2, 3), GROUND), 5.0, BANDS)
    assert b.shape == (4, 2, 3)
    assert b[0, 0, 0] == GROUND + 400 and b[1, 0, 0] == GROUND + 2200
    assert b[2, 1, 2] == GROUND + 3000 and b[3, 1, 2] == GROUND + 4600
    assert np.isnan(b).sum() == 4 * 6 - 4


def test_levels_and_heights_must_agree():
    with pytest.raises(ValueError):
        mh.deck_bounds(np.zeros((5, 1)), HHL[:5, None], np.array([GROUND]), 5.0, BANDS)


# ---- the byte ---------------------------------------------------------------------

def test_height_byte_is_50_m_steps_above_sea_level_half_up():
    b = mh.height_bytes(np.array([0.0, 24.9, 25.0, 50.0, 1400.0, 1425.0, 12700.0]))
    assert b.dtype == np.uint8
    assert b.tolist() == [0, 0, 1, 1, 28, 29, 254]


def test_height_byte_clips_and_keeps_255_for_none():
    assert mh.height_bytes(np.array([-86.0, 12725.0, 20000.0, np.nan])).tolist() == [0, 254, 254, 255]
    assert mh.NO_LEVEL == 255


# ---- the square, packed as the clue file packs --------------------------------------

def test_a_square_is_packed_node_by_node_each_nodes_steps_together():
    from pipeline import cluefile
    cube = np.arange(3 * 2 * 2 * 4, dtype=np.uint8).reshape(3, 2, 2, 4)     # [steps][rows][cols][4]
    blob = mh.pack_square(cube)
    raw = zlib.decompress(blob)
    assert raw == np.ascontiguousarray(np.moveaxis(cube, 0, 2)).tobytes()
    assert blob == zlib.compress(raw, cluefile.ZLIB_LEVEL)
    with pytest.raises(ValueError):
        mh.pack_square(cube.astype(np.uint16))
    with pytest.raises(ValueError):
        mh.pack_square(cube[..., :3])


# ---- DWD's listing ----------------------------------------------------------------

# Lines as opendata.dwd.de sends them (2026-11-17): the visible name cut short,
# the size last, and a CR LF line end.
LISTING = (
    '<a href="../">../</a>\r\n'
    '<a href="icon_global_icosahedral_time-invariant_2026111706_100_HHL.grib2.bz2">icon_global_icosahedral_'
    'time-invariant_20261117..&gt;</a> 17-Nov-2026 08:39:33             1206375\r\n'
    '<a href="icon_global_icosahedral_model-level_2026111706_000_120_CLC.grib2.bz2">icon_global_icosahedral_'
    'model-level_2026111706_..&gt;</a> 17-Nov-2026 08:40:06              446109\r\n'
    '</pre><hr></body>\r\n')


def test_listing_sizes_are_read_by_file_name():
    assert mh.listed_sizes(LISTING) == {
        "icon_global_icosahedral_time-invariant_2026111706_100_HHL.grib2.bz2": 1206375,
        "icon_global_icosahedral_model-level_2026111706_000_120_CLC.grib2.bz2": 446109}
    assert mh.listed_sizes("") == {} and mh.listed_sizes(None) == {}


def test_listed_levels_are_the_levels_a_listing_names():
    assert mh.listed_levels(LISTING, "HHL") == [100]
    assert mh.listed_levels(LISTING, "CLC") == [120]
