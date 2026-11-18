# tools/check_heights.py — the control for the cloud-height feed: the pure
# parts (which node, which raw cell, which step, the comparison) and one whole
# report from fake DWD fields. Nothing here touches the network or eccodes.
import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pipeline import dwd, heights
from pipeline.history import hour_of

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_heights", ROOT / "tools/check_heights.py")
ch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch)

RUN = dt.datetime(2026, 11, 16, 6, tzinfo=dt.timezone.utc)
FT = 3.28084


def test_the_docstring_says_a_one_level_difference_is_expected_and_reported():
    doc = " ".join(ch.__doc__.split())
    assert "0.25" in doc and "regrid" in doc
    assert "one model level" in doc and "reported, not hidden" in doc
    assert "publishes nothing" in doc


def test_the_node_is_the_lattice_point_nearest_the_place():
    d = {"sw_lat": 45, "sw_lon": -125, "spacing_mdeg": 250, "rows": 21, "cols": 21}
    assert ch.node_of(d, 45.52, -122.68) == (2, 9, 45.5, -122.75)
    assert ch.node_of(d, 45.0, -125.0) == (0, 0, 45.0, -125.0)
    assert ch.node_of(d, 50.0, -120.0) == (20, 20, 50.0, -120.0)
    for lat, lon in ((44.8, -122.0), (50.2, -122.0), (47.0, -125.2), (47.0, -119.8)):
        with pytest.raises(ValueError, match="not in this square"):
            ch.node_of(d, lat, lon)


def test_the_raw_cell_is_the_nearest_by_distance_on_the_ground():
    clat = np.array([45.0, 45.5, 45.5, 60.0])
    clon = np.array([-122.75, -122.75, -122.60, -122.75])
    i, km = ch.nearest_cell(clat, clon, 45.5, -122.74)
    assert i == 1 and km == pytest.approx(0.01 * np.cos(np.radians(45.5)) * 111.2, rel=1e-6)
    # A degree of longitude is shorter than one of latitude at 60 N.
    i, _ = ch.nearest_cell(np.array([60.4, 60.0]), np.array([10.0, 10.7]), 60.0, 10.0)
    assert i == 1
    # ... and the date line is no distance at all.
    i, km = ch.nearest_cell(np.array([52.0, 52.0]), np.array([179.9, 0.0]), 52.0, -179.9)
    assert i == 0 and km < 20


def test_the_step_is_found_by_its_hour():
    hours = [hour_of(RUN) - 1, hour_of(RUN), hour_of(RUN) + 17]
    assert ch.step_index(hours, RUN, 17) == 2
    assert ch.step_index(hours, RUN, 0) == 1
    with pytest.raises(ValueError, match="not in the file"):
        ch.step_index(hours, RUN, 5)


def test_levels_apart_counts_model_levels_between_two_heights():
    hhl = np.array([3000.0, 2000.0, 1200.0, 600.0, 200.0, 0.0])
    assert ch.levels_apart(1200.0, 1200.0, hhl) == 0
    assert ch.levels_apart(1210.0, 600.0, hhl) == 1            # the 50 m rounding is not a level
    assert ch.levels_apart(2000.0, 600.0, hhl) == 2


# ---- one whole report -------------------------------------------------------------

LEVELS = dwd.HEIGHT_MLEVELS                                  # 86..120
HALF = dwd.HEIGHT_HHL_LEVELS                                 # 86..121
GROUND = 300.0                                               # every height a multiple of the 50 m step
THICK = 150.0
NODE_LAT, NODE_LON = 45.5, -122.75


def _hhl_col():
    """Half-level heights of one column, the highest first: 150 m levels."""
    return GROUND + THICK * (HALF[-1] - np.array(HALF, dtype=float))


def _clc_col(cloudy):
    return np.array([70.0 if lv in cloudy else 0.0 for lv in LEVELS])


def _reader(cloudy, step=17):
    """read(field, name) for a world of 3 cells; cell 1 is the node's."""
    hhl, clc = _hhl_col(), _clc_col(cloudy)
    asked = []

    def read(field, name):
        asked.append(name)
        if name == dwd.invariant_file_name(RUN, "CLAT"):
            return np.array([NODE_LAT - 1, NODE_LAT + 0.01, NODE_LAT + 1])
        if name == dwd.invariant_file_name(RUN, "CLON"):
            return np.array([NODE_LON, NODE_LON, NODE_LON])
        for k, lv in enumerate(HALF):
            if name == dwd.invariant_file_name(RUN, "HHL", lv):
                return np.array([9999.0, hhl[k], 9999.0])
        for k, lv in enumerate(LEVELS):
            if name == dwd.model_file_name(RUN, step, lv, "CLC"):
                return np.array([100.0, clc[k], 100.0])
        raise AssertionError(f"not a file the control reads: {name}")

    read.asked = asked
    return read


def _ich(cloudy, step=17):
    """A square whose every node carries the record of that column."""
    rec = heights.height_record(_clc_col(cloudy)[:, None], _hhl_col()[:, None])[:, 0]
    hours = [hour_of(RUN) + s for s in (0, step)]
    cube = np.zeros((2, 21, 21, heights.N_BYTES), np.uint8)
    cube[0] = heights.unknown_record(())                    # the other hour: not what is asked for
    cube[1] = rec
    return heights.encode_height_square(45, -125, 250, hours, cube), rec


def test_a_report_shows_the_published_bytes_the_raw_column_and_no_difference():
    blob, rec = _ich({119, 118, 100})
    read = _reader({119, 118, 100})
    text = "\n".join(ch.report(blob, RUN, 17, 45.52, -122.68, read))
    assert "N45W125" in text and "row 2, column 9" in text and "45.500, -122.750" in text
    assert f"surface {GROUND:.0f} m" in text
    hhl = _hhl_col()
    base = hhl[LEVELS.index(119) + 1]
    line = next(l for l in text.splitlines() if l.startswith("low base"))
    pub = heights.byte_metres(rec[heights.LOW_BASE])
    assert f"byte {rec[heights.LOW_BASE]:3d}" in line
    assert f"{pub:.0f} m" in line and f"{pub * FT:.0f} ft" in line and f"{(pub - GROUND) * FT:.0f} ft above ground" in line
    assert f"raw {base:.0f} m" in line and f"{(base - GROUND) * FT:.0f} ft above ground" in line
    assert f"difference {pub - base:+.0f} m" in line and "0 levels" in line
    assert "SAME" in text and "DIFFERENT" not in text
    # The raw column is printed, cloudy levels with their deck.
    assert any(l.strip().startswith("L119") and "70" in l and "low" in l for l in text.splitlines())
    assert any(l.strip().startswith("L100") and "mid" in l for l in text.splitlines())
    # It read the raw icosahedral files of that run and step, and nothing else.
    want = {dwd.invariant_file_name(RUN, "CLAT"), dwd.invariant_file_name(RUN, "CLON")}
    want |= set(dwd.height_files(RUN, [17])["HHL"]) | set(dwd.height_files(RUN, [17])["CLC"])
    assert set(read.asked) == want


def test_a_one_level_difference_is_reported_not_hidden():
    blob, _ = _ich({119, 118})                               # published: the top is level 118's
    text = "\n".join(ch.report(blob, RUN, 17, 45.52, -122.68, _reader({119, 118, 117})))
    line = next(l for l in text.splitlines() if l.startswith("low top"))
    assert "1 level" in line and f"difference {-THICK:+.0f} m" in line
    assert "DIFFERENT" in text and "by one model level" in text


def test_no_level_and_unknown_are_said_in_words():
    blob, rec = _ich(set())
    text = "\n".join(ch.report(blob, RUN, 17, 45.52, -122.68, _reader(set())))
    assert rec[:4].tolist() == [heights.NO_LEVEL] * 4
    assert text.count("no cloud level in the deck") >= 8      # published and raw, four values
    blob, _ = _ich({119})
    text = "\n".join(ch.report(blob, RUN, 0, 45.52, -122.68, _reader({119}, step=0)))
    assert "byte 253" in text and "not read" in text
    assert "cannot be compared" in text


def test_the_flag_is_printed_for_both():
    blob, rec = _ich({86})
    assert rec[heights.FLAGS] == heights.FLAG_TOP_AT_REACH
    text = "\n".join(ch.report(blob, RUN, 17, 45.52, -122.68, _reader({86})))
    line = next(l for l in text.splitlines() if l.startswith("top above reach"))
    assert "published yes" in line and "raw yes" in line


def test_the_file_comes_from_a_path_or_a_url(tmp_path):
    blob, _ = _ich({119})
    p = tmp_path / "N45W125.ich"
    p.write_bytes(blob)
    assert ch.load_file(str(p), fetch=None) == blob
    got = []
    assert ch.load_file("https://example.invalid/texdev/heights/2026111606/N45W125.ich",
                        fetch=lambda url: got.append(url) or blob) == blob
    assert got == ["https://example.invalid/texdev/heights/2026111606/N45W125.ich"]


def test_the_tool_only_reads():
    src = (ROOT / "tools/check_heights.py").read_text()
    for word in ("put(", "upload", "git push", "blob.mjs", "texdev.mjs", "requests.post", "method=\"POST\""):
        assert word not in src, word
