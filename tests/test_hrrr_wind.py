import json
from datetime import datetime, timezone

import numpy as np
import pytest

import hrrr.build as hb
import hrrr.grib as hg
from hrrr.grid import SQUARES
from pipeline import wind
from tests.gribfake import Files, msg

UTC = timezone.utc
RUN = datetime(2026, 11, 16, 18, tzinfo=UTC)
NOW = datetime(2026, 11, 16, 20, 40, tzinfo=UTC)


def test_rotate_identity_at_lov():
    ue, ve = hg.rotate_to_earth(np.array([3.0, -7.0]), np.array([4.0, 2.0]), np.array([-97.5, -97.5]), -97.5, 38.5)
    assert np.allclose(ue, [3.0, -7.0], atol=1e-12) and np.allclose(ve, [4.0, 2.0], atol=1e-12)


def test_rotate_east_of_lov():
    ue, ve = hg.rotate_to_earth(np.array([10.0]), np.array([0.0]), np.array([-80.0]), -97.5, 38.5)
    assert ve[0] < 0
    assert abs(np.hypot(ue[0], ve[0]) - 10.0) < 1e-9
    # the lattice turns by sin(38.5 deg) x 17.5 deg = 10.9 deg
    a = np.sin(np.radians(38.5)) * np.radians(17.5)
    assert ue[0] == pytest.approx(10 * np.cos(a)) and ve[0] == pytest.approx(-10 * np.sin(a))


def test_rotate_lon_wraps_across_the_dateline():
    # 262.5 east and -97.5 are the same meridian
    a = hg.rotate_to_earth(np.array([5.0]), np.array([1.0]), np.array([-80.0]), 262.5, 38.5)
    b = hg.rotate_to_earth(np.array([5.0]), np.array([1.0]), np.array([-80.0]), -97.5, 38.5)
    assert np.allclose(a, b)


def _idx(fh):
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = ["1:0:d=2026111618:PRES:surface:" + want + ":"]
    off = 100
    for lev in (250, 500, 850):
        for comp in ("UGRD", "VGRD"):
            rows.append(f"{len(rows) + 1}:{off}:d=2026111618:{comp}:{lev} mb:{want}:")
            off += 50 if comp == "UGRD" else 70
    rows.append(f"{len(rows) + 1}:{off}:d=2026111618:UGRD:250 mb:{want}:")   # a duplicate must not win
    return "\n".join(rows)


def test_wind_rows():
    r = hg.wind_ranges(_idx(0), 0)
    assert r[(250, "u")] == (100, 149) and r[(250, "v")] == (150, 219)
    assert r[(500, "u")] == (220, 269) and r[(850, "v")] == (390, 459)
    assert len(r) == 6
    # f001 reads "1 hour fcst" rows only, and a missing row is a KeyError
    assert hg.wind_ranges(_idx(1), 1)[(250, "u")] == (100, 149)
    with pytest.raises(KeyError):
        hg.wind_ranges(_idx(0), 1)
    with pytest.raises(KeyError):
        hg.wind_ranges("\n".join(_idx(0).split("\n")[:5]), 0)


def _wind_files(fh, order=None, val=lambda lev, comp: (3.0 if comp == "u" else -4.0)):
    """One wrfsfc file: regular lat/lon 2x2 messages named u/v at the three levels."""
    import eccodes
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = []
    for lev in (250, 500, 850):
        for comp in ("u", "v"):
            h = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
            for k, v in (("Ni", 2), ("Nj", 2), ("latitudeOfFirstGridPointInDegrees", 41.0),
                         ("latitudeOfLastGridPointInDegrees", 40.0),
                         ("longitudeOfFirstGridPointInDegrees", 250.0),
                         ("longitudeOfLastGridPointInDegrees", 251.0),
                         ("iDirectionIncrementInDegrees", 1.0), ("jDirectionIncrementInDegrees", 1.0),
                         ("bitsPerValue", 24), ("discipline", 0), ("parameterCategory", 2),
                         ("parameterNumber", 2 if comp == "u" else 3), ("typeOfFirstFixedSurface", 100),
                         ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", lev * 100)):
                eccodes.codes_set(h, k, v)
            eccodes.codes_set_values(h, np.full(4, float(val(lev, comp))))
            b = eccodes.codes_get_message(h)
            eccodes.codes_release(h)
            rows.append(("UGRD" if comp == "u" else "VGRD", f"{lev} mb", want, b))
    if order:
        rows = [rows[i] for i in order]
    f = Files()
    f.add(hg.file_url(RUN, fh), rows)
    return f


def test_read_wind_decodes_all_six_on_a_latlon_grid_unrotated():
    f = _wind_files(0)
    d = hg.read_wind(RUN, 0, f.get)
    assert set(k for k in d if isinstance(k, tuple)) == {(l, c) for l in (850, 500, 250) for c in "uv"}
    assert np.allclose(d[(500, "u")], 3.0) and np.allclose(d[(500, "v")], -4.0)
    assert np.allclose(d["lon"], [-110.0, -109.0, -110.0, -109.0]) and np.allclose(d["lat"], [41, 41, 40, 40])


def test_read_wind_refuses_a_message_that_is_not_the_one_the_index_named():
    f = _wind_files(0)
    url = hg.file_url(RUN, 0) + ".idx"
    f.idx[url] = f.idx[url].replace(b"UGRD:250 mb", b"TMPX").replace(b"VGRD:250 mb", b"UGRD:250 mb").replace(b"TMPX", b"VGRD:250 mb")
    with pytest.raises(ValueError, match="wind: expected"):
        hg.read_wind(RUN, 0, f.get)


def test_hrrr_wind_failure_drops_key(tmp_path):
    def get(url, rng=None):
        raise OSError("boom")

    assert hb.build_wind(RUN, tmp_path, get, lambda s: None, [0, 1], 2) is None
    assert not (tmp_path / "wind").exists()


def test_build_wind_writes_groups_and_entry(tmp_path):
    files = {fh: _wind_files(fh) for fh in (0, 1)}

    def get(url, rng=None):
        for f in files.values():
            if url in f.blobs or url in f.idx:
                return f.get(url, rng)
        raise KeyError(url)

    entry = hb.build_wind(RUN, tmp_path, get, lambda s: None, [0, 1], 2)
    assert entry == wind.manifest_entry(RUN)
    base = int(RUN.timestamp() // 3600)
    got = wind.decode_icw1((tmp_path / "wind/2026111618/N30W120.icw").read_bytes())
    assert got["step_hours"] == [base, base + 1]
    # node (41, -110) holds 3, -4 at every level
    assert got["cube"][0, 0, 0, 11, 10] == 3 and got["cube"][1, 2, 1, 11, 10] == -4
