import json
from datetime import datetime, timezone

import numpy as np
import pytest

import hrrr.build as hb
import hrrr.grib as hg
from hrrr.grid import SQUARES
from pipeline import layerwind, wind
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


def test_read_wind_rotates_with_lov_then_latin1(monkeypatch):
    f = _wind_files(0, val=lambda lev, comp: 10.0 if comp == "u" else 0.0)
    # the fixture's cells sit at -110/-109; put LoV west of them so they are east of it
    monkeypatch.setattr(hg, "_wind_geometry", lambda g: (
        np.array([41.0, 41.0, 40.0, 40.0]), np.array([-80.0, -80.0, -80.0, -80.0]), (262.5, 38.5)))
    d = hg.read_wind(RUN, 0, f.get)
    a = np.sin(np.radians(38.5)) * np.radians(17.5)
    assert (d[(250, "v")] < 0).all()
    assert np.allclose(d[(250, "u")], 10 * np.cos(a), atol=1e-4) and np.allclose(d[(250, "v")], -10 * np.sin(a), atol=1e-4)
    assert np.allclose(np.hypot(d[(850, "u")], d[(850, "v")]), 10.0, atol=1e-4)



# ---- wind on HRRR's cloud layer ---------------------------------------------------
LEVELS = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150)
THR = layerwind.CONDENSATE_CLOUDY


def _lmsg(kind, lev, vals, surface_kind="surface"):
    """A 2 x 2 GRIB2 message on the 41..40 N, 250..251 E grid: kind u | v | clmr |
    cimixr on `lev` hPa, or 'pres' (Pa) on `surface_kind` ('surface' or 'cloudBase')."""
    import eccodes
    if kind == "pres":
        h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        spec = (("parameterCategory", 3), ("parameterNumber", 0),
                ("typeOfFirstFixedSurface", 1 if surface_kind == "surface" else 2),
                ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", 0))
    else:
        h = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
        cat, num = {"u": (2, 2), "v": (2, 3), "clmr": (1, 22), "cimixr": (1, 82)}[kind]
        spec = (("parameterCategory", cat), ("parameterNumber", num), ("typeOfFirstFixedSurface", 100),
                ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", lev * 100))
    for k, v in (("Ni", 2), ("Nj", 2), ("latitudeOfFirstGridPointInDegrees", 41.0),
                 ("latitudeOfLastGridPointInDegrees", 40.0), ("longitudeOfFirstGridPointInDegrees", 250.0),
                 ("longitudeOfLastGridPointInDegrees", 251.0), ("iDirectionIncrementInDegrees", 1.0),
                 ("jDirectionIncrementInDegrees", 1.0), ("bitsPerValue", 24), ("discipline", 0)) + spec:
        eccodes.codes_set(h, k, v)
    eccodes.codes_set_values(h, np.array(vals, dtype=float))
    b = eccodes.codes_get_message(h)
    eccodes.codes_release(h)
    return b


def _layer_files(fh, u_of=lambda lev: lev / 10.0, v_of=lambda lev: -1.0, clmr_of=lambda lev: 0.0,
                 cimixr_of=lambda lev: 0.0, ps_pa=(101300.0,) * 4, levels=LEVELS, skip=()):
    """The pressure file (u, v, a filler, CLMR, CIMIXR per level, as in NOAA's order, plus the
    25 mb levels between ours that the reader must ignore) and the surface file
    (PRES cloud base, then PRES surface). Returns a Files."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    prs = []
    for lev in sorted(set(levels) | {l + 25 for l in levels if l < 1000}, reverse=True):
        mine = lev in levels
        for kind, name in (("u", "UGRD"), ("v", "VGRD")):
            if (name, lev) not in skip:
                prs.append((name, f"{lev} mb", want, _lmsg(kind, lev, [(u_of if kind == "u" else v_of)(lev) if mine else 99.0] * 4)))
        prs.append(("TMP", f"{lev} mb", want, msg([280.0] * 4)))
        for kind, name, f in (("clmr", "CLMR", clmr_of), ("cimixr", "CIMIXR", cimixr_of)):
            if (name, lev) not in skip:
                prs.append((name, f"{lev} mb", want, _lmsg(kind, lev, [f(lev) if mine else 0.0] * 4)))
    sfc = [("PRES", "cloud base", want, _lmsg("pres", 0, [70000.0] * 4, "cloudBase"))]
    if ("PRES", "surface") not in skip:
        sfc.append(("PRES", "surface", want, _lmsg("pres", 0, list(ps_pa))))
    f = Files()
    f.add(hg.prs_url(RUN, fh), prs)
    f.add(hg.file_url(RUN, fh), sfc)
    return f


def _prs_idx(fh=6, skip=()):
    return _layer_files(fh, skip=skip).idx[hg.prs_url(RUN, fh) + ".idx"].decode()


def _sfc_idx(fh=6, skip=()):
    return _layer_files(fh, skip=skip).idx[hg.file_url(RUN, fh) + ".idx"].decode()


def test_layer_wind_levels_are_every_50_mb_1000_to_150():
    assert hg.HRRR_WIND_LEVELS == LEVELS


def test_layer_wind_ranges_cover_every_level_and_the_surface_pressure():
    f = _layer_files(6)
    prs_i, sfc_i = _prs_idx(), _sfc_idx()
    r = hg.layer_wind_ranges(prs_i, sfc_i, 6)
    assert len(r) == len(LEVELS) * 4 + 1
    for lev in LEVELS:
        assert {(lev, c) for c in ("u", "v", "clmr", "cimixr")} <= set(r)
    assert ("sfc", "pres") in r and not any(lev in r for lev in (975, 925, 825))
    # the surface PRES, not the cloud-base PRES that comes first in that file
    rows = [ln.split(":") for ln in sfc_i.strip().split("\n")]
    surf = next(i for i, x in enumerate(rows) if x[4] == "surface")
    assert r[("sfc", "pres")][0] == int(rows[surf][1])
    # rows of another forecast hour are not picked
    with pytest.raises(KeyError):
        hg.layer_wind_ranges(prs_i, sfc_i, 7)
    # f00 reads "anl"
    assert hg.layer_wind_ranges(_prs_idx(0), _sfc_idx(0), 0)[(500, "u")]


@pytest.mark.parametrize("name,lev", [("UGRD", 700), ("VGRD", 700), ("CLMR", 700), ("CIMIXR", 700),
                                      ("CIMIXR", 150), ("UGRD", 1000), ("PRES", "surface")])
def test_layer_wind_a_missing_row_raises_keyerror(name, lev):
    skip = ((name, lev),)
    with pytest.raises(KeyError, match=name):
        hg.layer_wind_ranges(_prs_idx(skip=skip), _sfc_idx(skip=skip), 6)


def _ref(u_of, cloudy, ps_hpa, v_of=lambda lev: -1.0):
    p = np.array(LEVELS, float)
    return layerwind.deck_winds(p, [np.array([u_of(l)]) for l in LEVELS], [np.array([v_of(l)]) for l in LEVELS],
                                [np.array([l in cloudy]) for l in LEVELS], np.array([ps_hpa]),
                                layerwind.deck_bounds("ncep", np.array([ps_hpa])))


def test_read_layer_wind_returns_deck_winds_over_the_cloud_layer():
    u_of = lambda lev: lev / 10.0
    cloudy = {700, 650}
    f = _layer_files(6, u_of=u_of, clmr_of=lambda lev: 2 * THR if lev in cloudy else 0.0,
                     ps_pa=(101300.0, 101300.0, 60000.0, 60000.0))
    d = hg.read_layer_wind(RUN, 6, f.get)
    assert set(k for k in d if isinstance(k, tuple)) == {(l, c) for l in (850, 500, 250) for c in "uv"}
    assert d[(850, "u")].dtype == np.float32 and d[(850, "u")].shape == (4,)
    assert np.allclose(d["lat"], [41, 41, 40, 40]) and np.allclose(d["lon"], [-110, -109, -110, -109])
    for pt, ps in ((0, 1013.0), (2, 600.0)):
        want = _ref(u_of, cloudy, ps)
        for key in (850, 500, 250):
            for c in "uv":
                got, w = d[(key, c)][pt], want[(key, c)][0]
                assert (np.isnan(got) and np.isnan(w)) or got == pytest.approx(w, rel=1e-5), (pt, key, c)
    # ps = 1013: 700 and 650 sit in the low deck (>= 642); 650 mb is its thickest run's top
    assert 65.0 <= d[(850, "u")][0] <= 70.0
    # ps = 600: the whole low deck is underground, NaN and not 0
    assert np.isnan(d[(850, "u")][2]) and np.isnan(d[(850, "v")][2])


def test_deck_winds_in_slices_equal_one_call(monkeypatch):
    f = _layer_files(6, clmr_of=lambda lev: 2 * THR if lev in (700, 400) else 0.0,
                     ps_pa=(101300.0, 95000.0, 60000.0, 70000.0))
    whole = hg.read_layer_wind(RUN, 6, f.get)
    monkeypatch.setattr(hg, "LAYER_CHUNK", 1)
    sliced = hg.read_layer_wind(RUN, 6, f.get)
    for k in whole:
        assert np.array_equal(whole[k], sliced[k], equal_nan=True), k


def test_cloudy_is_cloud_water_plus_cloud_ice_at_the_threshold(monkeypatch):
    seen = {}
    real = layerwind.deck_winds

    def spy(levels, u, v, cloudy, ps, bounds):
        seen["levels"], seen["cloudy"], seen["ps"] = list(levels), [np.asarray(c) for c in cloudy], ps
        return real(levels, u, v, cloudy, ps, bounds)

    monkeypatch.setattr(layerwind, "deck_winds", spy)
    # 600: each part alone is below the threshold, together they reach it; 500: water alone at it;
    # 400: ice alone just under; 300: both zero
    cl = {600: 0.5 * THR, 500: THR, 400: 0.0, 300: 0.0}
    ci = {600: 0.5 * THR, 500: 0.0, 400: 0.99 * THR, 300: 0.0}
    f = _layer_files(6, clmr_of=lambda lev: cl.get(lev, 0.0), cimixr_of=lambda lev: ci.get(lev, 0.0))
    hg.read_layer_wind(RUN, 6, f.get)
    on = lambda lev: bool(seen["cloudy"][seen["levels"].index(lev)][0])
    assert on(600) and on(500) and not on(400) and not on(300)
    assert np.allclose(seen["ps"], 1013.0)       # Pa -> hPa


def test_each_levels_wind_is_turned_before_the_layer_mean(monkeypatch):
    seen = {}
    real = layerwind.deck_winds

    def spy(levels, u, v, cloudy, ps, bounds):
        seen["levels"], seen["u"], seen["v"] = list(levels), u, v
        return real(levels, u, v, cloudy, ps, bounds)

    monkeypatch.setattr(layerwind, "deck_winds", spy)
    # cells at -80 with LoV 262.5 (= -97.5): a turn of sin(38.5) x 17.5 deg = 10.9 deg
    monkeypatch.setattr(hg, "_wind_geometry", lambda g: (
        np.array([41.0, 41.0, 40.0, 40.0]), np.array([-80.0] * 4), (262.5, 38.5)))
    f = _layer_files(6, u_of=lambda lev: lev / 10.0, v_of=lambda lev: lev / 100.0)
    d = hg.read_layer_wind(RUN, 6, f.get)
    a = np.sin(np.radians(38.5)) * np.radians(17.5)
    for k, lev in enumerate(seen["levels"]):
        u0, v0 = lev / 10.0, lev / 100.0
        assert np.allclose(seen["u"][k], u0 * np.cos(a) + v0 * np.sin(a), atol=1e-3), lev
        assert np.allclose(seen["v"][k], -u0 * np.sin(a) + v0 * np.cos(a), atol=1e-3), lev
    assert not np.allclose(seen["u"][0], 100.0)
    # the speed of a deck with no cloud (mean over its levels) keeps the speed of the mean, rotated once
    assert np.isfinite(d[(500, "u")]).all()


def test_wind_message_in_a_cloud_slot_is_rejected():
    for slot, other in (("clmr", "cimixr"), ("cimixr", "u"), ("clmr", "v")):
        f = _layer_files(6)
        url = hg.prs_url(RUN, 6)
        right, wrong = _lmsg(slot, 700, [0.0] * 4), _lmsg(other, 700, [0.0] * 4)
        assert len(right) == len(wrong) and f.blobs[url].count(right) == 1
        f.blobs[url] = f.blobs[url].replace(right, wrong)
        with pytest.raises(ValueError, match=f"expected {slot} 700"):
            hg.read_layer_wind(RUN, 6, f.get)


def test_a_cimixr_message_with_another_parameter_number_is_rejected():
    f = _layer_files(6)
    url = hg.prs_url(RUN, 6)
    right = _lmsg("cimixr", 500, [0.0] * 4)
    import eccodes
    h = eccodes.codes_new_from_message(right)
    eccodes.codes_set(h, "parameterNumber", 83)
    wrong = eccodes.codes_get_message(h); eccodes.codes_release(h)
    assert len(wrong) == len(right)
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="expected cimixr 500"):
        hg.read_layer_wind(RUN, 6, f.get)


def test_a_message_at_the_wrong_level_is_rejected():
    f = _layer_files(6)
    url = hg.prs_url(RUN, 6)
    right, wrong = _lmsg("u", 500, [50.0] * 4), _lmsg("u", 450, [50.0] * 4)
    assert len(right) == len(wrong)
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="expected u 500"):
        hg.read_layer_wind(RUN, 6, f.get)


def test_a_pres_that_is_not_the_surface_is_rejected():
    f = _layer_files(6)
    url = hg.file_url(RUN, 6)
    right, wrong = _lmsg("pres", 0, [101300.0] * 4), _lmsg("pres", 0, [101300.0] * 4, "cloudBase")
    assert len(right) == len(wrong) and right != wrong
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="surface pressure"):
        hg.read_layer_wind(RUN, 6, f.get)


def test_a_missing_row_fails_the_whole_step():
    f = _layer_files(6, skip=(("CIMIXR", 700),))
    with pytest.raises(KeyError, match="CIMIXR"):
        hg.read_layer_wind(RUN, 6, f.get)


def _two_step_get(files):
    def get(url, rng=None):
        for f in files.values():
            if url in f.blobs or url in f.idx:
                return f.get(url, rng)
        raise KeyError(url)
    return get


def test_build_wind_writes_groups_and_entry(tmp_path):
    files = {fh: _layer_files(fh, u_of=lambda lev: 3.0, v_of=lambda lev: -4.0) for fh in (0, 1)}
    entry = hb.build_wind(RUN, tmp_path, _two_step_get(files), lambda s: None, [0, 1], 2)
    assert entry == wind.manifest_entry(RUN, layer="cloud") and entry["layer"] == "cloud"
    base = int(RUN.timestamp() // 3600)
    got = wind.decode_icw1((tmp_path / "wind/2026111618/N30W120.icw").read_bytes())
    assert got["step_hours"] == [base, base + 1]
    # node (41, -110) holds 3, -4 in every deck (no cloud: the mean over each deck's levels)
    assert got["cube"][0, 0, 0, 11, 10] == 3 and got["cube"][1, 2, 1, 11, 10] == -4


def test_build_wind_drops_the_key_when_a_step_lacks_a_cloud_row(tmp_path):
    files = {0: _layer_files(0), 1: _layer_files(1, skip=(("CLMR", 850),))}
    assert hb.build_wind(RUN, tmp_path, _two_step_get(files), lambda s: None, [0, 1], 2) is None
    assert not (tmp_path / "wind").exists()


def _slow_build(monkeypatch, tmp_path, started_at):
    """Run build() with a fake clock that jumps ELAPSED seconds during thickness."""
    seen = []
    cur = [started_at]
    monkeypatch.setattr(hb, "last_hour", lambda run: 1)
    monkeypatch.setattr(hb, "build_wind", lambda *a, **k: seen.append(k.get("budget_s")))
    lat, lon = np.meshgrid(np.arange(30.0, 50.0, 0.1), np.arange(-120.0, -100.0, 0.1), indexing="ij")
    monkeypatch.setattr(hb, "read_fields", lambda run, fh, get: {
        "lat": lat.ravel(), "lon": lon.ravel(), **{n: np.full(lat.size, 40.0) for n, _ in hb.FIELDS}})

    def thick_then_jump(*a, **k):
        cur[0] = started_at + ELAPSED[0]

    monkeypatch.setattr(hb, "build_thick", thick_then_jump)
    hb.build(RUN, tmp_path, lambda u, r=None: b"", NOW, sleep=lambda s: None, clock=lambda: cur[0])
    return seen


ELAPSED = [0.0]


def test_wind_budget_shrinks_with_a_late_start(monkeypatch, tmp_path):
    ELAPSED[0] = 0.0
    assert _slow_build(monkeypatch, tmp_path / "a", 1000.0) == [600]
    ELAPSED[0] = 800.0     # 1200 - 800 - 120 = 280
    assert _slow_build(monkeypatch, tmp_path / "b", 1000.0) == [280]


def test_wind_skipped_when_no_time_left(monkeypatch, tmp_path, capsys):
    ELAPSED[0] = 1100.0    # 1200 - 1100 - 120 < 0
    assert _slow_build(monkeypatch, tmp_path, 5.0) == []
    assert "::warning::wind: no time left, skipped" in capsys.readouterr().out
    assert "wind" not in json.loads((tmp_path / "manifest.json").read_text())
