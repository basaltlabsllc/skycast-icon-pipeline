import json
from datetime import datetime, timezone

import eccodes
import numpy as np
import pytest

import rrfs.build as rb
import rrfs.grib as rg
from pipeline import layerwind, wind
from tests.gribfake import Files

UTC = timezone.utc
RUN = datetime(2026, 11, 16, 18, tzinfo=UTC)
NOW = datetime(2026, 11, 16, 22, 40, tzinfo=UTC)
POLE = (-35.0, 247.0)     # RRFS 13 km NA grid: latitude/longitude of the southern pole


def test_prs_url():
    assert rg.prs_url(datetime(2026, 11, 17, 12, tzinfo=UTC), 6).endswith("/rrfs.20261117/12/rrfs.t12z.prslev.13km.f006.na.grib2")
    assert rg.prs_url(RUN, 84).endswith("rrfs.t18z.prslev.13km.f084.na.grib2")
    assert rg.prs_url(RUN, 0).startswith("https://nomads.ncep.noaa.gov/")


# ---- the rotated-pole turn ----------------------------------------------------
def test_rotated_pole_identity_on_the_central_meridian():
    # the grid's central meridian is the south pole's longitude (-113 east): grid
    # north is geographic north at every latitude there
    lat = np.array([5.0, 40.0, 70.0])
    ue, ve = rg.rotate_rotated_pole(np.array([3.0, -7.0, 1.0]), np.array([4.0, 2.0, -9.0]), lat,
                                    np.full(3, -113.0), *POLE)
    assert np.allclose(ue, [3.0, -7.0, 1.0], atol=1e-9) and np.allclose(ve, [4.0, 2.0, -9.0], atol=1e-9)


def test_rotated_pole_keeps_the_speed():
    rng = np.random.default_rng(1)
    lat, lon = rng.uniform(0, 80, 50), rng.uniform(-180, -50, 50)
    u, v = rng.normal(0, 20, 50), rng.normal(0, 20, 50)
    ue, ve = rg.rotate_rotated_pole(u, v, lat, lon, *POLE)
    assert np.allclose(np.hypot(ue, ve), np.hypot(u, v), atol=1e-9)


def test_rotated_pole_sign_east_of_centre():
    # On paper: the rotated north pole is at (+35, 67 E). From (0 N, 23 W), 90 deg
    # of longitude east of the central meridian (-113), the pole lies to the
    # north-east: bearing = atan2(sin 90 cos 35, cos 0 sin 35 - 0) = atan2(cos 35,
    # sin 35) = 55 deg clockwise from north. So a wind blowing along grid north
    # (u = 0, v = 10) blows toward the north-east: east +10 sin 55, north +10 cos 55.
    # (The same tilt, mirrored, west of the central meridian.)
    ue, ve = rg.rotate_rotated_pole(np.array([0.0]), np.array([10.0]), np.array([0.0]), np.array([-23.0]), *POLE)
    assert ue[0] == pytest.approx(10 * np.sin(np.radians(55)), abs=1e-9) and ue[0] > 0
    assert ve[0] == pytest.approx(10 * np.cos(np.radians(55)), abs=1e-9)
    # grid east (u = 10, v = 0) then points south-east
    ue, ve = rg.rotate_rotated_pole(np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([-23.0]), *POLE)
    assert ue[0] == pytest.approx(10 * np.cos(np.radians(55)), abs=1e-9) and ve[0] < 0
    ue, ve = rg.rotate_rotated_pole(np.array([0.0]), np.array([10.0]), np.array([0.0]), np.array([157.0]), *POLE)
    assert ue[0] < 0      # west of the central meridian the tilt is the other way


# ---- a rotated-grid fixture ---------------------------------------------------
LEVELS = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150)
THR = layerwind.CONDENSATE_CLOUDY
CELLS = 4


def _rmsg(kind, lev, vals, uv_relative=1, sample=None, surface_kind="surface", pole=POLE, param=None):
    """A 2 x 2 GRIB2 message on a rotated lat/lon grid: kind u | v | clmr | icmr on `lev` hPa, or
    'pres' (Pa) on `surface_kind` ('surface' or 'cloudBase'). param overrides (category, number)."""
    if kind == "pres":
        h = eccodes.codes_grib_new_from_samples(sample or "rotated_ll_sfc_grib2")
        spec = (("parameterCategory", 3), ("parameterNumber", 0),
                ("typeOfFirstFixedSurface", 1 if surface_kind == "surface" else 2),
                ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", 0))
    else:
        h = eccodes.codes_grib_new_from_samples(sample or "rotated_ll_pl_grib2")
        cat, num = param or {"u": (2, 2), "v": (2, 3), "clmr": (1, 22), "icmr": (1, 23), "absv": (2, 10)}[kind]
        spec = (("parameterCategory", cat), ("parameterNumber", num), ("typeOfFirstFixedSurface", 100),
                ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", lev * 100))
    keys = [("Ni", 2), ("Nj", 2), ("latitudeOfFirstGridPointInDegrees", 1.0),
            ("latitudeOfLastGridPointInDegrees", 0.0), ("longitudeOfFirstGridPointInDegrees", 0.0),
            ("longitudeOfLastGridPointInDegrees", 40.0), ("iDirectionIncrementInDegrees", 40.0),
            ("jDirectionIncrementInDegrees", 1.0), ("bitsPerValue", 24), ("discipline", 0)] + list(spec)
    if (sample or "rotated").startswith("rotated"):
        keys += [("latitudeOfSouthernPoleInDegrees", pole[0]), ("longitudeOfSouthernPoleInDegrees", pole[1])]
    for k, v in keys:
        eccodes.codes_set(h, k, v)
    if uv_relative:
        eccodes.codes_set(h, "uvRelativeToGrid", 1)
    eccodes.codes_set_values(h, np.array(vals, dtype=float))
    b = eccodes.codes_get_message(h)
    eccodes.codes_release(h)
    return b


def _layer_files(fh, u_of=lambda lev: lev / 10.0, v_of=lambda lev: -1.0, clmr_of=lambda lev: 0.0,
                 icmr_of=lambda lev: 0.0, ps_pa=(101300.0,) * CELLS, levels=LEVELS, skip=(), run=RUN, **kw):
    """The prslev file in NOAA's row order per level (HGT, TMP, UGRD, VGRD, ABSV, CLMR, ICMR, RWMR: u/v
    touch and clmr/icmr touch, with ABSV between the pairs), plus 25 mb levels between ours that the
    reader must ignore; and the 2dfld file (PRES cloud base, then PRES surface). Returns a Files."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    filler = _rmsg("u", 250, [1.0] * CELLS, **kw)
    prs = []
    for lev in sorted(set(levels) | {l + 25 for l in levels if l < 1000}, reverse=True):
        mine = lev in levels
        prs += [("HGT", f"{lev} mb", want, filler), ("TMP", f"{lev} mb", want, filler)]
        for kind, name in (("u", "UGRD"), ("v", "VGRD")):
            if (name, lev) not in skip:
                x = (u_of if kind == "u" else v_of)(lev) if mine else 99.0
                prs.append((name, f"{lev} mb", want, _rmsg(kind, lev, [x] * CELLS, **kw)))
        prs.append(("ABSV", f"{lev} mb", want, _rmsg("absv", lev, [1e-5] * CELLS, **kw)))
        for kind, name, f in (("clmr", "CLMR", clmr_of), ("icmr", "ICMR", icmr_of)):
            if (name, lev) not in skip:
                prs.append((name, f"{lev} mb", want, _rmsg(kind, lev, [f(lev) if mine else 0.0] * CELLS, **kw)))
        prs.append(("RWMR", f"{lev} mb", want, filler))
    sfc = [("PRES", "cloud base", want, _rmsg("pres", 0, [70000.0] * CELLS, surface_kind="cloudBase", **kw))]
    if ("PRES", "surface") not in skip:
        sfc.append(("PRES", "surface", want, _rmsg("pres", 0, list(ps_pa), **kw)))
    f = Files()
    f.add(rg.prs_url(run, fh), prs)
    f.add(rg.file_url(run, fh), sfc)
    return f


def _prs_idx(fh=6, skip=()):
    return _layer_files(fh, skip=skip).idx[rg.prs_url(RUN, fh) + ".idx"].decode()


def _sfc_idx(fh=6, skip=()):
    return _layer_files(fh, skip=skip).idx[rg.file_url(RUN, fh) + ".idx"].decode()


def _true_latlon(**kw):
    g = eccodes.codes_new_from_message(_rmsg("u", 250, [0.0] * CELLS, **kw))
    try:
        lat = eccodes.codes_get_array(g, "latitudes")
        lon = eccodes.codes_get_array(g, "longitudes")
    finally:
        eccodes.codes_release(g)
    return lat, np.where(lon > 180, lon - 360, lon)


# ---- the level list and the idx rows -----------------------------------------
def test_layer_wind_levels_are_every_50_mb_1000_to_150():
    assert rg.RRFS_WIND_LEVELS == LEVELS


def test_layer_wind_ranges_cover_every_level_and_the_surface_pressure():
    prs_i, sfc_i = _prs_idx(), _sfc_idx()
    r = rg.layer_wind_ranges(prs_i, sfc_i, 6)
    assert len(r) == len(LEVELS) * 4 + 1
    for lev in LEVELS:
        assert {(lev, c) for c in ("u", "v", "clmr", "icmr")} <= set(r)
    assert ("sfc", "pres") in r and not any(k[0] in (975, 925, 825) for k in r)
    # the surface PRES, not the cloud-base PRES that comes first in that file
    rows = [ln.split(":") for ln in sfc_i.strip().split("\n")]
    surf = next(i for i, x in enumerate(rows) if x[4] == "surface")
    assert r[("sfc", "pres")][0] == int(rows[surf][1])
    with pytest.raises(KeyError):             # rows of another forecast hour are not picked
        rg.layer_wind_ranges(prs_i, sfc_i, 7)
    assert rg.layer_wind_ranges(_prs_idx(0), _sfc_idx(0), 0)[(500, "u")]    # f00 reads "anl"


@pytest.mark.parametrize("name,lev", [("UGRD", 700), ("VGRD", 700), ("CLMR", 700), ("ICMR", 700),
                                      ("ICMR", 150), ("UGRD", 1000), ("PRES", "surface")])
def test_layer_wind_a_missing_row_raises_keyerror(name, lev):
    skip = ((name, lev),)
    with pytest.raises(KeyError, match=name):
        rg.layer_wind_ranges(_prs_idx(skip=skip), _sfc_idx(skip=skip), 6)


def test_a_missing_row_fails_the_whole_step():
    for skip in ((("ICMR", 700),), (("CLMR", 150),), (("VGRD", 1000),), (("PRES", "surface"),)):
        f = _layer_files(6, skip=skip)
        with pytest.raises(KeyError, match=skip[0][0]):
            rg.read_layer_wind(RUN, 6, f.get)


# ---- requests per step --------------------------------------------------------
def test_a_level_is_one_request_and_a_step_makes_21_requests():
    f = _layer_files(6)
    rg.read_layer_wind(RUN, 6, f.get)
    prs, sfc = rg.prs_url(RUN, 6), rg.file_url(RUN, 6)
    urls = [u for u, _ in f.calls]
    assert urls.count(prs + ".idx") == 1 and urls.count(sfc + ".idx") == 1
    ranged = [(u, r) for u, r in f.calls if r is not None]
    assert len([1 for u, _ in ranged if u == prs]) == 18 and len([1 for u, _ in ranged if u == sfc]) == 1
    assert len(f.calls) == 2 + 18 + 1 == 21
    # each prslev request is u .. icmr of one level: five messages, u's start to icmr's end
    r = rg.layer_wind_ranges(f.idx[prs + ".idx"].decode(), f.idx[sfc + ".idx"].decode(), 6)
    want = {(r[(lev, "u")][0], r[(lev, "icmr")][1]) for lev in LEVELS}
    assert {rng for u, rng in ranged if u == prs} == want


def test_the_plan_names_the_absv_message_as_the_one_to_skip():
    _, plan = rg.layer_wind_plan(_prs_idx(), _sfc_idx(), 6)
    prs = [k for which, _, k in plan if which == "prs"]
    assert all(len(k) == 5 and k[2] is None and [x[1] for x in k if x] == ["u", "v", "clmr", "icmr"] for k in prs)


def _with_row_moved(prs_idx, lev, name_from, name_to):
    """Swap two rows' names in the idx text (offsets stay): the byte layout no longer matches the order
    the reader expects at that level."""
    rows = prs_idx.split("\n")
    ia = next(i for i, r in enumerate(rows) if f":{name_from}:{lev} mb:" in r)
    ib = next(i for i, r in enumerate(rows) if f":{name_to}:{lev} mb:" in r)
    ra, rb_ = rows[ia].split(":"), rows[ib].split(":")
    ra[3], rb_[3] = rb_[3], ra[3]
    rows[ia], rows[ib] = ":".join(ra), ":".join(rb_)
    return "\n".join(rows)


def test_a_level_whose_rows_are_not_in_the_expected_order_falls_back_to_separate_requests():
    idx = _with_row_moved(_prs_idx(), 700, "ABSV", "CLMR")           # ABSV and CLMR trade places in the idx
    _, plan = rg.layer_wind_plan(idx, _sfc_idx(), 6)
    at700 = [k for which, _, k in plan if which == "prs" and any(x and x[0] == 700 for x in k)]
    assert len(at700) > 1 and all(None not in k for k in at700)       # never one wide span, no skipped message
    assert sum(len(k) for k in at700) == 4
    # every other level is still one request of five
    others = [k for which, _, k in plan if which == "prs" and not any(x and x[0] == 700 for x in k)]
    assert len(others) == 17 and all(len(k) == 5 for k in others)
    # and a full step on such an idx costs 21 + (more) requests but still never spans more than five messages
    f = _layer_files(6)
    f.idx[rg.prs_url(RUN, 6) + ".idx"] = idx.encode()
    with pytest.raises(ValueError, match="expected"):                # the fixture's bytes disagree with the swapped names
        rg.read_layer_wind(RUN, 6, f.get)


def test_a_missing_absv_row_falls_back_instead_of_widening():
    rows = [r for r in _prs_idx().split("\n") if ":ABSV:500 mb:" not in r]
    _, plan = rg.layer_wind_plan("\n".join(rows), _sfc_idx(), 6)
    at500 = [k for which, _, k in plan if which == "prs" and any(x and x[0] == 500 for x in k)]
    assert all(None not in k for k in at500) and sum(len(k) for k in at500) == 4


def test_the_message_between_v_and_clmr_must_be_absv_by_parameter():
    f = _layer_files(6)
    url = rg.prs_url(RUN, 6)
    right, wrong = _rmsg("absv", 700, [1e-5] * CELLS), _rmsg("u", 700, [1e-5] * CELLS)
    assert len(right) == len(wrong) and f.blobs[url].count(right) == 1
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="absolute vorticity"):
        rg.read_layer_wind(RUN, 6, f.get)


# ---- values: the deck winds on the cloud layer --------------------------------
def _ref(u_of, cloudy, ps_hpa, v_of=lambda lev: -1.0):
    p = np.array(LEVELS, float)
    return layerwind.deck_winds(p, [np.array([u_of(l)]) for l in LEVELS], [np.array([v_of(l)]) for l in LEVELS],
                                [np.array([l in cloudy]) for l in LEVELS], np.array([ps_hpa]),
                                layerwind.deck_bounds("ncep", np.array([ps_hpa])))


def test_read_layer_wind_returns_deck_winds_over_the_cloud_layer():
    cloudy = {700, 650}
    # uv_relative=0: the components are already east/north, so the reference needs no turn
    f = _layer_files(6, clmr_of=lambda lev: 2 * THR if lev in cloudy else 0.0,
                     ps_pa=(101300.0, 101300.0, 60000.0, 60000.0), uv_relative=0)
    d = rg.read_layer_wind(RUN, 6, f.get)
    lat, lon = _true_latlon()
    assert set(k for k in d if isinstance(k, tuple)) == {(l, c) for l in (850, 500, 250) for c in "uv"}
    assert d[(850, "u")].dtype == np.float32 and d[(850, "u")].shape == (4,)
    assert np.allclose(d["lat"], lat) and np.allclose(d["lon"], lon)
    for pt, ps in ((0, 1013.0), (2, 600.0)):
        want = _ref(lambda lev: lev / 10.0, cloudy, ps)
        for key in (850, 500, 250):
            for c in "uv":
                got, w = d[(key, c)][pt], want[(key, c)][0]
                assert (np.isnan(got) and np.isnan(w)) or got == pytest.approx(w, rel=1e-5), (pt, key, c)
    assert 65.0 <= d[(850, "u")][0] <= 70.0       # ps 1013: 700 and 650 are the low deck's cloudy run
    assert np.isnan(d[(850, "u")][2]) and np.isnan(d[(850, "v")][2])     # ps 600: low deck underground, NaN not 0


def test_deck_winds_in_slices_equal_one_call(monkeypatch):
    f = _layer_files(6, clmr_of=lambda lev: 2 * THR if lev in (700, 400) else 0.0,
                     ps_pa=(101300.0, 95000.0, 60000.0, 70000.0))
    whole = rg.read_layer_wind(RUN, 6, f.get)
    monkeypatch.setattr(rg, "LAYER_CHUNK", 1)
    sliced = rg.read_layer_wind(RUN, 6, f.get)
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
    f = _layer_files(6, clmr_of=lambda lev: cl.get(lev, 0.0), icmr_of=lambda lev: ci.get(lev, 0.0))
    rg.read_layer_wind(RUN, 6, f.get)
    on = lambda lev: bool(seen["cloudy"][seen["levels"].index(lev)][0])
    assert on(600) and on(500) and not on(400) and not on(300)
    assert np.allclose(seen["ps"], 1013.0)       # Pa -> hPa


def test_each_levels_wind_is_turned_with_the_rotated_pole_before_the_layer_mean(monkeypatch):
    seen = {}
    real = layerwind.deck_winds

    def spy(levels, u, v, cloudy, ps, bounds):
        seen["levels"], seen["u"], seen["v"] = list(levels), u, v
        return real(levels, u, v, cloudy, ps, bounds)

    monkeypatch.setattr(layerwind, "deck_winds", spy)
    f = _layer_files(6, u_of=lambda lev: lev / 10.0, v_of=lambda lev: lev / 100.0)
    rg.read_layer_wind(RUN, 6, f.get)
    lat, lon = _true_latlon()
    for k, lev in enumerate(seen["levels"]):
        ue, ve = rg.rotate_rotated_pole(np.full(CELLS, lev / 10.0), np.full(CELLS, lev / 100.0), lat, lon, *POLE)
        assert np.allclose(seen["u"][k], ue, atol=1e-3) and np.allclose(seen["v"][k], ve, atol=1e-3), lev
    assert np.abs(seen["v"][0] - 10.0).max() > 1.0      # 1000 mb: the fixture's grid does turn the wind


def test_uv_relative_to_earth_is_left_alone(monkeypatch):
    seen = {}
    real = layerwind.deck_winds
    monkeypatch.setattr(layerwind, "deck_winds", lambda l, u, v, c, ps, b: (seen.update(u=u), real(l, u, v, c, ps, b))[1])
    rg.read_layer_wind(RUN, 6, _layer_files(6, u_of=lambda lev: 10.0, v_of=lambda lev: 0.0, uv_relative=0).get)
    assert np.allclose(seen["u"][0], 10.0)


def test_refuses_an_unsupported_grid():
    with pytest.raises(ValueError, match="wind: unsupported grid regular_ll"):
        rg.read_layer_wind(RUN, 6, _layer_files(6, sample="regular_ll_pl_grib2").get)


# ---- every message is checked by parameter, not only by level -----------------
def test_wind_message_in_a_cloud_slot_is_rejected():
    for slot, other in (("clmr", "icmr"), ("icmr", "u"), ("clmr", "v")):
        f = _layer_files(6)
        url = rg.prs_url(RUN, 6)
        right, wrong = _rmsg(slot, 700, [0.0] * CELLS), _rmsg(other, 700, [0.0] * CELLS)
        assert len(right) == len(wrong) and f.blobs[url].count(right) == 1
        f.blobs[url] = f.blobs[url].replace(right, wrong)
        with pytest.raises(ValueError, match=f"expected {slot} 700"):
            rg.read_layer_wind(RUN, 6, f.get)


def test_an_icmr_message_with_another_parameter_number_is_rejected():
    f = _layer_files(6)
    url = rg.prs_url(RUN, 6)
    right, wrong = _rmsg("icmr", 500, [0.0] * CELLS), _rmsg("icmr", 500, [0.0] * CELLS, param=(1, 24))
    assert len(wrong) == len(right) and right != wrong
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="expected icmr 500"):
        rg.read_layer_wind(RUN, 6, f.get)


def test_a_message_at_the_wrong_level_is_rejected():
    f = _layer_files(6)
    url = rg.prs_url(RUN, 6)
    right, wrong = _rmsg("u", 500, [50.0] * CELLS), _rmsg("u", 450, [50.0] * CELLS)
    assert len(right) == len(wrong)
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="expected u 500"):
        rg.read_layer_wind(RUN, 6, f.get)


def test_a_pres_that_is_not_the_surface_is_rejected():
    f = _layer_files(6)
    url = rg.file_url(RUN, 6)
    right = _rmsg("pres", 0, [101300.0] * CELLS)
    wrong = _rmsg("pres", 0, [101300.0] * CELLS, surface_kind="cloudBase")
    assert len(right) == len(wrong) and right != wrong
    f.blobs[url] = f.blobs[url].replace(right, wrong)
    with pytest.raises(ValueError, match="surface pressure"):
        rg.read_layer_wind(RUN, 6, f.get)


def test_wind_geometry_lambert_branch_reads_lov_and_latin1(monkeypatch):
    seen = {}
    vals = {"gridType": "lambert", "uvRelativeToGrid": 1, "LoVInDegrees": 262.5, "Latin1InDegrees": 38.5}
    monkeypatch.setattr(rg.eccodes, "codes_get", lambda g, k: vals[k])
    monkeypatch.setattr(rg.eccodes, "codes_get_array",
                        lambda g, k: np.array([41.0]) if k == "latitudes" else np.array([280.0]))
    lat, lon, turn = rg._wind_geometry(object())
    assert lon[0] == -80.0
    ue, ve = turn(np.array([10.0]), np.array([0.0]))
    a = np.sin(np.radians(38.5)) * np.radians(17.5)
    assert ve[0] == pytest.approx(-10 * np.sin(a))


# ---- build_wind ---------------------------------------------------------------
def _get_for(files):
    def get(url, rng=None):
        for f in files.values():
            if url in f.blobs or url in f.idx:
                return f.get(url, rng)
        raise KeyError(url)
    return get


def test_build_wind_failure_drops_key(tmp_path):
    def get(url, rng=None):
        raise OSError("boom")

    assert rb.build_wind(RUN, tmp_path, get, lambda s: None, [0, 1], 2) is None
    assert not (tmp_path / "wind").exists()


def test_build_wind_writes_the_cover_groups_and_entry_through_the_limiter(tmp_path):
    files = {fh: _layer_files(fh, u_of=lambda lev: 3.0, v_of=lambda lev: -4.0) for fh in (0, 1)}
    lim = rb.RateLimiter(1000)
    entry = rb.build_wind(RUN, tmp_path, rb.limited(_get_for(files), lim), lambda s: None, [0, 1], 2)
    assert entry == wind.manifest_entry(RUN, layer="cloud") and entry["layer"] == "cloud"
    assert len(lim._stamps) == 42                     # 21 requests an hour, every one counted
    base = int(RUN.timestamp() // 3600)
    names = sorted(p.name for p in (tmp_path / "wind/2026111618").glob("*.icw"))
    assert names and all(n in [g + ".icw" for g in rb.WIND_GROUPS] for n in names)
    lat, lon = _true_latlon()
    g = wind.group_of(round(lat[0]), round(lon[0]))
    got = wind.decode_icw1((tmp_path / f"wind/2026111618/{g}.icw").read_bytes())
    assert got["step_hours"] == [base, base + 1]


def test_wind_groups_are_the_cover_domain():
    assert len(rb.WIND_GROUPS) == 5 * 7 and rb.WIND_GROUPS[0] == "S10W180" and rb.WIND_GROUPS[-1] == "N70W060"


# ---- the time cap -------------------------------------------------------------
ELAPSED = [0.0]


def _slow_build(monkeypatch, tmp_path, started_at):
    seen = []
    cur = [started_at]
    monkeypatch.setattr(rb, "build_wind", lambda *a, **k: seen.append(k.get("budget_s")))
    lat, lon = np.meshgrid(np.arange(30.0, 50.0, 0.1), np.arange(-120.0, -100.0, 0.1), indexing="ij")

    def read_then_jump(run, fh, get):
        cur[0] = started_at + ELAPSED[0]
        return {"lat": lat.ravel(), "lon": lon.ravel(), **{n: np.full(lat.size, 40.0) for n, _ in rb.FIELDS}}

    monkeypatch.setattr(rb, "read_fields", read_then_jump)
    rb.build(RUN, tmp_path, lambda u, r=None: b"", NOW, sleep=lambda s: None, fhs=[0], clock=lambda: cur[0])
    return seen


def test_wind_budget_shrinks_with_a_late_start(monkeypatch, tmp_path):
    ELAPSED[0] = 0.0
    assert _slow_build(monkeypatch, tmp_path / "a", 1000.0) == [600]
    ELAPSED[0] = 2000.0     # 2400 - 2000 - 120 = 280
    assert _slow_build(monkeypatch, tmp_path / "b", 1000.0) == [280]


def test_wind_skipped_when_no_time_left(monkeypatch, tmp_path, capsys):
    ELAPSED[0] = 2300.0     # 2400 - 2300 - 120 < 0
    assert _slow_build(monkeypatch, tmp_path, 5.0) == []
    assert "::warning::wind: no time left, skipped" in capsys.readouterr().out
    assert "wind" not in json.loads((tmp_path / "manifest.json").read_text())


def test_workflow_timeout_matches_the_workflow_file():
    from pathlib import Path
    yml = (Path(__file__).resolve().parent.parent / ".github/workflows/rrfs.yml").read_text()
    assert f"timeout -k 10 {rb.WORKFLOW_TIMEOUT_S} python rrfs_build.py" in yml


def test_build_puts_the_wind_entry_in_the_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(rb, "build_wind", lambda *a, **k: wind.manifest_entry(RUN))
    lat, lon = np.meshgrid(np.arange(30.0, 50.0, 0.1), np.arange(-120.0, -100.0, 0.1), indexing="ij")
    monkeypatch.setattr(rb, "read_fields", lambda run, fh, get: {
        "lat": lat.ravel(), "lon": lon.ravel(), **{n: np.full(lat.size, 40.0) for n, _ in rb.FIELDS}})
    rb.build(RUN, tmp_path, lambda u, r=None: b"", NOW, sleep=lambda s: None, fhs=[0])
    assert json.loads((tmp_path / "manifest.json").read_text())["wind"] == wind.manifest_entry(RUN)


def test_build_wind_drops_the_key_when_a_step_lacks_a_cloud_row(tmp_path):
    files = {0: _layer_files(0), 1: _layer_files(1, skip=(("ICMR", 850),))}
    assert rb.build_wind(RUN, tmp_path, _get_for(files), lambda s: None, [0, 1], 2) is None
    assert not (tmp_path / "wind").exists()


def test_wind_hours_are_every_third_hour_and_the_cover_stays_hourly():
    assert rb.WIND_STEP_H == 3
    assert rb.wind_hours(list(rb.STEPS)) == list(range(0, 85, 3)) and len(rb.wind_hours(list(rb.STEPS))) == 29
    assert rb.STEPS == tuple(range(85))                      # the cover is untouched
    assert rb.wind_hours([1, 2, 4]) == []


def test_build_hands_wind_only_the_wind_hours_and_the_cover_every_hour(monkeypatch, tmp_path):
    got, covered = [], []
    monkeypatch.setattr(rb, "build_wind", lambda run, out, get, sleep, fhs, workers, **k: got.append(list(fhs)))
    lat, lon = np.meshgrid(np.arange(30.0, 50.0, 1.0), np.arange(-120.0, -100.0, 1.0), indexing="ij")

    def fields(run, fh, get):
        covered.append(fh)
        return {"lat": lat.ravel(), "lon": lon.ravel(), **{n: np.full(lat.size, 40.0) for n, _ in rb.FIELDS}}

    monkeypatch.setattr(rb, "read_fields", fields)
    rb.build(RUN, tmp_path, lambda u, r=None: b"", NOW, sleep=lambda s: None)
    assert got == [list(range(0, 85, 3))] and sorted(covered) == list(range(85))
