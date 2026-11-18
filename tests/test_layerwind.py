import numpy as np
import pytest
from pipeline import layerwind as lw


def _run(levels, u, cloudy, ps, kind="ncep", v=None):
    """One point: levels (hPa), u per level, cloudy per level, surface ps."""
    levels = np.asarray(levels, float)
    u = np.asarray(u, np.float32)[:, None]
    v = np.zeros_like(u) if v is None else np.asarray(v, np.float32)[:, None]
    cloudy = np.asarray(cloudy, bool)[:, None]
    ps = np.array([ps], float)
    return lw.deck_winds(levels, u, v, cloudy, ps, lw.deck_bounds(kind, ps))


def test_constants():
    assert lw.DECK_KEYS == (850, 500, 250)
    assert lw.CONDENSATE_CLOUDY == 1e-5 and lw.TCDC_CLOUDY == 5.0 and lw.MOIST_RH == 90.0


def test_no_cloud_is_weighted_deck_mean():
    # ncep low deck = ps 1013 .. 642. Each level's span inside the deck:
    # 1000: 1013 - 987.5 = 25.5   975: 987.5 - 962.5 = 25   950: 962.5 - 925 = 37.5
    # 900: 925 - 875 = 50         850: 875 - 775 = 100      700: 775 - 642 = 133
    levels = [1000, 975, 950, 900, 850, 700]
    out = _run(levels, [1, 1, 1, 1, 1, 10], [False] * 6, 1013.0)
    w = np.array([25.5, 25, 37.5, 50, 100, 133])
    want = (w * np.array([1, 1, 1, 1, 1, 10])).sum() / w.sum()   # 1568 / 371
    assert out[(850, "u")][0] == pytest.approx(want, rel=1e-6)
    assert out[(850, "u")][0] != pytest.approx(2.5)
    assert out[(850, "u")].dtype == np.float32
    assert out[(850, "v")][0] == 0.0
    # mid and high decks have no level here: NaN, not 0
    for d in (500, 250):
        assert np.isnan(out[(d, "u")][0]) and np.isnan(out[(d, "v")][0])


def test_thickest_run_wins():
    # runs {950} (span 975-925 = 50) and {850, 800, 750, 700}
    # (spans 875-825 = 50, 50, 50, 725-642 = 83; thickness 233)
    levels = [1000, 950, 900, 850, 800, 750, 700]
    u = [0, 5, 0, 10, 20, 30, 40]
    cloudy = [False, True, False, True, True, True, True]
    out = _run(levels, u, cloudy, 1013.0)
    want = (50 * 10 + 50 * 20 + 50 * 30 + 83 * 40) / 233.0
    assert out[(850, "u")][0] == pytest.approx(want, rel=1e-6)


def test_order_of_levels_does_not_matter():
    levels = [700, 1000, 850, 950, 800, 750, 900]
    u = [40, 0, 10, 5, 20, 30, 0]
    cloudy = [True, False, True, True, True, True, False]
    out = _run(levels, u, cloudy, 1013.0)
    assert out[(850, "u")][0] == pytest.approx((500 + 1000 + 1500 + 3320) / 233.0, rel=1e-6)


def test_tie_goes_to_lower_run():
    # {950} and {850}: both span 50 hPa
    levels = [1000, 950, 900, 850, 800]
    out = _run(levels, [0, 3, 0, 7, 0], [False, True, False, True, False], 1013.0, v=[0, -2, 0, 4, 0])
    assert out[(850, "u")][0] == pytest.approx(3.0)
    assert out[(850, "v")][0] == pytest.approx(-2.0)


def test_cloudy_run_without_wind_falls_back_to_deck():
    levels = [1000, 950, 900, 850]
    u = [2, np.nan, 2, 2]
    out = _run(levels, u, [False, True, False, False], 1013.0)
    assert out[(850, "u")][0] == pytest.approx(2.0)


def test_nan_wind_level_still_counts_for_run_thickness():
    # run A {950, 900}: 975-925 + 925-875 = 100, but 900's wind is NaN
    # run B {800}: 825-775 = 50. A is thicker; its wind is 950's alone.
    levels = [1000, 950, 900, 850, 800, 750]
    u = [0, 6, np.nan, 0, 9, 0]
    out = _run(levels, u, [False, True, True, False, True, False], 1013.0)
    assert out[(850, "u")][0] == pytest.approx(6.0)


def test_underground_levels_ignored():
    # ps 750: 1000/925/850/800 hPa are below the ground (value 99)
    levels = [1000, 925, 850, 800, 700, 500, 300, 250, 200]
    u = [99, 99, 99, 99, 1, 2, 3, 3, 3]
    for cloudy in ([False] * 9, [True] * 9):
        out = _run(levels, u, cloudy, 750.0)
        assert out[(850, "u")][0] == pytest.approx(1.0)
        assert out[(500, "u")][0] == pytest.approx(2.0)
        assert out[(250, "u")][0] == pytest.approx(3.0)


def test_level_at_surface_counts_in_low_deck():
    out = _run([850, 700], [4, 4], [False, False], 850.0, kind="icon")
    # icon low deck = ps .. 800: only the 850 level, exactly at the surface
    assert out[(850, "u")][0] == pytest.approx(4.0)


def test_deck_all_underground_is_nan():
    levels = [1000, 850, 700, 500, 300, 200]
    out = _run(levels, [5] * 6, [False] * 6, 600.0, kind="icon")
    assert np.isnan(out[(850, "u")][0]) and np.isnan(out[(850, "v")][0])
    assert out[(500, "u")][0] == pytest.approx(5.0)


def test_ncep_and_icon_bounds():
    ps = np.array([1013.0, 900.0])
    b = lw.deck_bounds("ncep", ps)
    assert [tuple(np.broadcast_to(x, ps.shape)) for x in b[0]] == [(642, 642), (1013, 900)]
    assert [tuple(np.broadcast_to(x, ps.shape)) for x in b[1]] == [(350, 350), (642, 642)]
    assert [tuple(np.broadcast_to(x, ps.shape)) for x in b[2]] == [(150, 150), (350, 350)]
    b = lw.deck_bounds("icon", ps)
    assert [tuple(np.broadcast_to(x, ps.shape)) for x in b[0]] == [(800, 800), (1013, 900)]
    assert [tuple(np.broadcast_to(x, ps.shape)) for x in b[1]] == [(400, 400), (800, 800)]
    assert [tuple(np.broadcast_to(x, ps.shape)) for x in b[2]] == [(150, 150), (400, 400)]
    with pytest.raises(ValueError):
        lw.deck_bounds("gfs", ps)


def test_ecmwf_bounds_follow_ps():
    ps = np.array([1000.0, 700.0])
    (lo0, hi0), (lo1, hi1), (lo2, hi2) = lw.deck_bounds("ecmwf", ps)
    assert np.allclose(lo0, [800, 560]) and np.allclose(hi0, [1000, 700])
    assert np.allclose(hi1, [800, 560]) and np.allclose(lo1, [450, 315])
    assert np.allclose(hi2, [450, 315]) and np.allclose(lo2, [150, 150])
    # 600 hPa is mid deck under ps 1000 and low deck under ps 700
    levels = np.array([600.0])
    u = np.array([[7.0, 7.0]], np.float32)
    out = lw.deck_winds(levels, u, np.zeros_like(u), np.zeros((1, 2), bool), ps, lw.deck_bounds("ecmwf", ps))
    assert np.isnan(out[(850, "u")][0]) and out[(500, "u")][0] == pytest.approx(7.0)
    assert out[(850, "u")][1] == pytest.approx(7.0) and np.isnan(out[(500, "u")][1])


def test_ecmwf_high_deck_empty_when_bound_below_150():
    ps = np.array([300.0])
    (_, _), (lo1, _), (lo2, hi2) = lw.deck_bounds("ecmwf", ps)
    assert np.all(np.asarray(hi2) <= np.asarray(lo2)) and np.all(np.asarray(lo1) >= 150)
    out = lw.deck_winds(np.array([200.0, 150.0]), np.ones((2, 1), np.float32), np.zeros((2, 1), np.float32),
                        np.zeros((2, 1), bool), ps, lw.deck_bounds("ecmwf", ps))
    assert np.isnan(out[(250, "u")][0])


def test_many_points_vectorised():
    rng = np.random.default_rng(1)
    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150], float)
    npts = 50
    u = rng.normal(0, 10, (levels.size, npts)).astype(np.float32)
    v = rng.normal(0, 10, (levels.size, npts)).astype(np.float32)
    cloudy = rng.random((levels.size, npts)) < 0.5
    ps = rng.uniform(600, 1030, npts)
    out = lw.deck_winds(levels, u, v, cloudy, ps, lw.deck_bounds("ncep", ps))
    for j in range(npts):
        one = lw.deck_winds(levels, u[:, j:j + 1], v[:, j:j + 1], cloudy[:, j:j + 1], ps[j:j + 1],
                            lw.deck_bounds("ncep", ps[j:j + 1]))
        for k in out:
            np.testing.assert_array_equal(out[k][j:j + 1], one[k])


# ICON's formulas (src/cloudclues.js rhWaterToIce), Pa, T in K:
#   esw = 611.2 exp(17.62 (T - 273.15) / (T - 30.03))
#   esi = exp(9.550426 - 5723.265 / T + 3.53068 ln T - 0.00728332 T)
# vapour pressure from specific humidity: e = q p / (0.622 + 0.378 q).
def _esw(t):
    return 611.2 * np.exp(17.62 * (t - 273.15) / (t - 30.03))


def _esi(t):
    return np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t) - 0.00728332 * t)


def test_moist_from_q_matches_icon_formula():
    # Inputs and the q that gives an effective RH of 92 % / 88 % (computed with
    # the formulas above, 2026-11-18):
    #   warm T 290 K, p 850 hPa (over water; esw 1914.99 Pa): q92 0.012993966552425476, q88 0.012424745661933382
    #   cold T 250 K, p 500 hPa (over ice;   esi 76.0239 Pa): q92 0.0008705385238246356, q88 0.0008326698698939572
    cases = [(290.0, 850.0, 0.012993966552425476, 0.012424745661933382),
             (250.0, 500.0, 0.0008705385238246356, 0.0008326698698939572)]
    for t, p, q92, q88 in cases:
        es = _esw(t) if t >= 273.15 else _esi(t)
        for q, rh in ((q92, 92.0), (q88, 88.0)):
            e = q * p * 100 / (0.622 + 0.378 * q)
            assert 100 * e / es == pytest.approx(rh, rel=1e-9)
    q = np.array([[0.012993966552425476, 0.012424745661933382, 0.0008705385238246356, 0.0008326698698939572]])
    t = np.array([[290.0, 290.0, 250.0, 250.0]])
    p = np.array([[850.0, 850.0, 500.0, 500.0]])
    assert lw.moist_from_q(q, t, p).tolist() == [[True, False, True, False]]
    # float32 inputs, NaN is never moist, p broadcast per level
    q32 = np.array([[0.0130, np.nan]], np.float32)
    assert lw.moist_from_q(q32, np.float32(290.0), np.array([[850.0]])).tolist() == [[True, False]]


def test_moist_from_rh_ice_conversion():
    # RH_water 85 % at 250 K = 85 * esw/esi = 106.98 % over ice: moist
    assert 85 * _esw(250.0) / _esi(250.0) == pytest.approx(106.98305404307902)
    out = lw.moist_from_rh(np.array([85.0, 85.0, 90.0, 89.9, np.nan]), np.array([250.0, 280.0, 280.0, 280.0, 280.0]))
    assert out.tolist() == [True, False, True, False, False]


def test_underground_deck_bottom_is_ps():
    # ncep, ps 600: 700 is underground. Mid deck = 600 .. 350 (bottom min(642, ps)):
    #   600: 600 - 550 = 50   500: 550 - 450 = 100   400: 450 - 350 = 100
    # High deck = 350 .. 150: 300 alone. Low deck (600 .. 642) is empty.
    levels = [700, 600, 500, 400, 300]
    u = [99, 2, 4, 8, 16]
    out = _run(levels, u, [False] * 5, 600.0)
    assert np.isnan(out[(850, "u")][0]) and np.isnan(out[(850, "v")][0])
    assert out[(500, "u")][0] == pytest.approx((50 * 2 + 100 * 4 + 100 * 8) / 250.0, rel=1e-6)
    assert out[(250, "u")][0] == pytest.approx(16.0)
    # runs {600} (50) and {400} (100): the thicker {400} wins
    out = _run(levels, u, [False, True, False, True, False], 600.0)
    assert out[(500, "u")][0] == pytest.approx(8.0)


def test_missing_t_or_humidity_is_never_moist():
    nan = np.nan
    assert lw.moist_from_rh(np.array([95.0, nan, 95.0]), np.array([nan, 280.0, np.inf])).tolist() == [False, False, False]
    assert lw.moist_from_q(np.array([0.013, nan]), np.array([nan, 290.0]), np.array([850.0, 850.0])).tolist() == [False, False]
