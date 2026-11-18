import dataclasses
import json
from datetime import datetime, timezone

import numpy as np
import pytest

import globalfeed.build as gb
from globalfeed import grib
from globalfeed.models import GFS, AIFS, IFS
from pipeline.squares import square_name

UTC = timezone.utc
RUN = datetime(2026, 11, 8, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 8, 5, 10, tzinfo=UTC)


GFS_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150)


def _ncep_rows(levels=GFS_LEVELS, fh=6, skip=()):
    """(name, level text, forecast text) rows in a GFS idx's order; `skip` drops a (name, level)."""
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [("HGT", "850 mb", want)]
    for lev in levels:
        for name in ("UGRD", "VGRD", "TCDC"):
            rows.append((name, f"{lev} mb", want))
    rows.append(("UGRD", "100 mb", want))          # above 150: never asked for
    rows.append(("PRES", "surface", want))
    rows.append(("UGRD", "250 mb", "anl"))          # another forecast hour's row
    return [r for r in rows if (r[0], r[1]) not in skip]


def _ncep_idx(levels=GFS_LEVELS, fh=6, skip=()):
    return "\n".join(f"{i + 1}:{i * 100}:d=2026110800:{n}:{lv}:{f}:"
                      for i, (n, lv, f) in enumerate(_ncep_rows(levels, fh, skip)))


def test_gfs_ranges_cover_every_deck_level():
    idx = _ncep_idx()
    rows = _ncep_rows()
    r = grib.wind_ranges(GFS, idx, 6)
    assert len(r) == 3 * len(GFS_LEVELS) + 1
    for lev in GFS_LEVELS:
        for comp, name in (("u", "UGRD"), ("v", "VGRD"), ("tcdc", "TCDC")):
            k = rows.index((name, f"{lev} mb", "6 hour fcst"))
            assert r[(lev, comp)] == (k * 100, (k + 1) * 100 - 1), (lev, comp)
    k = rows.index(("PRES", "surface", "6 hour fcst"))
    assert r[("sfc", "pres")] == (k * 100, (k + 1) * 100 - 1)
    assert (100, "u") not in r                      # 100 mb is above every deck
    with pytest.raises(KeyError):
        grib.wind_ranges(GFS, idx, 12)


def test_gfs_missing_tcdc_row_raises_keyerror():
    for skip in ((("TCDC", "700 mb"),), (("VGRD", "500 mb"),), (("PRES", "surface"),)):
        with pytest.raises(KeyError):
            grib.wind_ranges(GFS, _ncep_idx(skip=skip), 6)
    with pytest.raises(KeyError):
        grib.wind_ranges(GFS, "\n".join(_ncep_idx().split("\n")[:5]), 6)


def test_gfs_anl_rows_at_f000():
    r = grib.wind_ranges(GFS, _ncep_idx(fh=0), 0)
    assert (850, "tcdc") in r and ("sfc", "pres") in r


def _row(param, levtype, lev, off, ln):
    d = {"param": param, "levtype": levtype, "_offset": off, "_length": ln}
    if lev is not None:
        d["levelist"] = str(lev)
    return json.dumps(d)


EC_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150)


def _ec_index_rows(model, skip=()):
    """ECMWF .index rows (param, levtype, level) for the wind of `model`, plus decoys."""
    hums = ("q",) if model.name == "aifs" else ("r", "q")     # IFS lists both; AIFS has no r
    rows = [("u", "sfc", None)]                        # a 10 m-style row that must not be picked
    for lev in EC_LEVELS + (100, 50):                  # 100 and 50 hPa: above every deck, never asked for
        for p in ("v", "u") + hums + ("t", "w"):
            rows.append((p, "pl", lev))
    rows.append(("sp", "sfc", None))
    return [r for r in rows if r not in skip]


def _ec_index(model, skip=()):
    return "\n".join(_row(p, lt, lev, i * 100, 100) for i, (p, lt, lev) in enumerate(_ec_index_rows(model, skip)))


def test_ecmwf_wind_ranges_cover_every_deck_level():
    for model, hum in ((AIFS, "q"), (IFS, "q")):
        rows = _ec_index_rows(model)
        r = grib.wind_ranges(model, _ec_index(model), 6)
        assert len(r) == 4 * len(EC_LEVELS) + 1
        for lev in EC_LEVELS:
            for comp in ("u", "v", hum, "t"):
                k = rows.index((comp, "pl", lev))
                assert r[(lev, comp)] == (k * 100, k * 100 + 99), (model.name, lev, comp)
        k = rows.index(("sp", "sfc", None))
        assert r[("sfc", "sp")] == (k * 100, k * 100 + 99)
        assert (100, "u") not in r and (850, "w") not in r and (850, "r") not in r
    assert grib.AIFS_WIND_LEVELS == EC_LEVELS and grib.IFS_WIND_LEVELS == EC_LEVELS


def test_ecmwf_missing_humidity_or_sp_row_raises_keyerror():
    for model, skip in ((AIFS, ("q", "pl", 700)), (AIFS, ("t", "pl", 250)), (AIFS, ("u", "pl", 150)),
                        (AIFS, ("sp", "sfc", None)), (IFS, ("q", "pl", 925)), (IFS, ("v", "pl", 1000))):
        with pytest.raises(KeyError):
            grib.wind_ranges(model, _ec_index(model, skip=(skip,)), 6)
    # IFS's r is no stand-in for q (its r is over ice below 0 C): an IFS index whose
    # 925 hPa q row is missing fails even though r is there
    with pytest.raises(KeyError, match="q "):
        grib.wind_ranges(IFS, _ec_index(IFS, skip=(("q", "pl", 925),)), 6)


def _small(model, main, tail=()):
    return dataclasses.replace(model, main_steps=tuple(main), tail_steps=tuple(tail))


def _fake_wind(model, run, fh, get):
    get(model.index_url(run, fh))
    lat = np.arange(-90, 90.5, 1.0)
    lon = np.arange(-180, 180, 1.0)
    out = {"lat": lat, "lon": lon}
    for lev in (850, 500, 250):
        for c in "uv":
            out[(lev, c)] = np.full((lat.size, lon.size), 5.0 if c == "u" else -3.0)
    return out


def test_build_wind_writes_and_returns_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(gb, "read_wind_step", _fake_wind)
    m = _small(GFS, (0, 6), (9,))
    e = gb.build_wind(m, RUN, tmp_path, lambda u, r=None: b"", lambda s: None)
    assert e["format"] == "ICW1" and e["tiles"] == "wind/2026110800"
    from pipeline import wind
    got = wind.decode_icw1((tmp_path / "wind/2026110800/N30W120.icw").read_bytes())
    base = int(RUN.timestamp() // 3600)
    assert got["step_hours"] == [base, base + 6, base + 9]
    assert got["cube"][1, 0, 0, 5, 5] == 5 and got["cube"][1, 0, 1, 5, 5] == -3


def test_build_wind_failure_drops_key(tmp_path):
    m = _small(GFS, (0, 6))
    calls = []

    def get(url, rng=None):
        calls.append(url)
        raise OSError("boom")

    assert gb.build_wind(m, RUN, tmp_path, get, lambda s: None) is None
    assert not (tmp_path / "wind").exists()


def test_manifest_has_wind_only_when_built(monkeypatch, tmp_path):
    names = [square_name(40, -125)]
    monkeypatch.setattr(gb, "all_squares", lambda: list(names))
    monkeypatch.setattr(gb, "read_wind_step", _fake_wind)

    def fake_read(model, run, fh, get):
        lat, lon = np.meshgrid(np.arange(-90, 90.25, 0.25), np.arange(-180, 180, 0.25), indexing="ij")
        out = {"lat": lat.ravel(), "lon": lon.ravel()}
        for f in model.fields:
            out[f if isinstance(f, str) else f[0]] = np.full(lat.size, 40.0)
        return out

    monkeypatch.setattr(gb, "read_step", fake_read)
    get = lambda u, r=None: b""
    for model, has in ((_small(GFS, (0, 6)), True), (_small(dataclasses.replace(AIFS, wind=False), (0, 6)), False)):
        out = tmp_path / model.name
        gb.build(model, RUN, out, get, NOW, tmp_path / ("st" + model.name), sleep=lambda s: None)
        man = json.loads((out / "manifest.json").read_text())
        assert ("wind" in man) is has
        if has:
            assert man["wind"]["format"] == "ICW1" and (out / "wind/2026110800").is_dir()
        else:
            assert not (out / "wind").exists()


def test_build_wind_over_budget_drops_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gb, "read_wind_step", _fake_wind)
    now = [0.0]

    def get(url, rng=None):
        now[0] += 400.0                      # every index read costs 400 s
        return b""

    m = dataclasses.replace(_small(GFS, (0, 6, 12, 18)), workers=1)
    assert gb.build_wind(m, RUN, tmp_path, get, lambda s: None, budget_s=600, clock=lambda: now[0]) is None
    assert not (tmp_path / "wind").exists()
    assert "::warning::wind: over budget" in capsys.readouterr().out
    now[0] = 0.0
    assert gb.build_wind(_small(GFS, (0,)), RUN, tmp_path, get, lambda s: None, budget_s=600, clock=lambda: now[0])


def _timed_build(monkeypatch, tmp_path, started_at, elapsed):
    """build() with a fake clock that jumps `elapsed` s during thickness; returns build_wind's budgets."""
    seen = []
    cur = [started_at]
    names = [square_name(40, -125)]
    monkeypatch.setattr(gb, "all_squares", lambda: list(names))
    monkeypatch.setattr(gb, "build_wind", lambda *a, **k: seen.append(k.get("budget_s")))

    def fake_read(model, run, fh, get):
        lat, lon = np.meshgrid(np.arange(-90, 90.25, 0.25), np.arange(-180, 180, 0.25), indexing="ij")
        out = {"lat": lat.ravel(), "lon": lon.ravel()}
        for f in model.fields:
            out[f if isinstance(f, str) else f[0]] = np.full(lat.size, 40.0)
        return out

    def thick_then_jump(*a, **k):
        cur[0] = started_at + elapsed

    monkeypatch.setattr(gb, "read_step", fake_read)
    monkeypatch.setattr(gb, "build_thick", thick_then_jump)
    gb.build(_small(GFS, (0, 6)), RUN, tmp_path, lambda u, r=None: b"", NOW, tmp_path / "st",
             sleep=lambda s: None, clock=lambda: cur[0])
    return seen


def test_wind_budget_shrinks_with_a_late_start(monkeypatch, tmp_path):
    assert _timed_build(monkeypatch, tmp_path / "a", 1000.0, 0.0) == [600]
    assert _timed_build(monkeypatch, tmp_path / "b", 1000.0, 2200.0) == [380]   # 2700 - 2200 - 120


def test_wind_skipped_when_no_time_left(monkeypatch, tmp_path, capsys):
    assert _timed_build(monkeypatch, tmp_path, 5.0, 2600.0) == []              # 2700 - 2600 - 120 < 0
    assert "::warning::wind: no time left, skipped" in capsys.readouterr().out
    assert "wind" not in json.loads((tmp_path / "manifest.json").read_text())


def test_wind_flags():
    assert GFS.wind and AIFS.wind and IFS.wind
    assert GFS.wind_layer == "cloud" and AIFS.wind_layer == "humidity" and IFS.wind_layer == "humidity"


_EC_PARAM = {"u": (2, 2), "v": (2, 3), "t": (0, 0), "q": (1, 0), "r": (1, 1)}


def _ec_msg(kind, lev, vals):
    """A 2 x 2 ECMWF-style GRIB2 message: u / v / t / q / r on `lev` hPa, or sp at the surface (Pa)."""
    import eccodes
    if kind == "sp":
        h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        spec = (("parameterCategory", 3), ("parameterNumber", 0), ("typeOfFirstFixedSurface", 1),
                ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", 0))
    else:
        h = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
        cat, num = _EC_PARAM[kind]
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


def _esw(t):
    return 611.2 * np.exp(17.62 * (t - 273.15) / (t - 30.03))


def _q_at(rh, t, p_hpa):
    """Specific humidity (kg/kg) for RH over water `rh` % at T (K), p (hPa)."""
    e = rh / 100.0 * _esw(t)
    return 0.622 * e / (p_hpa * 100.0 - 0.378 * e)


# Columns for the four points (2 x 2): RH over water (%) and T (K) per level, the same at
# every point; ps differs per point. 925 is the only moist level in the low deck at 1013
# hPa, 600 the only one in the mid deck; 1000 is moist too but underground where ps = 900.
# 300 hPa is cold (250 K): RH 80 % over water is about 100 % over ice, so moist.
EC_RH = {1000: 95.0, 925: 95.0, 850: 20.0, 700: 30.0, 600: 96.0, 500: 40.0,
         400: 20.0, 300: 80.0, 250: 10.0, 200: 10.0, 150: 10.0}
EC_T = {lev: (250.0 if lev == 300 else 280.0) for lev in EC_LEVELS}
EC_PS = [101300.0, 101300.0, 90000.0, 90000.0]


def _ec_fixture(model, u_of=lambda lev: lev / 10.0, ps_pa=EC_PS, swap=None):
    """The model's wind file (2 x 2 fields) laid in a scrambled order like the real one,
    its .index, and a get() serving both. swap=(key, kind): that slot holds a `kind` message."""
    hum = grib.ECMWF_WIND[model.name][1]
    parts = []
    for lev in EC_LEVELS:
        for kind in ("v", "u", hum, "t"):
            val = {"u": u_of(lev), "v": -1.0, "t": EC_T[lev], "r": EC_RH[lev],
                   "q": _q_at(EC_RH[lev], EC_T[lev], lev)}[kind]
            parts.append(((lev, kind), _ec_msg(kind, lev, [val] * 4)))
    parts.append((("sfc", "sp"), _ec_msg("sp", 0, ps_pa)))
    parts = parts[1::2] + parts[0::2]                  # the real file is not in level order
    rows, buf = [], b""
    for key, m in parts:
        if swap and key == swap[0]:
            m = _ec_msg(swap[1], 850 if key[0] == "sfc" else key[0], [1.0] * 4)
        lev, kind = key
        rows.append(_row(kind, "sfc" if lev == "sfc" else "pl", None if lev == "sfc" else lev, len(buf), len(m)))
        buf += m
    idx = "\n".join(rows)

    def get(url, rng=None):
        if url.endswith(".index"):
            return idx.encode()
        return buf[rng[0]:rng[1] + 1]
    return get


def test_ecmwf_read_wind_step_returns_humidity_layer_winds():
    from pipeline import layerwind
    for model in (AIFS, IFS):
        out = grib.read_wind_step(model, RUN, 6, _ec_fixture(model))
        assert sorted(k for k in out if k not in ("lat", "lon")) == sorted((l, c) for l in (850, 500, 250) for c in "uv")
        assert out[(850, "u")].shape == (2, 2) and out[(850, "u")].dtype == np.float32
        # the expected values from Task 1's core, run by hand on the same column
        p = np.array(EC_LEVELS, float)
        t = [np.array([EC_T[l]]) for l in EC_LEVELS]
        moist = [layerwind.moist_from_q(np.array([_q_at(EC_RH[l], EC_T[l], l)]), t[k], l)
                 for k, l in enumerate(EC_LEVELS)]
        assert [l for k, l in enumerate(EC_LEVELS) if moist[k][0]] == [1000, 925, 600, 300], model.name
        for pt, ps in ((0, 1013.0), (2, 900.0)):
            want = layerwind.deck_winds(p, [np.array([l / 10.0]) for l in EC_LEVELS],
                                        [np.array([-1.0])] * len(EC_LEVELS), moist, np.array([ps]),
                                        layerwind.deck_bounds("ecmwf", np.array([ps])))
            for key in (850, 500, 250):
                for c in "uv":
                    got, w = out[(key, c)].reshape(-1)[pt], want[(key, c)][0]
                    assert got == pytest.approx(w, rel=1e-5), (model.name, pt, key, c)
        # ps 1013: low deck 1013..810 holds 1000, 925, 850; the thickest moist run is 1000-925
        # (weights 13 + 75 + ... ), mid 810..456 has 600 alone, high 456..150 has 300 alone
        w1000, w925 = 1013.0 - 962.5, 962.5 - 887.5
        assert out[(850, "u")][0, 0] == pytest.approx((100.0 * w1000 + 92.5 * w925) / (w1000 + w925), rel=1e-5)
        assert out[(500, "u")][0, 0] == pytest.approx(60.0)
        assert out[(250, "u")][0, 0] == pytest.approx(30.0)
        # ps 900: 1000 and 925 are underground, so the moist 1000-925 run never counts; the
        # low deck 900..720 holds 850 alone, which is dry: the deck mean is 850's wind
        assert out[(850, "u")][1, 0] == pytest.approx(85.0)
        assert out[(500, "u")][1, 0] == pytest.approx(60.0)


def test_ecmwf_read_wind_step_rejects_swapped_message():
    with pytest.raises(ValueError, match="expected q 700"):
        grib.read_wind_step(AIFS, RUN, 6, _ec_fixture(AIFS, swap=((700, "q"), "t")))
    with pytest.raises(ValueError, match="expected q 925"):
        grib.read_wind_step(IFS, RUN, 6, _ec_fixture(IFS, swap=((925, "q"), "r")))
    with pytest.raises(ValueError, match="expected u 500"):
        grib.read_wind_step(IFS, RUN, 6, _ec_fixture(IFS, swap=((500, "u"), "v")))
    with pytest.raises(ValueError, match="surface pressure"):
        grib.read_wind_step(AIFS, RUN, 6, _ec_fixture(AIFS, swap=(("sfc", "sp"), "t")))


def test_ecmwf_pressure_level_message_must_be_isobaric():
    import eccodes
    get = _ec_fixture(AIFS)
    good = _ec_msg("t", 850, [EC_T[850]] * 4)
    h = eccodes.codes_new_from_message(good)
    eccodes.codes_set(h, "typeOfFirstFixedSurface", 105)          # a model level numbered 850
    eccodes.codes_set(h, "scaledValueOfFirstFixedSurface", 850)
    bad = eccodes.codes_get_message(h); eccodes.codes_release(h)
    assert len(bad) == len(good)

    def g(url, rng=None):
        b = get(url, rng)
        return b if url.endswith(".index") else b.replace(good, bad)
    with pytest.raises(ValueError, match="expected t 850"):
        grib.read_wind_step(AIFS, RUN, 6, g)


def _gfs_msg(kind, lev, vals):
    """A 2 x 2 GRIB2 message: kind 'u' | 'v' | 'tcdc' on `lev` hPa, or 'pres' at the surface (Pa)."""
    import eccodes
    if kind == "pres":
        h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        spec = (("parameterCategory", 3), ("parameterNumber", 0), ("typeOfFirstFixedSurface", 1),
                ("scaleFactorOfFirstFixedSurface", 0), ("scaledValueOfFirstFixedSurface", 0))
    else:
        h = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
        cat, num = {"u": (2, 2), "v": (2, 3), "tcdc": (6, 1)}[kind]
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


def _gfs_fixture(u_of, tcdc_of, ps_pa, levels=GFS_LEVELS):
    """A GFS file with the 2 x 2 fields below, in idx order, and a get() serving it."""
    rows, buf, parts = [], b"", []
    for lev in levels:
        for kind, name in (("u", "UGRD"), ("v", "VGRD"), ("tcdc", "TCDC")):
            val = {"u": u_of(lev), "v": -1.0, "tcdc": tcdc_of(lev)}[kind]
            parts.append((name, f"{lev} mb", _gfs_msg(kind, lev, [val] * 4)))
    parts.append(("PRES", "surface", _gfs_msg("pres", 0, ps_pa)))
    for i, (name, lev, m) in enumerate(parts):
        rows.append(f"{i + 1}:{len(buf)}:d=2026110800:{name}:{lev}:6 hour fcst:")
        buf += m
    idx = "\n".join(rows)

    def get(url, rng=None):
        if url.endswith(".idx"):
            return idx.encode()
        return buf[rng[0]:(None if rng[1] is None else rng[1] + 1)]
    return get


def test_gfs_read_wind_step_returns_deck_layer_winds():
    from pipeline import layerwind
    levels = GFS_LEVELS
    u_of = lambda lev: lev / 10.0
    cloudy = {700, 600}                        # one cloudy run, in the low and mid decks
    get = _gfs_fixture(u_of, lambda lev: 80.0 if lev in cloudy else 1.0, [101300.0, 101300.0, 60000.0, 60000.0])
    out = grib.read_wind_step(GFS, RUN, 6, get)
    assert sorted(k for k in out if k not in ("lat", "lon")) == sorted((l, c) for l in (850, 500, 250) for c in "uv")
    assert out[(850, "u")].shape == (2, 2)
    # the expected values from Task 1's core, run by hand on the same column
    for pt, ps in ((0, 1013.0), (2, 600.0)):
        p = np.array(levels, float)
        want = layerwind.deck_winds(p, [np.array([u_of(l)]) for l in levels], [np.array([-1.0])] * len(levels),
                                    [np.array([l in cloudy]) for l in levels], np.array([ps]),
                                    layerwind.deck_bounds("ncep", np.array([ps])))
        for key in (850, 500, 250):
            for c in "uv":
                got, w = out[(key, c)].reshape(-1)[pt], want[(key, c)][0]
                assert (np.isnan(got) and np.isnan(w)) or got == pytest.approx(w, rel=1e-5), (pt, key, c)
    # at the 1013 hPa points 700 mb is the low deck's only cloudy level and 600 mb the mid deck's
    assert out[(850, "u")][0, 0] == pytest.approx(70.0)
    assert out[(500, "u")][0, 0] == pytest.approx(60.0)
    # at ps = 600 hPa the whole low deck (600..642) is underground: NaN, not 0
    assert np.isnan(out[(850, "u")][1, 0]) and out[(500, "u")][1, 0] == pytest.approx(60.0)


def test_gfs_read_wind_step_rejects_swapped_message():
    get = _gfs_fixture(lambda lev: 1.0, lambda lev: 1.0, [101300.0] * 4)
    idx = (get("x.idx").decode().replace("UGRD:850 mb", "@:850 mb").replace("VGRD:850 mb", "UGRD:850 mb")
           .replace("@:850 mb", "VGRD:850 mb"))                              # the idx swaps 850 mb u and v
    with pytest.raises(ValueError, match="expected . 850"):
        grib.read_wind_step(GFS, RUN, 6, lambda u, r=None: idx.encode() if u.endswith(".idx") else get(u, r))


def test_gfs_level_missing_its_ugrd_row_raises_keyerror():
    # the level list is fixed: a level lacking only UGRD must fail, not drop out
    with pytest.raises(KeyError, match="UGRD"):
        grib.wind_ranges(GFS, _ncep_idx(skip=(("UGRD", "700 mb"),)), 6)
    assert grib.GFS_WIND_LEVELS == GFS_LEVELS


def test_ecmwf_wind_fields_stay_float64_and_unchanged():
    for model in (AIFS, IFS):
        out = grib.read_wind_fields(model, RUN, 6, _ec_fixture(model))
        for lev in EC_LEVELS:
            a = out[(lev, "u")]
            assert a.dtype == np.float64 and a.shape == (2, 2)
            assert (a == lev / 10.0).all() and (out[(lev, "t")] == EC_T[lev]).all()
        assert out[("sfc", "sp")].dtype == np.float64


def _swap_get(get, idx_edit):
    def g(url, rng=None):
        return idx_edit(get(url, rng).decode()).encode() if url.endswith(".idx") else get(url, rng)
    return g


def test_gfs_wind_message_in_a_tcdc_slot_is_rejected():
    get = _gfs_fixture(lambda lev: 1.0, lambda lev: 1.0, [101300.0] * 4)
    tcdc, wind_u = _gfs_msg("tcdc", 850, [1.0] * 4), _gfs_msg("u", 850, [7.0] * 4)
    assert len(tcdc) == len(wind_u)                  # same size: offsets in the idx stay true

    def g(url, rng=None):
        b = get(url, rng)
        return b if url.endswith(".idx") else b.replace(tcdc, wind_u)   # a UGRD message in the TCDC 850 slot
    with pytest.raises(ValueError, match="expected tcdc 850"):
        grib.read_wind_step(GFS, RUN, 6, g)


def test_gfs_pres_that_is_not_the_surface_is_rejected():
    import eccodes
    get = _gfs_fixture(lambda lev: 1.0, lambda lev: 1.0, [101300.0] * 4)
    base_msg = _gfs_msg("pres", 0, [101300.0] * 4)
    h = eccodes.codes_new_from_message(base_msg)
    eccodes.codes_set(h, "typeOfFirstFixedSurface", 6)          # max wind: shortName pres, level 0, like the surface
    bad = eccodes.codes_get_message(h); eccodes.codes_release(h)
    assert len(bad) == len(base_msg)

    def g(url, rng=None):
        b = get(url, rng)
        return b if url.endswith(".idx") else b.replace(base_msg, bad)   # ranges merge touching messages
    with pytest.raises(ValueError, match="surface pressure"):
        grib.read_wind_step(GFS, RUN, 6, g)


# ---- throttled requests on the wind path (pipeline/throttle.py) ----------------

def _http_error(code, retry_after=None):
    import email.message
    import urllib.error
    h = email.message.Message()
    if retry_after is not None:
        h["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x/f.grib2", code, "err", h, None)


def _scripted_get(outcomes):
    """get() that raises / returns the scripted outcomes in order; records the calls."""
    calls = []

    def get(url, rng=None):
        calls.append((url, rng))
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o
    return get, calls


def test_throttle_429_then_success_after_one_sleep():
    from pipeline import throttle
    sleeps = []
    get, calls = _scripted_get([_http_error(429), b"ok"])
    assert throttle.retry_throttled(get, sleeps.append)("u", (0, 9)) == b"ok"
    assert sleeps == [2.0] and calls == [("u", (0, 9))] * 2


def test_throttle_honours_retry_after():
    from pipeline import throttle
    sleeps = []
    get, _ = _scripted_get([_http_error(429, "7"), _http_error(503, "Wed, 18 Nov 2026 12:00:00 GMT"), b"ok"])
    assert throttle.retry_throttled(get, sleeps.append)("u") == b"ok"
    assert sleeps == [7.0, 4.0]                    # a numeric Retry-After, then a date: the backoff
    get, _ = _scripted_get([_http_error(429, "900"), b"ok"])
    sleeps.clear()
    throttle.retry_throttled(get, sleeps.append)("u")
    assert sleeps == [throttle.CAP_S]              # capped


def test_throttle_503_retries_and_404_does_not():
    import urllib.error
    from pipeline import throttle
    sleeps = []
    get, _ = _scripted_get([_http_error(503), _http_error(503), b"ok"])
    assert throttle.retry_throttled(get, sleeps.append)("u") == b"ok" and sleeps == [2.0, 4.0]
    for code in (404, 403, 500):
        sleeps.clear()
        get, calls = _scripted_get([_http_error(code), b"never"])
        with pytest.raises(urllib.error.HTTPError) as ei:
            throttle.retry_throttled(get, sleeps.append)("u")
        assert ei.value.code == code and len(calls) == 1 and sleeps == []
    get, calls = _scripted_get([OSError("reset"), b"never"])
    with pytest.raises(OSError):
        throttle.retry_throttled(get, sleeps.append)("u")
    assert len(calls) == 1


def test_throttle_gives_up_after_the_cap_and_at_the_deadline():
    import urllib.error
    from pipeline import throttle
    sleeps = []
    get, calls = _scripted_get([_http_error(429)] * 10)
    with pytest.raises(urllib.error.HTTPError):
        throttle.retry_throttled(get, sleeps.append)("u")
    assert len(calls) == throttle.ATTEMPTS and sleeps == [2.0, 4.0, 8.0, 16.0]
    # with the budget clock: a wait that would end past the deadline is not made
    now = [0.0]

    def sleep(s):
        sleeps.append(s)
        now[0] += s
    sleeps.clear()
    get, calls = _scripted_get([_http_error(429)] * 10)
    with pytest.raises(urllib.error.HTTPError):
        throttle.retry_throttled(get, sleep, clock=lambda: now[0], deadline=5.0)("u")
    assert sleeps == [2.0] and len(calls) == 2      # 2 + 4 > 5: the third try is never made


def test_build_wind_survives_one_429_per_request(tmp_path):
    base = _ec_fixture(AIFS)
    seen, sleeps = set(), []

    def get(url, rng=None):
        k = (url, rng)
        if k not in seen:                              # every request is throttled once
            seen.add(k)
            raise _http_error(429)
        return base(url, rng)

    m = _small(AIFS, (0, 6))
    e = gb.build_wind(m, RUN, tmp_path, get, sleeps.append)
    assert e is not None and e["layer"] == "humidity"
    assert len(sleeps) == len(seen) and set(sleeps) == {2.0}
    assert any((tmp_path / "wind/2026110800").glob("*.icw"))
