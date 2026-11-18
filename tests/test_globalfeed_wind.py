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


def test_ecmwf_wind_ranges():
    rows = [_row("u", "sfc", None, 5, 5)]       # a 10 m-style row that must not be picked
    off = 1000
    for lev in (850, 500, 250):
        for p in "uv":
            rows.append(_row(p, "pl", lev, off, 50))
            off += 50
    r = grib.wind_ranges(AIFS, "\n".join(rows), 6)
    assert r[(850, "u")] == (1000, 1049) and r[(850, "v")] == (1050, 1099)
    assert r[(250, "v")] == (1250, 1299)
    assert len(r) == 6
    with pytest.raises(KeyError):
        grib.wind_ranges(IFS, "\n".join(rows[:-1]), 6)
    with pytest.raises(KeyError):
        grib.wind_ranges(IFS, "\n".join([_row("u", "sfc", None, 5, 5)]), 6)


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
    assert GFS.wind_layer == "cloud" and AIFS.wind_layer is None and IFS.wind_layer is None


def _wind_msg(comp, lev, val):
    import eccodes
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
    eccodes.codes_set_values(h, np.full(4, float(val)))
    b = eccodes.codes_get_message(h)
    eccodes.codes_release(h)
    return b


def _ifs_order_fixture(swap=False):
    """IFS lists v before u at each level; messages laid in that file order."""
    order = [(lev, c) for lev in (850, 500, 250) for c in "vu"]
    rows, buf = [], b""
    for i, (lev, c) in enumerate(order):
        mc = ("u" if c == "v" else "v") if (swap and (lev, c) == (500, "u")) else c
        m = _wind_msg(mc, lev, 10 * (i + 1))
        rows.append(_row(c, "pl", lev, len(buf), len(m)))
        buf += m
    idx = "\n".join(rows)

    def get(url, rng=None):
        if url.endswith(".index"):
            return idx.encode()
        return buf[rng[0]:rng[1] + 1]
    return get, order


def test_read_wind_step_ifs_order_maps_correctly():
    get, order = _ifs_order_fixture()
    out = grib.read_wind_step(IFS, RUN, 6, get)
    for i, (lev, c) in enumerate(order):
        assert out[(lev, c)][0, 0] == 10 * (i + 1)


def test_read_wind_step_rejects_swapped_message():
    get, _ = _ifs_order_fixture(swap=True)
    with pytest.raises(ValueError, match="expected u 500"):
        grib.read_wind_step(IFS, RUN, 6, get)


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
    get, order = _ifs_order_fixture()
    out = grib.read_wind_step(IFS, RUN, 6, get)
    for i, (lev, c) in enumerate(order):
        a = out[(lev, c)]
        assert a.dtype == np.float64 and a.shape == (2, 2)
        assert (a == float(10 * (i + 1))).all()
    # the same values the reader's own decode (eccodes + isclose-missing mask) gives, uncast
    import eccodes
    raw = _wind_msg("u", 850, 12.3456789)
    g = eccodes.codes_new_from_message(raw)
    want = np.where(np.isclose(eccodes.codes_get_values(g), eccodes.codes_get(g, "missingValue")), np.nan,
                    eccodes.codes_get_values(g))
    eccodes.codes_release(g)
    assert want.dtype == np.float64


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
