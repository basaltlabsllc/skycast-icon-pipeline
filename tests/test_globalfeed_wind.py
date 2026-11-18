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


def _ncep_idx():
    rows = ["1:0:d=2026110800:HGT:850 mb:6 hour fcst:",
            "2:100:d=2026110800:UGRD:850 mb:6 hour fcst:",
            "3:250:d=2026110800:VGRD:850 mb:6 hour fcst:",
            "4:400:d=2026110800:UGRD:500 mb:6 hour fcst:",
            "5:520:d=2026110800:VGRD:500 mb:6 hour fcst:",
            "6:700:d=2026110800:UGRD:250 mb:6 hour fcst:",
            "7:830:d=2026110800:VGRD:250 mb:6 hour fcst:",
            "8:960:d=2026110800:UGRD:250 mb:anl:"]
    return "\n".join(rows)


def test_ncep_wind_ranges():
    r = grib.wind_ranges(GFS, _ncep_idx(), 6)
    assert r[(850, "u")] == (100, 249) and r[(850, "v")] == (250, 399)
    assert r[(500, "u")] == (400, 519) and r[(500, "v")] == (520, 699)
    assert r[(250, "u")] == (700, 829) and r[(250, "v")] == (830, 959)
    with pytest.raises(KeyError):
        grib.wind_ranges(GFS, _ncep_idx(), 12)
    with pytest.raises(KeyError):
        grib.wind_ranges(GFS, "\n".join(_ncep_idx().split("\n")[:5]), 6)


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
