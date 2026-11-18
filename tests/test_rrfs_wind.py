import json
from datetime import datetime, timezone

import eccodes
import numpy as np
import pytest

import hrrr.grib as hg
import rrfs.build as rb
import rrfs.grib as rg
from pipeline import wind
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


# ---- idx rows and the ranged requests ----------------------------------------
def _idx(fh):
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [f"1:0:d=2026111618:HGT:2 mb:{want}:"]
    off = 100
    for lev in (250, 500, 850):
        for comp in ("UGRD", "VGRD"):
            rows.append(f"{len(rows) + 1}:{off}:d=2026111618:{comp}:{lev} mb:{want}:")
            off += 50 if comp == "UGRD" else 70
    rows.append(f"{len(rows) + 1}:{off}:d=2026111618:TMP:850 mb:{want}:")
    return "\n".join(rows)


def test_wind_rows_and_one_request_per_level():
    r = hg.wind_ranges(_idx(0), 0)
    assert r[(250, "u")] == (100, 149) and r[(250, "v")] == (150, 219)
    reqs = rg._requests(r)
    assert len(reqs) == 3 and all(len(k) == 2 for _, _, k in reqs)
    assert [(s, e) for s, e, _ in reqs] == [(r[(l, "u")][0], r[(l, "v")][1]) for l in (850, 500, 250)]


def test_requests_split_when_the_rows_do_not_touch():
    r = {(l, c): (10 * i, 10 * i + 4) for i, (l, c) in enumerate([(850, "u"), (850, "v"), (500, "u"), (500, "v"),
                                                                   (250, "u"), (250, "v")])}
    assert len(rg._requests(r)) == 6


# ---- a rotated-grid fixture ---------------------------------------------------
def _rot_msg(lev, comp, val, uv_relative=1, sample="rotated_ll_pl_grib2", pole=POLE):
    h = eccodes.codes_grib_new_from_samples(sample)
    keys = [("Ni", 2), ("Nj", 2), ("latitudeOfFirstGridPointInDegrees", 1.0),
            ("latitudeOfLastGridPointInDegrees", 0.0), ("longitudeOfFirstGridPointInDegrees", 0.0),
            ("longitudeOfLastGridPointInDegrees", 40.0), ("iDirectionIncrementInDegrees", 40.0),
            ("jDirectionIncrementInDegrees", 1.0), ("bitsPerValue", 24), ("discipline", 0),
            ("parameterCategory", 2), ("parameterNumber", 2 if comp == "u" else 3),
            ("typeOfFirstFixedSurface", 100), ("scaleFactorOfFirstFixedSurface", 0),
            ("scaledValueOfFirstFixedSurface", lev * 100)]
    if sample.startswith("rotated"):
        keys += [("latitudeOfSouthernPoleInDegrees", pole[0]), ("longitudeOfSouthernPoleInDegrees", pole[1])]
    for k, v in keys:
        eccodes.codes_set(h, k, v)
    if uv_relative:
        eccodes.codes_set(h, "uvRelativeToGrid", 1)
    eccodes.codes_set_values(h, np.full(4, float(val(lev, comp))))
    b = eccodes.codes_get_message(h)
    eccodes.codes_release(h)
    return b


def _wind_files(fh, val=lambda lev, comp: 10.0 if comp == "u" else 0.0, run=RUN, **kw):
    want = "anl" if fh == 0 else f"{fh} hour fcst"
    rows = [("HGT", "2 mb", want, _rot_msg(250, "u", lambda *a: 1.0, **kw))]
    for lev in (250, 500, 850):
        for comp in ("u", "v"):
            rows.append(("UGRD" if comp == "u" else "VGRD", f"{lev} mb", want, _rot_msg(lev, comp, val, **kw)))
    f = Files()
    f.add(rg.prs_url(run, fh), rows)
    return f


def _true_latlon(**kw):
    g = eccodes.codes_new_from_message(_rot_msg(250, "u", lambda *a: 0.0, **kw))
    try:
        lat = eccodes.codes_get_array(g, "latitudes")
        lon = eccodes.codes_get_array(g, "longitudes")
    finally:
        eccodes.codes_release(g)
    return lat, np.where(lon > 180, lon - 360, lon)


def test_read_wind_turns_with_the_messages_own_pole():
    f = _wind_files(0)
    d = rg.read_wind(RUN, 0, f.get)
    lat, lon = _true_latlon()
    assert np.allclose(d["lat"], lat) and np.allclose(d["lon"], lon)
    assert set(k for k in d if isinstance(k, tuple)) == {(l, c) for l in (850, 500, 250) for c in "uv"}
    ue, ve = rg.rotate_rotated_pole(np.full(4, 10.0), np.zeros(4), lat, lon, *POLE)
    assert np.allclose(d[(500, "u")], ue, atol=1e-4) and np.allclose(d[(500, "v")], ve, atol=1e-4)
    assert np.abs(ve).max() > 1.0             # the fixture does turn the wind
    # a different pole in the message turns it differently: the keys are read, not assumed
    d2 = rg.read_wind(RUN, 0, _wind_files(0, pole=(-50.0, 247.0)).get)
    assert not np.allclose(d2[(500, "v")], d[(500, "v")], atol=1e-3)


def test_read_wind_uv_relative_to_earth_is_left_alone():
    d = rg.read_wind(RUN, 0, _wind_files(0, uv_relative=0).get)
    assert np.allclose(d[(250, "u")], 10.0) and np.allclose(d[(250, "v")], 0.0)


def test_read_wind_refuses_an_unsupported_grid():
    with pytest.raises(ValueError, match="wind: unsupported grid regular_ll"):
        rg.read_wind(RUN, 0, _wind_files(0, sample="regular_ll_pl_grib2").get)


def test_read_wind_refuses_a_message_that_is_not_the_one_the_index_named():
    f = _wind_files(0)
    url = rg.prs_url(RUN, 0) + ".idx"
    f.idx[url] = (f.idx[url].replace(b"UGRD:250 mb", b"TMPX").replace(b"VGRD:250 mb", b"UGRD:250 mb")
                  .replace(b"TMPX", b"VGRD:250 mb"))
    with pytest.raises(ValueError, match="wind: expected"):
        rg.read_wind(RUN, 0, f.get)


def test_read_wind_makes_four_requests_an_hour():
    f = _wind_files(1)
    rg.read_wind(RUN, 1, f.get)
    assert len(f.calls) == 4 and f.calls[0][0].endswith(".idx") and all(r is not None for _, r in f.calls[1:])


def test_read_wind_lambert_uses_the_hrrr_turn_and_pins_the_wiring(monkeypatch):
    # the geometry call is replaced: LoV west of the cells, so they sit east of it
    f = _wind_files(0)
    lat = np.array([41.0, 41.0, 40.0, 40.0])
    lon = np.full(4, -80.0)
    monkeypatch.setattr(rg, "_wind_geometry", lambda g: (
        lat, lon, lambda u, v: hg.rotate_to_earth(u, v, lon, 262.5, 38.5)))
    d = rg.read_wind(RUN, 0, f.get)
    a = np.sin(np.radians(38.5)) * np.radians(17.5)
    assert (d[(250, "v")] < 0).all()
    assert np.allclose(d[(250, "u")], 10 * np.cos(a), atol=1e-4) and np.allclose(d[(250, "v")], -10 * np.sin(a), atol=1e-4)


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
    files = {fh: _wind_files(fh) for fh in (0, 1)}
    lim = rb.RateLimiter(1000)
    entry = rb.build_wind(RUN, tmp_path, rb.limited(_get_for(files), lim), lambda s: None, [0, 1], 2)
    assert entry == wind.manifest_entry(RUN)
    assert len(lim._stamps) == 8                      # 4 requests an hour, every one counted
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
