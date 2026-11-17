import datetime as dt
import numpy as np
import pytest
from pipeline import wind
from pipeline.site import wind_path


def test_group_names():
    g = wind.all_groups()
    assert len(g) == 162 and len(set(g)) == 162
    assert wind.group_name(30, -120) == "N30W120"
    assert wind.group_name(-10, 180) == "S10W180"
    with pytest.raises(ValueError):
        wind.group_name(40, -120)
    assert wind.group_name(-10, 20) == "S10E020"
    assert wind.parse_group("S90W180") == (-90, -180)
    assert wind.parse_group("N70E160") == (70, 160)


def test_group_of():
    assert wind.group_of(45, -110) == "N30W120"
    assert wind.group_of(90, 0) == "N70E000"
    assert wind.group_of(0, 180) == "S10W180"


def test_lattice_box_mean():
    lat = np.arange(-90, 90.001, 0.25)
    lon = np.arange(-180, 180, 0.25)
    field = np.tile(lon, (lat.size, 1))
    lat_t = wind.lattice_from_regular(lat, lon, field)
    assert lat_t.shape == (181, 360)
    assert abs(lat_t[90, 190] - 10.0) < 1e-9          # node (0, 10)
    band = np.zeros((lat.size, lon.size))
    band[(lat >= 44.5) & (lat <= 45.5), :] = 1.0
    lt = wind.lattice_from_regular(lat, lon, band)
    assert np.allclose(lt[135], 1.0)
    assert np.allclose(lt[133], 0.0)


def test_lattice_box_mean_wraps_and_descending_lat():
    lat = np.arange(90, -90.001, -0.5)
    lon = np.arange(0, 360, 0.5)                      # 0..360 convention
    field = np.ones((lat.size, lon.size))
    lt = wind.lattice_from_regular(lat, lon, field)
    assert np.allclose(lt, 1.0)                       # incl. lon -180 and the poles


def test_lattice_from_points_nearest_node():
    lt = wind.lattice_from_points(np.array([40.4, 40.6]), np.array([-100.2, -100.2]), np.array([2.0, 6.0]))
    assert lt.shape == (181, 360)
    assert lt[130, 80] == 2.0 and lt[131, 80] == 6.0
    assert np.isnan(lt[140, 80])


def test_round_trip():
    vals = np.array([-127, 0, 0.4, 0.6, 200, np.nan])
    lt = {(850, "u"): np.full((181, 360), np.nan)}
    cube = np.full((1, 3, 2, 21, 21), np.nan)
    cube[0, 0, 0, 0, :6] = vals
    from_cube = np.where(np.isnan(cube), wind.MISSING, np.clip(np.rint(cube), -127, 127)).astype(np.int8)
    blob = wind.encode_icw1(40, -120, [100, 101], np.concatenate([from_cube, from_cube]))
    out = wind.decode_icw1(blob)
    assert (out["sw_lat"], out["sw_lon"], out["cols"], out["rows"]) == (40, -120, 21, 21)
    assert out["step_hours"] == [100, 101] and out["levels"] == [850, 500, 250]
    assert out["cube"][0, 0, 0, 0, :6].tolist() == [-127, 0, 0, 1, 127, -128]


def test_dateline_edge():
    lt = {}
    base = np.full((181, 360), np.nan)
    base[90, 0] = 7.0                                  # lat 0, lon -180
    for lev in wind.LEVELS:
        lt[(lev, "u")] = base
        lt[(lev, "v")] = base
    cube = wind.group_cube([lt], "S10E160")
    assert cube.dtype == np.int8 and cube.shape == (1, 3, 2, 21, 21)
    assert cube[0, 0, 0, 10, 20] == 7
    assert cube[0, 0, 0, 10, 19] == wind.MISSING


def test_write_wind_skips_empty_groups(tmp_path):
    base = np.full((181, 360), np.nan)
    base[120:141, :] = 3.0                             # lat 30..50
    lt = {(lev, c): base for lev in wind.LEVELS for c in "uv"}
    run = dt.datetime(2026, 11, 17, 0, tzinfo=dt.timezone.utc)
    n = wind.write_wind(tmp_path, run, [100], [lt])
    files = sorted(p.name for p in (tmp_path / "wind" / "2026111700").iterdir())
    want = sorted(wind.group_name(la, lo) + ".icw" for la in (10, 30, 50) for lo in range(-180, 180, 20)
                  if la + 20 >= 30 and la <= 50)
    assert files == want
    assert n == sum(p.stat().st_size for p in (tmp_path / "wind" / "2026111700").iterdir())
    assert wind_path(run) == "wind/2026111700"
    assert wind.manifest_entry(run) == {"format": "ICW1", "run": "2026-11-17T00Z",
                                        "runMs": int(run.timestamp() * 1000),
                                        "tiles": "wind/2026111700", "levels": [850, 500, 250]}
