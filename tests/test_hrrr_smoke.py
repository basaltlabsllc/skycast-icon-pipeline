import zlib
from datetime import datetime, timezone

import numpy as np
import pytest

import hrrr.grib as hg
import hrrr.smoke as sm
from tests.gribfake import Files, msg

RUN = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
MASS = "8 m above ground"
COL = "entire atmosphere (considered as a single layer)"


def fcst(fh):
    return "anl" if fh == 0 else f"{fh} hour fcst"


def serve(files, fh, mass, aotk):
    rows = [("TMP", "2 m above ground", fcst(fh), msg([1, 2, 3, 4])),
            ("MASSDEN", "entire atmosphere", fcst(fh), msg([9, 9, 9, 9])),   # decoy
            ("MASSDEN", MASS, fcst(fh), msg(mass)),
            ("AOTK", COL, fcst(fh), msg(aotk)),
            ("PRES", "surface", fcst(fh), msg([1, 2, 3, 4]))]
    files.add(hg.file_url(RUN, fh), rows)


def idx_of(files, fh):
    return files.idx[hg.file_url(RUN, fh) + ".idx"].decode()


def test_smoke_ranges_pick_8m_massden_and_aotk():
    f = Files()
    serve(f, 6, [1, 2, 3, 4], [0.1] * 4)
    r = sm.smoke_ranges(idx_of(f, 6), 6)
    blob = f.blobs[hg.file_url(RUN, 6)]
    assert set(r) == {"MASSDEN", "AOTK"}
    rows = [ln.split(":") for ln in idx_of(f, 6).strip().split("\n")]
    assert r["MASSDEN"][0] == int(rows[2][1]) and r["MASSDEN"][1] == int(rows[3][1]) - 1
    assert r["AOTK"] == (int(rows[3][1]), int(rows[4][1]) - 1)
    assert blob[r["MASSDEN"][0]:r["MASSDEN"][0] + 4] == b"GRIB"


def test_smoke_ranges_f00_is_anl_and_missing_raises():
    f = Files()
    serve(f, 0, [1, 2, 3, 4], [0.1] * 4)
    assert sm.smoke_ranges(idx_of(f, 0), 0)
    with pytest.raises(KeyError):
        sm.smoke_ranges(idx_of(f, 0), 3)
    with pytest.raises(KeyError):
        sm.smoke_ranges("1:0:d=2026110912:TMP:2 m above ground:anl:\n", 0)


def test_read_smoke_units():
    f = Files()
    serve(f, 6, [1.61e-8, -1e-9, 0.0, 5e-9], [0.1, 0.2, -0.3, 0.4])
    out = sm.read_smoke(RUN, 6, f.get)
    assert out["pm25"][0] == pytest.approx(16.1, rel=1e-5)
    assert np.isnan(out["pm25"][1])                      # negative -> NaN
    assert out["pm25"][3] == pytest.approx(5.0, rel=1e-5)
    assert np.isnan(out["aod"][2])
    assert out["aod"][0] == pytest.approx(0.1, rel=1e-5)
    assert out["lat"].shape == out["lon"].shape == (4,)
    assert out["lon"].max() < 0                           # 250..251 east -> -110..-109


def test_lattice_means_cells_and_leaves_gaps_nan():
    lat = np.array([44.01, 43.99, 30.0])
    lon = np.array([-123.01, -122.99, -100.0])
    idx = sm.node_index(lat, lon)
    g = sm.lattice(np.array([10.0, 20.0, np.nan]), idx)
    assert g.shape == (273, 617) and g.dtype == np.float32
    r, c = round((44.0 - 20.0) / 0.125), round((-123.0 + 135.0) / 0.125)
    assert g[r, c] == pytest.approx(15.0)
    assert np.isnan(g[0, 0])
    assert np.isnan(g[round((30.0 - 20) / .125), round(35 / .125)])     # only a NaN cell there
    assert int(np.isfinite(g).sum()) == 1


def test_node_index_outside_is_minus_one():
    assert sm.node_index([10.0, 60.0, 40.0, 40.0], [-100.0, -100.0, -140.0, -50.0]).tolist() == [-1, -1, -1, -1]
    assert sm.node_index([20.0], [-135.0]).tolist() == [0]
    assert sm.node_index([54.0], [-58.0]).tolist() == [272 * 617 + 616]


# ---- EPA fixture, ported from src/airquality.js ------------------------------------
PM25_BP = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
           (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]
CATS = [50, 100, 150, 200, 300]


def us_aqi(pm):
    if pm <= 0:
        return 0
    for pl, ph, al, ah in PM25_BP:
        if pm <= ph:
            if pm < pl:
                return al
            return int(np.floor(al + (pm - pl) / (ph - pl) * (ah - al) + 0.5))
    return 500


def category(aqi):
    return next((i for i, m in enumerate(CATS) if aqi <= m), len(CATS))


def test_pm25_categories_survive_encoding():
    v = np.arange(0, 1000, 0.05)
    # float32 holds 35.4 as 35.4000015; round to 4 places the way a consumer's own AQI math would see it
    d = np.round(sm.decode_pm25(sm.encode_pm25(v)).astype(np.float64), 4)
    bad = [(float(a), float(b)) for a, b in zip(v, d) if category(us_aqi(a)) != category(us_aqi(float(b)))]
    assert bad == []


def test_pm25_breakpoints_decode_exactly():
    for x in (9.0, 35.4, 55.4, 125.4, 225.4, 325.4):
        assert float(sm.decode_pm25(sm.encode_pm25(np.array([x])))[0]) == pytest.approx(x, abs=1e-4)


def test_pm25_special_bytes():
    e = sm.encode_pm25(np.array([0.0, np.nan, 5000.0, 1000.0]))
    assert e.tolist() == [0, 255, 254, 254]
    d = sm.decode_pm25(np.array([0, 255, 254], dtype=np.uint8))
    assert d[0] == 0 and np.isnan(d[1]) and d[2] == pytest.approx(1000)
    assert sm.encode_pm25(np.array([9.0, 35.4])).tolist() == [90, 142]


def test_aod_log_bytes():
    e = sm.encode_aod(np.array([0.0005, 0.05, 1.0, 5.0, 9.0, np.nan]))
    assert e.tolist() == [0, 117, 206, 254, 254, 255]
    assert float(sm.decode_aod(np.array([117]))[0]) == pytest.approx(0.0497, rel=0.01)
    assert np.isnan(sm.decode_aod(np.array([255]))[0])


def test_smk_round_trip():
    pm = (np.arange(6, dtype=np.uint8).reshape(2, 3) * 40)
    ao = (np.arange(6, dtype=np.uint8).reshape(2, 3) + 7)
    raw = sm.smk_bytes(491234, pm, ao)
    assert raw[0] == 0x78
    h, a, b = sm.parse_smk(raw)
    assert h == 491234 and a.tolist() == pm.tolist() and b.tolist() == ao.tolist()
    d = zlib.decompress(raw)
    assert d == b"SMK1" + (491234).to_bytes(4, "little") + (2).to_bytes(2, "little") + (3).to_bytes(2, "little") + pm.tobytes() + ao.tobytes()
