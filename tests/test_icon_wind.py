import datetime as dt
import json
from pathlib import Path
import numpy as np
import pytest
import run_build
from pipeline import dwd, wind
from pipeline.squares import NX, NY

UTC = dt.timezone.utc
RUN = dt.datetime(2026, 9, 27, 0, tzinfo=UTC)


def test_wind_files_names_every_level_of_every_step():
    files = dwd.wind_files(RUN, [0, 1])
    assert "icon_global_icosahedral_pressure-level_2026092700_000_850_U.grib2.bz2" in files["U"]
    assert "icon_global_icosahedral_pressure-level_2026092700_001_250_V.grib2.bz2" in files["V"]
    assert len(files["U"]) == len(files["V"]) == 2 * 3
    assert dwd.WIND_LEVELS == (850, 500, 250)


def _listing_without(skip):
    def listing(hh, field):
        names = dwd.wind_files(RUN, [0, 1])[field]
        return "".join(f'<a href="{n}">x</a>\n' for n in names if n != skip)
    return listing


def test_missing_wind_files_reports_the_unlisted_250_hpa_v_file():
    gone = dwd.pressure_file_name(RUN, 1, 250, "V")
    assert dwd.missing_wind_files(_listing_without(gone), RUN, [0, 1]) == [gone]
    assert dwd.missing_wind_files(_listing_without(None), RUN, [0, 1]) == []


def _deps(tmp_path, download=None, regrid_values=None, listed=True):
    sleeps = []
    downloads = []

    def dl(url, dst):
        downloads.append(url)
        Path(dst).write_bytes(b"fake")

    def rv(path, kit, work):
        level, field = Path(path).name.split(".")[0].split("_")[-2:]
        value = {"U": 3.0, "V": -4.0}[field] + int(level) / 1000.0
        return np.full((NY, NX), value, dtype=np.float32)

    deps = run_build.Deps(listing=_listing_without(None) if listed else (lambda hh, f: ""),
                          fetch_text=None, download=download or dl, regrid=None,
                          ensure_kit=None, now=None, sleep=sleeps.append,
                          regrid_values=regrid_values or rv, validate=None)
    return deps, downloads, sleeps


def test_a_build_writes_icw_groups_and_the_manifest_entry(tmp_path):
    deps, downloads, _ = _deps(tmp_path)
    (tmp_path / "work").mkdir()
    entry = run_build.build_wind(tmp_path / "blob", tmp_path / "work", deps, Path("/kit"), RUN, [0, 1], 2)
    assert entry == wind.manifest_entry(RUN)
    assert len(downloads) == 12
    assert downloads[0].startswith("https://opendata.dwd.de/weather/nwp/icon/grib/00/u/")
    f = tmp_path / "blob" / "wind" / "2026092700" / "N30W100.icw"
    d = wind.decode_icw1(f.read_bytes())
    base = int(RUN.timestamp()) // 3600
    assert d["step_hours"] == [base, base + 1]
    # deck 0 = 850 hPa: u 3.85 -> 4, v -4 + .85 -> -3; deck 2 = 250: 3.25 -> 3, -3.75 -> -4
    assert (d["cube"][0, 0, 0] == 4).all() and (d["cube"][0, 0, 1] == -3).all()
    assert (d["cube"][1, 2, 0] == 3).all() and (d["cube"][1, 2, 1] == -4).all()
    assert list((tmp_path / "work").iterdir()) == []       # every file deleted after use


def test_a_failing_download_returns_none_and_leaves_no_wind_dir(tmp_path, capsys):
    deps, _, _ = _deps(tmp_path)
    ok = deps.download
    count = {"n": 0}

    def flaky(url, dst):
        count["n"] += 1
        if count["n"] == 5:
            raise OSError("boom")
        ok(url, dst)

    deps = deps._replace(download=flaky)
    (tmp_path / "work").mkdir()
    assert run_build.build_wind(tmp_path / "blob", tmp_path / "work", deps, Path("/kit"), RUN, [0, 1], 2) is None
    assert not (tmp_path / "blob" / "wind").exists()
    assert "::warning::wind failed" in capsys.readouterr().out
    assert list((tmp_path / "work").iterdir()) == []


def test_unlisted_wind_waits_then_publishes_without(tmp_path, capsys):
    deps, downloads, sleeps = _deps(tmp_path, listed=False)
    (tmp_path / "work").mkdir()
    assert run_build.build_wind(tmp_path / "blob", tmp_path / "work", deps, Path("/kit"), RUN, [0], 2) is None
    assert sum(sleeps) == run_build.WIND_WAIT_S and downloads == []
    assert "::warning::wind:" in capsys.readouterr().out


def test_over_budget_drops_the_wind(tmp_path, capsys):
    deps, _, _ = _deps(tmp_path)
    ticks = iter(range(0, 10000, 100))
    (tmp_path / "work").mkdir()
    out = run_build.build_wind(tmp_path / "blob", tmp_path / "work", deps, Path("/kit"), RUN, [0, 1], 1,
                               budget_s=250, clock=lambda: next(ticks))
    assert out is None and not (tmp_path / "blob" / "wind").exists()
    assert "::warning::wind: over budget" in capsys.readouterr().out
