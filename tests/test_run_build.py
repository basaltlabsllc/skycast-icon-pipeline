import datetime as dt
import json
import numpy as np
from pathlib import Path
from pipeline.blocks import decode_blocks
from pipeline.dwd import STEPS, FIELDS, TAIL_STEPS, THICK_FIELDS, file_name, _NAME
from pipeline.encode import decode_square, quantize
from pipeline.history import save_run_head, hour_of, HEAD_STEPS
from pipeline.squares import NX, NY
import pytest
import run_build

# A full build publishes every square on the globe; these tests use nine.
NINE = ["N35W120", "N35W125", "N35W130", "N40W120", "N40W125", "N40W130", "N45W120", "N45W125", "N45W130"]


@pytest.fixture(autouse=True)
def _nine_squares(monkeypatch):
    monkeypatch.setattr(run_build, "site_squares", lambda: list(NINE))

UTC = dt.timezone.utc
RUN = dt.datetime(2026, 9, 27, 0, tzinfo=UTC)


def _listing(run, field, steps=STEPS):
    return "".join(f'<a href="{file_name(run, s, field)}">x</a>\n' for s in steps)


def _no_columns(paths, kit, work, stride):
    raise AssertionError("these listings carry no column files: no column build")


# Cloud ice and water in kg/m² as the fake thickness files hold them: 27 g/m²
# of ice and 100 g/m² of water, bytes 99 and 138 (tests/test_thick.py).
THICK_KG = {"TQI_DIA": 0.027, "TQC_DIA": 0.1}


def _fake_deps(tmp_path, world_value=lambda step, field: 0, prev_manifest=None,
               thick_value=lambda step, field: THICK_KG[field]):
    calls = {"downloads": [], "regrids": [], "regrids_values": [], "sleeps": []}

    def listing(hh, field):
        return _listing(RUN, field) if hh == "00" else ""

    def fetch_text(url):
        assert url.endswith("manifest.json")
        if prev_manifest is None:
            raise OSError("404")
        return json.dumps(prev_manifest)

    def download(url, dst):
        calls["downloads"].append(url)
        Path(dst).write_bytes(b"fake")

    def regrid(grib_bz2, kit, work, scale=1.0):
        name = Path(grib_bz2).name
        step = int(name.split("_")[-2]); field = name.split("_")[-1].split(".")[0]
        calls["regrids"].append((step, field))
        return np.full((NY, NX), world_value(step, field), dtype=np.uint8)

    def regrid_values(grib_bz2, kit, work):
        _, step, field = _NAME.fullmatch(Path(grib_bz2).name).groups()
        step = int(step)
        calls["regrids_values"].append((step, field))
        return np.full((NY, NX), thick_value(step, field), dtype=np.float32)

    return run_build.Deps(listing=listing, fetch_text=fetch_text, download=download,
                          regrid=regrid, ensure_kit=lambda work: Path("/kit"),
                          now=lambda: dt.datetime(2026, 9, 27, 3, tzinfo=UTC),
                          lattice_levels=_no_columns, sleep=lambda s: calls["sleeps"].append(s),
                          regrid_values=regrid_values, validate=lambda world: []), calls


WHOLE = {"columns": {"x": 1}, "thick": {"x": 1}, "blocks": "blocks/x", "tail": {"x": 1}}


def _clear_build(tmp_path):
    deps, _ = _fake_deps(tmp_path, world_value=lambda s, f: 10)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=list(range(0, 79)), squares=["N40W125"])
    return rc, tmp_path / "site", tmp_path / "blob"


def _blob_files(blob, rel):
    root = blob / rel
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_clear_night_goes_to_blob_as_one_pack_and_the_blob_manifest_names_only_it(tmp_path):
    rc, site, blob = _clear_build(tmp_path)
    assert rc == run_build.BUILT
    cn = json.loads((blob / "manifest.json").read_text())["clearNight"]
    assert cn["format"] == "CLN2" and cn["keys"] == ["N30W140"] and cn["nights"]
    assert "tiles" not in cn
    assert cn["pack"].startswith("clearnight/2026092700/pack-")
    # Two Blob puts a run: the pack and its index; the per-file tiles are gone.
    name = cn["pack"].rsplit("/", 1)[1]
    assert _blob_files(blob, "clearnight/2026092700") == [f"{name}/all.bin", f"{name}/index.json"]
    idx = json.loads((blob / cn["pack"] / "index.json").read_text())
    assert f"{cn['nights'][0]}_N30W140.cln" in idx["names"]
    assert len(idx["names"]) == len(cn["nights"]) * len(cn["keys"])


def test_a_clear_night_pack_that_fails_falls_back_to_the_per_file_tiles(tmp_path, monkeypatch):
    real = run_build.pack.try_pack
    monkeypatch.setattr(run_build.pack, "try_pack", lambda root, rel, *a, **k: None if rel.startswith("clearnight/") else real(root, rel, *a, **k))
    rc, site, blob = _clear_build(tmp_path)
    assert rc == run_build.BUILT
    cn = json.loads((blob / "manifest.json").read_text())["clearNight"]
    assert cn["tiles"] == "clearnight/2026092700" and "pack" not in cn
    assert (blob / cn["tiles"] / f"{cn['nights'][0]}_N30W140.cln").exists()
    assert "clearNight" not in json.loads((site / "manifest.json").read_text())
    assert not (site / "clearnight").exists()


def test_a_failed_clear_night_never_fails_the_build(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no")
    monkeypatch.setattr(run_build.clearnight, "build_clearnight", boom)
    rc, _, blob = _clear_build(tmp_path)
    assert rc == run_build.BUILT
    assert "clearNight" not in json.loads((blob / "manifest.json").read_text())
    assert not (blob / "clearnight").exists()


def test_early_exit_when_the_site_already_has_this_run(tmp_path, capsys):
    deps, calls = _fake_deps(tmp_path, prev_manifest={"run": "2026-09-27T00Z", "complete": True, **WHOLE})
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=None, squares=None)
    assert rc == run_build.NOTHING_TO_DO
    assert calls["downloads"] == []
    assert not (tmp_path / "site").exists()


def test_a_build_writes_squares_manifest_and_state(tmp_path):
    deps, calls = _fake_deps(tmp_path, world_value=lambda step, field: min(100, step + FIELDS.index(field)))
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=[0, 1, 2, 3, 4, 5, 6], squares=["N40W125"])
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert m["run"] == "2026-09-27T00Z" and m["squares"] == ["N40W125"]
    assert m["stepsMs"] == [(hour_of(RUN) + s) * 3600000 for s in range(7)]
    assert m["historySteps"] == 0
    sq = decode_square((tmp_path / "site" / "tiles" / "2026092700" / "N40W125.icl").read_bytes())
    assert sq["step_hours"] == [hour_of(RUN) + s for s in range(7)]
    assert sq["cube"].shape == (7, 4, 41, 41)
    assert sq["cube"][6, 2, 0, 0] == 6 + 2                # step 6, field CLCH
    assert sorted(calls["regrids"])[:2] == [(0, "CLCH"), (0, "CLCL")]
    assert sum(1 for u in calls["downloads"] if any(f"_{f}." in u for f in FIELDS)) == 7 * 4
    # The cover and the thickness, and nothing else (no column files are listed).
    assert len(calls["downloads"]) == 7 * (len(FIELDS) + len(THICK_FIELDS))
    # The run's hours 0-5 are saved for the next build's history.
    assert (tmp_path / "state" / "history" / "2026092700.npz").exists()


def test_history_from_state_is_prepended(tmp_path):
    prev = RUN - dt.timedelta(hours=6)
    save_run_head(tmp_path / "state", prev, np.full((6, 4, NY, NX), 42, dtype=np.uint8))
    deps, _ = _fake_deps(tmp_path)
    run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                    deps=deps, pages_base="https://x", steps=[0, 1], squares=["N00E000"])
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert m["historySteps"] == 6
    assert m["stepsMs"][0] == hour_of(prev) * 3600000
    assert m["historyRuns"] == []                      # no publication recorded yet
    sq = decode_square((tmp_path / "site" / "tiles" / "2026092700" / "N00E000.icl").read_bytes())
    assert sq["cube"][0, 0, 0, 0] == 42 and sq["cube"][6, 0, 0, 0] == 0


def test_no_complete_run_is_nothing_to_do(tmp_path):
    deps, _ = _fake_deps(tmp_path)
    deps = deps._replace(listing=lambda hh, field: "")
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=None, squares=None)
    assert rc == run_build.NOTHING_TO_DO


def test_a_failed_file_fails_the_build_not_the_site(tmp_path):
    deps, _ = _fake_deps(tmp_path)
    def bad_download(url, dst):
        raise OSError("boom")
    deps = deps._replace(download=bad_download)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=[0], squares=["N00E000"])
    assert rc == run_build.FAILED
    assert not (tmp_path / "site" / "manifest.json").exists()


class _Resp:
    """A urlopen() result: headers plus a body served in one piece."""
    def __init__(self, body, length):
        self.headers = {"Content-Length": str(length)}
        self._body = body
    def read(self, n):
        b, self._body = self._body[:n], self._body[n:]
        return b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_download_retries_a_short_body_then_raises(tmp_path):
    bodies = [b"12345", b"12345", b"1234567890"]          # short, short, whole
    opens = []
    def urlopen(url, timeout):
        opens.append(url)
        return _Resp(bodies[len(opens) - 1], 10)
    run_build._download("u", tmp_path / "f", urlopen=urlopen, sleep=lambda s: None)
    assert len(opens) == 3 and (tmp_path / "f").read_bytes() == b"1234567890"
    opens.clear()
    import pytest
    with pytest.raises(OSError, match="got 2 of 10 bytes"):
        run_build._download("u", tmp_path / "g", urlopen=lambda url, timeout: _Resp(b"12", 10), sleep=lambda s: None)


def test_download_retries_a_dropped_connection_but_not_a_404(tmp_path):
    import http.client, urllib.error, pytest
    opens, naps = [], []
    def urlopen(url, timeout):
        opens.append(url)
        if len(opens) < 3:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return _Resp(b"1234567890", 10)
    run_build._download("u", tmp_path / "f", urlopen=urlopen, sleep=naps.append)
    assert len(opens) == 3 and naps == [2, 4] and (tmp_path / "f").read_bytes() == b"1234567890"
    opens.clear()
    def gone(url, timeout):
        opens.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        run_build._download("u", tmp_path / "g", urlopen=gone, sleep=naps.append)
    assert len(opens) == 1
    with pytest.raises(OSError, match="RemoteDisconnected"):
        run_build._download("u", tmp_path / "h", sleep=lambda s: None,
                            urlopen=lambda url, timeout: (_ for _ in ()).throw(http.client.RemoteDisconnected("x")))


def test_a_full_build_publishes_every_square_and_reads_no_land_mask(tmp_path):
    # Erik 2026-10-01: DWD ICON covers the whole globe, so an island or a pin
    # at sea must work the same — no land filter.
    deps, calls = _fake_deps(tmp_path)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=[0], squares=None)
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert m["squares"] == sorted(NINE)
    assert not any("FR_LAND" in u for u in calls["downloads"])


def test_site_squares_is_the_whole_globe(monkeypatch):
    from pipeline.squares import all_squares
    monkeypatch.undo()                      # the real one, not the nine above
    assert run_build.site_squares() == all_squares()


def test_values_are_published_in_2_percent_steps(tmp_path):
    deps, _ = _fake_deps(tmp_path, world_value=lambda step, field: 13)
    run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                    deps=deps, pages_base=None, steps=[0], squares=None)
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    sq = decode_square((tmp_path / "site" / m["tiles"] / "N40W125.icl").read_bytes())
    assert int(sq["cube"].max()) == 14 and int(sq["cube"].min()) == 14


def test_over_the_size_budget_the_squares_are_rewritten_in_4_percent_steps(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_build, "SITE_BUDGET_BYTES", 1)
    deps, _ = _fake_deps(tmp_path, world_value=lambda step, field: 13)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=[0], squares=None)
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    sq = decode_square((tmp_path / "site" / m["tiles"] / "N40W125.icl").read_bytes())
    assert int(sq["cube"].max()) == 12
    assert "::warning::" in capsys.readouterr().out


def _blocks_build(tmp_path, steps, value, squares=None):
    deps, _ = _fake_deps(tmp_path, world_value=value)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=steps, squares=squares)
    assert rc == run_build.BUILT
    return tmp_path / "site", tmp_path / "blob"


def test_blocks_go_to_blob_with_the_same_values_as_the_squares(tmp_path):
    site, blob = _blocks_build(tmp_path, list(range(7)), lambda s, f: min(100, s + FIELDS.index(f)),
                               squares=["N40W125"])
    feed = json.loads((blob / "manifest.json").read_text())
    assert feed["blocks"] == "blocks/2026092700"
    m = json.loads((site / "manifest.json").read_text())
    icl = decode_square((site / m["tiles"] / "N40W125.icl").read_bytes())
    icb = decode_blocks((blob / "blocks/2026092700/N40W125.icb").read_bytes(), 4)
    assert icb["magic"] == b"ICL2"
    assert icb["step_hours"] == icl["step_hours"]
    assert np.array_equal(icb["cube"], icl["cube"])


def test_pages_gets_no_blocks(tmp_path):
    site, _ = _blocks_build(tmp_path, [0, 1], lambda s, f: 0, squares=["N40W125"])
    assert not (site / "blocks").exists()
    assert json.loads((site / "manifest.json").read_text())["blocks"] is None


def test_blocks_use_the_2_percent_step_even_over_the_pages_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(run_build, "SITE_BUDGET_BYTES", 1)
    site, blob = _blocks_build(tmp_path, [0], lambda s, f: 13)
    m = json.loads((site / "manifest.json").read_text())
    icl = decode_square((site / m["tiles"] / "N40W125.icl").read_bytes())
    icb = decode_blocks((blob / "blocks/2026092700/N40W125.icb").read_bytes(), 4)
    assert int(icl["cube"].max()) == int(quantize(np.full(1, 13, np.uint8), 4)[0]) == 12
    assert int(icb["cube"].max()) == int(icb["cube"].min()) == 14


def test_a_partial_live_manifest_does_not_stop_the_full_build(tmp_path):
    deps, calls = _fake_deps(tmp_path, prev_manifest={"run": "2026-09-27T00Z", "complete": False})
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=[0], squares=["N00E000"])
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert m["complete"] is False                      # a test build never claims to be whole


def test_a_full_build_marks_the_manifest_complete(tmp_path):
    deps, _ = _fake_deps(tmp_path)
    run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                    deps=deps, pages_base=None, steps=None, squares=None)
    assert json.loads((tmp_path / "site" / "manifest.json").read_text())["complete"] is True


def _fake_deps_with_tail(tmp_path, tail_fails=False):
    deps, calls = _fake_deps(tmp_path, world_value=lambda step, field: 90 if step >= 123 else 10)

    def listing(hh, field):
        if hh != "00":
            return ""
        return _listing(RUN, field, steps=list(STEPS) + list(TAIL_STEPS))

    def download(url, dst):
        if tail_fails and "_123_" in url:
            raise OSError("tail down")
        calls["downloads"].append(url)
        Path(dst).write_bytes(b"fake")
    return deps._replace(listing=listing, download=download), calls


def test_a_full_build_publishes_the_tail(tmp_path):
    deps, calls = _fake_deps_with_tail(tmp_path)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=None, squares=None, workers=2)
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    base = int(RUN.timestamp()) // 3600
    assert m["tail"]["run"] == "2026-09-27T00Z"
    assert m["tail"]["tiles"] == "tail/2026092700"
    assert m["tail"]["stepsMs"] == [(base + s) * 3600 * 1000 for s in TAIL_STEPS]
    sq = decode_square((tmp_path / "site" / "tail" / "2026092700" / "N40W125.icl").read_bytes())
    assert sq["step_hours"] == [base + s for s in TAIL_STEPS]
    assert int(sq["cube"][0, 0, 0, 0]) == 90
    assert sum(1 for u in calls["downloads"] if "_123_" in u) == len(FIELDS)


def test_a_tail_failure_never_fails_the_build(tmp_path):
    deps, _ = _fake_deps_with_tail(tmp_path, tail_fails=True)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=None, squares=None, workers=2)
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert m["tail"] is None
    assert (tmp_path / "site" / "tiles" / "2026092700" / "N40W125.icl").exists()


def test_a_test_build_makes_no_tail(tmp_path):
    deps, calls = _fake_deps_with_tail(tmp_path)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=[0, 1], squares=["N40W125"], workers=2)
    assert rc == run_build.BUILT
    assert json.loads((tmp_path / "site" / "manifest.json").read_text())["tail"] is None
    assert not any("_123_" in u for u in calls["downloads"])


# ---- Cloud thickness ----------------------------------------------------------
# The fake files hold THICK_KG everywhere: ice byte 99, water byte 138
# (pinned by tests/test_thick.py).
ICE_BYTE, WATER_BYTE = 99, 138


def _thick_build(tmp_path, deps, steps, squares=("N40W125",)):
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=list(steps), squares=list(squares), workers=2)
    assert rc == run_build.BUILT
    return tmp_path / "site", tmp_path / "blob"


def _icl_of(site, name="N40W125"):
    m = json.loads((site / "manifest.json").read_text())
    return decode_square((site / m["tiles"] / f"{name}.icl").read_bytes())


def _ict_of(blob, name="N40W125"):
    return decode_blocks((blob / "thick" / "2026092700" / f"{name}.ict").read_bytes(), len(THICK_FIELDS))


def test_thickness_goes_to_blob_by_the_squares_hours(tmp_path):
    deps, calls = _fake_deps(tmp_path)
    site, blob = _thick_build(tmp_path, deps, range(7))
    feed = json.loads((blob / "manifest.json").read_text())
    assert _is_pack(feed["thick"].pop("pack"), "thick/2026092700")
    assert feed["thick"] == {"format": "ICT1", "run": "2026-09-27T00Z", "runMs": int(RUN.timestamp() * 1000),
                             "tiles": "thick/2026092700"}
    ict = _ict_of(blob)
    assert ict["magic"] == b"ICT1"
    assert ict["step_hours"] == _icl_of(site)["step_hours"] == [hour_of(RUN) + s for s in range(7)]
    assert ict["cube"].shape == (7, 2, 41, 41)
    assert (ict["cube"][:, 0] == ICE_BYTE).all() and (ict["cube"][:, 1] == WATER_BYTE).all()
    # Blob only: Pages carries no thickness and its manifest names none.
    assert not (site / "thick").exists()
    assert json.loads((site / "manifest.json").read_text())["thick"] is None
    assert sorted(calls["regrids_values"]) == sorted((s, f) for s in range(7) for f in THICK_FIELDS)


def test_hours_before_the_thickness_history_are_missing(tmp_path):
    prev = RUN - dt.timedelta(hours=6)
    save_run_head(tmp_path / "state", prev, np.full((6, 4, NY, NX), 42, dtype=np.uint8))
    deps, _ = _fake_deps(tmp_path)
    site, blob = _thick_build(tmp_path, deps, [0, 1])
    ict = _ict_of(blob)
    # The cover's history reaches back six hours; the thickness has none yet.
    assert ict["step_hours"] == _icl_of(site)["step_hours"] == [hour_of(prev) + i for i in range(8)]
    assert (ict["cube"][:6] == 255).all()
    assert (ict["cube"][6:, 0] == ICE_BYTE).all() and (ict["cube"][6:, 1] == WATER_BYTE).all()


def test_thickness_history_comes_from_state(tmp_path):
    prev = RUN - dt.timedelta(hours=6)
    save_run_head(tmp_path / "state", prev, np.full((6, 4, NY, NX), 42, dtype=np.uint8))
    save_run_head(tmp_path / "state", prev, np.full((6, 2, NY, NX), 50, dtype=np.uint8), sub="thick")
    deps, _ = _fake_deps(tmp_path)
    site, blob = _thick_build(tmp_path, deps, [0, 1])
    ict = _ict_of(blob)
    assert ict["step_hours"] == _icl_of(site)["step_hours"]
    assert (ict["cube"][:6] == 50).all()
    assert (ict["cube"][6:, 0] == ICE_BYTE).all() and (ict["cube"][6:, 1] == WATER_BYTE).all()


def test_a_run_saves_its_thickness_head(tmp_path):
    deps, _ = _fake_deps(tmp_path)
    _thick_build(tmp_path, deps, range(6))
    with np.load(tmp_path / "state" / "thick" / "2026092700.npz") as z:
        head = z["head"]
    assert head.shape == (6, 2, NY, NX) and head.dtype == np.uint8
    assert (head[:, 0] == ICE_BYTE).all() and (head[:, 1] == WATER_BYTE).all()


def test_unlisted_thickness_waits_then_publishes_without(tmp_path, capsys):
    deps, calls = _fake_deps(tmp_path)
    listed = deps.listing
    deps = deps._replace(listing=lambda hh, field: "" if field in THICK_FIELDS else listed(hh, field))
    site, blob = _thick_build(tmp_path, deps, [0, 1])
    # The fake listing carries no column files either, so the columns wait
    # their own 15 minutes after the thickness's.
    assert sum(calls["sleeps"]) == run_build.THICK_WAIT_S + run_build.WIND_WAIT_S + run_build.COLUMN_WAIT_S
    assert calls["regrids_values"] == []
    assert json.loads((blob / "manifest.json").read_text())["thick"] is None
    assert not (blob / "thick").exists()
    assert (blob / "blocks" / "2026092700" / "N40W125.icb").exists()
    assert "::warning::thickness:" in capsys.readouterr().out


def test_a_thickness_failure_never_fails_the_build(tmp_path, capsys):
    def thick_value(step, field):
        if (step, field) == (3, "TQC_DIA"):
            raise OSError("cdo died")
        return THICK_KG[field]
    deps, _ = _fake_deps(tmp_path, thick_value=thick_value)
    site, blob = _thick_build(tmp_path, deps, range(7))
    assert json.loads((blob / "manifest.json").read_text())["thick"] is None
    assert not (blob / "thick").exists()
    assert not (tmp_path / "state" / "thick").exists()
    assert (site / "tiles" / "2026092700" / "N40W125.icl").exists()
    assert (blob / "blocks" / "2026092700" / "N40W125.icb").exists()
    assert "::warning::thickness failed" in capsys.readouterr().out


# ---- The column feed --------------------------------------------------------
import re
from pipeline import columns, history
from pipeline.columns import decode_columns, dequantize, STATIC
from pipeline.dwd import COLUMN_PLEVELS, COLUMN_MLEVELS, COLUMN_QC_MLEVELS, column_files, column_qc_files

# Fake DWD values in DWD's units, chosen so every app-unit number below is
# computed from them, never typed in.
HSURF_M = 1117.0
# Half levels above ground, as DWD's measured: full 117 → 165 m, 118 → 95.5 m, 119 → 42 m.
HHL_AGL = {117: 200.0, 118: 130.0, 119: 61.0, 120: 23.0}
# Higher half levels for the cloud water, 300 m apart (ICON's are ~300 m apart
# at 3 km), and 121 on the ground. The fake GH (8000 − 7·hPa m) puts 1000 hPa
# at 1000 m, BELOW this 1117 m ground, and 700 hPa at 3100 m, inside.
HHL_AGL.update({lv: 200.0 + 300.0 * (117 - lv) for lv in range(1, 117)})
HHL_AGL[121] = 0.0


def _qc(level, step):
    """Fake cloud water, kg/kg: not linear in height, so a wrong pair shows."""
    return 1e-5 * (121 - level) ** 1.5 + 1e-6 * step


# Half a quantisation step (columns._QUANT), plus the float32 the remapped
# fields arrive in (~3e-5 K at 300 K).
TOL_T = columns._QUANT[columns.T][0] / 2 + 1e-4
TOL_1 = 0.5 + 1e-3


def _t_kelvin(level, step):
    return 280.0 + 0.5 * step - 0.004 * (1000 - level)


def _column_value(name):
    """The fake field a DWD column file holds, by its name."""
    m = re.search(r"_pressure-level_\d{10}_(\d{3})_(\d+)_([A-Z]+)\.", name)
    if m:
        step, lv, f = int(m.group(1)), int(m.group(2)), m.group(3)
        return lv, {"T": _t_kelvin(lv, step), "RELHUM": 50.0 + lv / 100, "FI": 9.80665 * (8000 - 7 * lv)}[f]
    m = re.search(r"_model-level_\d{10}_(\d{3})_(\d+)_T\.", name)
    if m:
        step, lv = int(m.group(1)), int(m.group(2))
        return lv, 275.0 + step + (118 - lv) * 1.5
    m = re.search(r"_model-level_\d{10}_(\d{3})_(\d+)_QC\.", name)
    if m:
        step, lv = int(m.group(1)), int(m.group(2))
        return lv, _qc(lv, step)
    m = re.search(r"_time-invariant_\d{10}_(\d+)_HHL\.", name)
    if m:
        return int(m.group(1)), HSURF_M + HHL_AGL[int(m.group(1))]
    if "_HSURF." in name:
        return None, HSURF_M
    m = re.search(r"_single-level_\d{10}_(\d{3})_(T_2M|RELHUM_2M|WW)\.", name)
    step = int(m.group(1))
    if m.group(2) == "WW":
        return None, _ww_code(step)
    return 2, {"T_2M": 283.15 + step, "RELHUM_2M": 91.0}[m.group(2)]


def _ww_code(step):
    """A WMO 4677 code per step: 45 fog, 61 rain, 71 snow, 95 thunder, ..."""
    return float([45, 61, 71, 95, 3, 51][step % 6])


def _fake_lattice_levels(paths, kit, work, stride, nearest=False):
    shape = columns.lattice(np.zeros((NY, NX), np.uint8), stride).shape
    out = {}
    for p in paths:
        # WW is categorical: it must be fetched nearest-link, every other field blended.
        assert nearest == ("_WW." in Path(p).name), (Path(p).name, nearest)
        lv, v = _column_value(Path(p).name)
        out[lv] = np.full(shape, v, dtype=np.float32)
    return out


def _deps_with_columns(tmp_path, steps, column_steps=None, qc_steps=None, **fake):
    deps, calls = _fake_deps(tmp_path, **fake)
    files = column_files(RUN, steps if column_steps is None else column_steps)
    qc = column_qc_files(RUN, steps if qc_steps is None else qc_steps)
    files["QC"] = qc["QC"]
    files["HHL"] = sorted(set(files["HHL"]) | set(qc["HHL"]))

    run = RUN

    def listing(hh, field):
        # The cloud and thickness fields as _fake_deps lists them (under 00);
        # the column fields under their run's own hour, as DWD does.
        if field in dwd_fields() or field in THICK_FIELDS:
            return _listing(run, field) if hh == "00" else ""
        if hh != f"{run:%H}":
            return ""
        return "".join(f'<a href="{n}">x</a>\n' for n in files.get(field, []))
    return deps._replace(listing=listing, lattice_levels=_fake_lattice_levels), calls


def dwd_fields():
    return FIELDS


def _columns_build(tmp_path, deps, steps=(0, 1, 2), squares=("N40W125",)):
    return run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                           deps=deps, pages_base="https://x", steps=list(steps), squares=list(squares), workers=2)


def _value(d, kind, lt, lv, r=0, c=0, s=0):
    for f, codes in zip(d["spec"], d["fields"]):
        if f[:3] == (kind, lt, lv):
            return float(dequantize(codes[r, c] if f[1] == STATIC else codes[r, c, s], f[3], f[4]))
    raise KeyError((kind, lt, lv))


def test_a_build_publishes_the_columns_in_app_units(tmp_path):
    steps = [0, 1, 2]
    deps, calls = _deps_with_columns(tmp_path, steps)
    assert _columns_build(tmp_path, deps, steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    col = m["columns"]
    base = hour_of(RUN)
    assert col["format"] == "ICC1" and col["run"] == "2026-09-27T00Z"
    assert col["tiles"] == "columns/2026092700"
    assert col["stepsMs"] == [(base + s) * 3600 * 1000 for s in steps]
    assert col["historySteps"] == 0
    assert col["spacing"] == run_build.COLUMN_STRIDE * 0.125
    assert col["side"] == 40 // run_build.COLUMN_STRIDE + 1
    d = decode_columns((tmp_path / "blob" / col["tiles"] / "N40W125.icc").read_bytes())
    assert d["step_hours"] == [base + s for s in steps]
    assert (d["rows"], d["cols"]) == (col["side"], col["side"])
    P, H = columns.PRESSURE_HPA, columns.HEIGHT_AGL_M
    for s in steps:
        for lv in COLUMN_PLEVELS:
            assert _value(d, columns.T, P, lv, s=s) == pytest.approx(_t_kelvin(lv, s) - 273.15, abs=TOL_T)
            assert _value(d, columns.RH, P, lv, s=s) == pytest.approx(50.0 + lv / 100, abs=TOL_1)
            assert _value(d, columns.GH, P, lv, s=s) == pytest.approx(8000 - 7 * lv, abs=TOL_1)
        assert _value(d, columns.T, H, 2, s=s) == pytest.approx(283.15 + s - 273.15, abs=TOL_T)
        assert _value(d, columns.RH, H, 2, s=s) == pytest.approx(91.0, abs=TOL_1)
        assert _value(d, columns.WW, H, 0, s=s) == _ww_code(s)            # the code itself, unchanged
    assert _value(d, columns.HSURF, STATIC, 0) == pytest.approx(HSURF_M, abs=TOL_1)


def test_ww_codes_outside_0_99_are_missing_never_clipped():
    # 99 is "thunderstorm with heavy hail": garbage must never decode as severe weather.
    arr = np.array([0.0, 3.2, 61.0, 99.0, 99.4, 99.6, 100.0, 255.0, -0.6, -1.0, np.nan, np.inf])
    got = run_build._to_codes(("WW", None), arr)
    assert got.tolist() == [0, 3, 61, 99, 99, 65535, 65535, 65535, 65535, 65535, 65535, 65535]


def test_t80_is_interpolated_in_height_between_the_two_levels_around_80_m(tmp_path):
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)
    _columns_build(tmp_path, deps, steps)
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    d = decode_columns((tmp_path / "blob" / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
    full = {lv: (HHL_AGL[lv] + HHL_AGL[lv + 1]) / 2 for lv in COLUMN_MLEVELS}
    up = max(lv for lv in COLUMN_MLEVELS if full[lv] >= 80)          # the lowest level at or above 80 m
    lo = up + 1
    assert full[lo] < 80 <= full[up]
    for s in steps:
        t_up = _column_value(f"x_model-level_2026092700_{s:03d}_{up}_T.g")[1]
        t_lo = _column_value(f"x_model-level_2026092700_{s:03d}_{lo}_T.g")[1]
        want = t_lo + (t_up - t_lo) * (80 - full[lo]) / (full[up] - full[lo]) - 273.15
        assert _value(d, columns.T, columns.HEIGHT_AGL_M, 80, s=s) == pytest.approx(want, abs=TOL_T)


def test_t80_weights_pick_the_bracketing_pair_per_node_and_never_extrapolate():
    # Four nodes: 80 m between 118 and 119; between 117 and 118 (118 low over
    # steep ground); above all three (clamped to 117); a missing height.
    levels = sorted(COLUMN_MLEVELS)
    full = np.array([[165.0, 95.5, 42.0], [120.0, 70.0, 35.0], [78.0, 60.0, 30.0], [np.nan, 90.0, 40.0]])
    hsurf = np.array([500.0, 0.0, 2000.0, 10.0])
    # Half levels whose midpoints are `full` (the lowest half level on the ground).
    half = {levels[-1] + 1: hsurf.copy()}
    for i, lv in reversed(list(enumerate(levels))):
        half[lv] = 2 * (full[:, i] + hsurf) - half[lv + 1]
    w = run_build.t80_weights(half, hsurf)
    assert sorted(w) == levels
    for node in range(3):
        t = {lv: 280.0 - 0.01 * full[node, i] for i, lv in enumerate(levels)}       # linear in height
        assert sum(w[lv][node] for lv in levels) == pytest.approx(1.0)
        assert all(0.0 <= w[lv][node] <= 1.0 for lv in levels)
        got = sum(w[lv][node] * t[lv] for lv in levels)
        z = full[node]
        want = 280.0 - 0.01 * (80.0 if z.min() <= 80 <= z.max() else z.min() if 80 < z.min() else z.max())
        assert got == pytest.approx(want)
    assert w[levels[0]][2] == 1.0                                    # clamped to the top level
    assert all(np.isnan(w[lv][3]) for lv in levels)


def test_a_full_build_publishes_columns_for_every_square(tmp_path):
    deps, _ = _deps_with_columns(tmp_path, STEPS, column_steps=STEPS)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=None, squares=None, workers=4)
    assert rc == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    assert m["complete"] is True and m["columns"] is not None
    got = sorted(p.stem for p in (tmp_path / "blob" / m["columns"]["tiles"]).glob("*.icc"))
    assert got == sorted(NINE) == m["squares"]
    assert len(m["columns"]["stepsMs"]) == len(STEPS)


def test_a_column_field_no_job_filled_fails_the_columns_never_ships_garbage(tmp_path, monkeypatch, capsys):
    # The jobs write into one uninitialised array: a field no job wrote must
    # fail the columns (as a missing key always did), never ship as noise.
    real = run_build.column_step

    def lossy(*a, **k):
        out = real(*a, **k)
        out.pop(("WW", None), None)
        return out
    monkeypatch.setattr(run_build, "column_step", lossy)
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)
    assert _columns_build(tmp_path, deps, steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    assert m["columns"] is None
    assert "never fetched" in capsys.readouterr().out


def test_column_files_not_yet_listed_wait_then_publish_without_columns(tmp_path, capsys):
    steps = [0, 1]
    deps, calls = _deps_with_columns(tmp_path, steps, column_steps=[0])         # step 1 never lands
    assert _columns_build(tmp_path, deps, steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    assert m["columns"] is None
    # (the fake listing carries no wind files either: that wait comes first)
    assert sum(calls["sleeps"]) == run_build.WIND_WAIT_S + run_build.COLUMN_WAIT_S
    assert (tmp_path / "site" / m["tiles"] / "N40W125.icl").exists()
    assert not (tmp_path / "blob" / "columns").exists()
    assert "::warning::columns" in capsys.readouterr().out


def test_a_column_failure_never_fails_the_build(tmp_path, capsys):
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)

    def broken(paths, kit, work, stride, nearest=False):
        if any("RELHUM_2M" in Path(p).name for p in paths):
            raise OSError("cdo died")
        return _fake_lattice_levels(paths, kit, work, stride, nearest=nearest)
    assert _columns_build(tmp_path, deps._replace(lattice_levels=broken), steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    assert m["columns"] is None
    assert not (tmp_path / "blob" / "columns").exists()
    assert "::warning::columns failed" in capsys.readouterr().out


def test_a_thickness_failure_still_publishes_the_columns(tmp_path, capsys):
    def thick_value(step, field):
        if (step, field) == (1, "TQC_DIA"):
            raise OSError("cdo died")
        return THICK_KG[field]
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps, thick_value=thick_value)
    assert _columns_build(tmp_path, deps, steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    assert m["thick"] is None and not (tmp_path / "blob" / "thick").exists()
    assert m["columns"]["tiles"] == "columns/2026092700"
    d = decode_columns((tmp_path / "blob" / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
    assert d["step_hours"] == [hour_of(RUN) + s for s in steps]
    assert _value(d, columns.T, columns.HEIGHT_AGL_M, 2, s=1) == pytest.approx(283.15 + 1 - 273.15, abs=TOL_T)
    assert "::warning::thickness failed" in capsys.readouterr().out


def test_column_history_comes_from_state(tmp_path):
    steps = list(range(6))
    deps, _ = _deps_with_columns(tmp_path, steps)
    _columns_build(tmp_path, deps, steps)
    assert len(list((tmp_path / "state" / "columns").glob("2026092700.h*.npz"))) == HEAD_STEPS
    # The next run, six hours on, reads those six hours as history.
    global RUN
    first, RUN = RUN, RUN + dt.timedelta(hours=6)
    try:
        deps, _ = _deps_with_columns(tmp_path, [0, 1])
        blob2 = tmp_path / "blob2"
        run_build.build(site=tmp_path / "site2", state=tmp_path / "state", work=tmp_path / "work", deps=deps,
                        pages_base="https://x", steps=[0, 1], squares=["N40W125"], workers=2, blob=blob2)
        m = json.loads((blob2 / "manifest.json").read_text())
        assert m["columns"]["historySteps"] == 6
        assert m["columns"]["stepsMs"][:6] == [(hour_of(first) + s) * 3600 * 1000 for s in range(6)]
        d = decode_columns((blob2 / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
        # History hour s is the first run's step s: T_2M = 283.15 + s K.
        for s in range(6):
            assert _value(d, columns.T, columns.HEIGHT_AGL_M, 2, s=s) == pytest.approx(283.15 + s - 273.15, abs=TOL_T)
    finally:
        RUN = first


def test_column_history_saved_before_ww_is_padded_with_missing(tmp_path):
    # A run head saved before WW joined the feed holds every dynamic field but
    # the last (WW). The next build still uses those six hours, WW missing.
    steps = list(range(6))
    deps, _ = _deps_with_columns(tmp_path, steps)
    _columns_build(tmp_path, deps, steps)
    for p in (tmp_path / "state" / "columns").glob("2026092700.h*.npz"):
        with np.load(p) as z:
            head, rh = z["head"], z["run_hour"]
        assert head.shape[1] == len(run_build.column_keys())
        with np.load(p) as z:                                         # new heads name their fields
            assert list(z["fields"]) == run_build.column_field_names()
        # the pre-WW layout (no field list): every field before WW (WW and the
        # later CLMR block gone).
        np.savez_compressed(p, head=head[:, :-(1 + len(COLUMN_PLEVELS))], run_hour=rh)
    global RUN
    first, RUN = RUN, RUN + dt.timedelta(hours=6)
    try:
        deps, _ = _deps_with_columns(tmp_path, [0, 1])
        blob2 = tmp_path / "blob2"
        run_build.build(site=tmp_path / "site2", state=tmp_path / "state", work=tmp_path / "work", deps=deps,
                        pages_base="https://x", steps=[0, 1], squares=["N40W125"], workers=2, blob=blob2)
        m = json.loads((blob2 / "manifest.json").read_text())
        assert m["columns"]["historySteps"] == 6
        d = decode_columns((blob2 / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
        H = columns.HEIGHT_AGL_M
        for s in range(6):
            assert _value(d, columns.T, H, 2, s=s) == pytest.approx(283.15 + s - 273.15, abs=TOL_T)
            assert np.isnan(_value(d, columns.WW, H, 0, s=s))
        for s in range(2):                                            # this run's own steps carry WW
            assert _value(d, columns.WW, H, 0, s=6 + s) == _ww_code(s)
    finally:
        RUN = first


def test_the_columns_go_to_blob_only_and_never_touch_the_pages_budget(tmp_path, monkeypatch, capsys):
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)
    _columns_build(tmp_path / "a", deps, steps)
    site, blob = tmp_path / "a" / "site", tmp_path / "a" / "blob"
    pages = json.loads((site / "manifest.json").read_text())
    feed = json.loads((blob / "manifest.json").read_text())
    # Pages: clouds and tail only, and a manifest that names no columns.
    assert not (site / "columns").exists() and pages["columns"] is None
    # Blob: the columns, and a manifest equal to Pages' but for its columns key.
    assert run_build._dir_bytes(blob / feed["columns"]["tiles"]) > 0
    # (the blocks, the thickness and the packs are Blob-only too: the manifests differ in those keys alone)
    blob_only = ("columns", "blocks", "thick", "tilesPack", "blocksPack")
    assert pages["thick"] is None and feed["thick"] is not None
    assert {k: v for k, v in feed.items() if k not in blob_only} == \
        {k: v for k, v in pages.items() if k not in blob_only}
    # blob/tiles holds the squares' pack alone (the squares themselves are in site/).
    assert sorted(p.relative_to(blob).parts[0] for p in blob.iterdir()) == \
        ["blocks", "columns", "manifest.json", "thick", "tiles"]
    assert sorted(p.relative_to(blob).as_posix() for p in (blob / "tiles").rglob("*") if p.is_file()) == \
        [f"{feed['tilesPack']}/all.bin", f"{feed['tilesPack']}/index.json"]
    clouds = run_build._dir_bytes(site / "tiles")
    capsys.readouterr()
    # A budget the clouds just fit: however big the columns, the squares stay in 2 % steps.
    monkeypatch.setattr(run_build, "SITE_BUDGET_BYTES", clouds)
    _columns_build(tmp_path / "b", deps, steps)
    out = capsys.readouterr().out
    assert "budget" not in out
    assert run_build._dir_bytes(tmp_path / "b" / "site" / "tiles") == clouds
    assert json.loads((tmp_path / "b" / "blob" / "manifest.json").read_text())["columns"] is not None


# ---- Contingencies (2026-10-28) ----------------------------------------------

def test_world_problems_names_a_constant_field_and_an_impossible_total_cover():
    real = np.tile(np.arange(NX, dtype=np.uint8) % 101, (NY, 1))         # mean ~50, varied
    ok = {(0, f): real for f in run_build.dwd.FIELDS}
    assert run_build.world_problems(ok) == []
    flat = dict(ok); flat[(0, "CLCL")] = np.zeros((NY, NX), np.uint8)
    assert run_build.world_problems(flat) == ["CLCL +0h is 0 everywhere"]
    clear = dict(ok); clear[(3, "CLCT")] = (real // 10).astype(np.uint8)  # varied but ~5 %
    assert any(p.startswith("CLCT +3h averages") for p in run_build.world_problems(clear))


def test_a_broken_run_is_never_published(tmp_path):
    deps, _ = _fake_deps(tmp_path)
    deps = deps._replace(validate=lambda world: ["CLCT +0h is 0 everywhere"])
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=[0], squares=["N00E000"])
    assert rc == run_build.FAILED
    assert not (tmp_path / "site" / "manifest.json").exists()


def test_the_real_deps_validate_with_world_problems():
    assert run_build.REAL.validate is None                                # resolved to world_problems in build()


def test_unreadable_history_costs_the_past_hours_not_the_run(tmp_path, monkeypatch):
    deps, _ = _fake_deps(tmp_path)
    def boom(*a, **k):
        raise ValueError("bad history")
    monkeypatch.setattr(run_build.history, "load_history_steps", boom)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base=None, steps=[0], squares=["N00E000"])
    assert rc == run_build.BUILT
    assert json.loads((tmp_path / "site" / "manifest.json").read_text())["historySteps"] == 0


def test_rebuilding_the_live_run_counts_a_repair(tmp_path, monkeypatch):
    # The live manifest (Blob's, here the fake's) is this run without its columns, repaired once.
    prev = {"run": "2026-09-27T00Z", "complete": True, **WHOLE, "columns": None, "repairs": 1}
    deps, _ = _fake_deps(tmp_path, prev_manifest=prev)
    for f in ("build_tail", "build_thick", "build_columns"):
        monkeypatch.setattr(run_build, f, lambda *a, **k: None)
    rc = run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                         deps=deps, pages_base="https://x", steps=None, squares=None, workers=2)
    assert rc == run_build.BUILT
    assert json.loads((tmp_path / "blob" / "manifest.json").read_text())["repairs"] == 2
    # A second repair is the last: the same manifest at 2 is left alone.
    deps, _ = _fake_deps(tmp_path, prev_manifest={**prev, "repairs": 2})
    assert run_build.build(site=tmp_path / "s2", state=tmp_path / "st2", work=tmp_path / "w2",
                           deps=deps, pages_base="https://x") == run_build.NOTHING_TO_DO


def test_history_runs_go_in_both_manifests(tmp_path):
    from pipeline import published
    prev_ms = int((RUN - dt.timedelta(hours=6)).timestamp() * 1000)
    published.record(tmp_path / "state", prev_ms, prev_ms + 4 * 3600000)
    site, blob = _blocks_build(tmp_path, [0, 1], lambda s, f: 10, squares=["N40W125"])
    want = [{"runMs": prev_ms, "publishedMs": prev_ms + 4 * 3600000}]
    assert json.loads((site / "manifest.json").read_text())["historyRuns"] == want
    assert json.loads((blob / "manifest.json").read_text())["historyRuns"] == want
    for d in (site, blob):
        assert json.loads((d / "manifest.json").read_text())["historyRecordSinceMs"] == prev_ms + 4 * 3600000


# ---- Packs (PACK1, pipeline/pack.py): Blob only ------------------------------

def _is_pack(path, folder):
    """folder/pack-<first 12 hex of sha256(all.bin)>."""
    return re.fullmatch(re.escape(folder) + r"/pack-[0-9a-f]{12}", path or "") is not None


def _pack_dirs(d):
    return [p.name for p in d.iterdir() if p.is_dir()] if d.is_dir() else []


def _pack_matches(files_root, pack_root, folder, pack_rel):
    """Every file of files_root/folder, byte-exact, at its span of the pack."""
    idx = json.loads((pack_root / pack_rel / "index.json").read_text())
    data = (pack_root / pack_rel / "all.bin").read_bytes()
    names = sorted(p.name for p in (files_root / folder).iterdir() if p.is_file())
    assert idx["format"] == "PACK1" and idx["names"] == names and names
    for i, n in enumerate(names):
        start = idx["ends"][i - 1] if i else 0
        assert data[start:idx["ends"][i]] == (files_root / folder / n).read_bytes(), n
    assert len(data) == idx["ends"][-1]


def test_a_full_build_packs_tiles_tail_blocks_and_thick_into_blob_only(tmp_path, monkeypatch):
    monkeypatch.delenv("FEED_TILE_FILES", raising=False)
    deps, _ = _fake_deps_with_tail(tmp_path)
    assert run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                           deps=deps, pages_base="https://x", steps=None, squares=None, workers=2) == run_build.BUILT
    site, blob = tmp_path / "site", tmp_path / "blob"
    feed = json.loads((blob / "manifest.json").read_text())
    pages = json.loads((site / "manifest.json").read_text())
    # Tile files on (the default): the per-file keys stay beside the pack keys.
    assert feed["tiles"] == "tiles/2026092700" and _is_pack(feed["tilesPack"], "tiles/2026092700")
    assert feed["blocks"] == "blocks/2026092700" and _is_pack(feed["blocksPack"], "blocks/2026092700")
    assert feed["tail"]["tiles"] == "tail/2026092700" and _is_pack(feed["tail"]["pack"], "tail/2026092700")
    assert feed["thick"]["tiles"] == "thick/2026092700" and _is_pack(feed["thick"]["pack"], "thick/2026092700")
    # tiles and tail are in site/ (Pages and Blob); their packs go to blob/ alone.
    _pack_matches(site, blob, "tiles/2026092700", feed["tilesPack"])
    _pack_matches(site, blob, "tail/2026092700", feed["tail"]["pack"])
    _pack_matches(blob, blob, "blocks/2026092700", feed["blocksPack"])
    _pack_matches(blob, blob, "thick/2026092700", feed["thick"]["pack"])
    # Pages: no pack files, no pack keys.
    assert not any(p.name.startswith("pack") for p in site.rglob("*"))
    assert "tilesPack" not in pages and "blocksPack" not in pages and "pack" not in (pages["tail"] or {})


def test_the_columns_get_a_pack_too(tmp_path, monkeypatch):
    monkeypatch.delenv("FEED_TILE_FILES", raising=False)
    deps, _ = _deps_with_columns(tmp_path, [0, 1])
    assert _columns_build(tmp_path, deps, [0, 1]) == run_build.BUILT
    blob = tmp_path / "blob"
    feed = json.loads((blob / "manifest.json").read_text())
    assert feed["columns"]["tiles"] == "columns/2026092700"
    assert _is_pack(feed["columns"]["pack"], "columns/2026092700")
    _pack_matches(blob, blob, "columns/2026092700", feed["columns"]["pack"])


def test_tile_files_off_drops_the_per_file_keys_from_the_blob_manifest_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEED_TILE_FILES", "0")
    deps, _ = _fake_deps_with_tail(tmp_path)
    assert run_build.build(site=tmp_path / "site", state=tmp_path / "state", work=tmp_path / "work",
                           deps=deps, pages_base="https://x", steps=None, squares=None, workers=2) == run_build.BUILT
    assert "tile files off" in capsys.readouterr().out
    site, blob = tmp_path / "site", tmp_path / "blob"
    feed = json.loads((blob / "manifest.json").read_text())
    assert "tiles" not in feed and _is_pack(feed["tilesPack"], "tiles/2026092700")
    assert "blocks" not in feed and _is_pack(feed["blocksPack"], "blocks/2026092700")
    assert "tiles" not in feed["tail"] and _is_pack(feed["tail"]["pack"], "tail/2026092700")
    assert "tiles" not in feed["thick"] and _is_pack(feed["thick"]["pack"], "thick/2026092700")
    # Pages is untouched: per-file squares and tail, named as before.
    pages = json.loads((site / "manifest.json").read_text())
    assert pages["tiles"] == "tiles/2026092700" and pages["tail"]["tiles"] == "tail/2026092700"
    assert (site / "tiles" / "2026092700" / "N40W125.icl").exists()


def test_a_failed_pack_drops_its_key_never_the_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEED_TILE_FILES", "0")
    from pipeline import pack
    real = pack.pack_folder

    def flaky(root, rel, dest=None):
        if rel.startswith("blocks/"):
            raise OSError("disk full")
        return real(root, rel, dest)
    monkeypatch.setattr(pack, "pack_folder", flaky)
    site, blob = _blocks_build(tmp_path, [0, 1], lambda s, f: 10, squares=["N40W125"])
    feed = json.loads((blob / "manifest.json").read_text())
    assert "::warning::pack blocks/2026092700 failed" in capsys.readouterr().out
    assert "blocksPack" not in feed and feed["blocks"] == "blocks/2026092700"   # still named: still uploaded
    assert _is_pack(feed["tilesPack"], "tiles/2026092700") and "tiles" not in feed
    assert _pack_dirs(blob / "blocks" / "2026092700") == []


def test_a_repair_that_changes_the_squares_gets_new_pack_folders(tmp_path, monkeypatch):
    # An ICON repair rebuilds the same run in place; all.bin is cached a year,
    # so changed bytes must land at a new path the new manifest names.
    monkeypatch.delenv("FEED_TILE_FILES", raising=False)
    _, blob_a = _blocks_build(tmp_path / "a", [0, 1], lambda s, f: 10, squares=["N40W125"])
    _, blob_b = _blocks_build(tmp_path / "b", [0, 1], lambda s, f: 10, squares=["N40W125"])
    _, blob_c = _blocks_build(tmp_path / "c", [0, 1], lambda s, f: 30, squares=["N40W125"])
    a, b, c = (json.loads((d / "manifest.json").read_text()) for d in (blob_a, blob_b, blob_c))
    assert a["tilesPack"] == b["tilesPack"] and a["blocksPack"] == b["blocksPack"]
    assert a["tilesPack"] != c["tilesPack"] and a["blocksPack"] != c["blocksPack"]
    assert a["tiles"] == c["tiles"]                              # the run folder itself is the same


# ---- Cloud water (CLMR) in the column feed ----------------------------------
TOL_CLMR = columns._QUANT[columns.CLMR][0] / 2 + 1e-9


def _full_msl(lv):
    return HSURF_M + (HHL_AGL[lv] + HHL_AGL[lv + 1]) / 2


def _clmr_want(p, step):
    """CLMR at p hPa from the fakes, computed independently: the bracketing
    pair found by a plain scan, the lowest level's value at or below it."""
    gh = 8000 - 7 * p
    levels = sorted(COLUMN_QC_MLEVELS, reverse=True)               # 120 (lowest) first
    if gh <= _full_msl(levels[0]):
        return _qc(levels[0], step)
    for lo, up in zip(levels, levels[1:]):
        if _full_msl(lo) <= gh <= _full_msl(up):
            f = (gh - _full_msl(lo)) / (_full_msl(up) - _full_msl(lo))
            return _qc(lo, step) + f * (_qc(up, step) - _qc(lo, step))
    return float("nan")


def test_clmr_is_qc_interpolated_in_height_to_each_levels_gh(tmp_path):
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)
    _columns_build(tmp_path, deps, steps)
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    d = decode_columns((tmp_path / "blob" / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
    assert 8000 - 7 * 1000 < HSURF_M                                   # the 1000 hPa case is below ground
    for s in steps:
        for p in COLUMN_PLEVELS:
            want = _clmr_want(p, s)
            assert np.isfinite(want)
            assert _value(d, columns.CLMR, columns.PRESSURE_HPA, p, s=s) == pytest.approx(want, abs=TOL_CLMR), p
        # the old fields are untouched
        assert _value(d, columns.WW, columns.HEIGHT_AGL_M, 0, s=s) == _ww_code(s)


def test_clmr_at_heights_brackets_clamps_below_and_never_invents():
    # Three levels at 100, 400, 1000 m MSL; QC 3e-4, 1e-4, 0.
    z = {120: np.full(7, 100.0), 119: np.full(7, 400.0), 118: np.full(7, 1000.0)}
    qc = {120: np.full(7, 3e-4), 119: np.full(7, 1e-4), 118: np.zeros(7)}
    qc[119][4] = np.nan                                                # a missing level at node 4
    z[118][6] = np.nan                                                 # a missing height at node 6
    gh = np.array([250.0, 700.0, 40.0, 1500.0, 250.0, np.nan, 250.0])
    got = run_build.clmr_at_heights(qc, z, gh)
    assert got[0] == pytest.approx(3e-4 + (250 - 100) / 300 * (1e-4 - 3e-4))   # between 120 and 119
    assert got[1] == pytest.approx(1e-4 + (700 - 400) / 600 * (0 - 1e-4))      # between 119 and 118
    assert got[2] == 3e-4                                              # below the lowest (under ground): 120's value
    assert np.isnan(got[3])                                            # above the top: missing, not extrapolated
    assert np.isnan(got[4])                                            # a bracketing level missing: missing
    assert np.isnan(got[5]) and np.isnan(got[6])                       # missing gh / height: missing
    # below ground needs only the lowest level: a missing level above it is irrelevant
    assert run_build.clmr_at_heights(qc, z, np.full(7, 40.0))[4] == 3e-4
    # exactly on a level: that level's value
    assert run_build.clmr_at_heights(qc, z, np.full(7, 400.0))[0] == pytest.approx(1e-4)
    assert run_build.clmr_at_heights(qc, z, np.full(7, 1000.0))[0] == pytest.approx(0.0)
    # 2-D lattices work the same
    z2 = {lv: a[:6].reshape(2, 3) for lv, a in z.items()}
    q2 = {lv: a[:6].reshape(2, 3) for lv, a in qc.items()}
    assert np.array_equal(run_build.clmr_at_heights(q2, z2, gh[:6].reshape(2, 3)).reshape(-1),
                          got[:6], equal_nan=True)


def test_full_level_heights_are_the_half_level_midpoints():
    hhl = {lv: np.array([HSURF_M + HHL_AGL[lv]]) for lv in range(99, 122)}
    z = run_build.full_level_heights(hhl, [100, 120])
    assert z[120][0] == pytest.approx(HSURF_M + (HHL_AGL[120] + 0.0) / 2)
    assert z[100][0] == pytest.approx(_full_msl(100))


def test_qc_files_never_hold_the_columns_back(tmp_path, capsys):
    # QC listed for step 0 only: the columns still publish; step 1's CLMR is
    # missing, step 0's is there, and no other field changes.
    steps = [0, 1]
    deps, calls = _deps_with_columns(tmp_path, steps, qc_steps=[0])
    assert _columns_build(tmp_path, deps, steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    d = decode_columns((tmp_path / "blob" / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
    P = columns.PRESSURE_HPA
    for p in COLUMN_PLEVELS:
        assert _value(d, columns.CLMR, P, p, s=0) == pytest.approx(_clmr_want(p, 0), abs=TOL_CLMR)
        assert np.isnan(_value(d, columns.CLMR, P, p, s=1))
        assert _value(d, columns.GH, P, p, s=1) == pytest.approx(8000 - 7 * p, abs=TOL_1)
    assert sum(calls["sleeps"]) == run_build.WIND_WAIT_S                # no wait on QC (only the fake's wind)
    assert not any("_001_" in u and "_QC." in u for u in calls["downloads"])
    assert "QC not listed for 1 of 2 steps" in capsys.readouterr().out


def test_a_cloud_water_failure_leaves_clmr_missing_and_the_rest_intact(tmp_path, capsys):
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)

    def broken(paths, kit, work, stride, nearest=False):
        if any("_001_" in Path(p).name and "_QC." in Path(p).name for p in paths):
            raise OSError("cdo died")
        return _fake_lattice_levels(paths, kit, work, stride, nearest=nearest)
    assert _columns_build(tmp_path, deps._replace(lattice_levels=broken), steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    d = decode_columns((tmp_path / "blob" / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
    for p in COLUMN_PLEVELS:
        assert np.isfinite(_value(d, columns.CLMR, columns.PRESSURE_HPA, p, s=0))
        assert np.isnan(_value(d, columns.CLMR, columns.PRESSURE_HPA, p, s=1))
        assert _value(d, columns.GH, columns.PRESSURE_HPA, p, s=1) == pytest.approx(8000 - 7 * p, abs=TOL_1)
    assert "cloud water +1 failed" in capsys.readouterr().out


def test_no_hhl_for_cloud_water_leaves_every_clmr_missing(tmp_path, capsys):
    steps = [0, 1]
    deps, _ = _deps_with_columns(tmp_path, steps)
    top = min(COLUMN_QC_MLEVELS)

    def broken(paths, kit, work, stride, nearest=False):
        if any(f"_{top}_HHL." in Path(p).name for p in paths):
            raise OSError("no such file")
        return _fake_lattice_levels(paths, kit, work, stride, nearest=nearest)
    assert _columns_build(tmp_path, deps._replace(lattice_levels=broken), steps) == run_build.BUILT
    m = json.loads((tmp_path / "blob" / "manifest.json").read_text())
    d = decode_columns((tmp_path / "blob" / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
    for s in steps:
        assert all(np.isnan(_value(d, columns.CLMR, columns.PRESSURE_HPA, p, s=s)) for p in COLUMN_PLEVELS)
        assert _value(d, columns.T, columns.HEIGHT_AGL_M, 80, s=s) == pytest.approx(
            _value(d, columns.T, columns.HEIGHT_AGL_M, 80, s=s))                  # T80 still built
        assert np.isfinite(_value(d, columns.T, columns.HEIGHT_AGL_M, 80, s=s))
    assert "no HHL for cloud water" in capsys.readouterr().out


def test_column_history_saved_before_clmr_is_padded_with_missing(tmp_path):
    # A head saved by the build before CLMR (2026-11-07 .. 11-19) names its 25
    # fields: they are a strict prefix of today's, so its six hours are used,
    # every old field as saved and CLMR missing.
    steps = list(range(6))
    deps, _ = _deps_with_columns(tmp_path, steps)
    _columns_build(tmp_path, deps, steps)
    names = run_build.column_field_names()
    old_names = [n for n in names if not n.startswith("CLMR@")]
    assert names[:len(old_names)] == old_names and len(old_names) == 25          # the 26-field table minus static HSURF
    for p in (tmp_path / "state" / "columns").glob("2026092700.h*.npz"):
        with np.load(p) as z:
            head, rh = z["head"], z["run_hour"]
        np.savez_compressed(p, head=head[:, :len(old_names)], run_hour=rh, fields=np.array(old_names))
    global RUN
    first, RUN = RUN, RUN + dt.timedelta(hours=6)
    try:
        deps, _ = _deps_with_columns(tmp_path, [0, 1])
        blob2 = tmp_path / "blob2"
        run_build.build(site=tmp_path / "site2", state=tmp_path / "state", work=tmp_path / "work", deps=deps,
                        pages_base="https://x", steps=[0, 1], squares=["N40W125"], workers=2, blob=blob2)
        m = json.loads((blob2 / "manifest.json").read_text())
        assert m["columns"]["historySteps"] == 6
        d = decode_columns((blob2 / m["columns"]["tiles"] / "N40W125.icc").read_bytes())
        H, P = columns.HEIGHT_AGL_M, columns.PRESSURE_HPA
        for s in range(6):
            assert _value(d, columns.T, H, 2, s=s) == pytest.approx(283.15 + s - 273.15, abs=TOL_T)
            assert _value(d, columns.WW, H, 0, s=s) == _ww_code(s)
            assert all(np.isnan(_value(d, columns.CLMR, P, p, s=s)) for p in COLUMN_PLEVELS)
        for s in (6, 7):                                                # the new run's own hours have CLMR
            assert _value(d, columns.CLMR, P, 700, s=s) == pytest.approx(_clmr_want(700, s - 6), abs=TOL_CLMR)
    finally:
        RUN = first


def test_a_head_whose_names_are_not_a_prefix_is_skipped(tmp_path):
    # Same field count, different names: never misread as today's layout.
    names = run_build.column_field_names()
    head = np.zeros((HEAD_STEPS, len(names), 2, 2), np.uint16)
    run = dt.datetime(2026, 9, 27, 0, tzinfo=dt.timezone.utc)
    history.save_run_head(tmp_path, run, head, sub="columns", dtype=np.uint16, split=True,
                          fields=list(reversed(names)))
    later = run + dt.timedelta(hours=6)
    assert history.load_history_steps(tmp_path, later, sub="columns", shape=(len(names), 2, 2),
                                      pad=columns.MISSING, fields=names) == []


def _clmr_on_a_stack(qc, z, gh):
    """The rule on a stack of every level (984df65's clmr_at_heights), kept
    as the reference: the stream must give the same float64 bits."""
    levels = sorted(z, reverse=True)
    zz = np.stack([np.asarray(z[lv], dtype=np.float64) for lv in levels])
    qq = np.stack([np.asarray(qc[lv], dtype=np.float64) for lv in levels])
    gh = np.asarray(gh, dtype=np.float64)
    n = len(levels)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.clip(np.sum(zz <= gh, axis=0), 1, n - 1)
        lo = up - 1
        pick = lambda a, i: np.take_along_axis(a, i[None], 0)[0]
        f = np.clip((gh - pick(zz, lo)) / (pick(zz, up) - pick(zz, lo)), 0.0, 1.0)
        out = pick(qq, lo) + f * (pick(qq, up) - pick(qq, lo))
        out = np.where(gh <= zz[0], qq[0], out)
        out = np.where(gh > zz[-1], np.nan, out)
    return np.where(~np.isfinite(gh) | np.any(~np.isfinite(zz), axis=0), np.nan, out)


def test_the_cloud_water_stream_gives_the_stack_rules_exact_bits():
    rng = np.random.default_rng(3)
    shape = (40, 50)
    levels = list(range(99, 121))
    # heights rising ~280 m a level from a ground at 0-3000 m, float32 as remapped
    ground = rng.uniform(0, 3000, shape)
    half = {lv: (ground + 20.0 * (121 - lv) ** 1.6).astype(np.float32) for lv in range(99, 122)}
    half[110][3, 4] = np.nan                                         # a missing height
    z = run_build.full_level_heights(half, levels)
    qc = {lv: (rng.lognormal(-10, 2, shape) * (rng.random(shape) > 0.3)).astype(np.float32) for lv in levels}
    qc[105][rng.random(shape) < 0.05] = np.nan                       # missing QC on one level
    ghs = []
    for base in (100.0, 1500.0, 3000.0, 9000.0):                     # below ground .. above the top
        g = (base + rng.uniform(-800, 800, shape)).astype(np.float32)
        g[0, :5] = z[120][0, :5].astype(np.float32)                  # on (or a hair off) the lowest level
        g[1, :5] = z[99][1, :5].astype(np.float32)                   # on the top level
        g[2, 0] = np.nan
        ghs.append(g)
    s = run_build.ClmrStream(ghs, lambda lv: run_build._full_level(half, lv), levels)
    for lv in s.levels:
        s.add(lv, qc[lv])
    for g, got in zip(ghs, s.result()):
        want = _clmr_on_a_stack(qc, z, g)
        assert np.array_equal(got, want, equal_nan=True)
        assert np.isfinite(got).any() or np.isnan(want).all()
    assert np.array_equal(run_build.clmr_at_heights(qc, z, ghs[1]), _clmr_on_a_stack(qc, z, ghs[1]), equal_nan=True)


def test_the_cloud_water_stream_holds_two_qc_levels_and_the_lowest_never_all():
    import gc
    import weakref
    levels = list(range(99, 121))
    half = {lv: np.full(3, 20.0 * (121 - lv) ** 1.6, np.float32) for lv in range(99, 122)}
    s = run_build.ClmrStream([np.full(3, 900.0, np.float32)], lambda lv: run_build._full_level(half, lv), levels)
    refs = []
    for lv in s.levels:
        a = np.full(3, 1e-5 * lv, np.float32)
        refs.append(weakref.ref(a))
        s.add(lv, a)
        del a
        gc.collect()
        alive = sum(r() is not None for r in refs)
        assert alive <= 3, alive                                     # the lowest, the previous, the latest
    with pytest.raises(ValueError):
        s.add(98, np.zeros(3, np.float32))                           # one level too many
    s2 = run_build.ClmrStream([np.zeros(3)], lambda lv: run_build._full_level(half, lv), levels)
    with pytest.raises(ValueError):
        s2.add(119, np.zeros(3))                                     # lowest first, always
    with pytest.raises(ValueError):
        s2.result()                                                  # not every level fed
