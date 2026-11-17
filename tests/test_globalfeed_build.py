import dataclasses
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import globalfeed.build as gb
from globalfeed import cli, history6
from globalfeed.latlon import to_feed
from globalfeed.models import GFS, AIFS
from pipeline import blocks, history
from pipeline.encode import decode_square, quantize
from pipeline.site import tiles_path, tail_path, blocks_path
from pipeline.squares import NX, NY, square_name, slice_square

UTC = timezone.utc
RUN = datetime(2026, 11, 8, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 8, 5, 10, tzinfo=UTC)
SQUARES = [square_name(40, -125), square_name(-5, 175), square_name(85, 0)]


def _small(model, main, tail):
    return dataclasses.replace(model, main_steps=tuple(main), tail_steps=tuple(tail))


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(gb, "all_squares", lambda: list(SQUARES))
    monkeypatch.setattr(gb.history, "HEAD_STEPS", 6)

    def fake_read(model, run, fh, get):
        get(model.index_url(run, fh))        # a failing get must fail the step
        lat, lon = np.meshgrid(np.arange(-90, 90.25, 0.25), np.arange(-180, 180, 0.25), indexing="ij")
        out = {"lat": lat.ravel(), "lon": lon.ravel()}
        for i, f in enumerate(model.fields):
            k = f if isinstance(f, str) else f[0]
            out[k] = ((lat.ravel() + 90) / 1.8 + fh * 2 + i * 7) % 101
        return out

    monkeypatch.setattr(gb, "read_step", fake_read)
    return fake_read


def _get(url, rng=None):
    return b""


def _head(value):
    return np.full((1, 4, NY, NX), value, dtype=np.uint8)


def test_newest_complete_run():
    present = {GFS.index_url(datetime(2026, 11, 7, 18, tzinfo=UTC), GFS.tail_steps[-1])}
    asked = []

    def exists(u):
        asked.append(u)
        return u in present

    assert gb.newest_complete_run(GFS, exists, NOW) == datetime(2026, 11, 7, 18, tzinfo=UTC)
    assert asked[0] == GFS.index_url(RUN, GFS.tail_steps[-1])
    assert gb.newest_complete_run(GFS, lambda u: False, NOW) is None


def test_build_layout_and_manifest(small, tmp_path):
    m = _small(GFS, range(0, 3), (3, 6))
    names = gb.build(m, RUN, tmp_path, _get, NOW, tmp_path / "state", sleep=lambda s: None)
    assert names == SQUARES
    man = json.loads((tmp_path / "manifest.json").read_text())
    base = int(RUN.timestamp() // 3600)
    assert man["tail"]["stepsMs"] == [(base + s) * 3600 * 1000 for s in m.tail_steps]
    assert man["tail"]["tiles"] == tail_path(RUN) and man["tail"]["run"] == man["run"]
    assert man["model"] == m.name and man["attribution"] == m.attribution
    assert "thick" not in man and "columns" not in man
    assert man["blocks"] == blocks_path(RUN) and man["tiles"] == tiles_path(RUN)
    assert man["historySteps"] == 0
    assert man["stepsMs"] == [(base + s) * 3600 * 1000 for s in m.main_steps]
    # Expected cube for one square, computed through the same public pieces.
    lat, lon = np.meshgrid(np.arange(-90, 90.25, 0.25), np.arange(-180, 180, 0.25), indexing="ij")
    for n in SQUARES:
        sl, so = int(n[1:3]) * (1 if n[0] == "N" else -1), int(n[4:]) * (1 if n[3] == "E" else -1)
        icl = decode_square((tmp_path / tiles_path(RUN) / f"{n}.icl").read_bytes())
        icb = blocks.decode_blocks((tmp_path / blocks_path(RUN) / f"{n}.icb").read_bytes(), 4)
        assert (tmp_path / tail_path(RUN) / f"{n}.icl").exists()
        assert np.array_equal(icl["cube"], icb["cube"])
        assert icl["step_hours"] == icb["step_hours"] == [base + s for s in m.main_steps]
        step = fake_cube(m, 1, lat, lon)
        want = quantize(slice_square(step, sl, so)[None], 2)[0]
        assert np.array_equal(icl["cube"][1], want)


def fake_cube(m, fh, lat, lon):
    return np.stack([to_feed(lat.ravel(), lon.ravel(), ((lat.ravel() + 90) / 1.8 + fh * 2 + i * 7) % 101)
                     for i in range(len(m.fields))])


def test_history_gfs(small, tmp_path):
    m = _small(GFS, range(0, 3), (3, 6))
    state = tmp_path / "state"
    earlier = RUN - timedelta(hours=6)
    history.save_run_head(state, earlier, np.zeros((6, 4, NY, NX), dtype=np.uint8), sub=m.name)
    gb.build(m, RUN, tmp_path / "o", _get, NOW, state, sleep=lambda s: None)
    man = json.loads((tmp_path / "o" / "manifest.json").read_text())
    base = int(earlier.timestamp() // 3600)
    assert man["historySteps"] == history.HEAD_STEPS
    assert man["stepsMs"][:history.HEAD_STEPS] == [(base + i) * 3600 * 1000 for i in range(history.HEAD_STEPS)]
    assert len(man["stepsMs"]) == history.HEAD_STEPS + len(m.main_steps)
    assert man["historyRuns"] == []                    # no publication recorded yet
    assert "historyRecordSinceMs" not in man


def test_history_runs_from_the_recorded_publications(small, tmp_path):
    from pipeline import published
    m = _small(AIFS, (0, 6, 12), (18, 24))
    state = tmp_path / "state"
    earlier = int((RUN - timedelta(hours=6)).timestamp() * 1000)
    published.record(state, earlier, earlier + 5 * 3600 * 1000)
    published.record(state, int(RUN.timestamp() * 1000), 1)   # a record of this run itself is not history
    gb.build(m, RUN, tmp_path / "o", _get, NOW, state, sleep=lambda s: None)
    man = json.loads((tmp_path / "o" / "manifest.json").read_text())
    assert man["historyRuns"] == [{"runMs": earlier, "publishedMs": earlier + 5 * 3600 * 1000}]
    assert man["historyRecordSinceMs"] == earlier + 5 * 3600 * 1000


def test_history_aifs(small, tmp_path):
    m = _small(AIFS, (0, 6, 12), (18, 24))
    state = tmp_path / "state"
    ages = (18, 12, 6)
    for a in ages:
        history6.save_run_head6(state, RUN - timedelta(hours=a), _head(a))
    history6.save_run_head6(state, RUN - timedelta(hours=66), _head(66))
    gb.build(m, RUN, tmp_path / "o", _get, NOW, state, sleep=lambda s: None)
    man = json.loads((tmp_path / "o" / "manifest.json").read_text())
    base = int(RUN.timestamp() // 3600)
    assert man["historySteps"] == len(ages)
    assert man["stepsMs"][:3] == [(base - a) * 3600 * 1000 for a in ages]
    gaps = np.diff(man["stepsMs"]) // 3600000
    assert set(gaps) == {6}
    assert list(history6.load_history6(state, RUN)) == [base - a for a in ages]
    # The build saved its own f000 for the next run.
    assert base in history6.load_history6(state, RUN + timedelta(hours=6))


def test_history6_keeps_newest_ten(tmp_path):
    for i in range(12):
        history6.save_run_head6(tmp_path, RUN - timedelta(hours=6 * i), _head(i))
    assert len(list((tmp_path / history6.SUB).glob("*.npz"))) == history.KEEP_RUNS


def test_cli_nothing_to_do(tmp_path):
    iso = RUN.strftime("%Y-%m-%dT%HZ")
    rc = cli.main(GFS, ["--out", str(tmp_path), "--state", str(tmp_path), "--live-manifest", "http://x"],
                  now=NOW, exists=lambda u: True, http_text=lambda u: json.dumps({"run": iso}))
    assert rc == 3


def test_cli_failure_is_1(tmp_path):
    def boom(url, rng=None):
        raise OSError("down")

    rc = cli.main(_small(GFS, range(0, 2), (3,)), ["--out", str(tmp_path), "--state", str(tmp_path)],
                  now=NOW, exists=lambda u: True, get=boom, sleep=lambda s: None)
    assert rc == 1


def test_failed_step_cancels_queued_steps(small, tmp_path):
    m = dataclasses.replace(_small(GFS, range(0, 40), (40,)), workers=1)
    fetched = []

    def get(url, rng=None):
        fetched.append(url)
        if url == m.index_url(RUN, 1):
            raise OSError("boom")
        return b""

    with pytest.raises(OSError):
        gb.build(m, RUN, tmp_path, get, NOW, tmp_path / "state", sleep=lambda s: None)
    assert m.index_url(RUN, 1) in fetched
    assert len(fetched) < len(m.main_steps) // 2   # the queued steps never ran


# --- Main hours the source never posted (GFS 2026-11-17 12Z f119) -----------

import urllib.error


def _missing(m, hours, code=404):
    """A get that answers `code` for the .idx of each hour in `hours`."""
    bad = {m.index_url(RUN, h) for h in hours}

    def get(url, rng=None):
        if url in bad:
            raise urllib.error.HTTPError(url, code, "nope", {}, None)
        return b""
    return get


def test_absent_hour_is_left_out_not_filled(small, tmp_path):
    m = _small(GFS, range(0, 10), (12,))
    gb.build(m, RUN, tmp_path, _missing(m, [7]), NOW, tmp_path / "state", sleep=lambda s: None)
    man = json.loads((tmp_path / "manifest.json").read_text())
    base = int(RUN.timestamp() // 3600)
    present = [s for s in m.main_steps if s != 7]
    assert man["stepsMs"] == [(base + s) * 3600 * 1000 for s in present]
    assert man["absentStepsMs"] == [(base + 7) * 3600 * 1000]
    assert man["complete"] is True
    lat, lon = np.meshgrid(np.arange(-90, 90.25, 0.25), np.arange(-180, 180, 0.25), indexing="ij")
    n = SQUARES[0]
    sl, so = int(n[1:3]) * (1 if n[0] == "N" else -1), int(n[4:]) * (1 if n[3] == "E" else -1)
    icl = decode_square((tmp_path / tiles_path(RUN) / f"{n}.icl").read_bytes())
    icb = blocks.decode_blocks((tmp_path / blocks_path(RUN) / f"{n}.icb").read_bytes(), 4)
    assert icl["step_hours"] == icb["step_hours"] == [base + s for s in present]
    assert np.array_equal(icl["cube"], icb["cube"])
    # Every stored step is the source's own hour: f006 and f008 sit side by
    # side, nothing stands in for f007.
    for i, s in enumerate(present):
        want = quantize(slice_square(fake_cube(m, s, lat, lon), sl, so)[None], 2)[0]
        assert np.array_equal(icl["cube"][i], want), s


def test_no_absent_key_when_every_hour_is_posted(small, tmp_path):
    m = _small(GFS, range(0, 8), (12,))
    gb.build(m, RUN, tmp_path, _get, NOW, tmp_path / "state", sleep=lambda s: None)
    assert "absentStepsMs" not in json.loads((tmp_path / "manifest.json").read_text())


def test_two_adjacent_absent_hours_pass(small, tmp_path):
    m = _small(GFS, range(0, 12), (12,))
    gb.build(m, RUN, tmp_path, _missing(m, [8, 9]), NOW, tmp_path / "state", sleep=lambda s: None)
    man = json.loads((tmp_path / "manifest.json").read_text())
    base = int(RUN.timestamp() // 3600)
    assert man["absentStepsMs"] == [(base + h) * 3600 * 1000 for h in (8, 9)]
    assert len(man["stepsMs"]) == 10


@pytest.mark.parametrize("code", [403, 500, 503])
def test_other_http_errors_still_fail(small, tmp_path, code):
    m = _small(GFS, range(0, 10), (12,))
    with pytest.raises(urllib.error.HTTPError):
        gb.build(m, RUN, tmp_path, _missing(m, [7], code), NOW, tmp_path / "state", sleep=lambda s: None)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("hours", [[0], [3], [9], [6, 7, 8]], ids=["first", "head", "last", "three"])
def test_absent_hours_beyond_the_limits_fail(small, tmp_path, hours):
    m = _small(GFS, range(0, 10), (12,))
    with pytest.raises(gb.AbsentSteps):
        gb.build(m, RUN, tmp_path, _missing(m, hours), NOW, tmp_path / "state", sleep=lambda s: None)
    assert not (tmp_path / "manifest.json").exists()


def test_absent_tail_hour_fails(small, tmp_path):
    m = _small(GFS, range(0, 8), (9, 12))
    with pytest.raises(urllib.error.HTTPError):
        gb.build(m, RUN, tmp_path, _missing(m, [9]), NOW, tmp_path / "state", sleep=lambda s: None)
    assert not (tmp_path / "manifest.json").exists()


def test_absent_hour_retried_before_it_counts(small, tmp_path, monkeypatch):
    m = _small(GFS, range(0, 10), (12,))
    monkeypatch.setattr(gb, "build_thick", lambda *a, **k: None)   # count the cover's own reads
    monkeypatch.setattr(gb, "build_wind", lambda *a, **k: None)
    asked = []
    inner = _missing(m, [7])

    def get(url, rng=None):
        asked.append(url)
        return inner(url, rng)

    gb.build(m, RUN, tmp_path, get, NOW, tmp_path / "state", sleep=lambda s: None)
    assert asked.count(m.index_url(RUN, 7)) == gb.RETRIES + 1


def test_six_hourly_feeds_never_skip():
    steps = list(AIFS.main_steps)
    assert gb._absent_problem(steps, [60], ()) is not None     # a 12 h hole
    assert gb._absent_problem(list(range(0, 121)), [119], range(6)) is None
    assert gb._absent_problem(list(range(0, 121)), [120], range(6)) is not None


def test_thickness_reads_the_posted_hours_and_wind_its_own(small, tmp_path, monkeypatch):
    m = _small(GFS, range(0, 10), (12,))
    seen = {}

    def thick(model, run, out_dir, get, sleep, names):
        seen["thick"] = model.main_steps
        return None

    def wnd(model, run, out_dir, get, sleep, budget_s=0, clock=None):
        seen["wind"] = model.main_steps
        return None

    monkeypatch.setattr(gb, "build_thick", thick)
    monkeypatch.setattr(gb, "build_wind", wnd)
    gb.build(m, RUN, tmp_path, _missing(m, [7]), NOW, tmp_path / "state", sleep=lambda s: None)
    assert seen["thick"] == tuple(s for s in m.main_steps if s != 7)
    assert seen["wind"] == m.main_steps


def test_history_head_saved_with_a_later_hour_absent(small, tmp_path):
    m = _small(GFS, range(0, 10), (12,))
    state = tmp_path / "state"
    gb.build(m, RUN, tmp_path, _missing(m, [7]), NOW, state, sleep=lambda s: None)
    nxt = RUN + timedelta(hours=6)
    found = history.load_history_steps(state, nxt, sub=m.name)
    assert [h for h, _ in found] == [int(RUN.timestamp() // 3600) + s for s in range(6)]
