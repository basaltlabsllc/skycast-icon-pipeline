# The cloud-height files (ICH1) of the texdev feed: which DWD files they read,
# where they are written, what the manifest says, and that a heights failure
# of any kind leaves the upper-air and clue files exactly as a build without
# heights writes them. Nothing here touches the network.
import datetime as dt
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import run_build
import texdev_build
from pipeline import columns, dwd, heights, site
from pipeline.history import HEAD_STEPS, hour_of
from pipeline.squares import NX, NY
from tests.test_run_build import HHL_AGL, HSURF_M
from tests.test_texdev_build import RUN, _deps, _fake_lattice

UTC = dt.timezone.utc
SQUARES = ["N40W125", "N40W120"]


@pytest.fixture(autouse=True)
def _one_clue_process(monkeypatch):
    # The clue files are not under test here (tests/test_texdev_build.py runs
    # them on several processes); one process keeps these builds quick.
    monkeypatch.setenv("CLUE_WORKERS", "1")

# ---- the fake air -----------------------------------------------------------------
# Half levels 117-120 are the column tests' (so the clue files are the same
# with and without heights); above them one every 150 m, and 121 is the ground.
TOP_FULL, GROUND_HALF = dwd.HEIGHT_MLEVELS[0], dwd.HEIGHT_HHL_LEVELS[-1]


def _agl(half):
    if half == GROUND_HALF:
        return 0.0
    return HHL_AGL[half] if half in HHL_AGL else HHL_AGL[117] + 150.0 * (117 - half)


def _cloudy_level(step):
    """The one full level with cloud at `step`: the second lowest on even
    steps (low deck everywhere), the highest read on odd steps (mid deck)."""
    return GROUND_HALF - 2 if step % 2 == 0 else TOP_FULL


def _height_value(name):
    m = re.search(r"_model-level_\d{10}_(\d{3})_(\d+)_CLC\.", name)
    if m:
        step, lv = int(m.group(1)), int(m.group(2))
        return 80.0 if lv == _cloudy_level(step) else 0.0
    m = re.search(r"_time-invariant_\d{10}_(\d+)_HHL\.", name)
    if m:
        return HSURF_M + _agl(int(m.group(1)))
    return None


def _lattice(base, calls=None, fail=None, clc_nan_level=None):
    def lattice_levels(paths, kit, work, stride, nearest=False):
        names = [Path(p).name for p in paths]
        vals = [_height_value(n) for n in names]
        if all(v is None for v in vals) or (len(names) > 1):
            return base(paths, kit, work, stride, nearest=nearest) if nearest else base(paths, kit, work, stride)
        (name,), (v,) = names, vals
        if calls is not None:
            calls.append(name)
        if fail and fail in name:
            raise OSError(f"fake: {name} is broken")
        shape = columns.lattice(np.zeros((NY, NX), np.uint8), stride).shape
        arr = np.full(shape, v, dtype=np.float32)
        if clc_nan_level is not None and f"_{clc_nan_level}_CLC." in name:
            arr[:] = np.nan
        return {"whatever cdo calls it": arr}
    return lattice_levels


def _hdeps(run=RUN, steps=(0, 1), unlisted=(), **fake):
    base = _deps(run=run, steps=steps)
    files = dwd.height_files(run, steps)

    def listing(hh, field):
        if field in files:
            if hh != f"{run:%H}":
                return ""
            return "".join(f'<a href="{n}">x</a>\n' for n in files[field] if not any(u in n for u in unlisted))
        return base.listing(hh, field)

    return base._replace(listing=listing, lattice_levels=_lattice(base.lattice_levels, **fake))


def _build(tmp_path, deps, steps=(0, 1), out="out", state="state", **kw):
    # "fork": the fake deps here are closures, which only a fork can hand to
    # the heights' process. The real build starts a fresh interpreter ("spawn").
    kw.setdefault("heights_start", "fork")
    return texdev_build.build(tmp_path / out, tmp_path / state, tmp_path / "work", deps,
                              steps=list(steps), workers=2, squares=SQUARES, **kw)


def _spawned_job(marker, entry):
    """A heights job a fresh interpreter can import (the real start method)."""
    Path(marker).write_text(f"{os.getpid()} {os.getsid(0)}")
    return entry


def _manifest(tmp_path, out="out"):
    return json.loads((tmp_path / out / "texdev" / "manifest.json").read_text())


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _direct(tmp_path, deps, steps=(0, 1), **kw):
    """run_build.build_heights in THIS process (the build runs it in a child,
    whose lists of calls a test cannot see)."""
    root, state, work = tmp_path / "direct" / "texdev", tmp_path / "direct-state", tmp_path / "direct-work"
    work.mkdir(parents=True, exist_ok=True)
    return run_build.build_heights(root, state, work, deps, Path("/kit"), RUN, list(steps), SQUARES,
                                   [hour_of(RUN) + s for s in steps], 2, **kw)


def _b(metres):
    return int(heights.height_bytes(np.array([metres]))[0])


# ---- DWD's files and the feed's path ------------------------------------------------

def test_height_levels_are_ground_to_5_km():
    assert dwd.HEIGHT_MLEVELS == list(range(86, 121))
    assert dwd.HEIGHT_HHL_LEVELS == list(range(86, 122))          # their bounds: one more


def test_height_files_are_clc_on_model_levels_and_hhl():
    files = dwd.height_files(RUN, [0, 7])
    assert sorted(files) == ["CLC", "HHL"]
    assert files["HHL"] == [dwd.invariant_file_name(RUN, "HHL", lv) for lv in range(86, 122)]
    assert files["CLC"] == [dwd.model_file_name(RUN, s, lv, "CLC") for s in (0, 7) for lv in range(86, 121)]
    assert files["CLC"][0] == "icon_global_icosahedral_model-level_2026111206_000_86_CLC.grib2.bz2"
    assert dwd.dir_url(RUN, "CLC", files["CLC"][0]).startswith("https://opendata.dwd.de/weather/nwp/icon/grib/06/clc/")


def test_missing_height_files_names_what_the_listings_lack():
    files = dwd.height_files(RUN, [0])
    gone = files["CLC"][3]
    pages = {f: "".join(f'<a href="{n}">x</a>\n' for n in names if n != gone) for f, names in files.items()}
    asked = []

    def listing(hh, field):
        asked.append((hh, field))
        return pages[field]

    assert dwd.missing_height_files(listing, RUN, [0]) == [gone]
    assert sorted(asked) == [("06", "CLC"), ("06", "HHL")]
    assert len(dwd.missing_height_files(lambda hh, field: None, RUN, [0])) == 35 + 36


def test_heights_live_beside_the_clues():
    assert site.heights_path(RUN) == "heights/2026111206"
    assert run_build.site_heights_root() == "heights"


# ---- a build with heights -----------------------------------------------------------

def test_the_manifest_names_the_heights(tmp_path):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    m = _manifest(tmp_path)
    e = m["heights"]
    assert e["format"] == "ICH1"
    assert e["run"] == dwd.run_iso(RUN) and e["runMs"] == int(RUN.timestamp() * 1000)
    assert e["tiles"] == f"heights/{dwd.run_id(RUN)}"
    assert e["steps"] == e["stepsMs"] == m["clues"]["steps"]
    assert e["historySteps"] == m["clues"]["historySteps"]
    assert e["spacing"] == m["clues"]["spacing"] and e["side"] == m["clues"]["side"]
    assert e["threshold"] == heights.CLC_THRESHOLD_PCT == 5
    assert e["levels"] == [86, 120]
    assert e["stepM"] == heights.HEIGHT_STEP_M == 50
    assert e["bytes"] == heights.N_BYTES == 5
    assert e["noLevel"] == heights.NO_LEVEL == 255
    assert e["unknown"] == heights.UNKNOWN == 253
    assert e["topAtReachFlag"] == heights.FLAG_TOP_AT_REACH == 1
    root = tmp_path / "out" / "texdev"
    assert sorted(p.name for p in root.iterdir()) == ["clues", "heights", "manifest.json", "upper"]
    assert sorted(p.name for p in (root / e["tiles"]).iterdir()) == sorted(f"{n}.ich" for n in SQUARES)
    # The entries the app reads today are what they were.
    assert m["clues"]["format"] == "ICX2" and m["upper"]["format"] == "ICU1"


def test_the_file_holds_the_clue_files_steps_and_the_decks_of_the_column(tmp_path):
    steps = [0, 1]
    _build(tmp_path, _hdeps(steps=steps), steps)
    root = tmp_path / "out" / "texdev"
    d = heights.decode_height_square((root / f"heights/{dwd.run_id(RUN)}/N40W125.ich").read_bytes())
    from pipeline.cluefile import decode_clue_square
    c = decode_clue_square((root / f"clues/{dwd.run_id(RUN)}/N40W125.icx").read_bytes())
    assert d["step_hours"] == c["step_hours"] == [hour_of(RUN) + s for s in steps]
    assert (d["sw_lat"], d["sw_lon"], d["spacing_mdeg"]) == (40, -125, 250)
    assert d["heights"].shape == c["clues"].shape[:3] + (heights.N_BYTES,)
    # Step 0: cloud on the second lowest level only: the low deck, from that
    # level's lower half level to its upper one, above SEA LEVEL.
    low = GROUND_HALF - 2
    want0 = [_b(HSURF_M + _agl(low + 1)), _b(HSURF_M + _agl(low)), heights.NO_LEVEL, heights.NO_LEVEL, 0]
    assert (d["heights"][0] == np.array(want0, np.uint8)).all()
    # Step 1: cloud on the highest level read only: the mid deck, and its top
    # is the top of the reach.
    want1 = [heights.NO_LEVEL, heights.NO_LEVEL, _b(HSURF_M + _agl(TOP_FULL + 1)), _b(HSURF_M + _agl(TOP_FULL)),
             heights.FLAG_TOP_AT_REACH]
    assert (d["heights"][1] == np.array(want1, np.uint8)).all()
    assert want0[0] < want0[1] < want1[2] < want1[3] <= heights.MAX_HEIGHT_BYTE


def test_each_model_level_file_is_decoded_on_its_own(tmp_path):
    calls = []
    steps = [0, 1]
    assert _direct(tmp_path, _hdeps(steps=steps, calls=calls), steps) is not None
    files = dwd.height_files(RUN, steps)
    clc = [n for n in calls if "_CLC." in n]
    assert sorted(clc) == sorted(files["CLC"])
    assert set(files["HHL"]) <= set(calls)


def test_a_level_without_cover_is_unknown_never_no_cloud(tmp_path):
    # Level 100 comes back all missing: whichever deck owns it was not read.
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps, clc_nan_level=100), steps) == run_build.BUILT
    e = _manifest(tmp_path)["heights"]
    d = heights.decode_height_square((tmp_path / "out" / "texdev" / e["tiles"] / "N40W125.ich").read_bytes())
    rec = d["heights"][0, 0, 0]
    assert heights.UNKNOWN in rec[:4].tolist()
    # ... and that deck's base and top are both unknown, not "no level".
    k = rec[:4].tolist().index(heights.UNKNOWN)
    assert rec[k ^ 1] == heights.UNKNOWN


# ---- heights never cost the run its other files ---------------------------------------

def _plain(tmp_path, steps):
    """What the build writes with no height files on DWD at all."""
    assert _build(tmp_path, _deps(steps=steps), steps, out="plain", state="state-plain") == run_build.BUILT
    return _files(tmp_path / "plain" / "texdev")


def _same_but_for_heights(tmp_path, steps, capsys, why):
    got = _files(tmp_path / "out" / "texdev")
    want = _plain(tmp_path, steps)
    assert sorted(got) == sorted(want)                        # no heights file, nothing else gone
    for name in want:
        if name != "manifest.json":
            assert got[name] == want[name], name
    m = _manifest(tmp_path)
    assert "heights" not in m
    assert m == json.loads(want["manifest.json"])
    assert not (tmp_path / "out" / "texdev" / "heights").exists()
    return m


def test_with_heights_the_other_files_are_byte_for_byte_the_same(tmp_path):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    got, want = _files(tmp_path / "out" / "texdev"), _plain(tmp_path, steps)
    for name in want:
        if name != "manifest.json":
            assert got[name] == want[name], name
    m, p = _manifest(tmp_path), json.loads(want["manifest.json"])
    assert {k: v for k, v in m.items() if k != "heights"} == p
    assert "heights" not in p


@pytest.mark.parametrize("unlisted", ["_000_100_CLC.", "_90_HHL."])
def test_a_file_missing_from_dwds_listing_drops_only_the_heights(tmp_path, capfd, unlisted):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps, unlisted=[unlisted]), steps) == run_build.BUILT
    out = capfd.readouterr().out
    assert "::warning::heights" in out and "not listed" in out and unlisted.strip("_.").split("_")[-1] in out
    _same_but_for_heights(tmp_path, steps, capfd, out)
    calls = []
    assert _direct(tmp_path, _hdeps(steps=steps, unlisted=[unlisted], calls=calls), steps) is None
    assert not [n for n in calls if "_CLC." in n or "_HHL." in n]     # nothing was downloaded for them


def test_a_broken_file_drops_only_the_heights(tmp_path, capfd):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps, fail="_001_95_CLC."), steps) == run_build.BUILT
    out = capfd.readouterr().out
    assert "::warning::heights failed" in out and "_001_95_CLC." in out
    _same_but_for_heights(tmp_path, steps, capfd, out)
    assert not list((tmp_path / "state").glob(f"{run_build.HEIGHTS_SUB}/*.npz"))   # no half run kept as history


def test_out_of_time_before_it_starts_drops_only_the_heights(tmp_path, capsys):
    steps = [0, 1]
    calls = []
    late = texdev_build.JOB_LIMIT_S - texdev_build.JOB_RESERVE_S          # the job began this long ago
    assert _build(tmp_path, _hdeps(steps=steps, calls=calls), steps,
                  env={"TEXDEV_JOB_START": "1000"}, clock=lambda: 1000.0 + late) == run_build.BUILT
    out = capsys.readouterr().out
    assert "::warning::heights skipped" in out and "no time left" in out
    assert not [n for n in calls if "_CLC." in n]
    _same_but_for_heights(tmp_path, steps, capsys, out)


def test_running_out_of_time_midway_drops_only_the_heights(tmp_path, capsys):
    # Each file read moves the clock a minute: the budget ends partway through.
    steps = [0, 1]
    calls = []
    deps = _hdeps(steps=steps, calls=calls)
    root, state, work = tmp_path / "out" / "texdev", tmp_path / "state", tmp_path / "work"
    work.mkdir(parents=True)
    hours = [hour_of(RUN) + s for s in steps]
    t = {"now": 0.0}
    inner = deps.lattice_levels

    def slow(paths, kit, work, stride, nearest=False):
        t["now"] += 60.0
        return inner(paths, kit, work, stride)

    got = run_build.build_heights(root, state, work, deps._replace(lattice_levels=slow), Path("/kit"), RUN, steps,
                                  SQUARES, hours, 2, budget_s=40 * 60.0, clock=lambda: t["now"])
    assert got is None
    out = capsys.readouterr().out
    assert "::warning::heights failed" in out and "out of time" in out
    assert len(calls) < 36 + 2 * 35                           # it stopped reading
    assert not (root / "heights").exists()
    assert not (state / run_build.HEIGHTS_SUB).exists()


def test_the_budget_is_the_smaller_of_its_own_and_what_the_job_has_left():
    own, limit, reserve = run_build.HEIGHTS_BUDGET_S, texdev_build.JOB_LIMIT_S, texdev_build.JOB_RESERVE_S
    assert own == 30 * 60 and reserve == 15 * 60
    assert texdev_build.heights_budget(1000.0, 1000.0 + 60 * 60) == own                    # plenty left
    assert texdev_build.heights_budget(1000.0, 1000.0 + limit - reserve - 600) == 600      # ten minutes left
    assert texdev_build.heights_budget(1000.0, 1000.0 + limit) == -reserve                 # none
    # The job's start comes from the workflow; without it, from what the build hands in.
    assert texdev_build.job_start({"TEXDEV_JOB_START": "1763373600"}, 5.0) == 1763373600.0
    for bad in ({}, {"TEXDEV_JOB_START": ""}, {"TEXDEV_JOB_START": "soon"}):
        assert texdev_build.job_start(bad, 5.0) == 5.0


def test_heights_run_after_everything_else_is_written_the_manifest_included(tmp_path):
    # What the heights' own process finds on disk when it starts is a whole
    # build without heights: whatever then happens to it costs nothing.
    steps = [0, 1]
    seen = tmp_path / "seen.json"
    root = tmp_path / "out" / "texdev"

    def job():
        seen.write_text(json.dumps({"manifest": json.loads((root / "manifest.json").read_text()),
                                    "files": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
                                    "pid": os.getpid()}))
        return None

    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=job) == run_build.BUILT
    got = json.loads(seen.read_text())
    want = _plain(tmp_path, steps)
    assert got["files"] == sorted(want)
    assert got["manifest"] == json.loads(want["manifest.json"]) and "heights" not in got["manifest"]
    assert got["pid"] != os.getpid()                              # their own process
    assert _manifest(tmp_path) == got["manifest"]


# ---- the heights' own process: a hard stop, and nothing it does costs the run -----------

def _dropped(tmp_path, steps, out, why):
    assert "::warning::heights failed" in out and why in out, out
    _same_but_for_heights(tmp_path, steps, None, out)
    assert not list((tmp_path / "state").glob(f"{run_build.HEIGHTS_SUB}/{dwd.run_id(RUN)}*"))


def _litter(tmp_path):
    """What a heights build leaves when it dies partway: files and a state head."""
    tiles = tmp_path / "out" / "texdev" / site.heights_path(RUN)
    tiles.mkdir(parents=True)
    (tiles / f"{SQUARES[0]}.ich").write_bytes(b"ICH1 half a file")
    head = tmp_path / "state" / run_build.HEIGHTS_SUB
    head.mkdir(parents=True, exist_ok=True)
    (head / f"{dwd.run_id(RUN)}.npz").write_bytes(b"half a head")


def test_a_heights_process_that_exits_non_zero_drops_only_the_heights(tmp_path, capfd):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=lambda: os._exit(3)) == run_build.BUILT
    _dropped(tmp_path, steps, capfd.readouterr().out, "exit code 3")


def test_a_heights_process_killed_by_a_signal_drops_only_the_heights(tmp_path, capfd):
    steps = [0, 1]
    job = lambda: os.kill(os.getpid(), signal.SIGKILL)
    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=job) == run_build.BUILT
    _dropped(tmp_path, steps, capfd.readouterr().out, f"signal {int(signal.SIGKILL)}")


def test_a_heights_process_that_writes_files_and_dies_leaves_none_of_them(tmp_path, capfd):
    steps = [0, 1]

    def job():
        _litter(tmp_path)
        os._exit(1)

    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=job) == run_build.BUILT
    _dropped(tmp_path, steps, capfd.readouterr().out, "exit code 1")


def test_a_heights_process_that_raises_drops_only_the_heights(tmp_path, capfd):
    steps = [0, 1]

    def job():
        _litter(tmp_path)
        raise MemoryError("fake")

    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=job) == run_build.BUILT
    _dropped(tmp_path, steps, capfd.readouterr().out, "exit code 1")


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_a_heights_process_that_outlives_its_time_is_killed_with_everything_it_started(tmp_path, capfd, monkeypatch):
    steps = [0, 1]
    monkeypatch.setattr(run_build, "HEIGHTS_BUDGET_S", 0.5)
    monkeypatch.setattr(run_build, "HEIGHTS_MIN_BUDGET_S", 0.0)
    pids = tmp_path / "pids"

    def job():
        _litter(tmp_path)
        hung = subprocess.Popen(["sleep", "600"])                 # a cdo that never returns
        pids.write_text(f"{os.getpid()} {hung.pid}")
        time.sleep(600)

    t0 = time.monotonic()
    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=job) == run_build.BUILT
    took = time.monotonic() - t0
    out = capfd.readouterr().out
    child, hung = (int(p) for p in pids.read_text().split())
    until = time.monotonic() + 5
    while not (_gone(child) and _gone(hung)) and time.monotonic() < until:
        time.sleep(0.05)
    assert _gone(child) and _gone(hung)
    _dropped(tmp_path, steps, out, "out of time")
    assert took < 60                                              # nowhere near the 600 s it wanted


def test_the_real_build_starts_the_heights_in_a_fresh_interpreter(tmp_path, monkeypatch):
    import functools
    import inspect
    import pickle
    assert inspect.signature(texdev_build.build).parameters["heights_start"].default == "spawn"
    assert inspect.signature(texdev_build.run_heights_child).parameters["start"].default == "spawn"
    # What the real build hands that interpreter can be handed over (pickled),
    # and is build_heights with the real deps and the build's own arguments.
    args = (tmp_path / "blob", tmp_path / "state", tmp_path / "work", Path("/kit"), RUN, [0, 1], SQUARES, [1, 2], 6)
    job = pickle.loads(pickle.dumps(texdev_build.heights_job_for(run_build.REAL, *args, budget_s=900.0)))
    got = []
    monkeypatch.setattr(run_build, "build_heights", lambda *a, **kw: got.append((a, kw)) or {"an": "entry"})
    assert job() == {"an": "entry"}
    ((a, kw),) = got
    assert a[:3] == args[:3] and a[3] is run_build.REAL and a[4:] == args[3:] and kw == {"budget_s": 900.0}
    monkeypatch.undo()
    names = list(inspect.signature(run_build.build_heights).parameters)[:10]
    assert names == ["blob_dir", "state", "work", "deps", "kit", "run", "steps", "squares", "hours", "workers"]
    # ... and a job run that way hands its result back from a session of its own.
    marker, result = tmp_path / "marker", tmp_path / "result.json"
    job = functools.partial(_spawned_job, str(marker), {"format": "ICH1"})
    assert texdev_build.run_heights_child(job, 300.0, result) is None
    assert json.loads(result.read_text()) == {"entry": {"format": "ICH1"}}
    pid, sid = (int(x) for x in marker.read_text().split())
    assert pid == sid and pid != os.getpid()


@pytest.mark.parametrize("what", ["not a dict", "another format", "other steps", "files missing", "no result"])
def test_a_result_that_is_not_what_was_asked_for_drops_only_the_heights(tmp_path, capfd, what):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps), steps, out="good", state="good-state") == run_build.BUILT
    entry = _manifest(tmp_path, "good")["heights"]
    capfd.readouterr()

    def job():
        if what == "files missing":
            return entry                                          # a fine entry, and no file written
        _litter(tmp_path)
        if what == "no result":
            os._exit(0)                                           # "fine", and nothing to show for it
        if what == "not a dict":
            return ["ICH1"]
        if what == "another format":
            return dict(entry, format="ICH9")
        return dict(entry, steps=entry["steps"][:-1])

    assert _build(tmp_path, _hdeps(steps=steps), steps, heights_job=job) == run_build.BUILT
    _dropped(tmp_path, steps, capfd.readouterr().out, "result")


def test_nothing_that_goes_wrong_around_the_heights_process_costs_the_run(tmp_path, capfd, monkeypatch):
    steps = [0, 1]

    def broken(job, budget_s, result, start):
        _litter(tmp_path)
        raise RuntimeError("fake: could not start a process")

    with monkeypatch.context() as patch:
        patch.setattr(texdev_build, "run_heights_child", broken)
        assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    _dropped(tmp_path, steps, capfd.readouterr().out, "could not start a process")


def test_an_error_after_the_height_files_are_written_still_only_drops_the_heights(tmp_path, capsys, monkeypatch):
    # The summary line and the manifest entry come after the files: an error
    # there is a heights failure like any other, never one that escapes.
    def boom(x):
        raise TypeError("fake: the entry cannot be built")

    monkeypatch.setattr(run_build, "_whole", boom)
    assert _direct(tmp_path, _hdeps()) is None
    out = capsys.readouterr().out
    assert "::warning::heights failed" in out and "the entry cannot be built" in out
    assert not (tmp_path / "direct" / "texdev" / "heights").exists()


# ---- time -------------------------------------------------------------------------------

def _budget_given(tmp_path, monkeypatch, env, clock, out="out"):
    """The budget the heights' process is started with (None: never started)."""
    got = []

    def spy(job, budget_s, result, start):
        got.append(budget_s)
        return "fake: not run"

    monkeypatch.setattr(texdev_build, "run_heights_child", spy)
    assert _build(tmp_path, _hdeps(), out=out, state=f"{out}-state", env=env, clock=clock) == run_build.BUILT
    return got[0] if got else None


def _clock(t0, later):
    """The build's first look at the clock is t0; every later one, `later`."""
    seen = []

    def clock():
        seen.append(1)
        return t0 if len(seen) == 1 else later

    return clock


def test_without_the_workflows_clock_the_job_is_taken_to_have_begun_20_min_before_the_build(tmp_path, monkeypatch):
    # The scheduled runs (main's texdev.yml) do not set TEXDEV_JOB_START. The
    # build then starts after checkout, cdo, pip and the whole test suite.
    limit, reserve, pre = texdev_build.JOB_LIMIT_S, texdev_build.JOB_RESERVE_S, texdev_build.PREBUILD_ALLOWANCE_S
    assert pre == 20 * 60
    t0, phase = 1_763_373_600.0, 100 * 60.0                       # upper air and clues took 100 min
    got = _budget_given(tmp_path, monkeypatch, {}, _clock(t0, t0 + phase))
    # It ends 15 min before the limit counted from 20 min before the build began.
    assert t0 + phase + got == (t0 - pre) + limit - reserve
    assert got == 15 * 60
    # Told when the job began (here: just before the build), it has its whole own budget.
    got = _budget_given(tmp_path, monkeypatch, {"TEXDEV_JOB_START": str(int(t0))}, _clock(t0, t0 + phase), out="told")
    assert got == run_build.HEIGHTS_BUDGET_S


def test_a_budget_too_small_to_finish_is_not_spent(tmp_path, capsys, monkeypatch):
    need = run_build.HEIGHTS_MIN_BUDGET_S
    assert need == 13 * 60
    limit, reserve = texdev_build.JOB_LIMIT_S, texdev_build.JOB_RESERVE_S
    env = {"TEXDEV_JOB_START": "1000"}
    at = lambda left: (lambda: 1000.0 + limit - reserve - left)   # `left` seconds of budget remain
    assert _budget_given(tmp_path, monkeypatch, env, at(need - 1)) is None
    out = capsys.readouterr().out
    assert "::warning::heights skipped" in out and "no time left" in out and "13 min" in out
    assert "heights" not in _manifest(tmp_path)
    assert _budget_given(tmp_path, monkeypatch, env, at(need), out="enough") == need
    # build_heights called by itself refuses the same way, reading nothing.
    calls = []
    assert _direct(tmp_path, _hdeps(calls=calls), budget_s=need - 1) is None
    assert calls == [] and "no time left" in capsys.readouterr().out


# ---- history ----------------------------------------------------------------------------

def test_height_history_is_kept_and_read_back(tmp_path):
    steps = list(range(HEAD_STEPS))
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    assert (tmp_path / "state" / run_build.HEIGHTS_SUB / f"{dwd.run_id(RUN)}.npz").exists()
    first = heights.decode_height_square(
        (tmp_path / "out" / "texdev" / f"heights/{dwd.run_id(RUN)}/N40W125.ich").read_bytes())
    nxt = RUN + dt.timedelta(hours=6)
    assert _build(tmp_path, _hdeps(run=nxt, steps=[0, 1]), [0, 1], out="out2") == run_build.BUILT
    m = _manifest(tmp_path, "out2")
    want = [hour_of(RUN) + s for s in range(HEAD_STEPS)] + [hour_of(nxt) + s for s in (0, 1)]
    assert m["heights"]["steps"] == [h * 3600 * 1000 for h in want] == m["clues"]["steps"]
    assert m["heights"]["historySteps"] == HEAD_STEPS
    d = heights.decode_height_square((tmp_path / "out2" / "texdev" / m["heights"]["tiles"] / "N40W125.ich").read_bytes())
    assert d["step_hours"] == want
    assert np.array_equal(d["heights"][:HEAD_STEPS], first["heights"])
    assert (d["heights"][:HEAD_STEPS, ..., :4] != heights.UNKNOWN).all()


def test_history_hours_with_no_heights_kept_are_unknown(tmp_path):
    # The first build had no height files; the second has them. Its file still
    # carries the clue file's history hours, every one marked unknown.
    steps = list(range(HEAD_STEPS))
    assert _build(tmp_path, _deps(steps=steps), steps) == run_build.BUILT
    nxt = RUN + dt.timedelta(hours=6)
    assert _build(tmp_path, _hdeps(run=nxt, steps=[0, 1]), [0, 1], out="out2") == run_build.BUILT
    m = _manifest(tmp_path, "out2")
    assert m["heights"]["steps"] == m["clues"]["steps"] and len(m["clues"]["steps"]) == HEAD_STEPS + 2
    d = heights.decode_height_square((tmp_path / "out2" / "texdev" / m["heights"]["tiles"] / "N40W125.ich").read_bytes())
    assert (d["heights"][:HEAD_STEPS, ..., :4] == heights.UNKNOWN).all()
    assert (d["heights"][:HEAD_STEPS, ..., 4] == 0).all()
    assert (d["heights"][HEAD_STEPS:, ..., :4] != heights.UNKNOWN).all()


def test_history_kept_under_another_rule_is_not_read(tmp_path, monkeypatch):
    steps = list(range(HEAD_STEPS))
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    monkeypatch.setattr(heights, "CLC_THRESHOLD_PCT", 10.0)
    nxt = RUN + dt.timedelta(hours=6)
    assert _build(tmp_path, _hdeps(run=nxt, steps=[0, 1]), [0, 1], out="out2") == run_build.BUILT
    m = _manifest(tmp_path, "out2")
    assert m["heights"]["threshold"] == 10
    d = heights.decode_height_square((tmp_path / "out2" / "texdev" / m["heights"]["tiles"] / "N40W125.ich").read_bytes())
    assert (d["heights"][:HEAD_STEPS, ..., :4] == heights.UNKNOWN).all()


def test_an_unreadable_kept_head_is_dropped_and_its_hours_are_unknown(tmp_path, capfd):
    steps = list(range(HEAD_STEPS))
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    d = tmp_path / "state" / run_build.HEIGHTS_SUB
    old = d / f"{dwd.run_id(RUN)}.npz"
    old.write_bytes(old.read_bytes()[:200])                       # cut short: not a zip any more
    capfd.readouterr()
    nxt = RUN + dt.timedelta(hours=6)
    for out in ("out2", "out3"):                                  # ... and it does not fail the next run either
        assert _build(tmp_path, _hdeps(run=nxt, steps=steps), steps, out=out) == run_build.BUILT
        m = _manifest(tmp_path, out)
        assert m["heights"]["steps"] == m["clues"]["steps"] and len(m["clues"]["steps"]) == 2 * HEAD_STEPS
        got = heights.decode_height_square((tmp_path / out / "texdev" / m["heights"]["tiles"] / "N40W125.ich").read_bytes())
        assert (got["heights"][:HEAD_STEPS, ..., :4] == heights.UNKNOWN).all()
        assert (got["heights"][HEAD_STEPS:, ..., :4] != heights.UNKNOWN).all()
        text = capfd.readouterr().out
        assert ("::warning::heights: kept head" in text and old.name in text) == (out == "out2")
    assert not old.exists()
    assert [p.name for p in d.glob("*.npz")] == [f"{dwd.run_id(nxt)}.npz"]
    with np.load(d / f"{dwd.run_id(nxt)}.npz") as z:
        assert z["head"].shape[0] == HEAD_STEPS


def test_a_height_that_is_not_falling_with_level_fails_the_heights(tmp_path, capfd, monkeypatch):
    # HHL files out of order (or not heights) must never be turned into bases and tops.
    steps = [0, 1]
    real = _agl
    monkeypatch.setitem(globals(), "_agl", lambda half: real(95) if half == 96 else real(half))
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    assert "::warning::heights failed" in capfd.readouterr().out
    assert "heights" not in _manifest(tmp_path)
