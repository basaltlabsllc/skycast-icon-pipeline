# The cloud-height files (ICH1) of the texdev feed: which DWD files they read,
# where they are written, what the manifest says, and that a heights failure
# of any kind leaves the upper-air and clue files exactly as a build without
# heights writes them. Nothing here touches the network.
import datetime as dt
import json
import re
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
    return texdev_build.build(tmp_path / out, tmp_path / state, tmp_path / "work", deps,
                              steps=list(steps), workers=2, squares=SQUARES, **kw)


def _manifest(tmp_path, out="out"):
    return json.loads((tmp_path / out / "texdev" / "manifest.json").read_text())


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


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
    _build(tmp_path, _hdeps(steps=steps, calls=calls), steps)
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
def test_a_file_missing_from_dwds_listing_drops_only_the_heights(tmp_path, capsys, unlisted):
    steps = [0, 1]
    calls = []
    assert _build(tmp_path, _hdeps(steps=steps, unlisted=[unlisted], calls=calls), steps) == run_build.BUILT
    out = capsys.readouterr().out
    assert "::warning::heights" in out and "not listed" in out and unlisted.strip("_.").split("_")[-1] in out
    assert not [n for n in calls if "_CLC." in n]             # nothing was downloaded for them
    _same_but_for_heights(tmp_path, steps, capsys, out)


def test_a_broken_file_drops_only_the_heights(tmp_path, capsys):
    steps = [0, 1]
    assert _build(tmp_path, _hdeps(steps=steps, fail="_001_95_CLC."), steps) == run_build.BUILT
    out = capsys.readouterr().out
    assert "::warning::heights failed" in out and "_001_95_CLC." in out
    _same_but_for_heights(tmp_path, steps, capsys, out)
    assert not (tmp_path / "state" / run_build.HEIGHTS_SUB).exists()      # no half run kept as history


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
    # The job's start comes from the workflow; without it, from the build's own start.
    assert texdev_build.job_start({"TEXDEV_JOB_START": "1763373600"}, 5.0) == 1763373600.0
    for bad in ({}, {"TEXDEV_JOB_START": ""}, {"TEXDEV_JOB_START": "soon"}):
        assert texdev_build.job_start(bad, 5.0) == 5.0


def test_heights_run_after_the_other_files_are_written():
    # A heights failure cannot touch what is already on disk, and the budget is
    # measured once the long part of the build is over.
    import inspect
    src = inspect.getsource(texdev_build.build)
    assert src.index("build_upper_and_clues") < src.index("heights_budget") < src.index("build_heights") \
        < src.index("manifest.json")


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


def test_a_height_that_is_not_falling_with_level_fails_the_heights(tmp_path, capsys, monkeypatch):
    # HHL files out of order (or not heights) must never be turned into bases and tops.
    steps = [0, 1]
    real = _agl
    monkeypatch.setitem(globals(), "_agl", lambda half: real(95) if half == 96 else real(half))
    assert _build(tmp_path, _hdeps(steps=steps), steps) == run_build.BUILT
    assert "::warning::heights failed" in capsys.readouterr().out
    assert "heights" not in _manifest(tmp_path)
