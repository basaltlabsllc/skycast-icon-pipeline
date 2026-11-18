#!/usr/bin/env python3
# texdev_build.py — the cloud-texture dev feed: upper air (ICU1) and texture
# clues (ICX2) of one ICON run, for the United States and Iceland, under texdev/.
#
# It takes the run a normal build would (the newest whole run on DWD), waits
# for its column and upper-air files the same way, and writes
#   <out>/texdev/upper/<run>/<square>.icu
#   <out>/texdev/clues/<run>/<square>.icx
#   <out>/texdev/heights/<run>/<square>.ich   (cloud heights, when they could be built)
#   <out>/texdev/manifest.json   {run, runMs, builtMs, squares, upper, clues[, heights]}
# The heights (ICH1, pipeline/heights.py: each deck's real base and top) are an
# extra beside the clue files, which they never change: the manifest's
# "heights" key is there only when the run's height files were written.
# The clues entry's "format" names the files' magic ("ICX2": 30 bytes a node,
# with each deck's wind direction; "ICX1" before it had 27). The app reads the
# magic of each file, so a feed of either kind loads.
# and nothing else: no cloud squares, no ICC1 columns, no live path. Its
# history lives in its own state dir: state/upper (run_build.UPPER_SUB) and
# state/texdev_columns (the column heads the clue history is rebuilt from).
import argparse
import functools
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import run_build
from pipeline import dwd
from pipeline.squares import square_name

# The United States and Iceland (Erik, 2026-11-13: "lets start with just the US
# and iceland for now, and then build the whole world later"), as boxes of 5°
# squares by their SW corners, each with about one square of sea or border
# around the land: a view from the coast looks hundreds of km out to sea, and
# the app draws no texture where a square is missing. The margin is about
# 830 km each way (the app's window at Denver took 3 x 4 squares; the first
# 206-square run left Reykjavik's window two squares short to the west:
# a 5-degree square is only 240 km wide at 64 N). The Aleutians west of
# 180° are left out: one square past the dateline would stretch the build's
# single window (run_build.square_window) round the whole globe.
#   (lat from, lat to, lon from, lon to), SW corners, inclusive
TEXDEV_BOXES = (
    (15, 55, -140, -60),    # the lower 48 states
    (45, 75, -180, -115),   # Alaska
    (10, 25, -170, -150),   # Hawaii
    (55, 70, -45, 0),       # Iceland
)
TEXDEV_SQUARES = sorted({square_name(lat, lon)
                         for la0, la1, lo0, lo1 in TEXDEV_BOXES
                         for lat in range(la0, la1 + 5, 5)
                         for lon in range(lo0, lo1 + 5, 5)})
PREFIX = "texdev"
COLUMN_SUB = "texdev_columns"


# The job's time limit (.github/workflows/texdev.yml, timeout-minutes) and what
# the cloud heights must leave of it. The reserve is for what follows them: a
# download already in flight when their time ends (run_build._download gives
# one file up to 4 tries of 120 s), then writing, the publish and the state
# push (under 2 min together in the five newest runs, read 2026-11-17). Those
# runs took 57 to 149 min of the 150, so the heights' own budget is not
# enough by itself: one of them would have had a minute.
JOB_LIMIT_S = 150 * 60
JOB_RESERVE_S = 15 * 60
JOB_START_ENV = "TEXDEV_JOB_START"       # unix seconds, set by this branch's workflow's second step
# The scheduled runs are fired by MAIN's copy of texdev.yml, which builds this
# branch's code and does not set that variable. The job is then taken to have
# begun this long before the build did: checkout, the cdo install, pip, the
# whole test suite and the state fetch come first (10 to 16 min by the review
# of 2026-11-17; the pytest step alone took 7 to 13 min in the five newest runs).
PREBUILD_ALLOWANCE_S = 20 * 60
# The heights' process hands its manifest entry back in this file in the work
# dir: {"entry": {...}} when the files are written, {"entry": null} when it
# gave up cleanly (it printed why). Written whole, then renamed into place.
HEIGHTS_RESULT = "heights-result.json"


def job_start(env, fallback):
    try:
        return float(int(env.get(JOB_START_ENV, "")))
    except ValueError:
        return float(fallback)


def heights_budget(job_start_s, now_s):
    """Seconds the cloud heights may take: their own budget, or what the job
    has left before its reserve when that is less (zero or under: none)."""
    return min(run_build.HEIGHTS_BUDGET_S, job_start_s + JOB_LIMIT_S - JOB_RESERVE_S - now_s)


def manifest(run, built, names, upper, clues, heights=None):
    out = {"run": dwd.run_iso(run), "runMs": int(run.timestamp() * 1000),
           "builtMs": int(built.timestamp() * 1000), "squares": sorted(names),
           "upper": upper, "clues": clues, "attribution": run_build.site.ATTRIBUTION}
    if heights is not None:
        # Absent, never null, when a run has no height files: a reader that
        # finds the key can rely on the files.
        out["heights"] = heights
    return json.dumps(out, separators=(",", ":"))


def _real_heights(blob_dir, state, work, *args, **kw):
    return run_build.build_heights(blob_dir, state, work, run_build.REAL, *args, **kw)


def heights_job_for(deps, blob_dir, state, work, *args, **kw):
    """What the heights' process runs: run_build.build_heights with these
    arguments (those after its `deps`). The real deps hold lambdas, which a
    fresh interpreter cannot be handed, so for them it looks REAL up itself."""
    if deps is run_build.REAL:
        return functools.partial(_real_heights, blob_dir, state, work, *args, **kw)
    return functools.partial(run_build.build_heights, blob_dir, state, work, deps, *args, **kw)


def _heights_child(job, result):
    os.setsid()                              # a process group of its own: cdo and all die with it
    entry = job()
    tmp = result.with_name(result.name + ".tmp")
    tmp.write_text(json.dumps({"entry": entry}))
    os.replace(tmp, result)


def run_heights_child(job, budget_s, result, start="spawn"):
    """Run job() in a child process for at most `budget_s` seconds; its return
    value goes to the file `result`. Returns None when the child exited 0,
    else why it did not. A child still running at the deadline, or one that
    died, is killed with its whole process group, so nothing it started
    outlives it. start: "spawn", a fresh interpreter, never a fork of this
    threaded one (as run_build.clue_cubes); the tests fork, to hand over fakes."""
    result.unlink(missing_ok=True)
    child = mp.get_context(start).Process(target=_heights_child, args=(job, result))
    child.start()
    why = "their process could not be waited for"
    try:
        child.join(budget_s)
        if child.is_alive():
            why = f"out of time: {budget_s / 60:.0f} min were allowed, and their process was stopped"
        elif child.exitcode == 0:
            why = None
        elif child.exitcode < 0:
            why = f"their process was killed by signal {-child.exitcode}"
        else:
            why = f"their process ended with exit code {child.exitcode}"
    finally:
        if why is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass                         # no such group: it never got as far as setsid, or all are gone
            if child.is_alive():
                child.kill()
            child.join()
    return why


def read_heights_result(result, root, run, names, clues):
    """(entry, None) from the child's result file — entry None when the child
    gave up cleanly — or (None, why) when the result is not a whole heights
    build of this run: the entry of this format and folder on the clue files'
    steps, and one file for every square."""
    try:
        got = json.loads(Path(result).read_text())
    except (OSError, ValueError) as e:
        return None, f"no result from their process ({type(e).__name__})"
    if not isinstance(got, dict) or "entry" not in got:
        return None, "malformed result from their process"
    entry = got["entry"]
    if entry is None:
        return None, None
    tiles = run_build.site.heights_path(run)
    if not isinstance(entry, dict) or entry.get("format") != run_build.heights.FORMAT or entry.get("tiles") != tiles \
            or entry.get("steps") != clues["steps"]:
        return None, "malformed result from their process (not this run's entry)"
    lacking = [n for n in names if not (Path(root) / tiles / f"{n}.ich").is_file()]
    if lacking:
        return None, f"malformed result from their process ({len(lacking)} files not written, first {lacking[0]})"
    return entry, None


def drop_heights(root, state, run, head=True):
    """Remove this build's height files and, with `head`, this run's kept head."""
    shutil.rmtree(Path(root) / run_build.site_heights_root(), ignore_errors=True)
    if head:
        d = run_build.history.history_dir(state, run_build.HEIGHTS_SUB)
        for p in d.glob(f"{dwd.run_id(run)}*"):
            try:
                p.unlink()
            except OSError:
                pass


def heights_extra(root, state, work, run, names, clues, budget, job, start="spawn"):
    """The cloud heights, run in a process of their own: the manifest entry,
    or None with a warning and no height file left. Never raises."""
    try:
        short = run_build.heights_too_short(budget)
        if short:
            print(f"::warning::heights skipped: {short}")
            return None
        result = Path(work) / HEIGHTS_RESULT
        why = run_heights_child(job, budget, result, start)
        if why is None:
            entry, why = read_heights_result(result, root, run, names, clues)
            if why is None:
                if entry is None:            # it gave up cleanly and said why; nothing is half-written
                    drop_heights(root, state, run, head=False)
                return entry
    except Exception as e:
        why = f"{type(e).__name__}: {e}"
    try:
        print(f"::warning::heights failed: {why}")
        drop_heights(root, state, run)
    except Exception:
        pass
    return None


def build(out, state, work, deps, steps=None, workers=None, squares=None, env=None, clock=time.time,
          heights_job=None, heights_start="spawn"):
    # heights_job: what the heights' process runs (testing); None for the real
    # thing. heights_start: how that process is started (run_heights_child).
    began = job_start(os.environ if env is None else env, clock() - PREBUILD_ALLOWANCE_S)
    run = dwd.newest_complete_run(deps.listing)
    if run is None:
        print("no complete run on DWD yet")
        return run_build.NOTHING_TO_DO
    steps = list(steps) if steps is not None else dwd.STEPS
    names = list(squares) if squares is not None else list(TEXDEV_SQUARES)
    root, work = Path(out) / PREFIX, Path(work)
    work.mkdir(parents=True, exist_ok=True)
    kit = deps.ensure_kit(work)
    workers = workers or 6
    print(f"texdev: run {dwd.run_iso(run)}, {len(steps)} steps, {len(names)} squares")
    upper, clues = run_build.build_upper_and_clues(root, Path(state), work, deps, kit, run, steps, names, workers,
                                                   column_sub=COLUMN_SUB)
    if upper is None or clues is None:
        print(f"::error::texdev: run {dwd.run_iso(run)} built no upper air — nothing published")
        return run_build.FAILED
    # The build without heights is whole on disk, manifest included, before the
    # heights start. They are an extra, run in a process of their own with a
    # hard deadline: only when that process ends well is the manifest written
    # again with their key, and whatever else happens this run is BUILT.
    built = deps.now()
    (root / "manifest.json").write_text(manifest(run, built, names, upper, clues))
    try:
        now = clock()
        budget = heights_budget(began, now)
        print(f"texdev: heights may take {budget / 60:.0f} min ({(now - began) / 60:.0f} min of the job's "
              f"{JOB_LIMIT_S // 60} used)")
        job = heights_job or heights_job_for(
            deps, root, Path(state), work, kit, run, steps, names,
            [ms // 3600000 for ms in clues["steps"]], workers, budget_s=budget)
        entry = heights_extra(root, Path(state), work, run, names, clues, budget, job, heights_start)
        if entry is not None:
            tmp = root / "manifest.json.tmp"
            try:
                tmp.write_text(manifest(run, built, names, upper, clues, entry))
                os.replace(tmp, root / "manifest.json")
            except Exception:
                tmp.unlink(missing_ok=True)
                drop_heights(root, Path(state), run)
                raise
    except Exception as e:
        print(f"::warning::heights failed: {type(e).__name__}: {e}")
    return run_build.BUILT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="blob", help="texdev/ is written under it")
    ap.add_argument("--state-dir", default="state-texdev")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--steps", default=None, help="comma list of forecast hours (testing)")
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split(",")] if a.steps else None
    sys.exit(build(a.out_dir, a.state_dir, a.work_dir, run_build.REAL, steps))


if __name__ == "__main__":
    main()
