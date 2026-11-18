#!/usr/bin/env python3
# tools/time_heights.py — the texdev build's cloud heights, run ALONE on real
# DWD data and not published: the same call texdev_build.build makes (their own
# spawned process with a hard deadline, run_build.build_heights, the build's
# squares, steps and worker count), without the upper air and clues before it.
#
# Why: in a whole build on the runner they did not finish in 30 min (run
# 44221721803), while tools/measure_heights.py had measured 10 min for the same
# files. This tells the two apart: slow here too, and the build's own path is
# slow; quick here, and it is what the upper air leaves behind (memory, disk,
# DWD's patience). run_build.fetch_heights prints where the time goes.
#
# Run by .github/workflows/columns-measure.yml (only = heights-build); it
# writes only under --out and never touches Pages, Blob or state.
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_build                                              # noqa: E402
import texdev_build                                           # noqa: E402
from pipeline import dwd, history                             # noqa: E402

BUILD_WORKERS = 6            # texdev_build.build: workers or 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="measure")
    ap.add_argument("--budget-min", type=float, default=75.0,
                    help="how long the heights may take here (the build allows run_build.HEIGHTS_BUDGET_S)")
    ap.add_argument("--steps", default=None, help="comma list of forecast hours (default: the build's, dwd.STEPS)")
    a = ap.parse_args()
    deps = run_build.REAL
    steps = [int(s) for s in a.steps.split(",")] if a.steps else list(dwd.STEPS)
    names = list(texdev_build.TEXDEV_SQUARES)
    out = Path(a.out)
    root, state, work = out / "blob" / texdev_build.PREFIX, out / "state", out / "work"
    for d in (root, state, work):
        d.mkdir(parents=True, exist_ok=True)
    run = dwd.newest_complete_run(deps.listing)
    if run is None:
        print("no complete run on DWD yet")
        raise SystemExit(1)
    budget = a.budget_min * 60
    print(f"time_heights: run {dwd.run_iso(run)}, {len(steps)} steps, {len(names)} squares, "
          f"{budget / 60:.0f} min allowed (nothing published)")
    kit = deps.ensure_kit(work)
    hours = [history.hour_of(run) + s for s in steps]
    job = texdev_build.heights_job_for(deps, root, state, work, kit, run, steps, names, hours, BUILD_WORKERS,
                                       budget_s=budget)
    t0 = time.monotonic()
    why = texdev_build.run_heights_child(job, budget, work / texdev_build.HEIGHTS_RESULT)
    took = time.monotonic() - t0
    print(f"time_heights: their process took {took:.0f} s ({took / 60:.1f} min): "
          f"{'ended well' if why is None else why}")


if __name__ == "__main__":
    main()
