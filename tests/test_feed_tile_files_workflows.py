# FEED_TILE_FILES (pipeline/pack.py) is set explicitly to "1" on the step that
# runs each feed's build, in all four Blob feeds' workflows: both layouts (per
# file and packs) go up until Erik flips it, after the app that reads packs is
# on phones. The packing never changes a workflow's time limits.
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# workflow -> (name of the step that runs the build, its build command, time limit it must keep)
STEPS = {
    "build": ("name: build", "python run_build.py", "timeout-minutes: 150"),
    "gfs": ("name: watch GFS, build each new run", "timeout -k 10 2700 python gfs_build.py", "timeout-minutes: 360"),
    "aifs": ("name: watch AIFS, build each new run", "timeout -k 10 2700 python aifs_build.py", "timeout-minutes: 360"),
    "ifs": ("name: watch IFS, build each new run", "timeout -k 10 2700 python ifs_build.py", "timeout-minutes: 360"),
}


def _step(text, name):
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.strip().endswith(name) and "- " in line)
    indent = len(lines[i]) - len(lines[i].lstrip())
    out = [lines[i]]
    for line in lines[i + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("wf", sorted(STEPS))
def test_feed_tile_files_is_on_for_the_build_step(wf):
    name, command, limit = STEPS[wf]
    text = (ROOT / f".github/workflows/{wf}.yml").read_text()
    step = _step(text, name)
    assert command in step
    env = step.split("run: |")[0]
    assert 'FEED_TILE_FILES: "1"' in env
    assert text.count("FEED_TILE_FILES") == 1
    assert limit in text


def test_hrrr_time_limit_untouched():
    assert "timeout -k 10 1200" in (ROOT / ".github/workflows/hrrr.yml").read_text()


def test_globalfeed_budget_matches_the_workflows():
    import globalfeed.build as gb
    assert gb.WORKFLOW_TIMEOUT_S == 2700
