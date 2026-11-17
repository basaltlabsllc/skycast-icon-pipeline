# .github/workflows/columns-measure.yml: the read-only measurement workflow.
# `only` picks what it measures: all / interp run tools/measure_columns.py as
# before, heights runs tools/measure_heights.py. It never publishes: a
# read-only token, no secrets, no Pages/Blob/state step.
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / ".github/workflows/columns-measure.yml").read_text()
CODE = "\n".join(line for line in TEXT.splitlines() if not line.lstrip().startswith("#"))


def measure_script():
    lines = TEXT.splitlines()
    i = next(n for n, line in enumerate(lines) if re.match(r"\s*- name: measure", line))
    i = next(n for n in range(i, len(lines)) if lines[n].strip() == "run: |")
    indent = len(lines[i]) - len(lines[i].lstrip())
    body = []
    for line in lines[i + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)) + "\n"


def run_measure(tmp_path, only):
    """The measure step under bash with python, sudo and the machine probes
    stubbed; returns the python command lines it ran."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "python.log"
    log.write_text("")
    stubs = {"python": f'echo "$@" >> "{log}"\n', "sudo": "exit 0\n", "nproc": "echo 4\n",
             "free": "exit 0\n", "df": "exit 0\n"}
    for name, body in stubs.items():
        p = bin_dir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "USER": "runner"}
    if only is not None:
        env["ONLY"] = only
    out = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", measure_script()],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return log.read_text().splitlines()


@pytest.mark.parametrize("only", ["all", "interp"])
def test_all_and_interp_run_the_column_measurement_as_before(tmp_path, only):
    assert run_measure(tmp_path, only) == [
        f"tools/measure_columns.py --out /mnt/measure --strides 2,4,8 --only {only}"]


@pytest.mark.parametrize("only", [None, ""])
def test_no_choice_is_all(tmp_path, only):
    assert run_measure(tmp_path, only) == ["tools/measure_columns.py --out /mnt/measure --strides 2,4,8 --only all"]


def test_heights_runs_the_height_measurement_instead(tmp_path):
    assert run_measure(tmp_path, "heights") == ["tools/measure_heights.py --out /mnt/measure"]


def test_the_input_names_its_three_choices():
    block = TEXT[TEXT.index("inputs:"):TEXT.index("permissions:")]
    assert "default: all" in block
    for choice in ("all", "interp", "heights"):
        assert re.search(rf"\b{choice} =", block), choice


def test_read_only_and_dispatch_only():
    assert re.search(r"^permissions:\n  contents: read\n", TEXT, re.M)
    assert not re.search(r":\s*write\b", CODE)            # no permission is write
    assert "secrets." not in CODE
    for word in ("schedule:", "push:", "pull_request", "cron:", "git push", "blob", ".mjs", "gh-pages",
                 "upload-pages", "deploy-pages", "gh workflow run"):
        assert word not in CODE, word
    assert re.findall(r"python (tools/\S+|-m pytest)", CODE) == ["-m pytest", "tools/measure_heights.py",
                                                                 "tools/measure_columns.py"]
