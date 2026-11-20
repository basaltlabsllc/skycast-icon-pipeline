import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import hrrr.grib as hg
import hrrr.smokebuild as sb
import hrrr_smoke_build as app
from tests.gribfake import Files, msg

UTC = timezone.utc
RUN = datetime(2026, 11, 20, 3, tzinfo=UTC)
MASS = "8 m above ground"
COL = "entire atmosphere (considered as a single layer)"


def fcst(fh):
    return "anl" if fh == 0 else f"{fh} hour fcst"


def serve_run(run, last):
    f = Files()
    for fh in range(last + 1):
        f.add(hg.file_url(run, fh), [("MASSDEN", MASS, fcst(fh), msg([1e-8, 2e-8, 0.0, 4e-8])),
                                     ("AOTK", COL, fcst(fh), msg([0.1, 0.2, 0.3, 0.4]))])
    return f


def exists_of(urls):
    have = {u + ".idx" for u in urls}
    return lambda u: u in have


def ms(run):
    return int(run.timestamp() * 1000)


def test_wanted_runs():
    now = datetime(2026, 11, 20, 5, 40, tzinfo=UTC)
    ex = exists_of([hg.file_url(datetime(2026, 11, 20, 3, tzinfo=UTC), 18),
                    hg.file_url(datetime(2026, 11, 20, 0, tzinfo=UTC), 48)])
    got = sb.wanted_runs(ex, now)
    assert got == {"short": datetime(2026, 11, 20, 3, tzinfo=UTC), "long": datetime(2026, 11, 20, 0, tzinfo=UTC)}
    ex = exists_of([hg.file_url(datetime(2026, 11, 20, 3, tzinfo=UTC), 18)])
    assert sb.wanted_runs(ex, now)["long"] is None


def test_build_run_writes_and_packs(tmp_path):
    f = serve_run(RUN, 2)
    rel = sb.build_run(RUN, 2, tmp_path, f.get, lambda s: None)
    assert re.fullmatch(r"smoke/2026112003/pack-[0-9a-f]{12}", rel)
    idx = json.loads((tmp_path / rel / "index.json").read_text())
    assert idx["names"] == ["00.smk", "01.smk", "02.smk"]
    assert not list((tmp_path / "smoke" / "2026112003").glob("*.smk"))


def test_build_run_over_budget_leaves_nothing(tmp_path):
    f = serve_run(RUN, 2)
    t = {"now": 0.0}

    def get(url, rng=None):
        if "f01" in url:
            t["now"] = 1000.0
        return f.get(url, rng)

    assert sb.build_run(RUN, 2, tmp_path, get, lambda s: None, budget_s=600,
                        clock=lambda: t["now"], workers=1) is None
    assert not (tmp_path / "smoke").exists()


def test_build_run_failure_leaves_nothing(tmp_path):
    f = serve_run(RUN, 2)

    def get(url, rng=None):
        if "f02" in url:
            raise IOError("boom")
        return f.get(url, rng)

    assert sb.build_run(RUN, 2, tmp_path, get, lambda s: None) is None
    assert not (tmp_path / "smoke").exists()


def test_next_index_rolls_past():
    now = datetime(2026, 11, 20, 6, tzinfo=UTC)
    h = lambda n: now - timedelta(hours=n)
    live = {"short": {"runMs": ms(h(1)), "last": 18, "pack": "smoke/a/pack-1"},
            "long": {"runMs": ms(h(48)), "last": 48, "pack": "smoke/b/pack-2"},
            "past": [{"runMs": ms(h(50)), "pack": "p50"}, {"runMs": ms(h(48)), "pack": "p48"},
                     {"runMs": ms(h(1)), "pack": "p1"}]}
    built = {"short": {"run": h(0), "last": 18, "pack": "p0"}}
    out = sb.next_index(live, built, now)
    assert [e["pack"] for e in out["past"]] == ["p0", "p1", "p48"]
    assert out["short"] == {"runMs": ms(h(0)), "last": 18, "pack": "p0"}
    assert out["long"] == live["long"]
    assert out["format"] == "SMK1" and out["attribution"] == "Smoke: NOAA HRRR"
    assert out["grid"]["nlat"] == 273


def test_next_index_long_replaces_own_short_entry():
    now = datetime(2026, 11, 20, 6, tzinfo=UTC)
    live = {"past": [{"runMs": ms(now), "pack": "short-pack"}]}
    out = sb.next_index(live, {"long": {"run": now, "last": 48, "pack": "long-pack"}}, now)
    assert out["past"] == [{"runMs": ms(now), "pack": "long-pack"}]


NOW = datetime(2026, 11, 20, 5, 40, tzinfo=UTC)
S, L = datetime(2026, 11, 20, 3, tzinfo=UTC), datetime(2026, 11, 20, 0, tzinfo=UTC)
EX = exists_of([hg.file_url(S, 18), hg.file_url(L, 48)])


def run_main(tmp_path, live, build):
    return app.main(["--out", str(tmp_path / "o")], fetch_live=lambda u: live, get=None, exists=EX,
                    now=lambda: NOW, sleep=lambda s: None, build=build)


def test_main_nothing_to_do(tmp_path):
    live = {"short": {"runMs": ms(S), "last": 18, "pack": "x"}, "long": {"runMs": ms(L), "last": 48, "pack": "y"}}
    calls = []
    assert run_main(tmp_path, live, lambda *a: calls.append(a)) == 3
    assert not calls and not (tmp_path / "o").exists()


def test_main_short_only_when_long_fails(tmp_path):
    live = {"long": {"runMs": ms(L) - 6 * 3600 * 1000, "last": 48, "pack": "old-long"}}

    def build(run, last, out, get, sleep, budget, clock):
        return "smoke/2026112003/pack-aaaaaaaaaaaa" if last == 18 else None

    assert run_main(tmp_path, live, build) == 0
    idx = json.loads((tmp_path / "o/smoke/latest.json").read_text())
    assert idx["long"]["pack"] == "old-long"
    assert idx["short"]["pack"] == "smoke/2026112003/pack-aaaaaaaaaaaa"


def test_main_fails_when_nothing_built(tmp_path):
    assert run_main(tmp_path, {}, lambda *a: None) == 1
    assert not (tmp_path / "o/smoke/latest.json").exists()
