# hrrr/smokebuild.py — the smoke-only pass: HRRR near-ground smoke (hrrr/smoke.py)
# for the newest short (18 h, any cycle) and long (48 h, 00/06/12/18Z) runs, one
# SMK1 file per forecast hour, packed per run, plus smoke/latest.json.
#
#   smoke/<YYYYMMDDHH>/pack-<h>/{all.bin,index.json}   00.smk ... <last>.smk
#   smoke/latest.json                                  see next_index
#
# "past" keeps every run built in the last KEEP_PAST_H hours, so the app can show
# the hours that already happened from the run that was current then.
import concurrent.futures as cf
import datetime as dt
import os
import shutil
import time
from pathlib import Path

from pipeline.pack import try_pack

from .build import RETRIES
from .grib import file_url
from .smoke import GRID, encode_aod, encode_pm25, lattice, node_index, read_smoke, smk_bytes

SMOKE_BUDGET_S = 600
KEEP_PAST_H = 49
ATTRIBUTION = "Smoke: NOAA HRRR"
SHORT_LAST, LONG_LAST = 18, 48


class _OverBudget(Exception):
    pass


def run_ms(run):
    return int(run.timestamp() * 1000)


def smoke_rel(run):
    return f"smoke/{run.strftime('%Y%m%d%H')}"


def wanted_runs(exists, now):
    """{"short": run | None, "long": run | None}. short: the newest of the last
    four cycle hours whose f18 idx exists. long: the newest 00/06/12/18Z run of
    the last 12 h whose f48 idx exists."""
    top = now.replace(minute=0, second=0, microsecond=0)
    short = long = None
    for back in range(4):
        run = top - dt.timedelta(hours=back)
        if exists(file_url(run, SHORT_LAST) + ".idx"):
            short = run
            break
    for back in range(13):
        run = top - dt.timedelta(hours=back)
        if run.hour % 6 == 0 and exists(file_url(run, LONG_LAST) + ".idx"):
            long = run
            break
    return {"short": short, "long": long}


def _read_step(run, fh, get, sleep):
    for attempt in range(RETRIES + 1):
        try:
            return read_smoke(run, fh, get)
        except Exception:
            if attempt == RETRIES:
                raise
            sleep(2 * (attempt + 1))


def build_run(run, last_fh, out_dir, get, sleep=time.sleep, budget_s=SMOKE_BUDGET_S,
              clock=time.monotonic, workers=None):
    """Write out_dir/smoke/<YYYYMMDDHH>/<fh:02d>.smk for f00..last_fh, pack the
    folder and remove the loose files. Returns the pack's path relative to
    out_dir, or None (any failure, or more than budget_s seconds) with nothing
    left behind."""
    out_dir = Path(out_dir)
    rel = smoke_rel(run)
    folder = out_dir / rel
    if workers is None:
        workers = int(os.environ.get("HRRR_WORKERS", "8"))
    t0 = clock()
    deadline = t0 + budget_s
    base_hour = int(run.timestamp() // 3600)

    def one(fh):
        if clock() > deadline:
            raise _OverBudget()
        d = _read_step(run, fh, get, sleep)
        idx = node_index(d["lat"], d["lon"])
        raw = smk_bytes(base_hour + fh,
                        encode_pm25(lattice(d["pm25"], idx)),
                        encode_aod(lattice(d["aod"], idx)))
        (folder / f"{fh:02d}.smk").write_bytes(raw)

    pack = None
    ex = cf.ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        try:
            futures = [ex.submit(one, fh) for fh in range(last_fh + 1)]
            for f in futures:
                try:
                    f.result(timeout=max(0.0, deadline - clock()))
                except cf.TimeoutError:
                    raise _OverBudget()
        except BaseException:
            ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
        if clock() > deadline:
            raise _OverBudget()
        pack = try_pack(out_dir, rel)
        if pack is None:
            raise RuntimeError("pack failed")
        for f in folder.glob("*.smk"):
            f.unlink()
    except Exception as e:
        if isinstance(e, _OverBudget):
            print(f"::warning::smoke {rel}: over budget ({clock() - t0:.0f} s), dropped")
        else:
            print(f"::warning::smoke {rel} failed: {type(e).__name__}: {e}")
        shutil.rmtree(folder, ignore_errors=True)
        pack = None
    # leave no empty smoke/ behind
    for d in (out_dir / "smoke",):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    if pack:
        print(f"smoke {rel}: f00..f{last_fh:02d}, {clock() - t0:.0f} s")
    return pack


def next_index(live, built, now):
    """The new smoke/latest.json. live is the live index ({} if none); built is
    {"short": {"run", "last", "pack"}, "long": {...}} for what this pass built
    (a key absent or None when not built). short and long keep the live entry
    unless rebuilt; past is the live past plus every run built now, within
    KEEP_PAST_H of now, one entry per runMs (a rebuild replaces), newest first."""
    out = {"format": "SMK1", "grid": dict(GRID), "attribution": ATTRIBUTION}
    for kind in ("short", "long"):
        b = (built or {}).get(kind)
        if b:
            out[kind] = {"runMs": run_ms(b["run"]), "last": b["last"], "pack": b["pack"]}
        elif live.get(kind):
            out[kind] = live[kind]
    past = {}
    for e in live.get("past") or []:
        past[e["runMs"]] = {"runMs": e["runMs"], "pack": e["pack"]}
    for b in (built or {}).values():
        if b:
            past[run_ms(b["run"])] = {"runMs": run_ms(b["run"]), "pack": b["pack"]}
    cutoff = run_ms(now) - KEEP_PAST_H * 3600 * 1000
    out["past"] = sorted((e for e in past.values() if e["runMs"] >= cutoff),
                         key=lambda e: -e["runMs"])
    return out
