#!/usr/bin/env python3
# tools/measure_heights.py — what CLOUD HEIGHTS per deck would cost the texture
# feed, measured on REAL DWD data and not published: for every node and step
# the texdev build covers, ICON's cloud cover on MODEL LEVELS (CLC) from the
# ground to 5 km above it, with the half-level heights (HHL), turned into the
# feed's five bytes (pipeline/heights.py: low-deck base, low-deck top, mid-deck
# base, mid-deck top in 50 m steps above sea level, and a flags byte). The
# decks, the bytes and the packing are the feed's own, imported from there.
#
# It reports which files DWD lists and their size, the download and build
# time (a timed sample, then every step with the texdev build's worker count),
# the bytes each square would grow by, and one decoded column as a sanity
# check. Run by .github/workflows/columns-measure.yml (only = heights); it
# writes only under --out and never touches Pages, Blob or state.
import argparse
import concurrent.futures as cf
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_build                                              # noqa: E402
import texdev_build                                           # noqa: E402
from pipeline import columns, dwd, heights, history           # noqa: E402
from pipeline.heights import (CLC_THRESHOLD_PCT, HEIGHT_STEP_M, NO_LEVEL, REACH_M, REACH_SHARE, UNKNOWN,   # noqa: E402
                              choose_top_level, full_levels, reach_share)

# The threshold, the reach, the decks (the model's own CLCL / CLCM levels), the
# bytes and the packing are the feed's: pipeline/heights.py.
# The sanity check's low cover: nodes where DWD's own CLCL is at least this.
CLCL_CHECK_PCT = 20.0
HISTORY_REPLAY = 60          # a texdev file carries 60 history hours; replay steps 0-59 to stand in
BEND = (44.06, -121.31)
FT = 3.28084


def out(line, summary):
    print(line, flush=True)
    if summary:
        with open(summary, "a") as f:
            f.write(line + "\n")


# ---- Pure parts (tests/test_measure_heights.py) ---------------------------------

_LISTED = re.compile(r'href="([^"/]+)">[^\r\n]*?\s(\d+)[ \t]*\r?$', re.M)   # DWD's lines end CR LF


def listed_sizes(html):
    """{file name: bytes} from one of DWD's directory listings (the name is
    the link's href: the visible text is cut short)."""
    return {name: int(size) for name, size in _LISTED.findall(html or "")}


def listed_levels(html, field):
    """The levels a listing names for `field` (…_<level>_<FIELD>.grib2.bz2), ascending."""
    return sorted({int(lv) for lv in re.findall(rf'href="[^"]*_(\d+)_{field}\.grib2\.bz2"', html or "")})


# ---- The measurement ---------------------------------------------------------------

def _clock():
    with run_build._CLOCK_LOCK:
        return dict(run_build._CLOCK)


def _since(before):
    now = _clock()
    return {k: now.get(k, 0.0) - before.get(k, 0.0) for k in ("download", "remap")}


def _stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.min()), float(np.median(v)), float(v.max())) if v.size else (float("nan"),) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="measure")
    ap.add_argument("--steps", default=None, help="comma list of forecast hours (default: the texdev build's, dwd.STEPS)")
    ap.add_argument("--sample", type=int, default=2, help="steps timed one file at a time before the rest")
    ap.add_argument("--workers", type=int, default=run_build.column_fetch_workers(),
                    help="fetch threads for the rest (default: the texdev build's)")
    a = ap.parse_args()
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    deps = run_build.REAL
    stride = run_build.UPPER_STRIDE
    feed_steps = list(dwd.STEPS)                   # texdev_build.build: steps default to dwd.STEPS
    steps = [int(s) for s in a.steps.split(",")] if a.steps else feed_steps
    names = list(texdev_build.TEXDEV_SQUARES)      # what texdev_build.build builds with no squares given
    work = Path(a.out) / "work"
    work.mkdir(parents=True, exist_ok=True)

    run = dwd.newest_complete_run(deps.listing)
    if run is None:
        out("no complete run on DWD yet", summary)
        raise SystemExit(1)
    hh = f"{run:%H}"
    out(f"## Cloud heights measurement — run {dwd.run_iso(run)} (nothing published)", summary)
    out(f"steps: {len(steps)} of the texdev build's {len(feed_steps)} (dwd.STEPS, which texdev_build.build uses "
        f"when no steps are given: hourly to 78 h, 3-hourly to 120 h); a published file also carries "
        f"{HISTORY_REPLAY} history hours", summary)
    n_side = columns.side(stride)
    out(f"squares: the {len(names)} of texdev_build.TEXDEV_SQUARES (all of them priced), 5° each, "
        f"{n_side} × {n_side} nodes at {columns.spacing_deg(stride)}° (run_build.UPPER_STRIDE = {stride})", summary)
    out(f"the feed's rule (pipeline/heights.py): cloud = CLC >= {CLC_THRESHOLD_PCT:g} %; decks = the levels ICON "
        f"itself counts into CLCL and CLCM (its 800 and 400 hPa reference-pressure levels, from HHL)", summary)

    # The nodes the feed holds: the squares' window of the lattice, and which of its points are a square's.
    window = run_build.square_window(names, stride)
    r0, r1, c0, c1 = window
    in_feed = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for name in names:
        rr, cc = run_build._local_nodes(name, stride, window)
        in_feed[np.ix_(rr, cc)] = True
    out(f"lattice window rows {r0}-{r1}, columns {c0}-{c1}: {in_feed.size} points, {int(in_feed.sum())} of them "
        f"nodes of a square", summary)

    kit = deps.ensure_kit(work)
    clc_html = deps.listing(hh, "CLC") or ""
    hhl_html = deps.listing(hh, "HHL") or ""
    clc_sizes, hhl_sizes = listed_sizes(clc_html), listed_sizes(hhl_html)
    hhl_listed = [lv for lv in listed_levels(hhl_html, "HHL")
                  if dwd.invariant_file_name(run, "HHL", lv) in hhl_sizes]
    clc_listed = listed_levels(clc_html, "CLC")
    if not hhl_listed:
        out(f"MISSING: DWD's hhl/ listing names no HHL file of run {dwd.run_id(run)}", summary)
        raise SystemExit(1)
    ground_lv = max(hhl_listed)
    out(f"DWD lists HHL half levels {hhl_listed[0]}-{ground_lv} ({len(hhl_listed)} files; {ground_lv} = the ground) "
        f"and CLC full levels {clc_listed[0] if clc_listed else '-'}-{clc_listed[-1] if clc_listed else '-'}", summary)

    # ---- Which levels span ground to 5 km: from HHL, going up until they do --------
    def hhl_level(lv):
        name = dwd.invariant_file_name(run, "HHL", lv)   # one cdo call each: no level axis to trust
        return run_build._only(run_build._fetch_levels(deps, kit, work, [name], run, "HHL", stride))

    t0 = time.monotonic()
    ground_world = hhl_level(ground_lv)
    ground = run_build._crop(ground_world, window)
    land = in_feed & (ground > 1.0)                   # as measure_columns: model surface above 1 m
    hhl = {ground_lv: ground}
    agl = {ground_lv: np.zeros(int(in_feed.sum()))}      # above ground, at the feed's nodes
    top_lv, shares = None, {}
    for lv in range(ground_lv - 1, hhl_listed[0] - 1, -1):
        if lv not in hhl_listed:
            out(f"MISSING: {dwd.invariant_file_name(run, 'HHL', lv)} is not in DWD's listing", summary)
            raise SystemExit(1)
        world = hhl_level(lv)
        hhl[lv] = run_build._crop(world, window)
        agl[lv] = (hhl[lv] - ground)[in_feed]
        shares[lv] = reach_share(agl[lv], REACH_M)
        top_lv = choose_top_level(agl, REACH_M, REACH_SHARE)
        if top_lv is not None:
            world_share = reach_share(world - ground_world, REACH_M)
            break
    hhl_s = time.monotonic() - t0
    if top_lv is None:
        out(f"no listed half level is {REACH_M:g} m above ground at {REACH_SHARE:.0%} of the nodes "
            f"(best: {max(shares.values(), default=0):.2%})", summary)
        raise SystemExit(1)
    levels = full_levels(top_lv, ground_lv)
    top_agl = hhl[top_lv] - ground
    lmin, lmed, lmax = _stats(top_agl[land])
    out(f"### Level set (derived from HHL, not assumed)", summary)
    out(f"chosen: full levels {levels[0]}-{levels[-1]} ({len(levels)} levels; half levels {top_lv}-{ground_lv}): "
        f"the fewest lowest levels whose top is >= {REACH_M:g} m above ground at >= {REACH_SHARE:.0%} of the feed's "
        f"nodes — **{shares[top_lv]:.3%}** of {int(in_feed.sum())} nodes "
        f"(one level fewer, top at half level {top_lv + 1}: {shares.get(top_lv + 1, float('nan')):.3%})", summary)
    out(f"top of the set above ground over the feed's {int(land.sum())} LAND nodes (model surface > 1 m): "
        f"min {lmin:.0f} m / median {lmed:.0f} m / max {lmax:.0f} m; over all its nodes: "
        + " / ".join(f"{x:.0f}" for x in _stats(top_agl[in_feed])) + " m", summary)
    out(f"for a later world build: the same top is >= {REACH_M:g} m above ground at {world_share:.3%} of the whole "
        f"{columns.spacing_deg(stride)}° world lattice", summary)
    out(f"HHL statics ({len(hhl)} half levels, download + cdo + lattice, one after another): {hhl_s:.0f} s, once per "
        f"run (they do not change between runs of one model version)", summary)
    hhl_stack = np.stack([hhl[lv] for lv in levels + [ground_lv]])       # [L+1][R][C], highest first

    # ---- What DWD lists, and its size -------------------------------------------
    clc_names = {s: [dwd.model_file_name(run, s, lv, "CLC") for lv in levels]
                 for s in sorted(set(steps) | set(feed_steps))}
    hhl_names = [dwd.invariant_file_name(run, "HHL", lv) for lv in levels + [ground_lv]]
    missing = [n for ns in clc_names.values() for n in ns if n not in clc_sizes]
    missing += [n for n in hhl_names if n not in hhl_sizes]
    out(f"### Files", summary)
    out(f"CLC model-level and HHL files missing from DWD's listing: {len(missing)} {missing[:5]}", summary)
    if missing:
        raise SystemExit(1)
    step_bytes = {s: sum(clc_sizes[n] for n in ns) for s, ns in clc_names.items()}
    clc_total = sum(step_bytes[s] for s in feed_steps)
    hhl_total = sum(hhl_sizes[n] for n in hhl_names)
    per = sorted(step_bytes[s] for s in feed_steps)
    out(f"download size, from DWD's listing (exact, compressed .bz2): CLC {len(levels)} levels × {len(feed_steps)} "
        f"steps = {len(levels) * len(feed_steps)} files, **{clc_total / 1e9:.2f} GB** per run "
        f"({per[0] / 1e6:.1f} / {per[len(per) // 2] / 1e6:.1f} / {per[-1] / 1e6:.1f} MB per step min / median / max); "
        f"HHL {len(hhl_names)} files, **{hhl_total / 1e6:.1f} MB**, once", summary)

    # ---- One step → the four bytes ------------------------------------------------
    def clc_level(s, lv):
        name = dwd.model_file_name(run, s, lv, "CLC")    # one cdo call each: no level axis to trust
        return run_build._crop(run_build._only(run_build._fetch_levels(deps, kit, work, [name], run, "CLC", stride)),
                               window)

    masks = heights.deck_masks(hhl_stack)

    def to_bytes(clc):
        return heights.height_record(clc, hhl_stack, CLC_THRESHOLD_PCT, masks)

    base_hour = history.hour_of(run)
    done = {}                                        # step -> u8 [5][R][C]

    # The sample: one file after another, so download and build time are wall time.
    sample = steps[:max(0, a.sample)]
    rest = steps[len(sample):]
    samp = {"download": 0.0, "remap": 0.0, "bounds": 0.0, "bytes": 0}
    first, clcl_hit = None, []
    clcl_html = (deps.listing(hh, "CLCL") or "") if sample else ""
    for s in sample:
        before = _clock()
        clc = np.stack([clc_level(s, lv) for lv in levels])
        d = _since(before)
        t0 = time.monotonic()
        done[s] = to_bytes(clc)
        samp["bounds"] += time.monotonic() - t0
        samp["download"] += d["download"]
        samp["remap"] += d["remap"]
        samp["bytes"] += step_bytes[s]
        if first is None:
            first = (s, clc)
        # DWD's own low cover against our low base (not part of the timing).
        name = dwd.file_name(run, s, "CLCL")
        if name in clcl_html:
            clcl = run_build._crop(run_build._only(
                run_build._fetch_levels(deps, kit, work, [name], run, "CLCL", stride)), window)
            low = in_feed & (clcl >= CLCL_CHECK_PCT)
            clcl_hit.append((s, int(low.sum()), int((low & (done[s][0] != NO_LEVEL)).sum()),
                             int((in_feed & (done[s][0] != NO_LEVEL)).sum())))
        else:
            clcl_hit.append((s, None, None, None))
        del clc
    cores = os.cpu_count() or 1
    n_feed = len(feed_steps)
    out(f"### Time ({cores} cores)", summary)
    if sample:
        build_s = samp["remap"] + samp["bounds"]
        dl_all = samp["download"] * clc_total / samp["bytes"]
        build_all = build_s / len(sample) * n_feed
        out(f"MEASURED, sample of {len(sample)} steps {sample} ({len(sample) * len(levels)} files, "
            f"{samp['bytes'] / 1e6:.0f} MB), one file at a time: download {samp['download']:.1f} s "
            f"({samp['bytes'] / 1e6 / max(samp['download'], 1e-9):.1f} MB/s); build {build_s:.1f} s "
            f"(bunzip2 + cdo copy + lattice {samp['remap']:.1f} s, bases/tops/bytes {samp['bounds']:.2f} s) "
            f"= {build_s / len(sample):.1f} s a step", summary)
        out(f"EXTRAPOLATED from the sample to all {n_feed} steps, one file at a time: download {dl_all / 60:.1f} min "
            f"(by bytes), build {build_all / 60:.1f} min (by steps); with the texdev build's {a.workers} fetch threads "
            f"that is at best download {dl_all / 60 / a.workers:.1f} min (if DWD serves {a.workers} connections as fast "
            f"as one) and build {build_all / 60 / min(a.workers, cores):.1f} min (if it scales over {cores} cores)",
            summary)

    # The rest: the texdev build's fetch threads, a job per file.
    if rest:
        before = _clock()
        t0 = time.monotonic()
        bounds_s = 0.0
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = {s: [pool.submit(clc_level, s, lv) for lv in levels] for s in rest}
            for s in rest:
                clc = np.stack([f.result() for f in futs.pop(s)])
                t1 = time.monotonic()
                done[s] = to_bytes(clc)
                bounds_s += time.monotonic() - t1
                del clc
        rest_s = time.monotonic() - t0
        d = _since(before)
        out(f"MEASURED, the other {len(rest)} steps ({len(rest) * len(levels)} files, "
            f"{sum(step_bytes[s] for s in rest) / 1e9:.2f} GB) with {a.workers} fetch threads (the texdev build's "
            f"column_fetch_workers): **{rest_s / 60:.1f} min** wall, download and build together = "
            f"{rest_s / len(rest):.1f} s a step; worker-seconds: download {d['download']:.0f}, bunzip2 + cdo copy + "
            f"lattice {d['remap']:.0f}, bases/tops/bytes {bounds_s:.0f}", summary)
        out(f"EXTRAPOLATED by steps from those {len(rest)} to all {n_feed}, {a.workers} threads: "
            f"**{rest_s / len(rest) * n_feed / 60:.1f} min** a run (+ HHL {hhl_s:.0f} s)", summary)

    # ---- Bytes per square -----------------------------------------------------------
    own = [(base_hour + s, done[s]) for s in steps]
    replay = [(base_hour - HISTORY_REPLAY + i, own[i][1]) for i in range(min(HISTORY_REPLAY, len(own)))]
    local = {name: run_build._local_nodes(name, stride, window) for name in names}
    out(f"### Bytes per square ({len(names)} squares, {heights.N_BYTES} bytes a node a step, zlib level "
        f"{heights.ZLIB_LEVEL}, the ICH1 file's stream (pipeline/heights.py); the file adds an "
        f"{heights._HEADER.size}-byte header + 4 bytes a step)", summary)
    out("| steps | raw KB per square | compressed KB per square min / median / max | all squares MB | zlib s |\n"
        "|---|---|---|---|---|", summary)
    rows = []
    for label, stacks in (("run only", own), (f"+{len(replay)} h history (steps 0-{len(replay) - 1} replayed as a "
                                               f"stand-in)", replay + own)):
        t0 = time.monotonic()
        sizes = []
        for name in names:
            rr, cc = local[name]
            cube = np.stack([b[:, rr][:, :, cc] for _, b in stacks])          # [steps][5][side][side]
            sizes.append(len(heights.pack_square(np.ascontiguousarray(np.moveaxis(cube, 1, 3)))))
        z_s = time.monotonic() - t0
        raw = len(stacks) * n_side * n_side * heights.N_BYTES
        srt = sorted(sizes)
        rows.append((len(stacks), sum(sizes)))
        out(f"| {len(stacks)} ({label}) | {raw / 1e3:.1f} | {srt[0] / 1e3:.1f} / {srt[len(srt) // 2] / 1e3:.1f} / "
            f"{srt[-1] / 1e3:.1f} | {sum(sizes) / 1e6:.2f} | {z_s:.1f} |", summary)
    if steps != feed_steps:
        n, total = rows[-1]
        want = n_feed + HISTORY_REPLAY
        out(f"EXTRAPOLATED (by steps; only {len(steps)} of the feed's {n_feed} steps were built): {want} steps ≈ "
            f"{total / n * want / 1e6:.2f} MB for all squares", summary)
    allb = np.stack([b for _, b in own])[:, :, in_feed]
    labels = ("low base", "low top", "mid base", "mid top")
    none = (allb[:, :heights.N_HEIGHT_BYTES] == NO_LEVEL).mean(axis=(0, 2))
    out("share of node-steps with no level in the deck (byte 255): "
        + ", ".join(f"{k} {v:.1%}" for k, v in zip(labels, none)), summary)
    unread = (allb[:, :heights.N_HEIGHT_BYTES] == UNKNOWN).mean(axis=(0, 2))
    out(f"share not read (byte {UNKNOWN}: a level's cover or a height missing): "
        + ", ".join(f"{k} {v:.3%}" for k, v in zip(labels, unread)), summary)
    out(f"share of node-steps whose mid top is the top of the highest level read (flag bit "
        f"{heights.FLAG_TOP_AT_REACH}): {(allb[:, heights.FLAGS] & heights.FLAG_TOP_AT_REACH > 0).mean():.2%}", summary)

    # ---- Sanity ---------------------------------------------------------------------
    out("### Sanity check", summary)
    for s, n_low, n_both, n_base in clcl_hit:
        if n_low is None:
            out(f"step {s}: MISSING {dwd.file_name(run, s, 'CLCL')} in DWD's listing — CLCL check skipped", summary)
        else:
            out(f"step {s}: of {n_low} nodes with DWD's single-level CLCL >= {CLCL_CHECK_PCT:g} %, "
                f"{n_both} ({n_both / max(n_low, 1):.1%}) got a low-deck base; {n_base} nodes got one in all "
                f"(the low deck is CLCL's own levels, so a miss is cover under {CLC_THRESHOLD_PCT:g} % on every "
                f"level, or the lattice's blend of neighbouring cells)", summary)
    if not clcl_hit:
        out("CLCL check skipped: no sample step", summary)
    sp = columns.spacing_deg(stride)
    br, bc = round((BEND[0] + 90) / sp) - r0, round((BEND[1] + 180) / sp) - c0
    if first is None:
        out("decoded column skipped: no sample step", summary)
    elif not (0 <= br < in_feed.shape[0] and 0 <= bc < in_feed.shape[1] and in_feed[br, bc]):
        out(f"decoded column skipped: no node of these squares near {BEND}", summary)
    else:
        s, clc = first
        g = float(ground[br, bc])
        out(f"node ({(br + r0) * sp - 90:.2f}, {(bc + c0) * sp - 180:.2f}) nearest {BEND}, step {s} "
            f"(hour +{s}), ground {g:.0f} m = {g * FT:.0f} ft above sea level; levels with CLC >= 1 %:", summary)
        shown = 0
        for i, lv in reversed(list(enumerate(levels))):
            c = float(clc[i, br, bc])
            if c >= 1.0:
                shown += 1
                out(f"  L{lv}: {(hhl_stack[i + 1, br, bc] - g) * FT:6.0f}-{(hhl_stack[i, br, bc] - g) * FT:6.0f} ft "
                    f"above ground, CLC {c:5.1f} %", summary)
        if not shown:
            out("  (none)", summary)
        b = done[s][:, br, bc]
        for k, label in enumerate(labels):
            if b[k] == NO_LEVEL:
                out(f"  {label}: byte 255 = no cloud level in that deck", summary)
            elif b[k] == UNKNOWN:
                out(f"  {label}: byte {UNKNOWN} = not read", summary)
            else:
                m = b[k] * HEIGHT_STEP_M
                out(f"  {label}: byte {b[k]} = {m:.0f} m above sea level = {(m - g) * FT:.0f} ft above ground", summary)


if __name__ == "__main__":
    main()
