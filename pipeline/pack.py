# pipeline/pack.py — one run folder of per-square files as one pack (PACK1).
#
# Vercel Blob bills $5 per million uploads, and ICON, GFS, AIFS and IFS upload
# one file per square per kind per run (145,152 put() calls on 2026-11-16).
# Every per-square folder <kind>/<run>/ also gets
#   <kind>/<run>/pack/all.bin     the folder's files concatenated, sorted by name
#   <kind>/<run>/pack/index.json  {"format":"PACK1","names":[...],"ends":[...]}
# File i spans bytes [ends[i-1] or 0, ends[i]) of all.bin, so the app reads one
# square by HTTP byte range. Names keep their extension. The pack sits inside
# the run folder, so prune (which keeps whole <kind>/<run>/ folders) keeps or
# drops it with its run. Same idea as the HRRR fog packs (hrrr/fogtiles.py).
#
# FEED_TILE_FILES (env, read once per build; only "0" is off): "1" keeps the
# per-file keys in the Blob manifest beside the pack keys (both layouts go up);
# "0" drops the per-file key of every kind whose pack was written, and
# publish/blob.mjs then skips uploading those kinds' per-file copies.
import json
import os
import shutil
from pathlib import Path

FORMAT = "PACK1"
PACK_DIR = "pack"
CHUNK = 8 * 1024 * 1024


def tile_files_on(env=os.environ):
    """FEED_TILE_FILES: "0" turns the per-file copies off; unset or anything else keeps them."""
    on = env.get("FEED_TILE_FILES", "1").strip() != "0"
    print(f"tile files {'on' if on else 'off'}")
    return on


def pack_folder(root, rel, dest=None):
    """Write <dest>/<rel>/pack/{all.bin,index.json} from the files directly in
    <root>/<rel> (sorted by name; subfolders, the pack's own included, are never
    packed). dest defaults to root. Returns rel + "/pack", or None for an empty
    or missing folder (nothing written)."""
    src = Path(root) / rel
    if not src.is_dir():
        return None
    files = sorted((p for p in src.iterdir() if p.is_file()), key=lambda p: p.name)
    if not files:
        return None
    out = Path(dest if dest is not None else root) / rel / PACK_DIR
    out.mkdir(parents=True, exist_ok=True)
    names, ends, total = [], [], 0
    with open(out / "all.bin", "wb") as w:
        for f in files:
            with open(f, "rb") as r:
                shutil.copyfileobj(r, w, CHUNK)
            total += f.stat().st_size
            names.append(f.name)
            ends.append(total)
    if (out / "all.bin").stat().st_size != total:
        raise OSError(f"{rel}: all.bin is {(out / 'all.bin').stat().st_size} bytes, the files {total}")
    (out / "index.json").write_text(json.dumps({"format": FORMAT, "names": names, "ends": ends},
                                               separators=(",", ":")))
    return f"{rel}/{PACK_DIR}"


def try_pack(root, rel, dest=None):
    """pack_folder, but a failure is a ::warning:: and None (the partial pack is
    removed): a failed pack drops its manifest key, never the build."""
    try:
        got = pack_folder(root, rel, dest)
    except Exception as e:
        print(f"::warning::pack {rel} failed: {type(e).__name__}: {e}")
        shutil.rmtree(Path(dest if dest is not None else root) / rel / PACK_DIR, ignore_errors=True)
        return None
    if got:
        idx = json.loads((Path(dest if dest is not None else root) / got / "index.json").read_text())
        print(f"pack {got}: {len(idx['names'])} files, {idx['ends'][-1] / 1e6:.1f} MB")
    return got


# Manifest key per kind: top-level strings get a sibling "<kind>Pack"; object
# entries (tail, thick, columns, wind) get "pack" inside, beside their "tiles".
TOP_LEVEL = ("tiles", "blocks")
ENTRIES = ("tail", "thick", "columns", "wind")


def add_pack_keys(manifest, packs, tile_files=True):
    """A copy of `manifest` with the pack keys of `packs` ({kind: pack path or
    None}) added. With tile_files False, a kind whose pack exists loses its
    per-file key (top-level "tiles"/"blocks", or the entry's "tiles"); a kind
    with no pack keeps it, so its files are still named and uploaded."""
    m = dict(manifest)
    for kind, path in packs.items():
        if not path:
            continue
        if kind in TOP_LEVEL:
            m[f"{kind}Pack"] = path
            if not tile_files:
                m.pop(kind, None)
        elif kind in ENTRIES:
            if not isinstance(m.get(kind), dict):
                continue
            e = dict(m[kind])
            e["pack"] = path
            if not tile_files:
                e.pop("tiles", None)
            m[kind] = e
        else:
            raise ValueError(f"unknown pack kind {kind!r}")
    return m
