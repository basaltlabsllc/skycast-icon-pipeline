# pipeline/pack.py — one run folder of per-square files as one pack (PACK1).
#
# Vercel Blob bills $5 per million uploads, and ICON, GFS, AIFS and IFS upload
# one file per square per kind per run (145,152 put() calls on 2026-11-16).
# Every per-square folder <kind>/<run>/ also gets
#   <kind>/<run>/pack-<h>/all.bin     the folder's files concatenated, sorted by name
#   <kind>/<run>/pack-<h>/index.json  {"format":"PACK1","names":[...],"ends":[...]}
# <h> is the first 12 hex digits of sha256(all.bin), and the manifest key names
# that exact folder: all.bin is cached for a year, and an ICON repair rebuilds a
# run in place (live.repairs_for), so a pack at a fixed path could serve old
# bytes against a new index. File i spans bytes [ends[i-1] or 0, ends[i]) of
# all.bin, so the app reads one square by HTTP byte range. Names keep their
# extension. The pack sits inside the run folder, so prune (which keeps whole
# <kind>/<run>/ folders) keeps or drops it with its run; an older pack-<h> of
# the same run (an earlier repair) goes with the run, or sooner once only packs
# are named. Same idea as the HRRR fog packs (hrrr/fogtiles.py).
#
# FEED_TILE_FILES (env, read once per build; only "0" is off): "1" keeps the
# per-file keys in the Blob manifest beside the pack keys (both layouts go up);
# "0" drops the per-file key of every kind whose pack was written, and
# publish/blob.mjs then skips uploading those kinds' per-file copies.
import hashlib
import json
import os
import shutil
from pathlib import Path

FORMAT = "PACK1"
PACK_DIR = "pack"
HASH_HEX = 12                 # pack-<first 12 hex digits of sha256(all.bin)>
TMP_DIR = ".pack-building"    # written here, renamed to pack-<hash> once hashed
CHUNK = 8 * 1024 * 1024


def tile_files_on(env=os.environ):
    """FEED_TILE_FILES: "0" turns the per-file copies off; unset or anything else keeps them."""
    on = env.get("FEED_TILE_FILES", "1").strip() != "0"
    print(f"tile files {'on' if on else 'off'}")
    return on


def pack_name(digest_hex):
    """The pack folder's name for an all.bin whose sha256 is digest_hex."""
    return f"{PACK_DIR}-{digest_hex[:HASH_HEX]}"


def pack_folder(root, rel, dest=None):
    """Write <dest>/<rel>/pack-<hash>/{all.bin,index.json} from the files
    directly in <root>/<rel> (sorted by name; subfolders, packs included, are
    never packed). <hash> is the first HASH_HEX hex digits of sha256(all.bin),
    so a run rebuilt in place (an ICON repair) never rewrites a year-cached
    all.bin at a path an older index points into. dest defaults to root.
    Returns rel + "/pack-<hash>", or None for an empty or missing folder
    (nothing written)."""
    src = Path(root) / rel
    if not src.is_dir():
        return None
    files = sorted((p for p in src.iterdir() if p.is_file()), key=lambda p: p.name)
    if not files:
        return None
    base = Path(dest if dest is not None else root) / rel
    tmp = base / TMP_DIR
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    names, ends, total = [], [], 0
    h = hashlib.sha256()
    with open(tmp / "all.bin", "wb") as w:
        for f in files:
            with open(f, "rb") as r:
                while chunk := r.read(CHUNK):
                    h.update(chunk)
                    w.write(chunk)
            total += f.stat().st_size
            names.append(f.name)
            ends.append(total)
    if (tmp / "all.bin").stat().st_size != total:
        raise OSError(f"{rel}: all.bin is {(tmp / 'all.bin').stat().st_size} bytes, the files {total}")
    (tmp / "index.json").write_text(json.dumps({"format": FORMAT, "names": names, "ends": ends},
                                               separators=(",", ":")))
    name = pack_name(h.hexdigest())
    shutil.rmtree(base / name, ignore_errors=True)      # the same content packed again: same bytes
    tmp.rename(base / name)
    return f"{rel}/{name}"


# A pack is skipped (no key, the per-file copies stay) when it would be bigger
# than node's readFile can hold (2 GiB; publish/blob.mjs reads each file whole)
# or would leave the runner less than MIN_FREE_BYTES of disk: an ENOSPC in a
# later step (publish state) fails the job, which a missing pack never does.
MAX_PACK_BYTES = int(1.9 * 2**30)
MIN_FREE_BYTES = 3 * 10**9


def folder_bytes(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return 0
    return sum(f.stat().st_size for f in folder.iterdir() if f.is_file())


def pack_skip_reason(root, rel, dest=None, free=None):
    """Why this folder must not be packed, or None."""
    size = folder_bytes(Path(root) / rel)
    if size > MAX_PACK_BYTES:
        return f"{size / 2**30:.2f} GiB is over the {MAX_PACK_BYTES / 2**30:.1f} GiB a publish can read"
    out = Path(dest if dest is not None else root)
    out.mkdir(parents=True, exist_ok=True)
    left = (shutil.disk_usage(out).free if free is None else free) - size
    if left < MIN_FREE_BYTES:
        return f"would leave {left / 1e9:.1f} GB free on the runner (minimum {MIN_FREE_BYTES / 1e9:.0f} GB)"
    return None


def try_pack(root, rel, dest=None, free=None):
    """pack_folder, but a failure is a ::warning:: and None (the partial pack is
    removed): a failed pack drops its manifest key, never the build."""
    try:
        why = pack_skip_reason(root, rel, dest, free)
        if why:
            print(f"::warning::pack {rel} skipped: {why}")
            return None
        got = pack_folder(root, rel, dest)
    except Exception as e:
        print(f"::warning::pack {rel} failed: {type(e).__name__}: {e}")
        shutil.rmtree(Path(dest if dest is not None else root) / rel / TMP_DIR, ignore_errors=True)
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
