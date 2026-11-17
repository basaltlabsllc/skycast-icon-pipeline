#!/usr/bin/env python3
"""Build Clear Night tiles from a LOCAL copy of the live feed's squares
(tools/fetch_feed_squares.sh), into a local folder — no DWD download, no
upload. For checking the tiles and the map on a real run before deploy.

The squares are the published 2 %-step cover, so the stats can differ from a
real build's (raw percent) by a point or so. World nodes no fetched square
covers are MISSING, so tiles past the fetched box read no data.

Usage: python3 tools/clearnight_from_feed.py <feedDir> <outDir>
Writes <outDir>/manifest.json ({"clearNight": entry, "run": …}) and
<outDir>/clearnight/<run>/<night>_<key>.cln; prints compute time and bytes.
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import clearnight                              # noqa: E402
from pipeline.encode import decode_square, MISSING           # noqa: E402
from pipeline.squares import NX, NY, row_indices, col_indices  # noqa: E402


def main(feed, out):
    feed, out = Path(feed), Path(out)
    m = json.loads((feed / "manifest.json").read_text())
    files = sorted((feed / m["tiles"]).glob("*.icl"))
    if not files:
        raise SystemExit(f"no squares under {feed / m['tiles']}")
    t0 = time.monotonic()
    first = decode_square(files[0].read_bytes())
    hours = first["step_hours"]
    world = np.full((len(hours), NY, NX), MISSING, np.uint8)
    names = []
    for f in files:
        sq = decode_square(f.read_bytes())
        if sq["step_hours"] != hours:
            raise SystemExit(f"{f.name}: hours differ from {files[0].name}")
        rows, cols = row_indices(sq["sw_lat"]), col_indices(sq["sw_lon"])
        world[:, rows[0]:rows[-1] + 1][:, :, cols] = sq["cube"][:, 3]      # field 3: CLCT
        names.append(f.stem)
    t1 = time.monotonic()
    run_id = m["tiles"].split("/")[-1]
    entry = clearnight.build_clearnight(out, run_id, hours, lambda i: world[i],
                                        keys=clearnight.tiles_for_squares(names),
                                        first_night=dt.datetime.fromtimestamp(m["runMs"] / 1000 - 86400, dt.timezone.utc).date().isoformat())
    t2 = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"run": m["run"], "runMs": m["runMs"], "clearNight": entry},
                                                  separators=(",", ":")))
    print(f"run {m['run']}: {len(names)} squares, {len(hours)} hours; decode {t1 - t0:.1f} s, "
          f"clear night {t2 - t1:.1f} s for {len(entry['keys'])} tiles × {len(entry['nights'])} nights")
    for n in entry["nights"]:
        print(f"  {n}: {entry['bytes'][n]:,} B in {len(entry['keys'])} tiles")


if __name__ == "__main__":
    main(*sys.argv[1:3])
