#!/bin/sh
# tools/fetch_feed_squares.sh — copy the LIVE feed's manifest and the ICL1
# squares of a lat/lon box into a local folder, read-only, for
# tools/clearnight_from_feed.py. Public URLs, no credentials.
#
# Usage: tools/fetch_feed_squares.sh <outDir> [south north west east]
#   default box: the 20° Clear Night tiles over the US, lat 10..70, lon -140..-60.
set -eu
OUT=$1
S=${2:-10}; N=${3:-70}; W=${4:--140}; E=${5:--60}
BASE=${FEED_BASE:-https://kq0vzbgeghiuc2ov.public.blob.vercel-storage.com}
mkdir -p "$OUT"
code=$(curl -sS -o "$OUT/manifest.json" -w '%{http_code}' "$BASE/manifest.json")
[ "$code" = 200 ] || { echo "manifest: HTTP $code" >&2; exit 1; }
TILES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tiles"])' "$OUT/manifest.json")
mkdir -p "$OUT/$TILES"
lat=$S
while [ "$lat" -lt "$N" ]; do
  lon=$W
  while [ "$lon" -lt "$E" ]; do
    name=$(python3 -c 'import sys; la,lo=int(sys.argv[1]),int(sys.argv[2]); print(("N" if la>=0 else "S")+f"{abs(la):02d}"+("E" if lo>=0 else "W")+f"{abs(lo):03d}")' "$lat" "$lon")
    f="$OUT/$TILES/$name.icl"
    if [ ! -s "$f" ]; then
      code=$(curl -sS -o "$f" -w '%{http_code}' "$BASE/$TILES/$name.icl")
      [ "$code" = 200 ] || { echo "$name: HTTP $code" >&2; rm -f "$f"; }
    fi
    lon=$((lon + 5))
  done
  lat=$((lat + 5))
done
echo "run $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run"])' "$OUT/manifest.json"): $(ls "$OUT/$TILES" | wc -l | tr -d ' ') squares in $OUT/$TILES"
