// publish/upload.mjs — the put() options for one run-folder file. Pure.
//
// The HRRR fog packs (fog/<run>/pack/<k>.iccp, hrrr/fogtiles.py) run to about
// 50–110 MB each. Those go up as @vercel/blob 2.8.0 multipart uploads
// (put(..., { multipart: true }): 8 MiB parts, up to 8 in flight, each part
// retried on its own), so a dropped connection resends 8 MiB, not the whole
// pack. Vercel bills a multipart upload as 1 advanced operation to start, 1 per
// part and 1 to complete (vercel.com/docs/vercel-blob/usage-and-pricing), so a
// 64 MB pack is 10 operations against 1 for a plain put: small files stay plain.
// The cut is 32 MiB: every pack of a full 6-hour block is above it (~7–16
// operations each, ~100 a run against the ~6,500 tile files it replaces), and
// every other feed file is far below it.
//
// Content type: a pack index (fog/<run>/pack/index.json, and the feed packs'
// <kind>/<run>/pack-<h>/index.json below) is application/json;
// everything else stays application/octet-stream as before (places' index.json
// included: unchanged here). All run-folder files keep the year cache: their
// paths change with every run, and a pack index never changes within one.

export const YEAR_S = 365 * 24 * 3600;
export const MULTIPART_BYTES = 32 * 1024 * 1024;

export function contentTypeFor(path) {
  return /(^|\/)pack(-[0-9a-f]+)?\/index\.json$/.test(path) ? 'application/json' : 'application/octet-stream';
}

/** put() options for a run-folder file of `size` bytes. */
export function putOptionsFor(path, size) {
  const o = {
    access: 'public', addRandomSuffix: false, allowOverwrite: true,
    cacheControlMaxAge: YEAR_S, contentType: contentTypeFor(path),
  };
  if (size >= MULTIPART_BYTES) o.multipart = true;
  return o;
}

// ---- Feed packs (PACK1, pipeline/pack.py: ICON, GFS, AIFS, IFS) -------------
//
// Every per-square folder <kind>/<run>/ also carries <kind>/<run>/pack-<h>/,
// <h> the first 12 hex digits of sha256(all.bin), the exact folder the
// manifest key names (all.bin + index.json; above: the year cache, multipart
// from the cut, the index as JSON). A rebuilt run (an ICON repair) with other
// bytes gets another folder, so a year-cached all.bin never changes under an
// index. The builds' FEED_TILE_FILES switch decides whether the Blob
// manifest still names the per-file folder: "1" names both (both layouts go
// up), "0" names only the pack. The manifest is the one source of truth here:
// a folder the manifest reaches only through its pack has its per-file copies
// left out of the upload, so no blob is ever written that no manifest names. A
// pack that failed leaves no pack key and the per-file key in place, so its
// files still go up. The HRRR fog entry has its own switch and is not touched.

const TOP_PACKS = [['tiles', 'tilesPack'], ['blocks', 'blocksPack']];
const ENTRY_PACKS = ['tail', 'thick', 'columns', 'wind', 'clearNight'];
const slash = (p) => p.replace(/\/?$/, '/');

/** [[runFolder/, runFolder/pack/]] for each folder the manifest names only by its pack. */
export function packedOnlyFolders(m) {
  const out = [];
  if (!m) return out;
  const add = (pack) => {
    const p = slash(pack);
    out.push([p.replace(/[^/]+\/$/, ''), p]);     // the run folder: the pack folder's parent
  };
  for (const [k, pk] of TOP_PACKS) {
    if (typeof m[pk] === 'string' && m[pk] && !(typeof m[k] === 'string' && m[k])) add(m[pk]);
  }
  for (const k of ENTRY_PACKS) {
    const e = m[k];
    if (e && typeof e.pack === 'string' && e.pack && !(typeof e.tiles === 'string' && e.tiles)) add(e.pack);
  }
  return out;
}

/** files ([{ path, dir }]) without the per-file copies of packed-only folders: { send, skipped }. */
export function splitPacked(files, m) {
  const folders = packedOnlyFolders(m);
  if (!folders.length) return { send: files, skipped: 0 };
  const send = files.filter(({ path }) => !folders.some(([run, pack]) => path.startsWith(run) && !path.startsWith(pack)));
  return { send, skipped: files.length - send.length };
}
