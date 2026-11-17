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
// Content type: a pack index (fog/<run>/pack/index.json) is application/json;
// everything else stays application/octet-stream as before (places' index.json
// included: unchanged here). All run-folder files keep the year cache: their
// paths change with every run, and a pack index never changes within one.

export const YEAR_S = 365 * 24 * 3600;
export const MULTIPART_BYTES = 32 * 1024 * 1024;

export function contentTypeFor(path) {
  return /(^|\/)pack\/index\.json$/.test(path) ? 'application/json' : 'application/octet-stream';
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
