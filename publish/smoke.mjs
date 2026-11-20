// publish/smoke.mjs — copy a built site-hrrr-smoke/ to the HRRR Blob store.
//
// hrrr_smoke_build.py writes smoke/<run>/pack-<h>/{all.bin,index.json} and
// smoke/latest.json (its short, long and past entries each name a `pack`).
// Same order as blob.mjs: the pack files first (their paths never change, so a
// year of cache), smoke/latest.json LAST (60 s), so no phone reads an index
// whose pack is not there yet. Then every smoke/ blob that neither the new
// index nor the PREVIOUS one names is deleted (a phone that read the old index
// a moment ago is still fetching its pack). A failed delete never fails the
// publish: the index is live, and the next publish's listing picks up what is
// left. blob.mjs's prune leaves smoke/ alone (prune.mjs PROTECTED_PREFIXES).
//
// Usage: BLOB_READ_WRITE_TOKEN=… node publish/smoke.mjs <dir>
import { readdir, readFile } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { put, list, del } from '@vercel/blob';
import { createRetry } from './pace.mjs';
import { deletePaced, DELETES_PER_SEC } from './sweep.mjs';
import { listAll as listStore } from './store.mjs';
import { putOptionsFor } from './upload.mjs';

export const INDEX_PATH = 'smoke/latest.json';
const INDEX_S = 60;               // Blob's shortest cache time
const CONCURRENCY = 8;
const slash = (p) => p.replace(/\/?$/, '/');

/** The pack folders (each ending in '/') an index names in short, long and past, once each. */
export function smokeKeep(index) {
  const out = new Set();
  if (!index) return [];
  for (const k of ['short', 'long']) {
    if (index[k] && typeof index[k].pack === 'string' && index[k].pack) out.add(slash(index[k].pack));
  }
  for (const e of Array.isArray(index.past) ? index.past : []) {
    if (e && typeof e.pack === 'string' && e.pack) out.add(slash(e.pack));
  }
  return [...out];
}

/**
 * The smoke/ pathnames to delete: every listed one under smoke/ that is not
 * smoke/latest.json and sits in no folder the new or the previous index names.
 * An older pack-<h> of a run that is still listed, and runs older than both
 * indexes, go. Throws when the NEW index names no pack at all (everything
 * would look old).
 */
export function smokeDeletes(listed, nextIndex, prevIndex) {
  const next = smokeKeep(nextIndex);
  if (!next.length) throw new Error('the new smoke index names no pack; refusing to prune');
  const keep = [...next, ...smokeKeep(prevIndex)];
  return listed.filter((p) => p.startsWith('smoke/') && p !== INDEX_PATH && !keep.some((k) => p.startsWith(k)));
}

/**
 * Upload `files` (paths relative to the store root, smoke/…), the index last,
 * then prune. Injected: put, del, listAll, readPrevious, read, retry.
 */
export async function publishSmoke({ files, read, index, put, del, listAll, readPrevious, retry, perSec = DELETES_PER_SEC, log = console.log }) {
  const packFiles = files.filter((p) => p !== INDEX_PATH);
  if (!files.includes(INDEX_PATH)) throw new Error(`no ${INDEX_PATH} to publish`);
  const before = (await listAll()).map((b) => b.pathname);
  // the index must not go up naming a pack that is neither in this upload nor already live
  for (const k of smokeKeep(index)) {
    if (!packFiles.includes(`${k}all.bin`) && !before.includes(`${k}all.bin`)) throw new Error(`index names ${k} but its all.bin is not uploaded or in the store`);
  }
  const previous = await readPrevious(before);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, packFiles.length) }, async () => {
    while (next < packFiles.length) {
      const path = packFiles[next++];
      const body = await read(path);
      await retry(path, () => put(path, body, putOptionsFor(path, body.length)));
    }
  }));
  log(`uploaded ${packFiles.length} smoke files`);
  await retry(INDEX_PATH, async () => put(INDEX_PATH, await read(INDEX_PATH), {
    access: 'public', addRandomSuffix: false, allowOverwrite: true,
    cacheControlMaxAge: INDEX_S, contentType: 'application/json',
  }));
  log(`${INDEX_PATH} live`);
  try {
    const doomed = smokeDeletes((await listAll()).map((b) => b.pathname), index, previous);
    const n = await deletePaced(doomed, { del, retry, perSec });
    log(`deleted ${n} old smoke files`);
  } catch (e) {
    log(`::warning::smoke publish: the index is live, but pruning old smoke files failed (${(e && e.message) || e})`);
  }
}

async function walk(dir, out = []) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p, out);
    else if (e.isFile()) out.push(p);
  }
  return out;
}

async function main() {
  const dir = process.argv[2];
  if (!dir) throw new Error('usage: node publish/smoke.mjs <dir>');
  if (!process.env.BLOB_READ_WRITE_TOKEN) throw new Error('BLOB_READ_WRITE_TOKEN is not set');
  const files = (await walk(dir)).map((p) => relative(dir, p).split(sep).join('/')).filter((p) => p.startsWith('smoke/'));
  const index = JSON.parse(await readFile(join(dir, INDEX_PATH), 'utf8'));
  const retry = createRetry({ attempts: 4 });
  const readPrevious = async (pathnames) => {
    const b = (await listStore(list, retry)).find((x) => x.pathname === INDEX_PATH);
    if (!b || !pathnames.includes(INDEX_PATH)) return null;
    try {
      const r = await fetch(`${b.url}?t=${Date.now()}`, { cache: 'no-store' });
      return r.ok ? await r.json() : null;
    } catch { return null; }
  };
  await publishSmoke({
    files, index, retry, put, del, readPrevious,
    read: (p) => readFile(join(dir, p)),
    listAll: () => listStore(list, retry),
    perSec: Number(process.env.BLOB_DELETES_PER_SEC) || DELETES_PER_SEC,
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((e) => { console.error(`::error::smoke publish failed: ${e.message || e}`); process.exit(1); });
}
