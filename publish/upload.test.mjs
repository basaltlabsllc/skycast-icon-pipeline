import { test } from 'node:test';
import assert from 'node:assert/strict';
import { putOptionsFor, contentTypeFor, MULTIPART_BYTES, YEAR_S, packedOnlyFolders, splitPacked } from './upload.mjs';

// ---- Feed packs (PACK1, pipeline/pack.py) ------------------------------------
// Pack folders are named by content: <kind>/<run>/pack-<12 hex of sha256(all.bin)>.
const H = 'pack-0123456789ab';
const H2 = 'pack-fedcba987654';

test('every feed pack all.bin gets the year cache as octet-stream (multipart from the cut); its index is JSON', () => {
  for (const k of ['tiles', 'blocks', 'tail', 'thick', 'columns', 'wind']) {
    const bin = putOptionsFor(`${k}/2026111700/${H}/all.bin`, 400e6);
    assert.equal(bin.cacheControlMaxAge, YEAR_S);
    assert.equal(bin.contentType, 'application/octet-stream');
    assert.equal(bin.multipart, true);
    assert.equal('multipart' in putOptionsFor(`${k}/2026111700/${H}/all.bin`, 2e6), false);
    const idx = putOptionsFor(`${k}/2026111700/${H}/index.json`, 3e5);
    assert.equal(idx.cacheControlMaxAge, YEAR_S);
    assert.equal(idx.contentType, 'application/json');
  }
  // Only a pack folder's index is JSON, not a lookalike.
  assert.equal(contentTypeFor('tiles/2026111700/packing/index.json'), 'application/octet-stream');
});

test('tile files on: every folder is still named per file, nothing is skipped', () => {
  const m = { tiles: 'tiles/R', tilesPack: `tiles/R/${H}`, blocks: 'blocks/R', blocksPack: `blocks/R/${H}`,
    tail: { tiles: 'tail/T', pack: `tail/T/${H}` }, wind: { tiles: 'wind/R', pack: `wind/R/${H}` } };
  assert.deepEqual(packedOnlyFolders(m), []);
  const files = [{ path: 'tiles/R/a.icl', dir: 's' }, { path: `tiles/R/${H}/all.bin`, dir: 'b' }];
  assert.deepEqual(splitPacked(files, m), { send: files, skipped: 0 });
});

test('tile files off: the per-file copies of every packed folder are skipped, the packs and the rest go up', () => {
  const m = { tilesPack: `tiles/R/${H}`, blocksPack: `blocks/R/${H2}`, tail: { pack: `tail/T/${H}` },
    thick: { pack: `thick/R/${H}` }, columns: { pack: `columns/R/${H}` }, wind: { pack: `wind/R/${H}` },
    fog: { pack: 'fog/R/pack' } };
  assert.deepEqual(packedOnlyFolders(m), [
    ['tiles/R/', `tiles/R/${H}/`], ['blocks/R/', `blocks/R/${H2}/`], ['tail/T/', `tail/T/${H}/`],
    ['thick/R/', `thick/R/${H}/`], ['columns/R/', `columns/R/${H}/`], ['wind/R/', `wind/R/${H}/`],
  ]);   // fog is HRRR's own switch (FOG_TILE_FILES): never touched here
  const paths = ['tiles/R/a.icl', `tiles/R/${H}/all.bin`, `tiles/R/${H}/index.json`, 'blocks/R/a.icb',
    `blocks/R/${H2}/all.bin`, 'tail/T/a.icl', 'thick/R/a.ict', 'columns/R/a.icc', 'wind/R/N30W120.icw',
    `wind/R/${H}/all.bin`, 'tiles/RX/a.icl', 'runs/2026111700.json', 'fog/R/N44W122/0.icc'];
  const { send, skipped } = splitPacked(paths.map((path) => ({ path, dir: 'd' })), m);
  assert.equal(skipped, 6);
  assert.deepEqual(send.map((f) => f.path), [`tiles/R/${H}/all.bin`, `tiles/R/${H}/index.json`,
    `blocks/R/${H2}/all.bin`, `wind/R/${H}/all.bin`, 'tiles/RX/a.icl', 'runs/2026111700.json', 'fog/R/N44W122/0.icc']);
});

test('tile files off with a failed pack: that folder has no pack key, keeps its per-file key and its files go up', () => {
  const m = { tilesPack: `tiles/R/${H}`, blocks: 'blocks/R' };
  const { send, skipped } = splitPacked([{ path: 'blocks/R/a.icb', dir: 'b' }, { path: 'tiles/R/a.icl', dir: 's' }], m);
  assert.deepEqual(send.map((f) => f.path), ['blocks/R/a.icb']);
  assert.equal(skipped, 1);
});

test('blob.mjs skips per-file copies through splitPacked with the manifest it publishes, and logs the count', async () => {
  const { readFile } = await import('node:fs/promises');
  const src = await readFile(new URL('./blob.mjs', import.meta.url), 'utf8');
  assert.match(src, /splitPacked\(files, current\)/);
  assert.match(src, /per-file copies of packed folders not uploaded/);
});

test('a file at or above the cut goes up multipart, below it a plain put', () => {
  assert.equal(MULTIPART_BYTES, 32 * 1024 * 1024);
  assert.equal(putOptionsFor('fog/2026111012/pack/0.iccp', 80e6).multipart, true);
  assert.equal(putOptionsFor('fog/2026111012/pack/0.iccp', MULTIPART_BYTES).multipart, true);
  assert.equal('multipart' in putOptionsFor('fog/2026111012/pack/8.iccp', MULTIPART_BYTES - 1), false);
  assert.equal('multipart' in putOptionsFor('tiles/2026111012/N40W125.icl', 40e3), false);
});

test('every run-folder file keeps the year cache, public, fixed path', () => {
  for (const [p, n] of [['fog/R/pack/0.iccp', 90e6], ['fog/R/pack/index.json', 50e3], ['fog/R/N44W122/0.icc', 60e3]]) {
    const o = putOptionsFor(p, n);
    assert.equal(o.cacheControlMaxAge, YEAR_S);
    assert.equal(o.access, 'public');
    assert.equal(o.addRandomSuffix, false);
    assert.equal(o.allowOverwrite, true);
  }
});

test('the pack index is JSON; packs, tiles and other files stay octet-stream', () => {
  assert.equal(contentTypeFor('fog/2026111012/pack/index.json'), 'application/json');
  assert.equal(contentTypeFor('fog/2026111012/pack/0.iccp'), 'application/octet-stream');
  assert.equal(contentTypeFor('fog/2026111012/N44W122/0.icc'), 'application/octet-stream');
  assert.equal(contentTypeFor('places/20261101/index.json'), 'application/octet-stream');
});

test('blob.mjs uploads run-folder files through putOptionsFor with the body size', async () => {
  const { readFile } = await import('node:fs/promises');
  const src = await readFile(new URL('./blob.mjs', import.meta.url), 'utf8');
  assert.match(src, /put\(path, body, putOptionsFor\(path, body\.length\)\)/);
});

test('Clear Night named only by its pack: its per-file copies are never uploaded', () => {
  const m = { clearNight: { format: 'CLN1', pack: 'clearnight/2026111712/pack-0123456789ab' } };
  assert.deepEqual(packedOnlyFolders(m), [['clearnight/2026111712/', 'clearnight/2026111712/pack-0123456789ab/']]);
  const files = [{ path: 'clearnight/2026111712/2026-11-17/N30W140.cln' }, { path: 'clearnight/2026111712/pack-0123456789ab/all.bin' }, { path: 'clearnight/2026111712/pack-0123456789ab/index.json' }];
  assert.deepEqual(splitPacked(files, m).send.map((f) => f.path), ['clearnight/2026111712/pack-0123456789ab/all.bin', 'clearnight/2026111712/pack-0123456789ab/index.json']);
});
