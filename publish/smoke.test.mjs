import { test } from 'node:test';
import assert from 'node:assert/strict';
import { smokeKeep, smokeDeletes, publishSmoke } from './smoke.mjs';

const idx = (short, long, past) => ({
  format: 'SMK1',
  short: short && { runMs: 1, last: 18, pack: short },
  long: long && { runMs: 2, last: 48, pack: long },
  past: (past || []).map((pack, i) => ({ runMs: 10 - i, pack })),
});
const files = (pack) => [`${pack}/all.bin`, `${pack}/index.json`];

test('smokeKeep names every folder an index names, once, with a trailing slash', () => {
  const i = idx('smoke/2026112000/pack-a', 'smoke/2026112000/pack-a', ['smoke/2026112000/pack-a', 'smoke/2026111918/pack-b/']);
  assert.deepEqual(smokeKeep(i), ['smoke/2026112000/pack-a/', 'smoke/2026111918/pack-b/']);
  assert.deepEqual(smokeKeep(null), []);
  assert.deepEqual(smokeKeep({}), []);
});

test('smokeDeletes keeps every pack of the new and the previous index, short, long and past', () => {
  const next = idx('smoke/R3/pack-a', 'smoke/R3/pack-a', ['smoke/R3/pack-a', 'smoke/R2/pack-b']);
  const prev = idx('smoke/R2/pack-b', 'smoke/R1/pack-c', ['smoke/R1/pack-c']);
  const listed = ['smoke/latest.json', ...files('smoke/R3/pack-a'), ...files('smoke/R2/pack-b'), ...files('smoke/R1/pack-c')];
  assert.deepEqual(smokeDeletes(listed, next, prev), []);
});

test('smokeDeletes deletes an older pack-<h> of a run that is still listed', () => {
  const next = idx('smoke/R3/pack-new', 'smoke/R3/pack-new', ['smoke/R3/pack-new']);
  const listed = [...files('smoke/R3/pack-new'), ...files('smoke/R3/pack-old')];
  assert.deepEqual(smokeDeletes(listed, next, null), files('smoke/R3/pack-old'));
});

test('smokeDeletes deletes runs older than both indexes, and never smoke/latest.json', () => {
  const next = idx('smoke/R3/pack-a', null, ['smoke/R3/pack-a']);
  const prev = idx('smoke/R2/pack-b', null, ['smoke/R2/pack-b']);
  const listed = ['smoke/latest.json', ...files('smoke/R3/pack-a'), ...files('smoke/R2/pack-b'), ...files('smoke/R0/pack-z')];
  assert.deepEqual(smokeDeletes(listed, next, prev), files('smoke/R0/pack-z'));
  assert.ok(!smokeDeletes(['smoke/latest.json'], next, prev).includes('smoke/latest.json'));
});

test('smokeDeletes never touches a path outside smoke/, and a prefix lookalike is not kept', () => {
  const next = idx('smoke/R3/pack-a', null, []);
  assert.deepEqual(smokeDeletes(['manifest.json', 'tiles/x/y', 'smoke/R3/pack-ab/all.bin'], next, null), ['smoke/R3/pack-ab/all.bin']);
});

test('smokeDeletes refuses an index that names nothing (it would delete everything)', () => {
  assert.throws(() => smokeDeletes(['smoke/R3/pack-a/all.bin'], idx(null, null, []), null));
});

// ---- the publish itself, with a fake store ----
function fakes(listed, prev) {
  const calls = [];
  return {
    calls,
    deps: {
      put: async (path, body, o) => { calls.push(['put', path, o.cacheControlMaxAge]); },
      del: async (paths) => { calls.push(['del', paths]); },
      listAll: async () => listed.map((pathname) => ({ pathname, url: `https://x/${pathname}` })),
      readPrevious: async () => prev,
      retry: async (_l, fn) => fn(),
      perSec: 1000,
    },
  };
}

test('publishSmoke uploads the packs, then latest.json last (60 s), then deletes what no index names', async () => {
  const next = idx('smoke/R3/pack-a', 'smoke/R3/pack-a', ['smoke/R3/pack-a']);
  const f = fakes(['smoke/latest.json', ...files('smoke/R9/pack-old')], null);
  const local = { 'smoke/R3/pack-a/all.bin': Buffer.from('a'), 'smoke/R3/pack-a/index.json': Buffer.from('{}'), 'smoke/latest.json': Buffer.from(JSON.stringify(next)) };
  await publishSmoke({ files: Object.keys(local), read: async (p) => local[p], index: next, ...f.deps });
  const puts = f.calls.filter((c) => c[0] === 'put');
  assert.deepEqual(puts.map((c) => c[1]), ['smoke/R3/pack-a/all.bin', 'smoke/R3/pack-a/index.json', 'smoke/latest.json']);
  assert.equal(puts[2][2], 60);
  assert.ok(puts[0][2] > 60);
  const dels = f.calls.filter((c) => c[0] === 'del');
  assert.deepEqual(dels.flatMap((c) => c[1]), files('smoke/R9/pack-old'));
  assert.ok(f.calls.indexOf(dels[0]) > f.calls.indexOf(puts[2]));
});

test('a failed delete does not fail the publish (the index is live)', async () => {
  const next = idx('smoke/R3/pack-a', null, ['smoke/R3/pack-a']);
  const f = fakes(files('smoke/R9/pack-old'), null);
  f.deps.del = async () => { throw new Error('boom'); };
  const local = { 'smoke/R3/pack-a/all.bin': Buffer.from('a'), 'smoke/latest.json': Buffer.from('{}') };
  await publishSmoke({ files: Object.keys(local), read: async (p) => local[p], index: next, ...f.deps });
});

test('a failed upload fails the publish before latest.json goes up', async () => {
  const next = idx('smoke/R3/pack-a', null, ['smoke/R3/pack-a']);
  const f = fakes([], null);
  const puts = [];
  f.deps.put = async (p) => { puts.push(p); if (p.endsWith('all.bin')) throw new Error('down'); };
  await assert.rejects(publishSmoke({ files: ['smoke/R3/pack-a/all.bin', 'smoke/latest.json'], read: async () => Buffer.from('x'), index: next, ...f.deps }));
  assert.ok(!puts.includes('smoke/latest.json'));
});
