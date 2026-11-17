import { test } from 'node:test';
import assert from 'node:assert/strict';
import { putOptionsFor, contentTypeFor, MULTIPART_BYTES, YEAR_S } from './upload.mjs';

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
