import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { putFileParts, PART_BYTES, IN_FLIGHT } from './bigfile.mjs';

async function fileOf(bytes) {
  const d = await mkdtemp(join(tmpdir(), 'bigfile-'));
  const f = join(d, 'all.bin');
  await writeFile(f, bytes);
  return f;
}

// A stand-in for @vercel/blob's createMultipartUploader: parts by number, how
// many were in flight at once, and an optional failure per part number.
function standIn({ failOnce = new Set() } = {}) {
  const s = { created: [], parts: new Map(), completed: null, inFlight: 0, maxInFlight: 0, failed: [] };
  s.create = async (pathname, options) => {
    s.created.push({ pathname, options });
    return {
      async uploadPart(n, body) {
        s.inFlight += 1;
        s.maxInFlight = Math.max(s.maxInFlight, s.inFlight);
        await new Promise((r) => setTimeout(r, 1));
        s.inFlight -= 1;
        if (failOnce.delete(n)) { s.failed.push(n); throw new Error(`part ${n} dropped`); }
        s.parts.set(n, Buffer.from(body));
        return { partNumber: n, etag: `e${n}` };
      },
      async complete(parts) { s.completed = parts; return { pathname }; },
    };
  };
  return s;
}

test('every byte goes up once, in numbered parts read at their own offsets; complete gets them in order', async () => {
  const data = Buffer.from(Array.from({ length: 23 }, (_, i) => i));
  const f = await fileOf(data);
  const s = standIn();
  const opts = { access: 'public', contentType: 'application/octet-stream' };
  const got = await putFileParts(s.create, 'columns/R/pack-x/all.bin', f, data.length, opts, { partBytes: 5, inFlight: 3 });
  assert.deepEqual(got, { pathname: 'columns/R/pack-x/all.bin' });
  assert.deepEqual(s.created, [{ pathname: 'columns/R/pack-x/all.bin', options: opts }]);
  assert.deepEqual([...s.parts.keys()].sort((a, b) => a - b), [1, 2, 3, 4, 5]);
  assert.deepEqual(s.parts.get(5), data.subarray(20));                 // the short last part
  assert.deepEqual(Buffer.concat([1, 2, 3, 4, 5].map((n) => s.parts.get(n))), data);
  assert.deepEqual(s.completed.map((p) => p.partNumber), [1, 2, 3, 4, 5]);
  assert.ok(s.maxInFlight <= 3);
});

test('a dropped part is retried on its own through attempt; the rest go up once', async () => {
  const data = Buffer.alloc(20, 7);
  const f = await fileOf(data);
  const s = standIn({ failOnce: new Set([3]) });
  let tries = 0;
  const attempt = async (fn) => { for (;;) { tries += 1; try { return await fn(); } catch (e) { if (tries > 10) throw e; } } };
  await putFileParts(s.create, 'p', f, data.length, {}, { partBytes: 5, inFlight: 2, attempt });
  assert.deepEqual(s.failed, [3]);
  assert.equal(tries, 5);                                              // 4 parts, part 3 twice
  assert.deepEqual(Buffer.concat([1, 2, 3, 4].map((n) => s.parts.get(n))), data);
});

test('a file shorter than the size it was listed with fails, never completes a truncated blob', async () => {
  const f = await fileOf(Buffer.alloc(12));
  const s = standIn();
  await assert.rejects(putFileParts(s.create, 'p', f, 15, {}, { partBytes: 5 }), /short read of part 3/);
  assert.equal(s.completed, null);
});

test('the defaults: 8 MiB parts, 8 in flight (about 64 MiB of parts in memory at once)', () => {
  assert.equal(PART_BYTES, 8 * 1024 * 1024);
  assert.equal(IN_FLIGHT, 8);
});
