import { test } from 'node:test';
import assert from 'node:assert/strict';
import { manifestFolders, pathsToDelete, runFolder, backlogToDelete, summarize, runIdOf, runIdMs, runFilePath } from './prune.mjs';

const cur = { tiles: 'tiles/2026100200', tail: { tiles: 'tail/2026100200' } };
const prev = { tiles: 'tiles/2026100118', tail: { tiles: 'tail/2026100112' } };

test('the folders a manifest names', () => {
  assert.deepEqual(manifestFolders(cur), ['tiles/2026100200/', 'tail/2026100200/']);
  assert.deepEqual(manifestFolders({ tiles: 'tiles/x/' }), ['tiles/x/']);
  assert.deepEqual(manifestFolders(null), []);
});

test('keeps the manifest, the current run and the previous one; deletes older runs', () => {
  const paths = [
    'manifest.json',
    'tiles/2026100200/N40W125.icl', 'tail/2026100200/N40W125.icl',
    'tiles/2026100118/N40W125.icl', 'tail/2026100112/N40W125.icl',
    'tiles/2026100112/N40W125.icl', 'tail/2026100100/N40W125.icl',
    'tiles/20261002000/N00E000.icl',   // a prefix lookalike is not the same folder
  ];
  assert.deepEqual(pathsToDelete(paths, { current: cur, previous: prev }), [
    'tiles/2026100112/N40W125.icl', 'tail/2026100100/N40W125.icl', 'tiles/20261002000/N00E000.icl',
  ]);
});

test('with no previous manifest only the current run is kept', () => {
  assert.deepEqual(pathsToDelete(['manifest.json', 'tiles/a/x.icl', 'tiles/b/x.icl'], { current: { tiles: 'tiles/b' }, previous: null }), ['tiles/a/x.icl']);
});

test('the column feed folder is kept like the tail; a null or old-style manifest has none', () => {
  const withCols = { ...cur, columns: { tiles: 'columns/2026100200' } };
  const prevCols = { ...prev, columns: { tiles: 'columns/2026100118' } };
  assert.deepEqual(manifestFolders(withCols), ['tiles/2026100200/', 'tail/2026100200/', 'columns/2026100200/']);
  assert.deepEqual(manifestFolders({ ...cur, columns: null }), manifestFolders(cur));
  const paths = ['columns/2026100200/N40W125.icc', 'columns/2026100118/N40W125.icc', 'columns/2026100112/N40W125.icc'];
  assert.deepEqual(pathsToDelete(paths, { current: withCols, previous: prevCols }), ['columns/2026100112/N40W125.icc']);
  // A build that published no columns deletes the older column folders.
  assert.deepEqual(pathsToDelete(paths, { current: cur, previous: prev }), paths);
});

test('the wind folder is kept like the others', () => {
  assert.deepEqual(manifestFolders({ wind: { tiles: 'wind/2026111700' } }), ['wind/2026111700/']);
});

test('the blocks and thickness folders are kept like the others; a manifest without them has none', () => {
  const m = { tiles: 'tiles/A', blocks: 'blocks/A', thick: { tiles: 'thick/A' }, tail: null, columns: null };
  assert.deepEqual(manifestFolders(m), ['tiles/A/', 'blocks/A/', 'thick/A/']);
  assert.deepEqual(manifestFolders({ tiles: 'tiles/A', blocks: null, thick: null }), ['tiles/A/']);
  const old = { tiles: 'tiles/Z' };
  const paths = ['blocks/A/x.icb', 'thick/A/x.ict', 'tiles/Z/x.icl', 'blocks/Q/x.icb', 'thick/Q/x.ict'];
  assert.deepEqual(pathsToDelete(paths, { current: m, previous: old }), ['blocks/Q/x.icb', 'thick/Q/x.ict']);
});

// From Nov 1 no prune got through its deletes, so a store held a backlog of
// many runs. The next prune that works must clear ALL of it, not just the run
// the previous manifest replaced: pathsToDelete works from the full listing.
test('a prune after failed ones clears the whole backlog, every family', () => {
  const runs = ['2026103000', '2026103006', '2026103012', '2026103018', '2026103100', '2026110100', '2026110106', '2026110112'];
  const fams = ['tiles', 'blocks', 'tail', 'columns', 'thick'];
  const paths = ['manifest.json'];
  for (const r of runs) for (const f of fams) paths.push(`${f}/${r}/N40W125.x`, `${f}/${r}/S10E020.x`);
  const m = (r) => ({ tiles: `tiles/${r}`, blocks: `blocks/${r}`, tail: { tiles: `tail/${r}` }, columns: { tiles: `columns/${r}` }, thick: { tiles: `thick/${r}` } });
  const doomed = pathsToDelete(paths, { current: m('2026110112'), previous: m('2026110106') });
  assert.equal(doomed.length, 6 * fams.length * 2);
  const left = paths.filter((p) => !doomed.includes(p));
  assert.deepEqual(left.sort(), ['manifest.json', ...fams.flatMap((f) => ['2026110106', '2026110112'].flatMap((r) => [`${f}/${r}/N40W125.x`, `${f}/${r}/S10E020.x`]))].sort());
});

test('runFolder', () => {
  assert.deepEqual(runFolder('tiles/2026100200/N40W125.icl'), { family: 'tiles', run: '2026100200' });
  assert.equal(runFolder('manifest.json'), null);
  assert.equal(runFolder('tiles/x'), null);
});

test('backlog: keeps the live run, the one before it and anything newer, per family; deletes the rest', () => {
  const live = { tiles: 'tiles/2026110112', blocks: 'blocks/2026110112', tail: { tiles: 'tail/2026110100' }, columns: null, thick: { tiles: 'thick/2026110112' } };
  const paths = [
    'manifest.json', 'robots.txt',
    'tiles/2026103100/a', 'tiles/2026110100/a', 'tiles/2026110106/a', 'tiles/2026110112/a', 'tiles/2026110118/a',  // 18Z: uploading now
    'blocks/2026110100/a', 'blocks/2026110112/a',             // no 06Z blocks: 00Z is the one before
    'tail/2026103012/a', 'tail/2026103100/a', 'tail/2026103112/a', 'tail/2026110100/a',
    'thick/2026110112/a',
    'columns/2026103100/a', 'columns/2026110106/a', 'columns/2026110112/a', // not named: measured against tiles' run
  ];
  assert.deepEqual(backlogToDelete(paths, live), [
    'tiles/2026103100/a', 'tiles/2026110100/a',
    'tail/2026103012/a', 'tail/2026103100/a',
    'columns/2026103100/a',
  ]);
});

test('backlog: a manifest naming no tiles folder deletes nothing', () => {
  assert.throws(() => backlogToDelete(['tiles/a/x'], null), /no tiles folder/);
  assert.throws(() => backlogToDelete(['tiles/a/x'], { tiles: null }), /no tiles folder/);
});

test('summarize: files and bytes by family and by run', () => {
  const s = summarize([
    { pathname: 'tiles/A/x', size: 10 }, { pathname: 'tiles/A/y', size: 5 }, { pathname: 'tiles/B/x', size: 1 },
    { pathname: 'tail/A/x', size: 100 }, { pathname: 'manifest.json', size: 2 },
  ]);
  assert.deepEqual(s.total, { files: 5, bytes: 118 });
  assert.deepEqual(s.families, [['.', { files: 1, bytes: 2 }], ['tail', { files: 1, bytes: 100 }], ['tiles', { files: 3, bytes: 16 }]]);
  assert.deepEqual(s.runs, [['tail/A', { files: 1, bytes: 100 }], ['tiles/A', { files: 2, bytes: 15 }], ['tiles/B', { files: 1, bytes: 1 }]]);
});

test('fog folders are named by the manifest and kept for current and previous', () => {
  assert.deepEqual(manifestFolders({ fog: { tiles: 'fog/2026111012' } }), ['fog/2026111012/']);
  const c = { tiles: 'tiles/B', fog: { tiles: 'fog/B' } }, p = { tiles: 'tiles/A', fog: { tiles: 'fog/A' } };
  const paths = ['manifest.json', 'fog/B/N44W122/0.icc', 'fog/A/N44W122/0.icc', 'fog/Z/N44W122/0.icc'];
  assert.deepEqual(pathsToDelete(paths, { current: c, previous: p }), ['fog/Z/N44W122/0.icc']);
  assert.deepEqual(pathsToDelete(['fog/A/N44W122/0.icc'], { current: { tiles: 'tiles/B', fog: { tiles: 'fog/B' } }, previous: { fog: { tiles: 'fog/A' } } }), []);
});

test('texdev/ is never pruned, by the publish or by the backlog sweep', () => {
  const paths = ['manifest.json', 'tiles/a/x.icl', 'texdev/2026111206/N40W125.icu', 'texdev/manifest.json'];
  assert.deepEqual(pathsToDelete(paths, { current: { tiles: 'tiles/b' }, previous: null }), ['tiles/a/x.icl']);
  assert.deepEqual(backlogToDelete(['tiles/a/x.icl', 'tiles/b/x.icl', 'texdev/2026111206/N40W125.icu', 'texdev/2026111200/N40W125.icu'], { tiles: 'tiles/b' }), []);
});

// RUN HISTORY: HRRR/RRFS keep 48 h of clouds + thickness + run files for SkyCast's late report fills.
// computed: python3 datetime → 2026-11-16T19:00Z = 1794855600000
test('run ids name the hour in UTC, both ways', () => {
  assert.equal(runIdOf(1794855600000), '2026111619');
  assert.equal(runIdMs('2026111619'), 1794855600000);
  assert.equal(runFilePath(1794855600000), 'runs/2026111619.json');
  assert.ok(Number.isNaN(runIdMs('20261116190')));
});

test('with keepHours, runs begun within it keep their clouds, thickness and run file; fog and older runs go', () => {
  const current = { runMs: 1794855600000, tiles: 'tiles/2026111619', thick: { tiles: 'thick/2026111619' }, fog: { tiles: 'fog/2026111619' } };
  const previous = { runMs: 1794852000000, tiles: 'tiles/2026111618', thick: { tiles: 'thick/2026111618' }, fog: { tiles: 'fog/2026111618' } };
  const paths = [
    'manifest.json',
    'tiles/2026111619/a.icl', 'thick/2026111619/a.ict', 'fog/2026111619/x/0.icc', 'runs/2026111619.json',
    'tiles/2026111618/a.icl', 'fog/2026111618/x/0.icc',
    // 19Z − 48 h = 2026-11-14T19Z: kept; 18Z that day is 49 h back: gone
    'tiles/2026111419/a.icl', 'thick/2026111419/a.ict', 'fog/2026111419/x/0.icc', 'runs/2026111419.json',
    'tiles/2026111418/a.icl', 'thick/2026111418/a.ict', 'runs/2026111418.json',
    'texdev/a.bin',
  ];
  assert.deepEqual(pathsToDelete(paths, { current, previous, keepHours: 48 }), [
    'fog/2026111419/x/0.icc',
    'tiles/2026111418/a.icl', 'thick/2026111418/a.ict', 'runs/2026111418.json',
  ]);
  // Without it (every other feed), only the current and previous runs stay, run files included.
  assert.deepEqual(pathsToDelete(paths, { current, previous }), [
    'runs/2026111619.json',
    'tiles/2026111419/a.icl', 'thick/2026111419/a.ict', 'fog/2026111419/x/0.icc', 'runs/2026111419.json',
    'tiles/2026111418/a.icl', 'thick/2026111418/a.ict', 'runs/2026111418.json',
  ]);
});

test('fog packs: an entry with only `pack` (FOG_TILE_FILES=0) keeps its pack folder for current and previous', () => {
  assert.deepEqual(manifestFolders({ fog: { pack: 'fog/2026111012/pack' } }), ['fog/2026111012/pack/']);
  const c = { tiles: 'tiles/B', fog: { pack: 'fog/B/pack' } }, p = { tiles: 'tiles/A', fog: { pack: 'fog/A/pack' } };
  const paths = ['manifest.json', 'fog/B/pack/0.iccp', 'fog/B/pack/index.json', 'fog/A/pack/3.iccp', 'fog/Z/pack/0.iccp', 'fog/Z/N44W122/0.icc'];
  assert.deepEqual(pathsToDelete(paths, { current: c, previous: p }), ['fog/Z/pack/0.iccp', 'fog/Z/N44W122/0.icc']);
});

test('fog packs: an entry with tiles AND pack keeps the whole fog run, tile files and packs', () => {
  const m = { fog: { tiles: 'fog/2026111012', pack: 'fog/2026111012/pack' } };
  assert.deepEqual(manifestFolders(m), ['fog/2026111012/', 'fog/2026111012/pack/']);
  const paths = ['fog/2026111012/N44W122/0.icc', 'fog/2026111012/pack/0.iccp', 'fog/2026111012/pack/index.json', 'fog/2026111011/pack/0.iccp'];
  assert.deepEqual(pathsToDelete(paths, { current: { tiles: 'tiles/2026111012', ...m }, previous: null }), ['fog/2026111011/pack/0.iccp']);
});

test('fog packs: the backlog cleanup reads a pack-only fog run', () => {
  const live = { tiles: 'tiles/2026111012', fog: { pack: 'fog/2026111012/pack' } };
  const paths = ['fog/2026111012/pack/0.iccp', 'fog/2026111011/pack/0.iccp', 'fog/2026111010/pack/0.iccp'];
  assert.deepEqual(backlogToDelete(paths, live), ['fog/2026111010/pack/0.iccp']);
});

// Feed packs (PACK1, pipeline/pack.py): ICON, GFS, AIFS, IFS. With
// FEED_TILE_FILES=0 a manifest names a folder only by its pack.
const packOnly = (R, T = R) => ({
  tilesPack: `tiles/${R}/pack`, blocksPack: `blocks/${R}/pack`, tail: { pack: `tail/${T}/pack` },
  thick: { pack: `thick/${R}/pack` }, columns: { pack: `columns/${R}/pack` }, wind: { pack: `wind/${R}/pack` },
});

test('feed packs: a pack-only manifest keeps every pack folder, wind included', () => {
  assert.deepEqual(manifestFolders(packOnly('B')), [
    'tiles/B/pack/', 'blocks/B/pack/', 'tail/B/pack/', 'columns/B/pack/', 'thick/B/pack/', 'wind/B/pack/',
  ]);
  const paths = ['manifest.json', 'tiles/B/pack/all.bin', 'tiles/B/pack/index.json', 'wind/B/pack/all.bin',
    'tiles/A/pack/all.bin', 'blocks/A/pack/index.json', 'tiles/Z/pack/all.bin', 'wind/Z/pack/all.bin', 'tail/Z/N40W125.icl'];
  assert.deepEqual(pathsToDelete(paths, { current: packOnly('B'), previous: packOnly('A') }),
    ['tiles/Z/pack/all.bin', 'wind/Z/pack/all.bin', 'tail/Z/N40W125.icl']);
});

test('feed packs: with tile files on, the whole run folder is kept, files and pack', () => {
  const m = { tiles: 'tiles/B', tilesPack: 'tiles/B/pack', blocks: 'blocks/B', blocksPack: 'blocks/B/pack',
    wind: { tiles: 'wind/B', pack: 'wind/B/pack' } };
  assert.deepEqual(manifestFolders(m), ['tiles/B/', 'blocks/B/', 'wind/B/', 'tiles/B/pack/', 'blocks/B/pack/', 'wind/B/pack/']);
  const paths = ['tiles/B/N40W125.icl', 'tiles/B/pack/all.bin', 'wind/B/N30W120.icw', 'wind/B/pack/index.json', 'tiles/A/pack/all.bin'];
  assert.deepEqual(pathsToDelete(paths, { current: m, previous: null }), ['tiles/A/pack/all.bin']);
});

test('feed packs: the backlog cleanup reads a pack-only manifest (no tiles key) without refusing', () => {
  const paths = ['tiles/2026111012/pack/all.bin', 'tiles/2026111006/pack/all.bin', 'tiles/2026111000/pack/all.bin',
    'wind/2026111012/pack/all.bin', 'wind/2026111000/N30W120.icw'];
  // wind's newest older run is kept (the previous manifest's), the one before that goes.
  assert.deepEqual(backlogToDelete([...paths, 'wind/2026110918/x.icw'], packOnly('2026111012')),
    ['tiles/2026111000/pack/all.bin', 'wind/2026110918/x.icw']);
});

test('feed packs: run history keeps a kept run\'s packs (tiles and thick folders)', () => {
  const cur = { ...packOnly('2026111619'), runMs: Date.UTC(2026, 10, 16, 19) };
  const paths = ['tiles/2026111618/pack/all.bin', 'thick/2026111618/pack/index.json', 'tiles/2026111500/pack/all.bin'];
  assert.deepEqual(pathsToDelete(paths, { current: cur, previous: null, keepHours: 2 }), ['tiles/2026111500/pack/all.bin']);
});
