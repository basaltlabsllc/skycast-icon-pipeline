import { test } from 'node:test';
import assert from 'node:assert/strict';
import { manifestFolders, pathsToDelete, runFolder, backlogToDelete, summarize, runIdOf, runIdMs, runFilePath, FOG_KEEP_HOURS } from './prune.mjs';

const cur = { tiles: 'tiles/2026100200', tail: { tiles: 'tail/2026100200' } };
const prev = { tiles: 'tiles/2026100118', tail: { tiles: 'tail/2026100112' } };

test('the folders a manifest names', () => {
  assert.deepEqual(manifestFolders(cur), ['tiles/2026100200/', 'tail/2026100200/']);
  assert.deepEqual(manifestFolders({ tiles: 'tiles/x/' }), ['tiles/x/']);
  assert.deepEqual(manifestFolders(null), []);
});

test('the Clear Night tiles a manifest names are kept', () => {
  const m = { tiles: 'tiles/2026111712', clearNight: { format: 'CLN1', tiles: 'clearnight/2026111712' } };
  assert.deepEqual(manifestFolders(m), ['tiles/2026111712/', 'clearnight/2026111712/']);
  assert.deepEqual(pathsToDelete(['clearnight/2026111712/2026-11-17/N30W140.cln', 'clearnight/2026111700/2026-11-17/N30W140.cln'],
    { current: m, previous: null }), ['clearnight/2026111700/2026-11-17/N30W140.cln']);
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

// FOG HISTORY: each run within FOG_KEEP_HOURS keeps its pack's first block and index.
test('with keepHours, runs begun within 24 h keep fog pack block 0 and its index; the rest of their fog goes', () => {
  assert.equal(FOG_KEEP_HOURS, 24);
  const current = { runMs: 1794855600000, tiles: 'tiles/2026111619', fog: { pack: 'fog/2026111619/pack' } };
  const previous = { runMs: 1794852000000, tiles: 'tiles/2026111618', fog: { pack: 'fog/2026111618/pack' } };
  const paths = [
    'fog/2026111619/pack/1.iccp',
    // 6 h back: block 0 and the index stay, the later blocks and per-tile files go
    'fog/2026111613/pack/0.iccp', 'fog/2026111613/pack/index.json', 'fog/2026111613/pack/1.iccp', 'fog/2026111613/N44W122/0.icc',
    // 19Z − 24 h = 2026-11-15T19Z: kept; 18Z that day is 25 h back: gone
    'fog/2026111519/pack/0.iccp', 'fog/2026111519/pack/index.json',
    'fog/2026111518/pack/0.iccp', 'fog/2026111518/pack/index.json',
    // not a pack's block 0
    'fog/2026111613/pack/0.iccp.tmp', 'fog/2026111613/pack/10.iccp',
  ];
  assert.deepEqual(pathsToDelete(paths, { current, previous, keepHours: 48 }), [
    'fog/2026111613/pack/1.iccp', 'fog/2026111613/N44W122/0.icc',
    'fog/2026111518/pack/0.iccp', 'fog/2026111518/pack/index.json',
    'fog/2026111613/pack/0.iccp.tmp', 'fog/2026111613/pack/10.iccp',
  ]);
  // Without keepHours (every other feed) nothing of an older run's fog stays.
  assert.deepEqual(pathsToDelete(['fog/2026111613/pack/0.iccp'], { current, previous }), ['fog/2026111613/pack/0.iccp']);
  // A keepHours under 24 bounds the fog history too.
  assert.deepEqual(pathsToDelete(['fog/2026111613/pack/0.iccp', 'fog/2026111617/pack/0.iccp'], { current, previous, keepHours: 3 }), ['fog/2026111613/pack/0.iccp']);
});

test('Clear Night packed only: the pack folder of the current and previous runs is kept, older runs go', () => {
  const c = { tiles: 'tiles/B', clearNight: { format: 'CLN1', pack: 'clearnight/B/pack-0123456789ab' } };
  const p = { tiles: 'tiles/A', clearNight: { format: 'CLN1', pack: 'clearnight/A/pack-ba9876543210' } };
  assert.deepEqual(manifestFolders(c), ['tiles/B/', 'clearnight/B/pack-0123456789ab/']);
  const paths = ['clearnight/B/pack-0123456789ab/all.bin', 'clearnight/B/pack-0123456789ab/index.json',
    'clearnight/A/pack-ba9876543210/all.bin', 'clearnight/Z/pack-aaaaaaaaaaaa/all.bin', 'clearnight/Z/2026-11-17/N30W140.cln'];
  assert.deepEqual(pathsToDelete(paths, { current: c, previous: p }), ['clearnight/Z/pack-aaaaaaaaaaaa/all.bin', 'clearnight/Z/2026-11-17/N30W140.cln']);
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
// Pack folders are named by content: <kind>/<run>/pack-<12 hex of sha256(all.bin)>.
const P = 'pack-0123456789ab';
const OLD = 'pack-fedcba987654';   // the same run's pack before an ICON repair rebuilt it
const packOnly = (R, T = R, h = P) => ({
  tilesPack: `tiles/${R}/${h}`, blocksPack: `blocks/${R}/${h}`, tail: { pack: `tail/${T}/${h}` },
  thick: { pack: `thick/${R}/${h}` }, columns: { pack: `columns/${R}/${h}` }, wind: { pack: `wind/${R}/${h}` },
});

test('feed packs: a pack-only manifest keeps every pack folder, wind included', () => {
  assert.deepEqual(manifestFolders(packOnly('B')), [
    `tiles/B/${P}/`, `blocks/B/${P}/`, `tail/B/${P}/`, `columns/B/${P}/`, `thick/B/${P}/`, `wind/B/${P}/`,
  ]);
  const paths = ['manifest.json', `tiles/B/${P}/all.bin`, `tiles/B/${P}/index.json`, `wind/B/${P}/all.bin`,
    `tiles/A/${P}/all.bin`, `blocks/A/${P}/index.json`, `tiles/Z/${P}/all.bin`, `wind/Z/${P}/all.bin`, 'tail/Z/N40W125.icl'];
  assert.deepEqual(pathsToDelete(paths, { current: packOnly('B'), previous: packOnly('A') }),
    [`tiles/Z/${P}/all.bin`, `wind/Z/${P}/all.bin`, 'tail/Z/N40W125.icl']);
});

test('feed packs: with tile files on, the whole run folder is kept, files and pack', () => {
  const m = { tiles: 'tiles/B', tilesPack: `tiles/B/${P}`, blocks: 'blocks/B', blocksPack: `blocks/B/${P}`,
    wind: { tiles: 'wind/B', pack: `wind/B/${P}` } };
  assert.deepEqual(manifestFolders(m), ['tiles/B/', 'blocks/B/', 'wind/B/', `tiles/B/${P}/`, `blocks/B/${P}/`, `wind/B/${P}/`]);
  const paths = ['tiles/B/N40W125.icl', `tiles/B/${P}/all.bin`, 'wind/B/N30W120.icw', `wind/B/${P}/index.json`, `tiles/A/${P}/all.bin`];
  assert.deepEqual(pathsToDelete(paths, { current: m, previous: null }), [`tiles/A/${P}/all.bin`]);
});

test('feed packs: an older pack-<hash> of a repaired run is harmless and goes with the run', () => {
  const old = [`tiles/B/${OLD}/all.bin`, `tiles/B/${OLD}/index.json`];
  const now = [`tiles/B/${P}/all.bin`, `tiles/B/${P}/index.json`];
  // Files on: the run folder is named, so the old pack stays while the run does.
  const on = { tiles: 'tiles/B', tilesPack: `tiles/B/${P}` };
  assert.deepEqual(pathsToDelete([...old, ...now, 'tiles/B/N40W125.icl'], { current: on, previous: null }), []);
  // Files off: kept while the previous manifest (the run before the repair) names it, deleted after.
  assert.deepEqual(pathsToDelete([...old, ...now], { current: packOnly('B'), previous: packOnly('B', 'B', OLD) }), []);
  assert.deepEqual(pathsToDelete([...old, ...now], { current: packOnly('B'), previous: packOnly('A') }), old);
  // The backlog cleanup works by run: the current run's old pack stays, a dropped run's packs all go.
  assert.deepEqual(backlogToDelete([...old, ...now, `tiles/0/${OLD}/all.bin`, `tiles/0/${P}/all.bin`,
    `tiles/1/${P}/all.bin`], packOnly('B')), [`tiles/0/${OLD}/all.bin`, `tiles/0/${P}/all.bin`]);
});

test('feed packs: the backlog cleanup reads a pack-only manifest (no tiles key) without refusing', () => {
  const paths = [`tiles/2026111012/${P}/all.bin`, `tiles/2026111006/${P}/all.bin`, `tiles/2026111000/${P}/all.bin`,
    `wind/2026111012/${P}/all.bin`, 'wind/2026111000/N30W120.icw'];
  // wind's newest older run is kept (the previous manifest's), the one before that goes.
  assert.deepEqual(backlogToDelete([...paths, 'wind/2026110918/x.icw'], packOnly('2026111012')),
    [`tiles/2026111000/${P}/all.bin`, 'wind/2026110918/x.icw']);
});

test('feed packs: run history keeps a kept run\'s packs (tiles and thick folders)', () => {
  const cur = { ...packOnly('2026111619'), runMs: Date.UTC(2026, 10, 16, 19) };
  const paths = [`tiles/2026111618/${P}/all.bin`, `thick/2026111618/${P}/index.json`, `tiles/2026111500/${P}/all.bin`];
  assert.deepEqual(pathsToDelete(paths, { current: cur, previous: null, keepHours: 2 }), [`tiles/2026111500/${P}/all.bin`]);
});
