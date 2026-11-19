# PACK1 packs (pipeline/pack.py) for the global models: GFS, AIFS and IFS (IFS
# gets every change even while its schedule is paused). Every per-square folder
# (tiles, blocks, tail, thick, and GFS's wind) gets <folder>/pack-<hash>/, and the
# manifest names it beside the per-file key; FEED_TILE_FILES=0 drops the
# per-file keys of the packed kinds.
import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import globalfeed.build as gb
from globalfeed.models import AIFS, GFS, IFS
from pipeline import pack
from pipeline.squares import square_name

UTC = timezone.utc
RUN = datetime(2026, 11, 8, 0, tzinfo=UTC)
NOW = datetime(2026, 11, 8, 5, 10, tzinfo=UTC)
SQUARES = [square_name(40, -125), square_name(-5, 175)]
R = "2026110800"


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(gb, "all_squares", lambda: list(SQUARES))
    monkeypatch.setattr(gb.history, "HEAD_STEPS", 6)
    grid = np.meshgrid(np.arange(-90, 90.25, 0.25), np.arange(-180, 180, 0.25), indexing="ij")

    def fake_read(model, run, fh, get):
        lat, lon = grid
        out = {"lat": lat.ravel(), "lon": lon.ravel()}
        for i, f in enumerate(model.fields):
            out[f if isinstance(f, str) else f[0]] = ((lat.ravel() + 90) / 1.8 + fh * 2 + i * 7) % 101
        return out

    def fake_thick(model, run, fh, get):
        lat, lon = grid
        return {"lat": lat.ravel(), "lon": lon.ravel(),
                "ice": np.full(lat.size, 0.02 + fh * 0.01), "water": np.full(lat.size, 0.1)}

    def fake_wind(model, run, fh, get):
        lat, lon = np.arange(-90, 90.5, 1.0), np.arange(-180, 180, 1.0)
        out = {"lat": lat, "lon": lon}
        for lev in (850, 500, 250):
            for c in "uv":
                out[(lev, c)] = np.full((lat.size, lon.size), 5.0 + fh if c == "u" else -3.0)
        return out

    monkeypatch.setattr(gb, "read_step", fake_read)
    monkeypatch.setattr(gb, "read_thick_step", fake_thick)
    monkeypatch.setattr(gb, "read_wind_step", fake_wind)


def _model(model):
    steps = {"gfs": ((0, 1, 2), (3,)), "aifs": ((0, 6), (12,)), "ifs": ((0, 6), (12,))}[model.name]
    return dataclasses.replace(model, main_steps=steps[0], tail_steps=steps[1])


def _build(model, out):
    gb.build(_model(model), RUN, out, lambda u, r=None: b"", NOW, out.parent / f"state-{model.name}",
             sleep=lambda s: None)
    return json.loads((out / "manifest.json").read_text())


def _is_pack(path, folder):
    """folder/pack-<first 12 hex of sha256(all.bin)>."""
    return re.fullmatch(re.escape(folder) + r"/pack-[0-9a-f]{12}", path or "") is not None


def _pack_matches(root, folder, pack_rel):
    idx = json.loads((root / pack_rel / "index.json").read_text())
    data = (root / pack_rel / "all.bin").read_bytes()
    names = sorted(p.name for p in (root / folder).iterdir() if p.is_file())
    assert idx["format"] == "PACK1" and idx["names"] == names and names
    for i, n in enumerate(names):
        assert data[(idx["ends"][i - 1] if i else 0):idx["ends"][i]] == (root / folder / n).read_bytes(), n
    assert len(data) == idx["ends"][-1]


# kinds each model writes: (manifest kind, folder)
KINDS = {
    "gfs": ["tiles", "blocks", "tail", "thick", "wind"],
    "aifs": ["tiles", "blocks", "tail", "wind"],
    "ifs": ["tiles", "blocks", "tail", "thick", "wind"],
}


def _per_file(man, kind):
    return man.get(kind) if kind in pack.TOP_LEVEL else (man.get(kind) or {}).get("tiles")


def _pack_key(man, kind):
    return man.get(f"{kind}Pack") if kind in pack.TOP_LEVEL else (man.get(kind) or {}).get("pack")


@pytest.mark.parametrize("model", [GFS, AIFS, IFS], ids=lambda m: m.name)
def test_every_folder_gets_a_byte_exact_pack_and_both_keys(small, tmp_path, monkeypatch, model, capsys):
    monkeypatch.setenv("FEED_TILE_FILES", "1")
    out = tmp_path / model.name
    man = _build(model, out)
    logged = capsys.readouterr().out
    assert "tile files on" in logged
    for kind in KINDS[model.name]:
        folder = f"{kind}/{R}"
        assert _per_file(man, kind) == folder, kind
        key = _pack_key(man, kind)
        assert _is_pack(key, folder), (kind, key)
        _pack_matches(out, folder, key)
        assert f"pack {key}: " in logged                          # each pack's size is logged
    for kind in {"tiles", "blocks", "tail", "thick", "wind"} - set(KINDS[model.name]):
        assert _pack_key(man, kind) is None and not (out / kind).exists()


@pytest.mark.parametrize("model", [GFS, AIFS, IFS], ids=lambda m: m.name)
def test_tile_files_off_names_only_the_packs(small, tmp_path, monkeypatch, model, capsys):
    monkeypatch.setenv("FEED_TILE_FILES", "0")
    out = tmp_path / model.name
    man = _build(model, out)
    assert "tile files off" in capsys.readouterr().out
    for kind in KINDS[model.name]:
        if kind in pack.KEEP_FILES:      # wind: the app reads it per file, so it stays named per file too
            assert _per_file(man, kind) == f"{kind}/{R}", kind
        else:
            assert _per_file(man, kind) is None, kind
        assert _is_pack(_pack_key(man, kind), f"{kind}/{R}"), kind
        assert (out / _pack_key(man, kind) / "all.bin").is_file()
    # The rest of the manifest is untouched.
    assert man["run"] == "2026-11-08T00Z" and man["model"] == model.name and man["squares"] == sorted(SQUARES)


def test_a_failed_pack_drops_its_key_never_the_build(small, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FEED_TILE_FILES", "0")
    real = pack.pack_folder

    def flaky(root, rel, dest=None):
        if rel.startswith("wind/"):
            raise OSError("disk full")
        return real(root, rel, dest)
    monkeypatch.setattr(pack, "pack_folder", flaky)
    out = tmp_path / "gfs"
    man = _build(GFS, out)
    assert "::warning::pack wind/2026110800 failed" in capsys.readouterr().out
    assert man["wind"]["tiles"] == f"wind/{R}" and "pack" not in man["wind"]   # still named per file
    assert _is_pack(man["tilesPack"], f"tiles/{R}") and "tiles" not in man
    assert not any(p.is_dir() for p in (out / "wind" / R).iterdir())


@pytest.mark.parametrize("model", [GFS, AIFS, IFS], ids=lambda m: m.name)
def test_pack_folders_are_named_by_content(small, tmp_path, monkeypatch, model):
    # The same run built twice from the same data: the same folders. A rebuild
    # of the run with different cover: new folders, in the same run folder.
    monkeypatch.setenv("FEED_TILE_FILES", "1")
    a = _build(model, tmp_path / "a")
    b = _build(model, tmp_path / "b")
    for kind in KINDS[model.name]:
        assert _pack_key(a, kind) == _pack_key(b, kind), kind
    # Changed cover: every cover folder gets a new pack folder.
    real = gb.read_step

    def brighter(m, run, fh, get):
        d = real(m, run, fh, get)
        return {k: (v if k in ("lat", "lon") else (v + 13) % 101) for k, v in d.items()}
    monkeypatch.setattr(gb, "read_step", brighter)
    c = _build(model, tmp_path / "c")
    for kind in ("tiles", "blocks", "tail"):
        assert _pack_key(a, kind) != _pack_key(c, kind), kind
        assert _per_file(a, kind) == _per_file(c, kind)
