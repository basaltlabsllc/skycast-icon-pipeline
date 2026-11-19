import hashlib
import json
import re

from pipeline import pack


def _read(root, pack_rel, name):
    idx = json.loads((root / pack_rel / "index.json").read_text())
    i = idx["names"].index(name)
    start = idx["ends"][i - 1] if i else 0
    return (root / pack_rel / "all.bin").read_bytes()[start:idx["ends"][i]]


def _packs(d):
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def test_offsets_round_trip_every_file_byte_exact(tmp_path):
    d = tmp_path / "tiles" / "2026111700"
    d.mkdir(parents=True)
    files = {"N40W125.icl": bytes(range(256)) * 3, "N00E000.icl": b"\x00" * 7, "S05W010.icl": b"abc",
             "N40W125.icb": b"other kind"}
    for n, b in files.items():
        (d / n).write_bytes(b)
    rel = pack.pack_folder(tmp_path, "tiles/2026111700")
    assert re.fullmatch(r"tiles/2026111700/pack-[0-9a-f]{12}", rel)
    idx = json.loads((tmp_path / rel / "index.json").read_text())
    assert idx["format"] == "PACK1"
    assert idx["names"] == sorted(files)
    data = (tmp_path / rel / "all.bin").read_bytes()
    assert idx["ends"][-1] == len(data) == sum(map(len, files.values()))
    for n, b in files.items():
        assert _read(tmp_path, rel, n) == b
    # The folder is named by its content: the first 12 hex digits of sha256(all.bin).
    assert rel.endswith("/pack-" + hashlib.sha256(data).hexdigest()[:12])
    assert _packs(d) == [rel.rsplit("/", 1)[1]]              # no build folder left behind


def test_same_content_same_folder_different_content_different_folder(tmp_path):
    for root in ("a", "b", "c"):
        (tmp_path / root / "tiles" / "R").mkdir(parents=True)
        (tmp_path / root / "tiles" / "R" / "x.icl").write_bytes(b"cover")
    (tmp_path / "c" / "tiles" / "R" / "x.icl").write_bytes(b"cover, repaired")
    a, b, c = (pack.pack_folder(tmp_path / r, "tiles/R") for r in ("a", "b", "c"))
    assert a == b                     # same bytes, same folder, wherever it is built
    assert a != c                     # a repair that changes a byte gets a new folder
    # Packed again in place: the same folder, still one pack.
    assert pack.pack_folder(tmp_path / "a", "tiles/R") == a
    assert _packs(tmp_path / "a" / "tiles" / "R") == [a.rsplit("/", 1)[1]]


def test_empty_or_missing_folder_gives_none_and_writes_nothing(tmp_path):
    assert pack.pack_folder(tmp_path, "tiles/none") is None
    (tmp_path / "blocks" / "R").mkdir(parents=True)
    assert pack.pack_folder(tmp_path, "blocks/R") is None
    assert _packs(tmp_path / "blocks" / "R") == []


def test_a_pack_is_never_packed_into_itself(tmp_path):
    d = tmp_path / "thick" / "R"
    d.mkdir(parents=True)
    (d / "a.ict").write_bytes(b"one")
    first = pack.pack_folder(tmp_path, "thick/R")
    (d / "b.ict").write_bytes(b"two")
    second = pack.pack_folder(tmp_path, "thick/R")          # again, with a pack already there
    idx = json.loads((tmp_path / second / "index.json").read_text())
    assert idx["names"] == ["a.ict", "b.ict"]
    assert (tmp_path / second / "all.bin").read_bytes() == b"onetwo"
    assert first != second and (tmp_path / first / "all.bin").read_bytes() == b"one"   # untouched


def test_dest_writes_the_pack_under_another_root_at_the_same_path(tmp_path):
    site, blob = tmp_path / "site", tmp_path / "blob"
    (site / "tiles" / "R").mkdir(parents=True)
    (site / "tiles" / "R" / "x.icl").write_bytes(b"xyz")
    rel = pack.pack_folder(site, "tiles/R", dest=blob)
    assert re.fullmatch(r"tiles/R/pack-[0-9a-f]{12}", rel)
    assert _packs(site / "tiles" / "R") == []
    assert (blob / rel / "all.bin").read_bytes() == b"xyz"


def test_tile_files_switch_only_zero_is_off(capsys):
    assert pack.tile_files_on({}) is True
    assert pack.tile_files_on({"FEED_TILE_FILES": "1"}) is True
    assert pack.tile_files_on({"FEED_TILE_FILES": " 0 "}) is False
    assert pack.tile_files_on({"FEED_TILE_FILES": "no"}) is True
    assert "tile files on" in capsys.readouterr().out
    pack.tile_files_on({"FEED_TILE_FILES": "0"})
    assert "tile files off" in capsys.readouterr().out


def test_try_pack_logs_its_size_and_a_failure_is_a_warning_not_an_exception(tmp_path, capsys, monkeypatch):
    (tmp_path / "tiles" / "R").mkdir(parents=True)
    (tmp_path / "tiles" / "R" / "a.icl").write_bytes(b"12345")
    got = pack.try_pack(tmp_path, "tiles/R")
    assert re.fullmatch(r"tiles/R/pack-[0-9a-f]{12}", got)
    assert f"pack {got}: 1 files" in capsys.readouterr().out

    # A failure half way (all.bin written, the index not): a warning, no key,
    # and the half-built pack removed, so nothing of it is uploaded.
    (tmp_path / "tiles" / "R" / "b.icl").write_bytes(b"678")

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pack.json, "dumps", boom)
    assert pack.try_pack(tmp_path, "tiles/R") is None
    monkeypatch.undo()
    assert "::warning::pack tiles/R failed: OSError: disk full" in capsys.readouterr().out
    assert _packs(tmp_path / "tiles" / "R") == [got.rsplit("/", 1)[1]]   # only the earlier, whole pack


def test_add_pack_keys_on_keeps_every_per_file_key():
    m = {"tiles": "tiles/R", "blocks": "blocks/R", "tail": {"tiles": "tail/T"}, "thick": {"tiles": "thick/R"},
         "columns": None}
    packs = {"tiles": "tiles/R/pack", "blocks": "blocks/R/pack", "tail": "tail/T/pack", "thick": None,
             "columns": "columns/R/pack"}
    out = pack.add_pack_keys(m, packs, tile_files=True)
    assert out["tiles"] == "tiles/R" and out["tilesPack"] == "tiles/R/pack"
    assert out["blocks"] == "blocks/R" and out["blocksPack"] == "blocks/R/pack"
    assert out["tail"] == {"tiles": "tail/T", "pack": "tail/T/pack"}
    assert out["thick"] == {"tiles": "thick/R"}                # its pack failed: no key
    assert out["columns"] is None                              # no entry: nothing to add to
    assert "tilesPack" not in m                                # the input is not changed


def test_add_pack_keys_off_drops_the_per_file_keys_of_packed_kinds_only():
    m = {"tiles": "tiles/R", "blocks": "blocks/R", "tail": {"tiles": "tail/T", "run": "x"},
         "thick": {"tiles": "thick/R"}, "wind": {"tiles": "wind/R"}}
    packs = {"tiles": "tiles/R/pack", "blocks": None, "tail": "tail/T/pack", "thick": None, "wind": "wind/R/pack"}
    out = pack.add_pack_keys(m, packs, tile_files=False)
    assert "tiles" not in out and out["tilesPack"] == "tiles/R/pack"
    assert out["blocks"] == "blocks/R" and "blocksPack" not in out   # no pack: the files stay named
    assert out["tail"] == {"run": "x", "pack": "tail/T/pack"}
    assert out["thick"] == {"tiles": "thick/R"}
    assert out["wind"] == {"tiles": "wind/R", "pack": "wind/R/pack"}   # KEEP_FILES: the app reads wind per file


def test_live_counts_a_pack_only_blob_manifest_as_whole():
    # FEED_TILE_FILES=0: the Blob manifest names blocksPack and no blocks; the
    # tail, thick and columns entries keep their objects with "pack" only.
    from pipeline import live
    m = {"run": "R", "complete": True, "blocksPack": "blocks/R/pack", "tilesPack": "tiles/R/pack",
         "columns": {"pack": "columns/R/pack"}, "thick": {"pack": "thick/R/pack"}, "tail": {"pack": "tail/T/pack"}}
    assert live.live_status(lambda url: json.dumps(m), None, blob_base="https://b") == ("R", True, 0)
    m2 = {k: v for k, v in m.items() if k != "blocksPack"}
    assert live.live_status(lambda url: json.dumps(m2), None, blob_base="https://b")[1] is False


def test_a_pack_too_big_or_too_short_of_disk_is_skipped(tmp_path, monkeypatch, capsys):
    d = tmp_path / "tiles" / "R"
    d.mkdir(parents=True)
    (d / "N40W125.icl").write_bytes(b"x" * 100)
    # too little disk: no pack, no folder, a warning
    assert pack.try_pack(tmp_path, "tiles/R", free=pack.MIN_FREE_BYTES + 50) is None
    assert not [p for p in d.iterdir() if p.is_dir()]
    assert "skipped" in capsys.readouterr().out
    # too big for node's readFile
    monkeypatch.setattr(pack, "MAX_PACK_BYTES", 99)
    assert pack.try_pack(tmp_path, "tiles/R", free=10**12) is None
    monkeypatch.setattr(pack, "MAX_PACK_BYTES", 10**9)
    # room enough: packed
    assert pack.try_pack(tmp_path, "tiles/R", free=10**12).startswith("tiles/R/pack-")
