import json

from pipeline import pack


def _read(root, pack_rel, name):
    idx = json.loads((root / pack_rel / "index.json").read_text())
    i = idx["names"].index(name)
    start = idx["ends"][i - 1] if i else 0
    return (root / pack_rel / "all.bin").read_bytes()[start:idx["ends"][i]]


def test_offsets_round_trip_every_file_byte_exact(tmp_path):
    d = tmp_path / "tiles" / "2026111700"
    d.mkdir(parents=True)
    files = {"N40W125.icl": bytes(range(256)) * 3, "N00E000.icl": b"\x00" * 7, "S05W010.icl": b"abc",
             "N40W125.icb": b"other kind"}
    for n, b in files.items():
        (d / n).write_bytes(b)
    rel = pack.pack_folder(tmp_path, "tiles/2026111700")
    assert rel == "tiles/2026111700/pack"
    idx = json.loads((d / "pack" / "index.json").read_text())
    assert idx["format"] == "PACK1"
    assert idx["names"] == sorted(files)
    assert idx["ends"][-1] == (d / "pack" / "all.bin").stat().st_size == sum(map(len, files.values()))
    for n, b in files.items():
        assert _read(tmp_path, rel, n) == b


def test_empty_or_missing_folder_gives_none_and_writes_nothing(tmp_path):
    assert pack.pack_folder(tmp_path, "tiles/none") is None
    (tmp_path / "blocks" / "R").mkdir(parents=True)
    assert pack.pack_folder(tmp_path, "blocks/R") is None
    assert not (tmp_path / "blocks" / "R" / "pack").exists()


def test_a_pack_is_never_packed_into_itself(tmp_path):
    d = tmp_path / "thick" / "R"
    d.mkdir(parents=True)
    (d / "a.ict").write_bytes(b"one")
    pack.pack_folder(tmp_path, "thick/R")
    (d / "b.ict").write_bytes(b"two")
    pack.pack_folder(tmp_path, "thick/R")          # again, with pack/ already there
    idx = json.loads((d / "pack" / "index.json").read_text())
    assert idx["names"] == ["a.ict", "b.ict"]
    assert (d / "pack" / "all.bin").read_bytes() == b"onetwo"


def test_dest_writes_the_pack_under_another_root_at_the_same_path(tmp_path):
    site, blob = tmp_path / "site", tmp_path / "blob"
    (site / "tiles" / "R").mkdir(parents=True)
    (site / "tiles" / "R" / "x.icl").write_bytes(b"xyz")
    assert pack.pack_folder(site, "tiles/R", dest=blob) == "tiles/R/pack"
    assert not (site / "tiles" / "R" / "pack").exists()
    assert (blob / "tiles" / "R" / "pack" / "all.bin").read_bytes() == b"xyz"


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
    assert pack.try_pack(tmp_path, "tiles/R") == "tiles/R/pack"
    out = capsys.readouterr().out
    assert "pack tiles/R/pack: 1 files" in out

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(pack, "pack_folder", boom)
    assert pack.try_pack(tmp_path, "tiles/R") is None
    assert "::warning::pack tiles/R failed: OSError: disk full" in capsys.readouterr().out
    assert not (tmp_path / "tiles" / "R" / "pack").exists()   # the partial pack is removed


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
    assert out["wind"] == {"pack": "wind/R/pack"}


def test_live_counts_a_pack_only_blob_manifest_as_whole():
    # FEED_TILE_FILES=0: the Blob manifest names blocksPack and no blocks; the
    # tail, thick and columns entries keep their objects with "pack" only.
    from pipeline import live
    m = {"run": "R", "complete": True, "blocksPack": "blocks/R/pack", "tilesPack": "tiles/R/pack",
         "columns": {"pack": "columns/R/pack"}, "thick": {"pack": "thick/R/pack"}, "tail": {"pack": "tail/T/pack"}}
    assert live.live_status(lambda url: json.dumps(m), None, blob_base="https://b") == ("R", True, 0)
    m2 = {k: v for k, v in m.items() if k != "blocksPack"}
    assert live.live_status(lambda url: json.dumps(m2), None, blob_base="https://b")[1] is False
