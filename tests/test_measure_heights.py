# tools/measure_heights.py — what the measurement tool still owns: reading
# DWD's listing (sizes, levels). Its rule, bytes and packing moved to
# pipeline/heights.py (tests/test_heights.py); this file also proves the tool
# still loads and takes them from there. Nothing here touches the network.
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("measure_heights", ROOT / "tools/measure_heights.py")
mh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mh)


def test_the_tool_takes_its_rule_from_the_feed_module():
    from pipeline import heights
    assert mh.heights is heights
    assert mh.CLC_THRESHOLD_PCT == heights.CLC_THRESHOLD_PCT == 5.0
    assert mh.REACH_M == heights.REACH_M == 5000.0
    assert mh.choose_top_level is heights.choose_top_level
    assert mh.full_levels is heights.full_levels
    for moved in ("deck_bounds", "height_bytes", "pack_square", "STANDIN_DECK_BANDS_M"):
        assert not hasattr(mh, moved), moved


def test_the_tool_names_nothing_it_no_longer_has():
    src = (ROOT / "tools/measure_heights.py").read_text()
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    assert not names & {"STANDIN_DECK_BANDS_M", "deck_bounds", "height_bytes", "pack_square", "cluefile",
                        "N_HEIGHT_BYTES", "zlib", "regrid", "clues", "DEG"}


def test_every_name_the_tool_reads_is_defined():
    # No test runs main() (it downloads); a name left behind by the move would
    # only fail in the workflow. pyflakes-style: every loaded global exists.
    src = (ROOT / "tools/measure_heights.py").read_text()
    tree = ast.parse(src)
    import builtins
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    assigned |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments)
                 for a in n.args + n.kwonlyargs + n.posonlyargs}
    assigned |= {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assigned |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
    known = assigned | set(vars(mh)) | set(vars(builtins))
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert loaded - known == set()


# ---- DWD's listing ----------------------------------------------------------------

# Lines as opendata.dwd.de sends them (2026-11-17): the visible name cut short,
# the size last, and a CR LF line end.
LISTING = (
    '<a href="../">../</a>\r\n'
    '<a href="icon_global_icosahedral_time-invariant_2026111706_100_HHL.grib2.bz2">icon_global_icosahedral_'
    'time-invariant_20261117..&gt;</a> 17-Nov-2026 08:39:33             1206375\r\n'
    '<a href="icon_global_icosahedral_model-level_2026111706_000_120_CLC.grib2.bz2">icon_global_icosahedral_'
    'model-level_2026111706_..&gt;</a> 17-Nov-2026 08:40:06              446109\r\n'
    '</pre><hr></body>\r\n')


def test_listing_sizes_are_read_by_file_name():
    assert mh.listed_sizes(LISTING) == {
        "icon_global_icosahedral_time-invariant_2026111706_100_HHL.grib2.bz2": 1206375,
        "icon_global_icosahedral_model-level_2026111706_000_120_CLC.grib2.bz2": 446109}
    assert mh.listed_sizes("") == {} and mh.listed_sizes(None) == {}


def test_listed_levels_are_the_levels_a_listing_names():
    assert mh.listed_levels(LISTING, "HHL") == [100]
    assert mh.listed_levels(LISTING, "CLC") == [120]
