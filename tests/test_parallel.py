"""Tests: effect-intersection parallelization."""
from osscontrib.compiler import Call, ContractGraphCompiler
from osscontrib.demo_tools import ALL_TOOLS
from osscontrib.parallel import check_compatible, find_conflicts, layerize


def _compiler():
    return ContractGraphCompiler.from_tools(ALL_TOOLS)


def test_compatible_independent_calls():
    comp = _compiler()
    a = comp.ground(Call("fetch_profile", {"user": "alice"}))
    b = comp.ground(Call("fetch_orders", {"user": "alice"}))
    assert check_compatible(a, b).ok


def test_write_write_conflict():
    comp = _compiler()
    a = comp.ground(Call("create_file", {"path": "/x", "content": "1"}))
    b = comp.ground(Call("delete_file", {"path": "/x"}))
    assert not check_compatible(a, b).ok


def test_cross_dependency_blocked():
    comp = _compiler()
    a = comp.ground(Call("parse_csv", {"path": "/d/a.csv"}))
    b = comp.ground(Call("filter_rows", {"table": "/d/a.csv"}))
    assert not check_compatible(a, b).ok  # filter reads what parse writes


def test_layerize_fanout():
    comp = _compiler()
    plan = [
        Call("authenticate", {"user": "alice"}),
        Call("fetch_profile", {"user": "alice"}),
        Call("fetch_orders", {"user": "alice"}),
    ]
    grounded = [comp.ground(c) for c in plan]
    layers, _ = layerize(grounded, set())
    assert len(layers) == 2
    assert [c.name for c in layers[0].calls] == ["authenticate"]
    assert sorted(c.name for c in layers[1].calls) == ["fetch_orders", "fetch_profile"]


def test_layerize_keeps_ordered_chain_sequential():
    comp = _compiler()
    plan = [
        Call("create_file", {"path": "/d/a.csv", "content": "x"}),
        Call("parse_csv", {"path": "/d/a.csv"}),
        Call("filter_rows", {"table": "/d/a.csv"}),
        Call("aggregate", {"table": "/d/a.csv"}),
    ]
    grounded = [comp.ground(c) for c in plan]
    layers, _ = layerize(grounded, {"file_not_exists(/d/a.csv)"})
    assert len(layers) == 4  # strict chain: no parallelization possible


def test_find_conflicts():
    comp = _compiler()
    calls = [
        comp.ground(Call("create_file", {"path": "/x", "content": "1"})),
        comp.ground(Call("delete_file", {"path": "/x"})),
    ]
    assert len(find_conflicts(calls)) == 1
