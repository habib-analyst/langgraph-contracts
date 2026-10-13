"""Tests: contract DSL."""
import pytest

from osscontrib.contracts import GroundedContract, ToolContract, substitute


def test_substitute():
    assert substitute("file_exists($path)", {"path": "/a.txt"}) == "file_exists(/a.txt)"
    with pytest.raises(KeyError):
        substitute("file_exists($path)", {})


def test_ground():
    c = ToolContract(name="t", pre=["file_exists($p)"], add=["ok($p)"])
    g = c.ground({"p": "/x"})
    assert g.pre == {"file_exists(/x)"}
    assert g.add == {"ok(/x)"}
    assert g.applicable({"file_exists(/x)"})
    assert not g.applicable(set())


def test_apply():
    c = ToolContract(name="t", add=["a"], delete=["b"])
    g = c.ground({})
    assert g.apply({"b", "c"}) == {"a", "c"}


def test_cond_effects():
    from osscontrib.contracts import CondEffect
    c = ToolContract(
        name="t", add=["base"],
        cond_effects=[CondEffect(when=["flag"], add=["extra"], delete=[])],
    )
    g = c.ground({})
    assert g.expected_state({"flag"}) == {"base", "extra", "flag"}
    assert g.expected_state(set()) == {"base"}


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        ToolContract(name="t", cost=-1.0)
