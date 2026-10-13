"""Tests: @contracted_tool adapter."""
import pytest
from langchain_core.tools import tool

from osscontrib.lg_adapter import (
    attach_contract,
    contracted_tool,
    get_contract,
    require_contract,
)
from osscontrib.contracts import ToolContract


def test_decorator_attaches_contract():
    @contracted_tool(pre=["file_exists($path)"], add=["file_read($path)"], cost=0.5)
    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return "x"

    c = get_contract(read_file)
    assert isinstance(c, ToolContract)
    assert c.name == "read_file"
    assert c.cost == 0.5
    g = c.ground({"path": "/a"})
    assert g.pre == {"file_exists(/a)"}


def test_decorator_order_does_not_matter():
    # @contracted_tool inside @tool: the name-keyed registry means the
    # contract survives the StructuredTool wrapping either way.
    @tool
    @contracted_tool(pre=["p"])
    def my_tool(x: int) -> str:
        """T."""
        return "y"

    c = get_contract(my_tool)
    assert c is not None
    assert c.pre == ["p"]


def test_require_contract_ok():
    @contracted_tool(add=["done"])
    @tool
    def t() -> str:
        """T."""
        return "y"

    assert require_contract(t).add == ["done"]


def test_attach_contract_plain_callable():
    def f(x):
        return x

    c = ToolContract(name="f", add=["did"])
    attach_contract(f, c)
    assert get_contract(f) is c


def test_demo_tools_all_contracted():
    from osscontrib.demo_tools import ALL_TOOLS
    assert len(ALL_TOOLS) == 11
    for t in ALL_TOOLS:
        assert get_contract(t) is not None, t.name
