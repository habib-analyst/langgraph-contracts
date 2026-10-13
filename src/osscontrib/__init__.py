"""osscontrib — contract-based tool orchestration for LangGraph.

Attach machine-checkable contracts to LangGraph tools, validate agent
tool plans statically before execution, and parallelize safe fan-outs
via effect-intersection analysis.

Sibling of contractorch (github.com/habib-analyst/contractorch), which
proved the contract-planning approach; this package is the LangGraph-native
integration, framed as a reusable pattern for the LangGraph community.
"""

from .compiler import (
    Call,
    ContractGraphCompiler,
    ContractToolNode,
    ValidationReport,
    Violation,
)
from .contracts import CondEffect, GroundedContract, ToolContract, WorldState
from .lg_adapter import (
    attach_contract,
    contracted_tool,
    get_contract,
    make_contract,
    require_contract,
)
from .parallel import Layer, check_compatible, find_conflicts, layerize

__all__ = [
    "Call",
    "CondEffect",
    "ContractGraphCompiler",
    "ContractToolNode",
    "GroundedContract",
    "Layer",
    "ToolContract",
    "ValidationReport",
    "Violation",
    "WorldState",
    "attach_contract",
    "check_compatible",
    "contracted_tool",
    "find_conflicts",
    "get_contract",
    "layerize",
    "make_contract",
    "require_contract",
]

__version__ = "0.1.0"
