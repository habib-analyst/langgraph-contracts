"""osscontrib.compiler — the contract-aware graph compiler.

Two jobs, both before a single tool runs:

1. **Static validation** (`validate_sequence`): fold a planned tool-call
   sequence through the contracts and report every precondition violation —
   with the exact unmet predicate and which earlier call should have
   established it (or that nothing in the plan does). Catches impossible
   transitions in a LangGraph agent's tool plan *before* execution.

2. **Safe fan-out** (`compile_plan`): compress the validated sequence into
   parallel layers via effect-intersection — two calls share a layer iff
   their contracts prove no write-write conflict and no cross-dependency.

Plus `ContractToolNode`: a drop-in LangGraph node that validates every
tool call against the live contract-state *at runtime* and refuses to
execute calls whose preconditions are unmet (recording the prevented
failure instead of letting the tool blow up downstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import GroundedContract, ToolContract
from .lg_adapter import get_contract
from .parallel import Layer, layerize, parallel_cost, sequential_cost


@dataclass
class Call:
    """One planned tool call: tool name + concrete parameters."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Violation:
    """A statically-detected contract violation in a planned sequence."""

    step: int  # index into the planned sequence
    call: Call
    missing: list[str]  # preconditions not established
    diagnosis: str  # human-readable: what should have established them


@dataclass
class ValidationReport:
    valid: bool
    violations: list[Violation] = field(default_factory=list)
    final_state: set[str] = field(default_factory=set)


class ContractGraphCompiler:
    """Compile + validate tool-call plans for contracted LangGraph tools."""

    def __init__(self, contracts: dict[str, ToolContract]) -> None:
        self.contracts = dict(contracts)

    @classmethod
    def from_tools(cls, tools: list[Any]) -> "ContractGraphCompiler":
        """Build a compiler from @contracted_tool-decorated tools."""
        contracts: dict[str, ToolContract] = {}
        for t in tools:
            c = get_contract(t)
            if c is None:
                name = getattr(t, "name", None) or getattr(t, "__name__", "?")
                raise ValueError(
                    f"tool '{name}' has no contract — decorate it with "
                    "@contracted_tool before compiling"
                )
            contracts[c.name] = c
        return cls(contracts)

    def ground(self, call: Call) -> GroundedContract:
        contract = self.contracts[call.tool]
        return contract.ground(call.params)

    def validate_sequence(
        self, calls: list[Call], initial_state: set[str]
    ) -> ValidationReport:
        """Statically validate a planned tool sequence against contracts.

        Walks the sequence, folding contracts through a simulated state.
        Every precondition violation is reported with a diagnosis of which
        earlier call should have established the missing predicate — or that
        no call in the plan establishes it at all.
        """
        state = set(initial_state)
        violations: list[Violation] = []
        # index: predicate -> list of step indices that add it
        producers: dict[str, list[int]] = {}
        for i, call in enumerate(calls):
            g = self.ground(call)
            missing = sorted(g.pre - state)
            if missing:
                diagnoses = []
                for pred in missing:
                    who = producers.get(pred, [])
                    if who:
                        diagnoses.append(
                            f"'{pred}' is established by step(s) "
                            f"{[c.tool for c in [calls[w] for w in who]]} "
                            f"but those run at/after this call — reorder needed"
                        )
                    else:
                        diagnoses.append(
                            f"'{pred}' is never established by any call in the plan"
                        )
                violations.append(
                    Violation(
                        step=i,
                        call=call,
                        missing=missing,
                        diagnosis="; ".join(diagnoses),
                    )
                )
            # fold expected effects even past a violation (best-effort state)
            adds = g.expected_add(state)
            state = g.expected_state(state)
            for pred in adds:
                producers.setdefault(pred, []).append(i)
        return ValidationReport(
            valid=not violations, violations=violations, final_state=state
        )

    def compile_plan(
        self, calls: list[Call], initial_state: set[str]
    ) -> tuple[list[Layer], ValidationReport]:
        """Validate, then compress into parallel layers. Raises on violation."""
        report = self.validate_sequence(calls, initial_state)
        if not report.valid:
            v = report.violations[0]
            raise ValueError(
                f"plan invalid at step {v.step} ({v.call.tool}): "
                f"missing {v.missing}. {v.diagnosis}"
            )
        grounded = [self.ground(c) for c in calls]
        layers, _ = layerize(grounded, initial_state)
        return layers, report

    def speedup(self, calls: list[Call], initial_state: set[str]) -> float:
        """Simulated speedup of the layered plan over sequential execution."""
        layers, _ = self.compile_plan(calls, initial_state)
        seq = sequential_cost([self.ground(c) for c in calls])
        par = parallel_cost(layers)
        return seq / par if par > 0 else 1.0


class ContractToolNode:
    """A LangGraph node that validates tool calls against contracts at runtime.

    Drop-in replacement for the tool-execution step of a ReAct-style agent:
    before invoking each tool it grounds the tool's contract against the
    call arguments and checks preconditions against the tracked contract
    state. Calls whose preconditions fail are *not executed* — a
    ToolMessage carrying the contract violation is returned instead, so the
    agent can replan rather than watch the tool fail downstream.

    The contract state is derived purely from declared effects, so it stays
    honest: if a tool's contract lies about its effects, the guarantees are
    void (same caveat as contractorch).
    """

    def __init__(self, tools: list[Any], initial_state: set[str] | None = None) -> None:
        self.compiler = ContractGraphCompiler.from_tools(tools)
        self.tools = {}
        for t in tools:
            tname = getattr(t, "name", None) or getattr(t, "__name__", None) or "?"
            self.tools[tname] = t
        self.state: set[str] = set(initial_state or set())
        self.prevented: list[dict[str, Any]] = []  # prevented-failure log
        self.executed: list[Call] = []

    def reset(self, initial_state: set[str] | None = None) -> None:
        self.state = set(initial_state or set())
        self.prevented = []
        self.executed = []

    def __call__(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node signature: takes graph state, returns message updates."""
        from langchain_core.messages import ToolMessage

        messages = graph_state.get("messages", [])
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) or []
        out_messages: list[ToolMessage] = []
        for tc in tool_calls:
            name = tc["name"]
            args = dict(tc.get("args", {}))
            impl = self.tools.get(name)
            if impl is None:
                out_messages.append(
                    ToolMessage(content=f"unknown tool: {name}", tool_call_id=tc["id"])
                )
                continue
            call = Call(tool=name, params=args)
            g = self.compiler.ground(call)
            missing = sorted(g.pre - self.state)
            if missing:
                self.prevented.append(
                    {"tool": name, "args": args, "missing": missing}
                )
                out_messages.append(
                    ToolMessage(
                        content=(
                            f"CONTRACT VIOLATION — call to '{name}' blocked: "
                            f"preconditions not met: {missing}. "
                            f"Replan with a call that establishes them first."
                        ),
                        tool_call_id=tc["id"],
                    )
                )
                continue
            result = impl.invoke(args)
            content = result if isinstance(result, str) else str(result)
            out_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
            self.state = g.expected_state(self.state)
            self.executed.append(call)
        return {"messages": out_messages}
