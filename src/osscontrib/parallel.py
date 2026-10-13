"""osscontrib.parallel — safe parallelism derived from contracts.

Two tool calls may run in parallel **iff** their contracts prove it safe:
  1. No write-write conflict:  effects(A) ∩ effects(B) = ∅
  2. No dependency either way: pre(A) ∩ effects(B) = ∅ and pre(B) ∩ effects(A) = ∅

A sequential call list is compressed into parallel *layers*: each call goes
into the earliest layer whose start-state satisfies its preconditions and in
which it is pairwise-compatible with every call already placed there.
Layer start-states are recomputed (folded) after every placement — placing a
call into an earlier layer changes what later layers start from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import GroundedContract


@dataclass
class Compatibility:
    ok: bool
    reason: str = ""


def check_compatible(a: GroundedContract, b: GroundedContract) -> Compatibility:
    """Decide whether two grounded calls may run in the same parallel layer."""
    write_conflict = a.effects & b.effects
    if write_conflict:
        return Compatibility(
            False,
            f"write-write conflict on {sorted(write_conflict)} "
            f"between {a.name} and {b.name}",
        )
    dep_ab = a.pre & b.effects
    if dep_ab:
        return Compatibility(
            False,
            f"{a.name} reads {sorted(dep_ab)} written by {b.name}",
        )
    dep_ba = b.pre & a.effects
    if dep_ba:
        return Compatibility(
            False,
            f"{b.name} reads {sorted(dep_ba)} written by {a.name}",
        )
    return Compatibility(True, "no shared writes, no cross dependencies")


def find_conflicts(calls: list[GroundedContract]) -> list[tuple[int, int, str]]:
    """All conflicting index pairs in a candidate batch, with reasons."""
    out: list[tuple[int, int, str]] = []
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            c = check_compatible(calls[i], calls[j])
            if not c.ok:
                out.append((i, j, c.reason))
    return out


@dataclass
class Layer:
    calls: list[GroundedContract] = field(default_factory=list)


def layerize(
    plan: list[GroundedContract], initial_state: set[str]
) -> tuple[list[Layer], list[set[str]]]:
    """Compress a sequential call list into parallel layers.

    Returns (layers, layer_start_states).
    """
    layers: list[Layer] = []

    def fold() -> list[set[str]]:
        out: list[set[str]] = []
        st = set(initial_state)
        for layer in layers:
            out.append(set(st))
            for c in layer.calls:
                st = c.apply(st)
        return out

    for step in plan:
        placed = False
        starts = fold()
        for li, layer in enumerate(layers):
            start = starts[li]
            if not step.pre <= start:
                continue
            if all(check_compatible(step, other).ok for other in layer.calls):
                layer.calls.append(step)
                placed = True
                break
        if not placed:
            starts = fold()
            end = set(initial_state)
            for layer in layers:
                for c in layer.calls:
                    end = c.apply(end)
            if not step.pre <= end:
                raise ValueError(
                    f"plan step {step.name}{step.params} has preconditions "
                    f"{sorted(step.pre - end)} never established by the plan"
                )
            layers.append(Layer([step]))

    return layers, fold()


def apply_layers(layers: list[Layer], initial_state: set[str]) -> set[str]:
    """Fold all layers into the final world state."""
    state = set(initial_state)
    for layer in layers:
        for c in layer.calls:
            state = c.apply(state)
    return state


def sequential_cost(plan: list[GroundedContract]) -> float:
    return sum(c.cost for c in plan)


def parallel_cost(layers: list[Layer]) -> float:
    """Simulated cost of the layered plan: each layer costs its slowest call."""
    return sum(max((c.cost for c in layer.calls), default=0.0) for layer in layers)
