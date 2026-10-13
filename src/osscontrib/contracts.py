"""osscontrib.contracts — the tool-contract DSL (LangGraph edition).

Adapted from the contractorch project (github.com/habib-analyst/contractorch):
a tool declares, in machine-checkable form,
  * preconditions: predicates that must hold before the call
  * effects:      predicates added / deleted after the call
  * conditional effects: (when, add, delete) branches verified at runtime

Predicates are ground strings like ``file_exists(/data/a.csv)``. Contract
templates use ``$var`` placeholders substituted from call parameters.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_VAR = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def substitute(template: str, params: dict[str, Any]) -> str:
    """Replace $var placeholders in a predicate template with call params."""

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in params:
            raise KeyError(f"contract template references unknown param '${name}'")
        return str(params[name])

    return _VAR.sub(repl, template)


class CondEffect(BaseModel):
    """A conditional effect: when `when` holds, apply add/delete after the call."""

    when: list[str] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)


class ToolContract(BaseModel):
    """The full contract of one tool."""

    name: str
    description: str = ""
    params: dict[str, str] = Field(default_factory=dict)  # param name -> type hint
    pre: list[str] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)
    cond_effects: list[CondEffect] = Field(default_factory=list)
    cost: float = 1.0  # simulated step cost for speedup measurement
    idempotent: bool = False

    @field_validator("cost")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cost must be non-negative")
        return v

    def ground(self, params: dict[str, Any]) -> "GroundedContract":
        """Instantiate the contract for one concrete call."""
        return GroundedContract(
            name=self.name,
            params=dict(params),
            pre={substitute(t, params) for t in self.pre},
            add={substitute(t, params) for t in self.add},
            delete={substitute(t, params) for t in self.delete},
            cond_effects=[
                CondEffect(
                    when=[substitute(t, params) for t in c.when],
                    add=[substitute(t, params) for t in c.add],
                    delete=[substitute(t, params) for t in c.delete],
                )
                for c in self.cond_effects
            ],
            cost=self.cost,
        )


class GroundedContract(BaseModel):
    """A contract with all $vars substituted — ready for planning/execution."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    pre: set[str] = Field(default_factory=set)
    add: set[str] = Field(default_factory=set)
    delete: set[str] = Field(default_factory=set)
    cond_effects: list[CondEffect] = Field(default_factory=list)
    cost: float = 1.0

    @property
    def effects(self) -> set[str]:
        """All predicates this call writes (adds + deletes)."""
        return self.add | self.delete

    def applicable(self, state: set[str]) -> bool:
        """True if every precondition holds in `state`."""
        return self.pre <= state

    def apply(self, state: set[str]) -> set[str]:
        """Return the state after this call's unconditional effects."""
        return (state | self.add) - self.delete

    def expected_add(self, state: set[str]) -> set[str]:
        """Unconditional adds + adds of conditional effects whose `when` holds."""
        out = set(self.add)
        for c in self.cond_effects:
            if set(c.when) <= state:
                out |= set(c.add)
        return out

    def expected_delete(self, state: set[str]) -> set[str]:
        out = set(self.delete)
        for c in self.cond_effects:
            if set(c.when) <= state:
                out |= set(c.delete)
        return out

    def expected_state(self, state: set[str]) -> set[str]:
        return (state | self.expected_add(state)) - self.expected_delete(state)


class WorldState(BaseModel):
    """A set of ground predicates describing the world."""

    predicates: set[str] = Field(default_factory=set)

    def holds(self, pred: str) -> bool:
        return pred in self.predicates

    def satisfies(self, goal: set[str]) -> bool:
        return goal <= self.predicates
