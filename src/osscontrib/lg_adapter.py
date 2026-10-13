"""osscontrib.lg_adapter — attach machine-checkable contracts to LangGraph tools.

The one-line integration:

    from osscontrib import contracted_tool
    from langchain_core.tools import tool

    @contracted_tool(
        pre=["authenticated($user)"],
        add=["profile_loaded($user)"],
        cost=2.0,
    )
    @tool
    def fetch_profile(user: str) -> str:
        ...

The decorator attaches a validated ``ToolContract`` to the LangChain
``StructuredTool`` as ``tool.contract``. It works on functions decorated
with ``@tool``, on ``StructuredTool`` instances, and (for graphs built
without langchain) on plain callables — in the last case the contract is
stored on the function itself.
"""

from __future__ import annotations

from typing import Any, Callable

from .contracts import CondEffect, ToolContract

_CONTRACT_ATTR = "contract"

# StructuredTool is a Pydantic model and rejects arbitrary attributes, so
# contracts live in a name-keyed registry (tool names are unique within a
# toolbelt). Plain functions still get the attribute as well.
_REGISTRY: dict[str, ToolContract] = {}


def _tool_name(target: Any) -> str:
    return getattr(target, "name", None) or getattr(target, "__name__", "tool")


def make_contract(
    name: str,
    description: str = "",
    params: dict[str, str] | None = None,
    pre: list[str] | None = None,
    add: list[str] | None = None,
    delete: list[str] | None = None,
    cond_effects: list[dict[str, list[str]]] | None = None,
    cost: float = 1.0,
    idempotent: bool = False,
) -> ToolContract:
    """Build a validated ToolContract from decorator kwargs."""
    return ToolContract(
        name=name,
        description=description,
        params=params or {},
        pre=list(pre or []),
        add=list(add or []),
        delete=list(delete or []),
        cond_effects=[CondEffect(**c) for c in (cond_effects or [])],
        cost=cost,
        idempotent=idempotent,
    )


def contracted_tool(
    pre: list[str] | None = None,
    add: list[str] | None = None,
    delete: list[str] | None = None,
    cond_effects: list[dict[str, list[str]]] | None = None,
    cost: float = 1.0,
    idempotent: bool = False,
    params: dict[str, str] | None = None,
) -> Callable[[Any], Any]:
    """Decorator factory: attach a contract to a LangChain tool.

    Apply *outside* ``@tool`` so the StructuredTool already exists::

        @contracted_tool(pre=[...], add=[...])
        @tool
        def my_tool(x: int) -> str: ...
    """

    def decorator(target: Any) -> Any:
        name = _tool_name(target)
        description = getattr(target, "description", "") or ""
        contract = make_contract(
            name=name,
            description=description,
            params=params,
            pre=pre,
            add=add,
            delete=delete,
            cond_effects=cond_effects,
            cost=cost,
            idempotent=idempotent,
        )
        return attach_contract(target, contract)

    return decorator


def attach_contract(target: Any, contract: ToolContract) -> Any:
    """Attach a contract to a tool object (StructuredTool, function, ...)."""
    _REGISTRY[_tool_name(target)] = contract
    try:
        setattr(target, _CONTRACT_ATTR, contract)
    except (ValueError, AttributeError, TypeError):
        # Pydantic models (e.g. StructuredTool) reject new attributes —
        # the registry entry is the source of truth for those.
        pass
    return target


def get_contract(target: Any) -> ToolContract | None:
    """Return the contract attached to a tool, or None if it has none."""
    c = getattr(target, _CONTRACT_ATTR, None)
    if c is not None:
        return c
    return _REGISTRY.get(_tool_name(target))


def require_contract(target: Any) -> ToolContract:
    """Return the contract, raising a clear error if the tool has none."""
    c = get_contract(target)
    if c is None:
        raise ValueError(
            f"tool '{_tool_name(target)}' has no contract — decorate it with @contracted_tool "
            "so the contract-aware compiler can validate it"
        )
    return c
