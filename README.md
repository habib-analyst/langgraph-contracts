# langgraph-contracts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/langgraph-contracts?style=social)](https://github.com/habib-analyst/langgraph-contracts)

**Stop letting your LangGraph agent call tools it can't legally call.** One decorator adds machine-checkable contracts to your tools — preconditions and effects the graph validates *before* execution and enforces *at runtime*.

## Quickstart

```bash
pip install osscontrib
```

```python
from osscontrib import contracted_tool, ContractGraphCompiler, ContractToolNode, Call
from langchain_core.tools import tool

@contracted_tool(
    pre=["authenticated($user)", "profile_loaded($user)"],
    add=["email_sent($to)"],
    cost=1.5,
)
@tool
def send_email(user: str, to: str) -> str:
    """Send an email to a recipient."""
    ...

# 1. Validate a planned sequence BEFORE running it
compiler = ContractGraphCompiler.from_tools([send_email, ...])
report = compiler.validate_sequence(
    [Call("authenticate", {"user": "alice"}),
     Call("send_email", {"user": "alice", "to": "bob@x.com"})],
    initial_state=set(),
)
print(report.valid)  # False
for v in report.violations:
    print(f"step {v.step} ({v.call.tool}): missing {v.missing}")
    print(f"  diagnosis: {v.diagnosis}")
# step 1 (send_email): missing ['profile_loaded(alice)']
#   diagnosis: 'profile_loaded(alice)' is never established by any call in the plan

# 2. Enforce at runtime inside your StateGraph
node = ContractToolNode([send_email, ...], initial_state=set())
graph.add_node("tools", node)  # drop-in for your tool-execution node
# A doomed call returns a ToolMessage instead of executing:
# "CONTRACT VIOLATION — call to 'send_email' blocked: preconditions not met:
#  ['authenticated(alice)', 'profile_loaded(alice)']. Replan with a call
#  that establishes them first."

# 3. Parallelize safe fan-outs automatically
layers, _ = compiler.compile_plan(calls, initial_state=set())
# authenticate → [fetch_profile, fetch_orders]  (1.5x simulated speedup)
```

## Measured results

20-task benchmark, 11-tool mock belt, 5 seeds. Both agents use the same
scripted description-overlap planner (a documented proxy for prompt-driven
routing); the contracted agent's planner reads the contracts.

| Metric (per 20 tasks, mean of 5 seeds) | Plain agent | Contracted |
|---|---|---|
| Precondition-violation failures | 27.0 | **0.0** |
| Goals reached | ~0/20 | **20/20** |
| Static violation probes caught | — | **3/3** |
| Plain's violations blocked by `ContractToolNode` | — | **27.0/27.0** |
| Fan-out speedup (auth → profile+orders) | 1.00× | **1.50×** |

Full method, ablations, and honest limitations in [paper/PAPER.md](paper/PAPER.md).

## How it works

```
@contracted_tool ──► ToolContract (Pydantic: pre/add/delete/cond_effects/cost)
        │
        ▼
ContractGraphCompiler ──► validate_sequence() → violations + diagnoses
        │                   compile_plan() → parallel layers (effect-intersection)
        ▼
ContractToolNode ──► validates every tool call inside your StateGraph
                     blocks violations, returns replan hints
```

**Effect-intersection parallelization** (from our sibling project
[ContractOrch](https://github.com/habib-analyst/contractorch)): two calls
share a parallel layer iff their contracts prove it safe — no write-write
conflict, no cross-dependency. Set arithmetic, not guesswork.

## Add contracts to your own tools

Three steps, one tool at a time — no rewrite needed:

1. **Decorate**: add `@contracted_tool(pre=[...], add=[...], delete=[...])`
   outside your existing `@tool`. Predicates are strings like
   `"authenticated($user)"`; `$var` binds to the call's arguments.
2. **Validate plans**: build a `ContractGraphCompiler.from_tools([...])`
   and call `validate_sequence()` on your agent's planned calls before
   executing — catch impossible transitions in CI, not production.
3. **Enforce in the graph**: replace your tool-execution node with
   `ContractToolNode(tools, initial_state={...})`. It validates every
   call against live contract state and blocks violations with a
   diagnosis the agent can replan from.

Start with your most failure-prone tools (anything with ordering
constraints: auth before API, create before read, parse before aggregate).
Contracts are trusted declarations — if a tool's contract lies about its
effects, the guarantees are void, so keep contracts next to the
implementation and review them together.

*Upstream note: this pattern is proposed as a candidate contribution to
the LangGraph ecosystem — contracts as first-class, optional metadata on
tools, with validation available both statically and inside `ToolNode`.*

## Citation

```bibtex
@software{rehman2026langgraphcontracts,
  author = {Rehman, Habib Ur},
  title = {langgraph-contracts: Contract-Based Tool Orchestration for LangGraph},
  year = {2026},
  url = {https://github.com/habib-analyst/langgraph-contracts}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
