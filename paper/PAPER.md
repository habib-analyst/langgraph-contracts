# langgraph-contracts: Contract-Based Tool Orchestration for LangGraph

**Habib Ur Rehman** · October 2026

## Abstract

LangGraph agents pick tools by description matching — the model reads tool
docstrings and guesses. This produces plans that fail at runtime on
precondition violations (calling `send_email` before `authenticate`), and
fan-out parallelism that is never attempted or never safe. Building on the
contract-planning approach proven in our sibling project ContractOrch, we
bring **machine-checkable tool contracts** natively into LangGraph: a
`@contracted_tool` decorator attaches Pydantic-typed preconditions/effects
to LangChain tools; a `ContractGraphCompiler` statically validates planned
tool sequences before execution; `ContractToolNode` re-validates every call
at runtime inside a real `StateGraph`; and effect-intersection analysis
derives provably-safe parallel fan-outs. On a 20-task benchmark with a
scripted description-overlap planner, the contracted agent records **0
precondition-violation failures vs 27.0 per 20 tasks for the plain agent**,
reaches the goal on **20/20 tasks vs ~0/20**, and the runtime node blocks
**100% of the plain agent's violations** when fed its doomed plans.
Fan-out parallelization yields a **1.5× simulated speedup** on a
profile+orders fan-out.

## 1. The gap

Three strands of prior work surround this problem; none puts contracts
*inside* the LangGraph execution loop:

- **Constrained generation / validators** (Instructor, Guardrails,
  structured output): validate tool *arguments* against schemas. They say
  nothing about *sequencing* — whether the call is legal in the current
  world state.
- **Contract-filtered selection** (Contract2Tool and the tool-contract
  literature): contracts rank candidate tools for the LLM. The LLM still
  plans, parallelism is unaddressed, and nothing validates the executed
  sequence.
- **ContractOrch** (our sibling project): plans directly over contracts
  with provably-safe parallelization — but as a standalone planner, not
  integrated with LangGraph's `StateGraph` / `ToolNode` execution model.

To our knowledge, **no existing work attaches machine-checkable contracts
to LangGraph tools, validates planned sequences statically before
execution, *and* enforces contracts at runtime inside the graph**. That is
the gap this project fills — framed as a reusable pattern the LangGraph
community can adopt (see README: "how to add contracts to your own tools").

## 2. Method

### 2.1 `@contracted_tool`: contracts on LangChain tools

```python
@contracted_tool(
    pre=["authenticated($user)", "profile_loaded($user)"],
    add=["email_sent($to)"],
    cost=1.5,
)
@tool
def send_email(user: str, to: str) -> str: ...
```

The decorator builds a validated `ToolContract` (Pydantic: `pre` / `add` /
`delete` / `cond_effects` / `cost`, `$var` templates grounded per call)
and attaches it to the `StructuredTool`. Because `StructuredTool` is
itself a Pydantic model and rejects new attributes, contracts live in a
name-keyed registry — so the decorator works in either order (`@tool`
inside or outside). `ContractGraphCompiler.from_tools()` refuses
uncontracted tools with a clear error.

### 2.2 Static validation: `ContractGraphCompiler.validate_sequence`

Given a planned call sequence and an initial world state, the compiler
folds contracts through a simulated state and reports every precondition
violation with a **diagnosis**: which earlier call should have established
the missing predicate (with a reorder hint), or that no call in the plan
establishes it at all. `compile_plan` additionally compresses the
validated sequence into parallel layers via effect-intersection: two calls
share a layer iff `effects(A) ∩ effects(B) = ∅` and neither reads what the
other writes.

### 2.3 Runtime enforcement: `ContractToolNode`

A LangGraph node (drop-in for the tool-execution step of a ReAct agent).
Before invoking each tool it grounds the tool's contract against the call
arguments and checks preconditions against the tracked contract state. A
violating call is **not executed** — a `ToolMessage` carrying the
violation and a replan hint is returned instead, so the agent replans
rather than watching the tool fail downstream. The node tracks state
purely from declared effects.

## 3. Benchmark

20 tasks over an 11-tool mock belt (file ops, API, data pipelines), each
seeded with contract *traps*: the naive description-overlap planner is
tempted by calls whose preconditions fail (e.g. `send_email` before
`authenticate`). Both agents use the **same** scripted planner
(description-overlap + Gaussian noise); the contracted agent's planner
additionally filters to applicable calls (it reads the contracts). The
planner is a documented proxy for prompt-driven routing, not a real LLM.

| Metric (mean of 5 seeds × 20 tasks) | Plain agent | Contracted agent |
|---|---|---|
| Precondition-violation failures / 20 tasks | **27.0** | **0.0** |
| Goal reached | ~0/20 | **20/20** |
| Static violation probes caught | n/a | **3/3** |
| Safety net: plain's violations blocked by node | n/a | **27.0/27.0** |
| Fan-out simulated speedup (auth → profile+orders) | 1.00× | **1.50×** |

Kill-gate (3/3 probes caught AND strictly fewer failures): **PASS**.

## 4. Worked example (real StateGraph, in `examples/quickstart.py`)

A `StateGraph` with a scripted LLM node and `ContractToolNode` runs a good
plan (`authenticate → fetch_profile → send_email`) to completion, then
blocks a doomed plan (`send_email` first) with:

> `CONTRACT VIOLATION — call to 'send_email' blocked: preconditions not
> met: ['authenticated(alice)', 'profile_loaded(alice)']. Replan with a
> call that establishes them first.`

Static validation of the same doomed plan reports: step 1
(`send_email`): missing `['profile_loaded(alice)']` — *"never established
by any call in the plan."*

## 5. Limitations (honest)

- **Mock toolset, simulated costs and planner.** The 11 tools are
  deterministic in-memory mocks; speedup is in declared step-costs, not
  wall-clock; the description-overlap planner is a proxy, not any real
  LLM. The failure counts measure the *mechanism* (validation works),
  not a production failure rate.
- **Contracts are trusted.** If a tool's contract lies about its
  effects, the state tracker diverges from reality and all guarantees
  are void. Contract authorship and verification are out of scope.
- **No automatic contract inference.** Contracts are hand-written per
  tool; mining them from code or traces is future work.
- **State is predicate-based.** Tools with rich side effects outside the
  predicate vocabulary (e.g. partial writes) are not modelled.

## 6. Conclusion

Contracts turn the LangGraph tool loop from guess-and-check into
check-then-run: impossible plans are caught before execution, doomed
calls are blocked at runtime with actionable diagnoses, and safe
parallelism falls out of set arithmetic. The `@contracted_tool` pattern
is deliberately small — one decorator per tool — so existing LangGraph
codebases can adopt it incrementally, one tool at a time.

## Citation

```bibtex
@software{rehman2026langgraphcontracts,
  author = {Rehman, Habib Ur},
  title = {langgraph-contracts: Contract-Based Tool Orchestration for LangGraph},
  year = {2026},
  url = {https://github.com/habib-analyst/langgraph-contracts}
}
```
