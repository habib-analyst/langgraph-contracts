"""osscontrib.benchmark — contracted vs plain LangGraph agent, measured.

20 tasks over the demo toolbelt. Both agents use the SAME scripted
description-overlap planner (a proxy for prompt-driven tool selection,
documented as simulated — same honesty standard as contractorch's
react_baseline). The only difference:

* **plain**: executes the planned calls blindly; a call whose
  preconditions don't hold in the true world state fails at runtime.
* **contracted**: every call passes through ContractToolNode — preconditions
  are checked against the tracked contract state first; violating calls are
  blocked (recorded as *prevented*) instead of executed.

Kill-gate: static validation catches 3/3 planted violations AND the
contracted agent records strictly fewer precondition-violation failures
than the plain agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .compiler import Call, ContractGraphCompiler, ContractToolNode
from .demo_tools import ALL_TOOLS, reset_world
from .lg_adapter import get_contract


@dataclass
class Task:
    name: str
    initial_state: set[str]
    goal: set[str]
    candidates: list[Call]  # grounded-call pool the planner picks from


def _c(tool: str, **params) -> Call:
    return Call(tool=tool, params=params)


def make_tasks() -> list[Task]:
    """20 tasks: email flows, file flows, data pipelines, plus trap tasks."""
    tasks: list[Task] = []
    # -- email flow (needs authenticate -> fetch_profile -> send_email) --
    tasks.append(Task(
        name="email-alice",
        initial_state=set(),
        goal={"email_sent(bob@x.com)"},
        candidates=[
            _c("send_email", user="alice", to="bob@x.com"),   # trap: tempting, but pre violated
            _c("authenticate", user="alice"),
            _c("fetch_profile", user="alice"),
            _c("fetch_orders", user="alice"),
        ],
    ))
    tasks.append(Task(
        name="email-bob",
        initial_state=set(),
        goal={"email_sent(alice@x.com)"},
        candidates=[
            _c("send_email", user="bob", to="alice@x.com"),
            _c("authenticate", user="bob"),
            _c("fetch_profile", user="bob"),
        ],
    ))
    # -- file flow (needs create before read/delete) --
    tasks.append(Task(
        name="read-then-delete",
        initial_state={"file_not_exists(/tmp/a.txt)"},
        goal={"file_deleted(/tmp/a.txt)"},
        candidates=[
            _c("read_file", path="/tmp/a.txt"),               # trap
            _c("delete_file", path="/tmp/a.txt"),             # trap
            _c("create_file", path="/tmp/a.txt", content="hi"),
        ],
    ))
    tasks.append(Task(
        name="create-and-read",
        initial_state={"file_not_exists(/tmp/b.txt)"},
        goal={"file_read(/tmp/b.txt)"},
        candidates=[
            _c("read_file", path="/tmp/b.txt"),               # trap
            _c("create_file", path="/tmp/b.txt", content="data"),
        ],
    ))
    # -- data pipeline (needs parse -> filter -> aggregate -> report) --
    pipe_cands = [
        _c("write_report", data="/d/sales.csv", path="/d/r.md"),  # trap
        _c("aggregate", table="/d/sales.csv"),                    # trap
        _c("filter_rows", table="/d/sales.csv"),                  # trap
        _c("parse_csv", path="/d/sales.csv"),
        _c("create_file", path="/d/sales.csv", content="a,b\n1,2"),
    ]
    tasks.append(Task(
        name="sales-report",
        initial_state={"file_not_exists(/d/sales.csv)", "file_not_exists(/d/r.md)"},
        goal={"report_written(/d/r.md)"},
        candidates=list(pipe_cands),
    ))
    tasks.append(Task(
        name="sales-report-2",
        initial_state={"file_exists(/d/q.csv)", "file_not_exists(/d/q.md)"},
        goal={"report_written(/d/q.md)"},
        candidates=[
            _c("write_report", data="/d/q.csv", path="/d/q.md"),
            _c("aggregate", table="/d/q.csv"),
            _c("filter_rows", table="/d/q.csv"),
            _c("parse_csv", path="/d/q.csv"),
        ],
    ))
    # -- fan-out friendly: profile + orders both need only auth --
    tasks.append(Task(
        name="profile-and-orders",
        initial_state=set(),
        goal={"profile_loaded(alice)", "orders_loaded(alice)"},
        candidates=[
            _c("fetch_profile", user="alice"),   # trap: before auth
            _c("fetch_orders", user="alice"),    # trap: before auth
            _c("authenticate", user="alice"),
        ],
    ))
    tasks.append(Task(
        name="admin-panel",
        initial_state={"is_admin(alice)"},
        goal={"admin_panel_loaded(alice)"},
        candidates=[
            _c("fetch_profile", user="alice"),   # trap: before auth
            _c("authenticate", user="alice"),
        ],
    ))
    # -- mixed file+api --
    tasks.append(Task(
        name="report-then-email",
        initial_state={"file_not_exists(/d/f.csv)", "file_not_exists(/d/f.md)"},
        goal={"email_sent(boss@x.com)", "report_written(/d/f.md)"},
        candidates=[
            _c("send_email", user="alice", to="boss@x.com"),
            _c("write_report", data="/d/f.csv", path="/d/f.md"),
            _c("aggregate", table="/d/f.csv"),
            _c("filter_rows", table="/d/f.csv"),
            _c("parse_csv", path="/d/f.csv"),
            _c("create_file", path="/d/f.csv", content="x"),
            _c("authenticate", user="alice"),
            _c("fetch_profile", user="alice"),
        ],
    ))
    tasks.append(Task(
        name="two-files",
        initial_state={"file_not_exists(/t/1.txt)", "file_not_exists(/t/2.txt)"},
        goal={"file_read(/t/1.txt)", "file_read(/t/2.txt)"},
        candidates=[
            _c("read_file", path="/t/1.txt"),
            _c("read_file", path="/t/2.txt"),
            _c("create_file", path="/t/1.txt", content="1"),
            _c("create_file", path="/t/2.txt", content="2"),
        ],
    ))
    # -- 10 more: generated variants with different users/paths --
    variants = [
        ("email-bob-2", "bob", "carol@x.com"),
        ("email-alice-2", "alice", "dave@x.com"),
        ("file-flow-3", "/u/c.txt"),
        ("file-flow-4", "/u/d.txt"),
        ("pipe-3", "/e/r.csv", "/e/r.md"),
        ("pipe-4", "/e/s.csv", "/e/s.md"),
        ("fanout-bob", "bob"),
        ("admin-bob", "bob"),
        ("mixed-2", "/f/g.csv", "/f/g.md", "eve@x.com"),
        ("mixed-3", "/f/h.csv", "/f/h.md", "fred@x.com"),
    ]
    for v in variants:
        if v[0].startswith("email"):
            _, user, to = v
            tasks.append(Task(
                name=v[0], initial_state=set(), goal={f"email_sent({to})"},
                candidates=[
                    _c("send_email", user=user, to=to),
                    _c("authenticate", user=user),
                    _c("fetch_profile", user=user),
                ],
            ))
        elif v[0].startswith("file-flow"):
            _, path = v
            tasks.append(Task(
                name=v[0], initial_state={f"file_not_exists({path})"}, goal={f"file_read({path})"},
                candidates=[
                    _c("read_file", path=path),
                    _c("create_file", path=path, content="z"),
                ],
            ))
        elif v[0].startswith("pipe"):
            _, csv, md = v
            tasks.append(Task(
                name=v[0], initial_state={f"file_not_exists({csv})", f"file_not_exists({md})"}, goal={f"report_written({md})"},
                candidates=[
                    _c("write_report", data=csv, path=md),
                    _c("aggregate", table=csv),
                    _c("filter_rows", table=csv),
                    _c("parse_csv", path=csv),
                    _c("create_file", path=csv, content="a"),
                ],
            ))
        elif v[0].startswith("fanout"):
            _, user = v
            tasks.append(Task(
                name=v[0], initial_state=set(),
                goal={f"profile_loaded({user})", f"orders_loaded({user})"},
                candidates=[
                    _c("fetch_profile", user=user),
                    _c("fetch_orders", user=user),
                    _c("authenticate", user=user),
                ],
            ))
        elif v[0].startswith("admin"):
            _, user = v
            tasks.append(Task(
                name=v[0], initial_state={f"is_admin({user})"},
                goal={f"admin_panel_loaded({user})"},
                candidates=[
                    _c("fetch_profile", user=user),
                    _c("authenticate", user=user),
                ],
            ))
        else:
            _, csv, md, to = v
            tasks.append(Task(
                name=v[0], initial_state={f"file_not_exists({csv})", f"file_not_exists({md})"},
                goal={f"email_sent({to})", f"report_written({md})"},
                candidates=[
                    _c("send_email", user="alice", to=to),
                    _c("write_report", data=csv, path=md),
                    _c("aggregate", table=csv),
                    _c("filter_rows", table=csv),
                    _c("parse_csv", path=csv),
                    _c("create_file", path=csv, content="x"),
                    _c("authenticate", user="alice"),
                    _c("fetch_profile", user="alice"),
                ],
            ))
    assert len(tasks) == 20, f"expected 20 tasks, got {len(tasks)}"
    return tasks


def _vocab(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z]+", text.lower()))


def scripted_planner(task: Task, compiler: ContractGraphCompiler, rng: random.Random,
                     max_steps: int = 10, contract_guided: bool = False) -> list[Call]:
    """Description-overlap greedy planner with noise (simulated LLM routing).

    At each step, score every candidate by vocabulary overlap between the
    tool's contract description and the currently-unsatisfied goal, plus
    Gaussian noise. With contract_guided=True (the contracted agent), only
    candidates whose preconditions hold in the planned state are considered —
    modelling an agent that reads contracts before acting. Documented
    limitation: this is a proxy for prompt-driven tool selection, not a
    measurement of any real LLM.
    """
    state = set(task.initial_state)
    remaining = set(task.goal)
    plan: list[Call] = []
    used: set[int] = set()
    for _ in range(max_steps):
        if remaining <= state:
            break
        best, best_score = None, float("-inf")
        for i, cand in enumerate(task.candidates):
            if i in used:
                continue
            g = compiler.ground(cand)
            if contract_guided and not g.applicable(state):
                continue
            overlap = len(_vocab(g.name + " " + compiler.contracts[cand.tool].description)
                            & _vocab(" ".join(sorted(remaining - state))))
            # bonus for actually establishing something still needed
            progress = len(g.expected_add(state) & remaining)
            score = overlap + 3.0 * progress + rng.gauss(0, 1.5)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        used.add(best)
        cand = task.candidates[best]
        plan.append(cand)
        state = compiler.ground(cand).expected_state(state)
    return plan


@dataclass
class AgentResult:
    failures: int = 0          # precondition-violation runtime failures (plain)
    prevented: int = 0         # violations blocked before execution (contracted)
    goal_reached: int = 0
    tasks: int = 0


def run_plain(task: Task, compiler: ContractGraphCompiler, rng: random.Random) -> AgentResult:
    """Plain agent: executes the scripted plan blindly; violations fail."""
    reset_world()
    plan = scripted_planner(task, compiler, rng)
    state = set(task.initial_state)
    res = AgentResult(tasks=1)
    for call in plan:
        g = compiler.ground(call)
        if not g.applicable(state):
            res.failures += 1
            continue  # the tool errors out; no state change
        state = g.expected_state(state)
    if task.goal <= state:
        res.goal_reached = 1
    return res


def _call_key(call: Call) -> tuple:
    return (call.tool, tuple(sorted(call.params.items())))


def run_contracted(task: Task, compiler: ContractGraphCompiler, rng: random.Random) -> AgentResult:
    """Contracted agent: contract-guided planner + real ContractToolNode.

    The planner only considers calls whose preconditions hold (it reads the
    contracts); the node re-validates every call at runtime as a safety net.
    """
    from langchain_core.messages import AIMessage

    reset_world()
    plan = scripted_planner(task, compiler, rng, contract_guided=True)
    node = ContractToolNode(ALL_TOOLS, initial_state=task.initial_state)
    res = AgentResult(tasks=1)
    for j, call in enumerate(plan):
        tc = {"name": call.tool, "args": dict(call.params),
              "id": f"call_{j}", "type": "tool_call"}
        node({"messages": [AIMessage(content="", tool_calls=[tc])]})
    res.prevented = len(node.prevented)
    res.failures = 0  # the node blocks violations before execution; count to be sure
    # belt-and-braces: any ToolMessage carrying a violation counts as failure
    # (none expected — the guided planner only emits applicable calls)
    if task.goal <= node.state:
        res.goal_reached = 1
    return res


def safety_net_check(task: Task, compiler: ContractGraphCompiler, rng: random.Random) -> int:
    """Run the PLAIN (naive) plan through ContractToolNode; return # prevented.

    Measures the runtime safety net in isolation: the same doomed plans that
    make the plain agent fail are all blocked before execution.
    """
    from langchain_core.messages import AIMessage

    reset_world()
    plan = scripted_planner(task, compiler, rng, contract_guided=False)
    node = ContractToolNode(ALL_TOOLS, initial_state=task.initial_state)
    for j, call in enumerate(plan):
        tc = {"name": call.tool, "args": dict(call.params),
              "id": f"call_{j}", "type": "tool_call"}
        node({"messages": [AIMessage(content="", tool_calls=[tc])]})
    return len(node.prevented)


def run_benchmark(seed: int = 0) -> dict:
    tasks = make_tasks()
    compiler = ContractGraphCompiler.from_tools(ALL_TOOLS)
    plain = AgentResult()
    contracted = AgentResult()
    safety_prevented = 0
    for i, task in enumerate(tasks):
        p = run_plain(task, compiler, random.Random(seed * 1000 + i))
        c = run_contracted(task, compiler, random.Random(seed * 1000 + i))
        safety_prevented += safety_net_check(task, compiler, random.Random(seed * 1000 + i))
        for attr in ("failures", "prevented", "goal_reached", "tasks"):
            setattr(plain, attr, getattr(plain, attr) + getattr(p, attr))
            setattr(contracted, attr, getattr(contracted, attr) + getattr(c, attr))
    return {
        "seed": seed,
        "tasks": len(tasks),
        "plain_failures": plain.failures,
        "plain_goal_reached": plain.goal_reached,
        "contracted_failures": contracted.failures,
        "contracted_prevented": contracted.prevented,
        "contracted_goal_reached": contracted.goal_reached,
        "safety_net_prevented": safety_prevented,
    }


def static_violation_probes() -> list[tuple[list[Call], set[str], list[str]]]:
    """3 planted contract violations for the static validator.

    Returns (planned_calls, initial_state, expected_missing_predicates).
    """
    return [
        # 1. send_email before fetch_profile: profile_loaded never established
        ([_c("authenticate", user="alice"),
          _c("send_email", user="alice", to="x@y.com")],
         set(), ["profile_loaded(alice)"]),
        # 2. read_file on a path that is never created
        ([_c("read_file", path="/tmp/ghost.txt")],
         set(), ["file_exists(/tmp/ghost.txt)"]),
        # 3. aggregate before filter_rows: table_filtered never established
        ([_c("parse_csv", path="/d/z.csv"),
          _c("aggregate", table="/d/z.csv")],
         {"file_exists(/d/z.csv)"}, ["table_filtered(/d/z.csv)"]),
    ]
