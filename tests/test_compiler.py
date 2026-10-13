"""Tests: static validation + ContractToolNode + benchmark kill-gate."""
from langchain_core.messages import AIMessage

from osscontrib.benchmark import (
    make_tasks,
    run_benchmark,
    run_contracted,
    run_plain,
    static_violation_probes,
)
from osscontrib.compiler import Call, ContractGraphCompiler, ContractToolNode
from osscontrib.demo_tools import ALL_TOOLS, reset_world


def _compiler():
    return ContractGraphCompiler.from_tools(ALL_TOOLS)


def test_from_tools_rejects_uncontracted():
    from langchain_core.tools import tool

    @tool
    def naked(x: int) -> str:
        """No contract."""
        return "y"

    try:
        ContractGraphCompiler.from_tools([naked])
    except ValueError as e:
        assert "no contract" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_static_validation_catches_all_planted():
    comp = _compiler()
    probes = static_violation_probes()
    assert len(probes) == 3
    for calls, initial, expected_missing in probes:
        report = comp.validate_sequence(calls, initial)
        assert not report.valid, f"probe not caught: {[c.tool for c in calls]}"
        found = sorted({m for v in report.violations for m in v.missing})
        assert found == sorted(expected_missing), f"{found} != {expected_missing}"
        # diagnosis must name the problem
        assert report.violations[0].diagnosis


def test_static_validation_passes_good_plan():
    comp = _compiler()
    calls = [
        Call("authenticate", {"user": "alice"}),
        Call("fetch_profile", {"user": "alice"}),
        Call("send_email", {"user": "alice", "to": "b@x.com"}),
    ]
    report = comp.validate_sequence(calls, set())
    assert report.valid
    assert {"email_sent(b@x.com)"} <= report.final_state


def test_compile_plan_raises_on_violation():
    comp = _compiler()
    calls = [Call("send_email", {"user": "alice", "to": "b@x.com"})]
    try:
        comp.compile_plan(calls, set())
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_speedup_greater_than_one_on_fanout():
    comp = _compiler()
    calls = [
        Call("authenticate", {"user": "alice"}),
        Call("fetch_profile", {"user": "alice"}),
        Call("fetch_orders", {"user": "alice"}),
    ]
    assert comp.speedup(calls, set()) > 1.0


def test_tool_node_blocks_violation():
    reset_world()
    node = ContractToolNode(ALL_TOOLS, initial_state=set())
    tc = {"name": "send_email", "args": {"user": "alice", "to": "b@x.com"},
          "id": "c0", "type": "tool_call"}
    out = node({"messages": [AIMessage(content="", tool_calls=[tc])]})
    assert len(node.prevented) == 1
    assert node.prevented[0]["missing"] == ["authenticated(alice)", "profile_loaded(alice)"]
    assert "CONTRACT VIOLATION" in out["messages"][0].content
    assert len(node.executed) == 0


def test_tool_node_executes_valid_call():
    reset_world()
    node = ContractToolNode(ALL_TOOLS, initial_state=set())
    tc = {"name": "authenticate", "args": {"user": "alice"},
          "id": "c0", "type": "tool_call"}
    out = node({"messages": [AIMessage(content="", tool_calls=[tc])]})
    assert len(node.executed) == 1
    assert "authenticated(alice)" in node.state
    assert "CONTRACT VIOLATION" not in out["messages"][0].content


def test_benchmark_kill_gate():
    # contracted must have strictly fewer precondition-violation failures
    plain_fail = cont_fail = 0
    for seed in (0, 1, 2):
        r = run_benchmark(seed=seed)
        assert r["tasks"] == 20
        plain_fail += r["plain_failures"]
        cont_fail += r["contracted_failures"]
        # safety net: every plain-agent violation is blocked by the node
        assert r["safety_net_prevented"] >= r["plain_failures"], (
            f"seed {seed}: node prevented {r['safety_net_prevented']} "
            f"but plain failed {r['plain_failures']}"
        )
    assert plain_fail > 0, "benchmark produced no violations for the plain agent"
    assert cont_fail < plain_fail
