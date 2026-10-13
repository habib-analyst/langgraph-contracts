"""Real LangGraph StateGraph wired with ContractToolNode.

A scripted "LLM" emits tool calls as AIMessages (no API key needed); the
graph routes them through ContractToolNode, which validates each call
against the live contract state. Run: ./venv/bin/python examples/quickstart.py
"""

import sys
sys.path.insert(0, "src")

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from osscontrib import ContractGraphCompiler, ContractToolNode
from osscontrib.demo_tools import ALL_TOOLS, reset_world


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # scripted plan injected by the "LLM"
    script: list[dict]


def scripted_llm(state: AgentState) -> dict:
    """Fake LLM: pops the next planned tool calls and emits them as an AIMessage."""
    script = list(state.get("script", []))
    if not script:
        return {"messages": [AIMessage(content="done")] , "script": []}
    batch = script[:2]  # emit up to 2 calls per turn
    rest = script[2:]
    tool_calls = [
        {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}
        for i, (name, args) in enumerate(batch)
    ]
    return {"messages": [AIMessage(content="", tool_calls=tool_calls)], "script": rest}


def main() -> None:
    reset_world()
    node = ContractToolNode(ALL_TOOLS, initial_state=set())

    graph = StateGraph(AgentState)
    graph.add_node("llm", scripted_llm)
    graph.add_node("tools", node)

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    app = graph.compile()

    # A good plan…
    good_script = [
        ("authenticate", {"user": "alice"}),
        ("fetch_profile", {"user": "alice"}),
        ("send_email", {"user": "alice", "to": "bob@x.com"}),
    ]
    out = app.invoke({"messages": [], "script": good_script})
    print("== good plan ==")
    for m in out["messages"]:
        if getattr(m, "tool_calls", None):
            print("  call:", [(tc["name"], tc["args"]) for tc in m.tool_calls])
        else:
            print("  result:", str(m.content)[:80])
    print("  contract state has email_sent(bob@x.com):",
          "email_sent(bob@x.com)" in node.state)

    # …and a doomed plan the node refuses to execute
    node.reset(set())
    bad_script = [("send_email", {"user": "alice", "to": "bob@x.com"})]
    out = app.invoke({"messages": [], "script": bad_script})
    print("== doomed plan ==")
    for m in out["messages"]:
        print("  result:", str(m.content)[:100])
    print("  prevented violations:", len(node.prevented))

    # Static validation before running anything
    compiler = ContractGraphCompiler.from_tools(ALL_TOOLS)
    from osscontrib import Call
    report = compiler.validate_sequence(
        [Call("authenticate", {"user": "alice"}),
         Call("send_email", {"user": "alice", "to": "bob@x.com"})],
        set(),
    )
    print("== static validation of doomed plan ==")
    print("  valid:", report.valid)
    for v in report.violations:
        print(f"  step {v.step} ({v.call.tool}): missing {v.missing}")
        print(f"    diagnosis: {v.diagnosis}")


if __name__ == "__main__":
    main()
