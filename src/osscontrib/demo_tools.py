"""osscontrib.demo_tools — a demo LangGraph toolbelt with contracts.

Ten tools across file ops, API calls, and data transforms, each a real
``@tool`` with a real in-memory implementation and a machine-checkable
contract attached via ``@contracted_tool``. The world is an in-memory dict
so the contract-state tracker can be cross-checked against ground truth.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .lg_adapter import contracted_tool

# ---------------------------------------------------------------- in-memory world
WORLD: dict[str, dict] = {
    "files": {},       # path -> content
    "users": {},       # user -> {"authenticated": bool, "admin": bool, "profile": dict|None, "orders": list}
    "tables": {},      # path -> {"rows": [...], "filtered": [...], "aggregated": ...}
    "reports": {},     # path -> content
    "emails": [],      # sent emails
}


def reset_world() -> None:
    WORLD["files"].clear()
    WORLD["users"].clear()
    WORLD["tables"].clear()
    WORLD["reports"].clear()
    WORLD["emails"].clear()
    WORLD["users"]["alice"] = {"authenticated": False, "admin": True,
                               "profile": {"name": "Alice"}, "orders": [1, 2, 3]}
    WORLD["users"]["bob"] = {"authenticated": False, "admin": False,
                             "profile": {"name": "Bob"}, "orders": [4]}


# ---------------------------------------------------------------- file ops
@contracted_tool(
    pre=["file_not_exists($path)"],
    add=["file_exists($path)"],
    cost=1.0,
)
@tool
def create_file(path: str, content: str) -> str:
    """Create a new file with the given content."""
    WORLD["files"][path] = content
    return f"created {path}"


@contracted_tool(
    pre=["file_exists($path)"],
    add=["file_read($path)"],
    cost=0.5,
)
@tool
def read_file(path: str) -> str:
    """Read an existing file."""
    return WORLD["files"][path]


@contracted_tool(
    pre=["file_exists($path)"],
    add=["file_deleted($path)", "file_not_exists($path)"],
    delete=["file_exists($path)"],
    cost=0.5,
)
@tool
def delete_file(path: str) -> str:
    """Delete an existing file."""
    del WORLD["files"][path]
    return f"deleted {path}"


# ---------------------------------------------------------------- API
@contracted_tool(
    add=["authenticated($user)"],
    cost=2.0,
)
@tool
def authenticate(user: str) -> str:
    """Authenticate a user, enabling API calls."""
    WORLD["users"][user]["authenticated"] = True
    return f"{user} authenticated"


@contracted_tool(
    pre=["authenticated($user)"],
    add=["profile_loaded($user)"],
    cond_effects=[{
        "when": ["is_admin($user)"],
        "add": ["admin_panel_loaded($user)"],
        "delete": [],
    }],
    cost=2.0,
)
@tool
def fetch_profile(user: str) -> str:
    """Fetch a user's profile (admins also unlock the admin panel)."""
    u = WORLD["users"][user]
    extra = " + admin panel" if u["admin"] else ""
    return f"profile({user}){extra}"


@contracted_tool(
    pre=["authenticated($user)"],
    add=["orders_loaded($user)"],
    cost=2.0,
)
@tool
def fetch_orders(user: str) -> str:
    """Fetch a user's order history."""
    return f"orders({user})={WORLD['users'][user]['orders']}"


@contracted_tool(
    pre=["authenticated($user)", "profile_loaded($user)"],
    add=["email_sent($to)"],
    cost=1.5,
)
@tool
def send_email(user: str, to: str) -> str:
    """Send an email to a recipient."""
    WORLD["emails"].append({"from": user, "to": to})
    return f"email sent to {to}"


# ---------------------------------------------------------------- data transforms
@contracted_tool(
    pre=["file_exists($path)"],
    add=["table_loaded($path)"],
    cost=1.5,
)
@tool
def parse_csv(path: str) -> str:
    """Parse a CSV file into a table."""
    WORLD["tables"][path] = {"rows": [1, 2, 3], "filtered": [], "aggregated": None}
    return f"table loaded from {path}"


@contracted_tool(
    pre=["table_loaded($table)"],
    add=["table_filtered($table)"],
    cost=1.0,
)
@tool
def filter_rows(table: str) -> str:
    """Filter table rows."""
    t = WORLD["tables"][table]
    t["filtered"] = [r for r in t["rows"] if r > 1]
    return f"filtered {table}"


@contracted_tool(
    pre=["table_filtered($table)"],
    add=["table_aggregated($table)", "report_source_ready($table)"],
    cost=1.5,
)
@tool
def aggregate(table: str) -> str:
    """Aggregate a filtered table."""
    t = WORLD["tables"][table]
    t["aggregated"] = sum(t["filtered"])
    return f"aggregated {table} = {t['aggregated']}"


@contracted_tool(
    pre=["report_source_ready($data)", "file_not_exists($path)"],
    add=["report_written($path)", "file_exists($path)"],
    cost=1.0,
)
@tool
def write_report(data: str, path: str) -> str:
    """Write an analysis report."""
    WORLD["reports"][path] = f"report on {data}"
    WORLD["files"][path] = WORLD["reports"][path]
    return f"report written to {path}"


ALL_TOOLS = [
    create_file, read_file, delete_file,
    authenticate, fetch_profile, fetch_orders, send_email,
    parse_csv, filter_rows, aggregate, write_report,
]
