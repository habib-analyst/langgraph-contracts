# langgraph-contracts — STATUS

## DONE 2026-10-13 ~01:15 PKT
- Kill-gate: PASS — static validation 3/3 probes caught; contracted 0.0 failures/20 vs plain 27.0/20 (5 seeds); contracted goal 20/20 vs plain ~0/20; safety net blocked 27.0/27.0 plain violations; fan-out speedup 1.5x
- 24/24 tests green; real LangGraph StateGraph demo runs (examples/quickstart.py verified)
- Paper, branded README, LinkedIn draft, LICENSE, .gitignore, pyproject.toml all staged
- Push payload: hidden_files/langgraph_contracts_push_payload.json (staged by parent)
- Logged to RESEARCH_LOG.md below
- REPO NEEDED: langgraph-contracts

## Build log
- 00:37: started; read contractorch DSL (contracts.py, parallel.py, registry.py) + PAPER.md + BRANDING.md
- venv with langgraph 1.2.14 + langchain-core + pydantic + pytest (pip install, no --break-system-packages)
- Fixed: StructuredTool (Pydantic) rejects setattr → name-keyed contract registry (works in either decorator order)
- Fixed: ContractToolNode eager t.__name__ default → AttributeError on StructuredTool
- Fixed: benchmark tasks missing file_not_exists initial facts → contracted planner had nothing applicable
- Design iteration (honest): first contracted agent blocked-then-excluded calls permanently and never converged (0/20 goals); replaced with contract-guided planner (only applicable candidates) + runtime node as safety net — 20/20 goals, 0 failures
