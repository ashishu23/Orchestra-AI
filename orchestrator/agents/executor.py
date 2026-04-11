import json
from typing import Any

from mcp.client import MCPClient, MCPError
from mcp.vault_client import VaultClient
from mcp.sandbox_client import SandboxClient
from config import settings

from .state import OrchestraState


def _get_clients() -> tuple[VaultClient, SandboxClient]:
    vault_mcp = MCPClient(settings.vault_url, timeout=settings.mcp_timeout)
    sandbox_mcp = MCPClient(settings.sandbox_url, timeout=settings.mcp_timeout)
    return VaultClient(vault_mcp), SandboxClient(sandbox_mcp)


async def executor_node(state: OrchestraState) -> dict[str, Any]:
    """
    Execute the current plan step by dispatching to the appropriate MCP tool.
    Updates the step's result in the plan.
    """
    plan = list(state["plan"])
    idx = state["current_step_index"]

    if idx >= len(plan):
        return {"status": "verifying"}

    step = dict(plan[idx])
    tool = step["tool"]
    args = step["arguments"]

    vault_client, sandbox_client = _get_clients()

    try:
        if tool in ("search_semantic", "search_hybrid", "search_keyword"):
            result = await vault_client._c.call(tool, args)
            # Serialize result chunks to a readable string for the critic
            result_str = json.dumps(result, indent=2)
            retrieved_chunks = list(state["retrieved_chunks"]) + [
                {"step_id": step["step_id"], "chunks": result}
            ]
            step["result"] = result_str
            plan[idx] = step
            return {
                "plan": plan,
                "retrieved_chunks": retrieved_chunks,
                "status": "verifying",
            }

        elif tool == "execute_python":
            result = await sandbox_client.execute_python(args.get("code", ""))
            result_str = json.dumps(result, indent=2)
            execution_results = list(state["execution_results"]) + [
                {"step_id": step["step_id"], "result": result}
            ]
            step["result"] = result_str
            plan[idx] = step
            return {
                "plan": plan,
                "execution_results": execution_results,
                "status": "verifying",
            }

    except MCPError as e:
        step["result"] = f"MCPError {e.code}: {e.message}"
        plan[idx] = step
        return {"plan": plan, "status": "verifying"}

    except Exception as e:
        step["result"] = f"ExecutionError: {str(e)}"
        plan[idx] = step
        return {"plan": plan, "status": "verifying"}

    finally:
        await vault_client._c.aclose()
        await sandbox_client._c.aclose()
