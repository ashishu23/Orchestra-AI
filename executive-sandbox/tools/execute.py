from sandbox.runner import run_sandboxed


async def execute_python(code: str) -> dict:
    """
    Execute Python code in an isolated sandbox with CPU, memory, and time limits.

    Useful for verifying numerical claims from retrieved research context
    (e.g., computing statistics, validating benchmarks).

    Args:
        code: Python source code to execute. Only standard library and
              pre-approved scientific modules are available.

    Returns:
        {
            "stdout": str,   - captured standard output
            "error": str | None,  - traceback string if execution failed
            "success": bool  - True if no exception was raised
        }
    """
    return await run_sandboxed(code)
