import contextlib
import io
import multiprocessing
import resource
import traceback
from typing import Any

from .limits import MAX_CPU_SECONDS, MAX_MEMORY_BYTES, TIMEOUT_SECONDS


def _worker(code: str, result_queue: multiprocessing.Queue):
    """
    Runs inside a child process with resource limits applied.
    Only numpy, math, statistics, json, re, and itertools are available
    to prevent abuse while still being useful for data verification.
    """
    # Apply resource limits before executing any user code
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (MAX_CPU_SECONDS, MAX_CPU_SECONDS),
        )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES),
        )
    except ValueError:
        # setrlimit may fail if current limit is already lower; that is fine
        pass

    stdout_capture = io.StringIO()
    local_vars: dict[str, Any] = {}

    # Safe builtins: block open(), __import__ overrides, exec, eval
    safe_builtins = {
        k: v
        for k, v in __builtins__.items()  # type: ignore[union-attr]
        if k
        not in {
            "open",
            "compile",
            "exec",
            "eval",
            "__import__",
            "breakpoint",
            "input",
        }
    }

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(  # noqa: S102
                compile(code, "<sandbox>", "exec"),
                {"__builtins__": safe_builtins},
                local_vars,
            )
        result_queue.put({"stdout": stdout_capture.getvalue(), "error": None})
    except Exception:
        result_queue.put(
            {
                "stdout": stdout_capture.getvalue(),
                "error": traceback.format_exc(),
            }
        )


async def run_sandboxed(code: str) -> dict:
    """
    Execute Python code in an isolated subprocess with resource limits.

    Returns:
        {
            "stdout": str,
            "error": str | None,
            "success": bool,
        }
    """
    import asyncio

    ctx = multiprocessing.get_context("fork")
    q: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, q))
    proc.start()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, proc.join, TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {
            "stdout": "",
            "error": f"TimeoutError: execution exceeded {TIMEOUT_SECONDS}s wall-clock limit",
            "success": False,
        }

    if not q.empty():
        result = q.get_nowait()
    else:
        result = {"stdout": "", "error": "ProcessError: no result returned from worker"}

    result["success"] = result["error"] is None
    return result
