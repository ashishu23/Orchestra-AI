"""Unit tests for the Python execution sandbox."""
import pytest

from sandbox.runner import run_sandboxed


@pytest.mark.asyncio
async def test_successful_execution():
    result = await run_sandboxed("print('hello')")
    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["error"] is None


@pytest.mark.asyncio
async def test_syntax_error():
    result = await run_sandboxed("def broken(: pass")
    assert result["success"] is False
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_runtime_error():
    result = await run_sandboxed("1 / 0")
    assert result["success"] is False
    assert "ZeroDivisionError" in (result["error"] or "")


@pytest.mark.asyncio
async def test_stdout_capture():
    result = await run_sandboxed("for i in range(3): print(i)")
    assert result["success"] is True
    assert "0" in result["stdout"]
    assert "2" in result["stdout"]


@pytest.mark.asyncio
async def test_timeout():
    result = await run_sandboxed("while True: pass")
    assert result["success"] is False
    assert "Timeout" in (result["error"] or "") or "timeout" in (result["error"] or "").lower()


@pytest.mark.asyncio
async def test_math_computation():
    code = """
import statistics
data = [1, 2, 3, 4, 5]
print(f"mean={statistics.mean(data)}")
print(f"stdev={statistics.stdev(data):.4f}")
"""
    result = await run_sandboxed(code)
    assert result["success"] is True
    assert "mean=3" in result["stdout"]
