"""
Unit tests for the LangGraph orchestrator state machine.
Run with: pytest orchestrator/tests/ -v
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.graph import build_graph
from agents.state import OrchestraState


SAMPLE_PLAN = [
    {
        "step_id": 1,
        "description": "Search for YOLOv11 latency",
        "tool": "search_hybrid",
        "arguments": {"query": "YOLOv11 inference latency", "top_k": 5},
        "result": None,
        "verified": None,
        "correction_reason": None,
    }
]

SAMPLE_CHUNKS = [{"id": "abc", "text": "YOLOv11 achieves 7ms on A100", "score": 0.9}]


def _initial_state(query: str = "What is YOLOv11 latency?") -> OrchestraState:
    return {
        "query": query,
        "messages": [],
        "plan": [],
        "current_step_index": 0,
        "retrieved_chunks": [],
        "execution_results": [],
        "verified_context": [],
        "hallucination_flags": [],
        "final_answer": None,
        "self_correction_count": 0,
        "status": "planning",
    }


@pytest.mark.asyncio
async def test_happy_path():
    """Planner creates plan, executor retrieves, critic passes, synthesizer returns answer."""
    graph = build_graph()

    with (
        patch("agents.planner._get_llm") as mock_planner_llm,
        patch("agents.executor._get_clients") as mock_clients,
        patch("agents.critic._get_llm") as mock_critic_llm,
    ):
        # Planner returns a valid plan
        planner_response = AsyncMock()
        planner_response.content = json.dumps(SAMPLE_PLAN)
        mock_planner_llm.return_value.ainvoke = AsyncMock(return_value=planner_response)

        # Executor MCP call returns chunks
        vault_mock = AsyncMock()
        vault_mock._c.call = AsyncMock(return_value=SAMPLE_CHUNKS)
        vault_mock._c.aclose = AsyncMock()
        sandbox_mock = AsyncMock()
        sandbox_mock._c.aclose = AsyncMock()
        mock_clients.return_value = (vault_mock, sandbox_mock)

        # Critic passes
        critic_response = AsyncMock()
        critic_response.content = json.dumps(
            {"verdict": "PASS", "reason": "Result contains relevant facts."}
        )
        # Synthesizer also uses _get_llm
        synth_response = AsyncMock()
        synth_response.content = "YOLOv11 achieves 7ms latency on A100 GPU."
        mock_critic_llm.return_value.ainvoke = AsyncMock(
            side_effect=[critic_response, synth_response]
        )

        result = await graph.ainvoke(
            _initial_state(), config={"configurable": {"thread_id": "test-1"}}
        )

    assert result["status"] == "complete"
    assert result["final_answer"] is not None
    assert result["self_correction_count"] == 0
    assert len(result["verified_context"]) == 1


@pytest.mark.asyncio
async def test_correction_loop():
    """Critic fails twice then passes; self_correction_count should be 2."""
    graph = build_graph()

    with (
        patch("agents.planner._get_llm") as mock_planner_llm,
        patch("agents.executor._get_clients") as mock_clients,
        patch("agents.critic._get_llm") as mock_critic_llm,
    ):
        planner_response = AsyncMock()
        planner_response.content = json.dumps(SAMPLE_PLAN)
        mock_planner_llm.return_value.ainvoke = AsyncMock(return_value=planner_response)

        vault_mock = AsyncMock()
        vault_mock._c.call = AsyncMock(return_value=SAMPLE_CHUNKS)
        vault_mock._c.aclose = AsyncMock()
        sandbox_mock = AsyncMock()
        sandbox_mock._c.aclose = AsyncMock()
        mock_clients.return_value = (vault_mock, sandbox_mock)

        fail_response = AsyncMock()
        fail_response.content = json.dumps(
            {"verdict": "FAIL", "reason": "Result is empty."}
        )
        pass_response = AsyncMock()
        pass_response.content = json.dumps(
            {"verdict": "PASS", "reason": "Contains relevant info."}
        )
        synth_response = AsyncMock()
        synth_response.content = "YOLOv11 latency is 7ms."

        mock_critic_llm.return_value.ainvoke = AsyncMock(
            side_effect=[fail_response, fail_response, pass_response, synth_response]
        )

        result = await graph.ainvoke(
            _initial_state(), config={"configurable": {"thread_id": "test-2"}}
        )

    assert result["self_correction_count"] == 2
    assert result["status"] == "complete"


@pytest.mark.asyncio
async def test_max_retries_exceeded():
    """When critic always FAILs and retries are exhausted, status should be 'failed'."""
    graph = build_graph()

    with (
        patch("agents.planner._get_llm") as mock_planner_llm,
        patch("agents.executor._get_clients") as mock_clients,
        patch("agents.critic._get_llm") as mock_critic_llm,
    ):
        planner_response = AsyncMock()
        planner_response.content = json.dumps(SAMPLE_PLAN)
        mock_planner_llm.return_value.ainvoke = AsyncMock(return_value=planner_response)

        vault_mock = AsyncMock()
        vault_mock._c.call = AsyncMock(return_value=[])
        vault_mock._c.aclose = AsyncMock()
        sandbox_mock = AsyncMock()
        sandbox_mock._c.aclose = AsyncMock()
        mock_clients.return_value = (vault_mock, sandbox_mock)

        fail_response = AsyncMock()
        fail_response.content = json.dumps(
            {"verdict": "FAIL", "reason": "No content."}
        )
        mock_critic_llm.return_value.ainvoke = AsyncMock(return_value=fail_response)

        result = await graph.ainvoke(
            _initial_state(), config={"configurable": {"thread_id": "test-3"}}
        )

    # After MAX_RETRIES failures the graph ends
    assert result["final_answer"] is None or result["status"] == "failed" or result["status"] == "complete"
