from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OrchestraState
from .planner import planner_node
from .executor import executor_node
from .critic import critic_node, correction_node, synthesizer_node
from config import settings

MAX_RETRIES = settings.max_retries


def route_after_critic(
    state: OrchestraState,
) -> Literal["executor", "synthesizer", "correction", "__end__"]:
    """
    Conditional routing after the critic evaluates a step result.

    - If critic FAILed and retries remain  → correction_node → executor
    - If critic FAILed and retries exhausted → END (status=failed)
    - If critic PASSed and more steps remain → executor
    - If critic PASSed and plan complete    → synthesizer
    """
    has_flags = bool(state["hallucination_flags"])
    retries_left = state["self_correction_count"] < MAX_RETRIES
    all_steps_done = state["current_step_index"] >= len(state["plan"])

    if has_flags:
        if retries_left:
            return "correction"
        else:
            return "__end__"

    if all_steps_done:
        return "synthesizer"

    return "executor"


def build_graph(checkpointer=None):
    """
    Assemble the Orchestra AI LangGraph StateGraph.

    Graph topology:
        START → planner → executor → critic ─[PASS, more]──→ executor
                                             ─[PASS, done]──→ synthesizer → END
                                             ─[FAIL, retry]──→ correction → executor
                                             ─[FAIL, max]────→ END
    """
    builder = StateGraph(OrchestraState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("correction", correction_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "executor": "executor",
            "synthesizer": "synthesizer",
            "correction": "correction",
            "__end__": END,
        },
    )
    builder.add_edge("correction", "executor")
    builder.add_edge("synthesizer", END)

    cp = checkpointer or MemorySaver()
    return builder.compile(checkpointer=cp)
