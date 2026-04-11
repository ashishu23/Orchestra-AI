from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchStep(TypedDict):
    step_id: int
    description: str
    tool: Literal["search_semantic", "search_hybrid", "search_keyword", "execute_python"]
    arguments: dict
    result: str | None
    verified: bool | None
    correction_reason: str | None


class OrchestraState(TypedDict):
    query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: list[ResearchStep]
    current_step_index: int
    retrieved_chunks: list[dict]
    execution_results: list[dict]
    verified_context: list[str]
    hallucination_flags: list[str]
    final_answer: str | None
    self_correction_count: int
    status: Literal["planning", "executing", "verifying", "correcting", "complete", "failed"]
