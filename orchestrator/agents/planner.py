import json
import pathlib
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .state import OrchestraState, ResearchStep
from config import settings

_PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "planner.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


def _get_llm():
    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


async def planner_node(state: OrchestraState) -> dict[str, Any]:
    """
    Generate a step-by-step research plan from the user query.
    Returns the plan list and resets execution state.
    """
    llm = _get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Research query: {state['query']}"),
    ]
    response = await llm.ainvoke(messages)
    raw = response.content.strip()

    # Strip markdown fences if the model wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    plan_data: list[dict] = json.loads(raw)
    plan: list[ResearchStep] = []
    for step in plan_data:
        plan.append(
            ResearchStep(
                step_id=step["step_id"],
                description=step["description"],
                tool=step["tool"],
                arguments=step["arguments"],
                result=None,
                verified=None,
                correction_reason=None,
            )
        )

    return {
        "plan": plan,
        "current_step_index": 0,
        "retrieved_chunks": [],
        "execution_results": [],
        "verified_context": [],
        "hallucination_flags": [],
        "self_correction_count": 0,
        "status": "executing",
        "messages": messages + [response],
    }
