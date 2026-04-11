import json
import pathlib
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .state import OrchestraState
from config import settings

_PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "critic.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()

_CORRECTION_STRATEGIES = {
    "search_semantic": ("search_hybrid", lambda q: q),
    "search_hybrid": ("search_keyword", lambda q: q),
    "search_keyword": ("search_keyword", lambda q: q),  # last resort: same tool
}


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


async def critic_node(state: OrchestraState) -> dict[str, Any]:
    """
    Evaluate the current step's result for hallucination / grounding.
    On PASS: appends to verified_context, increments current_step_index.
    On FAIL: appends to hallucination_flags.
    """
    plan = list(state["plan"])
    idx = state["current_step_index"]
    step = plan[idx]
    result_text = step.get("result") or ""

    llm = _get_llm()
    user_msg = (
        f"Original query: {state['query']}\n\n"
        f"Step description: {step['description']}\n"
        f"Tool used: {step['tool']}\n"
        f"Arguments: {json.dumps(step['arguments'])}\n\n"
        f"Result:\n{result_text}"
    )
    response = await llm.ainvoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    )
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    judgment = json.loads(raw)
    verdict = judgment.get("verdict", "FAIL")
    reason = judgment.get("reason", "")

    updated_plan = list(plan)
    step_copy = dict(step)

    if verdict == "PASS":
        step_copy["verified"] = True
        updated_plan[idx] = step_copy
        verified_context = list(state["verified_context"]) + [result_text]
        return {
            "plan": updated_plan,
            "verified_context": verified_context,
            "hallucination_flags": [],
            "current_step_index": idx + 1,
            "status": "executing",
        }
    else:
        step_copy["verified"] = False
        step_copy["correction_reason"] = reason
        updated_plan[idx] = step_copy
        flags = list(state["hallucination_flags"]) + [
            f"Step {step['step_id']}: {reason}"
        ]
        return {
            "plan": updated_plan,
            "hallucination_flags": flags,
            "status": "correcting",
        }


async def correction_node(state: OrchestraState) -> dict[str, Any]:
    """
    Mutate the failing step to try a different retrieval strategy.
    Clears hallucination_flags so the executor can retry.
    """
    plan = list(state["plan"])
    idx = state["current_step_index"]
    step = dict(plan[idx])
    current_tool = step["tool"]
    args = dict(step["arguments"])

    # Escalate retrieval strategy
    if current_tool in _CORRECTION_STRATEGIES:
        new_tool, query_transform = _CORRECTION_STRATEGIES[current_tool]
        if "query" in args:
            args["query"] = query_transform(args["query"])
        step["tool"] = new_tool
        step["arguments"] = args
        step["result"] = None
        step["verified"] = None

    plan[idx] = step
    return {
        "plan": plan,
        "hallucination_flags": [],
        "self_correction_count": state["self_correction_count"] + 1,
        "status": "executing",
    }


async def synthesizer_node(state: OrchestraState) -> dict[str, Any]:
    """
    Synthesize a final answer from all verified context chunks.
    """
    verified = state["verified_context"]
    if not verified:
        return {
            "final_answer": "Insufficient verified information found to answer the query.",
            "status": "complete",
        }

    # Truncate context to stay within reasonable token budget (~4000 tokens ≈ 16000 chars)
    MAX_CONTEXT_CHARS = 16_000
    combined = "\n\n---\n\n".join(verified)
    if len(combined) > MAX_CONTEXT_CHARS:
        combined = combined[:MAX_CONTEXT_CHARS] + "\n\n[...truncated...]"

    llm = _get_llm()
    prompt = (
        f"Answer the following research query based ONLY on the provided context. "
        f"Be precise and cite specific facts from the context.\n\n"
        f"Query: {state['query']}\n\n"
        f"Verified Context:\n{combined}"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {
        "final_answer": response.content,
        "status": "complete",
    }
