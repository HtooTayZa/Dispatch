"""
agents/graph.py
LangGraph state machine for the EscalationSync pipeline.

Graph topology
──────────────
  [START]
     │
     ▼
  triage_node          (Gemini 1.5 Flash – structured output: TicketAnalysis)
     │
     ▼ conditional_router
    ┌─────────────────┐
    │                 │
    ▼                 ▼
escalation_node   standard_resolution_node
(Claude 3.5)      (Gemini 1.5 Flash)
    │                 │
    └────────┬────────┘
             ▼
           [END]
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from config.settings import get_settings
from prompts.templates import (
    ESCALATION_SYSTEM_PROMPT,
    STANDARD_RESOLUTION_SYSTEM_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
    build_escalation_user_message,
    build_resolution_user_message,
    build_triage_user_message,
)
from schemas.ticket import AgentState, TicketAnalysis

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Retry stop-codes ───────────────────────────────────────────────────────────
# LangChain's .with_retry() will retry on these HTTP status codes.
_RETRY_STOP_CODES: list[int] = [429, 500, 502, 503, 504]


# ── LLM factory helpers ────────────────────────────────────────────────────────

def _make_gemini(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Return a Gemini model instance with retry configuration."""
    base = ChatGoogleGenerativeAI(
        model=settings.triage_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=temperature,
        max_retries=settings.llm_max_retries,
    )
    return base


def _make_claude(temperature: float = 0.4) -> ChatAnthropic:
    """Return a Claude model instance with retry configuration."""
    base = ChatAnthropic(
        model=settings.escalation_model,
        anthropic_api_key=settings.anthropic_api_key.get_secret_value(),
        temperature=temperature,
        max_retries=settings.llm_max_retries,
    )
    return base


# ── Node A: Triage ─────────────────────────────────────────────────────────────

async def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the incoming ticket using Gemini with structured output.
    Populates `state["analysis"]` with a TicketAnalysis instance.
    On failure, sets `state["error"]` and leaves analysis as None.
    """
    agent_state = AgentState(**state)
    ticket = agent_state.payload

    logger.info("triage_node | ticket_id=%s", ticket.ticket_id)

    triage_llm = _make_gemini(temperature=0.1).with_structured_output(TicketAnalysis)

    user_message = build_triage_user_message(
        raw_issue_text=ticket.raw_issue_text,
        ticket_id=ticket.ticket_id,
    )

    messages = [
        ("system", TRIAGE_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    try:
        analysis: TicketAnalysis = await triage_llm.ainvoke(messages)  # type: ignore[assignment]
        logger.info(
            "triage_node | category=%s sentiment=%s confidence=%.2f escalate=%s",
            analysis.category,
            analysis.sentiment,
            analysis.confidence_score,
            analysis.needs_human_escalation,
        )
        return {**state, "analysis": analysis}
    except Exception as exc:
        logger.error("triage_node | FAILED ticket_id=%s | %s", ticket.ticket_id, exc)
        return {**state, "error": f"Triage node error: {exc}"}


# ── Conditional Router ─────────────────────────────────────────────────────────

def conditional_router(state: dict[str, Any]) -> str:
    """
    Evaluate the TicketAnalysis and return the name of the next node.

    Escalation triggers (ANY of):
    • needs_human_escalation is True
    • sentiment is "Critical"
    • confidence_score < threshold (default 0.8)
    • analysis is None (triage failed)
    """
    raw_analysis = state.get("analysis")

    # State may be serialised to a plain dict by LangGraph between nodes.
    # Re-hydrate to a TicketAnalysis model if needed.
    if isinstance(raw_analysis, dict):
        try:
            analysis: TicketAnalysis | None = TicketAnalysis(**raw_analysis)
        except Exception:
            analysis = None
    else:
        analysis = raw_analysis  # already a TicketAnalysis or None

    if analysis is None:
        logger.warning("conditional_router | analysis is None – routing to escalation")
        return "escalation_node"

    threshold = settings.escalation_confidence_threshold

    should_escalate = (
        analysis.needs_human_escalation
        or analysis.sentiment == "Critical"
        or analysis.confidence_score < threshold
    )

    route = "escalation_node" if should_escalate else "standard_resolution_node"
    logger.info(
        "conditional_router | route=%s "
        "(human_flag=%s, sentiment=%s, confidence=%.2f, threshold=%.2f)",
        route,
        analysis.needs_human_escalation,
        analysis.sentiment,
        analysis.confidence_score,
        threshold,
    )
    return route


# ── Node B: Escalation Engine ──────────────────────────────────────────────────

async def escalation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Draft a high-empathy resolution script for human review using Claude.
    Populates `state["resolution_draft"]` and sets `state["routed_to"]`.
    """
    agent_state = AgentState(**state)
    ticket = agent_state.payload
    analysis = agent_state.analysis

    logger.info("escalation_node | ticket_id=%s", ticket.ticket_id)

    escalation_llm = _make_claude(temperature=0.5)

    analysis_summary = (
        f"Category: {analysis.category}, Sentiment: {analysis.sentiment}, "
        f"Confidence: {analysis.confidence_score:.2f}, "
        f"Tags: {', '.join(analysis.suggested_tags)}"
        if analysis
        else "Classification unavailable – triage failed."
    )

    user_message = build_escalation_user_message(
        raw_issue_text=ticket.raw_issue_text,
        ticket_id=ticket.ticket_id,
        customer_email=str(ticket.customer_email),
        analysis_summary=analysis_summary,
    )

    messages = [
        ("system", ESCALATION_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    try:
        response = await escalation_llm.ainvoke(messages)
        draft = response.content if hasattr(response, "content") else str(response)
        logger.info("escalation_node | draft generated | ticket_id=%s", ticket.ticket_id)
        return {**state, "resolution_draft": draft, "routed_to": "escalation"}
    except Exception as exc:
        logger.error(
            "escalation_node | FAILED ticket_id=%s | %s", ticket.ticket_id, exc
        )
        return {
            **state,
            "routed_to": "escalation",
            "error": f"Escalation node error: {exc}",
        }


# ── Node C: Standard Resolution ────────────────────────────────────────────────

async def standard_resolution_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a concise automated response using Gemini.
    Populates `state["resolution_draft"]` and sets `state["routed_to"]`.
    """
    agent_state = AgentState(**state)
    ticket = agent_state.payload
    analysis = agent_state.analysis

    logger.info("standard_resolution_node | ticket_id=%s", ticket.ticket_id)

    resolution_llm = _make_gemini(temperature=0.3)

    user_message = build_resolution_user_message(
        raw_issue_text=ticket.raw_issue_text,
        ticket_id=ticket.ticket_id,
        category=analysis.category if analysis else "Unknown",
        sentiment=analysis.sentiment if analysis else "Unknown",
    )

    messages = [
        ("system", STANDARD_RESOLUTION_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    try:
        response = await resolution_llm.ainvoke(messages)
        draft = response.content if hasattr(response, "content") else str(response)
        logger.info(
            "standard_resolution_node | draft generated | ticket_id=%s", ticket.ticket_id
        )
        return {**state, "resolution_draft": draft, "routed_to": "standard"}
    except Exception as exc:
        logger.error(
            "standard_resolution_node | FAILED ticket_id=%s | %s",
            ticket.ticket_id,
            exc,
        )
        return {
            **state,
            "routed_to": "standard",
            "error": f"Resolution node error: {exc}",
        }


# ── Graph compilation ──────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Assemble and compile the LangGraph state machine."""
    builder: StateGraph = StateGraph(dict)

    # Register nodes
    builder.add_node("triage_node", triage_node)
    builder.add_node("escalation_node", escalation_node)
    builder.add_node("standard_resolution_node", standard_resolution_node)

    # Entry edge
    builder.add_edge(START, "triage_node")

    # Conditional routing after triage
    builder.add_conditional_edges(
        "triage_node",
        conditional_router,
        {
            "escalation_node": "escalation_node",
            "standard_resolution_node": "standard_resolution_node",
        },
    )

    # Both resolution branches terminate the graph
    builder.add_edge("escalation_node", END)
    builder.add_edge("standard_resolution_node", END)

    compiled = builder.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


# Module-level compiled graph (lazy singleton via factory function)
_compiled_graph: Any = None


def get_compiled_graph() -> Any:
    """Return the module-level compiled graph, initialising it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


async def run_graph(initial_state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the compiled graph with the provided initial state.

    Integrates Langfuse callbacks when observability is configured.
    Returns the final settled state dict.
    """
    graph = get_compiled_graph()
    run_config: dict[str, Any] = {"recursion_limit": 10}

    if settings.observability_enabled:
        try:
            from langfuse.langchain import CallbackHandler as LangfuseCallback  # type: ignore

            langfuse_handler = LangfuseCallback(
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_host,
            )
            run_config["callbacks"] = [langfuse_handler]
            logger.info("run_graph | Langfuse tracing enabled")
        except ImportError:
            logger.warning(
                "run_graph | langfuse package not installed – tracing disabled"
            )

    final_state: dict[str, Any] = await graph.ainvoke(initial_state, config=run_config)
    return final_state
