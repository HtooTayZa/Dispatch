"""
agents/graph.py
LangGraph state machine for the EscalationSync pipeline.

Graph topology
──────────────
  [START]
     │
     ▼
  triage_node          (Ollama Qwen – structured output: TicketAnalysis)
     │
     ▼ conditional_router
    ┌─────────────────┐
    │                 │
    ▼                 ▼
escalation_node   standard_resolution_node
(Ollama Qwen)     (Ollama Qwen)
    │                 │
    └────────┬────────┘
             ▼
           [END]
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import ChatOllama
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

# ── LLM factory helpers ────────────────────────────────────────────────────────

# In agents/graph.py
def _make_local_llm(model_name: str = "phi4-mini:latest", temperature: float = 0.0) -> ChatOllama:
    """Return an Ollama local model instance."""
    return ChatOllama(
        model="phi4-mini:latest",  # Hardcoded here to ensure no override
        temperature=temperature,
    )

# ── Node A: Triage ─────────────────────────────────────────────────────────────

async def triage_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Classify the incoming ticket using Qwen with structured output.
    Populates `state["analysis"]` with a TicketAnalysis instance.
    """
    agent_state = AgentState(**state)
    ticket = agent_state.payload

    logger.info("triage_node | ticket_id=%s", ticket.ticket_id)

    # Initialize Qwen and bind it to our Pydantic schema
    triage_llm = _make_local_llm(
        model_name=settings.triage_model, temperature=0.1
    ).with_structured_output(TicketAnalysis)

    user_message = build_triage_user_message(
        raw_issue_text=ticket.raw_issue_text,
        ticket_id=ticket.ticket_id,
    )

    messages = [
        ("system", TRIAGE_SYSTEM_PROMPT),
        ("human", user_message),
    ]

    try:
        # THE REAL AI EXECUTION
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
    """
    raw_analysis = state.get("analysis")

    if isinstance(raw_analysis, dict):
        try:
            analysis: TicketAnalysis | None = TicketAnalysis(**raw_analysis)
        except Exception:
            analysis = None
    else:
        analysis = raw_analysis

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
        "conditional_router | route=%s (human_flag=%s, sentiment=%s, confidence=%.2f)",
        route,
        analysis.needs_human_escalation,
        analysis.sentiment,
        analysis.confidence_score,
    )
    return route


# ── Node B: Escalation Engine ──────────────────────────────────────────────────

async def escalation_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Draft a high-empathy resolution script for human review using Qwen.
    """
    agent_state = AgentState(**state)
    ticket = agent_state.payload
    analysis = agent_state.analysis

    logger.info("escalation_node | ticket_id=%s", ticket.ticket_id)

    escalation_llm = _make_local_llm(model_name=settings.escalation_model, temperature=0.5)

    analysis_summary = (
        f"Category: {analysis.category}, Sentiment: {analysis.sentiment}, "
        f"Confidence: {analysis.confidence_score:.2f}, "
        f"Tags: {', '.join(analysis.suggested_tags)}"
        if analysis else "Classification unavailable – triage failed."
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
        logger.error("escalation_node | FAILED ticket_id=%s | %s", ticket.ticket_id, exc)
        return {**state, "routed_to": "escalation", "error": f"Escalation node error: {exc}"}


# ── Node C: Standard Resolution ────────────────────────────────────────────────

async def standard_resolution_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a concise automated response using Qwen.
    """
    agent_state = AgentState(**state)
    ticket = agent_state.payload
    analysis = agent_state.analysis

    logger.info("standard_resolution_node | ticket_id=%s", ticket.ticket_id)

    resolution_llm = _make_local_llm(model_name=settings.resolution_model, temperature=0.3)

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
        logger.info("standard_resolution_node | draft generated | ticket_id=%s", ticket.ticket_id)
        return {**state, "resolution_draft": draft, "routed_to": "standard"}
        
    except Exception as exc:
        logger.error("standard_resolution_node | FAILED ticket_id=%s | %s", ticket.ticket_id, exc)
        return {**state, "routed_to": "standard", "error": f"Resolution node error: {exc}"}


# ── Graph compilation ──────────────────────────────────────────────────────────

def _build_graph() -> Any:
    builder: StateGraph = StateGraph(dict)

    builder.add_node("triage_node", triage_node)
    builder.add_node("escalation_node", escalation_node)
    builder.add_node("standard_resolution_node", standard_resolution_node)

    builder.add_edge(START, "triage_node")
    builder.add_conditional_edges(
        "triage_node",
        conditional_router,
        {
            "escalation_node": "escalation_node",
            "standard_resolution_node": "standard_resolution_node",
        },
    )

    builder.add_edge("escalation_node", END)
    builder.add_edge("standard_resolution_node", END)

    compiled = builder.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


_compiled_graph: Any = None

def get_compiled_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph

async def run_graph(initial_state: dict[str, Any]) -> dict[str, Any]:
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
        except ImportError:
            pass

    final_state: dict[str, Any] = await graph.ainvoke(initial_state, config=run_config)
    return final_state