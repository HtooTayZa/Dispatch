"""
schemas/ticket.py
All Pydantic v2 models used across the Dispatch pipeline.

• N8nWebhookPayload  – the inbound request body from n8n
• TicketAnalysis     – structured output produced by the Triage LLM node
• AgentState         – the full mutable state carried through the LangGraph
• ExecuteAgentResponse – the HTTP response returned to n8n after graph settlement
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# ── Inbound payload ────────────────────────────────────────────────────────────

class N8nWebhookPayload(BaseModel):
    """
    Validated input received from an n8n HTTP-Request node webhook.
    All fields are required; n8n must be configured to forward them.
    """

    ticket_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier assigned by the originating ticketing system",
        examples=["TKT-20240001", "JIRA-4821"],
    )
    customer_email: EmailStr = Field(
        ...,
        description="Validated email address of the submitting customer",
    )
    raw_issue_text: str = Field(
        ...,
        min_length=5,
        max_length=8_000,
        description="Free-form support ticket text verbatim from the customer",
    )
    # Optional metadata that n8n may attach
    source_channel: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Originating channel, e.g. 'email', 'chat', 'portal'",
    )
    priority_hint: Optional[Literal["low", "normal", "high", "urgent"]] = Field(
        default=None,
        description="Optional priority signal forwarded by n8n from upstream metadata",
    )

    @field_validator("ticket_id", mode="before")
    @classmethod
    def _strip_ticket_id(cls, v: Any) -> str:
        if isinstance(v, (int, float)):
            return str(v)
        return str(v).strip()

    @field_validator("raw_issue_text", mode="before")
    @classmethod
    def _strip_issue_text(cls, v: Any) -> str:
        return str(v).strip()


# ── LLM structured output ──────────────────────────────────────────────────────

CategoryLiteral = Literal["Billing", "Technical", "Account", "Spam"]
SentimentLiteral = Literal["Positive", "Neutral", "Frustrated", "Critical"]


class TicketAnalysis(BaseModel):
    """
    Strictly-typed structured output produced by the Triage node via
    `.with_structured_output(TicketAnalysis)`.  Every field is required;
    the LLM must populate them all.
    """

    category: CategoryLiteral = Field(
        ...,
        description=(
            "Primary classification of the support ticket. "
            "Choose exactly one: Billing, Technical, Account, or Spam."
        ),
    )
    sentiment: SentimentLiteral = Field(
        ...,
        description=(
            "Overall emotional tone expressed in the ticket text. "
            "Choose exactly one: Positive, Neutral, Frustrated, or Critical."
        ),
    )
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description=(
            "Model self-assessed confidence in the above classification, "
            "expressed as a probability in [0.0, 1.0]."
        ),
    )
    needs_human_escalation: bool = Field(
        ...,
        description=(
            "True if the ticket requires human review due to legal risk, "
            "account compromise, complex billing dispute, or safety concern."
        ),
    )
    suggested_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Short snake_case tags useful for downstream routing or CRM tagging, "
            "e.g. ['password_reset', 'refund_request']."
        ),
        max_length=10,
    )

    @field_validator("suggested_tags", mode="before")
    @classmethod
    def _lowercase_tags(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(t).lower().replace(" ", "_") for t in v]
        return []

    @model_validator(mode="after")
    def _validate_spam_not_escalated(self) -> "TicketAnalysis":
        """Spam tickets should never require human escalation."""
        if self.category == "Spam" and self.needs_human_escalation:
            # Auto-correct rather than hard-fail to avoid upstream disruption
            object.__setattr__(self, "needs_human_escalation", False)
        return self


# ── LangGraph state ────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    Mutable state dict carried through every node of the LangGraph.
    Uses BaseModel so it can be serialised/deserialised automatically.
    """

    # Injected at graph entry
    run_id: UUID = Field(default_factory=uuid4)
    payload: N8nWebhookPayload

    # Populated by the Triage node
    analysis: Optional[TicketAnalysis] = None

    # Populated by whichever resolution node runs
    resolution_draft: Optional[str] = None

    # Routing metadata (set by conditional edge logic)
    routed_to: Optional[Literal["escalation", "standard"]] = None

    # Error capture – if a node sets this, the API layer returns HTTP 422
    error: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


# ── API response ───────────────────────────────────────────────────────────────

class ExecuteAgentResponse(BaseModel):
    """
    Final JSON body returned to n8n after the LangGraph has settled.
    n8n can parse this directly with its JSON node.
    """

    run_id: UUID
    ticket_id: str
    customer_email: str
    raw_issue_text: Optional[str] = None
    analysis: Optional[TicketAnalysis]
    routed_to: Optional[Literal["escalation", "standard"]]
    resolution_draft: Optional[str]
    error: Optional[str] = None
    

    @classmethod
    def from_state(cls, state: AgentState) -> "ExecuteAgentResponse":
        return cls(
            run_id=state.run_id,
            ticket_id=state.payload.ticket_id,
            customer_email=str(state.payload.customer_email),
            analysis=state.analysis,
            routed_to=state.routed_to,
            resolution_draft=state.resolution_draft,
            error=state.error,
        )
