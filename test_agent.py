"""
tests/test_agent.py
Comprehensive pytest suite for EscalationSync.

Test strategy
─────────────
• All LLM invocations are patched at the node level so tests run offline,
  deterministically, and without API quota consumption.
• Schema validation tests exercise Pydantic directly.
• Routing tests inject pre-built TicketAnalysis fixtures and verify the
  conditional_router selects the correct branch.
• Integration tests drive the full HTTP layer via httpx.AsyncClient with
  the FastAPI app mounted in-process.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from agents.graph import conditional_router
from api.main import app
from schemas.ticket import (
    AgentState,
    ExecuteAgentResponse,
    N8nWebhookPayload,
    TicketAnalysis,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def standard_payload_dict() -> dict[str, Any]:
    """A routine billing inquiry – expected to route to standard_resolution."""
    return {
        "ticket_id": "TKT-0001",
        "customer_email": "alice@example.com",
        "raw_issue_text": (
            "Hi, I have a question about my invoice for March. "
            "I was charged twice for the Pro plan. Could you help?"
        ),
        "source_channel": "email",
    }


@pytest.fixture()
def escalation_payload_dict() -> dict[str, Any]:
    """A critical data-loss ticket – expected to route to escalation."""
    return {
        "ticket_id": "TKT-9999",
        "customer_email": "bob@enterprise.org",
        "raw_issue_text": (
            "URGENT: All of our production data has disappeared from your platform! "
            "We have lost 6 months of records and are facing a legal deadline tomorrow. "
            "This is completely unacceptable and I am contacting our lawyers."
        ),
        "priority_hint": "urgent",
    }


@pytest.fixture()
def standard_analysis() -> TicketAnalysis:
    return TicketAnalysis(
        category="Billing",
        sentiment="Neutral",
        confidence_score=0.92,
        needs_human_escalation=False,
        suggested_tags=["invoice_dispute", "double_charge"],
    )


@pytest.fixture()
def escalation_analysis() -> TicketAnalysis:
    return TicketAnalysis(
        category="Technical",
        sentiment="Critical",
        confidence_score=0.95,
        needs_human_escalation=True,
        suggested_tags=["data_loss", "legal_threat", "urgent_downtime"],
    )


@pytest.fixture()
def low_confidence_analysis() -> TicketAnalysis:
    return TicketAnalysis(
        category="Account",
        sentiment="Frustrated",
        confidence_score=0.65,  # Below 0.8 threshold
        needs_human_escalation=False,
        suggested_tags=["ambiguous"],
    )


@pytest_asyncio.fixture()
async def async_client() -> AsyncClient:
    """Async HTTP client bound to the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Schema Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestN8nWebhookPayload:
    def test_valid_payload_parses_correctly(
        self, standard_payload_dict: dict[str, Any]
    ) -> None:
        payload = N8nWebhookPayload(**standard_payload_dict)
        assert payload.ticket_id == "TKT-0001"
        assert str(payload.customer_email) == "alice@example.com"
        assert "twice" in payload.raw_issue_text

    def test_integer_ticket_id_coerced_to_string(self) -> None:
        payload = N8nWebhookPayload(
            ticket_id=12345,  # type: ignore[arg-type]
            customer_email="x@y.com",
            raw_issue_text="test issue text here",
        )
        assert payload.ticket_id == "12345"
        assert isinstance(payload.ticket_id, str)

    def test_invalid_email_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            N8nWebhookPayload(
                ticket_id="TKT-BAD",
                customer_email="not-an-email",
                raw_issue_text="some issue text",
            )
        assert "customer_email" in str(exc_info.value)

    def test_empty_issue_text_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            N8nWebhookPayload(
                ticket_id="TKT-EMPTY",
                customer_email="a@b.com",
                raw_issue_text="",  # min_length=5 violation
            )

    def test_too_short_issue_text_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            N8nWebhookPayload(
                ticket_id="TKT-SHORT",
                customer_email="a@b.com",
                raw_issue_text="Hi",  # only 2 chars, min 5
            )

    def test_whitespace_stripped_from_issue_text(self) -> None:
        payload = N8nWebhookPayload(
            ticket_id="TKT-WS",
            customer_email="a@b.com",
            raw_issue_text="   my billing issue   ",
        )
        assert not payload.raw_issue_text.startswith(" ")
        assert not payload.raw_issue_text.endswith(" ")

    def test_optional_fields_default_to_none(self) -> None:
        payload = N8nWebhookPayload(
            ticket_id="TKT-MIN",
            customer_email="a@b.com",
            raw_issue_text="minimal valid issue text",
        )
        assert payload.source_channel is None
        assert payload.priority_hint is None

    def test_invalid_priority_hint_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            N8nWebhookPayload(
                ticket_id="TKT-PRI",
                customer_email="a@b.com",
                raw_issue_text="some issue text",
                priority_hint="critical",  # not in Literal
            )


class TestTicketAnalysis:
    def test_valid_analysis_all_fields(self) -> None:
        analysis = TicketAnalysis(
            category="Technical",
            sentiment="Frustrated",
            confidence_score=0.87,
            needs_human_escalation=False,
            suggested_tags=["API Error", "Rate Limit"],
        )
        assert analysis.category == "Technical"
        # Tags should be lowercased and spaces replaced
        assert "api_error" in analysis.suggested_tags
        assert "rate_limit" in analysis.suggested_tags

    def test_spam_cannot_require_human_escalation(self) -> None:
        """Model validator must auto-correct spam + needs_human_escalation."""
        analysis = TicketAnalysis(
            category="Spam",
            sentiment="Neutral",
            confidence_score=0.99,
            needs_human_escalation=True,  # should be auto-corrected
            suggested_tags=[],
        )
        assert analysis.needs_human_escalation is False

    def test_confidence_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TicketAnalysis(
                category="Billing",
                sentiment="Neutral",
                confidence_score=1.5,  # > 1.0
                needs_human_escalation=False,
                suggested_tags=[],
            )

    def test_invalid_category_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            TicketAnalysis(
                category="Marketing",  # type: ignore[arg-type]
                sentiment="Neutral",
                confidence_score=0.8,
                needs_human_escalation=False,
                suggested_tags=[],
            )

    def test_invalid_sentiment_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            TicketAnalysis(
                category="Billing",
                sentiment="Angry",  # type: ignore[arg-type]
                confidence_score=0.8,
                needs_human_escalation=False,
                suggested_tags=[],
            )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Conditional Router Unit Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConditionalRouter:
    def _build_state(self, analysis: TicketAnalysis | None) -> dict[str, Any]:
        payload = N8nWebhookPayload(
            ticket_id="TKT-ROUTE",
            customer_email="test@test.com",
            raw_issue_text="routing test issue text",
        )
        state = AgentState(payload=payload, analysis=analysis)
        return state.model_dump(mode="python")

    def test_standard_route_on_clean_analysis(
        self, standard_analysis: TicketAnalysis
    ) -> None:
        state = self._build_state(standard_analysis)
        assert conditional_router(state) == "standard_resolution_node"

    def test_escalation_route_on_critical_sentiment(
        self, escalation_analysis: TicketAnalysis
    ) -> None:
        state = self._build_state(escalation_analysis)
        assert conditional_router(state) == "escalation_node"

    def test_escalation_route_on_low_confidence(
        self, low_confidence_analysis: TicketAnalysis
    ) -> None:
        state = self._build_state(low_confidence_analysis)
        assert conditional_router(state) == "escalation_node"

    def test_escalation_route_when_human_flag_true(self) -> None:
        analysis = TicketAnalysis(
            category="Billing",
            sentiment="Frustrated",
            confidence_score=0.91,  # above threshold
            needs_human_escalation=True,  # but flag is set
            suggested_tags=[],
        )
        state = self._build_state(analysis)
        assert conditional_router(state) == "escalation_node"

    def test_escalation_route_when_analysis_is_none(self) -> None:
        state = self._build_state(None)
        assert conditional_router(state) == "escalation_node"

    def test_standard_route_at_exact_threshold(self) -> None:
        """confidence_score == threshold should route to standard (not <)."""
        analysis = TicketAnalysis(
            category="Account",
            sentiment="Neutral",
            confidence_score=0.8,  # exactly at default threshold
            needs_human_escalation=False,
            suggested_tags=[],
        )
        state = self._build_state(analysis)
        assert conditional_router(state) == "standard_resolution_node"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Integration Tests  (full HTTP stack, LLM nodes mocked)
# ══════════════════════════════════════════════════════════════════════════════

MOCK_RESOLUTION_DRAFT = "Thank you for contacting us. We have reviewed your issue..."


def _make_mock_triage_node(analysis: TicketAnalysis):
    """Return an async mock that injects the given analysis into state."""

    async def _mock(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "analysis": analysis}

    return _mock


def _make_mock_resolution_node(route: str):
    """Return an async mock that sets resolution_draft and routed_to."""

    async def _mock(state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state,
            "resolution_draft": MOCK_RESOLUTION_DRAFT,
            "routed_to": route,
        }

    return _mock


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client: AsyncClient) -> None:
        resp = await async_client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body


def _make_mock_run_graph(analysis: TicketAnalysis, route: str):
    """
    Patch agents.graph.run_graph so tests bypass the compiled LangGraph
    and the LLM API entirely.  The compiled graph captures node references
    at build time, so patching individual nodes has no effect after compile.
    Patching run_graph (the coroutine that invokes ainvoke) is the correct
    injection point for integration tests.
    """

    async def _mock(initial_state: dict[str, Any]) -> dict[str, Any]:
        return {
            **initial_state,
            "analysis": analysis.model_dump(),
            "resolution_draft": MOCK_RESOLUTION_DRAFT,
            "routed_to": route,
            "error": None,
        }

    return _mock


class TestExecuteAgentStandardRoute:
    """Tests where triage produces a high-confidence, non-critical analysis."""

    @pytest.mark.asyncio
    async def test_standard_route_returns_200(
        self,
        async_client: AsyncClient,
        standard_payload_dict: dict[str, Any],
        standard_analysis: TicketAnalysis,
    ) -> None:
        with patch(
            "api.main.run_graph",
            side_effect=_make_mock_run_graph(standard_analysis, "standard"),
        ):
            resp = await async_client.post(
                "/v1/execute-agent", json=standard_payload_dict
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_standard_route_response_shape(
        self,
        async_client: AsyncClient,
        standard_payload_dict: dict[str, Any],
        standard_analysis: TicketAnalysis,
    ) -> None:
        with patch(
            "api.main.run_graph",
            side_effect=_make_mock_run_graph(standard_analysis, "standard"),
        ):
            resp = await async_client.post(
                "/v1/execute-agent", json=standard_payload_dict
            )

        body = resp.json()
        assert body["ticket_id"] == standard_payload_dict["ticket_id"]
        assert body["routed_to"] == "standard"
        assert body["resolution_draft"] == MOCK_RESOLUTION_DRAFT
        assert body["analysis"]["category"] == "Billing"
        assert body["analysis"]["sentiment"] == "Neutral"
        assert body["error"] is None
        # Validate run_id is a valid UUID
        UUID(body["run_id"])

    @pytest.mark.asyncio
    async def test_standard_route_analysis_fields_present(
        self,
        async_client: AsyncClient,
        standard_payload_dict: dict[str, Any],
        standard_analysis: TicketAnalysis,
    ) -> None:
        with patch(
            "api.main.run_graph",
            side_effect=_make_mock_run_graph(standard_analysis, "standard"),
        ):
            resp = await async_client.post(
                "/v1/execute-agent", json=standard_payload_dict
            )

        analysis = resp.json()["analysis"]
        assert "confidence_score" in analysis
        assert "needs_human_escalation" in analysis
        assert isinstance(analysis["suggested_tags"], list)
        assert analysis["needs_human_escalation"] is False


class TestExecuteAgentEscalationRoute:
    """Tests where triage produces a critical analysis requiring human review."""

    @pytest.mark.asyncio
    async def test_escalation_route_returns_200(
        self,
        async_client: AsyncClient,
        escalation_payload_dict: dict[str, Any],
        escalation_analysis: TicketAnalysis,
    ) -> None:
        with patch(
            "api.main.run_graph",
            side_effect=_make_mock_run_graph(escalation_analysis, "escalation"),
        ):
            resp = await async_client.post(
                "/v1/execute-agent", json=escalation_payload_dict
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_escalation_route_response_shape(
        self,
        async_client: AsyncClient,
        escalation_payload_dict: dict[str, Any],
        escalation_analysis: TicketAnalysis,
    ) -> None:
        with patch(
            "api.main.run_graph",
            side_effect=_make_mock_run_graph(escalation_analysis, "escalation"),
        ):
            resp = await async_client.post(
                "/v1/execute-agent", json=escalation_payload_dict
            )

        body = resp.json()
        assert body["ticket_id"] == escalation_payload_dict["ticket_id"]
        assert body["routed_to"] == "escalation"
        assert body["resolution_draft"] == MOCK_RESOLUTION_DRAFT
        assert body["analysis"]["sentiment"] == "Critical"
        assert body["analysis"]["needs_human_escalation"] is True
        assert body["error"] is None

    @pytest.mark.asyncio
    async def test_low_confidence_routes_to_escalation(
        self,
        async_client: AsyncClient,
        standard_payload_dict: dict[str, Any],
        low_confidence_analysis: TicketAnalysis,
    ) -> None:
        """Even a non-critical ticket should escalate if confidence is too low."""
        with patch(
            "api.main.run_graph",
            side_effect=_make_mock_run_graph(low_confidence_analysis, "escalation"),
        ):
            resp = await async_client.post(
                "/v1/execute-agent", json=standard_payload_dict
            )

        assert resp.status_code == 200
        assert resp.json()["routed_to"] == "escalation"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Input Validation at the HTTP Layer
# ══════════════════════════════════════════════════════════════════════════════

class TestInputValidationHTTP:
    @pytest.mark.asyncio
    async def test_missing_ticket_id_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        resp = await async_client.post(
            "/v1/execute-agent",
            json={
                "customer_email": "a@b.com",
                "raw_issue_text": "some issue text here",
                # ticket_id missing
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_email_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        resp = await async_client.post(
            "/v1/execute-agent",
            json={
                "ticket_id": "TKT-BAD-EMAIL",
                "customer_email": "not_valid",
                "raw_issue_text": "some issue text here",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        resp = await async_client.post(
            "/v1/execute-agent",
            json={},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_issue_text_too_short_returns_422(
        self, async_client: AsyncClient
    ) -> None:
        resp = await async_client.post(
            "/v1/execute-agent",
            json={
                "ticket_id": "TKT-SHORT",
                "customer_email": "a@b.com",
                "raw_issue_text": "Hi",  # < 5 chars
            },
        )
        assert resp.status_code == 422
