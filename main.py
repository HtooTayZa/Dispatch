"""
api/main.py
FastAPI application entry-point for EscalationSync.

Exposes:
  POST /v1/execute-agent  – receives n8n webhook payload, runs the LangGraph,
                            and returns the fully-settled state as JSON.
  GET  /healthz           – lightweight liveness probe for load-balancers / k8s
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.graph import get_compiled_graph, run_graph
from config.settings import get_settings
from schemas.ticket import (
    AgentState,
    ExecuteAgentResponse,
    N8nWebhookPayload,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm the compiled graph on startup so the first request is fast."""
    logger.info("EscalationSync starting up | env=%s", settings.environment)
    get_compiled_graph()  # pre-compile
    logger.info("LangGraph pre-compiled and ready")
    yield
    logger.info("EscalationSync shutting down")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered support-ticket routing and resolution engine bridging "
            "n8n workflow orchestration with a LangGraph multi-agent system."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):  # type: ignore[type-arg]
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routes ─────────────────────────────────────────────────────────────────
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:

    # ── Health probe ──────────────────────────────────────────────────────────
    @app.get(
        "/healthz",
        tags=["Operations"],
        summary="Liveness probe",
        response_description="Service is alive",
    )
    async def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "version": settings.app_version,
            "environment": settings.environment,
        }

    # ── Main agent endpoint ────────────────────────────────────────────────────
    @app.post(
        "/v1/execute-agent",
        response_model=ExecuteAgentResponse,
        status_code=status.HTTP_200_OK,
        tags=["Agent"],
        summary="Execute the EscalationSync multi-agent pipeline",
        response_description=(
            "Settled LangGraph state including TicketAnalysis, routing decision, "
            "and generated resolution draft."
        ),
    )
    async def execute_agent(
        payload: N8nWebhookPayload,
        request: Request,
    ) -> ExecuteAgentResponse:
        """
        Accepts an n8n webhook payload, feeds it into the LangGraph execution
        loop, waits for the state machine to settle, and returns the complete
        structured state back to n8n as a clean JSON response.

        **Webhook signature verification** (optional):
        If `N8N_WEBHOOK_SECRET` is configured, every request must include an
        `X-N8N-Signature` header containing the HMAC-SHA256 of the raw request
        body signed with the shared secret.
        """
        # ── Optional signature verification ───────────────────────────────────
        webhook_secret = settings.n8n_webhook_secret.get_secret_value()
        if webhook_secret:
            signature_header = request.headers.get("X-N8N-Signature", "")
            raw_body = await request.body()
            expected_sig = hmac.new(
                webhook_secret.encode(),
                raw_body,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected_sig, signature_header):
                logger.warning(
                    "execute_agent | invalid webhook signature | ticket_id=%s",
                    payload.ticket_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature",
                )

        logger.info(
            "execute_agent | START | ticket_id=%s email=%s",
            payload.ticket_id,
            payload.customer_email,
        )

        # ── Build initial state ────────────────────────────────────────────────
        initial_agent_state = AgentState(payload=payload)
        initial_state_dict = initial_agent_state.model_dump(mode="python")

        # ── Run the graph ──────────────────────────────────────────────────────
        try:
            final_state = await run_graph(initial_state_dict)
        except Exception as exc:
            logger.exception(
                "execute_agent | graph execution failed | ticket_id=%s",
                payload.ticket_id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent graph execution failed: {exc}",
            ) from exc

        # ── Reconstruct typed state ────────────────────────────────────────────
        try:
            final_agent_state = AgentState(**final_state)
        except Exception as exc:
            logger.error(
                "execute_agent | state deserialisation failed | ticket_id=%s | %s",
                payload.ticket_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Final state validation error: {exc}",
            ) from exc

        response = ExecuteAgentResponse.from_state(final_agent_state)

        logger.info(
            "execute_agent | END | ticket_id=%s routed_to=%s error=%s",
            payload.ticket_id,
            response.routed_to,
            response.error,
        )

        return response


# ── Module-level app instance ──────────────────────────────────────────────────
app = create_app()


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.environment == "development",
    )
