# Dispatch

An autonomous, privacy-first support triage system that sits between your inbound channels (web forms, email) and your team's workflow tools (Slack, Gmail). It uses a local-first agentic pipeline — built on LangGraph — to classify incoming tickets, decide whether they warrant escalation, and draft context-aware responses, all without sending data to a third-party LLM unless you explicitly choose to.

---

## How It Works

A ticket submitted through [Tally.so](https://tally.so) triggers a webhook in n8n Cloud, which forwards the payload to a local FastAPI server exposed via a `cloudflared` tunnel. From there, a LangGraph agent graph handles everything:

1. **Triage** — an LLM node reads the ticket and classifies it by category, sentiment, and urgency.
2. **Routing** — a conditional router compares the escalation confidence score against a configurable threshold. Tickets above the threshold (angry customers, complex issues, high-urgency flags) are sent down the escalation path; everything else takes the standard path.
3. **Drafting** — the appropriate node generates a response using a tone-specific system prompt: empathetic and thorough for escalations, concise and direct for standard replies.
4. **Delivery** — the drafted response is posted to Slack for human review, or sent directly via Gmail.

The entire pipeline is observable through [Langfuse](https://langfuse.com), which traces every LLM call so you can audit decisions and tune prompts over time.

## Demo

[[Watch the Dispatch Demo](https://img.youtube.com/vi/naLJ_bwnQgI/maxresdefault.jpg)](https://youtu.be/naLJ_bwnQgI))

*Click the image above to watch a walkthrough of the ticket triage and escalation pipeline.*

---

## Project Structure

```
Dispatch/
├── agents/          # LangGraph graph definition, nodes, and conditional router
├── api/             # FastAPI application and route handlers
├── config/          # Settings (loaded from .env via pydantic-settings)
├── prompts/         # System prompt templates for each agent node
├── schemas/         # Pydantic request/response models
├── tests/           # Test suite (pytest + pytest-asyncio)
├── .env.example     # All supported environment variables with documentation
└── requirements.txt
```

---

## Stack

| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Agent framework | LangGraph + LangChain |
| Default LLM | Ollama (`phi4-mini:latest`) — runs locally |
| Cloud LLM options | Google Gemini 1.5 Flash, Claude 3.5 Sonnet |
| Observability | Langfuse |
| Workflow orchestration | n8n Cloud |
| Intake form | Tally.so |
| Local tunnel | cloudflared |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally (for the default local LLM mode)
- An [n8n Cloud](https://n8n.io) account with a workflow configured to POST to your `/v1/execute-agent` endpoint
- A [Tally.so](https://tally.so) form that triggers the n8n webhook on submission
- [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) for exposing your local server

### Installation

```bash
git clone https://github.com/HtooTayZa/EscSync.git
cd EscSync
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

The key variables are:

```env
# Application
APP_NAME=EscalationSync
ENVIRONMENT=development
LOG_LEVEL=INFO

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# LLM model selection (Ollama model names by default)
TRIAGE_MODEL=phi4-mini:latest
ESCALATION_MODEL=phi4-mini:latest
RESOLUTION_MODEL=phi4-mini:latest

# Routing sensitivity — raise this to escalate less aggressively, lower it to escalate more
ESCALATION_CONFIDENCE_THRESHOLD=0.8

# Retry behaviour
LLM_MAX_RETRIES=3
LLM_RETRY_WAIT_SECONDS=2.0

# Langfuse observability (leave blank to disable tracing)
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# n8n webhook signature verification (leave blank to disable in development)
N8N_WEBHOOK_SECRET=

# Set to true to skip LLM calls entirely and return a mock response (useful for testing n8n plumbing)
USE_MOCK_BYPASS=false
```

### Running the Server

```bash
# Pull the default local model if you haven't already
ollama pull phi4-mini:latest

# Start the API
python -m api.main
```

In a separate terminal, create a public tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated `trycloudflare.com` URL into the HTTP Request node in your n8n workflow, pointing it at `[YOUR_URL]/v1/execute-agent`.

---

## Choosing an LLM Provider

Dispatch uses LangChain's provider abstraction, so swapping the underlying model requires only two changes: updating the factory functions in `agents/graph.py` and setting the relevant environment variables.

### Option A — Local (default)

Uses Ollama. No data leaves your machine and there are no API costs.

Best for: development, privacy-sensitive environments, cost control.

The default model is `phi4-mini:latest`. You can substitute any model available in your local Ollama instance by updating the `*_MODEL` variables in `.env`.

### Option B — Cloud

Uses Google Gemini or Anthropic Claude for higher reasoning capacity. Useful when tickets contain complex context or when draft quality is the priority.

1. Add `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` to your `.env`.
2. In `agents/graph.py`, replace `ChatOllama` with `ChatGoogleGenerativeAI` or `ChatAnthropic` in the model factory functions.
3. Update the `*_MODEL` variables to the appropriate model name (e.g. `gemini-1.5-flash`, `claude-3-5-sonnet-20241022`).

---

## Customisation

### Adjusting Routing Sensitivity

The escalation decision is driven by the confidence score returned from the triage node. The threshold is controlled by `ESCALATION_CONFIDENCE_THRESHOLD` in your `.env`. A value of `0.8` means a ticket must score at least 80% confidence for escalation before it is routed to the escalation path. Lower this number to escalate more aggressively; raise it to keep more tickets on the standard path.

For structural changes to routing logic, edit `conditional_router` in `agents/graph.py`.

### Changing Response Tone

Each path has its own system prompt. Edit `ESCALATION_SYSTEM_PROMPT` and `STANDARD_RESOLUTION_SYSTEM_PROMPT` in `prompts/templates.py` to control how the agent sounds — response length, formality, sign-off style, etc.

### Enabling Mock Mode

Set `USE_MOCK_BYPASS=true` to skip all LLM calls and return a fixed mock payload. This is useful for testing the end-to-end n8n workflow without consuming API quota or requiring Ollama to be running.

---

## Security

**Keep `.env` out of version control.** It is already listed in `.gitignore`, but double-check before pushing.

In production, set `N8N_WEBHOOK_SECRET` and configure n8n to sign its outgoing requests with the `X-N8N-Signature` header. The API will reject any request that does not carry a valid signature, ensuring only your n8n instance can trigger the agent.

---

## Observability

When Langfuse keys are present in `.env`, every LLM call — inputs, outputs, latency, token counts — is traced to your Langfuse project. This makes it straightforward to review why a given ticket was escalated, inspect the generated draft, and iterate on your prompts based on real data.

To disable tracing entirely, leave `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` blank.

---

## Running Tests

```bash
pytest tests/
```

The test suite uses `pytest-asyncio` for async route and agent tests.

---

## License

This project is for personal and experimental use. If you connect a cloud LLM provider (Gemini, Claude), monitor your API usage to avoid unexpected costs.
