# Agentic_AI_Cosmic_Mart

**End-to-end resolution agent** for Cosmic Mart customer support. It doesn't stop at answering questions. It takes each case all the way to the end: it checks order history, approves returns within a set authority limit, books courier pickups, issues refunds and schedules follow-ups. When it hits a limit or detects that the customer is frustrated, it hands the case to a human with the full context, so the customer never has to repeat themselves.

- **LLM:** any OpenAI-compatible endpoint. Built for a self-hosted **llama.cpp** `llama-server`, with no `--jinja` or native tool calling needed. The agent uses a prompt-based JSON tool protocol.
- **Server:** FastAPI, JSON endpoints only (the front end is built separately).
- **Memory:** `runtime/memory.xlsx`, one row per message tagged with `conversation_id`. It is a **rolling table of at most 50 rows**: the oldest rows are dropped once it's full.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then edit OPENAI_API_KEY / OPENAI_BASE_URL
python run.py
```

Open the interactive API docs at http://127.0.0.1:8000/docs. Check the LLM connection with `GET /health/llm`.

### `.env`

The only values you need are:

```
OPENAI_API_KEY=<key your llama-server was started with (--api-key), any value if none>
OPENAI_BASE_URL=http://<your-server>:8080/v1
```

`.env.example` lists every other setting: authority limit, return window, frustration threshold, memory cap, ports and CORS.

## How it works

```
POST /chat ──► frustration check (deterministic) ──► agent loop (LLM ⇄ tools, max N steps) ──► guardrails ──► reply
                       │                                     │                                    │
                       └── "talk to a human" / angry ────────┴── tool says requires_human ────────┴─► human handoff
```

| File | Purpose |
|---|---|
| `app/agent.py` | ReAct-style loop and **guardrails**: guaranteed handoff on frustration, authority-limit breach or step limit, even if the model forgets |
| `app/tools.py` | The agent's tools. **All business rules are enforced here in code**, not by the prompt |
| `app/protocol.py` | Parses the model's JSON (`action`/`action_input` or `final_answer`) and tolerates code fences and extra text |
| `app/prompts.py` | System prompt, rebuilt each turn with the live case state |
| `app/frustration.py` | Keyword, caps, punctuation and repetition scoring, plus detection of explicit "I want a human" requests |
| `app/memory.py` | Excel memory with a rolling row cap |
| `app/store.py` | Mock order system and case/ticket store (`runtime/state.json`, seeded from `data/seed_data.json`) |
| `app/llm.py` | OpenAI SDK client. Sends `response_format: json_object` (llama.cpp turns it into a grammar) and falls back automatically if the server rejects it |
| `app/main.py` | FastAPI endpoints |

### Agent tools

`verify_customer`, `get_order_history`, `get_order_details`, `check_return_eligibility`, `approve_return`, `get_pickup_slots`, `book_pickup`, `issue_refund`, `schedule_follow_up`, `escalate_to_human`, `resolve_case`

Rules enforced in code:
- Customers can only see their own orders.
- Returns are allowed only within the return window, for delivered orders and returnable categories, and never for more than the quantity left to return.
- The total approved per case must stay within `RETURN_AUTHORITY_LIMIT`. Going over the limit produces `requires_human`, which forces a handoff.
- Steps must happen in order: approve, then pickup, then refund.
- Once a case is escalated, the agent can't change any orders.
- After repeated failed identity checks, the case goes to a human.

### What a human receives (`GET /handoffs/{id}`)

The reason and priority, the agent's summary, the customer profile, all of the customer's orders, returns, pickups and refunds, every action the agent took (with inputs and outcomes), the frustration score and signals, and a transcript snapshot taken at handoff time. A live transcript is added as well. The specialist replies through `POST /handoffs/{id}/reply`, and the message appears in the same conversation.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{message, conversation_id?, customer_id?}` → reply, case status, actions taken, handoff info, frustration score |
| GET | `/conversations/{id}` | Customer-visible messages. Poll this to pick up follow-ups and human replies |
| GET | `/cases`, `/cases/{conversation_id}` | Case records with returns, pickups, refunds and follow-ups |
| GET | `/handoffs?status=open`, `/handoffs/{id}` | Human queue and the full context package |
| POST | `/handoffs/{id}/reply` | `{agent_name, message, close_case}` |
| GET | `/customers`, `/customers/{id}/orders` | Demo data |
| GET | `/memory?conversation_id=` | Rows of the memory table |
| POST | `/follow-ups/run?force=true` | Deliver follow-ups now (handy for demos). They are also delivered automatically every `FOLLOW_UP_POLL_SECONDS` |
| POST | `/admin/reset` | Restore the seed orders and clear cases |
| GET | `/health`, `/health/llm` | Status, and LLM connectivity via `/v1/models` |

Pass `customer_id` if your front end has already logged the customer in. If you leave it out, the agent verifies the customer by email and order number.

## Demo scenarios (seed data)

| Customer | Order | Scenario |
|---|---|---|
| C001 Alice (`alice.tan@example.com`) | CM-10001 headphones 89.90 | Happy path: return → pickup → refund → follow-up |
| C001 Alice | CM-10002 TV 749.00 | Over the authority limit → human handoff |
| C002 Ben (`ben.ortiz@example.com`) | CM-10003 shoes | Delivered 45 days ago → outside the return window |
| C002 Ben | CM-10004 gift card + phone case | Partial: the gift card is non-returnable |
| C003 Chloe (`chloe.ng@example.com`) | CM-10005 blender, kettle, coffee | Multi-item return; the coffee is perishable |
| C003 Chloe | CM-10006 lamp | Still in transit, can't be returned yet |
| C004 Dev (`dev.patel@example.com`) | CM-10007 / CM-10008 | Already returned / smartwatch 199 is over the limit |

Order dates are calculated relative to the first run. To reseed, call `POST /admin/reset` or delete `runtime/state.json`.

## Notes

- **Viewing memory in Excel:** open `runtime/memory.xlsx`. On Windows, Excel locks the file while it's open. The server keeps working from its in-memory copy and rewrites the file on the next message after you close Excel.
- With a 50-row cap, a long conversation that uses many tools will lose its oldest rows. The case record (actions, returns, refunds and handoff) is stored separately in `state.json` and goes into every prompt, so the agent doesn't lose track of the case.
- Tests use a scripted model and don't need the LLM server: `pytest`
