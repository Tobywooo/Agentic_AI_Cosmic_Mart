# Agentic_AI_Cosmic_Mart

**End-to-end resolution agent** for Cosmic Mart customer support. It doesn't stop at answering questions. It takes each case all the way to the end: it checks order history, approves returns within a set authority limit, books courier pickups, issues refunds and schedules follow-ups. When it hits a limit or detects that the customer is frustrated, it hands the case to a human with the full context, so the customer never has to repeat themselves.

- **LLM:** any OpenAI-compatible endpoint, accessed with the OpenAI Python SDK (`from openai import OpenAI`). Built for a self-hosted **llama.cpp** `llama-server`, with no `--jinja` or native tool calling needed. The agent uses a prompt-based JSON tool protocol. Tested with `qwen3.8-27b`.
- **Server:** FastAPI, JSON endpoints only. The chatbot front end is built separately.
- **Memory:** `runtime/memory.xlsx`, one row per message tagged with `conversation_id`. It is a **rolling table of at most 50 rows**: the oldest rows are dropped once it's full.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env, then edit it
python run.py
```

- API docs (try every endpoint in the browser): http://127.0.0.1:8000/docs
- Check the LLM connection: http://127.0.0.1:8000/health/llm should list your model.

### `.env`

Three values are needed. `.env` is git-ignored, so never commit it.

```
OPENAI_API_KEY=<key your llama-server was started with (--api-key); any value if none>
OPENAI_BASE_URL=http://<your-server>:8080/v1      # must end in /v1
LLM_MODEL=<model name your server reports>
```

`.env.example` lists every other setting: authority limit, return window, frustration threshold, memory cap, auto-resolve time, ports and CORS.

## Example exchange

```http
POST /chat
{ "message": "Hi, the headphones I got last week aren't for me. Can I return them?", "customer_id": "C001" }
```

```json
{
  "conversation_id": "3f9a1c2b7d4e",
  "case_ref": "CASE-00001",
  "reply": "Yes, you can! Your Nebula Wireless Headphones (order CM-10001) qualify for a full refund of $89.90. Shall I go ahead?",
  "message_id": 4,
  "case_status": "open",
  "escalated": false,
  "handoff_id": null,
  "frustration": { "score": 0.0, "requests_human": false, "signals": [] },
  "actions": [
    { "tool": "get_order_history", "input": {}, "result": { "ok": true, "orders": ["..."] } },
    { "tool": "check_return_eligibility", "input": { "order_id": "CM-10001" },
      "result": { "ok": true, "eligible_refund_total": 89.9, "within_authority": true, "...": "..." } }
  ]
}
```

Send the returned `conversation_id` with every later message in the same chat. In live testing, a full return took four messages: confirm the item, then approve the return, then book the pickup and refund, then close the case.

## How it works

```
POST /chat ──► frustration check (deterministic) ──► agent loop (LLM ⇄ tools, max N steps) ──► guardrails ──► reply
                       │                                     │                                    │
                       └── "talk to a human" / angry ────────┴── over limit / requires_human ─────┴─► human handoff
```

1. **The case is opened, or reopened.** Each conversation has one case (`CASE-00001`, …).
2. **Frustration check.** Before the model sees the message, the server scores it from 0 to 1 for angry phrases, capitals, repeated `!!` and repeated messages. It also detects requests for a person. A score of `FRUSTRATION_THRESHOLD` (0.6) or above, or any request for a person, means this turn must end in a handoff.
3. **Prompt.** The system prompt is rebuilt every turn with the policy, the tool list and a summary of the live case: whether the customer is verified, their returns, pickups, refunds, remaining authority and recent actions. Recent messages from memory follow it.
4. **Agent loop.** The model replies with one JSON object each step:
   - `{"thought": "...", "action": "<tool>", "action_input": {...}}` runs that tool. The result is sent back as an `OBSERVATION`, and the loop continues.
   - `{"thought": "...", "final_answer": "..."}` ends the loop, and that text is the reply.

   An invalid JSON reply gets one retry. If the reply is still not valid JSON, the plain text is used as the reply.
5. **Guardrails.** If a handoff was required and the model didn't create one, the server creates it. A handoff is required when:
   - the customer was frustrated or asked for a person
   - a tool flagged `requires_human`
   - an eligibility check came out over the limit
   - the agent ran out of steps
6. **Logging.** Every message and tool call is written to the memory table.

| File | Purpose |
|---|---|
| `app/agent.py` | Agent loop and guardrails: guaranteed handoff on frustration, authority-limit breach or step limit, even if the model forgets |
| `app/tools.py` | The agent's tools. **All business rules are enforced here in code**, not by the prompt |
| `app/prompts.py` | System prompt, rebuilt each turn with the live case state |
| `app/protocol.py` | Parses the model's JSON and tolerates code fences and extra text |
| `app/llm.py` | OpenAI SDK client. Sends `response_format: json_object` (llama.cpp turns it into a grammar) and falls back automatically if the server rejects it |
| `app/frustration.py` | Frustration scoring and detection of explicit "I want a human" requests |
| `app/memory.py` | Excel memory with a rolling row cap |
| `app/store.py` | Mock order system and case/ticket store (`runtime/state.json`, seeded from `data/seed_data.json`) |
| `app/followups.py` | Background jobs: sends scheduled follow-ups and auto-resolves inactive cases |
| `app/main.py` | FastAPI endpoints |
| `app/schemas.py` | Request/response models for `/chat` and human replies |
| `app/config.py` | Settings loaded from `.env` |

### Agent tools

`verify_customer`, `get_order_history`, `get_order_details`, `check_return_eligibility`, `approve_return`, `get_pickup_slots`, `book_pickup`, `issue_refund`, `schedule_follow_up`, `escalate_to_human`, `resolve_case`

Rules enforced in code:
- Customers can only see their own orders.
- Returns are allowed only within `RETURN_WINDOW_DAYS` (30), for delivered orders and returnable categories, and never for more than the quantity left to return.
- The total approved per case must stay within `RETURN_AUTHORITY_LIMIT` (150). Going over the limit forces a handoff.
- Steps must happen in order: approve, then pickup, then refund.
- Pickups are offered for the next `PICKUP_DAYS_AHEAD` (7) days, excluding Sundays.
- Once a case is escalated, the agent can't change any orders.
- After `MAX_VERIFICATION_ATTEMPTS` (3) failed identity checks, the case goes to a human.

### Customer identity

There are two ways to identify the customer:
- **Logged in:** the front end sends `customer_id` with `/chat`. The server trusts it and treats the customer as verified, so the agent goes straight to their orders.
- **Not logged in:** leave `customer_id` out. The agent asks for an email address and an order number, and `verify_customer` checks they match.

### Case status lifecycle

| Status | Set when |
|---|---|
| `open` | The first message in a conversation creates the case. A `resolved` case reopens when the customer writes again, and its old handoff is archived so a new one can be raised |
| `escalated` | A handoff is created, either by the model or forced by the server: frustration, a request for a person, a refund over the authority limit, or the step limit |
| `resolved` | The model calls `resolve_case`, a human replies with `close_case: true`, or the case has been idle for `AUTO_RESOLVE_HOURS` (72) with no return still waiting for a pickup or refund |

Handoffs have their own status: `open` → `in_progress` (after a human replies) → `resolved`.

### Human handoff

`GET /handoffs/{id}` gives the specialist everything they need:
- the reason and priority, plus the agent's summary
- the customer profile and all of their orders
- returns, pickups and refunds
- every action the agent took, with inputs and outcomes
- the frustration score and signals
- any earlier handoffs
- a transcript snapshot taken at handoff time, plus the live transcript

The specialist replies with `POST /handoffs/{id}/reply`, and the message appears in the customer's conversation as `role: "human_agent"`.

## Building the front end

**Chat loop**
1. Send `POST /chat` with `{ message, conversation_id, customer_id }`. Leave `conversation_id` out on the first message, then save the one returned (e.g. in `localStorage`).
2. Show a typing indicator and disable sending until the reply arrives. A message takes roughly **2–25 seconds**, because the agent may call the model several times. Don't set a short request timeout.
3. Every 3–5 seconds, poll `GET /conversations/{id}?after=<newest message_id you have>` for follow-ups and human replies. Pause polling while a `/chat` request is in progress.
4. Track messages by `message_id` and skip any you've already shown. The customer's own message and the reply both come back from polling, so show the customer's message as a temporary "sending…" bubble and let the polled version replace it.
5. Keep your own copy of the chat history. The memory table holds only 50 rows, so old messages drop out of `/conversations`.

**Message types** (`role` / `kind` in `/conversations`)

| `role` / `kind` | Show as |
|---|---|
| `user` | Customer bubble |
| `assistant` / `message` | Agent bubble |
| `assistant` / `follow_up` | Agent bubble, optionally labelled "Follow-up" |
| `human_agent` | Specialist bubble. The content looks like `"Sam: …"`, so split on the first `": "` to get the name |
| `system` / `auto_resolve` | Small grey "case closed" note |

When `escalated` is `true`, show a banner such as "A specialist has your case (ref HO-00001)".

**Errors**

| Status | Meaning | Suggested handling |
|---|---|---|
| `503` | The LLM server can't be reached | "We're having trouble, please try again" |
| `409` | The conversation belongs to a different customer | Start a new conversation |
| `422` | The message is empty or longer than 4,000 characters | Check the length before sending |
| `404` | Unknown `conversation_id` (e.g. after `/admin/reset`) | Clear the saved ID and start again |

**Specialist page (optional).** List the queue with `GET /handoffs?status=open`, show a case with `GET /handoffs/{id}`, and reply with `POST /handoffs/{id}/reply`. Without this page, you can do all of it from `/docs`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{message, conversation_id?, customer_id?}` → reply, `message_id`, case status, actions taken, handoff info, frustration score |
| GET | `/conversations/{id}?after=<message_id>` | Customer-visible messages, each with a `message_id`. Use `after` to get only new ones |
| GET | `/cases?status=`, `/cases/{conversation_id}` | Case records with returns, pickups, refunds and follow-ups |
| GET | `/handoffs?status=`, `/handoffs/{id}` | Human queue and the full context package |
| POST | `/handoffs/{id}/reply` | `{agent_name, message, close_case}` |
| GET | `/customers`, `/customers/{id}/orders` | Demo data |
| GET | `/memory?conversation_id=` | Rows of the memory table |
| POST | `/follow-ups/run?force=true` | Send follow-ups now (handy for demos). They are also sent automatically every `FOLLOW_UP_POLL_SECONDS` |
| POST | `/admin/reset` | Restore the seed orders and clear cases (memory is left as is) |
| GET | `/health`, `/health/llm` | Server status, and LLM connectivity via `/v1/models` |

`/chat` requests and responses are typed in `app/schemas.py`. Other endpoints return the dictionaries built in `app/main.py`. The full spec is at `/openapi.json`.

## Memory table (`runtime/memory.xlsx`)

| Column | Contents |
|---|---|
| `row_id` | Increasing ID, exposed to the front end as `message_id` |
| `timestamp` | UTC, ISO 8601 |
| `conversation_id` / `customer_id` | Which chat and customer the row belongs to |
| `role` | `user`, `assistant`, `tool`, `human_agent` or `system` |
| `tool_name` | Tool called (for `tool` rows), or `follow_up` / `auto_resolve` |
| `content` | The message, or for tool rows the model's reasoning, the tool input and the result as JSON |
| `case_status` | Case status when the row was written |

Once the table passes `MEMORY_MAX_ROWS` (50), the oldest rows are dropped. Tool calls use rows too: a full return conversation takes about 16. The case record in `runtime/state.json` isn't capped and goes into every prompt, so the agent doesn't lose track of a case when old messages drop out.

## Demo scenarios (seed data)

| Customer | Order | Scenario |
|---|---|---|
| C001 Alice (`alice.tan@example.com`) | CM-10001 headphones 89.90 | Happy path: return → pickup → refund → follow-up → resolved |
| C001 Alice | CM-10002 TV 749.00 | Over the authority limit → human handoff |
| C002 Ben (`ben.ortiz@example.com`) | CM-10003 shoes | Delivered 45 days ago → outside the return window → handoff for an exception |
| C002 Ben | CM-10004 gift card + phone case | Partial: the gift card is non-returnable |
| C003 Chloe (`chloe.ng@example.com`) | CM-10005 blender, kettle, coffee | Multi-item return; the coffee is perishable |
| C003 Chloe | CM-10006 lamp | Still in transit, can't be returned yet |
| C004 Dev (`dev.patel@example.com`) | CM-10007 / CM-10008 | Already returned / smartwatch 199 is over the limit |

Try frustration with: *"This is ridiculous!! I want to speak to a real person."*

Order dates are calculated from the first run. To reseed, call `POST /admin/reset` or delete `runtime/state.json`. Editing `data/seed_data.json` only takes effect after a reseed.

## Testing

```bash
pytest
```

The 16 tests use a scripted model, so they don't need the LLM server. They cover:
- the full return flow
- the authority limit
- frustration handoffs
- ownership checks
- verification
- invalid JSON from the model
- the step limit
- reopening a case and escalating it again
- auto-resolve
- the memory cap
- the API endpoints

The scenarios above were also run against a live `qwen3.8-27b` on llama.cpp. In each one the model chose the right action itself, so the server's backup handoff never had to step in.

## Troubleshooting

| Symptom | Check |
|---|---|
| `/health/llm` or `/chat` returns 503 | Is `OPENAI_BASE_URL` reachable, does it end in `/v1`, and is `OPENAI_API_KEY` correct? |
| Logged-in customer is asked for their email | Make sure the front end sends `customer_id` with `/chat` |
| Model replies aren't valid JSON | Keep `LLM_JSON_MODE=true` and lower `LLM_TEMPERATURE`. The server log shows a warning if the server rejects JSON mode |
| `memory.xlsx` isn't updating | Close it in Excel. Windows locks the file while it's open, and the server catches up on the next message |
| Old messages are missing from `/conversations` | Expected: memory is capped at 50 rows |
| Follow-ups never appear | They're due 48 h or more later. Use `POST /follow-ups/run?force=true` for demos |

## Security and limitations

- **There's no authentication.** Anyone who can reach the API can send any `customer_id`. Keep `HOST=127.0.0.1` unless you're on a trusted network, and add token auth before exposing it.
- Email plus order number is a weak identity check. A real deployment should add a one-time code.
- The order system is mock data in a JSON file. Replace `app/store.py` with real API clients to go to production.
- Message IDs come from the memory file. If `runtime/memory.xlsx` is deleted, they restart at 1, so reload the front end afterwards.
