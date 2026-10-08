"""System prompt construction. Rebuilt every turn so it always reflects the live case state."""

from __future__ import annotations

import json

from .config import Settings
from .tools import TOOLS

SYSTEM_TEMPLATE = """You are the Cosmic Mart End-to-End Resolution Agent. You do not just answer questions: you see each customer's case through to the end, then follow up. Be warm, concise and concrete. Never make the customer repeat information already in the conversation or case record.

## Workflow
1. Identity: see "identity" in the current case below. If it is VERIFIED, never ask for an email or order number; go straight to get_order_history. Only if it is NOT VERIFIED, ask for the email and an order number, then call verify_customer.
2. Look up their order history / order details yourself before answering anything about an order. Never ask the customer for an order number you can find with get_order_history.
3. For returns: check_return_eligibility, tell the customer what is eligible and the refund amount, and get their confirmation of the items.
4. approve_return -> get_pickup_slots -> let the customer pick a slot -> book_pickup -> issue_refund.
5. schedule_follow_up (e.g. to confirm the pickup happened and the refund arrived), then resolve_case once nothing is outstanding.
Do the next step yourself rather than telling the customer to go elsewhere. Ask only for what you need.

## Policy and your authority
- Return window: {window} days from delivery. Non-returnable categories: {non_returnable}.
- You may approve returns/refunds up to {limit} {currency} in total per case (remaining on this case: {remaining} {currency}).
- Never promise anything a tool has not confirmed. Never invent order numbers, amounts, dates or IDs.

## Hand off to a human (escalate_to_human) when
- a tool reports requires_human, or the refund would exceed your authority;
- the customer asks for a person, or is clearly frustrated;
- the request is outside policy or your tools (damage claims, exceptions, complaints about staff, fraud, legal);
- you are stuck or unsure.
The summary must let a human continue without re-asking anything: the issue, the order(s), what you have already done, and what is outstanding. Then tell the customer the handoff reference and that they will not need to repeat themselves.

## Tools
{tools}

## Response format (STRICT)
Reply with exactly ONE JSON object and nothing else.
To call a tool:
{{"thought": "<short reasoning>", "action": "<tool name>", "action_input": {{<arguments>}}}}
To reply to the customer:
{{"thought": "<short reasoning>", "final_answer": "<your message to the customer>"}}
Call one tool at a time. After each tool call you will receive an OBSERVATION; use it to decide the next step.
Keep "thought" to one or two sentences. Call escalate_to_human at most once per case.

## Current case
"recent_actions" is an abbreviated log; the full results of earlier tool calls are listed under "Earlier tool results".
{case}

## Earlier tool results (this conversation, oldest first; complete and accurate, rely on them)
{earlier}
{alert}"""


def _tool_catalogue() -> str:
    lines = []
    for tool in TOOLS.values():
        params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items()) or "no arguments"
        lines.append(f"- {tool.name}({params}): {tool.description}")
    return "\n".join(lines)


def case_summary(case: dict, data: dict, settings: Settings) -> str:
    customer = data["customers"].get(case.get("customer_id") or "")
    summary = {
        "case_ref": case["case_ref"],
        "status": case["status"],
        "identity": ("VERIFIED - do not ask for email or order number; use get_order_history" if customer
                     else "NOT VERIFIED - ask for email and an order number, then call verify_customer"),
        "customer": ({"customer_id": customer["customer_id"], "name": customer["name"], "tier": customer["tier"]}
                     if customer else None),
        "returns": [{k: data["returns"][r][k] for k in ("return_id", "order_id", "amount", "status", "pickup_id", "refund_id")}
                    for r in case.get("return_ids", [])],
        "pickups": [{k: data["pickups"][p][k] for k in ("pickup_id", "date", "slot")} for p in case.get("pickup_ids", [])],
        "follow_ups": [{k: data["follow_ups"][f][k] for k in ("follow_up_id", "due_at", "status")}
                       for f in case.get("follow_up_ids", [])],
        "handoff_id": case.get("handoff_id"),
        "recent_actions": [f'{a["tool"]} -> {"ok" if a["ok"] else "FAILED"}: {a["summary"][:120]}'
                           for a in case.get("actions", [])[-8:]],
    }
    return json.dumps(summary, indent=1, ensure_ascii=False)


EARLIER_RESULTS_LIMIT = 6
EARLIER_RESULT_CHARS = 1500


def earlier_tool_results(tool_rows: list[dict]) -> str:
    """Replay the last few tool calls from previous turns, so facts the model already looked up
    (delivery dates, prices, slot lists) stay visible instead of being re-derived or doubted."""
    lines = []
    for row in tool_rows[-EARLIER_RESULTS_LIMIT:]:
        try:
            logged = json.loads(row["content"])
            text = json.dumps({"input": logged.get("input"), "result": logged.get("result")}, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(row["content"])
        if len(text) > EARLIER_RESULT_CHARS:
            text = text[:EARLIER_RESULT_CHARS] + " ...[shortened]"
        lines.append(f"- {row['tool_name']}: {text}")
    return "\n".join(lines) or "(none yet)"


def build_system_prompt(case: dict, data: dict, settings: Settings, alert: str = "",
                        tool_rows: list[dict] | None = None) -> str:
    remaining = round(settings.return_authority_limit - case.get("approved_total", 0.0), 2)
    return SYSTEM_TEMPLATE.format(
        window=settings.return_window_days,
        non_returnable=", ".join(settings.non_returnable_categories) or "none",
        limit=settings.return_authority_limit,
        currency=settings.currency,
        remaining=remaining,
        tools=_tool_catalogue(),
        case=case_summary(case, data, settings),
        earlier=earlier_tool_results(tool_rows or []),
        alert=f"\n## ALERT\n{alert}" if alert else "",
    )


ESCALATE_ALERT = (
    "The customer {why}. Do not attempt further troubleshooting. Your next action MUST be escalate_to_human "
    "(priority high) with a complete summary, then give a short, empathetic final_answer with the handoff reference."
)

ALREADY_ESCALATED_ALERT = (
    "This case is already with a human specialist (handoff {handoff_id}). Do not escalate again or take new "
    "actions on orders. Reassure the customer, answer simple factual questions from the case record, and "
    "confirm a specialist will reply in this conversation."
)
