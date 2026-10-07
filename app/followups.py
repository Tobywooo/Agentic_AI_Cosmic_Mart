"""Background jobs: deliver scheduled follow-ups and close inactive cases.

Follow-ups are posted into the conversation; the front end picks them up by polling.
"""

from __future__ import annotations

from datetime import timedelta

from .memory import ExcelMemory
from .store import StateStore, iso, parse_iso, utcnow

OUTSTANDING_RETURN_STATUSES = ("approved", "pickup_booked")


def deliver_due_follow_ups(store: StateStore, memory: ExcelMemory, force: bool = False) -> list[dict]:
    """Mark due (or, with force=True, all pending) follow-ups as sent and post them to the conversation."""
    now = utcnow()
    delivered = []
    with store.transaction() as data:
        for fu in data["follow_ups"].values():
            if fu["status"] != "pending" or (not force and parse_iso(fu["due_at"]) > now):
                continue
            fu["status"] = "sent"
            fu["sent_at"] = iso(now)
            delivered.append(dict(fu))
        statuses = {f["conversation_id"]: data["cases"].get(f["conversation_id"], {}).get("status")
                    for f in delivered}
    for fu in delivered:
        memory.append(fu["conversation_id"], "assistant", fu["message"], customer_id=fu.get("customer_id"),
                      tool_name="follow_up", case_status=statuses[fu["conversation_id"]])
    return delivered


def auto_resolve_inactive_cases(store: StateStore, memory: ExcelMemory, hours: float) -> list[str]:
    """Resolve open cases idle for `hours` that have no return still awaiting pickup or refund.

    Escalated cases are never touched: the human specialist owns those. Pending follow-ups are
    still delivered afterwards, and a new customer message reopens the case.
    """
    if hours <= 0:
        return []
    cutoff = utcnow() - timedelta(hours=hours)
    resolved = []
    with store.transaction() as data:
        for case in data["cases"].values():
            last_activity = parse_iso(case.get("last_reply_at") or case["created_at"])
            outstanding = any(data["returns"][r]["status"] in OUTSTANDING_RETURN_STATUSES
                              for r in case.get("return_ids", []))
            if case["status"] != "open" or last_activity > cutoff or outstanding:
                continue
            case["status"] = "resolved"
            case["resolution"] = f"Auto-resolved after {hours:g}h without activity"
            case["resolved_at"] = iso(utcnow())
            resolved.append((case["conversation_id"], case.get("customer_id")))
    for conversation_id, customer_id in resolved:
        memory.append(conversation_id, "system", f"Case auto-resolved after {hours:g}h without activity.",
                      customer_id=customer_id, tool_name="auto_resolve", case_status="resolved")
    return [conversation_id for conversation_id, _ in resolved]
