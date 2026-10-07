"""Delivers scheduled follow-ups into the conversation (the front end picks them up by polling)."""

from __future__ import annotations

from .memory import ExcelMemory
from .store import StateStore, iso, parse_iso, utcnow


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
