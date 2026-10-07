"""Tools available to the resolution agent.

All business rules (authority limit, return window, ownership checks, step ordering)
are enforced here in code. The LLM decides *what* to do; these functions decide
whether it is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from .config import Settings
from .memory import ExcelMemory
from .store import StateStore, iso, parse_iso, utcnow

PICKUP_SLOTS = ("09:00-12:00", "12:00-15:00", "15:00-18:00")


class ToolError(Exception):
    """A tool refused or failed. `extra` is merged into the observation shown to the LLM.

    The tool's state changes are rolled back; `case_updates` are applied to the case afterwards
    (e.g. counting failed verification attempts).
    """

    def __init__(self, message: str, case_updates: dict | None = None, **extra):
        super().__init__(message)
        self.case_updates = case_updates or {}
        self.extra = extra


@dataclass
class ToolContext:
    store: StateStore
    memory: ExcelMemory
    settings: Settings
    conversation_id: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    required: tuple[str, ...]
    fn: Callable[[ToolContext, dict, dict, dict], dict]


# ---------------------------------------------------------------------------- helpers
def money(value: float) -> float:
    return round(float(value), 2)


def _require_customer(case: dict) -> str:
    if not case.get("customer_id"):
        raise ToolError(
            "Customer is not verified yet. Ask for their email address and an order number, "
            "then call verify_customer."
        )
    return case["customer_id"]


def _get_order(data: dict, case: dict, order_id) -> dict:
    customer_id = _require_customer(case)
    order = data["orders"].get(str(order_id).strip().upper())
    if not order or order["customer_id"] != customer_id:
        raise ToolError(f"Order {order_id} was not found on this customer's account.")
    return order


def _get_return(data: dict, case: dict, return_id) -> dict:
    rma = data["returns"].get(str(return_id).strip().upper())
    if not rma or rma["conversation_id"] != case["conversation_id"]:
        raise ToolError(f"Return {return_id} was not found on this case.")
    return rma


def _remaining_authority(ctx: ToolContext, case: dict) -> float:
    return money(ctx.settings.return_authority_limit - case.get("approved_total", 0.0))


def _requested_items(order: dict, items_arg) -> list[tuple[dict, int]]:
    """Resolve the `items` argument to (order_item, quantity). Defaults to everything not yet returned."""
    by_id = {i["item_id"]: i for i in order["items"]}
    if not items_arg:
        return [(i, i["quantity"] - i["returned_quantity"]) for i in order["items"]
                if i["quantity"] - i["returned_quantity"] > 0]
    if isinstance(items_arg, (str, dict)):
        items_arg = [items_arg]
    resolved = []
    for entry in items_arg:
        item_id = entry if isinstance(entry, str) else entry.get("item_id")
        qty = 1 if isinstance(entry, str) else entry.get("quantity", 1)
        item = by_id.get(str(item_id).strip().upper())
        if not item:
            raise ToolError(f"Item {item_id} is not part of order {order['order_id']}.",
                            valid_item_ids=list(by_id))
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            raise ToolError(f"Quantity for {item_id} must be a whole number.")
        if qty < 1:
            raise ToolError(f"Quantity for {item_id} must be at least 1.")
        resolved.append((item, qty))
    return resolved


def _assess(ctx: ToolContext, order: dict, items_arg) -> list[dict]:
    s = ctx.settings
    delivered_at = order.get("delivered_at")
    days_since = (utcnow() - parse_iso(delivered_at)).days if delivered_at else None
    results = []
    for item, qty in _requested_items(order, items_arg):
        remaining = item["quantity"] - item["returned_quantity"]
        reason = None
        if order["status"] not in ("delivered", "partially_returned"):
            reason = f"Order status is '{order['status']}'; only delivered orders can be returned."
        elif days_since is None or days_since > s.return_window_days:
            reason = f"Delivered {days_since} days ago; the return window is {s.return_window_days} days."
        elif item["category"] in s.non_returnable_categories:
            reason = f"Category '{item['category']}' is non-returnable."
        elif remaining <= 0:
            reason = "This item has already been returned."
        elif qty > remaining:
            reason = f"Only {remaining} unit(s) left to return."
        results.append({
            "item_id": item["item_id"],
            "name": item["name"],
            "quantity": qty,
            "unit_price": item["unit_price"],
            "refund_value": money(item["unit_price"] * qty),
            "eligible": reason is None,
            "reason": reason or "Eligible",
        })
    if not results:
        raise ToolError(f"Nothing left to return on order {order['order_id']}.")
    return results


def log_action(case: dict, tool: str, args: dict, ok: bool, summary: str) -> None:
    case.setdefault("actions", []).append(
        {"at": iso(utcnow()), "tool": tool, "input": args, "ok": ok, "summary": summary[:300]}
    )
    case["updated_at"] = iso(utcnow())


# ---------------------------------------------------------------------------- tools
def verify_customer(ctx, data, case, args):
    if case.get("customer_id"):
        customer = data["customers"][case["customer_id"]]
        return {"customer_id": customer["customer_id"], "name": customer["name"], "note": "Already verified."}
    email = str(args["email"]).strip().lower()
    order = data["orders"].get(str(args["order_id"]).strip().upper())
    customer = data["customers"].get(order["customer_id"]) if order else None
    if not customer or customer["email"].lower() != email:
        failures = {"verification_failures": case.get("verification_failures", 0) + 1}
        if failures["verification_failures"] >= ctx.settings.max_verification_attempts:
            raise ToolError("Verification failed too many times. Escalate to a human.",
                            case_updates=failures, requires_human=True)
        raise ToolError("The email and order number do not match our records. Ask the customer to double-check.",
                        case_updates=failures)
    case["customer_id"] = customer["customer_id"]
    return {"customer_id": customer["customer_id"], "name": customer["name"], "tier": customer["tier"]}


def get_order_history(ctx, data, case, args):
    customer_id = _require_customer(case)
    orders = sorted((o for o in data["orders"].values() if o["customer_id"] == customer_id),
                    key=lambda o: o["placed_at"], reverse=True)
    return {"orders": [{
        "order_id": o["order_id"],
        "status": o["status"],
        "placed_at": o["placed_at"][:10],
        "delivered_at": (o["delivered_at"] or "")[:10] or None,
        "total": o["total"],
        "items": [f'{i["item_id"]}: {i["name"]} x{i["quantity"]} @ {i["unit_price"]} ({i["category"]})'
                  + (f', {i["returned_quantity"]} returned' if i["returned_quantity"] else "")
                  for i in o["items"]],
    } for o in orders]}


def get_order_details(ctx, data, case, args):
    order = _get_order(data, case, args["order_id"])
    returns = [r for r in data["returns"].values() if r["order_id"] == order["order_id"]]
    return {"order": order, "returns": returns}


def check_return_eligibility(ctx, data, case, args):
    order = _get_order(data, case, args["order_id"])
    items = _assess(ctx, order, args.get("items"))
    total = money(sum(i["refund_value"] for i in items if i["eligible"]))
    remaining = _remaining_authority(ctx, case)
    return {
        "order_id": order["order_id"],
        "items": items,
        "eligible_refund_total": total,
        "agent_remaining_authority": remaining,
        "within_authority": total <= remaining,
        "note": ("You may approve this return." if total and total <= remaining else
                 "Exceeds your authority: escalate_to_human." if total else "No eligible items."),
    }


def approve_return(ctx, data, case, args):
    order = _get_order(data, case, args["order_id"])
    items = _assess(ctx, order, args.get("items"))
    ineligible = [i for i in items if not i["eligible"]]
    if ineligible:
        raise ToolError("Some requested items are not eligible for return.",
                        ineligible=[{"item_id": i["item_id"], "reason": i["reason"]} for i in ineligible])
    amount = money(sum(i["refund_value"] for i in items))
    remaining = _remaining_authority(ctx, case)
    if amount > remaining:
        raise ToolError(
            f"Refund value {amount} {ctx.settings.currency} exceeds your remaining authority of "
            f"{remaining} {ctx.settings.currency}. You must escalate_to_human.",
            requires_human=True,
        )
    return_id = ctx.store.next_id(data, "RMA")
    by_id = {i["item_id"]: i for i in order["items"]}
    for i in items:
        by_id[i["item_id"]]["returned_quantity"] += i["quantity"]
    order["status"] = ("returned" if all(i["returned_quantity"] >= i["quantity"] for i in order["items"])
                       else "partially_returned")
    data["returns"][return_id] = {
        "return_id": return_id,
        "order_id": order["order_id"],
        "conversation_id": case["conversation_id"],
        "customer_id": case["customer_id"],
        "items": [{k: i[k] for k in ("item_id", "name", "quantity", "unit_price")} for i in items],
        "amount": amount,
        "reason": str(args["reason"]),
        "status": "approved",
        "created_at": iso(utcnow()),
        "pickup_id": None,
        "refund_id": None,
    }
    case["approved_total"] = money(case.get("approved_total", 0.0) + amount)
    case.setdefault("return_ids", []).append(return_id)
    return {"return_id": return_id, "amount": amount, "status": "approved",
            "next_step": "Offer pickup slots with get_pickup_slots."}


def _available_slots(ctx: ToolContext) -> dict[str, list[str]]:
    today = utcnow().date()
    days = {}
    for offset in range(1, ctx.settings.pickup_days_ahead + 1):
        day = today + timedelta(days=offset)
        if day.weekday() != 6:  # no Sunday pickups
            days[day.isoformat()] = list(PICKUP_SLOTS)
    return days


def get_pickup_slots(ctx, data, case, args):
    rma = _get_return(data, case, args["return_id"])
    if rma["pickup_id"]:
        raise ToolError(f"Pickup {rma['pickup_id']} is already booked for this return.")
    customer = data["customers"][rma["customer_id"]]
    return {"return_id": rma["return_id"], "pickup_address": customer["address"],
            "available": {f"{d} ({date.fromisoformat(d):%A})": slots
                          for d, slots in _available_slots(ctx).items()}}


def book_pickup(ctx, data, case, args):
    rma = _get_return(data, case, args["return_id"])
    if rma["status"] != "approved" or rma["pickup_id"]:
        raise ToolError(f"Return {rma['return_id']} is '{rma['status']}' and cannot have a new pickup booked.")
    date_str = str(args["date"]).strip()[:10]
    slot = str(args["slot"]).strip()
    available = _available_slots(ctx)
    if date_str not in available:
        raise ToolError(f"{date_str} is not an available pickup date.", available_dates=list(available))
    if slot not in available[date_str]:
        raise ToolError(f"'{slot}' is not a valid slot.", valid_slots=list(PICKUP_SLOTS))
    customer = data["customers"][rma["customer_id"]]
    pickup_id = ctx.store.next_id(data, "PU")
    data["pickups"][pickup_id] = {
        "pickup_id": pickup_id, "return_id": rma["return_id"], "conversation_id": case["conversation_id"],
        "date": date_str, "slot": slot, "address": customer["address"], "status": "booked",
        "created_at": iso(utcnow()),
    }
    rma["pickup_id"] = pickup_id
    rma["status"] = "pickup_booked"
    case.setdefault("pickup_ids", []).append(pickup_id)
    return {"pickup_id": pickup_id, "date": date_str, "slot": slot, "address": customer["address"],
            "next_step": "Issue the refund with issue_refund."}


def issue_refund(ctx, data, case, args):
    rma = _get_return(data, case, args["return_id"])
    if rma["refund_id"]:
        refund = data["refunds"][rma["refund_id"]]
        return {"refund_id": refund["refund_id"], "amount": refund["amount"], "note": "Refund was already issued."}
    if rma["status"] != "pickup_booked":
        raise ToolError("A pickup must be booked before the refund can be issued (use get_pickup_slots / book_pickup).")
    if rma["amount"] > ctx.settings.return_authority_limit:  # defence in depth
        raise ToolError("Refund exceeds your authority limit. You must escalate_to_human.", requires_human=True)
    order = data["orders"][rma["order_id"]]
    refund_id = ctx.store.next_id(data, "RF")
    data["refunds"][refund_id] = {
        "refund_id": refund_id, "return_id": rma["return_id"], "conversation_id": case["conversation_id"],
        "amount": rma["amount"], "currency": ctx.settings.currency, "method": order["payment_method"],
        "status": "processed", "eta": "3-5 business days", "created_at": iso(utcnow()),
    }
    rma["refund_id"] = refund_id
    rma["status"] = "refunded"
    case.setdefault("refund_ids", []).append(refund_id)
    return {"refund_id": refund_id, "amount": rma["amount"], "currency": ctx.settings.currency,
            "method": order["payment_method"], "eta": "3-5 business days",
            "next_step": "Schedule a follow-up with schedule_follow_up."}


def schedule_follow_up(ctx, data, case, args):
    try:
        delay_hours = float(args.get("delay_hours", 48))
    except (TypeError, ValueError):
        raise ToolError("delay_hours must be a number.")
    if not 0 < delay_hours <= 24 * 30:
        raise ToolError("delay_hours must be between 0 and 720.")
    follow_up_id = ctx.store.next_id(data, "FU")
    due_at = iso(utcnow() + timedelta(hours=delay_hours))
    data["follow_ups"][follow_up_id] = {
        "follow_up_id": follow_up_id, "conversation_id": case["conversation_id"],
        "customer_id": case.get("customer_id"), "due_at": due_at, "message": str(args["message"]),
        "status": "pending", "created_at": iso(utcnow()),
    }
    case.setdefault("follow_up_ids", []).append(follow_up_id)
    return {"follow_up_id": follow_up_id, "due_at": due_at}


def escalate_to_human(ctx, data, case, args):
    if case.get("handoff_id"):
        return {"handoff_id": case["handoff_id"], "note": "Case is already with a human specialist."}
    return create_handoff(ctx, data, case, reason=str(args["reason"]), summary=str(args["summary"]),
                          priority=str(args.get("priority", "normal")))


def create_handoff(ctx: ToolContext, data: dict, case: dict, *, reason: str, summary: str,
                   priority: str = "normal", source: str = "agent") -> dict:
    """Package everything a human needs so the customer never has to repeat themselves."""
    priority = priority if priority in ("low", "normal", "high", "urgent") else "normal"
    handoff_id = ctx.store.next_id(data, "HO")
    customer = data["customers"].get(case.get("customer_id") or "")
    data["handoffs"][handoff_id] = {
        "handoff_id": handoff_id,
        "case_ref": case["case_ref"],
        "conversation_id": case["conversation_id"],
        "status": "open",
        "priority": priority,
        "raised_by": source,
        "reason": reason,
        "agent_summary": summary,
        "created_at": iso(utcnow()),
        "customer": customer,
        "orders": [o for o in data["orders"].values() if customer and o["customer_id"] == customer["customer_id"]],
        "returns": [data["returns"][r] for r in case.get("return_ids", [])],
        "pickups": [data["pickups"][p] for p in case.get("pickup_ids", [])],
        "refunds": [data["refunds"][r] for r in case.get("refund_ids", [])],
        "agent_authority_remaining": _remaining_authority(ctx, case),
        "frustration": case.get("frustration"),
        "actions_taken": list(case.get("actions", [])),
        "transcript": [{k: r[k] for k in ("timestamp", "role", "tool_name", "content")}
                       for r in ctx.memory.history(case["conversation_id"])],
        "human_replies": [],
    }
    case["handoff_id"] = handoff_id
    case["status"] = "escalated"
    return {"handoff_id": handoff_id, "priority": priority,
            "note": "A human specialist now has the full case history. Tell the customer the reference "
                    "and that they will not need to repeat themselves."}


def resolve_case(ctx, data, case, args):
    if case.get("status") == "escalated":
        raise ToolError("This case is with a human specialist; only they can close it.")
    case["status"] = "resolved"
    case["resolution"] = str(args["resolution_summary"])
    case["resolved_at"] = iso(utcnow())
    return {"case_ref": case["case_ref"], "status": "resolved"}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("verify_customer", "Verify identity by matching email to an order number. Required before any order access.",
         {"email": "string", "order_id": "string, e.g. CM-10001"}, ("email", "order_id"), verify_customer),
    Tool("get_order_history", "List the verified customer's orders, items and statuses.",
         {}, (), get_order_history),
    Tool("get_order_details", "Full details of one order, including existing returns.",
         {"order_id": "string"}, ("order_id",), get_order_details),
    Tool("check_return_eligibility", "Check which items can be returned, the refund value and whether it is within your authority.",
         {"order_id": "string", "items": "optional list of {item_id, quantity}; omit for all items"},
         ("order_id",), check_return_eligibility),
    Tool("approve_return", "Approve a return (creates an RMA). Only after the customer confirms which items.",
         {"order_id": "string", "items": "optional list of {item_id, quantity}; omit for all items",
          "reason": "string, customer's reason"}, ("order_id", "reason"), approve_return),
    Tool("get_pickup_slots", "List available courier pickup dates/slots for an approved return.",
         {"return_id": "string, e.g. RMA-00001"}, ("return_id",), get_pickup_slots),
    Tool("book_pickup", "Book a courier pickup for an approved return at the customer's address on file.",
         {"return_id": "string", "date": "YYYY-MM-DD", "slot": "one of " + ", ".join(PICKUP_SLOTS)},
         ("return_id", "date", "slot"), book_pickup),
    Tool("issue_refund", "Issue the refund to the original payment method once a pickup is booked.",
         {"return_id": "string"}, ("return_id",), issue_refund),
    Tool("schedule_follow_up", "Schedule a proactive follow-up message to the customer.",
         {"message": "string, the follow-up to send", "delay_hours": "number, default 48"},
         ("message",), schedule_follow_up),
    Tool("escalate_to_human", "Hand the case to a human specialist with full context.",
         {"reason": "string, why a human is needed", "summary": "string, issue, what was done, what is outstanding",
          "priority": "low | normal | high | urgent"}, ("reason", "summary"), escalate_to_human),
    Tool("resolve_case", "Close the case once the customer's issue is fully handled.",
         {"resolution_summary": "string"}, ("resolution_summary",), resolve_case),
]}

ALLOWED_WHEN_ESCALATED = frozenset({"get_order_history", "get_order_details", "check_return_eligibility",
                             "get_pickup_slots", "escalate_to_human"})


def execute_tool(ctx: ToolContext, name: str, args) -> dict:
    """Run a tool and return a JSON-serialisable observation. Never raises for tool-level problems."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool '{name}'.", "available_tools": list(TOOLS)}
    if not isinstance(args, dict):
        return {"ok": False, "error": "action_input must be a JSON object."}
    missing = [p for p in tool.required if args.get(p) in (None, "", [])]
    if missing:
        return {"ok": False, "error": f"Missing required argument(s): {', '.join(missing)}."}
    try:
        with ctx.store.transaction() as data:
            case = data["cases"][ctx.conversation_id]
            if case["status"] == "escalated" and name not in ALLOWED_WHEN_ESCALATED:
                raise ToolError("This case is with a human specialist; you cannot take further actions on it.")
            result = tool.fn(ctx, data, case, args)
            log_action(case, name, args, True, str(result))
        return {"ok": True, **result}
    except ToolError as exc:
        with ctx.store.transaction() as data:
            case = data["cases"][ctx.conversation_id]
            case.update(exc.case_updates)
            log_action(case, name, args, False, str(exc))
        return {"ok": False, "error": str(exc), **exc.extra}
