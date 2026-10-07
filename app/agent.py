"""The end-to-end resolution agent: a ReAct-style loop over the prompt-based JSON protocol.

Guardrails that must not depend on the model are enforced here:
  * frustration / "I want a human" -> guaranteed handoff, even if the model forgets;
  * a tool reporting `requires_human` (e.g. over the authority limit) -> guaranteed handoff;
  * the model running out of steps or failing to produce a reply -> handoff instead of a dead end.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field

from .config import Settings
from .frustration import FrustrationResult, detect_frustration
from .llm import ChatModel
from .memory import ExcelMemory
from .prompts import ALREADY_ESCALATED_ALERT, ESCALATE_ALERT, build_system_prompt
from .protocol import format_final, parse_agent_output
from .store import StateStore, iso, utcnow
from .tools import ToolContext, create_handoff, execute_tool, log_action

log = logging.getLogger(__name__)

FORMAT_REMINDER = (
    "Your last reply was not a valid JSON object in the required format. Reply again with exactly one JSON "
    'object: either {"thought": "...", "action": "<tool>", "action_input": {...}} or '
    '{"thought": "...", "final_answer": "..."}'
)
HANDOFF_REPLY = (
    "I'm sorry for the trouble. I've passed your case to one of our human specialists (reference {handoff_id}) "
    "together with everything we've discussed and done so far, so you won't need to repeat yourself. "
    "They'll pick it up and reply to you right here."
)
ESCALATED_FALLBACK = ("Your case is with one of our human specialists (reference {handoff_id}). "
                      "They have the full history and will reply to you here.")


class ConversationConflict(Exception):
    pass


@dataclass
class TurnResult:
    conversation_id: str
    case_ref: str
    reply: str
    message_id: int
    case_status: str
    escalated: bool
    handoff_id: str | None
    frustration: dict
    actions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class ResolutionAgent:
    def __init__(self, llm: ChatModel, store: StateStore, memory: ExcelMemory, settings: Settings):
        self.llm = llm
        self.store = store
        self.memory = memory
        self.settings = settings
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------ public API
    def handle_message(self, message: str, conversation_id: str | None = None,
                       customer_id: str | None = None) -> TurnResult:
        conversation_id = conversation_id or uuid.uuid4().hex[:12]
        with self._lock_for(conversation_id):
            return self._handle(conversation_id, message.strip(), customer_id)

    # ------------------------------------------------------------------ the loop
    def _handle(self, conversation_id: str, message: str, customer_id: str | None) -> TurnResult:
        case = self._ensure_case(conversation_id, customer_id)
        previous = [r["content"] for r in self.memory.history(conversation_id) if r["role"] == "user"]
        frustration = detect_frustration(message, previous)
        self._record_frustration(conversation_id, frustration)
        self._log(conversation_id, "user", message)

        already_escalated = case["status"] == "escalated"
        must_escalate = None
        if already_escalated:
            alert = ALREADY_ESCALATED_ALERT.format(handoff_id=case["handoff_id"])
        elif frustration.should_escalate(self.settings.frustration_threshold):
            why = ("asked to speak to a human" if frustration.requests_human
                   else f"is frustrated ({', '.join(frustration.signals)})")
            must_escalate = f"Customer {why}."
            alert = ESCALATE_ALERT.format(why=why)
        else:
            alert = ""

        ctx = ToolContext(self.store, self.memory, self.settings, conversation_id)
        messages = self._build_messages(conversation_id, message, alert)
        actions: list[dict] = []
        reply: str | None = None
        format_retries = 0
        over_limit: dict[str, str] = {}  # order_id -> why, from the latest eligibility check of that order

        for _ in range(self.settings.max_agent_steps):
            raw = self.llm.complete(messages)
            step = parse_agent_output(raw)
            if step is None:
                if format_retries == 0:
                    format_retries += 1
                    messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": FORMAT_REMINDER}]
                    continue
                reply = raw.strip() or None  # model ignored the protocol: treat plain text as the reply
                break
            if step.final_answer is not None:
                reply = step.final_answer
                break

            observation = execute_tool(ctx, step.action, step.action_input)
            actions.append({"tool": step.action, "input": step.action_input, "result": observation})
            self._log(conversation_id, "tool",
                      json.dumps({"thought": step.thought, "input": step.action_input, "result": observation},
                                 ensure_ascii=False, default=str),
                      tool_name=step.action)
            if observation.get("requires_human") and not must_escalate:
                must_escalate = f"Agent reached a limit: {observation.get('error')}"
            self._track_authority(step.action, observation, over_limit)
            messages += [
                {"role": "assistant", "content": json.dumps(
                    {"thought": step.thought, "action": step.action, "action_input": step.action_input},
                    ensure_ascii=False)},
                {"role": "user", "content": f"OBSERVATION from {step.action}:\n"
                                            f"{json.dumps(observation, ensure_ascii=False, default=str)}\n"
                                            "Respond with one JSON object."},
            ]
        else:
            if not already_escalated and not must_escalate:
                must_escalate = "Agent could not complete the request within its step limit."

        if over_limit and not must_escalate and not already_escalated:
            must_escalate = "Agent reached a limit: " + " ".join(over_limit.values())

        with self.store.read() as data:
            case = data["cases"][conversation_id]

        if must_escalate and not case.get("handoff_id"):
            # The model did not hand off when it had to: do it deterministically.
            handoff_id = self._force_handoff(ctx, must_escalate, message, frustration)
            reply = HANDOFF_REPLY.format(handoff_id=handoff_id)
        elif reply is None:
            reply = (ESCALATED_FALLBACK.format(handoff_id=case["handoff_id"]) if case.get("handoff_id") else
                     "Sorry, I didn't quite catch that. Could you tell me a bit more about what you need?")

        with self.store.transaction() as data:
            case = data["cases"][conversation_id]
            case["last_reply_at"] = iso(utcnow())
            case = dict(case)
        reply_row = self._log(conversation_id, "assistant", reply)

        return TurnResult(
            conversation_id=conversation_id,
            case_ref=case["case_ref"],
            reply=reply,
            message_id=reply_row["row_id"],
            case_status=case["status"],
            escalated=case["status"] == "escalated",
            handoff_id=case.get("handoff_id"),
            frustration=frustration.as_dict(),
            actions=actions,
        )

    # ------------------------------------------------------------------ helpers
    def _lock_for(self, conversation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(conversation_id, threading.Lock())

    def _ensure_case(self, conversation_id: str, customer_id: str | None) -> dict:
        with self.store.transaction() as data:
            if customer_id and customer_id not in data["customers"]:
                raise ConversationConflict(f"Unknown customer_id '{customer_id}'.")
            case = data["cases"].get(conversation_id)
            if case is None:
                case = {
                    "case_ref": self.store.next_id(data, "CASE"),
                    "conversation_id": conversation_id,
                    "customer_id": customer_id,
                    "status": "open",
                    "created_at": iso(utcnow()),
                    "updated_at": iso(utcnow()),
                    "approved_total": 0.0,
                    "frustration": {"latest": 0.0, "peak": 0.0, "signals": []},
                    "actions": [],
                    "return_ids": [], "pickup_ids": [], "refund_ids": [], "follow_up_ids": [],
                    "handoff_id": None,
                }
                data["cases"][conversation_id] = case
            elif customer_id and case["customer_id"] and case["customer_id"] != customer_id:
                raise ConversationConflict("This conversation belongs to a different customer.")
            elif customer_id and not case["customer_id"]:
                case["customer_id"] = customer_id
            if case["status"] == "resolved":  # customer came back: reopen
                case["status"] = "open"
                if case.get("handoff_id"):  # that handoff is finished; allow a fresh one for the new issue
                    case.setdefault("past_handoff_ids", []).append(case["handoff_id"])
                    case["handoff_id"] = None
            return dict(case)

    @staticmethod
    def _track_authority(tool: str, observation: dict, over_limit: dict[str, str]) -> None:
        """Remember orders whose latest eligibility check came out above the agent's authority.

        A later check of the same order that fits (e.g. fewer items), or a successful approval,
        clears it, so returning only the cheaper item of an order does not trigger a handoff.
        """
        order_id = observation.get("order_id")
        if not observation.get("ok") or not order_id:
            return
        if tool == "check_return_eligibility":
            if observation["eligible_refund_total"] > 0 and not observation["within_authority"]:
                over_limit[order_id] = (f"Refund of {observation['eligible_refund_total']} for order {order_id} "
                                        f"exceeds remaining authority of {observation['agent_remaining_authority']}.")
            else:
                over_limit.pop(order_id, None)
        elif tool == "approve_return":
            over_limit.pop(order_id, None)

    def _record_frustration(self, conversation_id: str, result: FrustrationResult) -> None:
        with self.store.transaction() as data:
            f = data["cases"][conversation_id]["frustration"]
            f["latest"] = round(result.score, 2)
            f["peak"] = max(f["peak"], f["latest"])
            f["signals"] = (f["signals"] + result.signals)[-10:]

    def _force_handoff(self, ctx: ToolContext, reason: str, message: str, frustration: FrustrationResult) -> str:
        with self.store.transaction() as data:
            case = data["cases"][ctx.conversation_id]
            done = [f'{a["tool"]} ({"ok" if a["ok"] else "failed"})' for a in case.get("actions", [])]
            summary = (f"{reason} Latest customer message: \"{message}\". "
                       f"Actions taken so far: {', '.join(done) or 'none'}. "
                       "See transcript and actions_taken for full detail.")
            priority = "high" if frustration.should_escalate(self.settings.frustration_threshold) else "normal"
            result = create_handoff(ctx, data, case, reason=reason, summary=summary,
                                    priority=priority, source="guardrail")
            log_action(case, "escalate_to_human", {"reason": reason}, True, f"auto-escalated: {result['handoff_id']}")
        log.info("Guardrail escalated conversation %s: %s", ctx.conversation_id, reason)
        return result["handoff_id"]

    def _build_messages(self, conversation_id: str, current_message: str, alert: str) -> list[dict]:
        with self.store.read() as data:
            system = build_system_prompt(data["cases"][conversation_id], data, self.settings, alert)

        turns: list[list] = []  # [role, text] with consecutive same-role turns merged
        for row in self.memory.history(conversation_id)[-self.settings.history_messages:]:
            if row["role"] == "user":
                role, text = "user", row["content"]
            elif row["role"] == "assistant":
                role, text = "assistant", row["content"]
            elif row["role"] == "human_agent":
                role, text = "assistant", f"[Message from human specialist] {row['content']}"
            else:
                continue  # tool rows: summarised in the case record instead
            if turns and turns[-1][0] == role:
                turns[-1][1] += "\n\n" + str(text)
            else:
                turns.append([role, str(text)])
        while turns and turns[0][0] == "assistant":  # chat templates expect user first after system
            turns.pop(0)
        if not turns or turns[-1][0] != "user":
            turns.append(["user", current_message])

        messages = [{"role": "system", "content": system}]
        for role, text in turns:
            messages.append({"role": role, "content": format_final(text) if role == "assistant" else text})
        return messages

    def _log(self, conversation_id: str, role: str, content: str, tool_name: str | None = None) -> dict:
        with self.store.read() as data:
            case = data["cases"][conversation_id]
        return self.memory.append(conversation_id, role, content, customer_id=case.get("customer_id"),
                           tool_name=tool_name, case_status=case["status"])
