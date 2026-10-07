"""FastAPI server. Start with `python run.py` (or `uvicorn app.main:create_app --factory`)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agent import ConversationConflict, ResolutionAgent
from .config import Settings
from .followups import deliver_due_follow_ups
from .llm import ChatModel, LLMUnavailable, OpenAICompatibleLLM
from .memory import ExcelMemory
from .schemas import ChatRequest, ChatResponse, HumanReply
from .store import StateStore, iso, utcnow

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, llm: ChatModel | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = StateStore(settings.state_path, settings.seed_path)
    memory = ExcelMemory(settings.memory_path, settings.memory_max_rows)
    llm = llm or OpenAICompatibleLLM(settings)
    agent = ResolutionAgent(llm, store, memory, settings)

    async def follow_up_loop():
        while True:
            await asyncio.sleep(settings.follow_up_poll_seconds)
            try:
                sent = await asyncio.to_thread(deliver_due_follow_ups, store, memory)
                if sent:
                    log.info("Delivered %d follow-up(s)", len(sent))
            except Exception:
                log.exception("Follow-up delivery failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(follow_up_loop())
        yield
        task.cancel()

    app = FastAPI(title="Cosmic Mart End-to-End Resolution Agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["*"], allow_headers=["*"])
    app.state.settings, app.state.store, app.state.memory, app.state.agent = settings, store, memory, agent

    @app.exception_handler(LLMUnavailable)
    async def _llm_down(_: Request, exc: LLMUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    # ---------------------------------------------------------------- health
    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok", "llm_base_url": settings.openai_base_url, "llm_model": settings.llm_model,
                "authority_limit": settings.return_authority_limit, "currency": settings.currency,
                "memory_rows": len(memory.all_rows()), "memory_max_rows": settings.memory_max_rows}

    @app.get("/health/llm", tags=["system"])
    def health_llm():
        if not isinstance(llm, OpenAICompatibleLLM):
            return {"status": "ok", "models": []}
        return {"status": "ok", "models": llm.list_models()}

    # ---------------------------------------------------------------- chat
    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(req: ChatRequest):
        try:
            return agent.handle_message(req.message, req.conversation_id, req.customer_id).as_dict()
        except ConversationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/conversations/{conversation_id}", tags=["chat"])
    def conversation(conversation_id: str):
        """Messages visible to the customer (poll this to receive follow-ups and human replies)."""
        with store.read() as data:
            case = data["cases"].get(conversation_id)
        if case is None:
            raise HTTPException(404, "Conversation not found")
        messages = [{"timestamp": r["timestamp"], "role": r["role"], "kind": r["tool_name"] or "message",
                     "content": r["content"]}
                    for r in memory.history(conversation_id) if r["role"] != "tool"]
        return {"conversation_id": conversation_id, "case_ref": case["case_ref"], "case_status": case["status"],
                "handoff_id": case.get("handoff_id"), "messages": messages}

    # ---------------------------------------------------------------- cases & data
    @app.get("/cases", tags=["cases"])
    def list_cases(status: str | None = None):
        with store.read() as data:
            cases = list(data["cases"].values())
        return [c for c in cases if status is None or c["status"] == status]

    @app.get("/cases/{conversation_id}", tags=["cases"])
    def get_case(conversation_id: str):
        with store.read() as data:
            case = data["cases"].get(conversation_id)
            if case is None:
                raise HTTPException(404, "Case not found")
            case["returns"] = [data["returns"][r] for r in case.get("return_ids", [])]
            case["pickups"] = [data["pickups"][p] for p in case.get("pickup_ids", [])]
            case["refunds"] = [data["refunds"][r] for r in case.get("refund_ids", [])]
            case["follow_ups"] = [data["follow_ups"][f] for f in case.get("follow_up_ids", [])]
        return case

    @app.get("/customers", tags=["data"])
    def list_customers():
        with store.read() as data:
            return list(data["customers"].values())

    @app.get("/customers/{customer_id}/orders", tags=["data"])
    def customer_orders(customer_id: str):
        with store.read() as data:
            if customer_id not in data["customers"]:
                raise HTTPException(404, "Customer not found")
            return [o for o in data["orders"].values() if o["customer_id"] == customer_id]

    @app.get("/memory", tags=["data"])
    def memory_rows(conversation_id: str | None = None):
        return memory.history(conversation_id) if conversation_id else memory.all_rows()

    # ---------------------------------------------------------------- human handoffs
    @app.get("/handoffs", tags=["handoffs"])
    def list_handoffs(status: str | None = None):
        with store.read() as data:
            handoffs = list(data["handoffs"].values())
        return [{k: h[k] for k in ("handoff_id", "case_ref", "conversation_id", "status", "priority",
                                   "reason", "created_at")}
                for h in handoffs if status is None or h["status"] == status]

    @app.get("/handoffs/{handoff_id}", tags=["handoffs"])
    def get_handoff(handoff_id: str):
        """Full context package for the human specialist."""
        with store.read() as data:
            handoff = data["handoffs"].get(handoff_id)
        if handoff is None:
            raise HTTPException(404, "Handoff not found")
        handoff["live_transcript"] = [r for r in memory.history(handoff["conversation_id"])]
        return handoff

    @app.post("/handoffs/{handoff_id}/reply", tags=["handoffs"])
    def reply_to_handoff(handoff_id: str, body: HumanReply):
        """A human specialist replies to the customer in the same conversation."""
        with store.transaction() as data:
            handoff = data["handoffs"].get(handoff_id)
            if handoff is None:
                raise HTTPException(404, "Handoff not found")
            handoff["status"] = "resolved" if body.close_case else "in_progress"
            handoff["human_replies"].append({"at": iso(utcnow()), "agent_name": body.agent_name,
                                             "message": body.message})
            case = data["cases"][handoff["conversation_id"]]
            if body.close_case:
                case["status"] = "resolved"
                case["resolution"] = f"Resolved by {body.agent_name}"
                case["resolved_at"] = iso(utcnow())
            status, customer_id = case["status"], case.get("customer_id")
        memory.append(handoff["conversation_id"], "human_agent", f"{body.agent_name}: {body.message}",
                      customer_id=customer_id, case_status=status)
        return {"handoff_id": handoff_id, "handoff_status": handoff["status"], "case_status": status}

    # ---------------------------------------------------------------- follow-ups / admin
    @app.post("/follow-ups/run", tags=["admin"])
    def run_follow_ups(force: bool = False):
        """Deliver due follow-ups now. `force=true` sends every pending one (handy for demos)."""
        return {"delivered": deliver_due_follow_ups(store, memory, force=force)}

    @app.post("/admin/reset", tags=["admin"])
    def reset_state():
        """Restore orders from seed data and clear cases/returns/handoffs (memory.xlsx is left as is)."""
        store.reset()
        return {"status": "reset"}

    return app
