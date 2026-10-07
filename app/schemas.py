from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(None, description="Omit to start a new conversation.")
    customer_id: str | None = Field(None, description="Set if the front end has already authenticated the customer.")


class ChatResponse(BaseModel):
    conversation_id: str
    case_ref: str
    reply: str
    case_status: str
    escalated: bool
    handoff_id: str | None
    frustration: dict[str, Any]
    actions: list[dict[str, Any]]


class HumanReply(BaseModel):
    agent_name: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=4000)
    close_case: bool = Field(False, description="Mark the handoff and case resolved after sending.")
