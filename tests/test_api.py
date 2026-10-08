from fastapi.testclient import TestClient

from app.llm import LLMUnavailable
from app.main import create_app
from tests.conftest import ScriptedLLM


def test_chat_handoff_and_human_reply(settings):
    llm = ScriptedLLM(
        {"thought": "", "action": "escalate_to_human",
         "action_input": {"reason": "Customer requested a human", "summary": "Wants help with order CM-10008"}},
        {"thought": "", "final_answer": "A specialist has your case (HO-00001)."},
    )
    with TestClient(create_app(settings, llm)) as client:
        assert client.get("/health").json()["memory_max_rows"] == 50

        res = client.post("/chat", json={"message": "speak to a manager please", "customer_id": "C004"})
        assert res.status_code == 200
        body = res.json()
        assert body["escalated"] and body["handoff_id"] == "HO-00001"

        handoff = client.get("/handoffs/HO-00001").json()
        assert handoff["customer"]["name"] == "Dev Patel" and handoff["live_transcript"]

        reply = client.post("/handoffs/HO-00001/reply",
                            json={"agent_name": "Sam", "message": "Hi Dev, I'm on it.", "close_case": True})
        assert reply.json()["case_status"] == "resolved"

        convo = client.get(f"/conversations/{body['conversation_id']}").json()
        assert [m["role"] for m in convo["messages"]] == ["user", "assistant", "human_agent"]
        assert convo["messages"][1]["message_id"] == body["message_id"]

        # Polling with after=<reply's message_id> returns only what arrived since.
        new = client.get(f"/conversations/{body['conversation_id']}", params={"after": body["message_id"]}).json()
        assert [m["content"] for m in new["messages"]] == ["Sam: Hi Dev, I'm on it."]

        assert client.post("/chat", json={"message": "hi", "conversation_id": body["conversation_id"],
                                          "customer_id": "C001"}).status_code == 409


def test_llm_offline_returns_503(settings):
    class DownLLM:
        def complete(self, messages):
            raise LLMUnavailable("connection refused")

    with TestClient(create_app(settings, DownLLM())) as client:
        res = client.post("/chat", json={"message": "hello", "customer_id": "C001"})
        assert res.status_code == 503 and "connection refused" in res.json()["detail"]


def test_root_serves_dashboard(settings):
    with TestClient(create_app(settings, ScriptedLLM())) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/html")
        assert "const API" in res.text
