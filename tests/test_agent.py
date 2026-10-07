from app.followups import deliver_due_follow_ups
from tests.conftest import last_observation


def act(action, **args):
    return {"thought": f"call {action}", "action": action, "action_input": args}


def final(text):
    return {"thought": "reply", "final_answer": text}


def book_first_slot(messages):
    slots = last_observation(messages)["available"]
    day, times = next(iter(slots.items()))
    return act("book_pickup", return_id="RMA-00001", date=day[:10], slot=times[0])


def test_end_to_end_return_refund_and_follow_up(agent, llm, store, memory):
    llm.push(act("check_return_eligibility", order_id="CM-10001"),
             final("Your headphones are eligible for an 89.90 refund. Shall I go ahead?"))
    first = agent.handle_message("I want to return my headphones", customer_id="C001")
    assert first.actions[0]["result"]["within_authority"] is True

    llm.push(act("approve_return", order_id="CM-10001", reason="No longer needed"),
             act("get_pickup_slots", return_id="RMA-00001"),
             book_first_slot,
             act("issue_refund", return_id="RMA-00001"),
             act("schedule_follow_up", message="Did your refund arrive?", delay_hours=48),
             act("resolve_case", resolution_summary="Return, pickup and refund completed"),
             final("All done: pickup booked and refund RF-00001 issued."))
    second = agent.handle_message("Yes please, the first slot is fine", conversation_id=first.conversation_id)

    assert [a["result"]["ok"] for a in second.actions] == [True] * 6
    assert second.case_status == "resolved" and not second.escalated
    with store.read() as data:
        assert data["returns"]["RMA-00001"]["status"] == "refunded"
        assert data["refunds"]["RF-00001"]["amount"] == 89.90
        assert data["orders"]["CM-10001"]["status"] == "returned"

    sent = deliver_due_follow_ups(store, memory, force=True)
    assert len(sent) == 1
    assert memory.history(first.conversation_id)[-1]["tool_name"] == "follow_up"


def test_over_authority_limit_is_handed_off_even_if_model_does_not(agent, llm, store):
    llm.push(act("approve_return", order_id="CM-10002", reason="Too big"),
             final("I've approved your TV return!"))  # model ignores requires_human
    result = agent.handle_message("Return my TV please", customer_id="C001")

    assert result.actions[0]["result"]["requires_human"] is True
    assert result.escalated and result.handoff_id == "HO-00001"
    assert "HO-00001" in result.reply and "approved" not in result.reply
    with store.read() as data:
        handoff = data["handoffs"]["HO-00001"]
        assert data["returns"] == {}
    assert handoff["customer"]["customer_id"] == "C001"
    assert any(t["role"] == "user" for t in handoff["transcript"])


def test_frustration_triggers_handoff_with_context(agent, llm, store):
    llm.push(final("Let me look into that for you."))
    result = agent.handle_message("This is RIDICULOUS!! I've already told you twice. Get me a real person.",
                                  customer_id="C002")
    assert result.frustration["requests_human"] is True
    assert result.escalated
    assert "ALERT" in llm.calls[0][0]["content"]  # model was told to escalate
    with store.read() as data:
        assert data["handoffs"][result.handoff_id]["priority"] == "high"

    # After the handoff the agent can no longer change orders.
    llm.push(act("approve_return", order_id="CM-10004", items=[{"item_id": "CM-10004-2"}], reason="x"),
             final("A specialist will reply here."))
    after = agent.handle_message("ok, any update?", conversation_id=result.conversation_id)
    assert after.actions[0]["result"]["ok"] is False


def test_agent_cannot_see_other_customers_orders(agent, llm):
    llm.push(act("get_order_details", order_id="CM-10001"), final("I couldn't find that order."))
    result = agent.handle_message("What's in order CM-10001?", customer_id="C002")
    assert result.actions[0]["result"]["ok"] is False


def test_verification_flow_and_non_returnable_items(agent, llm):
    llm.push(act("verify_customer", email="BEN.ORTIZ@example.com", order_id="cm-10004"),
             act("check_return_eligibility", order_id="CM-10004"),
             final("The phone case can be returned; gift cards cannot."))
    result = agent.handle_message("I want to return order CM-10004, I'm ben.ortiz@example.com")
    verify, check = (a["result"] for a in result.actions)
    assert verify["customer_id"] == "C002"
    eligible = {i["item_id"]: i["eligible"] for i in check["items"]}
    assert eligible == {"CM-10004-1": False, "CM-10004-2": True}
    assert check["eligible_refund_total"] == 19.99


def test_invalid_json_gets_one_retry_then_falls_back_to_text(agent, llm):
    llm.push("sure thing", "Hello! How can I help?")
    result = agent.handle_message("hi", customer_id="C003")
    assert result.reply == "Hello! How can I help?"
    assert "not a valid JSON" in llm.calls[1][-1]["content"]


def test_step_limit_escalates(agent, llm, settings):
    llm.push(*[act("get_order_history")] * settings.max_agent_steps)
    result = agent.handle_message("where are my orders", customer_id="C003")
    assert result.escalated and result.handoff_id
    assert len(result.actions) == settings.max_agent_steps
