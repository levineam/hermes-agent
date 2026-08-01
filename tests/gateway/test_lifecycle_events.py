"""Tests for the redacted, durable gateway lifecycle-event outbox."""

from __future__ import annotations

from types import SimpleNamespace

from gateway import lifecycle_events as events


def _correlation():
    return {
        "dispatch_id": "dispatch-1",
        "activation_attempt_id": "attempt-1",
        "route_revision": "route-r1",
        "destination_revision": "destination-r1",
        "plugin_revision": "plugin-r1",
        "expected_conversation_key": "agent:main:telegram:dm:chat-1",
    }


def _event():
    return events.build_delivery_event(
        correlation=_correlation(),
        continuation_message_ids=("telegram-1",),
        message_id="telegram-2",
    )


def test_delivery_event_carries_actual_provider_ids_without_a_session():
    event = _event()

    assert event["provider_message_ids"] == ["telegram-1", "telegram-2"]
    assert event["canonical_parent_message_id"] == "telegram-2"
    assert event["actual_session_id"] is None
    assert events.validate_event(event) == {"ok": True, "errors": []}


def test_rejects_unknown_or_raw_content_fields():
    event = _event()
    event["content"] = "must not persist"

    result = events.validate_event(event)

    assert result["ok"] is False
    assert "unknown field: content" in result["errors"]


def test_inbound_event_has_native_session_identity_but_no_raw_reply():
    event = events.build_inbound_event(
        provider_event_id="telegram-inbound-1",
        parent_message_id="telegram-parent-1",
        platform="telegram",
        chat_id="chat-1",
        thread_id=None,
        sender_id="user-1",
        gateway_profile="default",
        actual_session_id="session-1",
        actual_session_key="agent:main:telegram:dm:chat-1",
    )

    assert events.validate_event(event) == {"ok": True, "errors": []}
    event["reply"] = "must not persist"
    assert events.validate_event(event)["ok"] is False


def test_outbound_event_links_the_same_native_session_to_the_inbound_event():
    event = events.build_outbound_event(
        inbound_event_id="inbound-event-1",
        message_id="telegram-outbound-2",
        continuation_message_ids=("telegram-outbound-1",),
        platform="telegram",
        chat_id="chat-1",
        thread_id="topic-1",
        gateway_profile="default",
        actual_session_id="session-1",
        actual_session_key="agent:main:telegram:dm:chat-1:topic-1",
    )

    assert event["provider_message_ids"] == [
        "telegram-outbound-1",
        "telegram-outbound-2",
    ]
    assert events.validate_event(event) == {"ok": True, "errors": []}


def test_native_helpers_link_outbound_receipt_to_resolved_inbound_session(monkeypatch):
    persisted = []

    def _persist(event):
        persisted.append(event)
        return {"ok": True, "event_id": event["event_id"], "state": "pending"}

    monkeypatch.setattr(events, "enqueue_and_notify", _persist)
    monkeypatch.setattr(events, "lifecycle_observer_enabled", lambda: True)
    source = SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id="chat-1",
        thread_id=None,
        user_id="user-1",
        profile="default",
    )
    inbound = SimpleNamespace(
        source=source,
        message_id="telegram-inbound-1",
        reply_to_message_id="telegram-parent-1",
        metadata={},
    )

    inbound_result = events.record_inbound_from_event(
        inbound,
        actual_session_id="session-1",
        actual_session_key="agent:main:telegram:dm:chat-1",
    )
    outbound_result = events.record_outbound_from_result(
        inbound,
        SimpleNamespace(
            success=True,
            message_id="telegram-outbound-1",
            continuation_message_ids=(),
        ),
    )

    assert inbound_result["ok"] is True
    assert outbound_result["ok"] is True
    assert persisted[0]["event_type"] == "inbound"
    assert persisted[1]["event_type"] == "outbound"
    assert persisted[1]["causal_inbound_event_id"] == inbound_result["event_id"]
    assert persisted[1]["actual_session_id"] == "session-1"


def test_native_reply_is_not_persisted_without_an_explicit_observer(monkeypatch):
    monkeypatch.setattr(events, "lifecycle_observer_enabled", lambda: False)
    inbound = SimpleNamespace(
        source=SimpleNamespace(
            platform=SimpleNamespace(value="telegram"),
            chat_id="chat-1",
            thread_id=None,
            user_id="user-1",
            profile="default",
        ),
        message_id="telegram-inbound-1",
        reply_to_message_id="telegram-parent-1",
        metadata={},
    )

    result = events.record_inbound_from_event(
        inbound,
        actual_session_id="session-1",
        actual_session_key="agent:main:telegram:dm:chat-1",
    )

    assert result == {"ok": False, "skipped": "no_lifecycle_observer"}
    assert inbound.metadata == {}


def test_outbox_persists_before_notification_and_replays_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_db_path", lambda: tmp_path / "state.db")
    event = _event()
    assert events.enqueue(event)["state"] == "pending"

    seen = []
    assert events.drain(lambda payload: seen.append(payload) and False) == 0
    assert seen == [event]

    delivered = []
    assert events.drain(lambda payload: delivered.append(payload) or True) == 1
    assert delivered == [event]
    assert events.drain(lambda _payload: True) == 0


def test_inbound_provider_event_is_deduplicated_across_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_db_path", lambda: tmp_path / "state.db")
    first = events.build_inbound_event(
        provider_event_id="telegram-inbound-1",
        parent_message_id="telegram-parent-1",
        platform="telegram",
        chat_id="chat-1",
        thread_id=None,
        sender_id="user-1",
        gateway_profile="default",
        actual_session_id="session-1",
        actual_session_key="agent:main:telegram:dm:chat-1",
    )
    duplicate = {**first, "event_id": "new-event-id"}

    first_result = events.enqueue(first)
    duplicate_result = events.enqueue(duplicate)

    assert duplicate_result["event_id"] == first_result["event_id"]
