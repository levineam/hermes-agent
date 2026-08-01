"""Tests for the redacted, durable gateway lifecycle-event outbox."""

from __future__ import annotations

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
