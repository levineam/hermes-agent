"""Durable, redacted lifecycle events for gateway integrations.

The gateway may know that a provider accepted a message before an integration
plugin has received the associated receipt.  This module records a small,
allowlisted event before notifying plugins, so a plugin failure or process
restart cannot cause a second provider send.  It deliberately stores IDs and
revision metadata only: message text, prompts, provider errors, credentials,
and arbitrary plugin data are not lifecycle evidence.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from hermes_constants import get_hermes_home


EVENT_VERSION = 1
EVENT_TYPES = frozenset({"delivery"})
_REQUIRED_FIELDS = frozenset(
    {
        "event_id",
        "event_type",
        "event_version",
        "occurred_at",
        "dispatch_id",
        "activation_attempt_id",
        "route_revision",
        "destination_revision",
        "plugin_revision",
        "expected_conversation_key",
        "provider_message_ids",
        "canonical_parent_message_id",
        "actual_session_id",
    }
)
_ALLOWED_FIELDS = _REQUIRED_FIELDS | frozenset({"gateway_revision"})
_LEASE_SECONDS = 60


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (lifecycle_events)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS gateway_lifecycle_events (
            event_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            leased_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    """Validate the public, redacted lifecycle event shape.

    Direct delivery cannot create a native conversation.  Its actual session
    therefore must remain explicitly unresolved until the native inbound path
    provides one through ``build_session_key``.
    """
    if not isinstance(event, dict):
        return {"ok": False, "errors": ["event must be an object"]}
    unknown = sorted(set(event) - _ALLOWED_FIELDS)
    missing = sorted(name for name in _REQUIRED_FIELDS if name not in event)
    errors = [f"unknown field: {name}" for name in unknown]
    errors.extend(f"missing field: {name}" for name in missing)
    if errors:
        return {"ok": False, "errors": errors}

    if event["event_type"] not in EVENT_TYPES:
        errors.append("unsupported event_type")
    if event["event_version"] != EVENT_VERSION:
        errors.append("unsupported event_version")
    for field in (
        "event_id",
        "dispatch_id",
        "activation_attempt_id",
        "route_revision",
        "destination_revision",
        "plugin_revision",
        "expected_conversation_key",
    ):
        if not isinstance(event[field], str) or not event[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(event["occurred_at"], (int, float)):
        errors.append("occurred_at must be numeric")
    provider_ids = event["provider_message_ids"]
    if not isinstance(provider_ids, list) or not provider_ids or any(
        not isinstance(value, str) or not value for value in provider_ids
    ):
        errors.append("provider_message_ids must be a non-empty string list")
    elif len(set(provider_ids)) != len(provider_ids):
        errors.append("provider_message_ids must be unique")
    if (
        not isinstance(event["canonical_parent_message_id"], str)
        or event["canonical_parent_message_id"] not in provider_ids
    ):
        errors.append("canonical_parent_message_id must identify a provider message")
    if event["actual_session_id"] is not None:
        errors.append("delivery actual_session_id must remain unresolved")
    return {"ok": not errors, "errors": errors}


def build_delivery_event(
    *,
    correlation: dict[str, Any],
    message_id: str | None,
    continuation_message_ids: tuple | list = (),
    gateway_revision: str | None = None,
) -> dict[str, Any]:
    """Create a redacted receipt event from an actual provider ``SendResult``."""
    provider_ids = [str(value) for value in continuation_message_ids if value]
    if message_id and str(message_id) not in provider_ids:
        provider_ids.append(str(message_id))
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "delivery",
        "event_version": EVENT_VERSION,
        "occurred_at": time.time(),
        "dispatch_id": correlation.get("dispatch_id"),
        "activation_attempt_id": correlation.get("activation_attempt_id"),
        "route_revision": correlation.get("route_revision"),
        "destination_revision": correlation.get("destination_revision"),
        "plugin_revision": correlation.get("plugin_revision"),
        "expected_conversation_key": correlation.get("expected_conversation_key"),
        "provider_message_ids": provider_ids,
        "canonical_parent_message_id": str(message_id) if message_id else None,
        "actual_session_id": None,
    }
    if gateway_revision:
        event["gateway_revision"] = gateway_revision
    return event


def enqueue(event: dict[str, Any]) -> dict[str, Any]:
    """Persist an event before any plugin notification is attempted."""
    validation = validate_event(event)
    if not validation["ok"]:
        return validation
    payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
    now = time.time()
    with _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO gateway_lifecycle_events
               (event_id, payload_json, state, attempts, leased_at, created_at, updated_at)
               VALUES (?, ?, 'pending', 0, NULL, ?, ?)""",
            (event["event_id"], payload, now, now),
        )
        row = conn.execute(
            "SELECT state FROM gateway_lifecycle_events WHERE event_id=?",
            (event["event_id"],),
        ).fetchone()
    return {"ok": True, "event_id": event["event_id"], "state": row[0]}


def drain(notify: Callable[[dict[str, Any]], bool]) -> int:
    """Deliver pending events once each, leaving failures replayable.

    Notification is intentionally outside the transaction.  If the process
    dies while the callback runs, the short lease expires and the same event
    (never a provider send) is replayed on the next drain.
    """
    now = time.time()
    with _transaction() as conn:
        conn.execute(
            "UPDATE gateway_lifecycle_events SET state='pending', leased_at=NULL, updated_at=? "
            "WHERE state='delivering' AND leased_at < ?",
            (now, now - _LEASE_SECONDS),
        )
        rows = conn.execute(
            "SELECT event_id, payload_json FROM gateway_lifecycle_events "
            "WHERE state='pending' ORDER BY created_at"
        ).fetchall()

    delivered = 0
    for event_id, payload_json in rows:
        with _transaction() as conn:
            claimed = conn.execute(
                "UPDATE gateway_lifecycle_events SET state='delivering', leased_at=?, attempts=attempts+1, updated_at=? "
                "WHERE event_id=? AND state='pending'",
                (time.time(), time.time(), event_id),
            ).rowcount
        if not claimed:
            continue
        try:
            accepted = notify(json.loads(payload_json))
        except Exception:
            accepted = False
        with _transaction() as conn:
            conn.execute(
                "UPDATE gateway_lifecycle_events SET state=?, leased_at=NULL, updated_at=? WHERE event_id=?",
                ("delivered" if accepted else "pending", time.time(), event_id),
            )
        delivered += int(bool(accepted))
    return delivered


def notify_plugins(event: dict[str, Any]) -> bool:
    """Offer an event to every registered lifecycle observer.

    No observer means the event stays pending: an operator explicitly enabled
    lifecycle receipts on this route, so silently discarding its evidence is
    less safe than replaying it once the intended plugin is available.
    """
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    if not manager.has_hook("gateway_lifecycle_event"):
        return False
    _results, accepted = manager.invoke_hook_with_status(
        "gateway_lifecycle_event", event=event
    )
    return accepted


def enqueue_and_notify(event: dict[str, Any]) -> dict[str, Any]:
    """Durably enqueue an event, then best-effort drain the plugin outbox."""
    result = enqueue(event)
    if result.get("ok"):
        drain(notify_plugins)
    return result
