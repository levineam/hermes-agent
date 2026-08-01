"""Durable, redacted lifecycle events for gateway integrations.

The gateway may know that a provider accepted a message before an integration
plugin has received the associated receipt.  This module records a small,
allowlisted event before notifying plugins, so a plugin failure or process
restart cannot cause a second provider send.  It deliberately stores IDs and
revision metadata only: message text, prompts, provider errors, credentials,
and arbitrary plugin data are not lifecycle evidence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from hermes_constants import get_hermes_home


EVENT_VERSION = 1
EVENT_TYPES = frozenset({"delivery", "inbound", "outbound"})
_COMMON_REQUIRED_FIELDS = frozenset(
    {
        "event_id",
        "event_type",
        "event_version",
        "occurred_at",
    }
)
_DELIVERY_REQUIRED_FIELDS = frozenset(
    {
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
_INBOUND_REQUIRED_FIELDS = frozenset(
    {
        "provider_event_id",
        "parent_message_id",
        "platform",
        "chat_id",
        "thread_id",
        "sender_id",
        "gateway_profile",
        "actual_session_id",
        "actual_session_key",
    }
)
_OUTBOUND_REQUIRED_FIELDS = frozenset(
    {
        "provider_message_ids",
        "canonical_parent_message_id",
        "causal_inbound_event_id",
        "platform",
        "chat_id",
        "thread_id",
        "gateway_profile",
        "actual_session_id",
        "actual_session_key",
    }
)
_ALLOWED_FIELDS_BY_TYPE = {
    "delivery": _COMMON_REQUIRED_FIELDS | _DELIVERY_REQUIRED_FIELDS | frozenset({"gateway_revision"}),
    "inbound": _COMMON_REQUIRED_FIELDS | _INBOUND_REQUIRED_FIELDS | frozenset({"gateway_revision"}),
    "outbound": _COMMON_REQUIRED_FIELDS | _OUTBOUND_REQUIRED_FIELDS | frozenset({"gateway_revision"}),
}
_LEASE_SECONDS = 60


def is_opaque_correlation_id(value: Any) -> bool:
    """Return true only for a fixed-format digest safe to persist as an ID.

    Lifecycle correlation comes from an authenticated request body, but it is
    still untrusted input.  Requiring an exact SHA-256 hex digest prevents a
    raw prompt, credential, or provider error from being relabeled as an ID
    and copied into the durable outbox or an observer plugin.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


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
            dedupe_key TEXT,
            payload_json TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            leased_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(gateway_lifecycle_events)")
    }
    if "dedupe_key" not in columns:
        conn.execute("ALTER TABLE gateway_lifecycle_events ADD COLUMN dedupe_key TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS gateway_lifecycle_events_dedupe_key "
        "ON gateway_lifecycle_events(dedupe_key) WHERE dedupe_key IS NOT NULL"
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
    event_type = event.get("event_type")
    allowed = _ALLOWED_FIELDS_BY_TYPE.get(event_type, _COMMON_REQUIRED_FIELDS)
    required = allowed - {"gateway_revision"}
    unknown = sorted(set(event) - allowed)
    missing = sorted(name for name in required if name not in event)
    errors = [f"unknown field: {name}" for name in unknown]
    errors.extend(f"missing field: {name}" for name in missing)
    if event_type not in EVENT_TYPES:
        errors.append("unsupported event_type")
    if errors:
        return {"ok": False, "errors": errors}
    if event["event_version"] != EVENT_VERSION:
        errors.append("unsupported event_version")
    for field in ("event_id",):
        if not isinstance(event[field], str) or not event[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(event["occurred_at"], (int, float)):
        errors.append("occurred_at must be numeric")
    if event_type == "delivery":
        for field in (
            "dispatch_id",
            "activation_attempt_id",
            "route_revision",
            "destination_revision",
            "plugin_revision",
            "expected_conversation_key",
        ):
            if not isinstance(event[field], str) or not event[field].strip():
                errors.append(f"{field} must be a non-empty string")
        for field in ("dispatch_id", "activation_attempt_id"):
            if not is_opaque_correlation_id(event[field]):
                errors.append(f"{field} must be a 64-character lowercase hexadecimal digest")
        _validate_provider_ids(event, errors)
        if event["actual_session_id"] is not None:
            errors.append("delivery actual_session_id must remain unresolved")
    elif event_type == "inbound":
        for field in (
            "provider_event_id",
            "parent_message_id",
            "platform",
            "chat_id",
            "sender_id",
            "actual_session_id",
            "actual_session_key",
        ):
            if not isinstance(event[field], str) or not event[field].strip():
                errors.append(f"{field} must be a non-empty string")
        for field in ("thread_id", "gateway_profile"):
            if event[field] is not None and not isinstance(event[field], str):
                errors.append(f"{field} must be a string or null")
    elif event_type == "outbound":
        for field in (
            "causal_inbound_event_id",
            "platform",
            "chat_id",
            "actual_session_id",
            "actual_session_key",
        ):
            if not isinstance(event[field], str) or not event[field].strip():
                errors.append(f"{field} must be a non-empty string")
        for field in ("thread_id", "gateway_profile"):
            if event[field] is not None and not isinstance(event[field], str):
                errors.append(f"{field} must be a string or null")
        _validate_provider_ids(event, errors)
    return {"ok": not errors, "errors": errors}


def _validate_provider_ids(event: dict[str, Any], errors: list[str]) -> None:
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


def build_inbound_event(
    *,
    provider_event_id: str,
    parent_message_id: str,
    platform: str,
    chat_id: str,
    thread_id: str | None,
    sender_id: str,
    gateway_profile: str | None,
    actual_session_id: str,
    actual_session_key: str,
    gateway_revision: str | None = None,
) -> dict[str, Any]:
    """Build a redacted native inbound event after session resolution."""
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "inbound",
        "event_version": EVENT_VERSION,
        "occurred_at": time.time(),
        "provider_event_id": provider_event_id,
        "parent_message_id": parent_message_id,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "sender_id": sender_id,
        "gateway_profile": gateway_profile,
        "actual_session_id": actual_session_id,
        "actual_session_key": actual_session_key,
    }
    if gateway_revision:
        event["gateway_revision"] = gateway_revision
    return event


def build_outbound_event(
    *,
    inbound_event_id: str,
    message_id: str | None,
    continuation_message_ids: tuple | list = (),
    platform: str,
    chat_id: str,
    thread_id: str | None,
    gateway_profile: str | None,
    actual_session_id: str,
    actual_session_key: str,
    gateway_revision: str | None = None,
) -> dict[str, Any]:
    """Build a redacted native outbound receipt causally tied to an inbound event."""
    provider_ids = [str(value) for value in continuation_message_ids if value]
    if message_id and str(message_id) not in provider_ids:
        provider_ids.append(str(message_id))
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "outbound",
        "event_version": EVENT_VERSION,
        "occurred_at": time.time(),
        "provider_message_ids": provider_ids,
        "canonical_parent_message_id": str(message_id) if message_id else None,
        "causal_inbound_event_id": inbound_event_id,
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "gateway_profile": gateway_profile,
        "actual_session_id": actual_session_id,
        "actual_session_key": actual_session_key,
    }
    if gateway_revision:
        event["gateway_revision"] = gateway_revision
    return event


def record_inbound_from_event(
    message_event: Any,
    *,
    actual_session_id: str,
    actual_session_key: str,
) -> dict[str, Any]:
    """Persist an exact native reply after Hermes has resolved its session.

    The caller deliberately supplies the resolved session values from the
    gateway's normal session store.  This helper never constructs or predicts a
    session identity from a transport event.
    """
    source = getattr(message_event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None)
    provider_event_id = getattr(message_event, "message_id", None)
    parent_message_id = getattr(message_event, "reply_to_message_id", None)
    sender_id = getattr(source, "user_id", None)
    chat_id = getattr(source, "chat_id", None)
    if not all((platform, provider_event_id, parent_message_id, sender_id, chat_id)):
        return {"ok": False, "skipped": "not_an_exact_native_reply"}
    if not lifecycle_observer_enabled():
        return {"ok": False, "skipped": "no_lifecycle_observer"}
    event = build_inbound_event(
        provider_event_id=str(provider_event_id),
        parent_message_id=str(parent_message_id),
        platform=str(platform),
        chat_id=str(chat_id),
        thread_id=(str(source.thread_id) if getattr(source, "thread_id", None) is not None else None),
        sender_id=str(sender_id),
        gateway_profile=(str(source.profile) if getattr(source, "profile", None) is not None else None),
        actual_session_id=str(actual_session_id),
        actual_session_key=str(actual_session_key),
    )
    result = enqueue_and_notify(event)
    if result.get("ok"):
        metadata = getattr(message_event, "metadata", None)
        if isinstance(metadata, dict):
            metadata["gateway_lifecycle_inbound_event_id"] = result["event_id"]
            metadata["gateway_lifecycle_actual_session_id"] = str(actual_session_id)
            metadata["gateway_lifecycle_actual_session_key"] = str(actual_session_key)
    return result


def record_outbound_from_result(message_event: Any, send_result: Any) -> dict[str, Any]:
    """Persist an outbound receipt only for a lifecycle-marked native reply."""
    metadata = getattr(message_event, "metadata", None)
    source = getattr(message_event, "source", None)
    if not isinstance(metadata, dict) or source is None:
        return {"ok": False, "skipped": "no_lifecycle_context"}
    inbound_event_id = metadata.get("gateway_lifecycle_inbound_event_id")
    actual_session_id = metadata.get("gateway_lifecycle_actual_session_id")
    actual_session_key = metadata.get("gateway_lifecycle_actual_session_key")
    platform = getattr(getattr(source, "platform", None), "value", None)
    chat_id = getattr(source, "chat_id", None)
    message_id = getattr(send_result, "message_id", None)
    if not (
        getattr(send_result, "success", False)
        and inbound_event_id
        and actual_session_id
        and actual_session_key
        and platform
        and chat_id
        and message_id
    ):
        return {"ok": False, "skipped": "no_authoritative_outbound_receipt"}
    return enqueue_and_notify(
        build_outbound_event(
            inbound_event_id=str(inbound_event_id),
            message_id=str(message_id),
            continuation_message_ids=getattr(send_result, "continuation_message_ids", ()) or (),
            platform=str(platform),
            chat_id=str(chat_id),
            thread_id=(str(source.thread_id) if getattr(source, "thread_id", None) is not None else None),
            gateway_profile=(str(source.profile) if getattr(source, "profile", None) is not None else None),
            actual_session_id=str(actual_session_id),
            actual_session_key=str(actual_session_key),
        )
    )


def _dedupe_key(event: dict[str, Any]) -> str:
    """Return a deterministic, unambiguous provider-identity key.

    The key is an internal index only.  Canonical JSON prevents delimiter
    ambiguity between provider identifiers, and hashing keeps the SQLite index
    bounded without adding an internal field to a signed receipt payload.
    """
    if event["event_type"] == "delivery":
        identity = (
            event["event_type"],
            event["dispatch_id"],
            event["activation_attempt_id"],
            event["canonical_parent_message_id"],
        )
    elif event["event_type"] == "inbound":
        identity = (
            event["event_type"],
            event["platform"],
            event["gateway_profile"],
            event["chat_id"],
            event["provider_event_id"],
        )
    else:
        identity = (
            event["event_type"],
            event["platform"],
            event["gateway_profile"],
            event["chat_id"],
            event["canonical_parent_message_id"],
            event["causal_inbound_event_id"],
        )
    encoded = json.dumps(identity, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def enqueue(event: dict[str, Any]) -> dict[str, Any]:
    """Persist an event before any plugin notification is attempted."""
    validation = validate_event(event)
    if not validation["ok"]:
        return validation
    payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
    dedupe_key = _dedupe_key(event)
    now = time.time()
    with _transaction() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO gateway_lifecycle_events
               (event_id, dedupe_key, payload_json, state, attempts, leased_at, created_at, updated_at)
               VALUES (?, ?, ?, 'pending', 0, NULL, ?, ?)""",
            (event["event_id"], dedupe_key, payload, now, now),
        )
        row = conn.execute(
            "SELECT event_id, state FROM gateway_lifecycle_events WHERE dedupe_key=?",
            (dedupe_key,),
        ).fetchone()
    return {"ok": True, "event_id": row[0], "state": row[1]}


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


def lifecycle_observer_enabled() -> bool:
    """Return whether a native lifecycle observer is explicitly installed."""
    from hermes_cli.plugins import get_plugin_manager

    return get_plugin_manager().has_hook("gateway_lifecycle_event")


def enqueue_and_notify(event: dict[str, Any]) -> dict[str, Any]:
    """Durably enqueue an event, then best-effort drain the plugin outbox."""
    result = enqueue(event)
    if result.get("ok"):
        drain(notify_plugins)
    return result
