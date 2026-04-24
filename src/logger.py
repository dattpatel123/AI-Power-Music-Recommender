import json
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "logs" / "sessions.json"


def log_session(entry: dict) -> None:
    """
    Append a session entry to logs/sessions.json.
    The file is kept as a JSON array; each call appends one object.
    """
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()

    if _LOG_PATH.exists() and _LOG_PATH.stat().st_size > 0:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    else:
        sessions = []

    sessions.append(entry)

    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)


def log_event(event_type: str, detail: dict) -> None:
    """
    Append a lightweight event (clamp, retry, error) to sessions.json as its
    own top-level entry.  Used by guardrails to record every clamp/retry
    without requiring a full session object.
    """
    log_session({"event": event_type, **detail})
