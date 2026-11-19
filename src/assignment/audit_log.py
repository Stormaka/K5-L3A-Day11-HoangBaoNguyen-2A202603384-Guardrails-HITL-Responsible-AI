"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.

Thiết kế:
  * Mỗi request có ``request_id`` (tự sinh nếu không truyền) → ghép input/output.
  * Văn bản được che PII/secret trước khi ghi (log cũng là một kênh rò rỉ).
    Để không mất bằng chứng điều tra, mỗi bản ghi giữ thêm ``input_sha256``
    (so khớp với bản gốc mà không lưu bản gốc), và các trường kỹ thuật như
    ``destination`` của egress được ghi nguyên văn qua ``**extra``.
  * Che PII lỗi → ghi ``[SCRUB_FAILED]`` (fail-closed), không ghi bản thô.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCRUB_FAILED = "[SCRUB_FAILED]"


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}          # request_id → start time
        self._pending: dict[str, dict] = {}        # request_id → partial entry
        self._last_by_user: dict[str, str] = {}    # user_id → latest open request_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id.

        Returns the request_id so callers can correlate input/output.
        """
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = time.perf_counter()
        self._last_by_user[user_id] = request_id
        self._pending[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": _scrub(text),
            "input_sha256": _sha256(text),
            "input_chars": len(text or ""),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        **extra,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs.

        Without ``request_id`` the user's most recent open request is closed,
        so input, latency and output still end up in one record.
        """
        if request_id is None:
            request_id = self._last_by_user.get(user_id)
        if request_id and self._last_by_user.get(user_id) == request_id:
            self._last_by_user.pop(user_id, None)

        started = self._open.pop(request_id, None) if request_id else None
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 2) if started is not None else None
        )
        entry = self._pending.pop(request_id, None) if request_id else None
        if entry is None:  # output without a recorded input — still log it
            entry = {
                "request_id": request_id or uuid.uuid4().hex[:12],
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input": None,
                "orphan_output": True,
            }
        entry.update(
            {
                "output": _scrub(text),
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
                "completed_at": utc_now_iso(),
            }
        )
        for k, v in extra.items():
            if v is not None:
                entry[k] = v
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def _sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _scrub(text: str | None) -> str | None:
    """Redact PII / secrets before persisting (reuse the CP2 output filter)."""
    if text is None:
        return None
    try:
        from guardrails.output_guardrails import content_filter

        return content_filter(text)["redacted"]
    except Exception:  # fail-closed: never persist raw text we could not scrub
        return SCRUB_FAILED


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
