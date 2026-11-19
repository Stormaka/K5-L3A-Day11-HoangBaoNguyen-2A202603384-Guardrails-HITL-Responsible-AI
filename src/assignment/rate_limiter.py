"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0
        self.last_wait_seconds = 0.0

    @staticmethod
    def _bucket_key(invocation_context) -> str:
        """Per-user bucket; fall back to the session id so anonymous callers do
        not all share (and exhaust) a single "anonymous" bucket."""
        user_id = getattr(invocation_context, "user_id", None)
        if user_id:
            return str(user_id)
        session = getattr(invocation_context, "session", None)
        session_id = getattr(session, "id", None)
        return f"session:{session_id}" if session_id else "anonymous"

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = self._bucket_key(invocation_context)
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Sliding window: drop timestamps that fell out of the window
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        # 2. Over budget → block WITHOUT calling the LLM (cost / flooding attack).
        #    Blocked requests are not recorded, so a spammer cannot extend
        #    their own lock-out forever — the window simply drains.
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            self.last_wait_seconds = max(0.0, wait)
            return self._block_response(
                f"Rate limit exceeded ({self.max_requests} requests / "
                f"{self.window_seconds}s). Try again in {max(1, round(wait))}s."
            )

        # 3. Within budget → record and let it through
        window.append(now)
        self.last_wait_seconds = 0.0
        return None

    def remaining(self, user_id: str = "anonymous") -> int:
        """Requests still available for ``user_id`` in the current window."""
        now = time.time()
        window = self.user_windows[user_id]
        while window and window[0] <= now - self.window_seconds:
            window.popleft()
        return max(0, self.max_requests - len(window))
