"""
Assignment 11 — Monitoring & Alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.

Thiết kế:
  * Bộ đếm cộng dồn (toàn phiên) để báo cáo, nhưng ALERT tính trên cửa sổ
    trượt ``window_seconds`` (mặc định 5 phút) → alert phản ánh tình hình
    hiện tại, không bị "kẹt" vì sự kiện cũ.
  * Rate-limit alert tính theo từng user (ai đang flood), kèm danh sách user.
  * Judge chỉ được tính khi đi qua ``record(judged=True, ...)`` từ traffic thật.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class _Event:
    ts: float
    user_id: str
    blocked: bool
    rate_limited: bool
    judged: bool
    judge_failed: bool


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters (cumulative) — updated by record() after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    # Breakdown: số lần chặn theo từng lớp (input_guardrail:injection, ...)
    blocked_by_layer: dict = field(default_factory=dict)
    redacted_responses: int = 0
    llm_errors: int = 0

    # Sliding window for alerting
    window_seconds: int = 300
    min_requests_for_rate_alert: int = 5
    _events: deque = field(default_factory=deque, repr=False)

    def record(
        self,
        *,
        blocked: bool,
        layer: str | None = None,
        rate_limited: bool = False,
        redacted: bool = False,
        judged: bool = False,
        judge_failed: bool = False,
        llm_error: bool = False,
        user_id: str = "anonymous",
        now: float | None = None,
    ) -> None:
        """Update counters after one request went through the pipeline."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if rate_limited:
            self.rate_limit_hits += 1
        if redacted:
            self.redacted_responses += 1
        if judged:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1
        if llm_error:
            self.llm_errors += 1
        self._events.append(_Event(
            ts=time.time() if now is None else now,
            user_id=user_id,
            blocked=blocked,
            rate_limited=rate_limited,
            judged=judged,
            judge_failed=judge_failed,
        ))

    def _window(self, now: float | None = None) -> list[_Event]:
        now = time.time() if now is None else now
        while self._events and self._events[0].ts <= now - self.window_seconds:
            self._events.popleft()
        return list(self._events)

    def window_stats(self, now: float | None = None) -> dict:
        events = self._window(now)
        total = len(events)
        blocked = sum(e.blocked for e in events)
        judged = sum(e.judged for e in events)
        judge_failed = sum(e.judge_failed for e in events)
        rl_by_user: dict[str, int] = {}
        for e in events:
            if e.rate_limited:
                rl_by_user[e.user_id] = rl_by_user.get(e.user_id, 0) + 1
        return {
            "window_seconds": self.window_seconds,
            "requests": total,
            "blocked": blocked,
            "block_rate": blocked / total if total else 0.0,
            "judge_checks": judged,
            "judge_fail_rate": judge_failed / judged if judged else 0.0,
            "rate_limit_hits_by_user": rl_by_user,
        }

    def check_metrics(self, now: float | None = None) -> list[Alert]:
        """Compute windowed rates and (re)build the alert list.

        Alerts are recomputed from scratch on each call so repeated checks do
        not create duplicates, and stale events age out of the window.
        """
        w = self.window_stats(now)
        alerts: list[Alert] = []
        if w["requests"] >= self.min_requests_for_rate_alert and w["block_rate"] > self.block_rate_threshold:
            alerts.append(Alert(
                metric="block_rate",
                value=round(w["block_rate"], 4),
                threshold=self.block_rate_threshold,
                message=(
                    f"Block rate {w['block_rate']:.0%} over the last {self.window_seconds}s "
                    f"(> {self.block_rate_threshold:.0%}): possible attack campaign or "
                    "over-blocking false positives — review audit log."
                ),
            ))
        floods = {
            u: n for u, n in w["rate_limit_hits_by_user"].items()
            if n >= self.rate_limit_hit_threshold
        }
        if floods:
            worst = max(floods.values())
            alerts.append(Alert(
                metric="rate_limit_hits",
                value=float(worst),
                threshold=float(self.rate_limit_hit_threshold),
                message=(
                    f"Rate-limit hits ≥ {self.rate_limit_hit_threshold} in {self.window_seconds}s "
                    f"for user(s) {sorted(floods)}: possible flooding / cost attack."
                ),
            ))
        if w["judge_checks"] and w["judge_fail_rate"] > self.judge_fail_rate_threshold:
            alerts.append(Alert(
                metric="judge_fail_rate",
                value=round(w["judge_fail_rate"], 4),
                threshold=self.judge_fail_rate_threshold,
                message=(
                    f"LLM judge rejected {w['judge_fail_rate']:.0%} of responses "
                    f"(> {self.judge_fail_rate_threshold:.0%}): model may be leaking or hallucinating."
                ),
            ))
        self.alerts = alerts
        return alerts

    def export_json(self, filepath: str | None = None) -> str:
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot()
        payload.update({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window": self.window_stats(),
            "blocked_by_layer": dict(sorted(self.blocked_by_layer.items())),
            "redacted_responses": self.redacted_responses,
            "llm_errors": self.llm_errors,
            "thresholds": {
                "block_rate": self.block_rate_threshold,
                "rate_limit_hits_per_user": self.rate_limit_hit_threshold,
                "judge_fail_rate": self.judge_fail_rate_threshold,
                "window_seconds": self.window_seconds,
            },
        })
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
