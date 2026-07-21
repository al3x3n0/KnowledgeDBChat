"""
Research monitor profile service.

Builds and persists lightweight token-score profiles from Research Inbox triage.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.orm import load_only
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.models.agent_job import AgentJob
from app.models.notification import Notification, NotificationType
from app.models.research_inbox import ResearchInboxItem
from app.models.research_monitor_profile import ResearchMonitorProfile
from app.services.autonomy_service import (
    build_monitor_follow_up_autonomy_compat,
    build_monitor_policy_compat_fields,
    build_monitor_policy_history_compat_entry,
    normalize_monitor_allowed_recommendations,
    normalize_monitor_policy_mode,
    resolve_monitor_automation_contract,
)
from app.services.scientific_validation_service import normalize_portfolio_automation_profile


class ResearchMonitorProfileService:
    SAFE_AUTONOMY_RECOMMENDATIONS = ["deep_dive_chain", "single_research_job"]
    POLICY_HISTORY_KEY = "follow_up_policy_history"
    BUDGET_HISTORY_KEY = "autonomy_budget_history"
    CUSTOMER_REBALANCE_HISTORY_KEY = "customer_rebalance_history"
    POLICY_EVALUATION_TARGET_COUNT = 8
    CUSTOMER_REBALANCE_EVALUATION_TARGET_COUNT = 8
    DEFAULT_AUTONOMY_BUDGET = {
        "auto_launch_limit_24h": 3,
        "approval_queue_limit_24h": 6,
        "alert_limit_24h": 4,
        "queue_backlog_cap": 8,
    }
    DEFAULT_CUSTOMER_AUTONOMY_BUDGET = {
        "auto_launch_limit_24h": 0,
        "approval_queue_limit_24h": 0,
        "alert_limit_24h": 0,
        "queue_backlog_cap": 0,
    }
    STOPWORDS = {
        "the","and","for","with","from","that","this","into","over","under","when","where","what","which","while",
        "your","you","are","our","their","they","them","then","than","also","only","just","more","most","less",
        "use","using","used","make","made","help","helps","via","can","could","should","would","may","might","will",
        "data","dataset","datasets","model","models","train","training","eval","evaluate","evaluation","assistant",
        "job","jobs","paper","papers","doc","docs","document","documents","research","monitor",
    }

    def tokenize(self, text: str) -> list[str]:
        raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
        out: list[str] = []
        for w in raw:
            w = w.strip("_-")
            if len(w) < 3:
                continue
            if w in self.STOPWORDS:
                continue
            out.append(w)
        return out

    def extract_phrases(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        phrases: list[str] = []
        for idx in range(len(tokens) - 1):
            phrase = f"{tokens[idx]} {tokens[idx + 1]}".strip()
            if phrase:
                phrases.append(phrase)
        return phrases

    @staticmethod
    def _safe_rate(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return round((float(numerator) / float(denominator)) * 100.0, 1)

    @staticmethod
    def _score_trend(*, completed_count: int, failed_count: int, cancelled_count: int) -> str:
        negative = failed_count + cancelled_count
        if completed_count > negative:
            return "positive"
        if negative > completed_count:
            return "negative"
        return "mixed"

    def _normalize_policy_mode(self, raw: object) -> str:
        return normalize_monitor_policy_mode(raw)

    def _normalize_allowed_recommendations(self, raw: object) -> list[str]:
        return normalize_monitor_allowed_recommendations(raw, default_allowed=self.SAFE_AUTONOMY_RECOMMENDATIONS)

    def _normalize_policy_config(self, raw: object) -> dict[str, Any]:
        policy = raw if isinstance(raw, dict) else {}
        return {
            "mode": self._normalize_policy_mode(policy.get("mode")),
            "allowed_recommendations": self._normalize_allowed_recommendations(policy.get("allowed_recommendations")),
        }

    def _monitor_follow_up_autonomy_from_policy(
        self,
        *,
        automation_profile: object,
        automation_policy: object,
        effective_policy: object,
    ) -> dict[str, Any]:
        return build_monitor_follow_up_autonomy_compat(
            automation_profile=automation_profile,
            automation_policy=automation_policy,
            effective_policy=effective_policy,
            default_allowed=self.SAFE_AUTONOMY_RECOMMENDATIONS,
        )

    def resolve_monitor_automation_config(self, config: object) -> dict[str, Any]:
        return resolve_monitor_automation_contract(
            config,
            default_allowed=self.SAFE_AUTONOMY_RECOMMENDATIONS,
        )

    def _normalize_budget_config(self, raw: object, *, defaults: Optional[dict[str, int]] = None) -> dict[str, int]:
        budget = raw if isinstance(raw, dict) else {}
        default_budget = defaults or self.DEFAULT_AUTONOMY_BUDGET
        out: dict[str, int] = {}
        for key, default in default_budget.items():
            try:
                value = int(budget.get(key))
            except Exception:
                value = default
            out[key] = max(0, min(value, 10000))
        return out

    def _normalize_customer_budget_config(self, raw: object) -> dict[str, int]:
        return self._normalize_budget_config(raw, defaults=self.DEFAULT_CUSTOMER_AUTONOMY_BUDGET)

    def _normalize_budget_history_entry(self, raw: object) -> Optional[dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        at_value = raw.get("at")
        parsed_at: Optional[datetime] = None
        if isinstance(at_value, datetime):
            parsed_at = at_value
        elif isinstance(at_value, str):
            try:
                parsed_at = datetime.fromisoformat(at_value.replace("Z", "+00:00"))
            except Exception:
                parsed_at = None
        if parsed_at is None:
            return None
        return {
            "id": str(raw.get("id") or "").strip() or None,
            "at": parsed_at,
            "actor_user_id": (str(raw.get("actor_user_id") or "").strip() or None),
            "change_source": (str(raw.get("change_source") or "").strip() or "manual_override"),
            "change_reason": (str(raw.get("change_reason") or "").strip() or None),
            "previous_autonomy_budget": self._normalize_budget_config(raw.get("previous_autonomy_budget")),
            "next_autonomy_budget": self._normalize_budget_config(raw.get("next_autonomy_budget")),
            "guidance_context": raw.get("guidance_context") if isinstance(raw.get("guidance_context"), dict) else {},
        }

    def _normalize_customer_rebalance_history_entry(self, raw: object) -> Optional[dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        at_value = raw.get("at")
        parsed_at: Optional[datetime] = None
        if isinstance(at_value, datetime):
            parsed_at = at_value
        elif isinstance(at_value, str):
            try:
                parsed_at = datetime.fromisoformat(at_value.replace("Z", "+00:00"))
            except Exception:
                parsed_at = None
        if parsed_at is None:
            return None
        changes: list[dict[str, Any]] = []
        raw_changes = raw.get("changes") if isinstance(raw.get("changes"), list) else []
        for change in raw_changes:
            if not isinstance(change, dict):
                continue
            changes.append(
                {
                    "monitor_job_id": change.get("monitor_job_id"),
                    "monitor_name": str(change.get("monitor_name") or "").strip() or "Unknown monitor",
                    "customer": (str(change.get("customer") or "").strip() or None),
                    "current_budget": self._normalize_budget_config(change.get("current_budget")),
                    "proposed_budget": self._normalize_budget_config(change.get("proposed_budget")),
                    "delta_budget": {
                        "auto_launch_limit_24h": int((change.get("delta_budget") or {}).get("auto_launch_limit_24h", 0) or 0),
                        "approval_queue_limit_24h": int((change.get("delta_budget") or {}).get("approval_queue_limit_24h", 0) or 0),
                        "alert_limit_24h": int((change.get("delta_budget") or {}).get("alert_limit_24h", 0) or 0),
                        "queue_backlog_cap": int((change.get("delta_budget") or {}).get("queue_backlog_cap", 0) or 0),
                    },
                    "reasons": [str(reason).strip() for reason in (change.get("reasons") or []) if str(reason).strip()],
                }
            )
        return {
            "id": str(raw.get("id") or "").strip() or None,
            "at": parsed_at,
            "actor_user_id": (str(raw.get("actor_user_id") or "").strip() or None),
            "change_source": (str(raw.get("change_source") or "").strip() or "customer_rebalance_guidance"),
            "change_reason": (str(raw.get("change_reason") or "").strip() or None),
            "changes": changes,
            "before_capacity": self._normalize_budget_config(raw.get("before_capacity"), defaults={
                "auto_launch_limit_24h": 0,
                "approval_queue_limit_24h": 0,
                "alert_limit_24h": 0,
                "queue_backlog_cap": 0,
            }),
            "after_capacity": self._normalize_budget_config(raw.get("after_capacity"), defaults={
                "auto_launch_limit_24h": 0,
                "approval_queue_limit_24h": 0,
                "alert_limit_24h": 0,
                "queue_backlog_cap": 0,
            }),
            "evaluation_target_count": max(3, int(raw.get("evaluation_target_count") or self.CUSTOMER_REBALANCE_EVALUATION_TARGET_COUNT)),
            "evaluation_state": (str(raw.get("evaluation_state") or "").strip().lower() or "active"),
        }

    def _budget_history_for_job(self, job: Optional[AgentJob]) -> list[dict[str, Any]]:
        if not job or not isinstance(getattr(job, "results", None), dict):
            return []
        raw_entries = job.results.get(self.BUDGET_HISTORY_KEY)
        if not isinstance(raw_entries, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw in raw_entries:
            entry = self._normalize_budget_history_entry(raw)
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda row: row["at"], reverse=True)
        return entries[:10]

    def _customer_rebalance_history_for_profile(self, profile: Optional[ResearchMonitorProfile]) -> list[dict[str, Any]]:
        if not profile:
            return []
        raw_entries = getattr(profile, "customer_rebalance_history", None)
        if not isinstance(raw_entries, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw in raw_entries:
            entry = self._normalize_customer_rebalance_history_entry(raw)
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda row: row["at"], reverse=True)
        return entries[:10]

    def append_customer_rebalance_history_entry(
        self,
        *,
        profile: ResearchMonitorProfile,
        actor_user_id: object,
        change_source: Optional[str],
        change_reason: Optional[str],
        changes: list[dict[str, Any]],
        before_capacity: dict[str, int],
        after_capacity: dict[str, int],
    ) -> dict[str, Any]:
        from uuid import uuid4

        history = self._customer_rebalance_history_for_profile(profile)
        normalized_changes: list[dict[str, Any]] = []
        for change in changes:
            if not isinstance(change, dict):
                continue
            normalized_changes.append(
                {
                    "monitor_job_id": str(change.get("monitor_job_id") or "").strip() or None,
                    "monitor_name": str(change.get("monitor_name") or "").strip() or "Unknown monitor",
                    "customer": (str(change.get("customer") or "").strip() or None),
                    "current_budget": self._normalize_budget_config(change.get("current_budget")),
                    "proposed_budget": self._normalize_budget_config(change.get("proposed_budget")),
                    "delta_budget": {
                        "auto_launch_limit_24h": int((change.get("delta_budget") or {}).get("auto_launch_limit_24h", 0) or 0),
                        "approval_queue_limit_24h": int((change.get("delta_budget") or {}).get("approval_queue_limit_24h", 0) or 0),
                        "alert_limit_24h": int((change.get("delta_budget") or {}).get("alert_limit_24h", 0) or 0),
                        "queue_backlog_cap": int((change.get("delta_budget") or {}).get("queue_backlog_cap", 0) or 0),
                    },
                    "reasons": [str(reason).strip() for reason in (change.get("reasons") or []) if str(reason).strip()],
                }
            )
        entry = {
            "id": str(uuid4()),
            "at": datetime.utcnow(),
            "actor_user_id": (str(actor_user_id or "").strip() or None),
            "change_source": (str(change_source or "").strip() or "customer_rebalance_guidance"),
            "change_reason": (str(change_reason or "").strip() or None),
            "changes": normalized_changes,
            "before_capacity": self._normalize_budget_config(before_capacity, defaults={
                "auto_launch_limit_24h": 0,
                "approval_queue_limit_24h": 0,
                "alert_limit_24h": 0,
                "queue_backlog_cap": 0,
            }),
            "after_capacity": self._normalize_budget_config(after_capacity, defaults={
                "auto_launch_limit_24h": 0,
                "approval_queue_limit_24h": 0,
                "alert_limit_24h": 0,
                "queue_backlog_cap": 0,
            }),
            "evaluation_target_count": self.CUSTOMER_REBALANCE_EVALUATION_TARGET_COUNT,
            "evaluation_state": "active",
        }
        history.insert(0, entry)
        normalized = sorted(history, key=lambda row: row["at"], reverse=True)[:20]
        profile.customer_rebalance_history = [
            {
                **row,
                "at": row["at"].isoformat(),
            }
            for row in normalized
        ]
        return entry

    def append_budget_history_entry(
        self,
        *,
        job: AgentJob,
        previous_budget: dict[str, int],
        next_budget: dict[str, int],
        actor_user_id: object,
        change_source: Optional[str],
        change_reason: Optional[str],
        guidance_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        history = self._budget_history_for_job(job)
        from uuid import uuid4

        entry = {
            "id": str(uuid4()),
            "at": datetime.utcnow(),
            "actor_user_id": (str(actor_user_id or "").strip() or None),
            "change_source": (str(change_source or "").strip() or "manual_override"),
            "change_reason": (str(change_reason or "").strip() or None),
            "previous_autonomy_budget": self._normalize_budget_config(previous_budget),
            "next_autonomy_budget": self._normalize_budget_config(next_budget),
            "guidance_context": guidance_context if isinstance(guidance_context, dict) else {},
        }
        history.insert(0, entry)
        normalized = sorted(history, key=lambda row: row["at"], reverse=True)[:20]
        results = dict(job.results or {}) if isinstance(job.results, dict) else {}
        results[self.BUDGET_HISTORY_KEY] = [
            {
                **row,
                "at": row["at"].isoformat(),
            }
            for row in normalized
        ]
        job.results = results
        return entry

    @staticmethod
    def _empty_budget_usage() -> dict[str, int]:
        return {
            "auto_launch_count_24h": 0,
            "approval_queue_count_24h": 0,
            "alert_count_24h": 0,
            "queue_backlog_count": 0,
        }

    def _budget_remaining(self, budget_config: dict[str, int], budget_usage: dict[str, int]) -> dict[str, int]:
        return {
            "auto_launch_count_24h": max(0, int(budget_config.get("auto_launch_limit_24h", 0) or 0) - int(budget_usage.get("auto_launch_count_24h", 0) or 0)),
            "approval_queue_count_24h": max(0, int(budget_config.get("approval_queue_limit_24h", 0) or 0) - int(budget_usage.get("approval_queue_count_24h", 0) or 0)),
            "alert_count_24h": max(0, int(budget_config.get("alert_limit_24h", 0) or 0) - int(budget_usage.get("alert_count_24h", 0) or 0)),
            "queue_backlog_count": max(0, int(budget_config.get("queue_backlog_cap", 0) or 0) - int(budget_usage.get("queue_backlog_count", 0) or 0)),
        }

    def _derive_budget_throttle(
        self,
        *,
        policy_mode: str,
        budget_config: dict[str, int],
        budget_usage: dict[str, int],
    ) -> tuple[str, list[str]]:
        reasons: list[str] = []
        auto_exhausted = int(budget_usage.get("auto_launch_count_24h", 0) or 0) >= int(budget_config.get("auto_launch_limit_24h", 0) or 0) > 0
        approval_exhausted = int(budget_usage.get("approval_queue_count_24h", 0) or 0) >= int(budget_config.get("approval_queue_limit_24h", 0) or 0) > 0
        backlog_exhausted = int(budget_usage.get("queue_backlog_count", 0) or 0) >= int(budget_config.get("queue_backlog_cap", 0) or 0) > 0
        alert_exhausted = int(budget_usage.get("alert_count_24h", 0) or 0) >= int(budget_config.get("alert_limit_24h", 0) or 0) > 0

        if auto_exhausted:
            reasons.append("Auto-launch budget exhausted in the last 24 hours.")
        if approval_exhausted:
            reasons.append("Approval queue budget exhausted in the last 24 hours.")
        if backlog_exhausted:
            reasons.append("Queue backlog cap reached for this monitor.")
        if alert_exhausted:
            reasons.append("Alert budget exhausted in the last 24 hours.")

        normalized_mode = self._normalize_policy_mode(policy_mode)
        if normalized_mode == "auto_launch_safe":
            if approval_exhausted or backlog_exhausted:
                return "manual_only_clamped", reasons
            if auto_exhausted:
                return "auto_launch_throttled", reasons
        elif normalized_mode == "queue_for_approval":
            if approval_exhausted or backlog_exhausted:
                return "manual_only_clamped", reasons
        return "normal", reasons

    def _budget_ratio(self, usage: dict[str, Any], budget: dict[str, Any], usage_key: str, budget_key: str) -> float:
        limit = int(budget.get(budget_key, 0) or 0)
        used = int(usage.get(usage_key, 0) or 0)
        if limit <= 0:
            return 0.0
        return float(used) / float(limit)

    def _build_customer_rebalance_guidance(
        self,
        *,
        customer_row: dict[str, Any],
        monitor_rows: list[dict[str, Any]],
    ) -> tuple[str, list[str], Optional[str], list[dict[str, Any]]]:
        if len(monitor_rows) < 2:
            return "none", [], None, []

        customer_throttle = str(customer_row.get("customer_budget_throttle_state") or "normal").strip().lower()
        portfolio_status = str(customer_row.get("portfolio_status") or "normal").strip().lower()
        actionable = (
            customer_throttle != "normal"
            or portfolio_status in {"monitor_throttled", "backlog_heavy", "alert_heavy", "nearing_saturation", "customer_budget_throttled"}
        )
        if not actionable:
            return "none", [], None, []

        ranked_pressure = sorted(
            monitor_rows,
            key=lambda row: (
                -(self._budget_ratio(row.get("budget_usage") or {}, row.get("autonomy_budget") or {}, "queue_backlog_count", "queue_backlog_cap")
                  + self._budget_ratio(row.get("budget_usage") or {}, row.get("autonomy_budget") or {}, "approval_queue_count_24h", "approval_queue_limit_24h")
                  + self._budget_ratio(row.get("budget_usage") or {}, row.get("autonomy_budget") or {}, "auto_launch_count_24h", "auto_launch_limit_24h")),
                0 if str(row.get("health_bucket") or "") == "weak" else 1 if str(row.get("health_bucket") or "") == "mixed" else 2,
                -int(row.get("accepted_count", 0) or 0),
            ),
        )
        ranked_relief = sorted(
            monitor_rows,
            key=lambda row: (
                self._budget_ratio(row.get("budget_usage") or {}, row.get("autonomy_budget") or {}, "auto_launch_count_24h", "auto_launch_limit_24h")
                + self._budget_ratio(row.get("budget_usage") or {}, row.get("autonomy_budget") or {}, "approval_queue_count_24h", "approval_queue_limit_24h")
                + self._budget_ratio(row.get("budget_usage") or {}, row.get("autonomy_budget") or {}, "queue_backlog_count", "queue_backlog_cap"),
                0 if str(row.get("health_bucket") or "") == "strong" else 1 if str(row.get("health_bucket") or "") == "mixed" else 2,
                str(row.get("monitor_name") or ""),
            ),
        )

        pressure_monitor = ranked_pressure[0]
        relief_monitor = next((row for row in ranked_relief if row.get("monitor_job_id") != pressure_monitor.get("monitor_job_id")), None)
        if not relief_monitor:
            return "none", [], None, []

        field_specs = [
            ("auto_launch_limit_24h", "auto_launch_count_24h", "auto-launch headroom"),
            ("approval_queue_limit_24h", "approval_queue_count_24h", "approval queue headroom"),
            ("queue_backlog_cap", "queue_backlog_count", "backlog headroom"),
            ("alert_limit_24h", "alert_count_24h", "alert headroom"),
        ]
        pressure_budget = self._normalize_budget_config(pressure_monitor.get("autonomy_budget"))
        relief_budget = self._normalize_budget_config(relief_monitor.get("autonomy_budget"))
        pressure_usage = pressure_monitor.get("budget_usage") if isinstance(pressure_monitor.get("budget_usage"), dict) else {}
        relief_usage = relief_monitor.get("budget_usage") if isinstance(relief_monitor.get("budget_usage"), dict) else {}
        next_pressure_budget = dict(pressure_budget)
        next_relief_budget = dict(relief_budget)
        pressure_reasons: list[str] = []
        relief_reasons: list[str] = []

        for budget_key, usage_key, label in field_specs:
            pressure_ratio = self._budget_ratio(pressure_usage, pressure_budget, usage_key, budget_key)
            relief_ratio = self._budget_ratio(relief_usage, relief_budget, usage_key, budget_key)
            if pressure_ratio < 0.8 or relief_ratio > 0.5:
                continue
            if int(pressure_budget.get(budget_key, 0) or 0) <= 1:
                continue
            next_pressure_budget[budget_key] = max(1, int(next_pressure_budget.get(budget_key, 0) or 0) - 1)
            next_relief_budget[budget_key] = int(next_relief_budget.get(budget_key, 0) or 0) + 1
            pressure_reasons.append(f"Reduce {label} on {pressure_monitor.get('monitor_name')}.")
            relief_reasons.append(f"Reassign {label} to {relief_monitor.get('monitor_name')}.")

        changes: list[dict[str, Any]] = []
        if next_pressure_budget != pressure_budget:
            changes.append(
                {
                    "monitor_job_id": pressure_monitor.get("monitor_job_id"),
                    "monitor_name": pressure_monitor.get("monitor_name"),
                    "customer": pressure_monitor.get("customer"),
                    "current_budget": pressure_budget,
                    "proposed_budget": next_pressure_budget,
                    "delta_budget": {
                        key: int(next_pressure_budget.get(key, 0) or 0) - int(pressure_budget.get(key, 0) or 0)
                        for key in self.DEFAULT_AUTONOMY_BUDGET.keys()
                    },
                    "reasons": pressure_reasons or ["This monitor is driving disproportionate portfolio pressure."],
                }
            )
        if next_relief_budget != relief_budget:
            changes.append(
                {
                    "monitor_job_id": relief_monitor.get("monitor_job_id"),
                    "monitor_name": relief_monitor.get("monitor_name"),
                    "customer": relief_monitor.get("customer"),
                    "current_budget": relief_budget,
                    "proposed_budget": next_relief_budget,
                    "delta_budget": {
                        key: int(next_relief_budget.get(key, 0) or 0) - int(relief_budget.get(key, 0) or 0)
                        for key in self.DEFAULT_AUTONOMY_BUDGET.keys()
                    },
                    "reasons": relief_reasons or ["This monitor is strong and currently under-utilized."],
                }
            )

        if not changes:
            return "none", [], None, []

        reasons = [
            f"{pressure_monitor.get('monitor_name')} is consuming the most customer budget pressure.",
            f"{relief_monitor.get('monitor_name')} has stronger spare headroom.",
        ]
        summary = f"Shift budget headroom from {pressure_monitor.get('monitor_name')} to {relief_monitor.get('monitor_name')}."
        return "actionable", reasons, summary, changes

    def _customer_capacity_from_changes(
        self,
        *,
        base_rows: list[dict[str, Any]],
        changes_by_monitor: dict[str, dict[str, int]],
    ) -> dict[str, int]:
        totals = {
            "auto_launch_limit_24h": 0,
            "approval_queue_limit_24h": 0,
            "alert_limit_24h": 0,
            "queue_backlog_cap": 0,
        }
        for row in base_rows:
            monitor_id = str(row.get("monitor_job_id") or "")
            budget = changes_by_monitor.get(monitor_id) or self._normalize_budget_config(row.get("autonomy_budget"))
            for key in totals.keys():
                totals[key] += int(budget.get(key, 0) or 0)
        return totals

    def _budget_usage_for_monitor(
        self,
        *,
        items: list[ResearchInboxItem],
        alert_count_24h: int,
        now: Optional[datetime] = None,
    ) -> dict[str, int]:
        usage = self._empty_budget_usage()
        since = (now or datetime.utcnow()).replace(microsecond=0)
        window_start = since.timestamp() - 86400
        for item in items:
            item_time = self._item_sort_time(item).timestamp()
            launch_time = getattr(item, "follow_up_launched_at", None)
            launch_ts = launch_time.timestamp() if isinstance(launch_time, datetime) else item_time
            decision = str(item.follow_up_decision or "").strip().lower()
            launch_status = str(item.follow_up_launch_status or "").strip().lower()
            operator_decision = str(item.follow_up_operator_decision or "").strip().lower()
            if decision == "auto_launched" and launch_ts >= window_start:
                usage["auto_launch_count_24h"] += 1
            if launch_status == "pending_approval" and item_time >= window_start:
                usage["approval_queue_count_24h"] += 1
            if launch_status in {"pending_approval", "blocked", "failed"} or decision in {"manual", "queued_for_approval", "launch_failed"}:
                if operator_decision not in {"rejected", "approved_launch"} and launch_status not in {"rejected", "launched"}:
                    usage["queue_backlog_count"] += 1
        usage["alert_count_24h"] = max(0, int(alert_count_24h or 0))
        return usage

    async def build_monitor_budget_snapshot(
        self,
        *,
        db: AsyncSession,
        user_id,
        monitor_job: Optional[AgentJob],
    ) -> dict[str, Any]:
        if not monitor_job:
            budget_config = self._normalize_budget_config(None)
            usage = self._empty_budget_usage()
            remaining = self._budget_remaining(budget_config, usage)
            return {
                "autonomy_budget": budget_config,
                "budget_usage": usage,
                "budget_remaining": remaining,
                "budget_throttle_state": "normal",
                "budget_throttle_reasons": [],
            }
        config = monitor_job.config if isinstance(getattr(monitor_job, "config", None), dict) else {}
        automation = self.resolve_monitor_automation_config(config)
        effective_policy = automation["effective_policy"] if isinstance(automation.get("effective_policy"), dict) else {}
        budget_config = self._normalize_budget_config(config.get("autonomy_budget"))
        now = datetime.utcnow()
        since = now.timestamp() - 86400
        inbox_stmt = (
            select(ResearchInboxItem)
            .where(
                ResearchInboxItem.user_id == user_id,
                ResearchInboxItem.job_id == monitor_job.id,
                ResearchInboxItem.status == "accepted",
            )
            .options(load_only(
                ResearchInboxItem.follow_up_decision,
                ResearchInboxItem.follow_up_launch_status,
                ResearchInboxItem.follow_up_operator_decision,
                ResearchInboxItem.follow_up_launched_at,
                ResearchInboxItem.updated_at,
                ResearchInboxItem.created_at,
            ))
        )
        inbox_rows = list((await db.execute(inbox_stmt)).scalars().all())
        notif_stmt = (
            select(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.notification_type.in_([
                    NotificationType.QUEUE_URGENCY_ALERT,
                    NotificationType.POLICY_GUARDRAIL_ALERT,
                    NotificationType.AUTONOMY_BUDGET_ALERT,
                ]),
            )
        )
        notifications = list((await db.execute(notif_stmt)).scalars().all())
        alert_count = 0
        for notification in notifications:
            created_at = getattr(notification, "created_at", None)
            if not isinstance(created_at, datetime) or created_at.timestamp() < since:
                continue
            data = notification.data if isinstance(notification.data, dict) else {}
            if str(data.get("job_id") or "").strip() == str(monitor_job.id):
                alert_count += 1
        usage = self._budget_usage_for_monitor(items=inbox_rows, alert_count_24h=alert_count, now=now)
        remaining = self._budget_remaining(budget_config, usage)
        throttle_state, throttle_reasons = self._derive_budget_throttle(
            policy_mode=str(effective_policy.get("follow_up_review_mode") or "manual_only"),
            budget_config=budget_config,
            budget_usage=usage,
        )
        return {
            "autonomy_budget": budget_config,
            "budget_usage": usage,
            "budget_remaining": remaining,
            "budget_throttle_state": throttle_state,
            "budget_throttle_reasons": throttle_reasons,
        }

    async def build_customer_budget_snapshot(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: Optional[str],
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:
        customer_name = str(customer or "").strip() or None
        budget_config = self._normalize_customer_budget_config(None)
        if not customer_name:
            usage = self._empty_budget_usage()
            return {
                "customer": None,
                "customer_budget": budget_config,
                "customer_budget_usage": usage,
                "customer_budget_remaining": self._budget_remaining(budget_config, usage),
                "customer_budget_throttle_state": "normal",
                "customer_budget_throttle_reasons": [],
            }

        profile = await self.get_profile(db=db, user_id=user_id, customer=customer_name)
        if profile is not None:
            budget_config = self._normalize_customer_budget_config(getattr(profile, "customer_budget_config", None))

        current_time = now or datetime.utcnow()
        since = current_time.timestamp() - 86400
        inbox_stmt = (
            select(ResearchInboxItem)
            .where(
                ResearchInboxItem.user_id == user_id,
                ResearchInboxItem.customer == customer_name,
                ResearchInboxItem.status == "accepted",
            )
            .options(load_only(
                ResearchInboxItem.follow_up_decision,
                ResearchInboxItem.follow_up_launch_status,
                ResearchInboxItem.follow_up_operator_decision,
                ResearchInboxItem.follow_up_launched_at,
                ResearchInboxItem.updated_at,
                ResearchInboxItem.created_at,
            ))
        )
        inbox_rows = list((await db.execute(inbox_stmt)).scalars().all())

        notif_stmt = (
            select(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.notification_type.in_([
                    NotificationType.QUEUE_URGENCY_ALERT,
                    NotificationType.POLICY_GUARDRAIL_ALERT,
                    NotificationType.AUTONOMY_BUDGET_ALERT,
                    NotificationType.CUSTOMER_AUTONOMY_BUDGET_ALERT,
                ]),
            )
        )
        notifications = list((await db.execute(notif_stmt)).scalars().all())
        alert_count = 0
        for notification in notifications:
            created_at = getattr(notification, "created_at", None)
            if not isinstance(created_at, datetime) or created_at.timestamp() < since:
                continue
            data = notification.data if isinstance(notification.data, dict) else {}
            if str(data.get("customer") or "").strip() == customer_name:
                alert_count += 1

        usage = self._budget_usage_for_monitor(items=inbox_rows, alert_count_24h=alert_count, now=current_time)
        remaining = self._budget_remaining(budget_config, usage)
        throttle_state, throttle_reasons = self._derive_budget_throttle(
            policy_mode="auto_launch_safe",
            budget_config=budget_config,
            budget_usage=usage,
        )
        return {
            "customer": customer_name,
            "customer_budget": budget_config,
            "customer_budget_usage": usage,
            "customer_budget_remaining": remaining,
            "customer_budget_throttle_state": throttle_state,
            "customer_budget_throttle_reasons": throttle_reasons,
        }

    def _policy_history_for_job(self, job: Optional[AgentJob]) -> list[dict[str, Any]]:
        if not job or not isinstance(getattr(job, "results", None), dict):
            return []
        raw_entries = job.results.get(self.POLICY_HISTORY_KEY)
        if not isinstance(raw_entries, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            at_value = raw.get("at")
            parsed_at: Optional[datetime] = None
            if isinstance(at_value, datetime):
                parsed_at = at_value
            elif isinstance(at_value, str):
                try:
                    parsed_at = datetime.fromisoformat(at_value.replace("Z", "+00:00"))
                except Exception:
                    parsed_at = None
            if parsed_at is None:
                continue
            entries.append(
                {
                    "id": str(raw.get("id") or "").strip() or None,
                    "at": parsed_at,
                    "actor_user_id": (str(raw.get("actor_user_id") or "").strip() or None),
                    "change_source": (str(raw.get("change_source") or "").strip() or None),
                    "change_reason": (str(raw.get("change_reason") or "").strip() or None),
                    "previous_automation_profile": normalize_portfolio_automation_profile(raw.get("previous_automation_profile"), default="balanced"),
                    "next_automation_profile": normalize_portfolio_automation_profile(raw.get("next_automation_profile"), default="balanced"),
                    "previous_automation_policy": raw.get("previous_automation_policy") if isinstance(raw.get("previous_automation_policy"), dict) else {},
                    "next_automation_policy": raw.get("next_automation_policy") if isinstance(raw.get("next_automation_policy"), dict) else {},
                    "previous_effective_policy": raw.get("previous_effective_policy") if isinstance(raw.get("previous_effective_policy"), dict) else {},
                    "next_effective_policy": raw.get("next_effective_policy") if isinstance(raw.get("next_effective_policy"), dict) else {},
                    "effective_clamp_state": (str(raw.get("effective_clamp_state") or "").strip() or None),
                    "effective_clamp_reasons": [str(reason).strip() for reason in (raw.get("effective_clamp_reasons") or []) if str(reason).strip()],
                    "analytics_context": raw.get("analytics_context") if isinstance(raw.get("analytics_context"), dict) else {},
                    "evaluation_target_count": max(
                        3,
                        int(raw.get("evaluation_target_count") or self.POLICY_EVALUATION_TARGET_COUNT),
                    ),
                    "evaluation_state": (str(raw.get("evaluation_state") or "").strip().lower() or "active"),
                }
            )
            entries[-1].update(
                build_monitor_policy_history_compat_entry(
                    previous_snapshot={
                        "follow_up_autonomy": self._normalize_policy_config(raw.get("previous_follow_up_autonomy")),
                    },
                    next_snapshot={
                        "follow_up_autonomy": self._normalize_policy_config(raw.get("next_follow_up_autonomy")),
                    },
                )
            )
        entries.sort(key=lambda entry: entry["at"], reverse=True)
        return entries[:5]

    @staticmethod
    def _customer_portfolio_status_for_bucket(bucket: dict[str, Any]) -> tuple[str, list[str]]:
        reasons: list[str] = []
        auto_capacity = int(bucket.get("auto_launch_capacity_24h", 0) or 0)
        auto_used = int(bucket.get("auto_launch_used_24h", 0) or 0)
        approval_capacity = int(bucket.get("approval_queue_capacity_24h", 0) or 0)
        approval_used = int(bucket.get("approval_queue_used_24h", 0) or 0)
        alert_capacity = int(bucket.get("alert_capacity_24h", 0) or 0)
        alert_used = int(bucket.get("alert_used_24h", 0) or 0)
        backlog_capacity = int(bucket.get("backlog_capacity", 0) or 0)
        backlog_used = int(bucket.get("backlog_used", 0) or 0)
        throttled_monitor_count = int(bucket.get("throttled_monitor_count", 0) or 0)

        if throttled_monitor_count > 0:
            reasons.append(f"{throttled_monitor_count} monitor(s) are currently throttled.")
        if backlog_capacity > 0 and backlog_used >= max(1, int(backlog_capacity * 0.8)):
            reasons.append(f"Backlog is using {backlog_used}/{backlog_capacity} available headroom.")
        if alert_capacity > 0 and alert_used >= max(1, int(alert_capacity * 0.8)):
            reasons.append(f"Alerts used {alert_used}/{alert_capacity} of configured headroom.")
        if (
            (auto_capacity > 0 and auto_used >= max(1, int(auto_capacity * 0.8)))
            or (approval_capacity > 0 and approval_used >= max(1, int(approval_capacity * 0.8)))
        ):
            reasons.append("Launch or approval capacity is nearing saturation.")

        if throttled_monitor_count > 0:
            return "monitor_throttled", reasons[:3]
        if backlog_capacity > 0 and backlog_used >= max(1, int(backlog_capacity * 0.8)):
            return "backlog_heavy", reasons[:3]
        if alert_capacity > 0 and alert_used >= max(1, int(alert_capacity * 0.8)):
            return "alert_heavy", reasons[:3]
        if (
            (auto_capacity > 0 and auto_used >= max(1, int(auto_capacity * 0.8)))
            or (approval_capacity > 0 and approval_used >= max(1, int(approval_capacity * 0.8)))
        ):
            return "nearing_saturation", reasons[:3]
        return "normal", reasons[:3]

    @staticmethod
    def _to_naive_utc(value: Any) -> Optional[datetime]:
        """Coerce a datetime or ISO string to a naive-UTC datetime for comparison.

        The codebase stores naive-UTC datetimes, but policy-history "at" values
        may arrive as tz-aware datetimes or ISO strings (with a Z suffix);
        comparing those directly raises "offset-naive vs offset-aware".
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc).replace(tzinfo=None)
            return value
        return None

    @staticmethod
    def _item_sort_time(item: ResearchInboxItem) -> datetime:
        raw = (
            getattr(item, "updated_at", None)
            or getattr(item, "created_at", None)
            or datetime.utcnow()
        )
        return ResearchMonitorProfileService._to_naive_utc(raw) or datetime.utcnow()

    def _empty_policy_evaluation_counts(self) -> dict[str, int]:
        return {
            "accepted_count": 0,
            "auto_launched_count": 0,
            "approval_launched_count": 0,
            "queued_for_approval_count": 0,
            "manual_only_count": 0,
            "blocked_count": 0,
            "follow_up_completed_count": 0,
            "follow_up_failed_count": 0,
            "follow_up_cancelled_count": 0,
        }

    def _policy_evaluation_counts_for_items(self, items: list[ResearchInboxItem]) -> dict[str, int]:
        counts = self._empty_policy_evaluation_counts()
        for item in items:
            counts["accepted_count"] += 1
            decision = str(item.follow_up_decision or "").strip().lower()
            launch_status = str(item.follow_up_launch_status or "").strip().lower()
            operator_decision = str(item.follow_up_operator_decision or "").strip().lower()
            outcome_status = str(item.follow_up_outcome_status or "").strip().lower()

            if decision == "manual":
                counts["manual_only_count"] += 1
            if decision == "auto_launched":
                counts["auto_launched_count"] += 1
            if operator_decision == "approved_launch":
                counts["approval_launched_count"] += 1
            if launch_status == "pending_approval":
                counts["queued_for_approval_count"] += 1
            if launch_status == "blocked":
                counts["blocked_count"] += 1
            if outcome_status == "completed":
                counts["follow_up_completed_count"] += 1
            elif outcome_status == "failed":
                counts["follow_up_failed_count"] += 1
            elif outcome_status == "cancelled":
                counts["follow_up_cancelled_count"] += 1
        return counts

    def _customer_rebalance_evaluation_counts(
        self,
        *,
        items: list[ResearchInboxItem],
        monitor_rows: list[dict[str, Any]],
    ) -> dict[str, int]:
        policy_counts = self._policy_evaluation_counts_for_items(items)
        return {
            "accepted_count": int(policy_counts["accepted_count"]),
            "blocked_count": int(policy_counts["blocked_count"]),
            "follow_up_completed_count": int(policy_counts["follow_up_completed_count"]),
            "follow_up_failed_count": int(policy_counts["follow_up_failed_count"]),
            "follow_up_cancelled_count": int(policy_counts["follow_up_cancelled_count"]),
            "auto_launch_used_24h": sum(int((row.get("budget_usage") or {}).get("auto_launch_count_24h", 0) or 0) for row in monitor_rows),
            "approval_queue_used_24h": sum(int((row.get("budget_usage") or {}).get("approval_queue_count_24h", 0) or 0) for row in monitor_rows),
            "alert_used_24h": sum(int((row.get("budget_usage") or {}).get("alert_count_24h", 0) or 0) for row in monitor_rows),
            "backlog_used": sum(int((row.get("budget_usage") or {}).get("queue_backlog_count", 0) or 0) for row in monitor_rows),
            "throttled_monitor_count": sum(1 for row in monitor_rows if str(row.get("budget_throttle_state") or "normal").strip().lower() != "normal"),
        }

    def _customer_rebalance_delta_counts(self, before_counts: dict[str, int], after_counts: dict[str, int]) -> dict[str, int]:
        return {
            key: int(after_counts.get(key, 0) or 0) - int(before_counts.get(key, 0) or 0)
            for key in before_counts.keys()
        }

    def _customer_rebalance_sample_items(
        self,
        items: list[ResearchInboxItem],
        *,
        period: str,
        jobs_by_id: Optional[dict[Any, AgentJob]] = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for item in items[:limit]:
            title = str(item.title or "").strip() or "Untitled inbox item"
            summary = str(item.summary or "").strip() or None
            job = (jobs_by_id or {}).get(item.job_id)
            samples.append(
                {
                    "item_id": item.id,
                    "title": title,
                    "period": period,
                    "launch_status": (str(item.follow_up_launch_status or "").strip().lower() or None),
                    "outcome_status": (str(item.follow_up_outcome_status or "").strip().lower() or None),
                    "recommendation_key": (str(item.follow_up_recommendation_key or "").strip() or None),
                    "summary": summary,
                    "monitor_job_id": item.job_id,
                    "monitor_name": str(getattr(job, "name", "") or "").strip() or None,
                }
            )
        return samples

    def _policy_evaluation_samples_for_items(
        self,
        items: list[ResearchInboxItem],
        *,
        period: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for item in items[:limit]:
            title = str(item.title or "").strip() or "Untitled inbox item"
            summary = str(item.summary or "").strip() or None
            samples.append(
                {
                    "item_id": item.id,
                    "title": title,
                    "period": period,
                    "launch_status": (str(item.follow_up_launch_status or "").strip().lower() or None),
                    "outcome_status": (str(item.follow_up_outcome_status or "").strip().lower() or None),
                    "recommendation_key": (str(item.follow_up_recommendation_key or "").strip() or None),
                    "summary": summary,
                }
            )
        return samples

    @staticmethod
    def _policy_evaluation_delta_counts(before_counts: dict[str, int], after_counts: dict[str, int]) -> dict[str, int]:
        return {
            key: int(after_counts.get(key, 0) or 0) - int(before_counts.get(key, 0) or 0)
            for key in before_counts.keys()
        }

    def _evaluate_policy_history_entry(
        self,
        *,
        monitor_job_id: Any,
        history_entry: dict[str, Any],
        items: list[ResearchInboxItem],
    ) -> dict[str, Any]:
        target_count = max(3, int(history_entry.get("evaluation_target_count") or self.POLICY_EVALUATION_TARGET_COUNT))
        sorted_items = sorted(items, key=self._item_sort_time)
        effective_at = self._to_naive_utc(history_entry["at"]) or datetime.utcnow()
        before_candidates = [item for item in sorted_items if self._item_sort_time(item) < effective_at]
        after_candidates = [item for item in sorted_items if self._item_sort_time(item) >= effective_at]
        before_items = before_candidates[-target_count:]
        after_items = after_candidates[:target_count]
        before_counts = self._policy_evaluation_counts_for_items(before_items)
        after_counts = self._policy_evaluation_counts_for_items(after_items)
        delta_counts = self._policy_evaluation_delta_counts(before_counts, after_counts)
        sample_count = int(after_counts["accepted_count"])
        before_terminal = (
            before_counts["follow_up_completed_count"]
            + before_counts["follow_up_failed_count"]
            + before_counts["follow_up_cancelled_count"]
        )
        after_terminal = (
            after_counts["follow_up_completed_count"]
            + after_counts["follow_up_failed_count"]
            + after_counts["follow_up_cancelled_count"]
        )
        before_completion_rate = self._safe_rate(before_counts["follow_up_completed_count"], before_terminal)
        after_completion_rate = self._safe_rate(after_counts["follow_up_completed_count"], after_terminal)
        before_negative = before_counts["follow_up_failed_count"] + before_counts["follow_up_cancelled_count"]
        after_negative = after_counts["follow_up_failed_count"] + after_counts["follow_up_cancelled_count"]
        before_block_rate = self._safe_rate(before_counts["blocked_count"], before_counts["accepted_count"])
        after_block_rate = self._safe_rate(after_counts["blocked_count"], after_counts["accepted_count"])
        reasons: list[str] = []

        if sample_count < min(target_count, 3):
            status = "insufficient_data"
            reasons.append(f"Only {sample_count} accepted signal(s) observed after this policy change")
        else:
            improving_signals = 0
            degrading_signals = 0
            if after_completion_rate > before_completion_rate:
                improving_signals += 1
                reasons.append(f"Completion rate improved from {before_completion_rate:.1f}% to {after_completion_rate:.1f}%")
            elif after_completion_rate < before_completion_rate:
                degrading_signals += 1
                reasons.append(f"Completion rate fell from {before_completion_rate:.1f}% to {after_completion_rate:.1f}%")
            if after_negative < before_negative:
                improving_signals += 1
                reasons.append("Failed and cancelled follow-ups declined")
            elif after_negative > before_negative:
                degrading_signals += 1
                reasons.append("Failed and cancelled follow-ups increased")
            if after_block_rate < before_block_rate:
                improving_signals += 1
                reasons.append("Fewer accepted items are getting blocked by policy")
            elif after_block_rate > before_block_rate:
                degrading_signals += 1
                reasons.append("More accepted items are getting blocked by policy")

            if degrading_signals >= 2 and improving_signals == 0:
                status = "degrading"
            elif improving_signals >= 2 and degrading_signals == 0:
                status = "improving"
            else:
                status = "mixed"
            if not reasons:
                reasons.append("Launch mix and downstream outcomes are roughly flat after this policy change")

        return {
            "monitor_job_id": monitor_job_id,
            "history_entry_id": str(history_entry.get("id") or ""),
            "evaluation_status": status,
            "evaluation_sample_count": sample_count,
            "evaluation_target_count": target_count,
            "evaluation_reasons": reasons[:3],
            "before_counts": before_counts,
            "after_counts": after_counts,
            "delta_counts": delta_counts,
            "sample_items": [
                *self._policy_evaluation_samples_for_items(list(reversed(before_items)), period="before"),
                *self._policy_evaluation_samples_for_items(after_items, period="after"),
            ],
        }

    def build_policy_evaluation_detail(
        self,
        *,
        monitor_job_id: Any,
        history_entry: dict[str, Any],
        items: list[ResearchInboxItem],
    ) -> dict[str, Any]:
        return self._evaluate_policy_history_entry(
            monitor_job_id=monitor_job_id,
            history_entry=history_entry,
            items=items,
        )

    def _evaluate_customer_rebalance_history_entry(
        self,
        *,
        customer: str,
        history_entry: dict[str, Any],
        items: list[ResearchInboxItem],
        monitor_rows: list[dict[str, Any]],
        jobs_by_id: Optional[dict[Any, AgentJob]] = None,
    ) -> dict[str, Any]:
        target_count = max(3, int(history_entry.get("evaluation_target_count") or self.CUSTOMER_REBALANCE_EVALUATION_TARGET_COUNT))
        sorted_items = sorted(items, key=self._item_sort_time)
        effective_at = self._to_naive_utc(history_entry["at"]) or datetime.utcnow()
        before_candidates = [item for item in sorted_items if self._item_sort_time(item) < effective_at]
        after_candidates = [item for item in sorted_items if self._item_sort_time(item) >= effective_at]
        before_items = before_candidates[-target_count:]
        after_items = after_candidates[:target_count]

        changed_monitor_ids = {
            str(change.get("monitor_job_id") or "").strip()
            for change in (history_entry.get("changes") or [])
            if str(change.get("monitor_job_id") or "").strip()
        }
        before_monitor_rows = [row for row in monitor_rows if str(row.get("monitor_job_id") or "").strip() in changed_monitor_ids]
        if not before_monitor_rows:
            before_monitor_rows = monitor_rows
        after_monitor_rows = monitor_rows

        before_counts = self._customer_rebalance_evaluation_counts(items=before_items, monitor_rows=before_monitor_rows)
        after_counts = self._customer_rebalance_evaluation_counts(items=after_items, monitor_rows=after_monitor_rows)
        delta_counts = self._customer_rebalance_delta_counts(before_counts, after_counts)
        sample_count = int(after_counts["accepted_count"])

        reasons: list[str] = []
        if sample_count < min(target_count, 3):
            status = "insufficient_data"
            reasons.append(f"Only {sample_count} accepted signal(s) observed after this rebalance")
        else:
            improving_signals = 0
            degrading_signals = 0
            if int(after_counts["throttled_monitor_count"]) < int(before_counts["throttled_monitor_count"]):
                improving_signals += 1
                reasons.append("Fewer monitors are throttled after the rebalance")
            elif int(after_counts["throttled_monitor_count"]) > int(before_counts["throttled_monitor_count"]):
                degrading_signals += 1
                reasons.append("More monitors are throttled after the rebalance")
            if int(after_counts["backlog_used"]) < int(before_counts["backlog_used"]):
                improving_signals += 1
                reasons.append("Queue backlog pressure declined")
            elif int(after_counts["backlog_used"]) > int(before_counts["backlog_used"]):
                degrading_signals += 1
                reasons.append("Queue backlog pressure increased")
            if int(after_counts["blocked_count"]) < int(before_counts["blocked_count"]):
                improving_signals += 1
                reasons.append("Fewer accepted signals are being blocked")
            elif int(after_counts["blocked_count"]) > int(before_counts["blocked_count"]):
                degrading_signals += 1
                reasons.append("More accepted signals are being blocked")
            if int(after_counts["follow_up_completed_count"]) > int(before_counts["follow_up_completed_count"]):
                improving_signals += 1
                reasons.append("More launched follow-ups are completing")
            elif int(after_counts["follow_up_failed_count"]) + int(after_counts["follow_up_cancelled_count"]) > int(before_counts["follow_up_failed_count"]) + int(before_counts["follow_up_cancelled_count"]):
                degrading_signals += 1
                reasons.append("Failed or cancelled follow-ups increased")

            if degrading_signals >= 2 and improving_signals == 0:
                status = "degrading"
            elif improving_signals >= 2 and degrading_signals == 0:
                status = "improving"
            else:
                status = "mixed"
            if not reasons:
                reasons.append("Portfolio pressure and downstream outcomes are roughly flat after this rebalance")

        return {
            "customer": customer,
            "history_entry_id": str(history_entry.get("id") or ""),
            "evaluation_status": status,
            "evaluation_sample_count": sample_count,
            "evaluation_target_count": target_count,
            "evaluation_reasons": reasons[:3],
            "before_counts": before_counts,
            "after_counts": after_counts,
            "delta_counts": delta_counts,
            "sample_items": [
                *self._customer_rebalance_sample_items(list(reversed(before_items)), period="before", jobs_by_id=jobs_by_id),
                *self._customer_rebalance_sample_items(after_items, period="after", jobs_by_id=jobs_by_id),
            ],
        }

    def build_customer_rebalance_evaluation_detail(
        self,
        *,
        customer: str,
        history_entry: dict[str, Any],
        items: list[ResearchInboxItem],
        monitor_rows: list[dict[str, Any]],
        jobs_by_id: Optional[dict[Any, AgentJob]] = None,
    ) -> dict[str, Any]:
        return self._evaluate_customer_rebalance_history_entry(
            customer=customer,
            history_entry=history_entry,
            items=items,
            monitor_rows=monitor_rows,
            jobs_by_id=jobs_by_id,
        )

    def _empty_policy_simulation_counts(self) -> dict[str, int]:
        return {
            "auto_launch_safe_count": 0,
            "queue_for_approval_count": 0,
            "manual_only_count": 0,
            "blocked_count": 0,
            "insufficient_context_count": 0,
        }

    def _simulate_follow_up_policy_for_item(
        self,
        item: ResearchInboxItem,
        *,
        policy: dict[str, Any],
        learning_profile: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        from app.api.endpoints.agent_jobs import _build_follow_up_actions_for_inbox_item

        normalized_policy = self._normalize_policy_config(policy)
        mode = normalized_policy["mode"]
        allowlist = set(normalized_policy["allowed_recommendations"])
        actions = _build_follow_up_actions_for_inbox_item(item, learning_profile=learning_profile)
        preferred_action = next((action for action in actions if action.recommended), actions[0] if actions else None)
        selected_action = preferred_action
        if mode != "manual_only":
            for action in actions:
                recommendation_key = str(action.recommendation_key or "").strip()
                eligibility = str(action.autonomy_eligibility or "").strip().lower() or "manual_only"
                if eligibility == "auto_launchable" and recommendation_key in allowlist:
                    selected_action = action
                    break

        if selected_action is None:
            return {
                "classification": "insufficient_context",
                "recommendation_key": None,
                "reason": "No bounded follow-up recommendation could be rebuilt for this inbox item.",
            }

        recommendation_key = str(selected_action.recommendation_key or "").strip() or None
        eligibility = str(selected_action.autonomy_eligibility or "").strip().lower() or "manual_only"

        if mode == "manual_only":
            return {
                "classification": "manual_only",
                "recommendation_key": recommendation_key,
                "reason": "Policy would keep this signal manual.",
            }
        if eligibility != "auto_launchable":
            return {
                "classification": "blocked",
                "recommendation_key": recommendation_key,
                "reason": "Best bounded follow-up is not safe for autonomous launch.",
            }
        if recommendation_key and recommendation_key not in allowlist:
            return {
                "classification": "blocked",
                "recommendation_key": recommendation_key,
                "reason": "The recommended follow-up is outside the proposed allowlist.",
            }
        if mode == "queue_for_approval":
            return {
                "classification": "queue_for_approval",
                "recommendation_key": recommendation_key,
                "reason": "Policy would queue this safe follow-up for operator approval.",
            }
        return {
            "classification": "auto_launch_safe",
            "recommendation_key": recommendation_key,
            "reason": "Policy would auto-launch this safe bounded follow-up.",
        }

    def build_policy_simulation_snapshot(
        self,
        *,
        monitor_job: AgentJob,
        items: list[ResearchInboxItem],
        proposed_policy: dict[str, Any],
        learning_profile: Optional[dict[str, Any]] = None,
        history_limit: int = 25,
    ) -> dict[str, Any]:
        current_config = monitor_job.config if isinstance(getattr(monitor_job, "config", None), dict) else {}
        current_automation = self.resolve_monitor_automation_config(current_config)
        current_policy = current_automation["follow_up_autonomy"]
        proposed_raw = proposed_policy if isinstance(proposed_policy, dict) else {}
        proposed_automation = self.resolve_monitor_automation_config(
            {
                **current_config,
                "automation_profile": proposed_raw.get("automation_profile", current_automation["automation_profile"]),
                "automation_policy": (
                    proposed_raw.get("automation_policy")
                    if isinstance(proposed_raw.get("automation_policy"), dict)
                    else current_automation["automation_policy"]
                ),
            }
        )
        proposed = proposed_automation["follow_up_autonomy"]
        baseline_counts = self._empty_policy_simulation_counts()
        simulated_counts = self._empty_policy_simulation_counts()
        baseline_recommendations: Counter[str] = Counter()
        simulated_recommendations: Counter[str] = Counter()
        sample_items: list[dict[str, Any]] = []

        for item in items[:history_limit]:
            baseline = self._simulate_follow_up_policy_for_item(
                item,
                policy=current_policy,
                learning_profile=learning_profile,
            )
            simulated = self._simulate_follow_up_policy_for_item(
                item,
                policy=proposed,
                learning_profile=learning_profile,
            )
            baseline_counts[f"{baseline['classification']}_count"] += 1
            simulated_counts[f"{simulated['classification']}_count"] += 1
            if baseline.get("recommendation_key") and baseline["classification"] != "insufficient_context":
                baseline_recommendations[str(baseline["recommendation_key"])] += 1
            if simulated.get("recommendation_key") and simulated["classification"] != "insufficient_context":
                simulated_recommendations[str(simulated["recommendation_key"])] += 1
            if len(sample_items) < 6 and (
                baseline["classification"] != simulated["classification"]
                or baseline.get("recommendation_key") != simulated.get("recommendation_key")
            ):
                sample_items.append(
                    {
                        "item_id": item.id,
                        "title": str(item.title or "").strip() or "Untitled inbox item",
                        "recommendation_key": simulated.get("recommendation_key") or baseline.get("recommendation_key"),
                        "current_outcome": baseline["classification"],
                        "simulated_outcome": simulated["classification"],
                        "reason": str(simulated["reason"] or baseline["reason"]),
                    }
                )

        recommendation_keys = sorted(set(baseline_recommendations.keys()) | set(simulated_recommendations.keys()))
        recommendation_deltas = [
            {
                "recommendation_key": key,
                "baseline_count": int(baseline_recommendations.get(key) or 0),
                "simulated_count": int(simulated_recommendations.get(key) or 0),
                "delta_count": int(simulated_recommendations.get(key) or 0) - int(baseline_recommendations.get(key) or 0),
            }
            for key in recommendation_keys
        ]
        recommendation_deltas.sort(
            key=lambda row: (-abs(int(row["delta_count"])), row["recommendation_key"])
        )

        delta_counts = {
            key: int(simulated_counts[key]) - int(baseline_counts[key])
            for key in baseline_counts.keys()
        }
        return {
            "monitor_job_id": monitor_job.id,
            "current_policy": current_policy,
            "proposed_policy": proposed,
            "current_automation_profile": current_automation["automation_profile"],
            "current_automation_policy": current_automation["automation_policy"],
            "current_effective_policy": current_automation["effective_policy"],
            "proposed_automation_profile": proposed_automation["automation_profile"],
            "proposed_automation_policy": proposed_automation["automation_policy"],
            "proposed_effective_policy": proposed_automation["effective_policy"],
            "history_limit": int(history_limit),
            "baseline_counts": baseline_counts,
            "simulated_counts": simulated_counts,
            "delta_counts": delta_counts,
            "top_recommendation_deltas": recommendation_deltas[:4],
            "sample_items": sample_items,
            "insufficient_context_count": int(simulated_counts["insufficient_context_count"]),
        }

    async def build_policy_simulation(
        self,
        *,
        db: AsyncSession,
        user_id: UUID,
        monitor_job_id: UUID,
        proposed_policy: dict[str, Any],
        history_limit: int = 25,
    ) -> tuple[Optional[AgentJob], dict[str, Any]]:
        monitor_job = await db.get(AgentJob, monitor_job_id)
        if not monitor_job or monitor_job.user_id != user_id:
            return None, {}
        stmt = (
            select(ResearchInboxItem)
            .where(
                ResearchInboxItem.user_id == user_id,
                ResearchInboxItem.job_id == monitor_job_id,
                ResearchInboxItem.status == "accepted",
            )
            .order_by(ResearchInboxItem.updated_at.desc())
            .limit(int(max(5, min(history_limit, 100))))
        )
        res = await db.execute(stmt)
        items = list(res.scalars().all())
        from app.api.endpoints.agent_jobs import _load_follow_up_learning_profile

        customer = next((str(item.customer or "").strip() for item in items if str(item.customer or "").strip()), None)

        learning_profile = await _load_follow_up_learning_profile(
            db=db,
            user_id=user_id,
            customer=customer or None,
        )
        return monitor_job, self.build_policy_simulation_snapshot(
            monitor_job=monitor_job,
            items=items,
            proposed_policy=proposed_policy,
            learning_profile=learning_profile,
            history_limit=history_limit,
        )

    def _health_summary_for_bucket(self, bucket: dict[str, Any]) -> tuple[float, str, list[str]]:
        discovered = int(bucket.get("discovered_count", 0) or 0)
        accepted = int(bucket.get("accepted_count", 0) or 0)
        auto_launched = int(bucket.get("auto_launched_count", 0) or 0)
        approval_launched = int(bucket.get("approval_launched_count", 0) or 0)
        blocked = int(bucket.get("blocked_count", 0) or 0)
        completed = int(bucket.get("follow_up_completed_count", 0) or 0)
        failed = int(bucket.get("follow_up_failed_count", 0) or 0)
        cancelled = int(bucket.get("follow_up_cancelled_count", 0) or 0)

        launched = auto_launched + approval_launched
        terminal = completed + failed + cancelled
        acceptance_rate = self._safe_rate(accepted, discovered)
        completion_rate = self._safe_rate(completed, terminal)
        autonomy_rate = self._safe_rate(launched, accepted)

        score = (acceptance_rate * 0.45) + (completion_rate * 0.35) + (autonomy_rate * 0.20)
        reasons: list[str] = []

        if discovered < 3:
            score = min(score, 55.0)
            reasons.append("Limited history")
        if acceptance_rate >= 60.0:
            reasons.append("High acceptance rate")
        elif discovered >= 4 and acceptance_rate < 25.0:
            score -= 15.0
            reasons.append("Low acceptance rate")

        if terminal > 0 and completed >= max(2, failed + cancelled):
            reasons.append("Launched follow-ups are completing reliably")
        elif terminal > 0 and (failed + cancelled) > completed:
            score -= 15.0
            reasons.append("More launched follow-ups are failing than completing")

        if accepted > 0 and blocked >= max(2, accepted // 2 or 1):
            score -= 10.0
            reasons.append("Many accepted items are blocked by policy")

        score = max(0.0, min(round(score, 1), 100.0))
        if score >= 70.0:
            bucket_name = "strong"
        elif score >= 40.0:
            bucket_name = "mixed"
        else:
            bucket_name = "weak"

        if not reasons:
            reasons.append("Needs more operator feedback")
        return score, bucket_name, reasons[:3]

    def _policy_guidance_for_bucket(self, bucket: dict[str, Any]) -> tuple[str, list[str], list[str], str]:
        effective_policy = bucket.get("effective_policy") if isinstance(bucket.get("effective_policy"), dict) else {}
        automation_policy = bucket.get("automation_policy") if isinstance(bucket.get("automation_policy"), dict) else {}
        current_mode = str(
            effective_policy.get("follow_up_review_mode")
            or automation_policy.get("follow_up_review_mode")
            or "manual_only"
        ).strip().lower() or "manual_only"
        current_allowed = self._normalize_allowed_recommendations(
            effective_policy.get("allowed_recommendations")
            if isinstance(effective_policy.get("allowed_recommendations"), list)
            else automation_policy.get("allowed_recommendations")
        )
        recommendation_rows = list(bucket.get("recommendation_counts", {}).values())

        accepted = int(bucket.get("accepted_count", 0) or 0)
        blocked = int(bucket.get("blocked_count", 0) or 0)
        completed = int(bucket.get("follow_up_completed_count", 0) or 0)
        failed = int(bucket.get("follow_up_failed_count", 0) or 0)
        cancelled = int(bucket.get("follow_up_cancelled_count", 0) or 0)
        health_bucket = str(bucket.get("health_bucket") or "").strip().lower()
        acceptance_rate = float(bucket.get("acceptance_rate", 0.0) or 0.0)
        latest_evaluation_status = str(bucket.get("latest_policy_evaluation_status") or "").strip().lower()
        latest_evaluation_sample_count = int(bucket.get("latest_policy_evaluation_sample_count", 0) or 0)

        recommended_mode = current_mode
        policy_reasons: list[str] = []

        if latest_evaluation_status == "degrading" and latest_evaluation_sample_count >= 3:
            recommended_mode = "queue_for_approval" if current_mode == "auto_launch_safe" else "manual_only"
            policy_reasons.append("Latest policy change is degrading follow-up outcomes")
        elif health_bucket == "strong" and acceptance_rate >= 60.0 and completed >= max(2, failed + cancelled):
            recommended_mode = "auto_launch_safe"
            policy_reasons.append("Discovery quality and follow-up outcomes support safe auto-launch")
        elif health_bucket == "weak" or failed > completed or blocked >= max(2, accepted // 2 or 1):
            recommended_mode = "manual_only"
            if blocked >= max(2, accepted // 2 or 1):
                policy_reasons.append("Too many accepted items are blocked or failing")
            else:
                policy_reasons.append("Recent follow-up outcomes are too weak for autonomy")
        else:
            recommended_mode = "queue_for_approval"
            policy_reasons.append("Signals are useful, but follow-ups still need operator confirmation")

        recommended_allowed: list[str] = []
        for key in self.SAFE_AUTONOMY_RECOMMENDATIONS:
            row = next((candidate for candidate in recommendation_rows if candidate.get("recommendation_key") == key), None)
            if row is None:
                if recommended_mode != "manual_only":
                    recommended_allowed.append(key)
                continue
            completed_count = int(row.get("completed_count", 0) or 0)
            failed_count = int(row.get("failed_count", 0) or 0)
            cancelled_count = int(row.get("cancelled_count", 0) or 0)
            launch_count = int(row.get("launch_count", 0) or 0)
            negative_count = failed_count + cancelled_count
            if launch_count >= 2 and negative_count > completed_count:
                policy_reasons.append(f"Remove {key}: recent launches underperform")
                continue
            recommended_allowed.append(key)

        if recommended_mode == "manual_only":
            recommended_allowed = recommended_allowed or current_allowed or list(self.SAFE_AUTONOMY_RECOMMENDATIONS)
        elif not recommended_allowed:
            recommended_allowed = list(self.SAFE_AUTONOMY_RECOMMENDATIONS)

        if recommended_mode == current_mode and recommended_allowed == current_allowed:
            policy_reasons.append("Current policy already matches observed monitor performance")

        confidence = "low"
        if health_bucket == "strong" and accepted >= 4:
            confidence = "high"
        elif accepted >= 2:
            confidence = "medium"

        deduped_reasons: list[str] = []
        for reason in policy_reasons:
            if reason not in deduped_reasons:
                deduped_reasons.append(reason)
        return recommended_mode, recommended_allowed, deduped_reasons[:3], confidence

    def _policy_guardrail_for_bucket(self, bucket: dict[str, Any]) -> tuple[Optional[str], Optional[str], list[str], Optional[str], Optional[dict[str, Any]]]:
        latest_status = str(bucket.get("latest_policy_evaluation_status") or "").strip().lower()
        sample_count = int(bucket.get("latest_policy_evaluation_sample_count", 0) or 0)
        effective_policy = bucket.get("effective_policy") if isinstance(bucket.get("effective_policy"), dict) else {}
        automation_policy = bucket.get("automation_policy") if isinstance(bucket.get("automation_policy"), dict) else {}
        current_policy = self._normalize_policy_config(
            {
                "mode": effective_policy.get("follow_up_review_mode") or automation_policy.get("follow_up_review_mode"),
                "allowed_recommendations": (
                    effective_policy.get("allowed_recommendations")
                    if isinstance(effective_policy.get("allowed_recommendations"), list)
                    else automation_policy.get("allowed_recommendations")
                ),
            }
        )
        recent_history = list(bucket.get("recent_policy_history") or [])
        if latest_status != "degrading" or sample_count < 3 or not recent_history:
            return None, None, [], None, None

        latest_entry = recent_history[0]
        latest_entry_id = str(latest_entry.get("id") or "").strip() or None
        previous_policy = self._normalize_policy_config(latest_entry.get("previous_follow_up_autonomy"))
        reasons = list(bucket.get("latest_policy_evaluation_reasons") or [])[:2]
        if reasons:
            reasons = [*reasons, "Apply a more conservative policy until outcomes recover"][:3]
        else:
            reasons = ["Latest policy change is degrading follow-up outcomes", "Apply a more conservative policy until outcomes recover"]

        if latest_entry_id and previous_policy != current_policy:
            return "active", "rollback", reasons[:3], latest_entry_id, previous_policy

        current_mode = str(current_policy.get("mode") or "manual_only").strip().lower() or "manual_only"
        if current_mode == "auto_launch_safe":
            next_policy = {
                "mode": "queue_for_approval",
                "allowed_recommendations": list(current_policy.get("allowed_recommendations") or list(self.SAFE_AUTONOMY_RECOMMENDATIONS)),
            }
        elif current_mode == "queue_for_approval":
            next_policy = {
                "mode": "manual_only",
                "allowed_recommendations": list(current_policy.get("allowed_recommendations") or list(self.SAFE_AUTONOMY_RECOMMENDATIONS)),
            }
        else:
            return None, None, [], None, None
        return "active", "downgrade", reasons[:3], None, next_policy

    def build_effectiveness_snapshot(
        self,
        *,
        items: list[ResearchInboxItem],
        jobs_by_id: dict[Any, AgentJob],
        notification_counts_by_job: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        monitor_buckets: dict[str, dict[str, Any]] = {}
        recommendation_buckets: dict[str, dict[str, Any]] = {}
        totals = {
            "total_monitors": 0,
            "discovered_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "auto_launched_count": 0,
            "approval_launched_count": 0,
            "blocked_count": 0,
            "follow_up_completed_count": 0,
            "follow_up_failed_count": 0,
            "follow_up_cancelled_count": 0,
            "strong_monitors": 0,
            "mixed_monitors": 0,
            "weak_monitors": 0,
        }

        for item in items:
            customer = (str(item.customer or "").strip() or None)
            job = jobs_by_id.get(item.job_id)
            monitor_key = str(item.job_id) if item.job_id else f"unattributed:{customer or 'global'}"
            monitor_bucket = monitor_buckets.setdefault(
                monitor_key,
                {
                    "monitor_job_id": item.job_id,
                    "monitor_name": str(job.name).strip() if job and getattr(job, "name", None) else "Unattributed inbox items",
                    "monitor_job_type": str(job.job_type).strip() if job and getattr(job, "job_type", None) else None,
                    "customer": customer,
                    "automation_profile": "balanced",
                    "automation_policy": {},
                    "effective_policy": {},
                    "autonomy_mode": "balanced",
                    "current_policy_mode": "manual_only",
                    "current_allowed_recommendations": list(self.SAFE_AUTONOMY_RECOMMENDATIONS),
                    "autonomy_budget": dict(self.DEFAULT_AUTONOMY_BUDGET),
                    "budget_usage": self._empty_budget_usage(),
                    "budget_remaining": self._empty_budget_usage(),
                    "budget_throttle_state": "normal",
                    "budget_throttle_reasons": [],
                    "budget_history_count": 0,
                    "latest_budget_changed_at": None,
                    "latest_budget_change_source": None,
                    "latest_budget_actor_user_id": None,
                    "latest_budget_change_reason": None,
                    "policy_history_count": 0,
                    "latest_policy_changed_at": None,
                    "latest_policy_change_source": None,
                    "latest_policy_actor_user_id": None,
                    "latest_policy_evaluation_status": None,
                    "latest_policy_evaluation_sample_count": 0,
                    "latest_policy_evaluation_target_count": 0,
                    "latest_policy_evaluation_reasons": [],
                    "policy_guardrail_status": None,
                    "policy_guardrail_action": None,
                    "policy_guardrail_reasons": [],
                    "policy_guardrail_target_history_entry_id": None,
                    "policy_guardrail_follow_up_autonomy": None,
                    "discovered_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "auto_launched_count": 0,
                    "approval_launched_count": 0,
                    "queued_for_approval_count": 0,
                    "manual_only_count": 0,
                    "blocked_count": 0,
                    "follow_up_completed_count": 0,
                    "follow_up_failed_count": 0,
                    "follow_up_cancelled_count": 0,
                    "relaunch_count": 0,
                    "suppressed_relaunches_count": 0,
                    "policy_mode_counts": Counter(),
                    "recent_policy_history": [],
                    "recommendation_counts": {},
                    "_accepted_items": [],
                },
            )
            if job and isinstance(getattr(job, "config", None), dict):
                automation = self.resolve_monitor_automation_config(job.config)
                effective_policy = automation["effective_policy"] if isinstance(automation.get("effective_policy"), dict) else {}
                automation_policy = automation["automation_policy"] if isinstance(automation.get("automation_policy"), dict) else {}
                monitor_bucket["automation_profile"] = automation["automation_profile"]
                monitor_bucket["automation_policy"] = automation_policy
                monitor_bucket["effective_policy"] = effective_policy
                monitor_bucket["autonomy_mode"] = automation["automation_profile"]
                monitor_bucket.update(
                    build_monitor_policy_compat_fields(
                        automation_profile=automation["automation_profile"],
                        automation_policy=automation_policy,
                        effective_policy=effective_policy,
                        default_allowed=list(self.SAFE_AUTONOMY_RECOMMENDATIONS),
                    )
                )
                monitor_bucket["autonomy_budget"] = self._normalize_budget_config(job.config.get("autonomy_budget"))
            if job:
                budget_history = self._budget_history_for_job(job)
                if budget_history:
                    monitor_bucket["budget_history_count"] = (
                        len(job.results.get(self.BUDGET_HISTORY_KEY))
                        if isinstance(getattr(job, "results", None), dict) and isinstance(job.results.get(self.BUDGET_HISTORY_KEY), list)
                        else len(budget_history)
                    )
                    monitor_bucket["latest_budget_changed_at"] = budget_history[0]["at"]
                    monitor_bucket["latest_budget_change_source"] = budget_history[0].get("change_source")
                    monitor_bucket["latest_budget_actor_user_id"] = budget_history[0].get("actor_user_id")
                    monitor_bucket["latest_budget_change_reason"] = budget_history[0].get("change_reason")
                policy_history = self._policy_history_for_job(job)
                if policy_history:
                    monitor_bucket["policy_history_count"] = (
                        len(job.results.get(self.POLICY_HISTORY_KEY))
                        if isinstance(getattr(job, "results", None), dict) and isinstance(job.results.get(self.POLICY_HISTORY_KEY), list)
                        else len(policy_history)
                    )
                    monitor_bucket["latest_policy_changed_at"] = policy_history[0]["at"]
                    monitor_bucket["latest_policy_change_source"] = policy_history[0].get("change_source")
                    monitor_bucket["latest_policy_actor_user_id"] = policy_history[0].get("actor_user_id")
                    monitor_bucket["recent_policy_history"] = policy_history

            totals["discovered_count"] += 1
            monitor_bucket["discovered_count"] += 1

            status = str(item.status or "").strip().lower()
            if status == "accepted":
                totals["accepted_count"] += 1
                monitor_bucket["accepted_count"] += 1
                monitor_bucket["_accepted_items"].append(item)
            elif status == "rejected":
                totals["rejected_count"] += 1
                monitor_bucket["rejected_count"] += 1

            policy_mode = str(item.follow_up_policy_mode or "").strip().lower()
            if policy_mode:
                monitor_bucket["policy_mode_counts"][policy_mode] += 1

            decision = str(item.follow_up_decision or "").strip().lower()
            launch_status = str(item.follow_up_launch_status or "").strip().lower()
            operator_decision = str(item.follow_up_operator_decision or "").strip().lower()
            outcome_status = str(item.follow_up_outcome_status or "").strip().lower()
            recommendation_key = str(item.follow_up_recommendation_key or "").strip()
            budget_decision = str(item.follow_up_budget_decision or "").strip().lower()
            customer_budget_decision = str(item.follow_up_customer_budget_decision or "").strip().lower()

            if decision == "manual":
                monitor_bucket["manual_only_count"] += 1
            if decision == "auto_launched":
                totals["auto_launched_count"] += 1
                monitor_bucket["auto_launched_count"] += 1
            if operator_decision == "approved_launch":
                totals["approval_launched_count"] += 1
                monitor_bucket["approval_launched_count"] += 1
            if launch_status == "pending_approval":
                monitor_bucket["queued_for_approval_count"] += 1
            if launch_status == "blocked":
                totals["blocked_count"] += 1
                monitor_bucket["blocked_count"] += 1
            if decision == "relaunched":
                monitor_bucket["relaunch_count"] += 1
            if launch_status in {"blocked", "rejected"} and decision in {"manual", "queued_for_approval"}:
                monitor_bucket["suppressed_relaunches_count"] += 1
            if budget_decision or customer_budget_decision:
                monitor_bucket["suppressed_relaunches_count"] += 1

            if outcome_status == "completed":
                totals["follow_up_completed_count"] += 1
                monitor_bucket["follow_up_completed_count"] += 1
            elif outcome_status == "failed":
                totals["follow_up_failed_count"] += 1
                monitor_bucket["follow_up_failed_count"] += 1
            elif outcome_status == "cancelled":
                totals["follow_up_cancelled_count"] += 1
                monitor_bucket["follow_up_cancelled_count"] += 1

            if recommendation_key:
                monitor_recommendations = monitor_bucket["recommendation_counts"]
                monitor_recommendation_bucket = monitor_recommendations.setdefault(
                    recommendation_key,
                    {
                        "recommendation_key": recommendation_key,
                        "launch_count": 0,
                        "auto_launch_count": 0,
                        "approval_launch_count": 0,
                        "blocked_count": 0,
                        "completed_count": 0,
                        "failed_count": 0,
                        "cancelled_count": 0,
                    },
                )
                recommendation_bucket = recommendation_buckets.setdefault(
                    recommendation_key,
                    {
                        "recommendation_key": recommendation_key,
                        "launch_count": 0,
                        "auto_launch_count": 0,
                        "approval_launch_count": 0,
                        "blocked_count": 0,
                        "completed_count": 0,
                        "failed_count": 0,
                        "cancelled_count": 0,
                        "monitor_keys": set(),
                    },
                )
                recommendation_bucket["monitor_keys"].add(monitor_key)

                if launch_status == "launched":
                    monitor_recommendation_bucket["launch_count"] += 1
                    recommendation_bucket["launch_count"] += 1
                if decision == "auto_launched":
                    monitor_recommendation_bucket["auto_launch_count"] += 1
                    recommendation_bucket["auto_launch_count"] += 1
                if operator_decision == "approved_launch":
                    monitor_recommendation_bucket["approval_launch_count"] += 1
                    recommendation_bucket["approval_launch_count"] += 1
                if launch_status == "blocked":
                    monitor_recommendation_bucket["blocked_count"] += 1
                    recommendation_bucket["blocked_count"] += 1
                if outcome_status == "completed":
                    monitor_recommendation_bucket["completed_count"] += 1
                    recommendation_bucket["completed_count"] += 1
                elif outcome_status == "failed":
                    monitor_recommendation_bucket["failed_count"] += 1
                    recommendation_bucket["failed_count"] += 1
                elif outcome_status == "cancelled":
                    monitor_recommendation_bucket["cancelled_count"] += 1
                    recommendation_bucket["cancelled_count"] += 1

        monitor_rows: list[dict[str, Any]] = []
        for bucket in monitor_buckets.values():
            bucket["acceptance_rate"] = self._safe_rate(bucket["accepted_count"], bucket["discovered_count"])
            accepted_items = list(bucket.get("_accepted_items") or [])
            if bucket.get("monitor_job_id") and bucket.get("recent_policy_history"):
                evaluated_history = [
                    {
                        **entry,
                        **self._evaluate_policy_history_entry(
                            monitor_job_id=bucket["monitor_job_id"],
                            history_entry=entry,
                            items=accepted_items,
                        ),
                    }
                    for entry in bucket["recent_policy_history"]
                ]
                bucket["recent_policy_history"] = evaluated_history
                latest_evaluation = evaluated_history[0]
                bucket["latest_policy_evaluation_status"] = latest_evaluation.get("evaluation_status")
                bucket["latest_policy_evaluation_sample_count"] = int(latest_evaluation.get("evaluation_sample_count", 0) or 0)
                bucket["latest_policy_evaluation_target_count"] = int(latest_evaluation.get("evaluation_target_count", 0) or 0)
                bucket["latest_policy_evaluation_reasons"] = list(latest_evaluation.get("evaluation_reasons") or [])

            (
                bucket["policy_guardrail_status"],
                bucket["policy_guardrail_action"],
                bucket["policy_guardrail_reasons"],
                bucket["policy_guardrail_target_history_entry_id"],
                bucket["policy_guardrail_target_policy"],
            ) = self._policy_guardrail_for_bucket(bucket)
            budget_usage = self._budget_usage_for_monitor(
                items=accepted_items,
                alert_count_24h=(notification_counts_by_job or {}).get(str(bucket.get("monitor_job_id") or ""), 0),
            )
            budget_config = self._normalize_budget_config(bucket.get("autonomy_budget"))
            budget_remaining = self._budget_remaining(budget_config, budget_usage)
            budget_throttle_state, budget_throttle_reasons = self._derive_budget_throttle(
                policy_mode=str(
                    ((bucket.get("effective_policy") or {}) if isinstance(bucket.get("effective_policy"), dict) else {}).get(
                        "follow_up_review_mode"
                    )
                    or "manual_only"
                ),
                budget_config=budget_config,
                budget_usage=budget_usage,
            )
            bucket["autonomy_budget"] = budget_config
            bucket["budget_usage"] = budget_usage
            bucket["budget_remaining"] = budget_remaining
            bucket["budget_throttle_state"] = budget_throttle_state
            bucket["budget_throttle_reasons"] = budget_throttle_reasons
            bucket["budget_clamp_state"] = None if budget_throttle_state == "normal" else budget_throttle_state
            bucket["budget_clamp_reasons"] = list(budget_throttle_reasons)

            health_score, health_bucket, health_reasons = self._health_summary_for_bucket(bucket)
            bucket["health_score"] = health_score
            bucket["health_bucket"] = health_bucket
            bucket["health_reasons"] = health_reasons
            (
                bucket["recommended_policy_mode"],
                bucket["recommended_allowed_recommendations"],
                bucket["policy_reasons"],
                bucket["policy_confidence"],
            ) = self._policy_guidance_for_bucket(bucket)
            bucket["policy_guardrail_state"] = bucket.get("policy_guardrail_status")
            bucket.update(
                build_monitor_policy_compat_fields(
                    automation_profile=bucket.get("automation_profile"),
                    automation_policy=bucket.get("automation_policy"),
                    effective_policy=bucket.get("effective_policy"),
                    default_allowed=list(self.SAFE_AUTONOMY_RECOMMENDATIONS),
                    target_policy=bucket.get("policy_guardrail_target_policy"),
                )
            )
            bucket["policy_mode_counts"] = {
                str(k): int(v) for k, v in dict(bucket["policy_mode_counts"]).items() if int(v) > 0
            }
            bucket["follow_up_review_counts"] = {
                "auto_launch_safe": int(bucket.get("auto_launched_count", 0) or 0),
                "queue_for_approval": int(bucket.get("queued_for_approval_count", 0) or 0),
                "manual_only": int(bucket.get("manual_only_count", 0) or 0),
                "blocked": int(bucket.get("blocked_count", 0) or 0),
            }
            bucket["scheduler_summary"] = {
                "scheduling_mode": "continuous",
                "last_evaluated_at": None,
                "last_dispatched_at": None,
                "auto_launches_count": int(bucket.get("auto_launched_count", 0) or 0),
                "queued_approvals_count": int(bucket.get("queued_for_approval_count", 0) or 0),
                "manual_recommendations_count": int(bucket.get("manual_only_count", 0) or 0),
                "blocked_by_policy_count": int(bucket.get("blocked_count", 0) or 0),
                "blocked_by_budget_count": 0 if budget_throttle_state == "normal" else 1,
                "suppressed_relaunches_count": int(bucket.get("suppressed_relaunches_count", 0) or 0),
            }

            top_recommendations: list[dict[str, Any]] = []
            for recommendation in bucket["recommendation_counts"].values():
                success_rate = self._safe_rate(
                    recommendation["completed_count"],
                    recommendation["completed_count"] + recommendation["failed_count"] + recommendation["cancelled_count"],
                )
                top_recommendations.append(
                    {
                        **recommendation,
                        "success_rate": success_rate,
                        "score_trend": self._score_trend(
                            completed_count=recommendation["completed_count"],
                            failed_count=recommendation["failed_count"],
                            cancelled_count=recommendation["cancelled_count"],
                        ),
                        "monitor_count": 1,
                    }
                )
            top_recommendations.sort(
                key=lambda row: (
                    -(row["completed_count"] * 3 + row["launch_count"]),
                    row["recommendation_key"],
                )
            )
            bucket["top_recommendations"] = top_recommendations[:3]
            bucket.pop("recommendation_counts", None)
            bucket.pop("_accepted_items", None)
            monitor_rows.append(bucket)

            if health_bucket == "strong":
                totals["strong_monitors"] += 1
            elif health_bucket == "mixed":
                totals["mixed_monitors"] += 1
            else:
                totals["weak_monitors"] += 1

        monitor_rows.sort(
            key=lambda row: (
                -float(row.get("health_score", 0.0) or 0.0),
                -int(row.get("accepted_count", 0) or 0),
                str(row.get("monitor_name") or ""),
            )
        )
        totals["total_monitors"] = len(monitor_rows)

        customer_buckets: dict[str, dict[str, Any]] = {}
        for row in monitor_rows:
            customer_name = str(row.get("customer") or "").strip() or "Unassigned"
            bucket = customer_buckets.setdefault(
                customer_name,
                {
                    "customer": customer_name,
                    "monitor_count": 0,
                    "strong_monitor_count": 0,
                    "mixed_monitor_count": 0,
                    "weak_monitor_count": 0,
                    "auto_launch_used_24h": 0,
                    "auto_launch_capacity_24h": 0,
                    "approval_queue_used_24h": 0,
                    "approval_queue_capacity_24h": 0,
                    "alert_used_24h": 0,
                    "alert_capacity_24h": 0,
                    "backlog_used": 0,
                    "backlog_capacity": 0,
                    "throttled_monitor_count": 0,
                    "accepted_count": 0,
                    "blocked_count": 0,
                    "follow_up_completed_count": 0,
                    "follow_up_failed_count": 0,
                    "follow_up_cancelled_count": 0,
                    "rebalance_guidance_status": "none",
                    "rebalance_guidance_reasons": [],
                    "rebalance_guidance_summary": None,
                    "rebalance_guidance_changes": [],
                    "latest_rebalance_evaluation_status": None,
                    "latest_rebalance_evaluation_sample_count": 0,
                    "latest_rebalance_evaluation_target_count": 0,
                    "latest_rebalance_evaluation_reasons": [],
                    "recent_rebalance_history": [],
                    "_launch_rows": [],
                    "_backlog_rows": [],
                    "_alert_rows": [],
                    "_throttled_rows": [],
                },
            )
            bucket["monitor_count"] += 1
            if str(row.get("health_bucket") or "") == "strong":
                bucket["strong_monitor_count"] += 1
            elif str(row.get("health_bucket") or "") == "mixed":
                bucket["mixed_monitor_count"] += 1
            else:
                bucket["weak_monitor_count"] += 1

            budget_usage = row.get("budget_usage") if isinstance(row.get("budget_usage"), dict) else {}
            budget_config = row.get("autonomy_budget") if isinstance(row.get("autonomy_budget"), dict) else {}
            auto_used = int(budget_usage.get("auto_launch_count_24h", 0) or 0)
            approval_used = int(budget_usage.get("approval_queue_count_24h", 0) or 0)
            alert_used = int(budget_usage.get("alert_count_24h", 0) or 0)
            backlog_used = int(budget_usage.get("queue_backlog_count", 0) or 0)
            auto_capacity = int(budget_config.get("auto_launch_limit_24h", 0) or 0)
            approval_capacity = int(budget_config.get("approval_queue_limit_24h", 0) or 0)
            alert_capacity = int(budget_config.get("alert_limit_24h", 0) or 0)
            backlog_capacity = int(budget_config.get("queue_backlog_cap", 0) or 0)
            throttle_state = str(row.get("budget_throttle_state") or "normal").strip().lower() or "normal"

            bucket["auto_launch_used_24h"] += auto_used
            bucket["auto_launch_capacity_24h"] += auto_capacity
            bucket["approval_queue_used_24h"] += approval_used
            bucket["approval_queue_capacity_24h"] += approval_capacity
            bucket["alert_used_24h"] += alert_used
            bucket["alert_capacity_24h"] += alert_capacity
            bucket["backlog_used"] += backlog_used
            bucket["backlog_capacity"] += backlog_capacity
            bucket["accepted_count"] += int(row.get("accepted_count", 0) or 0)
            bucket["blocked_count"] += int(row.get("blocked_count", 0) or 0)
            bucket["follow_up_completed_count"] += int(row.get("follow_up_completed_count", 0) or 0)
            bucket["follow_up_failed_count"] += int(row.get("follow_up_failed_count", 0) or 0)
            bucket["follow_up_cancelled_count"] += int(row.get("follow_up_cancelled_count", 0) or 0)
            if throttle_state != "normal":
                bucket["throttled_monitor_count"] += 1

            contributor_base = {
                "monitor_job_id": row.get("monitor_job_id"),
                "monitor_name": str(row.get("monitor_name") or "Unknown monitor"),
                "customer": row.get("customer"),
                "throttle_state": throttle_state if throttle_state != "normal" else None,
            }
            bucket["_launch_rows"].append({**contributor_base, "value": auto_used})
            bucket["_backlog_rows"].append({**contributor_base, "value": backlog_used})
            bucket["_alert_rows"].append({**contributor_base, "value": alert_used})
            if throttle_state != "normal":
                severity = 2 if throttle_state == "manual_only_clamped" else 1
                bucket["_throttled_rows"].append({**contributor_base, "value": severity})

        customer_rows: list[dict[str, Any]] = []
        for bucket in customer_buckets.values():
            status, reasons = self._customer_portfolio_status_for_bucket(bucket)
            bucket["portfolio_status"] = status
            bucket["portfolio_reasons"] = reasons
            bucket["top_launch_monitors"] = sorted(
                [row for row in bucket.pop("_launch_rows", []) if int(row.get("value", 0) or 0) > 0],
                key=lambda row: (-int(row.get("value", 0) or 0), str(row.get("monitor_name") or "")),
            )[:3]
            bucket["top_backlog_monitors"] = sorted(
                [row for row in bucket.pop("_backlog_rows", []) if int(row.get("value", 0) or 0) > 0],
                key=lambda row: (-int(row.get("value", 0) or 0), str(row.get("monitor_name") or "")),
            )[:3]
            bucket["top_alert_monitors"] = sorted(
                [row for row in bucket.pop("_alert_rows", []) if int(row.get("value", 0) or 0) > 0],
                key=lambda row: (-int(row.get("value", 0) or 0), str(row.get("monitor_name") or "")),
            )[:3]
            bucket["throttled_monitors"] = sorted(
                bucket.pop("_throttled_rows", []),
                key=lambda row: (-int(row.get("value", 0) or 0), str(row.get("monitor_name") or "")),
            )[:3]
            (
                bucket["rebalance_guidance_status"],
                bucket["rebalance_guidance_reasons"],
                bucket["rebalance_guidance_summary"],
                bucket["rebalance_guidance_changes"],
            ) = self._build_customer_rebalance_guidance(
                customer_row=bucket,
                monitor_rows=[
                    row for row in monitor_rows
                    if str(row.get("customer") or "").strip() == str(bucket.get("customer") or "").strip()
                ],
            )
            customer_rows.append(bucket)
        customer_rows.sort(
            key=lambda row: (
                -int(row.get("throttled_monitor_count", 0) or 0),
                -int(row.get("backlog_used", 0) or 0),
                str(row.get("customer") or ""),
            )
        )

        recommendation_rows: list[dict[str, Any]] = []
        for recommendation in recommendation_buckets.values():
            success_rate = self._safe_rate(
                recommendation["completed_count"],
                recommendation["completed_count"] + recommendation["failed_count"] + recommendation["cancelled_count"],
            )
            recommendation_rows.append(
                {
                    "recommendation_key": recommendation["recommendation_key"],
                    "launch_count": int(recommendation["launch_count"]),
                    "auto_launch_count": int(recommendation["auto_launch_count"]),
                    "approval_launch_count": int(recommendation["approval_launch_count"]),
                    "blocked_count": int(recommendation["blocked_count"]),
                    "completed_count": int(recommendation["completed_count"]),
                    "failed_count": int(recommendation["failed_count"]),
                    "cancelled_count": int(recommendation["cancelled_count"]),
                    "success_rate": success_rate,
                    "score_trend": self._score_trend(
                        completed_count=recommendation["completed_count"],
                        failed_count=recommendation["failed_count"],
                        cancelled_count=recommendation["cancelled_count"],
                    ),
                    "monitor_count": len(recommendation["monitor_keys"]),
                }
            )
        recommendation_rows.sort(
            key=lambda row: (
                -int(row.get("completed_count", 0) or 0),
                -int(row.get("launch_count", 0) or 0),
                str(row.get("recommendation_key") or ""),
            )
        )

        return {
            "generated_at": datetime.utcnow(),
            "totals": totals,
            "customers": customer_rows,
            "monitors": monitor_rows,
            "recommendations": recommendation_rows[:8],
        }

    async def build_effectiveness_analytics(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: Optional[str] = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        stmt = (
            select(ResearchInboxItem)
            .where(ResearchInboxItem.user_id == user_id)
            .order_by(ResearchInboxItem.updated_at.desc())
            .limit(int(max(100, min(limit, 5000))))
        )
        if customer:
            stmt = stmt.where(ResearchInboxItem.customer == customer)
        res = await db.execute(stmt)
        items = list(res.scalars().all())

        job_ids = [item.job_id for item in items if item.job_id is not None]
        jobs_by_id: dict[Any, AgentJob] = {}
        if job_ids:
            jobs_stmt = select(AgentJob).where(AgentJob.id.in_(job_ids))
            jobs_res = await db.execute(jobs_stmt)
            jobs_by_id = {job.id: job for job in jobs_res.scalars().all()}
        notification_counts_by_job: dict[str, int] = {}
        if job_ids:
            window_start = datetime.utcnow().timestamp() - 86400
            notif_stmt = select(Notification).where(
                Notification.user_id == user_id,
                Notification.notification_type.in_([
                    NotificationType.QUEUE_URGENCY_ALERT,
                    NotificationType.POLICY_GUARDRAIL_ALERT,
                    NotificationType.AUTONOMY_BUDGET_ALERT,
                ]),
            )
            notif_res = await db.execute(notif_stmt)
            for notification in notif_res.scalars().all():
                created_at = getattr(notification, "created_at", None)
                if not isinstance(created_at, datetime) or created_at.timestamp() < window_start:
                    continue
                data = notification.data if isinstance(notification.data, dict) else {}
                job_key = str(data.get("job_id") or "").strip()
                if job_key:
                    notification_counts_by_job[job_key] = notification_counts_by_job.get(job_key, 0) + 1

        snapshot = self.build_effectiveness_snapshot(
            items=items,
            jobs_by_id=jobs_by_id,
            notification_counts_by_job=notification_counts_by_job,
        )
        customer_names = [
            str(row.get("customer") or "").strip()
            for row in snapshot.get("customers", [])
            if str(row.get("customer") or "").strip()
        ]
        if customer_names:
            profiles_stmt = select(ResearchMonitorProfile).where(
                and_(
                    ResearchMonitorProfile.user_id == user_id,
                    ResearchMonitorProfile.customer.in_(customer_names),
                )
            )
            profiles = {
                str(profile.customer or "").strip(): profile
                for profile in (await db.execute(profiles_stmt)).scalars().all()
                if str(profile.customer or "").strip()
            }
            for row in snapshot.get("customers", []):
                customer_name = str(row.get("customer") or "").strip()
                profile = profiles.get(customer_name)
                budget_config = self._normalize_customer_budget_config(
                    getattr(profile, "customer_budget_config", None) if profile is not None else None
                )
                usage = {
                    "auto_launch_count_24h": int(row.get("auto_launch_used_24h", 0) or 0),
                    "approval_queue_count_24h": int(row.get("approval_queue_used_24h", 0) or 0),
                    "alert_count_24h": int(row.get("alert_used_24h", 0) or 0),
                    "queue_backlog_count": int(row.get("backlog_used", 0) or 0),
                }
                throttle_state, throttle_reasons = self._derive_budget_throttle(
                    policy_mode="auto_launch_safe",
                    budget_config=budget_config,
                    budget_usage=usage,
                )
                row["customer_budget"] = budget_config
                row["customer_budget_usage"] = usage
                row["customer_budget_remaining"] = self._budget_remaining(budget_config, usage)
                row["customer_budget_throttle_state"] = throttle_state
                row["customer_budget_throttle_reasons"] = throttle_reasons
                if throttle_state != "normal":
                    existing_reasons = [str(reason).strip() for reason in (row.get("portfolio_reasons") or []) if str(reason).strip()]
                    row["portfolio_reasons"] = list(dict.fromkeys([*throttle_reasons[:3], *existing_reasons]))[:4]
                    if str(row.get("portfolio_status") or "") == "normal":
                        row["portfolio_status"] = "customer_budget_throttled"
                (
                    row["rebalance_guidance_status"],
                    row["rebalance_guidance_reasons"],
                    row["rebalance_guidance_summary"],
                    row["rebalance_guidance_changes"],
                ) = self._build_customer_rebalance_guidance(
                    customer_row=row,
                    monitor_rows=[
                        monitor_row for monitor_row in snapshot.get("monitors", [])
                        if str(monitor_row.get("customer") or "").strip() == customer_name
                    ],
                )
                rebalance_history = self._customer_rebalance_history_for_profile(profile)
                if rebalance_history:
                    accepted_items = [
                        item for item in items
                        if str(item.customer or "").strip() == customer_name and str(item.status or "").strip().lower() == "accepted"
                    ]
                    customer_monitor_rows = [
                        monitor_row for monitor_row in snapshot.get("monitors", [])
                        if str(monitor_row.get("customer") or "").strip() == customer_name
                    ]
                    evaluated_history = [
                        {
                            **entry,
                            **self._evaluate_customer_rebalance_history_entry(
                                customer=customer_name,
                                history_entry=entry,
                                items=accepted_items,
                                monitor_rows=customer_monitor_rows,
                                jobs_by_id=jobs_by_id,
                            ),
                        }
                        for entry in rebalance_history
                    ]
                    row["recent_rebalance_history"] = evaluated_history
                    latest_rebalance = evaluated_history[0]
                    row["latest_rebalance_evaluation_status"] = latest_rebalance.get("evaluation_status")
                    row["latest_rebalance_evaluation_sample_count"] = int(latest_rebalance.get("evaluation_sample_count", 0) or 0)
                    row["latest_rebalance_evaluation_target_count"] = int(latest_rebalance.get("evaluation_target_count", 0) or 0)
                    row["latest_rebalance_evaluation_reasons"] = list(latest_rebalance.get("evaluation_reasons") or [])
                    if str(latest_rebalance.get("evaluation_status") or "").strip().lower() == "degrading":
                        row["rebalance_guidance_status"] = "none"
                        row["rebalance_guidance_summary"] = None
                        row["rebalance_guidance_changes"] = []
                        row["rebalance_guidance_reasons"] = list(dict.fromkeys([
                            "The latest rebalance degraded customer portfolio outcomes.",
                            *list(latest_rebalance.get("evaluation_reasons") or []),
                        ]))[:3]
        return snapshot

    async def build_customer_rebalance_preview(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: str,
        monitor_budget_updates: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        snapshot = await self.build_effectiveness_analytics(db=db, user_id=user_id, customer=customer)
        customer_name = str(customer or "").strip()
        customer_row = next(
            (row for row in snapshot.get("customers", []) if str(row.get("customer") or "").strip() == customer_name),
            None,
        )
        monitor_rows = [
            row for row in snapshot.get("monitors", [])
            if str(row.get("customer") or "").strip() == customer_name
        ]
        before_capacity = self._customer_capacity_from_changes(base_rows=monitor_rows, changes_by_monitor={})

        default_changes = customer_row.get("rebalance_guidance_changes") if isinstance(customer_row, dict) else []
        raw_changes = monitor_budget_updates if isinstance(monitor_budget_updates, list) and monitor_budget_updates else default_changes
        normalized_changes: list[dict[str, Any]] = []
        changes_by_monitor: dict[str, dict[str, int]] = {}
        monitor_by_id = {str(row.get("monitor_job_id") or ""): row for row in monitor_rows}
        for change in raw_changes:
            if not isinstance(change, dict):
                continue
            monitor_id = str(change.get("monitor_job_id") or "").strip()
            if not monitor_id or monitor_id not in monitor_by_id:
                continue
            monitor_row = monitor_by_id[monitor_id]
            current_budget = self._normalize_budget_config(
                change.get("current_budget") if isinstance(change.get("current_budget"), dict) else monitor_row.get("autonomy_budget")
            )
            proposed_budget = self._normalize_budget_config(
                change.get("proposed_budget") if isinstance(change.get("proposed_budget"), dict) else change
            )
            changes_by_monitor[monitor_id] = proposed_budget
            normalized_changes.append(
                {
                    "monitor_job_id": monitor_row.get("monitor_job_id"),
                    "monitor_name": monitor_row.get("monitor_name"),
                    "customer": monitor_row.get("customer"),
                    "current_budget": current_budget,
                    "proposed_budget": proposed_budget,
                    "delta_budget": {
                        key: int(proposed_budget.get(key, 0) or 0) - int(current_budget.get(key, 0) or 0)
                        for key in self.DEFAULT_AUTONOMY_BUDGET.keys()
                    },
                    "reasons": [str(reason).strip() for reason in (change.get("reasons") or []) if str(reason).strip()],
                }
            )
        after_capacity = self._customer_capacity_from_changes(base_rows=monitor_rows, changes_by_monitor=changes_by_monitor)
        return {
            "customer": customer_name,
            "guidance_status": str(customer_row.get("rebalance_guidance_status") or "none") if isinstance(customer_row, dict) else "none",
            "guidance_summary": customer_row.get("rebalance_guidance_summary") if isinstance(customer_row, dict) else None,
            "guidance_reasons": list(customer_row.get("rebalance_guidance_reasons") or []) if isinstance(customer_row, dict) else [],
            "before_capacity": before_capacity,
            "after_capacity": after_capacity,
            "changes": normalized_changes,
        }

    async def recompute_profile(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: Optional[str],
        limit: int = 500,
    ) -> ResearchMonitorProfile:
        """
        Recompute token scores from accepted/rejected inbox items and upsert the profile.
        """
        stmt = (
            select(ResearchInboxItem.status, ResearchInboxItem.title, ResearchInboxItem.summary)
            .where(
                ResearchInboxItem.user_id == user_id,
                ResearchInboxItem.status.in_(["accepted", "rejected"]),
            )
            .order_by(ResearchInboxItem.updated_at.desc())
            .limit(int(max(50, min(limit, 2000))))
        )
        if customer:
            stmt = stmt.where(ResearchInboxItem.customer == customer)
        else:
            stmt = stmt.where(ResearchInboxItem.customer.is_(None))

        res = await db.execute(stmt)
        rows = res.all()

        pos = Counter()
        neg = Counter()
        pos_phrases = Counter()
        neg_phrases = Counter()
        recommendation_scores = Counter()
        source_type_scores = Counter()
        outcome_counters = Counter()
        for status, title, summary in rows:
            text = f"{title or ''} {summary or ''}"
            toks = self.tokenize(text)
            phrases = self.extract_phrases(text)
            if not toks:
                continue
            if str(status) == "accepted":
                pos.update(toks)
                pos_phrases.update(phrases)
            elif str(status) == "rejected":
                neg.update(toks)
                neg_phrases.update(phrases)

        outcome_stmt = (
            select(
                ResearchInboxItem.follow_up_recommendation_key,
                ResearchInboxItem.follow_up_launch_status,
                ResearchInboxItem.follow_up_operator_decision,
                ResearchInboxItem.item_type,
                ResearchInboxItem.follow_up_outcome_status,
            )
            .where(
                ResearchInboxItem.user_id == user_id,
                ResearchInboxItem.status == "accepted",
            )
            .order_by(ResearchInboxItem.updated_at.desc())
            .limit(int(max(50, min(limit, 2000))))
        )
        if customer:
            outcome_stmt = outcome_stmt.where(ResearchInboxItem.customer == customer)
        else:
            outcome_stmt = outcome_stmt.where(ResearchInboxItem.customer.is_(None))

        outcome_rows = (await db.execute(outcome_stmt)).all()
        for recommendation_key, launch_status, operator_decision, item_type, outcome_status in outcome_rows:
            rec_key = str(recommendation_key or "").strip()
            launch_status_text = str(launch_status or "").strip().lower()
            operator_decision_text = str(operator_decision or "").strip().lower()
            item_type_text = str(item_type or "").strip().lower()
            outcome_status_text = str(outcome_status or "").strip().lower()

            if item_type_text:
                if launch_status_text == "launched":
                    source_type_scores[item_type_text] += 2
                elif launch_status_text in {"failed", "blocked", "rejected"}:
                    source_type_scores[item_type_text] -= 1

            if rec_key:
                if launch_status_text == "launched":
                    recommendation_scores[rec_key] += 3
                    outcome_counters["launched"] += 1
                elif launch_status_text == "pending_approval":
                    recommendation_scores[rec_key] += 1
                    outcome_counters["pending_approval"] += 1
                elif launch_status_text == "failed":
                    recommendation_scores[rec_key] -= 3
                    outcome_counters["failed"] += 1
                elif launch_status_text == "blocked":
                    recommendation_scores[rec_key] -= 1
                    outcome_counters["blocked"] += 1
                elif launch_status_text == "rejected":
                    recommendation_scores[rec_key] -= 2
                    outcome_counters["rejected"] += 1

            if operator_decision_text == "approved_launch":
                outcome_counters["approved_launch"] += 1
                if rec_key:
                    recommendation_scores[rec_key] += 2
            elif operator_decision_text == "rejected":
                outcome_counters["operator_rejected"] += 1
                if rec_key:
                    recommendation_scores[rec_key] -= 2

            if outcome_status_text == "completed":
                outcome_counters["completed_follow_up"] += 1
                if rec_key:
                    recommendation_scores[rec_key] += 5
                if item_type_text:
                    source_type_scores[item_type_text] += 3
            elif outcome_status_text == "failed":
                outcome_counters["failed_follow_up"] += 1
                if rec_key:
                    recommendation_scores[rec_key] -= 4
                if item_type_text:
                    source_type_scores[item_type_text] -= 2
            elif outcome_status_text == "cancelled":
                outcome_counters["cancelled_follow_up"] += 1
                if rec_key:
                    recommendation_scores[rec_key] -= 2
                if item_type_text:
                    source_type_scores[item_type_text] -= 1

        scores: dict[str, int] = {}
        for t, c in pos.items():
            scores[t] = scores.get(t, 0) + int(c)
        for t, c in neg.items():
            scores[t] = scores.get(t, 0) - int(c)

        phrase_scores: dict[str, int] = {}
        for t, c in pos_phrases.items():
            phrase_scores[t] = phrase_scores.get(t, 0) + int(c)
        for t, c in neg_phrases.items():
            phrase_scores[t] = phrase_scores.get(t, 0) - int(c)

        # Keep only meaningful tokens to bound JSON size.
        pruned = {t: int(s) for t, s in scores.items() if abs(int(s)) >= 2}
        pruned_phrases = {t: int(s) for t, s in phrase_scores.items() if abs(int(s)) >= 2}
        pruned_recommendations = {t: int(s) for t, s in recommendation_scores.items() if abs(int(s)) >= 1}
        pruned_source_types = {t: int(s) for t, s in source_type_scores.items() if abs(int(s)) >= 1}
        normalized_outcomes = {t: int(v) for t, v in outcome_counters.items() if int(v) != 0}

        existing_res = await db.execute(
            select(ResearchMonitorProfile).where(
                and_(
                    ResearchMonitorProfile.user_id == user_id,
                    (ResearchMonitorProfile.customer == customer) if customer else ResearchMonitorProfile.customer.is_(None),
                )
            ).limit(1)
        )
        profile = existing_res.scalar_one_or_none()
        if profile:
            profile.token_scores = pruned
            profile.phrase_scores = pruned_phrases
            profile.recommendation_scores = pruned_recommendations
            profile.source_type_scores = pruned_source_types
            profile.outcome_counters = normalized_outcomes
            profile.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(profile)
            return profile

        profile = ResearchMonitorProfile(
            user_id=user_id,
            customer=customer,
            token_scores=pruned,
            phrase_scores=pruned_phrases,
            recommendation_scores=pruned_recommendations,
            source_type_scores=pruned_source_types,
            outcome_counters=normalized_outcomes,
            customer_budget_config=self._normalize_customer_budget_config(None),
            customer_rebalance_history=[],
            muted_tokens=[],
            muted_patterns=[],
            notes=None,
        )
        db.add(profile)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            # Race; load and update.
            existing_res = await db.execute(
                select(ResearchMonitorProfile).where(
                    and_(
                        ResearchMonitorProfile.user_id == user_id,
                        (ResearchMonitorProfile.customer == customer) if customer else ResearchMonitorProfile.customer.is_(None),
                    )
                ).limit(1)
            )
            profile = existing_res.scalar_one()
            profile.token_scores = pruned
            profile.phrase_scores = pruned_phrases
            profile.recommendation_scores = pruned_recommendations
            profile.source_type_scores = pruned_source_types
            profile.outcome_counters = normalized_outcomes
            profile.updated_at = datetime.utcnow()
            await db.commit()
        await db.refresh(profile)
        return profile

    async def get_profile(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: Optional[str],
    ) -> Optional[ResearchMonitorProfile]:
        stmt = select(ResearchMonitorProfile).where(ResearchMonitorProfile.user_id == user_id)
        if customer:
            stmt = stmt.where(ResearchMonitorProfile.customer == customer)
        else:
            stmt = stmt.where(ResearchMonitorProfile.customer.is_(None))
        res = await db.execute(stmt.limit(1))
        return res.scalar_one_or_none()


research_monitor_profile_service = ResearchMonitorProfileService()
