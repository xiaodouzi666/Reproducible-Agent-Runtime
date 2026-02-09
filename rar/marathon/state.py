"""
Marathon runtime state schemas for checkpoint/resume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointCursor(BaseModel):
    """Cursor for resumable planner loop progress."""

    round: int = 0
    step: int = 0


class WaitDirective(BaseModel):
    """WAIT request metadata."""

    wait_type: Literal["wait_seconds", "wait_until"] = "wait_seconds"
    reason: str = ""
    seconds: int = 0
    until_iso: str = ""
    next_run_at: str = ""


class CheckpointState(BaseModel):
    """Serializable checkpoint payload for marathon execution."""

    run_id: str
    source_run_id: str = ""
    task_description: str = ""
    mode: str = ""
    model: str = ""
    thinking_level: str = ""
    status: Literal["running", "waiting", "completed"] = "running"
    cursor: CheckpointCursor = Field(default_factory=CheckpointCursor)
    pending_actions: list[str] = Field(default_factory=list)
    completed_actions: list[dict[str, Any]] = Field(default_factory=list)
    subtask_results: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    audit_rounds: list[dict[str, Any]] = Field(default_factory=list)
    latest_audit_challenges: list[dict[str, Any]] = Field(default_factory=list)
    has_research: bool = False
    has_audit: bool = False
    last_audit_approved: Optional[bool] = None
    final_payload: dict[str, Any] = Field(default_factory=dict)
    gemini_history: list[Any] = Field(default_factory=list)
    wait: Optional[WaitDirective] = None
    next_run_at: str = ""
    wait_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utcnow_iso)
    updated_at: str = Field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        """Stable dict dump for JSON file persistence."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointState":
        """Validate checkpoint payload from JSON."""
        return cls.model_validate(payload or {})
