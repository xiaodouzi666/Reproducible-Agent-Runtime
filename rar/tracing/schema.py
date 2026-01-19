"""
Schema definitions for trace entries.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime
from enum import Enum
import json
import hashlib


class TraceEventType(Enum):
    """Types of events that can be traced."""
    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"

    # Communication
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"

    # BDI (Belief-Desire-Intention)
    BELIEF_UPDATE = "belief_update"
    DESIRE_SET = "desire_set"
    INTENTION_FORM = "intention_form"

    # Contract Net
    CFP_ISSUED = "cfp_issued"
    BID_SUBMITTED = "bid_submitted"
    CONTRACT_AWARDED = "contract_awarded"

    # Tool usage
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Task/workflow
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
    SUBTASK_CREATE = "subtask_create"

    # Audit/Challenge
    CHALLENGE_RAISED = "challenge_raised"
    EVIDENCE_REQUESTED = "evidence_requested"
    JUSTIFICATION_PROVIDED = "justification_provided"
    REPLAN_TRIGGERED = "replan_triggered"

    # System
    SYSTEM_INFO = "system_info"
    ERROR = "error"


@dataclass
class EvidenceAnchor:
    """
    Evidence anchor for linking results to sources.
    """
    doc_id: str
    doc_title: str = ""
    location: str = ""  # e.g., "paragraph_3", "line_42"
    content_hash: str = ""
    snippet: str = ""
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceAnchor":
        return cls(**data)


@dataclass
class TraceEntry:
    """
    A single entry in the execution trace.
    Designed to capture all relevant information for replay and audit.
    """
    # Core identifiers
    run_id: str
    step_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Event type
    event_type: TraceEventType = TraceEventType.SYSTEM_INFO

    # Agent info
    agent_id: str = ""
    agent_role: str = ""  # planner, researcher, executor, auditor

    # Communication
    performative: str = ""  # ACL speech act
    message_id: str = ""
    conversation_id: str = ""
    sender: str = ""
    receiver: str = ""

    # Content
    content: Any = None  # Main content/payload
    content_summary: str = ""  # Short summary for display

    # Tool info
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: Any = None
    tool_output_hash: str = ""

    # Evidence
    evidence_anchors: list = field(default_factory=list)

    # Artifacts (files, figures, etc.)
    artifacts: dict = field(default_factory=dict)

    # Metrics
    latency_ms: float = 0.0
    token_count: int = 0
    cost: float = 0.0

    # Status
    status: str = "success"  # success, error, pending
    error_message: str = ""

    # BDI state (for planner)
    beliefs: dict = field(default_factory=dict)
    desires: list = field(default_factory=list)
    intentions: list = field(default_factory=list)

    # For reproducibility
    input_hash: str = ""
    output_hash: str = ""
    deterministic: bool = True  # Whether this step is deterministic

    def __post_init__(self):
        # Compute hashes if not provided
        if self.tool_input and not self.input_hash:
            self.input_hash = self._compute_hash(self.tool_input)
        if self.tool_output and not self.output_hash:
            self.output_hash = self._compute_hash(self.tool_output)

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute hash for comparison."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(data).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "TraceEntry":
        """Create from dictionary."""
        data = data.copy()
        data["event_type"] = TraceEventType(data["event_type"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TraceEntry":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def format_for_display(self) -> str:
        """Format for human-readable display."""
        lines = [
            f"[{self.step_id:03d}] {self.timestamp}",
            f"      Event: {self.event_type.value}",
            f"      Agent: {self.agent_id} ({self.agent_role})"
        ]

        if self.performative:
            lines.append(f"      Performative: {self.performative}")
        if self.tool_name:
            lines.append(f"      Tool: {self.tool_name}")
        if self.content_summary:
            lines.append(f"      Summary: {self.content_summary}")
        if self.evidence_anchors:
            lines.append(f"      Evidence: {len(self.evidence_anchors)} anchor(s)")
        if self.latency_ms:
            lines.append(f"      Latency: {self.latency_ms:.2f}ms")
        if self.status != "success":
            lines.append(f"      Status: {self.status}")
            if self.error_message:
                lines.append(f"      Error: {self.error_message}")

        return "\n".join(lines)


@dataclass
class RunMetadata:
    """Metadata about a complete run."""
    run_id: str
    task_description: str
    start_time: str
    end_time: str = ""
    status: str = "running"  # running, completed, failed, replayed

    # Configuration
    spec_file: str = ""
    seed: Optional[int] = None
    llm_mode: bool = False

    # Summary stats
    total_steps: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    agents_used: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)

    # Final result
    final_answer: str = ""
    evidence_summary: list = field(default_factory=list)

    # For comparison
    is_replay: bool = False
    original_run_id: str = ""
    matches_original: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        return cls(**data)
