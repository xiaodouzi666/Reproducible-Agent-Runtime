"""
Tracer - Main interface for recording execution traces.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from .schema import TraceEntry, TraceEventType, RunMetadata, EvidenceAnchor
from .store import TraceStore


class Tracer:
    """
    Main tracing interface for recording multi-agent execution.
    Provides convenient methods for logging different types of events.
    """

    def __init__(self, run_id: Optional[str] = None, store: Optional[TraceStore] = None):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.store = store or TraceStore()
        self._step_counter = 0
        self._metadata: Optional[RunMetadata] = None
        self._conversation_id = str(uuid.uuid4())[:8]

    def start_run(
        self,
        task_description: str,
        spec_file: str = "",
        seed: Optional[int] = None,
        llm_mode: bool = False,
        is_replay: bool = False,
        original_run_id: str = ""
    ) -> str:
        """Initialize a new run."""
        self._metadata = RunMetadata(
            run_id=self.run_id,
            task_description=task_description,
            start_time=datetime.now().isoformat(),
            spec_file=spec_file,
            seed=seed,
            llm_mode=llm_mode,
            is_replay=is_replay,
            original_run_id=original_run_id
        )
        self.store.create_run(self.run_id, self._metadata)

        # Log system start
        self._log(
            event_type=TraceEventType.SYSTEM_INFO,
            agent_id="system",
            content={"action": "run_started", "task": task_description},
            content_summary=f"Run started: {task_description[:50]}..."
        )

        return self.run_id

    def end_run(
        self,
        status: str = "completed",
        final_answer: str = "",
        evidence_summary: list = None
    ):
        """Finalize a run."""
        if self._metadata:
            self._metadata.end_time = datetime.now().isoformat()
            self._metadata.status = status
            self._metadata.total_steps = self._step_counter
            self._metadata.final_answer = final_answer
            self._metadata.evidence_summary = evidence_summary or []

            # Calculate totals
            entries = self.store.get_entries(self.run_id)
            self._metadata.total_latency_ms = sum(e.latency_ms for e in entries)

            # Collect unique agents and tools used
            self._metadata.agents_used = list(set(e.agent_id for e in entries if e.agent_id))
            self._metadata.tools_used = list(set(e.tool_name for e in entries if e.tool_name))

            self.store.update_metadata(self.run_id, self._metadata)

        # Log system end
        self._log(
            event_type=TraceEventType.SYSTEM_INFO,
            agent_id="system",
            content={"action": "run_ended", "status": status},
            content_summary=f"Run ended: {status}"
        )

    def _log(self, **kwargs) -> TraceEntry:
        """Internal logging method."""
        self._step_counter += 1
        entry = TraceEntry(
            run_id=self.run_id,
            step_id=self._step_counter,
            conversation_id=self._conversation_id,
            **kwargs
        )
        self.store.append_entry(self.run_id, entry)
        return entry

    # ===== Agent lifecycle =====

    def log_agent_start(self, agent_id: str, agent_role: str, beliefs: dict = None):
        """Log agent activation."""
        return self._log(
            event_type=TraceEventType.AGENT_START,
            agent_id=agent_id,
            agent_role=agent_role,
            beliefs=beliefs or {},
            content_summary=f"Agent {agent_id} ({agent_role}) started"
        )

    def log_agent_end(self, agent_id: str, agent_role: str):
        """Log agent deactivation."""
        return self._log(
            event_type=TraceEventType.AGENT_END,
            agent_id=agent_id,
            agent_role=agent_role,
            content_summary=f"Agent {agent_id} ended"
        )

    # ===== BDI logging =====

    def log_belief_update(
        self,
        agent_id: str,
        beliefs: dict,
        reason: str = ""
    ):
        """Log a belief update."""
        return self._log(
            event_type=TraceEventType.BELIEF_UPDATE,
            agent_id=agent_id,
            beliefs=beliefs,
            content={"reason": reason},
            content_summary=f"Beliefs updated: {reason}"
        )

    def log_desire(self, agent_id: str, desires: list):
        """Log agent desires/goals."""
        return self._log(
            event_type=TraceEventType.DESIRE_SET,
            agent_id=agent_id,
            desires=desires,
            content_summary=f"Desires set: {len(desires)} goal(s)"
        )

    def log_intention(self, agent_id: str, intentions: list):
        """Log agent intentions/plans."""
        return self._log(
            event_type=TraceEventType.INTENTION_FORM,
            agent_id=agent_id,
            intentions=intentions,
            content_summary=f"Intentions formed: {len(intentions)} plan(s)"
        )

    # ===== Message logging =====

    def log_message(
        self,
        sender: str,
        receiver: str,
        performative: str,
        content: Any,
        message_id: str = "",
        tool_name: str = "",
        evidence_anchors: list = None,
        latency_ms: float = 0.0
    ):
        """Log an ACL message."""
        content_preview = str(content)[:100]
        return self._log(
            event_type=TraceEventType.MESSAGE_SENT,
            agent_id=sender,
            sender=sender,
            receiver=receiver,
            performative=performative,
            message_id=message_id,
            content=content,
            content_summary=f"[{performative.upper()}] {sender}→{receiver}: {content_preview}",
            tool_name=tool_name,
            evidence_anchors=evidence_anchors or [],
            latency_ms=latency_ms
        )

    # ===== Contract Net logging =====

    def log_cfp(
        self,
        manager: str,
        task_id: str,
        task_type: str,
        description: str
    ):
        """Log a Call for Proposals."""
        return self._log(
            event_type=TraceEventType.CFP_ISSUED,
            agent_id=manager,
            content={
                "task_id": task_id,
                "task_type": task_type,
                "description": description
            },
            content_summary=f"CFP issued: {task_type} - {description[:50]}"
        )

    def log_bid(
        self,
        bidder: str,
        task_id: str,
        cost: float,
        latency: float,
        success_prob: float
    ):
        """Log a bid submission."""
        return self._log(
            event_type=TraceEventType.BID_SUBMITTED,
            agent_id=bidder,
            content={
                "task_id": task_id,
                "cost": cost,
                "latency": latency,
                "success_probability": success_prob
            },
            content_summary=f"Bid: cost={cost:.2f}, prob={success_prob:.0%}"
        )

    def log_contract_award(
        self,
        manager: str,
        contractor: str,
        task_id: str
    ):
        """Log a contract award."""
        return self._log(
            event_type=TraceEventType.CONTRACT_AWARDED,
            agent_id=manager,
            receiver=contractor,
            content={"task_id": task_id, "contractor": contractor},
            content_summary=f"Contract awarded to {contractor}"
        )

    # ===== Tool logging =====

    def log_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        tool_input: dict,
        description: str = ""
    ):
        """Log a tool invocation."""
        return self._log(
            event_type=TraceEventType.TOOL_CALL,
            agent_id=agent_id,
            tool_name=tool_name,
            tool_input=tool_input,
            content_summary=f"Tool call: {tool_name}"
        )

    def log_tool_result(
        self,
        agent_id: str,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        latency_ms: float,
        success: bool,
        evidence_anchors: list = None,
        artifacts: dict = None,
        error_message: str = ""
    ):
        """Log a tool result."""
        return self._log(
            event_type=TraceEventType.TOOL_RESULT,
            agent_id=agent_id,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            latency_ms=latency_ms,
            status="success" if success else "error",
            error_message=error_message,
            evidence_anchors=evidence_anchors or [],
            artifacts=artifacts or {},
            content_summary=f"Tool result: {tool_name} ({'success' if success else 'error'})"
        )

    # ===== Task logging =====

    def log_task_start(self, agent_id: str, task_id: str, description: str):
        """Log task start."""
        return self._log(
            event_type=TraceEventType.TASK_START,
            agent_id=agent_id,
            content={"task_id": task_id, "description": description},
            content_summary=f"Task started: {description[:50]}"
        )

    def log_task_complete(
        self,
        agent_id: str,
        task_id: str,
        result: Any,
        evidence_anchors: list = None
    ):
        """Log task completion."""
        return self._log(
            event_type=TraceEventType.TASK_COMPLETE,
            agent_id=agent_id,
            content={"task_id": task_id, "result": result},
            evidence_anchors=evidence_anchors or [],
            content_summary=f"Task completed: {task_id}"
        )

    def log_task_fail(self, agent_id: str, task_id: str, error: str):
        """Log task failure."""
        return self._log(
            event_type=TraceEventType.TASK_FAIL,
            agent_id=agent_id,
            status="error",
            error_message=error,
            content={"task_id": task_id, "error": error},
            content_summary=f"Task failed: {error[:50]}"
        )

    # ===== Audit logging =====

    def log_challenge(
        self,
        auditor_id: str,
        target_agent: str,
        target_step: int,
        reason: str
    ):
        """Log a challenge raised by auditor."""
        return self._log(
            event_type=TraceEventType.CHALLENGE_RAISED,
            agent_id=auditor_id,
            receiver=target_agent,
            content={
                "target_step": target_step,
                "reason": reason
            },
            content_summary=f"Challenge: {reason[:50]}"
        )

    def log_justification(
        self,
        agent_id: str,
        challenge_step: int,
        justification: str,
        evidence_anchors: list = None
    ):
        """Log justification in response to challenge."""
        return self._log(
            event_type=TraceEventType.JUSTIFICATION_PROVIDED,
            agent_id=agent_id,
            content={
                "challenge_step": challenge_step,
                "justification": justification
            },
            evidence_anchors=evidence_anchors or [],
            content_summary=f"Justification: {justification[:50]}"
        )

    def log_replan(self, planner_id: str, reason: str, new_plan: list):
        """Log a replan triggered by audit."""
        return self._log(
            event_type=TraceEventType.REPLAN_TRIGGERED,
            agent_id=planner_id,
            intentions=new_plan,
            content={"reason": reason, "new_plan": new_plan},
            content_summary=f"Replan: {reason[:50]}"
        )

    # ===== Utility =====

    def get_entries(self) -> list[TraceEntry]:
        """Get all entries for current run."""
        return self.store.get_entries(self.run_id)

    def get_metadata(self) -> Optional[RunMetadata]:
        """Get metadata for current run."""
        return self._metadata
