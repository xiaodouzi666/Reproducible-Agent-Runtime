"""
Base agent class with BDI architecture support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from enum import Enum

from ..protocols.acl import ACLMessage, Performative
from ..tracing.tracer import Tracer


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BDIState:
    """
    Belief-Desire-Intention state for an agent.
    Visible in traces for debugging and audit.
    """
    # Beliefs: What the agent believes about the world
    beliefs: dict = field(default_factory=dict)

    # Desires: Goals/objectives the agent wants to achieve
    desires: list = field(default_factory=list)

    # Intentions: Committed plans to achieve desires
    intentions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def update_belief(self, key: str, value: Any):
        """Update a belief."""
        self.beliefs[key] = value

    def add_desire(self, desire: str):
        """Add a desire/goal."""
        if desire not in self.desires:
            self.desires.append(desire)

    def remove_desire(self, desire: str):
        """Remove a desire."""
        if desire in self.desires:
            self.desires.remove(desire)

    def add_intention(self, intention: dict):
        """Add an intention/plan step."""
        self.intentions.append(intention)

    def clear_intentions(self):
        """Clear all intentions (for replanning)."""
        self.intentions = []


class BaseAgent(ABC):
    """
    Base class for all agents in the RAR framework.
    Implements BDI architecture with tracing support.
    """

    # Class-level attributes
    role: str = "base"
    description: str = "Base agent"

    def __init__(
        self,
        agent_id: str,
        tracer: Optional[Tracer] = None,
        tools: dict = None
    ):
        self.agent_id = agent_id
        self.tracer = tracer
        self.tools = tools or {}

        # State
        self.state = AgentState.IDLE
        self.bdi = BDIState()

        # Message handling
        self.inbox: list[ACLMessage] = []
        self.outbox: list[ACLMessage] = []

        # Results
        self.result: Any = None
        self.evidence: list = []

    def activate(self):
        """Activate the agent."""
        self.state = AgentState.ACTIVE
        if self.tracer:
            self.tracer.log_agent_start(
                self.agent_id,
                self.role,
                self.bdi.beliefs
            )

    def deactivate(self):
        """Deactivate the agent."""
        self.state = AgentState.COMPLETED
        if self.tracer:
            self.tracer.log_agent_end(self.agent_id, self.role)

    def receive_message(self, message: ACLMessage):
        """Receive a message into inbox."""
        self.inbox.append(message)

    def send_message(
        self,
        receiver: str,
        performative: Performative,
        content: Any,
        **kwargs
    ) -> ACLMessage:
        """Send a message."""
        message = ACLMessage(
            performative=performative,
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            **kwargs
        )
        self.outbox.append(message)

        if self.tracer:
            self.tracer.log_message(
                sender=self.agent_id,
                receiver=receiver,
                performative=performative.value,
                content=content,
                message_id=message.message_id,
                evidence_anchors=kwargs.get("evidence", [])
            )

        return message

    def update_beliefs(self, updates: dict, reason: str = ""):
        """Update beliefs with tracing."""
        for key, value in updates.items():
            self.bdi.update_belief(key, value)

        if self.tracer:
            self.tracer.log_belief_update(
                self.agent_id,
                self.bdi.beliefs,
                reason
            )

    def set_desires(self, desires: list):
        """Set desires/goals."""
        self.bdi.desires = desires
        if self.tracer:
            self.tracer.log_desire(self.agent_id, desires)

    def set_intentions(self, intentions: list):
        """Set intentions/plan."""
        self.bdi.intentions = intentions
        if self.tracer:
            self.tracer.log_intention(self.agent_id, intentions)

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a tool with tracing."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not available")

        tool = self.tools[tool_name]

        if self.tracer:
            self.tracer.log_tool_call(
                self.agent_id,
                tool_name,
                kwargs
            )

        result = tool(**kwargs)

        if self.tracer:
            self.tracer.log_tool_result(
                agent_id=self.agent_id,
                tool_name=tool_name,
                tool_input=kwargs,
                tool_output=result.output,
                latency_ms=result.latency_ms,
                success=result.success,
                evidence_anchors=result.evidence_anchors,
                artifacts=result.artifacts,
                error_message=result.error or ""
            )

        return result

    @abstractmethod
    def process(self, task: dict) -> dict:
        """
        Process a task and return result.
        Must be implemented by subclasses.

        Args:
            task: Task specification dict

        Returns:
            Result dict with at least 'success', 'output', and 'evidence' keys
        """
        pass

    def get_state_summary(self) -> dict:
        """Get a summary of agent state for display."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "state": self.state.value,
            "bdi": self.bdi.to_dict(),
            "inbox_count": len(self.inbox),
            "outbox_count": len(self.outbox)
        }
