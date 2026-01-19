"""
ACL (Agent Communication Language) Protocol Implementation
Based on FIPA ACL specification with speech acts.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
from datetime import datetime
import uuid
import json


class Performative(Enum):
    """Speech act types for agent communication."""
    # Informative acts
    INFORM = "inform"           # Share information/facts
    CONFIRM = "confirm"         # Confirm a previously communicated proposition
    DISCONFIRM = "disconfirm"   # Deny a previously communicated proposition

    # Directive acts
    REQUEST = "request"         # Request an action to be performed
    QUERY = "query"             # Ask for information

    # Commissive acts
    PROPOSE = "propose"         # Propose a course of action
    COMMIT = "commit"           # Commit to performing an action
    ACCEPT = "accept"           # Accept a proposal
    REJECT = "reject"           # Reject a proposal

    # Declarative acts
    DECLARE = "declare"         # Make a declaration (change world state)

    # Contract Net specific
    CFP = "cfp"                 # Call for proposals
    BID = "bid"                 # Submit a bid
    AWARD = "award"             # Award a contract

    # Audit/Challenge acts
    CHALLENGE = "challenge"     # Challenge a statement/result
    RETRACT = "retract"         # Retract a previous statement
    JUSTIFY = "justify"         # Provide justification/evidence


@dataclass
class ACLMessage:
    """
    ACL Message structure for inter-agent communication.
    Designed to be fully traceable and serializable.
    """
    # Core fields
    performative: Performative
    sender: str
    receiver: str
    content: Any

    # Metadata
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None  # message_id of the message being replied to
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Protocol tracking
    protocol: Optional[str] = None  # e.g., "contract-net", "query-ref"

    # Evidence and artifacts
    evidence: Optional[list] = None  # List of evidence anchors
    artifacts: Optional[dict] = None  # Any produced artifacts (files, results)

    # Execution metadata
    tool_used: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[Any] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["performative"] = self.performative.value
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "ACLMessage":
        """Create from dictionary."""
        data = data.copy()
        data["performative"] = Performative(data["performative"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ACLMessage":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def create_reply(
        self,
        performative: Performative,
        sender: str,
        content: Any,
        **kwargs
    ) -> "ACLMessage":
        """Create a reply message to this message."""
        return ACLMessage(
            performative=performative,
            sender=sender,
            receiver=self.sender,
            content=content,
            conversation_id=self.conversation_id,
            reply_to=self.message_id,
            protocol=self.protocol,
            **kwargs
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        content_preview = str(self.content)[:50]
        if len(str(self.content)) > 50:
            content_preview += "..."
        return (
            f"[{self.performative.value.upper()}] "
            f"{self.sender} → {self.receiver}: {content_preview}"
        )

    def format_for_display(self) -> str:
        """Format for UI display with full details."""
        lines = [
            f"━━━ ACL Message [{self.message_id}] ━━━",
            f"Performative: {self.performative.value.upper()}",
            f"From: {self.sender} → To: {self.receiver}",
            f"Time: {self.timestamp}",
        ]
        if self.conversation_id:
            lines.append(f"Conversation: {self.conversation_id}")
        if self.reply_to:
            lines.append(f"Reply to: {self.reply_to}")
        if self.protocol:
            lines.append(f"Protocol: {self.protocol}")

        lines.append(f"Content: {self.content}")

        if self.tool_used:
            lines.append(f"Tool: {self.tool_used}")
        if self.evidence:
            lines.append(f"Evidence: {len(self.evidence)} anchor(s)")
        if self.latency_ms:
            lines.append(f"Latency: {self.latency_ms:.2f}ms")

        return "\n".join(lines)


class MessageBus:
    """
    Simple message bus for routing ACL messages between agents.
    All messages are logged for tracing.
    """

    def __init__(self):
        self.messages: list[ACLMessage] = []
        self.subscribers: dict[str, list[callable]] = {}

    def subscribe(self, agent_id: str, callback: callable):
        """Subscribe an agent to receive messages."""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    def send(self, message: ACLMessage) -> None:
        """Send a message and notify the receiver."""
        self.messages.append(message)

        # Notify receiver's callbacks
        if message.receiver in self.subscribers:
            for callback in self.subscribers[message.receiver]:
                callback(message)

        # Also handle broadcast (receiver = "*")
        if message.receiver == "*":
            for agent_id, callbacks in self.subscribers.items():
                if agent_id != message.sender:
                    for callback in callbacks:
                        callback(message)

    def get_conversation(self, conversation_id: str) -> list[ACLMessage]:
        """Get all messages in a conversation."""
        return [m for m in self.messages if m.conversation_id == conversation_id]

    def get_messages_for(self, agent_id: str) -> list[ACLMessage]:
        """Get all messages received by an agent."""
        return [m for m in self.messages if m.receiver == agent_id or m.receiver == "*"]
