"""
Base classes for tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime
import hashlib
import json


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None

    # Metadata for tracing
    tool_name: str = ""
    input_data: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Evidence anchors (for search tools)
    evidence_anchors: list = field(default_factory=list)

    # Artifacts (for exec tools)
    artifacts: dict = field(default_factory=dict)

    # For reproducibility
    output_hash: str = ""

    def __post_init__(self):
        if self.output and not self.output_hash:
            self.output_hash = self._compute_hash(self.output)

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute a hash of the output for comparison."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(data).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceAnchor:
    """
    An evidence anchor linking a result to its source.
    Used for building the evidence chain.
    """
    doc_id: str                    # Document identifier
    doc_title: str = ""            # Document title
    location: str = ""             # Line number, section, or offset
    content_hash: str = ""         # Hash of the matched content
    snippet: str = ""              # The actual matched text
    relevance_score: float = 0.0   # How relevant this evidence is

    def to_dict(self) -> dict:
        return asdict(self)

    def format_citation(self) -> str:
        """Format as a citation reference."""
        return f"[{self.doc_id}:{self.location}]"


class BaseTool(ABC):
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "Base tool"

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the tool.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._call_count = 0

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def __call__(self, **kwargs) -> ToolResult:
        """Make the tool callable."""
        self._call_count += 1
        return self.execute(**kwargs)

    def get_schema(self) -> dict:
        """Get the tool's input schema for documentation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {}
        }
