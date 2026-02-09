"""
Structured output schemas for Gemini responses.
"""

from __future__ import annotations

from typing import Literal, Any

from pydantic import BaseModel, Field


# ===== Plan =====


class PlanStep(BaseModel):
    """One planner step in structured plan output."""

    step_id: str = Field(description="Unique subtask identifier")
    step_type: Literal["research", "execute", "audit", "synthesize"] = Field(
        description="Subtask category"
    )
    description: str = Field(description="Concise step description")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Optional step inputs")


class PlanSchema(BaseModel):
    """Structured planner output."""

    task: str
    rationale: str = ""
    steps: list[PlanStep] = Field(default_factory=list)


# ===== Audit =====


class AuditChallenge(BaseModel):
    """One challenge produced by auditor."""

    target_subtask_id: str = ""
    reason: str
    severity: Literal["low", "medium", "high"] = "medium"
    suggested_fix_query: str = ""
    suggested_calc: str = ""


class AuditSchema(BaseModel):
    """Structured auditor output."""

    approved: bool
    challenges: list[AuditChallenge] = Field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "medium"
    fix_suggestions: list[str] = Field(default_factory=list)


# ===== Finalize =====


class FinalizeClaim(BaseModel):
    """A claim in finalized answer."""

    claim: str
    status: Literal["supported", "contested", "uncertain"] = "supported"
    evidence_anchor_ids: list[str] = Field(default_factory=list)


class FinalizeEvidence(BaseModel):
    """One evidence reference used in finalized answer."""

    doc_id: str = ""
    location: str = ""
    content_hash: str = ""
    note: str = ""


class FinalizeSchema(BaseModel):
    """Structured finalize output required by F7."""

    final_answer_markdown: str
    claims: list[FinalizeClaim] = Field(default_factory=list)
    evidence_used: list[FinalizeEvidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: list[str] = Field(default_factory=list)


# ===== Argument Graph (F8) =====


class ArgumentNode(BaseModel):
    """A node in an argument graph."""

    id: str
    type: Literal["claim", "evidence", "counterclaim"]
    text: str
    source_anchor_id: str = ""


class ArgumentEdge(BaseModel):
    """A relation edge in an argument graph."""

    source: str
    target: str
    relation: Literal["supports", "attacks"]


class ArgumentGraphSchema(BaseModel):
    """OWL Full argument graph schema."""

    nodes: list[ArgumentNode] = Field(default_factory=list)
    edges: list[ArgumentEdge] = Field(default_factory=list)
    accepted_claim_ids: list[str] = Field(default_factory=list)
    rationale: dict[str, str] = Field(default_factory=dict)


def model_to_json_schema(model_cls: type[BaseModel]) -> dict:
    """Convert a Pydantic model class to JSON schema."""
    return model_cls.model_json_schema()
