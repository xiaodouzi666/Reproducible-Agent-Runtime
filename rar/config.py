"""
Runtime mode configuration for OWL Lite / DL / Full.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for one reasoning mode."""

    mode: str
    label: str
    model: str
    thinking_level: str
    audit_level: str
    argumentation: bool
    redteam_model: str = ""
    evidence_rules: str = ""
    require_audit: bool = True
    require_finalize: bool = False
    allow_finalize_without_audit: bool = False
    min_evidence_per_claim: int = 1
    max_replans: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


MODE_CONFIGS: dict[str, ModeConfig] = {
    "owl_lite": ModeConfig(
        mode="owl_lite",
        label="OWL Lite",
        model="gemini-3-flash-preview",
        thinking_level="minimal",
        audit_level="light",
        argumentation=False,
        evidence_rules="At least 1 evidence anchor per key claim.",
        require_audit=False,
        require_finalize=False,
        allow_finalize_without_audit=True,
        min_evidence_per_claim=1,
        max_replans=0,
    ),
    "owl_dl": ModeConfig(
        mode="owl_dl",
        label="OWL DL",
        model="gemini-3-pro-preview",
        thinking_level="high",
        audit_level="strict",
        argumentation=False,
        evidence_rules="Coverage + consistency checks; audit must pass or replan.",
        require_audit=True,
        require_finalize=True,
        allow_finalize_without_audit=False,
        min_evidence_per_claim=1,
        max_replans=2,
    ),
    "owl_full": ModeConfig(
        mode="owl_full",
        label="OWL Full",
        model="gemini-3-pro-preview",
        thinking_level="high",
        audit_level="argumentation",
        argumentation=True,
        redteam_model="gemini-3-flash-preview",
        evidence_rules="Claim-support-attack structure required; produce argument graph.",
        require_audit=True,
        require_finalize=True,
        allow_finalize_without_audit=False,
        min_evidence_per_claim=1,
        max_replans=2,
    ),
}


def get_mode_config(mode: str) -> ModeConfig:
    """Get static mode config with fallback to owl_lite."""
    normalized = (mode or "owl_lite").strip().lower()
    return MODE_CONFIGS.get(normalized, MODE_CONFIGS["owl_lite"])


def resolve_mode_config(
    mode: str,
    model_override: Optional[str] = None,
    thinking_override: Optional[str] = None,
) -> ModeConfig:
    """
    Resolve mode config and apply optional model/thinking overrides.
    """
    base = get_mode_config(mode)
    if not model_override and not thinking_override:
        return base
    return ModeConfig(
        mode=base.mode,
        label=base.label,
        model=model_override or base.model,
        thinking_level=thinking_override or base.thinking_level,
        audit_level=base.audit_level,
        argumentation=base.argumentation,
        redteam_model=base.redteam_model,
        evidence_rules=base.evidence_rules,
        require_audit=base.require_audit,
        require_finalize=base.require_finalize,
        allow_finalize_without_audit=base.allow_finalize_without_audit,
        min_evidence_per_claim=base.min_evidence_per_claim,
        max_replans=base.max_replans,
    )
