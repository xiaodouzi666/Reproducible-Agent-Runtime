"""
Auditor Agent - Quality assurance and evidence verification.
"""

import re
from typing import Any, Optional

from .base import BaseAgent, AgentState
from ..protocols.acl import Performative
from ..tracing.tracer import Tracer


class AuditorAgent(BaseAgent):
    """
    Auditor agent responsible for:
    - Verifying evidence quality and relevance
    - Challenging conclusions when evidence is weak
    - Requesting additional evidence or replanning
    """

    role = "auditor"
    description = "Quality assurance and evidence verification"

    MODE_THRESHOLDS = {
        "owl_lite": {
            "min_evidence_count": 1,
            "min_relevance_score": 0.15,
            "min_confidence": 0.45,
            "argumentation_required": False,
        },
        "owl_dl": {
            "min_evidence_count": 2,
            "min_relevance_score": 0.30,
            "min_confidence": 0.65,
            "argumentation_required": False,
        },
        "owl_full": {
            "min_evidence_count": 3,
            "min_relevance_score": 0.35,
            "min_confidence": 0.70,
            "argumentation_required": True,
        },
    }

    # Runtime failure signatures (actual execution/reporting failures).
    ERROR_PATTERNS = [
        re.compile(r"\btraceback\b", re.IGNORECASE),
        re.compile(r"\bexception\b", re.IGNORECASE),
        re.compile(r"\b(?:value|type|runtime|key|index|name|syntax)error\b", re.IGNORECASE),
        re.compile(r"\berror:\s", re.IGNORECASE),
        re.compile(r"\bfailed\b", re.IGNORECASE),
        re.compile(r"\bstack trace\b", re.IGNORECASE),
        re.compile(r"\bsegmentation fault\b", re.IGNORECASE),
    ]

    # Benign scientific contexts that include "error" terminology.
    SAFE_ERROR_CONTEXT = [
        re.compile(r"\bstandard error\b", re.IGNORECASE),
        re.compile(r"\berror analysis\b", re.IGNORECASE),
        re.compile(r"\bmeasurement error\b", re.IGNORECASE),
        re.compile(r"\berror propagation\b", re.IGNORECASE),
        re.compile(r"\bmean (absolute|squared) error\b", re.IGNORECASE),
        re.compile(r"\bconfidence interval\b", re.IGNORECASE),
        re.compile(r"\buncertainty\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        agent_id: str = "auditor",
        tracer: Optional[Tracer] = None,
        tools: dict = None
    ):
        super().__init__(agent_id, tracer, tools)
        self.challenges_raised = []

    def process(self, task: dict) -> dict:
        """
        Process an audit task.

        Args:
            task: {
                "id": str,
                "type": "audit",
                "description": str,
                "target_subtasks": list,  # Subtask IDs to audit
                "results": dict           # Results from other subtasks
            }

        Returns:
            {
                "success": bool,
                "output": str,            # Audit summary
                "evidence": list,         # Audit evidence
                "challenges": list,       # Any challenges raised
                "approved": bool          # Whether results are approved
            }
        """
        self.activate()

        results = task.get("results", {})
        target_subtasks = task.get("target_subtasks", list(results.keys()))
        mode = str(task.get("mode", "owl_lite"))
        policy = self._resolve_policy(mode=mode, policy=task.get("policy", {}))

        # Update beliefs
        self.update_beliefs({
            "task_id": task.get("id", ""),
            "targets": target_subtasks,
            "results_to_audit": len(results),
            "audit_mode": mode,
            "audit_level": policy.get("audit_level", ""),
        }, reason="Audit task received")

        # Set desires
        self.set_desires(["Verify evidence quality", "Challenge weak conclusions"])

        # Audit each result
        audit_results = {}
        challenges = []
        all_approved = True

        for subtask_id in target_subtasks:
            if subtask_id not in results:
                continue

            result = results[subtask_id]
            audit = self._audit_result(subtask_id, result, policy, mode)
            audit_results[subtask_id] = audit

            if not audit["approved"]:
                all_approved = False
                reason_text = "; ".join(audit.get("issues", [])) if audit.get("issues") else "Audit failed"
                challenge = {
                    "target_subtask": subtask_id,
                    "target_subtask_id": subtask_id,
                    "reason": reason_text,
                    "issues": audit.get("issues", []),
                    "recommendation": audit.get("recommendation", "Review and improve the approach"),
                    "suggested_fix_query": audit.get("suggested_fix_query", ""),
                    "suggested_calc": audit.get("suggested_calc", ""),
                    "severity": audit.get("severity", "medium"),
                }
                challenges.append(challenge)

                # Log challenge
                if self.tracer:
                    self.tracer.log_challenge(
                        self.agent_id,
                        "planner",
                        0,  # Would reference actual step
                        reason_text
                    )

        # Synthesize audit report
        report = self._generate_audit_report(audit_results, challenges)

        # Update beliefs
        self.update_beliefs({
            "audit_completed": True,
            "all_approved": all_approved,
            "challenges_raised": len(challenges)
        }, reason="Audit completed")

        # Send result
        if challenges:
            self.send_message(
                receiver="planner",
                performative=Performative.CHALLENGE,
                content={
                    "challenges": challenges,
                    "recommendation": "Consider replanning or gathering more evidence"
                }
            )
        else:
            self.send_message(
                receiver="planner",
                performative=Performative.CONFIRM,
                content={
                    "status": "approved",
                    "message": "All results meet quality standards"
                }
            )

        self.result = {
            "success": True,
            "output": report,
            "evidence": [],  # Audit itself doesn't produce new evidence
            "challenges": challenges,
            "approved": all_approved,
            "audit_details": audit_results,
            "audit_summary": {
                "mode": mode,
                "audit_level": policy.get("audit_level", ""),
                "approved": all_approved,
                "challenges_count": len(challenges),
                "targets_count": len(target_subtasks),
            },
        }

        self.deactivate()
        return self.result

    def _audit_result(self, subtask_id: str, result: dict, policy: dict, mode: str) -> dict:
        """Audit a single subtask result."""
        issues = []
        score = 1.0
        min_evidence_count = int(policy.get("min_evidence_count", 2))
        min_relevance_score = float(policy.get("min_relevance_score", 0.3))
        min_confidence = float(policy.get("min_confidence", 0.6))
        argumentation_required = bool(policy.get("argumentation_required", False))
        subtask_kind = self._infer_subtask_kind(result)

        # Check success
        if not result.get("success", False):
            issues.append("Task failed")
            suggested_fix_query = self._suggest_research_query(result) if subtask_kind == "research" else ""
            suggested_calc = (
                "Recompute with explicit units, show intermediate equations, and report uncertainty bounds."
                if subtask_kind == "execute"
                else ""
            )
            return {
                "approved": False,
                "score": 0,
                "issues": issues,
                "recommendation": "Retry the task",
                "suggested_fix_query": suggested_fix_query,
                "suggested_calc": suggested_calc,
                "severity": "high",
                "subtask_kind": subtask_kind,
            }

        # Check evidence
        evidence = result.get("evidence", [])
        if len(evidence) < min_evidence_count:
            issues.append(f"Insufficient evidence ({len(evidence)} < {min_evidence_count})")
            score -= 0.3

        require_content_hash = mode in {"owl_dl", "owl_full"}
        anchor_issues = self._check_evidence_anchors(evidence, require_content_hash=require_content_hash)
        if anchor_issues:
            issues.extend(anchor_issues)
            score -= min(0.3, 0.05 * len(anchor_issues))

        # Check relevance scores
        if evidence and mode in {"owl_dl", "owl_full"}:
            avg_relevance = sum(
                e.get("relevance_score", 0) for e in evidence
            ) / len(evidence)

            if avg_relevance < min_relevance_score:
                issues.append(f"Low average relevance ({avg_relevance:.2f} < {min_relevance_score})")
                score -= 0.2

        output = str(result.get("output", ""))

        # Check runtime-error indicators (avoid false positives like "standard error").
        has_runtime_error, runtime_markers = self.has_runtime_error_indicators(result, output)
        if has_runtime_error:
            marker_preview = ", ".join(runtime_markers[:3]) if runtime_markers else "unknown marker"
            issues.append(f"Output contains runtime error indicators: {marker_preview}")
            score -= 0.2

        if mode in {"owl_dl", "owl_full"} and subtask_kind == "execute":
            consistency_issues = self._check_execute_consistency(output)
            if consistency_issues:
                issues.extend(consistency_issues)
                score -= min(0.25, 0.08 * len(consistency_issues))

        # Lite only enforces minimal coverage; DL/Full enforces stronger output quality.
        if mode == "owl_lite":
            if len(evidence) == 0 and len(output.strip()) < 20:
                issues.append("No evidence anchors and output too short for verification")
                score -= 0.2
        else:
            if len(output.strip()) < 50:
                issues.append("Output too brief")
                score -= 0.1

        if argumentation_required:
            claim_count = self._estimate_claim_count(output)
            if claim_count < 2:
                issues.append("Argumentation mode expects explicit claims/counter-claims")
                score -= 0.2

        # Determine approval
        blocking_markers = ["Task failed", "Output contains runtime error indicators"]
        if mode in {"owl_dl", "owl_full"}:
            blocking_markers.extend(
                [
                    "Insufficient evidence",
                    "missing doc_id",
                    "missing location",
                    "missing content_hash",
                    "Inconsistent unit",
                    "Inconsistent numeric value",
                ]
            )
        has_blocking_issue = any(
            any(marker in issue for marker in blocking_markers)
            for issue in issues
        )
        approved = (score >= min_confidence and len(issues) <= 1) and not has_blocking_issue

        # Generate recommendation
        if not approved:
            if "Insufficient evidence" in str(issues) or "missing doc_id" in str(issues):
                recommendation = "Gather more evidence with additional searches"
            elif subtask_kind == "execute" and any("unit" in i.lower() or "inconsistent" in i.lower() for i in issues):
                recommendation = "Re-run computation with explicit units and consistency checks"
            elif "Low average relevance" in str(issues):
                recommendation = "Refine search query for more relevant results"
            else:
                recommendation = "Review and improve the approach"
        else:
            recommendation = "Approved"

        suggested_fix_query = ""
        suggested_calc = ""
        if not approved:
            if subtask_kind == "research":
                suggested_fix_query = self._suggest_research_query(result)
            elif subtask_kind == "execute":
                suggested_calc = (
                    "Recompute with explicit units, show intermediate equations, and report uncertainty bounds."
                )

        severity = "low"
        if not approved:
            severity = "medium"
            if len(issues) >= 3 or any("failed" in i.lower() for i in issues):
                severity = "high"

        return {
            "approved": approved,
            "score": max(0, score),
            "issues": issues,
            "recommendation": recommendation,
            "suggested_fix_query": suggested_fix_query,
            "suggested_calc": suggested_calc,
            "severity": severity,
            "subtask_kind": subtask_kind,
            "argumentation_suggestions": self._argumentation_suggestions(result) if mode == "owl_full" else [],
        }

    def _resolve_policy(self, mode: str, policy: dict) -> dict:
        """Resolve audit policy from mode defaults + task-level overrides."""
        mode_key = mode if mode in self.MODE_THRESHOLDS else "owl_lite"
        base = dict(self.MODE_THRESHOLDS[mode_key])
        base["audit_level"] = "light" if mode_key == "owl_lite" else ("strict" if mode_key == "owl_dl" else "argumentation")

        if isinstance(policy, dict):
            base.update(policy)
        return base

    def _estimate_claim_count(self, output: str) -> int:
        """Heuristic claim counter for argumentation mode."""
        text = str(output or "").lower()
        markers = ["claim", "counter", "support", "attack", "evidence"]
        return sum(1 for m in markers if m in text)

    def _check_evidence_anchors(self, evidence: list, require_content_hash: bool) -> list[str]:
        """Validate evidence anchor structure for reproducibility."""
        issues = []
        for idx, item in enumerate(evidence or [], 1):
            if not isinstance(item, dict):
                issues.append(f"Evidence #{idx} is not a structured anchor")
                continue
            if not item.get("doc_id"):
                issues.append(f"Evidence #{idx} missing doc_id")
            if not item.get("location"):
                issues.append(f"Evidence #{idx} missing location")
            doc_id = str(item.get("doc_id", ""))
            if require_content_hash and not item.get("content_hash") and not doc_id.startswith("computation_"):
                issues.append(f"Evidence #{idx} missing content_hash")
        return issues

    def _infer_subtask_kind(self, result: dict) -> str:
        """Infer producer type from payload shape."""
        if not isinstance(result, dict):
            return "unknown"
        if "raw_results" in result:
            return "research"
        if "raw_output" in result or "artifacts" in result:
            return "execute"
        if "audit_details" in result or "approved" in result:
            return "audit"
        return "unknown"

    def _check_execute_consistency(self, output: str) -> list[str]:
        """Simple numeric/unit consistency checks for DL/Full."""
        issues: list[str] = []
        text = str(output or "")
        if not text.strip():
            return ["Execution output is empty"]

        pattern = re.compile(
            r"([A-Za-z][A-Za-z0-9_\-\(\)\s]{1,40})\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([A-Za-z/%\^\-0-9]*)",
            re.IGNORECASE,
        )
        by_metric: dict[str, list[tuple[float, str]]] = {}
        for match in pattern.finditer(text):
            metric = " ".join(match.group(1).strip().lower().split())
            try:
                value = float(match.group(2))
            except ValueError:
                continue
            unit = match.group(3).strip().lower()
            by_metric.setdefault(metric, []).append((value, unit))

        for metric, values in by_metric.items():
            if len(values) < 2:
                continue
            units = {u for _, u in values if u}
            if len(units) > 1:
                issues.append(f"Inconsistent unit for '{metric}': {', '.join(sorted(units))}")
            nums = [v for v, _ in values]
            max_v = max(nums)
            min_v = min(nums)
            if max_v != 0 and abs(max_v - min_v) / abs(max_v) > 0.25:
                issues.append(f"Inconsistent numeric value for '{metric}' across output lines")

        lowered = text.lower()
        if "activation energy" in lowered or "ea" in lowered:
            ea_values = []
            for metric, values in by_metric.items():
                if "activation energy" in metric or metric.strip() in {"ea", "ea_kj_mol"}:
                    ea_values.extend([v for v, _ in values])
            for value in ea_values:
                if value <= 0 or value > 2000:
                    issues.append(f"Activation energy appears out of expected range: {value}")
                    break
        return issues

    def _suggest_research_query(self, result: dict) -> str:
        """Generate a focused follow-up search query from current output."""
        output = str(result.get("output", "")).lower()
        if "activation energy" in output:
            return (
                "activation energy thermal analysis methods uncertainty comparison "
                "Friedman FWO KAS Starink ICTAC recommendations"
            )
        if "uncertainty" in output:
            return "method uncertainty confidence interval error sources evidence anchors"
        return "expand evidence with method comparison and uncertainty quantification"

    def _argumentation_suggestions(self, result: dict) -> list[dict]:
        """Generate minimal claim/counter-claim suggestions for OWL Full."""
        if not str(result.get("output", "")).strip():
            return []
        return [
            {
                "claim": "Primary conclusion is supported by current evidence.",
                "counterclaim": "Evidence coverage may be incomplete or biased to one method family.",
                "required_evidence": "Add at least one independent source with conflicting interpretation.",
            }
        ]

    def has_runtime_error_indicators(self, result: dict, output_text: str) -> tuple[bool, list[str]]:
        """
        Detect runtime failures using structured signals first, then regex scan.

        Returns:
            (has_error, matched_markers)
        """
        markers: list[str] = []

        # 1) Structured execution path signals (highest precision).
        raw_output = result.get("raw_output")
        if isinstance(raw_output, dict):
            raw_success = raw_output.get("success")
            if raw_success is False:
                markers.append("raw_output.success=False")
            stderr_text = str(raw_output.get("stderr", "") or "").strip()
            if stderr_text:
                markers.append("stderr")
            err_text = str(raw_output.get("error", "") or "").strip()
            if err_text:
                markers.append(f"error:{err_text[:60]}")

        if result.get("success") is False:
            markers.append("result.success=False")
        top_level_error = str(result.get("error", "") or "").strip()
        if top_level_error:
            markers.append(f"result.error:{top_level_error[:60]}")

        if markers:
            return True, markers

        # 2) Text scan fallback with safe-context filtering.
        text = str(output_text or "")
        if not text.strip():
            return False, []

        normalized = text.lower()
        matched = []
        for pattern in self.ERROR_PATTERNS:
            for match in pattern.finditer(normalized):
                token = match.group(0)
                # Ignore known benign scientific contexts around this token.
                window_start = max(0, match.start() - 40)
                window_end = min(len(normalized), match.end() + 40)
                context = normalized[window_start:window_end]
                if any(safe.search(context) for safe in self.SAFE_ERROR_CONTEXT):
                    continue
                matched.append(token)
                if len(matched) >= 5:
                    break
            if len(matched) >= 5:
                break

        # Stronger positive: explicit traceback/error prefixes anywhere.
        if re.search(r"(traceback \(most recent call last\):|error:\s|exception:)", normalized):
            matched.append("explicit_error_prefix")

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for item in matched:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return (len(deduped) > 0), deduped

    def _generate_audit_report(self, audit_results: dict, challenges: list) -> str:
        """Generate a human-readable audit report."""
        parts = ["═══ Audit Report ═══\n"]

        for subtask_id, audit in audit_results.items():
            status = "✓ APPROVED" if audit["approved"] else "✗ NEEDS ATTENTION"
            parts.append(f"\n{subtask_id}: {status}")
            parts.append(f"  Quality Score: {audit['score']:.2f}")

            if audit["issues"]:
                parts.append("  Issues:")
                for issue in audit["issues"]:
                    parts.append(f"    - {issue}")

            parts.append(f"  Recommendation: {audit['recommendation']}")

        if challenges:
            parts.append(f"\n═══ Challenges Raised: {len(challenges)} ═══")
            for i, challenge in enumerate(challenges, 1):
                parts.append(f"\n{i}. Target: {challenge['target_subtask']}")
                parts.append(f"   Reason: {challenge['reason']}")
                parts.append(f"   Recommendation: {challenge['recommendation']}")

        return "\n".join(parts)

    def verify_evidence_chain(self, evidence: list, claim: str) -> dict:
        """Verify that evidence supports a specific claim."""
        if not evidence:
            return {
                "verified": False,
                "confidence": 0,
                "reason": "No evidence provided"
            }

        # Simple keyword matching to check relevance
        claim_words = set(claim.lower().split())
        supporting_count = 0

        for e in evidence:
            snippet = e.get("snippet", "").lower()
            snippet_words = set(snippet.split())
            overlap = len(claim_words & snippet_words)
            if overlap >= 2:  # At least 2 words match
                supporting_count += 1

        confidence = supporting_count / len(evidence) if evidence else 0

        return {
            "verified": confidence >= 0.5,
            "confidence": confidence,
            "supporting_evidence": supporting_count,
            "total_evidence": len(evidence)
        }
