"""
Auditor Agent - Quality assurance and evidence verification.
"""

from typing import Optional

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

    # Thresholds for quality checks
    MIN_EVIDENCE_COUNT = 2
    MIN_RELEVANCE_SCORE = 0.3
    MIN_CONFIDENCE = 0.6

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

        # Update beliefs
        self.update_beliefs({
            "task_id": task.get("id", ""),
            "targets": target_subtasks,
            "results_to_audit": len(results)
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
            audit = self._audit_result(subtask_id, result)
            audit_results[subtask_id] = audit

            if not audit["approved"]:
                all_approved = False
                challenge = {
                    "target_subtask": subtask_id,
                    "reason": audit["issues"],
                    "recommendation": audit["recommendation"]
                }
                challenges.append(challenge)

                # Log challenge
                if self.tracer:
                    self.tracer.log_challenge(
                        self.agent_id,
                        "planner",
                        0,  # Would reference actual step
                        "; ".join(audit["issues"])
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
            "audit_details": audit_results
        }

        self.deactivate()
        return self.result

    def _audit_result(self, subtask_id: str, result: dict) -> dict:
        """Audit a single subtask result."""
        issues = []
        score = 1.0

        # Check success
        if not result.get("success", False):
            issues.append("Task failed")
            return {
                "approved": False,
                "score": 0,
                "issues": issues,
                "recommendation": "Retry the task"
            }

        # Check evidence
        evidence = result.get("evidence", [])
        if len(evidence) < self.MIN_EVIDENCE_COUNT:
            issues.append(f"Insufficient evidence ({len(evidence)} < {self.MIN_EVIDENCE_COUNT})")
            score -= 0.3

        # Check relevance scores
        if evidence:
            avg_relevance = sum(
                e.get("relevance_score", 0) for e in evidence
            ) / len(evidence)

            if avg_relevance < self.MIN_RELEVANCE_SCORE:
                issues.append(f"Low average relevance ({avg_relevance:.2f} < {self.MIN_RELEVANCE_SCORE})")
                score -= 0.2

        # Check output quality
        output = result.get("output", "")
        if isinstance(output, str) and len(output) < 50:
            issues.append("Output too brief")
            score -= 0.1

        # Check for error indicators
        if "error" in str(output).lower() or "failed" in str(output).lower():
            issues.append("Output contains error indicators")
            score -= 0.2

        # Determine approval
        approved = score >= self.MIN_CONFIDENCE and len(issues) <= 1

        # Generate recommendation
        if not approved:
            if "Insufficient evidence" in str(issues):
                recommendation = "Gather more evidence with additional searches"
            elif "Low average relevance" in str(issues):
                recommendation = "Refine search query for more relevant results"
            else:
                recommendation = "Review and improve the approach"
        else:
            recommendation = "Approved"

        return {
            "approved": approved,
            "score": max(0, score),
            "issues": issues,
            "recommendation": recommendation
        }

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
