"""
Diff Engine - Compare two runs and generate detailed diff reports.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from difflib import unified_diff, SequenceMatcher
import json

from ..tracing import TraceStore, TraceEntry, RunMetadata


@dataclass
class DiffReport:
    """A comprehensive diff report between two runs."""
    run_id_a: str
    run_id_b: str

    # Overall status
    identical: bool = False
    similarity_score: float = 0.0

    # Answer comparison
    answer_a: str = ""
    answer_b: str = ""
    answer_diff: list = field(default_factory=list)
    answer_match: bool = False

    # Step comparison
    steps_a: int = 0
    steps_b: int = 0
    step_diffs: list = field(default_factory=list)  # List of step differences
    matching_steps: int = 0
    different_steps: int = 0

    # Cost comparison
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    latency_diff_ms: float = 0.0
    latency_diff_percent: float = 0.0

    # Tool usage comparison
    tools_a: dict = field(default_factory=dict)  # tool_name -> count
    tools_b: dict = field(default_factory=dict)
    tool_diffs: list = field(default_factory=list)

    # Evidence comparison
    evidence_count_a: int = 0
    evidence_count_b: int = 0
    common_evidence: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)


class DiffEngine:
    """
    Engine for comparing two runs and generating diff reports.
    """

    def __init__(self, store: Optional[TraceStore] = None):
        self.store = store or TraceStore()

    def compare(self, run_id_a: str, run_id_b: str) -> DiffReport:
        """
        Compare two runs and generate a comprehensive diff report.

        Args:
            run_id_a: First run ID
            run_id_b: Second run ID

        Returns:
            DiffReport with detailed comparison
        """
        report = DiffReport(run_id_a=run_id_a, run_id_b=run_id_b)

        # Load data
        metadata_a = self.store.get_metadata(run_id_a)
        metadata_b = self.store.get_metadata(run_id_b)
        entries_a = self.store.get_entries(run_id_a)
        entries_b = self.store.get_entries(run_id_b)

        if not metadata_a or not metadata_b:
            return report

        # Compare answers
        self._compare_answers(report, metadata_a, metadata_b)

        # Compare steps
        self._compare_steps(report, entries_a, entries_b)

        # Compare costs
        self._compare_costs(report, entries_a, entries_b)

        # Compare tool usage
        self._compare_tools(report, entries_a, entries_b)

        # Compare evidence
        self._compare_evidence(report, entries_a, entries_b)

        # Calculate overall similarity
        report.similarity_score = self._calculate_similarity(report)
        report.identical = report.similarity_score >= 0.99

        return report

    def _compare_answers(
        self,
        report: DiffReport,
        metadata_a: RunMetadata,
        metadata_b: RunMetadata
    ):
        """Compare final answers."""
        report.answer_a = metadata_a.final_answer or ""
        report.answer_b = metadata_b.final_answer or ""

        # Check exact match (normalized)
        norm_a = " ".join(report.answer_a.lower().split())
        norm_b = " ".join(report.answer_b.lower().split())
        report.answer_match = norm_a == norm_b

        # Generate unified diff
        if not report.answer_match:
            lines_a = report.answer_a.splitlines(keepends=True)
            lines_b = report.answer_b.splitlines(keepends=True)
            diff = list(unified_diff(
                lines_a, lines_b,
                fromfile=f"Run A ({report.run_id_a})",
                tofile=f"Run B ({report.run_id_b})",
                lineterm=""
            ))
            report.answer_diff = diff

    def _compare_steps(
        self,
        report: DiffReport,
        entries_a: list[TraceEntry],
        entries_b: list[TraceEntry]
    ):
        """Compare execution steps."""
        report.steps_a = len(entries_a)
        report.steps_b = len(entries_b)

        # Align steps by event type for comparison
        max_steps = max(report.steps_a, report.steps_b)
        matching = 0
        different = 0

        for i in range(max_steps):
            entry_a = entries_a[i] if i < len(entries_a) else None
            entry_b = entries_b[i] if i < len(entries_b) else None

            diff_entry = {
                "step": i + 1,
                "status": "match",
                "details": {}
            }

            if entry_a is None:
                diff_entry["status"] = "missing_in_a"
                diff_entry["details"]["event_type_b"] = entry_b.event_type.value if entry_b else ""
                different += 1
            elif entry_b is None:
                diff_entry["status"] = "missing_in_b"
                diff_entry["details"]["event_type_a"] = entry_a.event_type.value
                different += 1
            else:
                # Both exist, compare
                if entry_a.event_type != entry_b.event_type:
                    diff_entry["status"] = "event_type_mismatch"
                    diff_entry["details"] = {
                        "event_type_a": entry_a.event_type.value,
                        "event_type_b": entry_b.event_type.value
                    }
                    different += 1
                elif entry_a.output_hash != entry_b.output_hash:
                    diff_entry["status"] = "output_mismatch"
                    diff_entry["details"] = {
                        "event_type": entry_a.event_type.value,
                        "hash_a": entry_a.output_hash,
                        "hash_b": entry_b.output_hash,
                        "summary_a": entry_a.content_summary[:100] if entry_a.content_summary else "",
                        "summary_b": entry_b.content_summary[:100] if entry_b.content_summary else ""
                    }
                    different += 1
                else:
                    matching += 1
                    continue  # Don't add matching steps to reduce noise

            if diff_entry["status"] != "match":
                report.step_diffs.append(diff_entry)

        report.matching_steps = matching
        report.different_steps = different

    def _compare_costs(
        self,
        report: DiffReport,
        entries_a: list[TraceEntry],
        entries_b: list[TraceEntry]
    ):
        """Compare execution costs (latency)."""
        report.latency_a_ms = sum(e.latency_ms for e in entries_a)
        report.latency_b_ms = sum(e.latency_ms for e in entries_b)
        report.latency_diff_ms = abs(report.latency_a_ms - report.latency_b_ms)

        if report.latency_a_ms > 0:
            report.latency_diff_percent = (report.latency_diff_ms / report.latency_a_ms) * 100

    def _compare_tools(
        self,
        report: DiffReport,
        entries_a: list[TraceEntry],
        entries_b: list[TraceEntry]
    ):
        """Compare tool usage."""
        # Count tool usage in each run
        for entry in entries_a:
            if entry.tool_name:
                report.tools_a[entry.tool_name] = report.tools_a.get(entry.tool_name, 0) + 1

        for entry in entries_b:
            if entry.tool_name:
                report.tools_b[entry.tool_name] = report.tools_b.get(entry.tool_name, 0) + 1

        # Find differences
        all_tools = set(report.tools_a.keys()) | set(report.tools_b.keys())
        for tool in all_tools:
            count_a = report.tools_a.get(tool, 0)
            count_b = report.tools_b.get(tool, 0)
            if count_a != count_b:
                report.tool_diffs.append({
                    "tool": tool,
                    "count_a": count_a,
                    "count_b": count_b,
                    "diff": count_b - count_a
                })

    def _compare_evidence(
        self,
        report: DiffReport,
        entries_a: list[TraceEntry],
        entries_b: list[TraceEntry]
    ):
        """Compare evidence collected."""
        evidence_a = []
        evidence_b = []

        for entry in entries_a:
            evidence_a.extend(entry.evidence_anchors)

        for entry in entries_b:
            evidence_b.extend(entry.evidence_anchors)

        report.evidence_count_a = len(evidence_a)
        report.evidence_count_b = len(evidence_b)

        # Find common evidence by doc_id
        docs_a = set(e.get("doc_id", "") for e in evidence_a if isinstance(e, dict))
        docs_b = set(e.get("doc_id", "") for e in evidence_b if isinstance(e, dict))
        report.common_evidence = len(docs_a & docs_b)

    def _calculate_similarity(self, report: DiffReport) -> float:
        """Calculate overall similarity score."""
        scores = []

        # Answer similarity
        if report.answer_a and report.answer_b:
            answer_sim = SequenceMatcher(
                None,
                report.answer_a.lower(),
                report.answer_b.lower()
            ).ratio()
            scores.append(answer_sim * 0.4)  # 40% weight
        elif report.answer_match:
            scores.append(0.4)

        # Step similarity
        total_steps = max(report.steps_a, report.steps_b)
        if total_steps > 0:
            step_sim = report.matching_steps / total_steps
            scores.append(step_sim * 0.4)  # 40% weight

        # Tool usage similarity
        all_tools = set(report.tools_a.keys()) | set(report.tools_b.keys())
        if all_tools:
            tool_matches = sum(
                1 for t in all_tools
                if report.tools_a.get(t, 0) == report.tools_b.get(t, 0)
            )
            tool_sim = tool_matches / len(all_tools)
            scores.append(tool_sim * 0.2)  # 20% weight

        return sum(scores) if scores else 0.0

    def format_report(self, report: DiffReport) -> str:
        """Format a diff report for human-readable display."""
        lines = [
            "═" * 60,
            f"DIFF REPORT: {report.run_id_a} vs {report.run_id_b}",
            "═" * 60,
            "",
            f"Overall Status: {'IDENTICAL' if report.identical else 'DIFFERENT'}",
            f"Similarity Score: {report.similarity_score:.1%}",
            "",
            "─" * 40,
            "ANSWER COMPARISON",
            "─" * 40,
            f"Match: {'Yes' if report.answer_match else 'No'}",
        ]

        if not report.answer_match and report.answer_diff:
            lines.append("\nDiff:")
            for line in report.answer_diff[:20]:  # Limit to 20 lines
                lines.append(f"  {line.rstrip()}")
            if len(report.answer_diff) > 20:
                lines.append(f"  ... ({len(report.answer_diff) - 20} more lines)")

        lines.extend([
            "",
            "─" * 40,
            "STEP COMPARISON",
            "─" * 40,
            f"Run A Steps: {report.steps_a}",
            f"Run B Steps: {report.steps_b}",
            f"Matching: {report.matching_steps}",
            f"Different: {report.different_steps}",
        ])

        if report.step_diffs:
            lines.append("\nStep Differences:")
            for diff in report.step_diffs[:10]:  # Limit to 10
                lines.append(f"  Step {diff['step']}: {diff['status']}")
                if diff.get("details"):
                    for k, v in diff["details"].items():
                        lines.append(f"    {k}: {v}")

        lines.extend([
            "",
            "─" * 40,
            "COST COMPARISON",
            "─" * 40,
            f"Latency A: {report.latency_a_ms:.2f}ms",
            f"Latency B: {report.latency_b_ms:.2f}ms",
            f"Difference: {report.latency_diff_ms:.2f}ms ({report.latency_diff_percent:.1f}%)",
        ])

        lines.extend([
            "",
            "─" * 40,
            "TOOL USAGE",
            "─" * 40,
        ])

        if report.tool_diffs:
            lines.append("Differences:")
            for diff in report.tool_diffs:
                lines.append(f"  {diff['tool']}: {diff['count_a']} → {diff['count_b']} ({diff['diff']:+d})")
        else:
            lines.append("Tool usage identical")

        lines.extend([
            "",
            "─" * 40,
            "EVIDENCE COMPARISON",
            "─" * 40,
            f"Evidence A: {report.evidence_count_a}",
            f"Evidence B: {report.evidence_count_b}",
            f"Common: {report.common_evidence}",
            "",
            "═" * 60,
        ])

        return "\n".join(lines)

    def save_report(self, report: DiffReport, output_path: str):
        """Save a diff report to a file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.format_report(report))
            f.write("\n\n--- JSON Data ---\n\n")
            f.write(report.to_json())
