"""
Replay Engine - Reproduce previous runs deterministically.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

from ..tracing import TraceStore, TraceEntry, RunMetadata
from ..orchestrator import Orchestrator


class ReplayEngine:
    """
    Engine for replaying previous runs.
    Ensures reproducibility by using the same seed and configuration.
    """

    def __init__(self, store: Optional[TraceStore] = None, output_dir: str = "runs"):
        self.store = store or TraceStore(output_dir)
        self.output_dir = Path(output_dir)

    def replay(
        self,
        original_run_id: str,
        new_run_id: Optional[str] = None
    ) -> dict:
        """
        Replay a previous run.

        Args:
            original_run_id: ID of the run to replay
            new_run_id: Optional new run ID (auto-generated if not provided)

        Returns:
            {
                "run_id": str,
                "original_run_id": str,
                "success": bool,
                "matches_original": bool,
                "diff_summary": dict
            }
        """
        # Load original run metadata
        original_metadata = self.store.get_metadata(original_run_id)
        if not original_metadata:
            return {
                "success": False,
                "error": f"Original run not found: {original_run_id}"
            }

        # Generate new run ID
        new_run_id = new_run_id or f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Create orchestrator with same configuration
        orchestrator = Orchestrator(
            output_dir=str(self.output_dir),
            seed=original_metadata.seed
        )

        # Set up orchestrator tracer for replay
        orchestrator.tracer = None  # Will be created in run()

        # Execute the same task
        result = orchestrator.run(
            task_description=original_metadata.task_description,
            spec_file=original_metadata.spec_file,
            run_id=new_run_id
        )

        # Update metadata to mark as replay
        new_metadata = self.store.get_metadata(new_run_id)
        if new_metadata:
            new_metadata.is_replay = True
            new_metadata.original_run_id = original_run_id
            self.store.update_metadata(new_run_id, new_metadata)

        # Compare with original
        comparison = self.compare_runs(original_run_id, new_run_id)

        result["original_run_id"] = original_run_id
        result["matches_original"] = comparison.get("identical", False)
        result["diff_summary"] = comparison

        return result

    def replay_from_spec(self, spec_path: str) -> dict:
        """
        Replay using a spec file.
        The spec file should contain the run configuration.

        Args:
            spec_path: Path to spec YAML file

        Returns:
            Replay result dict
        """
        spec_path = Path(spec_path)
        if not spec_path.exists():
            return {
                "success": False,
                "error": f"Spec file not found: {spec_path}"
            }

        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        # Create orchestrator
        orchestrator = Orchestrator(
            corpus_dir=spec.get("corpus_dir", "demo_data/corpus"),
            output_dir=str(self.output_dir),
            seed=spec.get("seed")
        )

        # Generate replay run ID
        new_run_id = f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Run
        result = orchestrator.run(
            task_description=spec.get("task", spec.get("description", "")),
            spec_file=str(spec_path),
            run_id=new_run_id
        )

        # If original_run_id specified, compare
        original_run_id = spec.get("original_run_id")
        if original_run_id and self.store.run_exists(original_run_id):
            comparison = self.compare_runs(original_run_id, new_run_id)
            result["original_run_id"] = original_run_id
            result["matches_original"] = comparison.get("identical", False)
            result["diff_summary"] = comparison

        return result

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """
        Compare two runs for reproducibility.

        Returns:
            {
                "identical": bool,
                "answer_match": bool,
                "step_differences": list,
                "cost_comparison": dict
            }
        """
        entries_a = self.store.get_entries(run_id_a)
        entries_b = self.store.get_entries(run_id_b)
        metadata_a = self.store.get_metadata(run_id_a)
        metadata_b = self.store.get_metadata(run_id_b)

        if not entries_a or not entries_b:
            return {
                "identical": False,
                "error": "Could not load entries for comparison"
            }

        # Compare final answers
        answer_a = metadata_a.final_answer if metadata_a else ""
        answer_b = metadata_b.final_answer if metadata_b else ""
        answer_match = self._normalize_answer(answer_a) == self._normalize_answer(answer_b)

        # Compare step-by-step
        step_differences = []
        max_steps = max(len(entries_a), len(entries_b))

        for i in range(max_steps):
            entry_a = entries_a[i] if i < len(entries_a) else None
            entry_b = entries_b[i] if i < len(entries_b) else None

            if entry_a is None or entry_b is None:
                step_differences.append({
                    "step": i,
                    "type": "missing_step",
                    "in_a": entry_a is not None,
                    "in_b": entry_b is not None
                })
                continue

            # Compare key fields
            if entry_a.output_hash != entry_b.output_hash:
                step_differences.append({
                    "step": i,
                    "type": "output_mismatch",
                    "event_type": entry_a.event_type.value,
                    "hash_a": entry_a.output_hash,
                    "hash_b": entry_b.output_hash
                })

        # Cost comparison
        total_latency_a = sum(e.latency_ms for e in entries_a)
        total_latency_b = sum(e.latency_ms for e in entries_b)

        cost_comparison = {
            "steps_a": len(entries_a),
            "steps_b": len(entries_b),
            "latency_a_ms": total_latency_a,
            "latency_b_ms": total_latency_b,
            "latency_diff_ms": abs(total_latency_a - total_latency_b)
        }

        return {
            "identical": answer_match and len(step_differences) == 0,
            "answer_match": answer_match,
            "step_differences": step_differences,
            "cost_comparison": cost_comparison,
            "steps_compared": max_steps
        }

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        # Remove whitespace variations
        return " ".join(answer.lower().split())

    def get_replay_spec(self, run_id: str) -> dict:
        """
        Generate a replay spec from an existing run.

        Args:
            run_id: ID of the run

        Returns:
            Spec dict that can be saved as YAML
        """
        metadata = self.store.get_metadata(run_id)
        if not metadata:
            return {}

        return {
            "task": metadata.task_description,
            "seed": metadata.seed,
            "original_run_id": run_id,
            "corpus_dir": "demo_data/corpus",
            "replay_mode": True
        }

    def save_replay_spec(self, run_id: str, output_path: str) -> bool:
        """Save a replay spec to a YAML file."""
        spec = self.get_replay_spec(run_id)
        if not spec:
            return False

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(spec, f, default_flow_style=False)

        return True
