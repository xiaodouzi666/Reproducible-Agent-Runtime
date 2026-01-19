"""
Trace storage backend - JSONL file-based storage.
"""

import os
import json
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

from .schema import TraceEntry, RunMetadata


class TraceStore:
    """
    File-based trace storage using JSONL format.
    Each run creates a directory with trace.jsonl and metadata.json.
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_run_dir(self, run_id: str) -> Path:
        """Get the directory for a specific run."""
        return self.base_dir / run_id

    def create_run(self, run_id: str, metadata: RunMetadata) -> Path:
        """Create a new run directory and initialize files."""
        run_dir = self._get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write initial metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # Create empty trace file
        trace_path = run_dir / "trace.jsonl"
        trace_path.touch()

        # Create artifacts directory
        (run_dir / "artifacts").mkdir(exist_ok=True)

        return run_dir

    def append_entry(self, run_id: str, entry: TraceEntry) -> None:
        """Append a trace entry to the run's trace file."""
        trace_path = self._get_run_dir(run_id) / "trace.jsonl"
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

    def update_metadata(self, run_id: str, metadata: RunMetadata) -> None:
        """Update the run's metadata."""
        metadata_path = self._get_run_dir(run_id) / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    def get_metadata(self, run_id: str) -> Optional[RunMetadata]:
        """Get metadata for a run."""
        metadata_path = self._get_run_dir(run_id) / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as f:
            return RunMetadata.from_dict(json.load(f))

    def get_entries(self, run_id: str) -> list[TraceEntry]:
        """Get all trace entries for a run."""
        trace_path = self._get_run_dir(run_id) / "trace.jsonl"
        if not trace_path.exists():
            return []

        entries = []
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(TraceEntry.from_json(line))
        return entries

    def iter_entries(self, run_id: str) -> Iterator[TraceEntry]:
        """Iterate over trace entries without loading all into memory."""
        trace_path = self._get_run_dir(run_id) / "trace.jsonl"
        if not trace_path.exists():
            return

        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield TraceEntry.from_json(line)

    def get_entry_by_step(self, run_id: str, step_id: int) -> Optional[TraceEntry]:
        """Get a specific trace entry by step ID."""
        for entry in self.iter_entries(run_id):
            if entry.step_id == step_id:
                return entry
        return None

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        runs = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                runs.append(item.name)
        return sorted(runs, reverse=True)  # Most recent first

    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return (self._get_run_dir(run_id) / "metadata.json").exists()

    def save_artifact(self, run_id: str, name: str, data: bytes) -> str:
        """Save an artifact file and return its path."""
        artifacts_dir = self._get_run_dir(run_id) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        artifact_path = artifacts_dir / name
        with open(artifact_path, "wb") as f:
            f.write(data)
        return str(artifact_path)

    def get_artifact(self, run_id: str, name: str) -> Optional[bytes]:
        """Get an artifact file."""
        artifact_path = self._get_run_dir(run_id) / "artifacts" / name
        if artifact_path.exists():
            with open(artifact_path, "rb") as f:
                return f.read()
        return None

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its data."""
        import shutil
        run_dir = self._get_run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return True
        return False

    def get_run_summary(self, run_id: str) -> dict:
        """Get a summary of a run for display."""
        metadata = self.get_metadata(run_id)
        if not metadata:
            return {}

        entries = self.get_entries(run_id)

        # Count events by type
        event_counts = {}
        for entry in entries:
            event_type = entry.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Get evidence anchors
        all_evidence = []
        for entry in entries:
            all_evidence.extend(entry.evidence_anchors)

        return {
            "run_id": run_id,
            "task": metadata.task_description,
            "status": metadata.status,
            "start_time": metadata.start_time,
            "end_time": metadata.end_time,
            "total_steps": len(entries),
            "event_counts": event_counts,
            "total_latency_ms": sum(e.latency_ms for e in entries),
            "evidence_count": len(all_evidence),
            "final_answer": metadata.final_answer,
            "is_replay": metadata.is_replay
        }
