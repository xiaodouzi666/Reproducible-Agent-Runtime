"""
Marathon checkpoint I/O helper.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import CheckpointState


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarathonRunner:
    """Read/write checkpoint state for long-running or paused executions."""

    def __init__(self, output_dir: str = "runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, run_id: str) -> Path:
        return self.output_dir / run_id / "checkpoint.json"

    def checkpoint_exists(self, run_id: str) -> bool:
        return self.checkpoint_path(run_id).exists()

    def save_checkpoint(self, state: CheckpointState) -> Path:
        path = self.checkpoint_path(state.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not state.created_at:
            state.created_at = _utcnow_iso()
        state.updated_at = _utcnow_iso()

        payload = self._encode_bytes(state.to_dict())
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        tmp_path.replace(path)
        return path

    def load_checkpoint(self, run_id: str) -> CheckpointState:
        path = self.checkpoint_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        decoded = self._decode_bytes(raw)
        return CheckpointState.from_dict(decoded)

    def build_fallback_resume_context(self, run_id: str) -> dict[str, Any]:
        """Fallback context when checkpoint read fails."""
        run_dir = self.output_dir / run_id
        final_path = run_dir / "final.json"
        metadata_path = run_dir / "metadata.json"

        fallback: dict[str, Any] = {
            "task_description": "",
            "summary": "",
            "warnings": [],
            "mode": "",
            "model": "",
            "thinking_level": "",
        }

        try:
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                fallback["task_description"] = str(meta.get("task_description", "") or "")
                fallback["mode"] = str(meta.get("owl_mode", "") or "")
                fallback["model"] = str(meta.get("llm_model", "") or "")
                fallback["thinking_level"] = str(meta.get("llm_thinking_level", "") or "")
        except Exception:
            pass

        try:
            if final_path.exists():
                final = json.loads(final_path.read_text(encoding="utf-8"))
                fallback["summary"] = str(final.get("answer", "") or "")[:4000]
                warnings = final.get("warnings", [])
                if isinstance(warnings, list):
                    fallback["warnings"] = [str(w) for w in warnings if str(w).strip()]
        except Exception:
            pass

        return fallback

    def _encode_bytes(self, value: Any) -> Any:
        if isinstance(value, (bytes, bytearray)):
            return {"__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, dict):
            return {str(k): self._encode_bytes(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._encode_bytes(v) for v in value]
        return value

    def _decode_bytes(self, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value.keys()) == {"__bytes_b64__"}:
                encoded = str(value.get("__bytes_b64__", "") or "")
                if encoded:
                    try:
                        return base64.b64decode(encoded)
                    except Exception:
                        return b""
                return b""
            return {str(k): self._decode_bytes(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._decode_bytes(v) for v in value]
        return value
