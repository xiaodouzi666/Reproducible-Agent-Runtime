"""
Orchestrator - Main workflow engine for multi-agent execution.
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import yaml

from .agents import PlannerAgent, ResearcherAgent, ExecutorAgent, AuditorAgent
from .config import resolve_mode_config, ModeConfig
from .tools import LocalSearchTool, PythonExecTool
from .tracing import Tracer, TraceStore


class Orchestrator:
    """
    Main orchestrator for multi-agent workflow execution.
    Coordinates agents, manages tools, and records traces.
    """

    def __init__(
        self,
        corpus_dir: str = "demo_data/corpus",
        output_dir: str = "runs",
        seed: Optional[int] = None,
        llm_mode: bool = False,
        owl_mode: str = "",
        llm_provider: str = "",
        llm_model: str = "",
        llm_thinking_level: str = "",
        llm_use_cache: bool = True,
        llm_global_cache_path: Optional[str] = "runs/_global_llm_cache.jsonl",
        llm_readonly_cache_paths: Optional[list[str]] = None,
        llm_cache_readonly: bool = False,
        llm_allow_missing_api_key: bool = False,
        marathon_context: Optional[dict] = None,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.llm_mode = llm_mode
        self.owl_mode = owl_mode
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_thinking_level = llm_thinking_level
        self.llm_use_cache = llm_use_cache
        self.llm_global_cache_path = llm_global_cache_path
        self.llm_readonly_cache_paths = llm_readonly_cache_paths or []
        self.llm_cache_readonly = llm_cache_readonly
        self.llm_allow_missing_api_key = llm_allow_missing_api_key
        self.marathon_context = marathon_context or {}
        self.mode_config: ModeConfig = resolve_mode_config(
            mode=self.owl_mode or "owl_lite",
            model_override=self.llm_model or None,
            thinking_override=self.llm_thinking_level or None,
        )
        self.owl_mode = self.mode_config.mode
        if not self.llm_model:
            self.llm_model = self.mode_config.model
        if not self.llm_thinking_level:
            self.llm_thinking_level = self.mode_config.thinking_level

        # Initialize trace store
        self.store = TraceStore(str(self.output_dir))

        # Tools will be initialized per run
        self.tools = {}

        # Agents will be initialized per run
        self.agents = {}

        # Current run state
        self.tracer: Optional[Tracer] = None
        self.run_id: Optional[str] = None

    def _init_tools(self) -> dict:
        """Initialize tools with current seed."""
        artifacts_dir = self.output_dir / (self.run_id or "temp") / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        return {
            "local_search": LocalSearchTool(
                corpus_dir=str(self.corpus_dir),
                seed=self.seed
            ),
            "python_exec": PythonExecTool(
                seed=self.seed,
                output_dir=str(artifacts_dir)
            )
        }

    def _init_agents(self, tracer: Tracer, tools: dict) -> dict:
        """Initialize all agents."""
        planner_llm_client = None
        if self.llm_mode and (self.llm_provider or "gemini").lower() == "gemini":
            try:
                from .llm import GeminiClient

                run_dir = self.output_dir / (self.run_id or "temp")
                planner_llm_client = GeminiClient(
                    run_dir=str(run_dir),
                    enable_cache=self.llm_use_cache,
                    global_cache_path=self.llm_global_cache_path,
                    readonly_cache_paths=self.llm_readonly_cache_paths,
                    cache_readonly=self.llm_cache_readonly,
                    allow_missing_api_key=self.llm_allow_missing_api_key,
                    auto_fallback=True,
                )
            except Exception:
                # Graceful degradation: keep runtime alive with rule-based planner.
                planner_llm_client = None
                self.llm_mode = False

        return {
            "planner": PlannerAgent(
                agent_id="planner",
                tracer=tracer,
                tools=tools,
                llm_client=planner_llm_client,
            ),
            "researcher": ResearcherAgent(
                agent_id="researcher",
                tracer=tracer,
                tools=tools
            ),
            "executor": ExecutorAgent(
                agent_id="executor",
                tracer=tracer,
                tools=tools
            ),
            "auditor": AuditorAgent(
                agent_id="auditor",
                tracer=tracer,
                tools=tools
            )
        }

    def run(
        self,
        task_description: str,
        spec_file: str = "",
        run_id: Optional[str] = None
    ) -> dict:
        """
        Execute a complete workflow for a task.

        Args:
            task_description: The task to accomplish
            spec_file: Path to spec file (if any)
            run_id: Optional run ID (auto-generated if not provided)

        Returns:
            {
                "run_id": str,
                "success": bool,
                "final_answer": str,
                "evidence": list,
                "trace_path": str
            }
        """
        # Generate run ID
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Initialize tracer
        self.tracer = Tracer(run_id=self.run_id, store=self.store)
        self.tracer.start_run(
            task_description=task_description,
            spec_file=spec_file,
            seed=self.seed,
            llm_mode=self.llm_mode,
            owl_mode=self.owl_mode,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            llm_thinking_level=self.llm_thinking_level,
        )

        # Initialize tools and agents
        self.tools = self._init_tools()
        self.agents = self._init_agents(self.tracer, self.tools)

        try:
            # Run the workflow
            result = self._execute_workflow(task_description)
            result = self._post_finalize_best_effort(result)
            run_status = result.get("run_status") or ("completed" if result.get("success") else "failed")

            # Finalize
            self.tracer.end_run(
                status=run_status,
                final_answer=result.get("final_answer", ""),
                evidence_summary=result.get("evidence", []),
                finalize_missing=result.get("finalize_missing"),
                llm_finalize_called=result.get("llm_finalize_called"),
                argument_graph_generated=result.get("argument_graph_generated"),
                next_run_at=result.get("next_run_at", ""),
            )

            result["run_id"] = self.run_id
            result["trace_path"] = str(self.output_dir / self.run_id)

            # Auto-generate final.json and run_spec.yaml
            self._save_final_and_spec(
                task_description=task_description,
                result=result,
                spec_file=spec_file
            )

            return result

        except Exception as e:
            # Handle errors
            self.tracer.end_run(
                status="failed",
                final_answer=f"Error: {str(e)}",
                finalize_missing=False,
                llm_finalize_called=False,
                argument_graph_generated=False,
                next_run_at="",
            )

            # Still generate final.json and run_spec.yaml for failed runs
            error_result = {
                "run_id": self.run_id,
                "success": False,
                "error": str(e),
                "final_answer": f"Error: {str(e)}",
                "evidence": [],
                "trace_path": str(self.output_dir / self.run_id)
            }
            self._save_final_and_spec(
                task_description=task_description,
                result=error_result,
                spec_file=spec_file
            )

            return error_result

    def _execute_workflow(self, task_description: str) -> dict:
        """Execute the multi-agent workflow."""
        planner = self.agents["planner"]
        researcher = self.agents["researcher"]
        executor = self.agents["executor"]
        auditor = self.agents["auditor"]

        # Step 1: Planner decomposes task and coordinates
        task = {
            "description": task_description,
            "context": {
                "llm_enabled": bool(self.llm_mode and planner.llm_client is not None),
                "owl_mode": self.owl_mode or "owl_lite",
                "llm_provider": self.llm_provider or "gemini",
                "llm_model": self.llm_model,
                "llm_thinking_level": self.llm_thinking_level,
                "llm_use_cache": self.llm_use_cache,
                "llm_max_steps": 10,
                "audit_level": self.mode_config.audit_level,
                "argumentation": self.mode_config.argumentation,
                "evidence_rules": self.mode_config.evidence_rules,
                "max_replans": self.mode_config.max_replans,
                "marathon": {
                    "enabled": True,
                    "output_dir": str(self.output_dir),
                    **(self.marathon_context if isinstance(self.marathon_context, dict) else {}),
                },
            },
            "workers": [researcher, executor, auditor]
        }

        planner_result = planner.process(task)

        # For LLM-driven planner, prefer planner-produced final answer.
        if planner_result.get("llm_loop"):
            final_answer = planner_result.get("output", "")
            if not final_answer:
                final_answer = self._synthesize_final_answer(task_description, planner_result)
        else:
            final_answer = self._synthesize_final_answer(task_description, planner_result)

        if self.llm_mode:
            final_answer = self._ensure_mode_header(final_answer)
        final_answer = self._append_audit_sections(final_answer, planner_result)

        return {
            "success": planner_result.get("success", False),
            "run_status": planner_result.get("run_status", "completed" if planner_result.get("success", False) else "failed"),
            "final_answer": final_answer,
            "evidence": planner_result.get("evidence", []),
            "plan": planner_result.get("plan", []),
            "executed_plan": planner_result.get("executed_plan", []),
            "subtask_results": planner_result.get("subtask_results", {}),
            "audit_challenges": planner_result.get("audit_challenges", []),
            "all_audit_challenges": planner_result.get("all_audit_challenges", []),
            "audit_rounds": planner_result.get("audit_rounds", []),
            "audit_notes": planner_result.get("audit_notes", ""),
            "uncertainty": bool(planner_result.get("uncertainty", False)),
            "claims": planner_result.get("claims", []),
            "evidence_used": planner_result.get("evidence_used", []),
            "confidence": planner_result.get("confidence"),
            "uncertainty_details": planner_result.get("uncertainty_details", []),
            "warnings": planner_result.get("warnings", []),
            "outstanding_challenges": planner_result.get("outstanding_challenges", []),
            "structured_plan": planner_result.get("structured_plan", {}),
            "structured_finalize": planner_result.get("structured_finalize", {}),
            "argument_graph_ref": planner_result.get("argument_graph_ref", ""),
            "argument_graph_status": planner_result.get("argument_graph_status", "skipped"),
            "argument_graph_generated": bool(planner_result.get("argument_graph_generated", False)),
            "finalize_missing": bool(planner_result.get("finalize_missing", False)),
            "llm_finalize_called": bool(planner_result.get("llm_finalize_called", False)),
            "checkpoint_path": planner_result.get("checkpoint_path", ""),
            "next_run_at": planner_result.get("next_run_at", ""),
            "wait_reason": planner_result.get("wait_reason", ""),
            "llm_loop": planner_result.get("llm_loop", {}),
        }

    def _save_final_and_spec(
        self,
        task_description: str,
        result: dict,
        spec_file: str = ""
    ):
        """Save final.json and run_spec.yaml after a run."""
        run_dir = self.output_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Calculate total latency from trace if available
        total_latency_ms = 0.0
        try:
            trace_path = run_dir / "trace.jsonl"
            if trace_path.exists():
                with open(trace_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        if "latency_ms" in entry:
                            total_latency_ms += entry.get("latency_ms", 0)
        except Exception:
            pass

        warnings = self._dedupe_strings(result.get("warnings", []))
        argument_graph_ref = str(result.get("argument_graph_ref", "") or "")
        argument_graph_status = str(result.get("argument_graph_status", "skipped") or "skipped")
        argument_graph_generated = bool(result.get("argument_graph_generated", False))

        final_data = {
            "run_id": self.run_id,
            "task": task_description,
            "answer": result.get("final_answer", ""),
            "evidence_anchors": result.get("evidence", []),
            "status": result.get("run_status", "completed" if result.get("success") else "failed"),
            "total_latency_ms": total_latency_ms,
            "mode": self.owl_mode,
            "llm_enabled": bool(self.llm_mode),
            "model": self.llm_model,
            "thinking_level": self.llm_thinking_level,
            "audit_level": self.mode_config.audit_level,
            "audit_summary": self._extract_audit_summary(result),
            "argument_graph_ref": argument_graph_ref,
            "argument_graph_status": argument_graph_status,
            "argument_graph_generated": argument_graph_generated,
            "finalize_missing": bool(result.get("finalize_missing", False)),
            "llm_finalize_called": bool(result.get("llm_finalize_called", False)),
            "llm_loop": result.get("llm_loop", {}),
            "checkpoint_path": result.get("checkpoint_path", ""),
            "next_run_at": result.get("next_run_at", ""),
            "wait_reason": result.get("wait_reason", ""),
            "audit_rounds": result.get("audit_rounds", []),
            "audit_notes": result.get("audit_notes", ""),
            "uncertainty": bool(result.get("uncertainty", False)),
            "confidence": result.get("confidence"),
            "uncertainty_details": result.get("uncertainty_details", []),
            "warnings": warnings,
            "outstanding_challenges": result.get("outstanding_challenges", []),
            "structured_plan": result.get("structured_plan", {}),
            "structured_finalize": result.get("structured_finalize", {}),
        }

        final_path = run_dir / "final.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False, default=str)

        # Generate run_spec.yaml
        spec_data = {
            "task": task_description,
            "seed": self.seed,
            "corpus_dir": str(self.corpus_dir),
            "run_id": self.run_id
        }

        if self.llm_mode or self.llm_model:
            spec_data.update({
                "llm_enabled": bool(self.llm_mode),
                "llm_mode": self.llm_mode,
                "owl_mode": self.owl_mode,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "llm_thinking_level": self.llm_thinking_level,
                "llm_use_cache": self.llm_use_cache,
            })
        if isinstance(self.marathon_context, dict) and self.marathon_context:
            spec_data["marathon"] = self.marathon_context

        spec_path = run_dir / "run_spec.yaml"
        with open(spec_path, "w", encoding="utf-8") as f:
            yaml.dump(spec_data, f, allow_unicode=True, default_flow_style=False)

    def _synthesize_final_answer(self, task: str, result: dict) -> str:
        """Synthesize final answer with citations."""
        parts = [f"# Task: {task}\n\n"]

        # Add findings
        parts.append("## Findings\n\n")

        subtask_results = result.get("subtask_results", {})
        for subtask_id, sr in subtask_results.items():
            if sr.get("success"):
                output = sr.get("output", "")
                if output:
                    parts.append(f"### {subtask_id}\n{output}\n\n")

        # Add evidence citations
        evidence = result.get("evidence", [])
        if evidence:
            parts.append("## Evidence Sources\n\n")
            seen_docs = set()
            for i, e in enumerate(evidence, 1):
                doc_id = e.get("doc_id", "unknown")
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    title = e.get("doc_title", doc_id)
                    location = e.get("location", "")
                    parts.append(f"{i}. [{title}] - {location}\n")

        # Add audit notes if any
        challenges = result.get("audit_challenges", [])
        if challenges:
            parts.append("\n## Audit Notes\n\n")
            for c in challenges:
                reason = c.get("reason", "Unknown issue")
                if isinstance(reason, list):
                    reason = "; ".join(str(item) for item in reason)
                parts.append(f"- {reason}\n")

        return "".join(parts)

    def run_from_spec(self, spec_path: str) -> dict:
        """
        Run a workflow from a YAML spec file.

        Args:
            spec_path: Path to the spec YAML file

        Returns:
            Workflow result dict
        """
        spec_path = Path(spec_path)
        if not spec_path.exists():
            return {
                "success": False,
                "error": f"Spec file not found: {spec_path}"
            }

        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        # Extract spec fields
        task_description = spec.get("task", spec.get("description", ""))
        self.seed = spec.get("seed", self.seed)

        # Override corpus dir if specified
        if "corpus_dir" in spec:
            self.corpus_dir = Path(spec["corpus_dir"])

        # Optional LLM config for reproducibility.
        self.llm_mode = spec.get("llm_mode", spec.get("llm_enabled", self.llm_mode))
        self.owl_mode = spec.get("owl_mode", self.owl_mode)
        self.llm_provider = spec.get("llm_provider", self.llm_provider)
        self.llm_model = spec.get("llm_model", self.llm_model)
        self.llm_thinking_level = spec.get("llm_thinking_level", self.llm_thinking_level)
        self.llm_use_cache = spec.get("llm_use_cache", self.llm_use_cache)
        if isinstance(spec.get("marathon"), dict):
            self.marathon_context = spec.get("marathon", {})
        self.mode_config = resolve_mode_config(
            mode=self.owl_mode,
            model_override=self.llm_model or None,
            thinking_override=self.llm_thinking_level or None,
        )
        self.owl_mode = self.mode_config.mode
        self.llm_model = self.mode_config.model
        self.llm_thinking_level = self.mode_config.thinking_level

        return self.run(
            task_description=task_description,
            spec_file=str(spec_path),
            run_id=spec.get("run_id")
        )

    def _ensure_mode_header(self, answer: str) -> str:
        """Ensure final answer starts with mode/model/thinking metadata."""
        answer = answer or ""
        if answer.startswith("Mode: "):
            return answer
        mode_label = self.mode_config.label if self.mode_config else (self.owl_mode or "OWL")
        header = (
            f"Mode: {mode_label} | Model: {self.llm_model} | Thinking: {self.llm_thinking_level}\n"
        )
        return header + "\n" + answer

    def _extract_audit_summary(self, result: dict) -> dict:
        """Extract audit summary from planner subtask results."""
        summary = {
            "mode": self.owl_mode,
            "audit_level": self.mode_config.audit_level,
            "audits_total": 0,
            "audits_approved": 0,
            "challenges_total": 0,
            "replan_rounds": len(result.get("audit_rounds", []) or []),
            "uncertainty": bool(result.get("uncertainty", False)),
        }
        for subtask_id, subtask_result in (result.get("subtask_results", {}) or {}).items():
            if not isinstance(subtask_result, dict):
                continue
            if "approved" not in subtask_result:
                continue
            summary["audits_total"] += 1
            if subtask_result.get("approved"):
                summary["audits_approved"] += 1
            summary["challenges_total"] += len(subtask_result.get("challenges", []) or [])
        return summary

    def _append_audit_sections(self, answer: str, planner_result: dict) -> str:
        """Append audit/replan notes and uncertainty when missing in answer body."""
        answer = answer or ""
        if str(planner_result.get("run_status", "") or "") == "waiting":
            return answer
        has_audit_heading = "## Audit Notes" in answer or "### Audit Notes" in answer
        has_conf_heading = "## Confidence" in answer or "## Uncertainty" in answer

        audit_rounds = planner_result.get("audit_rounds", []) or []
        challenges = planner_result.get("audit_challenges", []) or []
        audit_notes = str(planner_result.get("audit_notes", "") or "").strip()
        uncertainty = bool(planner_result.get("uncertainty", False))

        if not has_audit_heading and (audit_rounds or challenges or audit_notes):
            lines = ["\n\n## Audit Notes\n"]
            if audit_notes:
                lines.append(f"- {audit_notes}\n")
            elif audit_rounds:
                for item in audit_rounds:
                    round_idx = item.get("round", 0)
                    approved = bool(item.get("approved", False))
                    challenge_count = len(item.get("challenges", []) or [])
                    lines.append(f"- Round {round_idx}: {'approved' if approved else 'challenged'} ({challenge_count} challenge(s))\n")
            if challenges:
                lines.append("- Outstanding challenges:\n")
                for c in challenges:
                    reason = c.get("reason", "Unknown issue")
                    if isinstance(reason, list):
                        reason = "; ".join(str(item) for item in reason)
                    lines.append(f"  - {reason}\n")
            answer += "".join(lines)

        if not has_conf_heading:
            confidence_header = "\n\n## Uncertainty\n" if uncertainty else "\n\n## Confidence\n"
            confidence_line = (
                "- Some audit challenges remain unresolved after replans.\n"
                if uncertainty
                else "- Latest audit approved the current evidence and reasoning path.\n"
            )
            answer += confidence_header + confidence_line

        return answer

    def _post_finalize_best_effort(self, result: dict) -> dict:
        """
        Post-finalize best-effort extensions.
        Finalize-first invariant:
        - finalize success is the hard gate
        - argument graph is best-effort and cannot flip run to failed
        """
        enriched = dict(result or {})
        warnings = list(enriched.get("warnings", []) or [])

        finalize_ok = bool(enriched.get("llm_finalize_called")) and not bool(enriched.get("finalize_missing"))
        run_status = str(enriched.get("run_status", "completed" if finalize_ok else "failed") or "failed")
        waiting_status = run_status == "waiting"
        if waiting_status:
            enriched.setdefault("argument_graph_ref", "")
            enriched.setdefault("argument_graph_status", "skipped")
            enriched.setdefault("argument_graph_generated", False)
            warnings = self._dedupe_strings(warnings)
            enriched["warnings"] = warnings
            enriched["run_status"] = "waiting"
            enriched["success"] = True
            return enriched
        if finalize_ok and run_status == "failed":
            run_status = "completed_with_warnings" if warnings else "completed"

        enriched.setdefault("argument_graph_ref", "")
        enriched.setdefault("argument_graph_status", "skipped")
        enriched.setdefault("argument_graph_generated", False)

        if self.owl_mode == "owl_full" and finalize_ok:
            run_dir = self.output_dir / (self.run_id or "temp")
            run_dir.mkdir(parents=True, exist_ok=True)
            graph_ref, graph_status, graph_generated, graph_warnings = self._try_generate_argument_graph(
                run_dir=run_dir,
                result=enriched,
            )
            enriched["argument_graph_ref"] = graph_ref
            enriched["argument_graph_status"] = graph_status
            enriched["argument_graph_generated"] = graph_generated
            warnings.extend(graph_warnings)
            if graph_status == "failed" and run_status == "completed":
                run_status = "completed_with_warnings"
        elif self.owl_mode == "owl_full":
            # Finalize not completed: skip graph and keep failure semantics.
            enriched["argument_graph_ref"] = ""
            enriched["argument_graph_status"] = "skipped"
            enriched["argument_graph_generated"] = False

        warnings = self._dedupe_strings(warnings)
        enriched["warnings"] = warnings
        enriched["run_status"] = run_status
        enriched["success"] = run_status in {"completed", "completed_with_warnings", "waiting"}
        return enriched

    def _try_generate_argument_graph(
        self,
        run_dir: Path,
        result: dict,
    ) -> tuple[str, str, bool, list[str]]:
        """
        Best-effort argument graph generation.
        Returns: (graph_ref, graph_status, generated, warnings)
        """
        warnings: list[str] = []
        claims = []
        structured_finalize = result.get("structured_finalize", {}) if isinstance(result.get("structured_finalize"), dict) else {}
        if isinstance(structured_finalize.get("claims"), list):
            claims = structured_finalize.get("claims", [])
        if not claims:
            claims = result.get("claims", []) if isinstance(result.get("claims"), list) else []

        evidence = result.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        latest_challenges = []
        audit_rounds = result.get("audit_rounds", [])
        if isinstance(audit_rounds, list) and audit_rounds:
            latest = audit_rounds[-1] if isinstance(audit_rounds[-1], dict) else {}
            challenges = latest.get("challenges", [])
            if isinstance(challenges, list):
                latest_challenges = challenges

        if not claims and not evidence:
            reason = "argument_graph_failed: missing claims/evidence inputs"
            if self.tracer:
                self.tracer.log_argument_graph_failed("planner", reason)
            return "", "failed", False, [reason]

        anchor_rows = []
        allowed_anchor_ids: set[str] = set()
        for idx, ev in enumerate(evidence, 1):
            if not isinstance(ev, dict):
                continue
            anchor_id = self._build_anchor_id(ev, idx)
            allowed_anchor_ids.add(anchor_id)
            anchor_rows.append(
                {
                    "anchor_id": anchor_id,
                    "doc_id": str(ev.get("doc_id", "") or ""),
                    "location": str(ev.get("location", "") or ""),
                    "content_hash": str(ev.get("content_hash", "") or ""),
                    "snippet": str(ev.get("snippet", "") or "")[:280],
                }
            )

        try:
            from .llm import GeminiClient, ArgumentGraphSchema

            client = GeminiClient(
                run_dir=str(run_dir),
                enable_cache=self.llm_use_cache,
                global_cache_path=self.llm_global_cache_path,
                readonly_cache_paths=self.llm_readonly_cache_paths,
                cache_readonly=self.llm_cache_readonly,
                allow_missing_api_key=self.llm_allow_missing_api_key,
                auto_fallback=True,
            )
            payload = {
                "mode": self.owl_mode,
                "claims": claims,
                "evidence_anchors": anchor_rows,
                "latest_audit_challenges": latest_challenges,
            }
            prompt = self._build_argument_graph_prompt(payload)
            record = client.generate_json(
                prompt=prompt,
                model=self.llm_model,
                thinking_level=self.llm_thinking_level,
                json_schema=ArgumentGraphSchema,
                system_prompt=(
                    "You build Dung-style claim-evidence-counterclaim graphs for OWL Full. "
                    "Use only provided evidence anchors. Do not invent citations or anchors."
                ),
                schema_name="argument_graph",
                max_schema_retries=1,
                on_schema_violation=(
                    lambda info: self.tracer.log_schema_violation(
                        agent_id="planner",
                        schema_name="argument_graph",
                        attempt=int(info.get("attempt", 1)),
                        max_attempts=int(info.get("max_attempts", 1)),
                        error=str(info.get("error", "")),
                        cache_key=str(info.get("cache_key", "")),
                        model=str(info.get("model", "")),
                        thinking_level=str(info.get("thinking_level", "")),
                        response_hash=str(info.get("response_hash", "")),
                        response_text_preview=str(info.get("response_text_preview", "")),
                    )
                    if self.tracer
                    else None
                ),
            )

            graph_raw = record.response_json if isinstance(record.response_json, dict) else {}
            graph_data = self._normalize_argument_graph(graph_raw, allowed_anchor_ids)

            graph_path = run_dir / "argument_graph.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2, default=str)

            if self.tracer:
                self.tracer.log_argument_graph_ok(
                    agent_id="planner",
                    node_count=len(graph_data.get("nodes", [])),
                    edge_count=len(graph_data.get("edges", [])),
                    graph_ref=graph_path.name,
                )

            return graph_path.name, "ok", True, warnings
        except Exception as e:
            reason = f"argument_graph_failed: {self._short_error(e)}"
            warnings.append(reason)
            if self.tracer:
                self.tracer.log_argument_graph_failed("planner", reason)
            return "", "failed", False, warnings

    def _build_argument_graph_prompt(self, payload: dict) -> str:
        return (
            "Build an OWL Full argument graph from the provided finalized claims and evidence anchors.\n"
            "Constraints:\n"
            "1) Evidence nodes must reference existing anchor_id via source_anchor_id.\n"
            "2) Do not invent papers/documents/citations.\n"
            "3) Use relation labels supports/attacks only.\n"
            "4) Keep graph compact and faithful to input.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)}\n"
        )

    def _build_anchor_id(self, evidence: dict, idx: int) -> str:
        doc_id = str(evidence.get("doc_id", "") or "doc")
        location = str(evidence.get("location", "") or f"loc_{idx}")
        content_hash = str(evidence.get("content_hash", "") or "")
        if content_hash:
            return f"{doc_id}|{location}|{content_hash}"
        return f"{doc_id}|{location}|anchor_{idx}"

    def _normalize_argument_graph(self, raw: dict, allowed_anchor_ids: set[str]) -> dict:
        nodes_out = []
        edges_out = []
        node_ids = set()

        raw_nodes = raw.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raw_nodes = []
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id", item.get("node_id", "")) or "").strip()
            node_type = str(item.get("type", item.get("node_type", "")) or "").strip().lower()
            text = str(item.get("text", "") or "").strip()
            if not node_id or not text:
                continue
            node_type = {
                "counter_claim": "counterclaim",
                "counter-claim": "counterclaim",
                "counterclaim": "counterclaim",
                "claim": "claim",
                "evidence": "evidence",
            }.get(node_type, "")
            if node_type not in {"claim", "evidence", "counterclaim"}:
                continue

            source_anchor_id = str(item.get("source_anchor_id", "") or "").strip()
            if node_type == "evidence":
                if source_anchor_id not in allowed_anchor_ids:
                    continue
            else:
                source_anchor_id = ""

            node_payload = {
                "id": node_id,
                "type": node_type,
                "text": text,
                "source_anchor_id": source_anchor_id,
            }
            nodes_out.append(node_payload)
            node_ids.add(node_id)

        raw_edges = raw.get("edges", [])
        if not isinstance(raw_edges, list):
            raw_edges = []
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "") or "").strip()
            target = str(item.get("target", "") or "").strip()
            relation = str(item.get("relation", "") or "").strip().lower()
            relation = {
                "support": "supports",
                "supports": "supports",
                "attack": "attacks",
                "attacks": "attacks",
            }.get(relation, "")
            if not source or not target or relation not in {"supports", "attacks"}:
                continue
            if source not in node_ids or target not in node_ids:
                continue
            edges_out.append({"source": source, "target": target, "relation": relation})

        accepted = raw.get("accepted_claim_ids", raw.get("winner_claims", []))
        if not isinstance(accepted, list):
            accepted = []
        accepted_ids = [str(cid) for cid in accepted if str(cid) in node_ids]

        rationale_raw = raw.get("rationale", {})
        rationale = {}
        if isinstance(rationale_raw, dict):
            for key, value in rationale_raw.items():
                claim_id = str(key)
                if claim_id in node_ids:
                    rationale[claim_id] = str(value)

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "accepted_claim_ids": accepted_ids,
            "rationale": rationale,
        }

    def _dedupe_strings(self, values: list[Any]) -> list[str]:
        output: list[str] = []
        seen = set()
        for item in values or []:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output

    def _short_error(self, exc: Exception, limit: int = 180) -> str:
        text = str(exc or "").strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."


def run_task(
    task: str,
    corpus_dir: str = "demo_data/corpus",
    output_dir: str = "runs",
    seed: Optional[int] = None
) -> dict:
    """Convenience function to run a single task."""
    orchestrator = Orchestrator(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        seed=seed
    )
    return orchestrator.run(task)
