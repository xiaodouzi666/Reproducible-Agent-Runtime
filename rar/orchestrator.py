"""
Orchestrator - Main workflow engine for multi-agent execution.
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import yaml

from .agents import PlannerAgent, ResearcherAgent, ExecutorAgent, AuditorAgent
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
        seed: Optional[int] = None
    ):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed

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
        return {
            "planner": PlannerAgent(
                agent_id="planner",
                tracer=tracer,
                tools=tools
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
            llm_mode=False
        )

        # Initialize tools and agents
        self.tools = self._init_tools()
        self.agents = self._init_agents(self.tracer, self.tools)

        try:
            # Run the workflow
            result = self._execute_workflow(task_description)

            # Finalize
            self.tracer.end_run(
                status="completed" if result["success"] else "failed",
                final_answer=result.get("final_answer", ""),
                evidence_summary=result.get("evidence", [])
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
                final_answer=f"Error: {str(e)}"
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
            "context": {},
            "workers": [researcher, executor, auditor]
        }

        planner_result = planner.process(task)

        # Check if auditor raised challenges
        audit_result = None
        for subtask_id, result in planner_result.get("subtask_results", {}).items():
            if "audit" in subtask_id.lower() or (
                isinstance(result, dict) and result.get("challenges")
            ):
                audit_result = result
                break

        # If there are challenges, attempt one replan
        if audit_result and audit_result.get("challenges") and not audit_result.get("approved", True):
            # For now, just note the challenges in the result
            # A more sophisticated system would actually replan
            planner_result["audit_challenges"] = audit_result.get("challenges", [])

        # Synthesize final answer with evidence
        final_answer = self._synthesize_final_answer(
            task_description,
            planner_result
        )

        return {
            "success": planner_result.get("success", False),
            "final_answer": final_answer,
            "evidence": planner_result.get("evidence", []),
            "plan": planner_result.get("plan", []),
            "subtask_results": planner_result.get("subtask_results", {}),
            "audit_challenges": planner_result.get("audit_challenges", [])
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

        # Generate final.json
        final_data = {
            "run_id": self.run_id,
            "task": task_description,
            "answer": result.get("final_answer", ""),
            "evidence_anchors": result.get("evidence", []),
            "status": "completed" if result.get("success") else "failed",
            "total_latency_ms": total_latency_ms
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
                parts.append(f"- {c.get('reason', 'Unknown issue')}\n")

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

        return self.run(
            task_description=task_description,
            spec_file=str(spec_path),
            run_id=spec.get("run_id")
        )


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
