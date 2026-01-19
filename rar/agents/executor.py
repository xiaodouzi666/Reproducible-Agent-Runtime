"""
Executor Agent - Code execution and computation.
"""

from typing import Optional

from .base import BaseAgent, AgentState
from ..protocols.acl import Performative
from ..tracing.tracer import Tracer


class ExecutorAgent(BaseAgent):
    """
    Executor agent responsible for:
    - Running Python code for calculations
    - Data analysis and visualization
    - Returning computation results with artifacts
    """

    role = "executor"
    description = "Code execution and computation"

    def __init__(
        self,
        agent_id: str = "executor",
        tracer: Optional[Tracer] = None,
        tools: dict = None
    ):
        super().__init__(agent_id, tracer, tools)

    def process(self, task: dict) -> dict:
        """
        Process an execution task.

        Args:
            task: {
                "id": str,
                "type": "execute",
                "description": str,
                "code": str
            }

        Returns:
            {
                "success": bool,
                "output": dict,     # Computation results
                "evidence": list,   # Evidence (computation trace)
                "artifacts": dict   # Generated files (plots, etc.)
            }
        """
        self.activate()

        # Update beliefs
        self.update_beliefs({
            "task_id": task.get("id", ""),
            "code_to_execute": task.get("code", "")[:100] + "..."
        }, reason="Execution task received")

        # Set desires
        self.set_desires([f"Execute computation: {task.get('description', '')}"])

        code = task.get("code", "")

        if not code:
            self.deactivate()
            return {
                "success": False,
                "output": {},
                "error": "No code provided",
                "evidence": [],
                "artifacts": {}
            }

        # Check if execution tool is available
        if "python_exec" not in self.tools:
            self.deactivate()
            return {
                "success": False,
                "output": {},
                "error": "Python execution tool not available",
                "evidence": [],
                "artifacts": {}
            }

        # Execute code
        exec_result = self.use_tool(
            "python_exec",
            code=code,
            description=task.get("description", ""),
            save_figures=True
        )

        if not exec_result.success:
            self.deactivate()
            return {
                "success": False,
                "output": exec_result.output,
                "error": exec_result.error,
                "evidence": [],
                "artifacts": {}
            }

        # Process results
        output = exec_result.output
        artifacts = exec_result.artifacts

        # Create evidence from computation
        evidence = self._create_computation_evidence(
            task.get("id", ""),
            code,
            output
        )

        # Update beliefs
        self.update_beliefs({
            "computation_completed": True,
            "has_artifacts": bool(artifacts),
            "result_variables": list(output.get("variables", {}).keys())
        }, reason="Computation completed")

        # Send result message
        self.send_message(
            receiver="planner",
            performative=Performative.INFORM,
            content={
                "task_id": task.get("id", ""),
                "result": output.get("variables", {}),
                "stdout": output.get("stdout", ""),
                "has_figures": bool(artifacts)
            },
            artifacts=artifacts
        )

        self.result = {
            "success": True,
            "output": self._format_output(output),
            "evidence": evidence,
            "artifacts": artifacts,
            "raw_output": output
        }

        self.deactivate()
        return self.result

    def _format_output(self, output: dict) -> str:
        """Format execution output for display."""
        parts = []

        # Add stdout if present
        stdout = output.get("stdout", "").strip()
        if stdout:
            parts.append("Output:\n" + stdout)

        # Add variables
        variables = output.get("variables", {})
        if variables:
            parts.append("\nComputed values:")
            for name, value in variables.items():
                parts.append(f"  {name} = {value}")

        return "\n".join(parts) if parts else "Execution completed (no output)"

    def _create_computation_evidence(
        self,
        task_id: str,
        code: str,
        output: dict
    ) -> list:
        """Create evidence anchors from computation."""
        evidence = []

        # The code itself is evidence
        evidence.append({
            "doc_id": f"computation_{task_id}",
            "doc_title": "Python Computation",
            "location": "code",
            "snippet": code[:500],
            "relevance_score": 1.0
        })

        # Output is also evidence
        stdout = output.get("stdout", "")
        if stdout:
            evidence.append({
                "doc_id": f"computation_{task_id}",
                "doc_title": "Computation Output",
                "location": "stdout",
                "snippet": stdout[:500],
                "relevance_score": 1.0
            })

        return evidence

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Basic code validation before execution."""
        # Check for dangerous patterns
        dangerous_patterns = [
            "import os",
            "import sys",
            "subprocess",
            "__import__",
            "eval(",
            "exec(",
            "open(",
            "file(",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Potentially unsafe code pattern: {pattern}"

        return True, "Code appears safe"
