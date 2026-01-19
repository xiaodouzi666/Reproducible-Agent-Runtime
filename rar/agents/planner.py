"""
Planner Agent - BDI-based task decomposition and coordination.
"""

from typing import Optional
import re

from .base import BaseAgent, AgentState, BDIState
from ..protocols.acl import Performative
from ..protocols.contract_net import ContractNetProtocol, TaskAnnouncement, Bid
from ..tracing.tracer import Tracer


class PlannerAgent(BaseAgent):
    """
    Planner agent responsible for:
    - Task decomposition (breaking complex tasks into subtasks)
    - Contract net coordination (issuing CFPs, collecting bids, awarding contracts)
    - Plan monitoring and replanning
    """

    role = "planner"
    description = "Task decomposition and multi-agent coordination"

    def __init__(
        self,
        agent_id: str = "planner",
        tracer: Optional[Tracer] = None,
        tools: dict = None,
        llm_client: Optional[callable] = None
    ):
        super().__init__(agent_id, tracer, tools)
        self.llm_client = llm_client
        self.contract_net = ContractNetProtocol(self.agent_id)
        self.subtask_counter = 0

    def process(self, task: dict) -> dict:
        """
        Process a task by decomposing it and coordinating execution.

        Args:
            task: {
                "description": str,  # Task description
                "context": dict,     # Additional context
                "workers": list      # Available worker agents
            }

        Returns:
            {
                "success": bool,
                "output": str,       # Final answer
                "evidence": list,    # Evidence chain
                "plan": list,        # Executed plan
                "subtask_results": dict
            }
        """
        self.activate()

        # Initialize beliefs
        self.update_beliefs({
            "task": task["description"],
            "context": task.get("context", {}),
            "available_workers": [w.agent_id for w in task.get("workers", [])],
            "completed_subtasks": [],
            "pending_subtasks": []
        }, reason="Task received")

        # Set desire
        self.set_desires([f"Complete task: {task['description']}"])

        # Step 1: Decompose task into subtasks
        subtasks = self._decompose_task(task["description"], task.get("context", {}))

        # Set intentions (the plan)
        self.set_intentions([
            {"action": "execute_subtask", "subtask": st}
            for st in subtasks
        ])

        # Step 2: Execute subtasks via contract net
        workers = task.get("workers", [])
        results = {}
        all_evidence = []

        for subtask in subtasks:
            # Issue CFP
            task_announcement = TaskAnnouncement(
                task_id=subtask["id"],
                task_type=subtask["type"],
                description=subtask["description"],
                requirements=subtask.get("requirements", {})
            )

            self.contract_net.announce_task(task_announcement)

            if self.tracer:
                self.tracer.log_cfp(
                    self.agent_id,
                    subtask["id"],
                    subtask["type"],
                    subtask["description"]
                )

            # Collect bids from appropriate workers
            bids = self._collect_bids(subtask, workers)

            if not bids:
                # No worker available, try to handle directly if possible
                results[subtask["id"]] = {
                    "success": False,
                    "error": "No worker available for this task type"
                }
                continue

            # Select winner and award contract
            for bid in bids:
                self.contract_net.submit_bid(bid)
                if self.tracer:
                    self.tracer.log_bid(
                        bid.bidder,
                        subtask["id"],
                        bid.estimated_cost,
                        bid.estimated_latency,
                        bid.success_probability
                    )

            winner = self.contract_net.evaluate_bids(subtask["id"])
            contract = self.contract_net.award_contract(subtask["id"], winner)

            if self.tracer:
                self.tracer.log_contract_award(
                    self.agent_id,
                    winner.bidder,
                    subtask["id"]
                )

            # Send task to winner
            winner_agent = next(
                (w for w in workers if w.agent_id == winner.bidder),
                None
            )

            if winner_agent:
                self.send_message(
                    receiver=winner.bidder,
                    performative=Performative.REQUEST,
                    content=subtask,
                    protocol="contract-net"
                )

                # Execute subtask
                self.contract_net.start_execution(subtask["id"])
                subtask_result = winner_agent.process(subtask)

                if subtask_result["success"]:
                    self.contract_net.complete_task(
                        subtask["id"],
                        subtask_result["output"],
                        subtask_result.get("evidence", [])
                    )
                else:
                    self.contract_net.fail_task(
                        subtask["id"],
                        subtask_result.get("error", "Unknown error")
                    )

                results[subtask["id"]] = subtask_result
                all_evidence.extend(subtask_result.get("evidence", []))

                # Update beliefs
                self.update_beliefs({
                    "completed_subtasks": list(results.keys()),
                    f"result_{subtask['id']}": subtask_result["output"]
                }, reason=f"Subtask {subtask['id']} completed")

        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answer(task["description"], results)

        self.result = {
            "success": all(r.get("success", False) for r in results.values()) if results else False,
            "output": final_answer,
            "evidence": all_evidence,
            "plan": subtasks,
            "subtask_results": results
        }

        self.deactivate()
        return self.result

    def _decompose_task(self, description: str, context: dict) -> list:
        """
        Decompose a task into subtasks.
        Uses LLM if available, otherwise uses rule-based decomposition.
        """
        subtasks = []

        if self.llm_client:
            # LLM-based decomposition
            prompt = f"""Decompose this research task into subtasks:

Task: {description}

Context: {context}

Return a list of subtasks, each with:
- type: "research" (for information gathering) or "execute" (for computation)
- description: what to do
- query: search query (for research) or code (for execute)
"""
            # Would call LLM here
            pass

        # Rule-based decomposition (fallback/default)
        self.subtask_counter += 1

        # Always start with research
        subtasks.append({
            "id": f"subtask_{self.subtask_counter}",
            "type": "research",
            "description": f"Search for relevant information about: {description}",
            "query": self._extract_search_query(description)
        })

        # Check if computation is needed
        if self._needs_computation(description):
            self.subtask_counter += 1
            subtasks.append({
                "id": f"subtask_{self.subtask_counter}",
                "type": "execute",
                "description": f"Perform calculations for: {description}",
                "code": self._generate_computation_code(description, context)
            })

        # Always end with synthesis/validation
        self.subtask_counter += 1
        subtasks.append({
            "id": f"subtask_{self.subtask_counter}",
            "type": "audit",
            "description": "Validate findings and check evidence quality",
            "target_subtasks": [st["id"] for st in subtasks[:-1]]
        })

        return subtasks

    def _extract_search_query(self, description: str) -> str:
        """Extract a search query from task description."""
        # Simple extraction - take key phrases
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'when', 'where', 'which'}
        words = description.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(keywords[:5])

    def _needs_computation(self, description: str) -> bool:
        """Check if task requires computation."""
        compute_keywords = [
            'calculate', 'compute', 'fit', 'plot', 'graph', 'analyze',
            'statistics', 'regression', 'curve', 'data', 'numeric',
            'temperature', 'pressure', 'rate', 'coefficient', 'equation'
        ]
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in compute_keywords)

    def _generate_computation_code(self, description: str, context: dict) -> str:
        """Generate computation code based on task."""
        # Simple template-based code generation
        if 'activation energy' in description.lower() or 'arrhenius' in description.lower():
            return """
import numpy as np
from scipy import stats

# Arrhenius equation: k = A * exp(-Ea/RT)
# ln(k) = ln(A) - Ea/(R*T)

# Example data (would come from research results)
T = np.array([300, 350, 400, 450, 500])  # Temperature in K
k = np.array([1e-3, 5e-3, 2e-2, 8e-2, 0.3])  # Rate constants

R = 8.314  # J/(mol*K)

# Linear regression: ln(k) vs 1/T
x = 1/T
y = np.log(k)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

Ea = -slope * R / 1000  # kJ/mol
A = np.exp(intercept)

print(f"Activation Energy (Ea): {Ea:.2f} kJ/mol")
print(f"Pre-exponential factor (A): {A:.2e} s^-1")
print(f"R-squared: {r_value**2:.4f}")

result = {"Ea_kJ_mol": Ea, "A": A, "R_squared": r_value**2}
"""
        elif 'thermal' in description.lower() or 'decomposition' in description.lower():
            return """
import numpy as np

# Thermal analysis calculation
# Example: Calculate weight loss rate

T = np.linspace(300, 800, 100)  # Temperature range in K
# Simulated TGA curve (weight fraction)
w = 1 - 0.5 * (1 + np.tanh((T - 550) / 50))

# Weight loss rate (derivative)
dw_dT = np.gradient(w, T)

# Find peak decomposition temperature
peak_idx = np.argmin(dw_dT)
T_peak = T[peak_idx]

print(f"Peak decomposition temperature: {T_peak:.1f} K ({T_peak-273.15:.1f} °C)")
print(f"Final weight fraction: {w[-1]:.3f}")

result = {"T_peak_K": T_peak, "final_weight_fraction": w[-1]}
"""
        else:
            return """
import numpy as np
import statistics

# Generic data analysis
data = [1.2, 2.3, 3.1, 4.5, 5.2, 6.1]

mean_val = statistics.mean(data)
std_val = statistics.stdev(data)

print(f"Mean: {mean_val:.3f}")
print(f"Standard deviation: {std_val:.3f}")

result = {"mean": mean_val, "std": std_val}
"""

    def _collect_bids(self, subtask: dict, workers: list) -> list[Bid]:
        """Collect bids from workers for a subtask."""
        bids = []
        task_type = subtask.get("type", "")

        for worker in workers:
            # Check if worker can handle this task type
            if task_type == "research" and worker.role == "researcher":
                bids.append(Bid(
                    task_id=subtask["id"],
                    bidder=worker.agent_id,
                    estimated_cost=1.0,
                    estimated_latency=0.5,
                    success_probability=0.9,
                    capability_match=1.0,
                    strategy="BM25 search over local corpus"
                ))
            elif task_type == "execute" and worker.role == "executor":
                bids.append(Bid(
                    task_id=subtask["id"],
                    bidder=worker.agent_id,
                    estimated_cost=2.0,
                    estimated_latency=1.0,
                    success_probability=0.85,
                    capability_match=1.0,
                    strategy="Python code execution in sandbox"
                ))
            elif task_type == "audit" and worker.role == "auditor":
                bids.append(Bid(
                    task_id=subtask["id"],
                    bidder=worker.agent_id,
                    estimated_cost=0.5,
                    estimated_latency=0.3,
                    success_probability=0.95,
                    capability_match=1.0,
                    strategy="Evidence verification and quality check"
                ))

        return bids

    def _synthesize_answer(self, task: str, results: dict) -> str:
        """Synthesize final answer from subtask results."""
        parts = [f"Task: {task}\n", "Findings:\n"]

        for task_id, result in results.items():
            if result.get("success"):
                output = result.get("output", "")
                if isinstance(output, dict):
                    output = str(output)
                parts.append(f"- {task_id}: {output[:200]}\n")
            else:
                parts.append(f"- {task_id}: Failed - {result.get('error', 'Unknown error')}\n")

        return "".join(parts)

    def handle_challenge(self, challenge: dict, workers: list) -> dict:
        """Handle a challenge from the auditor by potentially replanning."""
        reason = challenge.get("reason", "")
        target_subtask = challenge.get("target_subtask", "")

        if self.tracer:
            self.tracer.log_replan(
                self.agent_id,
                reason,
                [{"action": "retry_subtask", "subtask_id": target_subtask}]
            )

        # Simple replan: retry the challenged subtask
        # In a more sophisticated system, this could involve alternative strategies
        return {"action": "retry", "subtask_id": target_subtask}
