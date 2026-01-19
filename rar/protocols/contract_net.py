"""
Contract Net Protocol Implementation
A simplified version of FIPA Contract Net for task allocation.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Status of a task in the contract net."""
    ANNOUNCED = "announced"     # CFP sent, waiting for bids
    BIDDING = "bidding"         # Collecting bids
    AWARDED = "awarded"         # Contract awarded
    IN_PROGRESS = "in_progress" # Work started
    COMPLETED = "completed"     # Work done
    FAILED = "failed"           # Task failed
    CANCELLED = "cancelled"     # Task cancelled


@dataclass
class TaskAnnouncement:
    """A task announcement (Call for Proposals)."""
    task_id: str
    task_type: str  # "research", "execute", "audit"
    description: str
    requirements: dict = field(default_factory=dict)
    deadline: Optional[str] = None
    priority: int = 1  # 1-5, higher is more important

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Bid:
    """A bid submitted by an agent in response to a CFP."""
    bid_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    bidder: str = ""

    # Bid metrics (for selection)
    estimated_cost: float = 0.0      # Abstract cost unit
    estimated_latency: float = 0.0   # Seconds
    success_probability: float = 1.0  # 0-1
    capability_match: float = 1.0     # How well agent matches task requirements

    # Additional info
    strategy: str = ""               # How the agent plans to accomplish the task
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def score(self) -> float:
        """Calculate a composite bid score (higher is better)."""
        # Simple weighted formula - can be customized
        return (
            self.success_probability * 0.4 +
            self.capability_match * 0.3 +
            (1 / (1 + self.estimated_cost)) * 0.15 +
            (1 / (1 + self.estimated_latency)) * 0.15
        )


@dataclass
class Contract:
    """An awarded contract between manager and contractor."""
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    manager: str = ""       # The agent who issued the CFP
    contractor: str = ""    # The agent who won the bid
    bid: Optional[Bid] = None
    status: TaskStatus = TaskStatus.AWARDED
    result: Any = None
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        if self.bid:
            data["bid"] = self.bid.to_dict()
        return data


class ContractNetProtocol:
    """
    Manages the Contract Net protocol flow.

    Flow:
    1. Manager announces task (CFP)
    2. Contractors submit bids
    3. Manager evaluates bids and awards contract
    4. Contractor executes task
    5. Contractor reports result
    """

    def __init__(self, manager_id: str):
        self.manager_id = manager_id
        self.tasks: dict[str, TaskAnnouncement] = {}
        self.bids: dict[str, list[Bid]] = {}  # task_id -> list of bids
        self.contracts: dict[str, Contract] = {}  # task_id -> contract
        self.task_status: dict[str, TaskStatus] = {}

    def announce_task(self, task: TaskAnnouncement) -> str:
        """Announce a new task (CFP)."""
        self.tasks[task.task_id] = task
        self.bids[task.task_id] = []
        self.task_status[task.task_id] = TaskStatus.ANNOUNCED
        return task.task_id

    def submit_bid(self, bid: Bid) -> bool:
        """Submit a bid for a task."""
        if bid.task_id not in self.tasks:
            return False
        if self.task_status.get(bid.task_id) not in [TaskStatus.ANNOUNCED, TaskStatus.BIDDING]:
            return False

        self.bids[bid.task_id].append(bid)
        self.task_status[bid.task_id] = TaskStatus.BIDDING
        return True

    def evaluate_bids(
        self,
        task_id: str,
        custom_evaluator: Optional[Callable[[list[Bid]], Bid]] = None
    ) -> Optional[Bid]:
        """Evaluate bids and select a winner."""
        if task_id not in self.bids or not self.bids[task_id]:
            return None

        bids = self.bids[task_id]

        if custom_evaluator:
            winner = custom_evaluator(bids)
        else:
            # Default: select by highest score
            winner = max(bids, key=lambda b: b.score)

        return winner

    def award_contract(self, task_id: str, winning_bid: Bid) -> Contract:
        """Award a contract to the winning bidder."""
        contract = Contract(
            task_id=task_id,
            manager=self.manager_id,
            contractor=winning_bid.bidder,
            bid=winning_bid,
            status=TaskStatus.AWARDED
        )
        self.contracts[task_id] = contract
        self.task_status[task_id] = TaskStatus.AWARDED
        return contract

    def start_execution(self, task_id: str) -> bool:
        """Mark a task as started."""
        if task_id in self.contracts:
            self.contracts[task_id].status = TaskStatus.IN_PROGRESS
            self.task_status[task_id] = TaskStatus.IN_PROGRESS
            return True
        return False

    def complete_task(self, task_id: str, result: Any, evidence: list = None) -> bool:
        """Mark a task as completed with result."""
        if task_id in self.contracts:
            self.contracts[task_id].status = TaskStatus.COMPLETED
            self.contracts[task_id].result = result
            self.contracts[task_id].evidence = evidence or []
            self.task_status[task_id] = TaskStatus.COMPLETED
            return True
        return False

    def fail_task(self, task_id: str, reason: str) -> bool:
        """Mark a task as failed."""
        if task_id in self.contracts:
            self.contracts[task_id].status = TaskStatus.FAILED
            self.contracts[task_id].result = {"error": reason}
            self.task_status[task_id] = TaskStatus.FAILED
            return True
        return False

    def get_task_summary(self, task_id: str) -> dict:
        """Get a summary of task status for display."""
        if task_id not in self.tasks:
            return {}

        task = self.tasks[task_id]
        bids = self.bids.get(task_id, [])
        contract = self.contracts.get(task_id)

        return {
            "task": task.to_dict(),
            "status": self.task_status.get(task_id, TaskStatus.ANNOUNCED).value,
            "num_bids": len(bids),
            "bids": [b.to_dict() for b in bids],
            "contract": contract.to_dict() if contract else None
        }

    def format_negotiation_log(self, task_id: str) -> str:
        """Format the negotiation process for display."""
        if task_id not in self.tasks:
            return "Task not found"

        task = self.tasks[task_id]
        bids = self.bids.get(task_id, [])
        contract = self.contracts.get(task_id)

        lines = [
            f"═══ Contract Net: {task.task_type.upper()} Task ═══",
            f"Task ID: {task_id}",
            f"Description: {task.description}",
            f"Status: {self.task_status.get(task_id, TaskStatus.ANNOUNCED).value}",
            "",
            "── Bids Received ──"
        ]

        if bids:
            for i, bid in enumerate(bids, 1):
                lines.append(
                    f"  {i}. {bid.bidder}: "
                    f"cost={bid.estimated_cost:.2f}, "
                    f"latency={bid.estimated_latency:.2f}s, "
                    f"success={bid.success_probability:.0%}, "
                    f"score={bid.score:.3f}"
                )
        else:
            lines.append("  (No bids received)")

        if contract:
            lines.extend([
                "",
                "── Contract Awarded ──",
                f"  Winner: {contract.contractor}",
                f"  Status: {contract.status.value}",
            ])
            if contract.result:
                lines.append(f"  Result: {str(contract.result)[:100]}...")

        return "\n".join(lines)
