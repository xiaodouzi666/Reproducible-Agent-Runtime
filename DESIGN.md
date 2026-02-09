# RAR Design Document

## 1. Overview

### 1.1 Goals
RAR (Reproducible Agent Runtime) is a multi-agent framework designed to:
1. Execute complex AI4S tasks through agent collaboration
2. Record every step for full traceability
3. Enable reproducible replay of any execution
4. Support comparison between runs

### 1.2 Design Principles
- **Transparency**: All agent decisions and communications are visible
- **Reproducibility**: Same inputs + seed = same outputs
- **Auditability**: Evidence chain links conclusions to sources
- **Simplicity**: Minimal dependencies, easy to deploy

## 2. Architecture

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                   (Streamlit UI / CLI)                          │
├─────────────────────────────────────────────────────────────────┤
│                        Orchestrator                              │
│              (Workflow coordination, run management)            │
├─────────────────────────────────────────────────────────────────┤
│                         Agent Layer                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐│
│  │   Planner    │ │  Researcher  │ │   Executor   │ │ Auditor ││
│  │ (BDI/Coord)  │ │   (Search)   │ │  (Compute)   │ │(Verify) ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘│
├─────────────────────────────────────────────────────────────────┤
│                       Protocol Layer                             │
│  ┌────────────────────────┐  ┌────────────────────────────────┐ │
│  │   Contract Net         │  │         ACL Messages           │ │
│  │ (CFP→Bid→Award)        │  │  (Performatives: inform,       │ │
│  │                        │  │   request, challenge, etc.)    │ │
│  └────────────────────────┘  └────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                         Tool Layer                               │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  LocalSearchTool │      │  PythonExecTool  │                 │
│  │  (BM25 over      │      │  (Sandbox exec   │                 │
│  │   local corpus)  │      │   for compute)   │                 │
│  └──────────────────┘      └──────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│                        Tracing Layer                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │   Tracer    │  │   TraceStore    │  │   EvidenceAnchor     │ │
│  │ (API)       │  │   (JSONL)       │  │   (Citations)        │ │
│  └─────────────┘  └─────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Descriptions

#### Orchestrator
- Entry point for task execution
- Initializes agents and tools
- Manages run lifecycle
- Coordinates replay and diff operations

#### Agents

| Agent | Role | Key Responsibilities |
|-------|------|---------------------|
| Planner | Coordinator | Task decomposition, Contract Net management, plan synthesis |
| Researcher | Information | Corpus search, evidence collection, citation generation |
| Executor | Computation | Python code execution, data analysis, visualization |
| Auditor | Quality | Evidence verification, challenge weak conclusions, trigger replan |

#### Protocols

**Contract Net Protocol**:
```
Planner                    Workers
   │                          │
   │──── CFP (task) ─────────>│
   │<─── Bid (cost,prob) ─────│
   │──── Award (winner) ─────>│
   │<─── Result ──────────────│
```

**ACL Speech Acts**:
- Informative: INFORM, CONFIRM, DISCONFIRM
- Directive: REQUEST, QUERY
- Commissive: PROPOSE, COMMIT, ACCEPT, REJECT
- Contract Net: CFP, BID, AWARD
- Audit: CHALLENGE, JUSTIFY, RETRACT

## 3. Trace Schema

### 3.1 TraceEntry Fields

```python
@dataclass
class TraceEntry:
    # Identifiers
    run_id: str
    step_id: int
    timestamp: str

    # Event type
    event_type: TraceEventType

    # Agent info
    agent_id: str
    agent_role: str

    # Communication
    performative: str
    message_id: str
    sender: str
    receiver: str

    # Content
    content: Any
    content_summary: str

    # Tool info
    tool_name: str
    tool_input: dict
    tool_output: Any
    tool_output_hash: str

    # Evidence
    evidence_anchors: list

    # Metrics
    latency_ms: float
    status: str

    # BDI state
    beliefs: dict
    desires: list
    intentions: list

    # Reproducibility
    input_hash: str
    output_hash: str
```

### 3.2 Event Types

| Category | Events |
|----------|--------|
| Agent | agent_start, agent_end |
| BDI | belief_update, desire_set, intention_form |
| Communication | message_sent, message_received |
| Contract Net | cfp_issued, bid_submitted, contract_awarded |
| Tool | tool_call, tool_result |
| Task | task_start, task_complete, task_fail |
| Audit | challenge_raised, justification_provided, replan_triggered |

## 4. Workflow Example

### 4.1 Task: "Calculate activation energy from thermal analysis data"

```
Step 1: Planner receives task
        - Updates beliefs: {task: "...", workers: [...]}
        - Sets desire: "Complete task"
        - Forms intentions: [research, execute, audit]

Step 2: Planner issues CFP for research subtask
        - Contract Net: CFP → Researcher bids → Award

Step 3: Researcher executes search
        - Tool: local_search(query="activation energy thermal")
        - Returns: 5 relevant passages with evidence anchors
        - Message: INFORM(findings, evidence)

Step 4: Planner issues CFP for execute subtask
        - Contract Net: CFP → Executor bids → Award

Step 5: Executor runs Arrhenius calculation
        - Tool: python_exec(code="...")
        - Returns: Ea = 125.34 kJ/mol, R² = 0.9876
        - Message: INFORM(result, computation_evidence)

Step 6: Auditor validates results
        - Checks evidence count (>= 2) ✓
        - Checks relevance scores (>= 0.3) ✓
        - Message: CONFIRM(approved)

Step 7: Planner synthesizes final answer
        - Combines findings with citations
        - Generates evidence chain
```

## 5. Reproducibility Design

### 5.1 Determinism

To ensure reproducibility:
1. All tools accept optional `seed` parameter
2. NumPy/random seeds are fixed per run
3. Search results are deterministic (BM25 scoring)
4. Output hashes recorded for comparison

### 5.2 Replay Mechanism

```python
# Replay a previous run
engine = ReplayEngine()
result = engine.replay(original_run_id)

# Compare with original
comparison = engine.compare_runs(original_run_id, new_run_id)
# Returns: {identical: bool, step_differences: [...]}
```

### 5.3 Diff Report

Compares:
1. Final answers (text diff)
2. Step outputs (hash comparison)
3. Latency/cost (numerical diff)
4. Tool usage (count comparison)
5. Evidence collected (document overlap)

## 6. Evidence Chain

### 6.1 EvidenceAnchor Structure

```python
@dataclass
class EvidenceAnchor:
    doc_id: str           # "paper_001"
    doc_title: str        # "Thermal Analysis Methods"
    location: str         # "paragraph_3"
    content_hash: str     # "abc123..."
    snippet: str          # "The Kissinger method..."
    relevance_score: float # 0.85
```

### 6.2 Citation Generation

Final answer format:
```markdown
## Findings
The activation energy is 125.34 kJ/mol [1][2].

## Evidence Sources
[1] Thermal Analysis Methods - paragraph_3
[2] Polymer Decomposition Study - paragraph_5
```

## 7. Evaluation Criteria

### 7.1 KAS Alignment

| Requirement | Implementation |
|-------------|----------------|
| Multi-agent | 4 agents with distinct roles |
| BDI visible | Beliefs/desires/intentions in trace |
| Contract Net | CFP/Bid/Award workflow |
| ACL messages | Performatives in every message |
| Traceability | JSONL trace with all steps |
| Reproducibility | Seed-based determinism + replay |
| Evidence chain | Anchors link results to sources |

### 7.2 Demo Metrics

- **Startup time**: < 5 seconds
- **Task execution**: < 30 seconds (no LLM)
- **UI responsiveness**: Interactive
- **Offline capable**: Pre-generated sample run

## 8. MVP Scope and Future Extensions

### 8.1 Current MVP Scope

The current demo provides an MVP closed loop (RunSpec/Trace/Replay/Diff + auditable collaboration). Core capabilities include:
- **RunSpec**: YAML task specification with fixed seed and corpus configuration.
- **Trace**: End-to-end JSONL logging, with per-step agent/tool/performative/evidence records.
- **Replay**: One-command rerun for deterministic consistency checks.
- **Diff**: Multi-dimensional comparison (answer/steps/cost/tools/evidence).
- **Auditable collaboration**: Visible BDI state, traceable Contract Net flow, and ACL messages with performatives.

### 8.2 Future Extensions

1. **Artifact Store (hot/cold tiering)**: Tiered storage for large files/model weights; trace stores references only.
2. **Dashboard**: Real-time monitoring for concurrent runs and visualization of agent interaction topology.
3. **Dung-style argument adjudication**: Conflict detection and resolution based on argumentation frameworks.
4. **LLM Integration**: Optional OpenAI/Anthropic integrations for stronger planner intelligence.
5. **Distributed Execution**: Cross-machine distributed agent scheduling.
6. **Real-time Collaboration**: Multi-user observation/intervention during execution.
7. **Formal Verification**: Formal proofs for reproducibility properties.
