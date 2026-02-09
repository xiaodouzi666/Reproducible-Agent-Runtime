# RAR (Reproducible Agent Runtime): Design and Implementation of a Reproducible Multi-Agent System for AI4S

---

## Abstract

As artificial intelligence becomes increasingly widely applied in scientific research, the reproducibility issue of AI for Science (AI4S) is becoming increasingly prominent. This paper presents the RAR (Reproducible Agent Runtime) system—a reproducible multi-agent runtime framework for scientific computing scenarios. The system integrates classical theoretical frameworks including BDI (Belief-Desire-Intention) cognitive architecture, Contract Net protocol, and ACL (Agent Communication Language) message protocol. Through complete execution tracing, reproducible runtime mechanisms, and auditable evidence chains, it builds a trustworthy AI scientific computing framework. This paper elaborates on the design principles, technical implementation, construction challenges, and practical value of the RAR system, providing a feasible solution for reproducibility issues in the AI4S field.

**Keywords**: Multi-Agent System, Reproducibility, AI for Science, BDI Architecture, Evidence Chain, Audit Mechanism

---

## Chapter 1 Introduction

### 1.1 Research Background

Artificial intelligence technology is increasingly widely applied in scientific research, from drug discovery to materials science, from bioinformatics to climate modeling, AI is reshaping the paradigm of scientific research. However, with the rapid development of AI4S, a serious problem is gradually emerging—the **reproducibility crisis**.

The core challenges currently facing the AI for Science field include:

- **Black Box Problem**: The decision-making process of AI models lacks transparency, making it difficult to trace reasoning paths
- **Non-reproducible Results**: The same task may produce different results in different runs, lacking deterministic guarantees
- **Broken Evidence Chains**: The connection between scientific conclusions and original data/literature is unclear
- **Difficult to Audit Collaboration**: Decisions in multi-agent collaboration processes cannot be effectively reviewed

### 1.2 Research Motivation and Objectives

The core value of scientific research lies in reproducibility and falsifiability. However, the probabilistic nature of LLMs, complex multi-agent interactions, and lack of unified evidence representation make it difficult to guarantee reproducibility in traditional AI systems.

The RAR system aims to address these problems, with core objectives including:

1. **Completely Transparent Execution Tracing**: Record every step of system operation
2. **Reproducible Runtime Mechanism**: Guarantee result reproducibility through deterministic algorithms and caching mechanisms
3. **Auditable Evidence Chain**: Establish clear connections between conclusions and source materials
4. **Flexible Configuration Modes**: Support cost-quality trade-offs for different scenarios

### 1.3 Paper Structure

This paper is divided into seven chapters with the following structure:

- **Chapter 1**: Introduction, expounding research background, motivation, and objectives
- **Chapter 2**: Theoretical Foundation, introducing the classical theoretical frameworks integrated by the system
- **Chapter 3**: System Architecture, detailing RAR's layered design and implementation of each layer
- **Chapter 4**: Key Technologies and Implementation, deeply exploring technical details of core modules
- **Chapter 5**: Construction Challenges and Solutions, analyzing technical difficulties during development
- **Chapter 6**: Practical Value and Reflection, summarizing gains and insights from the project
- **Chapter 7**: Conclusion and Outlook, summarizing the full text and envisioning future work

---

## Chapter 2 Theoretical Foundation

### 2.1 BDI (Belief-Desire-Intention) Cognitive Architecture

The BDI architecture originates from the field of cognitive science, proposed by scholars such as Bratman for modeling the decision-making processes of rational agents. The architecture divides an agent's cognitive state into three levels:

```
Beliefs → Agent's knowledge of world state
Desires → Goals the agent wishes to achieve
Intentions → Plans the agent commits to execute
```

In the RAR system, the BDI architecture is used to model each agent's decision-making process. When an agent's beliefs change, the system automatically triggers tracing records, persisting state changes to ensure complete reconstruction of the agent's decision-making process during subsequent audits.

### 2.2 Contract Net Protocol

The Contract Net protocol is a standardized multi-agent task allocation protocol by FIPA (Foundation for Intelligent Physical Agents). This protocol simulates the bidding-contract award process in reality, including the following stages:

```
1. CFP (Call for Proposals) - Bidding stage
2. BID - Tender stage
3. EVALUATION - Evaluation stage
4. AWARD - Contract award stage
5. REPORT - Result reporting
```

The RAR system adopts the Contract Net protocol to implement task allocation between agents, selecting the optimal executor through multi-dimensional evaluation (cost, latency, success probability, capability matching).

### 2.3 ACL (Agent Communication Language) Message Protocol

The ACL protocol is based on Speech Act Theory, treating communication between agents as a speech act. The protocol defines multiple speech act types, categorized by function:

| Category | Speech Acts |
|----------|-------------|
| Informative | INFORM, CONFIRM, DISCONFIRM |
| Directive | REQUEST, QUERY |
| Commissive | PROPOSE, COMMIT, ACCEPT, REJECT |
| Contract Net | CFP, BID, AWARD |
| Audit | CHALLENGE, RETRACT, JUSTIFY |

### 2.4 Dung Argumentation Framework

The Dung argumentation framework is a formal argumentation theory for handling conflicting viewpoints. The RAR system introduces an argumentation graph generation mechanism in the highest-level mode (OWL Full), using claim-support-attack structure to represent argumentation relationships, making the reasoning process of scientific conclusions more transparent and auditable.

---

## Chapter 3 System Architecture

### 3.1 Overall Architecture Design

The RAR system adopts a layered architecture design, following the "separation of concerns" principle, with each layer focusing on specific functional responsibilities. From bottom to top: tracing layer, tool layer, protocol layer, agent layer, orchestration layer, and interface layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chapter 6 UI Layer                            │
│                   Streamlit UI / CLI                           │
├─────────────────────────────────────────────────────────────────┤
│                    Chapter 5 Orchestration Layer                 │
│                   Orchestrator (Core Engine)                     │
├─────────────────────────────────────────────────────────────────┤
│                    Chapter 4 Agent Layer                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐│
│  │   Planner    │ │  Researcher  │ │   Executor   │ │ Auditor ││
│  │  (Planner)   │ │ (Researcher)  │ │  (Executor)  │ │(Auditor)││
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    Chapter 3 Protocol Layer                      │
│  ┌──────────────────┐  ┌──────────────────────────────────┐    │
│  │  Contract Net    │  │      ACL Message Protocol        │    │
│  │ (Task Bidding)   │  │  (CFP/BID/AWARD/INFORM/CHALLENGE)│    │
│  └──────────────────┘  └──────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                    Chapter 2 Tool Layer                          │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  LocalSearchTool │      │  PythonExecTool  │                 │
│  │  (BM25 Search)   │      │  (Sandbox Exec)  │                 │
│  └──────────────────┘      └──────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│                    Chapter 1 Tracing Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────┐   │
│  │   Tracer    │  │ TraceStore  │  │   EvidenceAnchor        │   │
│  │ (Tracing API)│  │ (JSONL Store)│  │   (Evidence Anchor)    │   │
│  └─────────────┘  └─────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Tracing Layer (Trace Layer)

The tracing layer is the core infrastructure of the RAR system, responsible for recording all execution details and ensuring complete traceability and reproducibility.

#### 3.2.1 TraceEntry Data Structure

Each trace entry records complete information about an execution event, including the following dimensions:

| Dimension | Fields |
|-----------|--------|
| Identification | run_id, step_id, timestamp |
| Event Type | agent_start, tool_call, message_sent, llm_call, etc. |
| Agent Info | agent_id, agent_role, BDI state |
| Communication Info | performative, sender, receiver, message_id |
| Tool Info | tool_name, tool_input, tool_output, latency_ms |
| LLM Info | model, thinking_level, cache_key, response_hash |
| Evidence Info | evidence_anchors list |
| Reproducibility Info | input_hash, output_hash, deterministic |

#### 3.2.2 JSONL Storage Format

Trace logs are stored in JSONL (JSON Lines) format, offering advantages such as incremental writing, stream processing, scalability, and compressibility.

### 3.3 Tool Layer

#### 3.3.1 LocalSearchTool - BM25 Full-Text Search

Uses BM25 (Best Matching 25) algorithm for full-text retrieval:

```
BM25(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D| / avgdl))
```

#### 3.3.2 PythonExecTool - Sandbox Execution Environment

Security execution mechanisms include: restricted global namespace, preloaded scientific computing libraries, exclusion of dangerous modules, output redirection capture, seed fixation for reproducibility.

### 3.4 Protocol Layer

#### 3.4.1 ACL Message Structure

```python
@dataclass
class ACLMessage:
    performative: Performative      # Speech act type
    sender: str                     # Sender identifier
    receiver: str                   # Receiver identifier
    content: Any                    # Message content
    message_id: str                 # Unique message identifier
    conversation_id: str            # Conversation identifier
    reply_to: Optional[str]         # Reply-to message identifier
    protocol: Optional[str]         # Protocol type
    evidence: List[EvidenceAnchor]  # Evidence anchor list
    artifacts: List[str]            # Artifact list
```

#### 3.4.2 Contract Net State Machine and Bid Evaluation

The protocol maintains a state machine for each task: `ANNOUNCED → BIDDING → AWARDED → IN_PROGRESS → COMPLETED/FAILED`.

Comprehensive score calculation formula:

```
score = success_probability × 0.4 +
         capability_match × 0.3 +
         (1 / (1 + estimated_cost)) × 0.15 +
         (1 / (1 + estimated_latency)) × 0.15
```

### 3.5 Agent Layer

The agent layer implements four specialized agents based on BDI architecture:

| Agent | Responsibilities |
|-------|------------------|
| PlannerAgent | Task decomposition, subtask allocation, result synthesis, replanning |
| ResearcherAgent | BM25 search, evidence anchor generation, relevance filtering |
| ExecutorAgent | Sandbox code execution, variable extraction, chart saving, output capture |
| AuditorAgent | Multi-level audit strategy, multi-dimensional quality checks, challenge generation |

### 3.6 Orchestration Layer

The orchestration layer is the system's core engine, coordinating the entire operational flow. The function calling loop drives tool calls through LLM, continuing to run until finalize or maximum steps are reached.

Each run has clear lifecycle states: `PENDING → RUNNING → COMPLETED/COMPLETED_WITH_WARNINGS/FAILED`

### 3.7 User Interface Layer

Provides two interaction methods: command-line interface and Web interface.

---

## Chapter 4 Key Technologies and Implementation

### 4.1 LLM Integration Layer

#### 4.1.1 Gemini Client Wrapper

The system wraps Google Gemini API, providing deterministic caching, automatic fallback, and structured output functionality.

**Deterministic Caching Mechanism**: Standardize request payload then calculate SHA256 hash as cache key, query cache before executing request.

**Automatic Fallback Strategy**:

```python
try:
    return call_gemini_pro(request)
except TransientError:
    logger.warning("Pro model failed, falling back to Flash")
    return call_gemini_flash(request)
```

**Structured Output**: Convert Pydantic model to JSON Schema, pass to Gemini API, construct repair prompt and retry on validation failure.

#### 4.1.2 Function Calling Design

| Tool | Input | Output | Evidence Support |
|------|-------|--------|------------------|
| research | query, num_results | findings, anchors | ✓ |
| execute | code, seed | output, variables, figures | ✓ |
| audit | subtask_result | passed, challenges | ✓ |
| finalize | answer | summary, answer | ✓ |

### 4.2 Reproducibility Guarantee Mechanism

#### 4.2.1 Multi-level Deterministic Guarantee

| Level | Mechanism |
|-------|-----------|
| Random Number Generation | Fix numpy and random seed |
| Search Algorithm | BM25 deterministic result ordering |
| LLM Call | Hash-based caching mechanism |
| Tool Execution | Input/output hash verification |
| Logging | JSONL append-only writing |

#### 4.2.2 Replay Engine and Difference Analysis

Replay flow: Load original run metadata → Configuration alignment → Cache reuse → Execute replay → Result comparison.

Difference analysis includes: answer matching, step differences, cost differences, cache hit rate.

### 4.3 Evidence Anchor System

```python
@dataclass
class EvidenceAnchor:
    doc_id: str              # Document unique identifier
    doc_title: str           # Document title
    location: str            # Precise location description
    content_hash: str        # Content hash (MD5/SHA)
    content_snippet: str     # Content snippet
    relevance_score: float   # Relevance score
```

Content hash ensures evidence verifiability, calculating and verifying integrity through SHA256.

### 4.4 Configuration System

#### 4.4.1 Three-Level Operating Modes

| Mode | Model | Thinking Level | Audit Level | Evidence Requirements | Use Cases |
|------|-------|----------------|-------------|----------------------|-----------|
| OWL Lite | Flash | minimal | light | 1 anchor, score>0.15 | Quick exploration |
| OWL DL | Pro | high | strict | 2 anchors, score>0.30 | Production |
| OWL Full | Pro | high | argumentation | 3 anchors, score>0.35, argumentation graph | Highest reliability |

#### 4.4.2 Multi-dimensional Audit Checks

Six-dimensional checks: success status, evidence quantity, evidence structure, relevance, runtime errors, unit consistency.

---

## Chapter 5 Construction Challenges and Solutions

### 5.1 Technical Challenges

#### 5.1.1 Determinism and Reproducibility

**Core Challenges**: Probabilistic nature of LLMs, multi-agent concurrent interactions, random factors in tool execution.

**Solutions**: Global seed management, LLM response caching (deterministic hash based on request payload), deterministic algorithms (BM25), output hash verification, JSONL append-only writing.

#### 5.1.2 State Consistency

**Solution**: Adopt sequential execution model to avoid concurrent state update issues, all state changes recorded to JSONL, each TraceEntry contains complete snapshot.

#### 5.1.3 Evidence Chain Integrity

**Solution**: Strict evidence anchor structure (document ID, title, location, hash, snippet), MD5/SHA hash calculation, relevance score filtering.

### 5.2 Design Challenges

#### 5.2.1 Complexity vs. Understandability

**Solutions**: Detailed tracing logs, Streamlit UI visualization, pre-generated example runs, layered architecture, progressive modes.

#### 5.2.2 Performance and Resource Consumption

**Solutions**: JSONL incremental writing, LLM response caching, mode grading, artifact separation.

#### 5.2.3 LLM Integration Uncertainty

**Solutions**: Automatic fallback strategy (Pro → Flash), schema retry mechanism, graceful degradation, rich error handling.

### 5.3 Engineering Challenges

#### 5.3.1 Error Handling and Degradation

**"finalize-first" Invariant**: Final confirmation is a hard threshold, argumentation graph generation and other additional features are best-effort.

**Status Codes**: completed, completed_with_warnings, failed, waiting.

#### 5.3.2 Testing and Validation

**Solutions**: Pre-generated example runs, fixed seed testing, modular design, mock mechanism.

---

## Chapter 6 Practical Value and Reflection

### 6.1 Theoretical Knowledge Deepening

Through practical implementation of BDI architecture, understood how to transform abstract cognitive science theories into executable code. State tracing mechanisms make agents' decision-making processes completely transparent, more interpretable than traditional "black box" AI systems.

Contract Net protocol demonstrates fair and efficient task allocation mechanism design, ACL message protocol embodies the application of speech act theory in actual systems.

### 6.2 Engineering Practice Skills

| Feature | Application |
|---------|-------------|
| dataclass | Define immutable data structures, reduce boilerplate code |
| Enum | Type-safe state management |
| ABC | Define and enforce interface contracts |
| contextlib | Output redirection and resource management |

The six-layer architecture design demonstrates the power of layered architecture, with design patterns including dependency injection, strategy pattern, and factory pattern fully applied.

For data persistence, JSONL format supports incremental writing, stream processing, scalability, and compressibility; content-addressed storage supports deduplication and verification; metadata separation keeps file responsibilities clear.

### 6.3 Cross-disciplinary Knowledge Integration

| Domain | Knowledge Gained |
|--------|------------------|
| Cognitive Science | BDI architecture transforms rational agent models into executable code |
| Philosophy of Language | Speech act theory provides formal semantic framework for communication |
| Game Theory | Contract Net protocol demonstrates auction mechanism in task allocation |
| Evidence Theory | Evidence anchors and hash verification embody rigor of evidence theory |
| Philosophy of Science | Reproducibility and falsifiability are core principles of scientific research |

### 6.4 Software Engineering Thinking

**Reproducibility Engineering**: All random sources configurable with seed, all inputs/outputs recorded, model versions and parameters tracked.

**Progressive Enhancement**: owl_lite/owl_dl/owl_full design allows users to choose appropriate modes based on needs, argumentation graph generation and other additional features designed as best-effort.

**Error Handling Philosophy**: Fallback to rule-based planning when LLM unavailable, each state has clear definition, error messages include context and suggestions.

---

## Chapter 7 Conclusion and Outlook

### 7.1 Research Summary

The RAR project is a complex system combining theory and practice with the following characteristics:

1. **Solves Practical Problems**: Addresses reproducibility crisis in AI4S field
2. **Integrates Multiple Theories**: BDI, Contract Net, ACL, Dung argumentation framework
3. **Excellent Engineering Implementation**: Clear architecture, complete tracing, flexible configuration
4. **High Learning Value**: Covers multi-agent systems, LLM engineering, distributed systems, and other fields

Project Innovations:

| Innovation | Description |
|------------|-------------|
| Completeness | From BDI architecture to Contract Net protocol, from evidence chain to argumentation graph |
| Reproducibility | Multiple mechanisms including seed fixation, LLM caching, deterministic algorithms |
| Transparency | Complete trace records make every decision step traceable |
| Flexibility | Three-level mode design allows cost-reliability trade-off |

### 7.2 Contributions and Significance

**Academic Contributions**: Proposed a reproducible multi-agent system framework for AI4S, implemented complete evidence anchor system, designed multi-level audit mechanism, provided reproducibility engineering practice paradigm.

**Practical Significance**: Provided trustworthy AI auxiliary tools for scientific computing scenarios, provided reference for observability design of multi-agent systems, provided practical experience for LLM application engineering.

### 7.3 Future Work

**Function Expansion**: Distributed execution, integration of more scientific computing tools, interactive argumentation graph visualization, multi-format result export.

**Performance Optimization**: Incremental tracing, intelligent caching based on semantic similarity, large-scale data stream processing.

**Ecosystem Building**: Plugin system supporting third-party extensions, promoting standardization of reproducible AI systems, establishing open source community.

### 7.4 Conclusion

The RAR project provides a feasible solution for reproducibility issues in the AI for Science field. By integrating theories from multiple domains including cognitive science, philosophy of language, and game theory, combined with software engineering best practices, we have built a trustworthy, reproducible, and auditable AI system framework.

This project not only solves practical problems but more importantly demonstrates how to build a responsible AI system—one that can explain its decision-making process, be audited and verified, and produce reliable results. This is the core capability needed for the future AI for Science field.

```
Transparency + Reproducibility + Auditability = Trustworthy AI4S
```

---

## Appendix

### A. System Directory Structure

```
RAR/
├── src/
│   ├── agents/           # Agent implementations
│   ├── protocols/        # Protocol implementations
│   ├── tools/            # Tool implementations
│   ├── tracing/          # Tracing system
│   ├── llm/              # LLM integration
│   └── orchestrator.py   # Orchestrator
├── corpus/               # Local corpus
├── runs/                 # Run data
├── ui/                   # User interface
├── tests/                # Test code
├── DESIGN.md             # Design document
└── README.md             # User guide
```

### B. Configuration File Example

```yaml
# run_spec.yaml
task_description: "Analyze the application of quantum computing in cryptography"
mode: "owl_dl"
model: "gemini-3-pro-preview"
thinking_level: "high"
seed: 42
corpus_path: "./corpus"
max_steps: 20
```

### C. Trace Entry Example

```json
{
  "run_id": "run_20250109_120000",
  "step_id": 5,
  "timestamp": "2025-01-09T12:00:15.123Z",
  "event_type": "llm_call",
  "agent_id": "planner_001",
  "agent_role": "planner",
  "model": "gemini-3-pro-preview",
  "thinking_level": "high",
  "cache_key": "abc123...",
  "response_hash": "def456...",
  "deterministic": true
}
```

---

**Document Version**: v1.0
**Last Updated**: 2025-01-09
**Author**: RAR Project Team
