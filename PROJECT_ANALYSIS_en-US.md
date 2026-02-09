# RAR (Reproducible Agent Runtime) - Comprehensive Project Analysis

## Table of Contents

1. [Project Inspiration and Background](#project-inspiration-and-background)
2. [Technology Stack and Implementation](#technology-stack-and-implementation)
3. [Challenges and Pain Points](#challenges-and-pain-points)
4. [Learning Outcomes and Growth](#learning-outcomes-and-growth)

---

## Project Inspiration and Background

### 1.1 Core Inspiration

**AI4S (AI for Science) Reproducibility Crisis**

This project originated from deep reflection on the reproducibility issues of AI applications in scientific research. In the current AI for Science domain, the following pain points exist:

- **Black Box Problem**: AI model decision-making processes lack transparency and are difficult to trace
- **Irreproducible Results**: The same task may produce different results across different runs
- **Broken Evidence Chains**: The connection between scientific conclusions and original data/literature is unclear
- **Difficult-to-Audit Collaboration**: Decisions in multi-agent collaboration processes cannot be effectively reviewed

### 1.2 Integration of Theoretical Foundations

The project cleverly integrates multiple classic theoretical frameworks:

- **BDI (Belief-Desire-Intention) Architecture**: Agent modeling methodology from cognitive science
- **Contract Net Protocol**: FIPA standard multi-agent task allocation protocol
- **ACL (Agent Communication Language)**: Communication protocol based on Speech Act Theory
- **Dung Argumentation Framework**: Formal argumentation theory for handling conflicting viewpoints

### 1.3 Design Philosophy

```
Transparency + Reproducibility + Auditability = Trustworthy AI4S
```

The core philosophy of the project: To build a trustworthy AI scientific computing framework through completely transparent execution tracing, reproducible execution mechanisms, and auditable evidence chains.

---

## Technology Stack and Implementation

### 2.1 Overall Architecture Design

The RAR system employs a layered architecture design, comprising from bottom to top: Tracing Layer, Tool Layer, Protocol Layer, Agent Layer, Orchestration Layer, and Interface Layer. This design follows the "separation of concerns" principle, where each layer focuses on specific functional responsibilities, and layers communicate through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                     │
│                   Streamlit UI / CLI                           │
├─────────────────────────────────────────────────────────────────┤
│                       Orchestration Layer                       │
│                   Orchestrator (Core Engine)                    │
├─────────────────────────────────────────────────────────────────┤
│                          Agent Layer                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐│
│  │   Planner    │ │  Researcher  │ │   Executor   │ │ Auditor ││
│  │(Task Decomp)  │ │(Info Retrieval)│ │ (Code Exec)  │ │(Quality)││
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘│
├─────────────────────────────────────────────────────────────────┤
│                         Protocol Layer                          │
│  ┌──────────────────┐  ┌──────────────────────────────────┐    │
│  │  Contract Net    │  │      ACL Message Protocol        │    │
│  │(Task Bidding)    │  │  (CFP/BID/AWARD/INFORM/CHALLENGE)│    │
│  └──────────────────┘  └──────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                          Tool Layer                             │
│  ┌──────────────────┐      ┌──────────────────┐                 │
│  │  LocalSearchTool │      │  PythonExecTool  │                 │
│  │(BM25 Full-Search)│      │ (Sandbox Code Exec)│                 │
│  └──────────────────┘      └──────────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│                         Tracing Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────┐   │
│  │   Tracer    │  │ TraceStore  │  │   EvidenceAnchor        │   │
│  │(Tracing API)│  │(JSONL Store)│  │  (Evidence Anchor Sys)   │   │
│  └─────────────┘  └─────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Layer: BDI Architecture Implementation

#### 2.2.1 Agent Abstract Base Class Design

The agent layer design is based on the BDI (Belief-Desire-Intention) cognitive architecture. This architecture originates from the field of cognitive science and is used to model the decision-making process of rational agents. The `BaseAgent` class defines common behavioral patterns for all agents.

**BDI State Structure**:

Agents maintain three core cognitive states:
- **Beliefs**: The agent's knowledge of the current world state, including task information, available tools, other agent states, etc.
- **Desires**: The goals the agent wishes to achieve, such as "complete task," "verify evidence," etc.
- **Intentions**: Specific plan steps the agent commits to execute

In implementation, when an agent's beliefs change, the system automatically triggers trace recording, persisting state changes to the trace log. This ensures the agent's decision-making process can be completely reconstructed during subsequent auditing.

**Lifecycle Management**:

Each agent has clear lifecycle states: `IDLE`, `ACTIVE`, `WAITING`, `COMPLETED`, `FAILED`. Agents record corresponding trace events when activated and deactivated, forming a complete lifecycle audit trail.

**Message Communication Mechanism**:

Communication between agents occurs through ACL (Agent Communication Language) messages. When sending messages, the system automatically records message content, sender, receiver, speech act type, and other information to the trace log. This design ensures all communication is traceable.

#### 2.2.2 Planner Agent

The planner agent serves as the coordination center for the entire system, responsible for task decomposition, subtask allocation, and result synthesis.

**Task Decomposition Strategy**:

The planner agent supports two task decomposition modes: LLM-driven intelligent planning and rule-based fallback planning. In LLM-driven mode, the planner uses Gemini API's function calling feature, defining tool schemas (such as research, execute, audit, finalize) to let the large model autonomously decide how to complete complex tasks.

In the function calling loop, the system maintains a conversation history including user prompts, model responses, and tool execution results. Each LLM call generates one or more function call requests; the system invokes corresponding tool handlers based on the requests and returns results to the LLM for continued processing. This process continues until the LLM calls the finalize function or reaches the maximum step limit.

**Contract Net Protocol Implementation**:

For traditional task allocation, the planner agent implements the Contract Net Protocol. This protocol simulates the bidding-bid-awarding process in reality:

1. **Call for Proposals (CFP) Phase**: The planner agent publishes a task announcement, including task type (research/execute/audit), description, and requirements
2. **Bid Phase**: Worker agents evaluate the task and submit bids, including estimated cost, latency, success probability, and capability match
3. **Evaluation Phase**: The planner agent calculates a composite score for each bid and selects the optimal bidder
4. **Contract Award (AWARD)**: Awards the contract to the winner to begin task execution
5. **Result Reporting**: The contract executor reports results to the planner agent upon completion

**Replanning Mechanism**:

When the auditor agent challenges results, the planner agent triggers a replanning process. The system maintains a maximum replanning count parameter; after exceeding this limit, even if unresolved challenges remain, the system completes the task with a "completed_with_warnings" status, ensuring the system doesn't fall into an infinite loop.

#### 2.2.3 Researcher Agent

The researcher agent is responsible for retrieving relevant information from the local corpus and collecting evidence.

**BM25 Search Algorithm**:

The researcher uses the BM25 (Best Matching 25) algorithm for full-text search. BM25 is a ranking function based on probabilistic retrieval models that considers term frequency (TF) of query terms in documents and inverse document frequency (IDF) of query terms in the entire corpus, effectively assessing document-query relevance.

During initialization, the system loads all documents from the corpus, splits each document into paragraphs, and builds BM25 indexes for all paragraphs. For a given query, the system calculates BM25 scores for query terms against each paragraph, returning the most relevant paragraphs.

**Evidence Anchor Generation**:

For each search result, the system generates an evidence anchor. Evidence anchors are the key mechanism connecting conclusions to source documents, containing the following elements:

- **Document Identifier**: Uniquely identifies the source document
- **Document Title**: Title of the source document
- **Location Information**: Precise location description (such as paragraph number)
- **Content Hash**: Content hash calculated using MD5 or SHA algorithms for evidence integrity verification
- **Content Snippet**: Relevant text content
- **Relevance Score**: Relevance score calculated by BM25 algorithm

The content hash design ensures evidence verifiability. If someone questions the correctness of a conclusion, the hash value can verify whether the cited content has been tampered with.

#### 2.2.4 Executor Agent

The executor agent is responsible for executing Python code in a secure sandbox environment for scientific computation and data analysis.

**Sandbox Execution Environment**:

To ensure system security, the executor creates a restricted global namespace. This namespace only contains safe built-in functions (such as abs, len, max, etc.) and preloaded scientific computing libraries (numpy, pandas, scipy, matplotlib). Dangerous modules and functions (such as os, subprocess, eval, exec, etc.) are excluded.

Code is preprocessed before execution to remove import statements for already preloaded modules, avoiding duplicate imports and potential conflicts.

**Output Capture Mechanism**:

The system uses Python's `contextlib.redirect_stdout` and `redirect_stderr` to capture standard output and standard error during code execution. This allows the system to record all code output, including printed intermediate results and error information.

**Variable Extraction and Figure Saving**:

After execution completes, the system extracts all non-private variables from the local namespace (variables not starting with "_"). For serializable basic types (int, float, str, bool, list, dict), their values are saved directly; for other types, they are converted to string representations.

If the code generates matplotlib figures, the system automatically detects and saves them. Each figure is saved as a PNG file, with a base64 encoded version also generated for easy embedding in web interfaces.

#### 2.2.5 Auditor Agent

The auditor agent is responsible for verifying result quality, challenging weak conclusions, and ensuring system reliability.

**Multi-level Audit Strategy**:

The system supports three runtime modes, each corresponding to different audit strictness levels:

- **OWL Lite**: Minimum requirements (1 evidence anchor, relevance score > 0.15), optional audit
- **OWL DL**: Moderate requirements (2 evidence anchors, relevance score > 0.30), audit must pass
- **OWL Full**: Highest requirements (3 evidence anchors, relevance score > 0.35, requires argumentation graph), audit must pass

This progressive design allows users to choose appropriate strictness levels based on their needs, balancing computational cost and result quality.

**Multi-dimensional Quality Checks**:

The auditor agent validates subtask results from multiple dimensions:

1. **Success Status Check**: Verifies whether the task completed successfully
2. **Evidence Count Check**: Ensures evidence anchor count meets minimum requirements
3. **Evidence Structure Check**: Verifies evidence anchors contain necessary fields (doc_id, location, content_hash)
4. **Relevance Check**: Calculates average relevance score to ensure evidence quality
5. **Runtime Error Detection**: Uses regex pattern matching to detect error indicators in output (such as "traceback", "exception", "error", etc.)
6. **Unit Consistency Check**: For execution tasks, checks whether values and units in output are consistent

**Challenge Generation and Repair Suggestions**:

When audit fails, the system generates specific challenge information, including:

- **Challenge Reason**: Describes the specific problem (such as "insufficient evidence", "low relevance")
- **Severity Level**: low, medium, high
- **Repair Suggestions**: Targeted improvement recommendations
- **Follow-up Query Suggestions**: For research tasks, provides improved search queries
- **Computation Suggestions**: For execution tasks, provides recomputation guidance

### 2.3 Protocol Layer Implementation

#### 2.3.1 ACL Message Protocol

The ACL (Agent Communication Language) protocol is based on Speech Act Theory, treating communication between agents as a speech act.

**Speech Act Type Classification**:

The protocol defines multiple speech act types, categorized by function:

- **Informative**: INFORM (share information), CONFIRM (confirm), DISCONFIRM (deny)
- **Directive**: REQUEST (request action), QUERY (ask for information)
- **Commissive**: PROPOSE (propose), COMMIT (commit), ACCEPT (accept), REJECT (reject)
- **Contract Net Specific**: CFP (call for proposals), BID (submit bid), AWARD (award contract)
- **Audit Specific**: CHALLENGE (challenge), RETRACT (retract), JUSTIFY (provide justification)

Each speech act has clear semantics and usage scenarios. For example, when the auditor agent discovers a problem, it uses the CHALLENGE speech act to issue a challenge to the planner agent; when the challenge is resolved, it can send a JUSTIFY message to provide reasons and evidence.

**Message Structure Design**:

ACL messages contain the following core elements:

- **Performative**: Speech act type
- **Sender/Receiver**: Sender and receiver identifiers
- **Content**: Message content
- **Message ID**: Unique message identifier
- **Conversation ID**: Conversation identifier for correlating related messages
- **Reply To**: Identifier of the message being replied to, supporting conversation chains
- **Protocol**: Protocol type (such as "contract-net")
- **Evidence**: Attached evidence anchor list
- **Artifacts**: Attached artifacts (such as generated files)

This structure design makes all communication traceable and auditable.

#### 2.3.2 Contract Net Protocol

The Contract Net Protocol is a classic multi-agent task allocation protocol simulating the bidding process in reality.

**Protocol State Machine**:

The protocol maintains a state machine for each task, including the following states:
- ANNOUNCED: CFP published, waiting for bids
- BIDDING: Collecting bids
- AWARDED: Contract awarded
- IN_PROGRESS: Task in progress
- COMPLETED: Task completed
- FAILED: Task failed
- CANCELLED: Task cancelled

**Bid Evaluation Algorithm**:

Each bid contains multiple evaluation dimensions:
- **Estimated Cost**: Abstract cost unit
- **Estimated Latency**: Estimated execution latency
- **Success Probability**: Success probability (0-1)
- **Capability Match**: Capability match degree

The system uses a weighted formula to calculate composite scores:
```
score = success_probability * 0.4 +
         capability_match * 0.3 +
         (1 / (1 + estimated_cost)) * 0.15 +
         (1 / (1 + estimated_latency)) * 0.15
```

This design makes bid evaluation a multi-dimensional balance rather than considering a single factor.

### 2.4 Tool Layer Implementation

#### 2.4.1 Local Search Tool (LocalSearchTool)

The local search tool is responsible for full-text retrieval of the local corpus.

**Corpus Loading and Index Building**:

During tool initialization, it traverses all documents in the specified directory (supporting .txt, .md, .markdown formats), processing each document as follows:

1. **Title Extraction**: Preferentially extracted from Markdown titles, otherwise using filename
2. **Paragraph Splitting**: Splits document by double newlines, filtering out overly short paragraphs
3. **Tokenization**: Uses regex to extract words, converted to lowercase
4. **BM25 Index Building**: Builds BM25 indexes for tokenized results of all paragraphs

For environments where BM25 library is unavailable, the system automatically degrades to simple keyword matching algorithms, ensuring basic system availability.

**Relevance Calculation**:

For a given query, the system tokenizes it and uses the BM25 algorithm to calculate scores for each paragraph. BM25 considers term frequency (TF) and inverse document frequency (IDF), effectively assessing document-query relevance.

#### 2.4.2 Python Execution Tool (PythonExecTool)

The Python execution tool is responsible for executing scientific computing code in a secure sandbox environment.

**Secure Execution Mechanism**:

The tool implements multi-layer security protection mechanisms:

1. **Restricted Built-in Functions**: Only provides safe built-in functions, removing dangerous functions (such as open, file, __import__, etc.)
2. **Preloaded Scientific Libraries**: Preloads numpy, pandas, scipy, matplotlib and other libraries, users need not import
3. **Code Preprocessing**: Removes redundant import statements to avoid conflicts
4. **Output Redirection**: Uses contextlib to capture standard output and errors
5. **Seed Fixing**: If seed parameter is provided, fixes numpy and random random seeds to ensure reproducibility

**Artifact Management**:

The tool automatically detects and saves figures generated by matplotlib. Each figure is saved as a PNG file, with base64 encoding generated for easy display in web interfaces.

### 2.5 Tracing Layer Implementation

The tracing layer is the core infrastructure of the RAR system, responsible for recording all execution details and ensuring complete traceability and reproducibility.

#### 2.5.1 Tracing Schema Definition

**TraceEntry**:

Each trace entry records complete information about an execution event, including:

- **Identification Information**: run_id, step_id, timestamp
- **Event Type**: agent_start, tool_call, message_sent, llm_call, and over 30 other types
- **Agent Information**: agent_id, agent_role, BDI state
- **Communication Information**: performative, sender, receiver, message_id
- **Tool Information**: tool_name, tool_input, tool_output, latency_ms
- **LLM Information**: model, thinking_level, cache_key, response_hash
- **Evidence Information**: evidence_anchors list
- **Reproducibility Information**: input_hash, output_hash, deterministic

This design ensures each execution step is completely recorded, allowing precise reconstruction later.

**Hash Calculation Mechanism**:

The system uses the SHA256 algorithm to calculate hash values for inputs and outputs for reproducibility verification. For non-serializable objects, they are converted to strings before hashing. These hash values are used in diff comparisons to quickly determine whether two runs produced identical results.

#### 2.5.2 Tracer Implementation

The tracer provides convenient logging interfaces for recording different types of events.

**Event Classification**:

The tracer supports recording the following categories of events:

- **Agent Lifecycle**: agent_start, agent_end
- **BDI State Changes**: belief_update, desire_set, intention_form
- **Message Communication**: message_sent, message_received
- **Contract Net**: cfp_issued, bid_submitted, contract_awarded
- **Tool Calls**: tool_call, tool_result
- **LLM Calls**: llm_call, llm_result, llm_cache_hit, schema_violation
- **Task Execution**: task_start, task_complete, task_fail
- **Audit**: challenge_raised, justification_provided, replan_triggered

Each event type has dedicated recording methods that automatically populate relevant fields.

**JSONL Storage Format**:

Trace logs are stored in JSONL (JSON Lines) format, with one JSON object per line. The advantages of this format include:

1. **Incremental Writing**: Can append directly to file end without loading entire content
2. **Stream Processing**: Can read and process line by line with small memory footprint
3. **Scalability**: New lines can be appended directly without rewriting entire file
4. **Compressibility**: Text format facilitates compressed storage

#### 2.5.3 Trace Store (TraceStore)

The trace store is responsible for managing trace data for all runs.

**File Organization Structure**:

Each run has its own directory containing the following files:
- `metadata.json`: Run metadata (task description, configuration, status, etc.)
- `trace.jsonl`: Trace log (one TraceEntry per line)
- `final.json`: Final result and audit information
- `run_spec.yaml`: Run specification (for reproduction)
- `llm_cache.jsonl`: LLM response cache
- `artifacts/`: Artifacts directory (figures, etc.)

This organization makes each run self-contained, facilitating management and distribution.

### 2.6 LLM Integration Layer

#### 2.6.1 Gemini Client Wrapper

The system wraps Google Gemini API, providing caching, degradation, and structured output capabilities.

**Deterministic Caching Mechanism**:

To ensure reproducibility and reduce costs, the system implements a hash-based caching mechanism:

1. **Cache Key Calculation**: Standardizes request payload (prompt, model, thinking_level, system_prompt, etc.) and calculates SHA256 hash
2. **Cache Query**: Queries cache before executing request, returns directly if hit
3. **Cache Write**: Writes result to cache after executing request
4. **Read-only Cache Paths**: Supports specifying read-only cache paths for reproduction scenarios

The caching strategy significantly improves system reproducibility. Identical requests will always return identical results as long as the original cache exists.

**Automatic Degradation Strategy**:

The system implements automatic degradation from Pro model to Flash model. When a Pro model request fails and the error belongs to transient errors (such as 429 rate limiting, timeout, resource exhaustion), the system automatically uses the Flash model to retry the request. This design improves system robustness but records degradation information for auditing.

**Structured Output and Schema Validation**:

The system supports forcing LLM output to comply with predefined Pydantic schemas. Implementation steps:

1. **Schema Normalization**: Converts Pydantic model class to JSON Schema format
2. **Request Building**: Passes JSON Schema to Gemini API's response_json_schema parameter
3. **Response Validation**: Uses Pydantic's model_validate_json to validate response
4. **Error Retry**: Constructs repair prompt and retries when validation fails

This mechanism ensures structured LLM output for easier subsequent processing.

#### 2.6.2 Function Calling Loop

The function calling loop is the core mechanism of LLM-driven execution.

**Conversation History Maintenance**:

The system maintains a complete conversation history including all user messages, model responses, and tool results. This ensures the LLM can understand previous interaction context.

Specifically, for model responses containing function calls, the system preserves complete thought signatures and function call information, ensuring context consistency for subsequent requests.

**Tool Handler Mapping**:

Each tool (research, execute, audit, finalize) has a corresponding handler function responsible for executing actual operations. Tool handler input is parameters parsed by the LLM, and output is a structured result object.

**Termination Conditions**:

The loop terminates under the following conditions:
1. LLM calls finalize function and returns finalized=true
2. LLM calls wait_seconds or wait_until function (for pausing long-running tasks)
3. Maximum step limit is reached

If maximum steps are reached without finalize, the system attempts a "force finalize," explicitly informing the LLM that it must call the finalize function.

### 2.7 Configuration System

#### 2.7.1 Runtime Mode Configuration

The system supports three runtime modes corresponding to different computational costs and strictness levels:

**OWL Lite (Lightweight Mode)**:

- Model: gemini-3-flash-preview
- Thinking Level: minimal
- Audit Level: light
- Evidence Requirements: At least 1 evidence anchor per claim
- Audit Requirements: Optional
- Replanning: Not supported
- Use Cases: Quick exploration, cost-sensitive applications

**OWL DL (Description Logic Mode)**:

- Model: gemini-3-pro-preview
- Thinking Level: high
- Audit Level: strict
- Evidence Requirements: Coverage and consistency checks
- Audit Requirements: Must pass, otherwise replan
- Replanning: Maximum 2 rounds
- Use Cases: Production environment, requiring higher reliability

**OWL Full (Complete Mode)**:

- Model: gemini-3-pro-preview
- Thinking Level: high
- Audit Level: argumentation
- Evidence Requirements: Must include claim-support-attack structure
- Audit Requirements: Must pass
- Replanning: Maximum 2 rounds
- Additional Features: Generate argumentation graph (Dung-style)
- Use Cases: Highest reliability requirements, need complete argumentation chain

### 2.8 Replay and Reproducibility

#### 2.8.1 Replay Engine Design

The replay engine is responsible for reproducing previous runs and verifying system reproducibility.

**Replay Process**:

1. **Load Original Run Metadata**: Read task description, configuration, seed, and other information
2. **Configuration Alignment**: Use same seed, model, thinking_level, and other configurations
3. **Cache Reuse**: Set original run's LLM cache path to read-only, ensuring consistent LLM responses
4. **Execute Replay**: Run the same task
5. **Result Comparison**: Compare whether results from both runs are consistent

**Reproducibility Guarantees**:

The system ensures reproducibility through the following mechanisms:

1. **Fixed Seed**: numpy and random random seeds are fixed
2. **Deterministic Algorithms**: BM25 search results are deterministic
3. **LLM Caching**: Identical requests return cached results
4. **Input/Output Hashes**: Record input_hash and output_hash for each step for diff comparison

**Difference Analysis**:

After replay completion, the system compares both runs:

- **Answer Match**: Whether final answers are identical (after normalization)
- **Step Differences**: Compare output hash of each step step by step
- **Cost Differences**: Differences in latency, step count, and other performance metrics
- **Cache Hit Rate**: LLM cache hit statistics

### 2.9 User Interface

#### 2.9.1 Command Line Interface

The CLI provides a complete command set for system operations:

- **run**: Run new task, supporting YAML spec file or command-line task description
- **replay**: Replay previous run
- **diff**: Compare differences between two runs
- **list**: List all runs
- **show**: Display detailed run information, including trace logs

The command-line interface supports rich parameter configuration such as seed, model selection, thinking level, corpus path, etc., providing high flexibility.

#### 2.9.2 Web Interface (Streamlit)

The Streamlit web interface provides graphical operation with support for:

- **Real-time Progress Display**: Shows currently executing steps and status
- **Run History Browsing**: View all historical runs
- **Interactive Replay**: Select and replay historical runs
- **Difference Comparison**: Side-by-side comparison of two runs
- **Trace Visualization**: View complete trace logs

---

## Challenges and Pain Points

### 3.1 Technical Challenges

#### 3.1.1 Determinism and Reproducibility

**Challenge Analysis**:

The inherent probabilistic nature of LLMs presents a fundamental challenge to reproducibility. Even with identical inputs and parameters, LLMs may produce different outputs. This is fatal in scientific computing scenarios, where reproducibility is the cornerstone of scientific research.

The concurrent characteristics of multi-agent systems add complexity. Message passing order between agents may vary due to system scheduling differences; even with identical logic, actual execution order may differ.

Random factors in tool execution (such as search, computation) also need to be controlled. numpy and random number generation, hash table iteration order, etc., can all introduce non-determinism.

**Solutions**:

The system adopts multi-level reproducibility guarantee mechanisms:

1. **Global Seed Management**: seed parameter persists throughout the run lifecycle, affecting numpy and random number generation
2. **LLM Response Caching**: Implements caching based on deterministic hashing of request payload, identical requests directly return cached results
3. **Deterministic Algorithms**: BM25 search algorithm ensures identical queries return identical ordering
4. **Output Hash Verification**: Records input_hash and output_hash for each step for diff comparison
5. **JSONL Append Writing**: Avoids ordering differences caused by memory accumulation

#### 3.1.2 State Consistency

**Challenge Analysis**:

Maintaining state consistency in multi-agent concurrent execution scenarios is a challenge. BDI state update timing, message passing reliability, and concurrent conflict resolution all require careful design.

**Solutions**:

The system adopts a sequential rather than concurrent execution model, avoiding concurrent state update issues. All state changes are recorded to JSONL files, with each TraceEntry containing complete snapshots rather than incremental updates, ensuring accurate state restoration during subsequent auditing and replay.

#### 3.1.3 Evidence Chain Integrity

**Challenge Analysis**:

Ensuring evidence chain integrity requires solving the following problems:

1. Each conclusion must have a traceable source
2. Evidence fragments must correspond precisely to original documents
3. Content hash verification must be correct
4. Evidence anchors cannot be tampered with

**Solutions**:

The system designs a strict evidence anchor structure where each anchor must contain document ID, document title, precise location, content hash, and content fragment. Hashes are calculated using MD5 or SHA algorithms, ensuring any content tampering can be detected. Relevance score filtering ensures low-quality evidence is excluded.

### 3.2 Design Challenges

#### 3.2.1 Balancing Complexity and Understandability

**Pain Point Analysis**:

The combination of BDI + Contract Net + ACL brings high system complexity. Users (especially non-technical users) struggle to understand interactions between agents, making debugging and problem location difficult.

**Solutions**:

The system reduces complexity through various means:

1. **Detailed Trace Logs**: Each step has clear log records
2. **Streamlit UI Visualization**: Graphically displays execution process
3. **Pre-generated Example Runs**: Users can quickly view demonstrations
4. **Layered Architecture**: Each layer has clear responsibilities, facilitating understanding and modification
5. **Progressive Modes**: owl_lite/owl_dl/owl_full gradually increase complexity

#### 3.2.2 Performance and Resource Consumption

**Pain Point Analysis**:

Complete tracing generates large amounts of log data, potentially affecting performance. LLM calls can be slow and expensive. Replay mechanisms require storing all intermediate states, occupying significant storage space.

**Solutions**:

1. **JSONL Incremental Writing**: Avoids memory accumulation
2. **LLM Response Caching**: Reduces duplicate calls while improving reproducibility
3. **Mode Grading**: Users can choose appropriate modes based on needs
4. **Artifact Separation**: Large artifacts (such as figures) are stored separately in artifacts directory

#### 3.2.3 LLM Integration Uncertainty

**Pain Point Analysis**:

LLM APIs may rate limit or fail, schema validation may require retries, and different models have significantly different behaviors—all increasing system unpredictability.

**Solutions**:

1. **Automatic Degradation Strategy**: Automatically degrades from Pro model to Flash model
2. **Schema Retry Mechanism**: Constructs repair prompt and retries when validation fails
3. **Graceful Degradation**: Falls back to rule-based planning when LLM unavailable
4. **Rich Error Handling**: Detailed error messages and degradation prompts

### 3.3 Engineering Challenges

#### 3.3.1 Error Handling and Degradation Strategies

**Challenge Analysis**:

The system needs to handle various error scenarios: LLM unavailability, tool execution failures, network errors, schema validation errors, etc. Partial function failures should not cause entire system collapse.

**Solutions**:

The system adopts a "finalize-first" invariant: final confirmation is a hard threshold, while additional functions like argumentation graph generation are best-effort. The system defines rich status codes: completed, completed_with_warnings, failed, waiting, each with clear semantics.

#### 3.3.2 Testing and Validation

**Challenge Analysis**:

Unit testing of multi-agent systems is complex, LLM response uncertainty makes testing difficult, and integration testing requires a complete environment.

**Solutions**:

1. **Pre-generated Example Runs**: Provides demonstration-ready complete runs
2. **Fixed Seed Testing**: Uses fixed seed for deterministic testing
3. **Modular Design**: Each component can be tested independently
4. **Mock Mechanism**: Can mock LLM responses for testing

---

## Learning Outcomes and Growth

### 4.1 Deepening Theoretical Knowledge

#### 4.1.1 Multi-Agent System Theory

**Understanding BDI Architecture**:

The BDI architecture provides an elegant way to model agent decision-making processes. Beliefs represent the agent's knowledge of the world, Desires represent the agent's goals, and Intentions represent the plans the agent commits to executing. This three-layer cognitive structure makes agent behavior more understandable and predictable.

Through practical implementation of BDI architecture, I understood how to transform abstract cognitive science theories into executable code. The state tracking mechanism makes the agent's decision-making process completely transparent, which is more interpretable than traditional "black box" AI systems.

**Design Philosophy of Negotiation Protocols**:

The Contract Net Protocol demonstrates how to design a fair and efficient task allocation mechanism. By introducing bidding, evaluation, and contract award processes, the system can automatically select the most suitable worker to complete specific tasks. This design has universal applicability in multi-agent systems.

**Application of Speech Act Theory**:

The ACL message protocol demonstrates how to apply speech act theory to practical system design. Each speech act has clear semantics and usage rules, making communication between agents more standardized and understandable. Speech acts like INFORM, REQUEST, and CHALLENGE not only convey information but also specify the nature of interaction and expected responses.

#### 4.1.2 Distributed System Design

**Maintaining State Consistency**:

Maintaining state consistency in distributed environments is a complex problem. The RAR system avoids concurrency complexity through sequential execution and complete snapshot recording. This taught me that in certain scenarios, sacrificing some concurrent performance for correctness is worthwhile.

**Message Passing Reliability**:

The system's message bus design ensures reliable message delivery. All messages are recorded to trace logs, meaning messages won't be lost and can be replayed and audited at any time. This is crucial for building trustworthy systems.

**Observability Design Principles**:

The tracing layer design demonstrates how to build a highly observable system. By recording all relevant information for each event, the system can achieve complete auditing and reproduction. This taught me that observability should be considered from the beginning when designing systems, not added afterward.

### 4.2 Engineering Practice Skills

#### 4.2.1 Python Advanced Features

**Application of Data Classes**:

The project extensively uses dataclass to define data structures, demonstrating how to define immutable data structures declaratively. dataclass automatically generates `__init__`, `__repr__`, `__eq__`, and other methods, significantly reducing boilerplate code.

**Enum Type State Management**:

Using enum types to define states (such as AgentState, TraceEventType, Performative) makes state types safer, avoiding spelling errors that string constants might bring.

**Abstract Base Class Interface Definition**:

BaseAgent uses abstract base classes to define interfaces, forcing subclasses to implement the process method. This demonstrates how to use Python's type system to define and enforce interface contracts.

#### 4.2.2 System Architecture Design

**Advantages of Layered Architecture**:

The RAR project's six-layer design (UI/Orchestrator/Agent/Protocol/Tool/Tracer) demonstrates the power of layered architecture. Each layer has clear responsibilities, and layers communicate through well-defined interfaces. This design makes the system easy to understand, test, and modify.

**Use of Dependency Injection**:

Injecting tracer and tools through constructors reduces coupling between components, facilitating unit testing and module replacement.

**Application of Strategy Pattern**:

Different protocols and tools can be flexibly replaced, demonstrating the strategy pattern design concept. For example, LocalSearchTool can be replaced with other search implementations as long as the same interface is maintained.

#### 4.2.3 Data Persistence

**Choice of JSONL Format**:

The project uses JSONL (JSON Lines) format to store trace logs, demonstrating a storage strategy for streaming data. JSONL format supports incremental writing, is suitable for large data scenarios, and maintains readability.

**Content-Addressed Storage**:

The design of referencing content through hash values supports deduplication and verification. If two contents are identical, their hashes are also identical, so only one copy needs to be stored. Meanwhile, hash values can verify content integrity.

**Metadata Separation**:

Separating storage of metadata, trace logs, and final results makes each file's responsibilities clear, facilitating individual processing and querying.

### 4.3 AI/LLM Engineering Practice

#### 4.3.1 LLM Caching Strategy

**Hash-Based Input Caching**:

By calculating the deterministic hash of request payload as the cache key, an efficient caching mechanism is implemented. The advantages of this design include:
- Identical requests always hit cache
- Cache hits improve response speed
- Reduces API call costs
- Ensures output reproducibility

**Multi-level Cache Architecture**:

The system supports run-level cache, global cache, and read-only cache paths. This multi-level architecture is particularly useful in reproduction scenarios—can directly use the original run's cache, avoiding duplicate LLM calls.

#### 4.3.2 Schema Validation and Retry

**Enforcing Structured Output**:

Through Pydantic schema and Gemini's response_json_schema parameter, LLM output is forced to comply with expected structures. This avoids the complexity of manually parsing LLM output and improves system reliability.

**Automatic Retry Mechanism**:

When validation fails, the system constructs repair prompts and retries. This design is particularly useful when LLM output is unstable, significantly improving system robustness.

**Degradation Strategy**:

When advanced models fail, automatically degrade to base models. This design ensures the system can still provide services under poor network conditions or API rate limiting.

#### 4.3.3 Function Calling Design

Function calling is a key technology for decomposing complex tasks into callable units. By defining clear tool schemas, LLMs can autonomously decide how to complete tasks. This is more flexible and powerful than the traditional "single prompt + single response" mode.

Each tool has clear input/output schemas and supports evidence anchors for tool results. This design makes the LLM's decision-making process more transparent and auditable.

### 4.4 Software Engineering Thinking

#### 4.4.1 Reproducibility Engineering

**Determinism-First Principle**:

The project demonstrates how to build a deterministic system: all random sources should be configurable with seed, all inputs and outputs should be recorded, and model versions and parameters should be tracked. These principles are particularly important in scientific computing scenarios.

**Complete Tracing Design**:

Recording all inputs, outputs, and intermediate states is the foundation for achieving reproducibility. The RAR system's tracing layer design provides a complete example.

**Version Control Practices**:

Recording model version, parameter configuration, dependency version, and other information enables results to be reproduced across different environments.

#### 4.4.2 Progressive Enhancement

**Three-Tier Mode Design**:

The owl_lite/owl_dl/owl_full design demonstrates how to achieve progressive enhancement. Users can choose appropriate modes based on their needs, gradually increasing function depth and strictness from simple exploration to complete argumentation chains.

**Optional Feature Design**:

Additional features like argumentation graph generation are designed as best-effort, where failure doesn't affect the main flow. This design philosophy is worth learning from: core functions must be reliable, while additional features can be added progressively.

#### 4.4.3 Error Handling Philosophy

**Graceful Degradation**:

The system's graceful degradation strategy demonstrates how to handle partial function failures. When LLM is unavailable, fall back to rule-based planning; when advanced models fail, degrade to base models—these strategies ensure system robustness.

**Clear State Semantics**:

Each state has clear definitions and transition conditions, making system behavior predictable and facilitating debugging and problem location.

**Rich Diagnostic Information**:

Error messages contain sufficient context, including failure reasons, suggested repair solutions, etc., significantly reducing debugging difficulty.

### 4.5 Interdisciplinary Knowledge Integration

| Domain | Knowledge Gained |
|--------|------------------|
| Cognitive Science | BDI architecture transforms rational agent models from cognitive science into executable code |
| Philosophy of Language | Speech act theory provides a formal semantic framework for agent communication |
| Game Theory | Contract Net protocol demonstrates auction mechanism application in task allocation |
| Evidence Theory | Evidence anchors and hash verification design reflect the rigor of evidence theory |
| Philosophy of Science | Reproducibility and falsification are core principles of scientific research |

### 4.6 Project Management and Collaboration

#### 4.6.1 Documentation-Driven Development

The project demonstrates how to improve code quality through documentation-driven development:

- **DESIGN.md**: Detailed architecture design documentation guides development
- **README.md**: Clear usage guides reduce learning curve
- **Code Comments**: Each module has detailed docstrings
- **Type Annotations**: Python type hints improve code readability

#### 4.6.2 Version Control Practices

- **Reasonable .gitignore Configuration**: Excludes unnecessary files
- **Clear Commit Messages**: Each commit has a clear purpose
- **Pre-generated runs/**: Provides demonstration data, lowering usage barrier

---

## Conclusion

The RAR project is a complex system **combining theory and practice**, which:

1. **Solves Real Problems**: The reproducibility crisis in the AI4S domain
2. **Integrates Multiple Theories**: BDI, Contract Net, ACL, Dung argumentation framework
3. **Excellent Engineering Implementation**: Clear architecture, complete tracing, flexible configuration
4. **High Learning Value**: Covers multi-agent systems, LLM engineering, distributed systems, and more

Through this project, one can deeply understand **how to build a trustworthy, reproducible, and auditable AI system**—which is the core capability needed for the future AI for Science domain.

The project's innovations lie in:

1. **Completeness**: From BDI architecture to Contract Net protocol, from evidence chains to argumentation graphs, forming a complete auditable AI system
2. **Reproducibility**: Multiple mechanisms including seed fixing, LLM caching, and deterministic algorithms ensure reproducible results
3. **Transparency**: Complete trace records make every decision step traceable and auditable
4. **Flexibility**: Three-tier mode design allows users to balance cost and reliability

This project provides a feasible solution to the reproducibility problem in the AI for Science domain, holding significant academic value and practical meaning.
