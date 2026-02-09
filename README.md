# RAR - Reproducible Agent Runtime

![RAR Logo](logo.png)

A traceable, replayable, and comparable multi-agent execution and audit framework for AI4S (AI for Science) applications.

## Quick Start (1 minute)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit UI
streamlit run app.py

# Or use the CLI
python run.py run "What methods are used to calculate activation energy?"
```

## Features

- **Multi-Agent Collaboration**: 4 specialized agents (Planner, Researcher, Executor, Auditor) with BDI architecture
- **Contract Net Protocol**: Visible task allocation with bidding and contract awarding
- **ACL/Speech Acts**: Messages with performatives (inform, request, propose, challenge, etc.)
- **Full Traceability**: JSONL-based trace recording for every execution step
- **Evidence Chain**: Every conclusion linked to source documents with citations
- **Replay**: Reproduce any previous run with `--replay` flag
- **Diff Comparison**: Compare two runs to analyze differences

## Public Demo

- **Public Project Link:** [YouTube Demo](https://www.youtube.com/watch?v=BDPJ2wSYzlM)
- The demo link should be publicly accessible without login or paywall.
- Recommended default demo path: run -> trace -> evidence -> replay/diff.

## Completion Semantics

- `completed`: finalize success and no major warnings.
- `completed_with_warnings`: finalize success but outstanding challenges OR optional modules failed (for example, argument graph best-effort generation).
- `failed`: finalize missing (for example, repeated schema/LLM failures).
- `waiting`: planner requested `wait_seconds`/`wait_until`; checkpoint is saved and the run can be resumed.

### Finalize-first invariant

`structured_finalize` is the hard completion gate. Optional modules are best-effort and must not block finalize.  
Examples: argument graph generation, checkpoint write/read, and other enrichments can emit warnings, but they cannot flip a finalized run to `failed`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Orchestrator                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Planner  │  │ Researcher │  │ Executor │  │  Auditor  │ │
│  │  (BDI)   │──│  (Search)  │──│ (Python) │──│ (Verify)  │ │
│  └──────────┘  └────────────┘  └──────────┘  └───────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌─────────────────────────────────┐ │
│  │  Contract Net     │  │        ACL Messages             │ │
│  │  (Task Allocation)│  │  (CFP/BID/AWARD/INFORM/...)     │ │
│  └───────────────────┘  └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ LocalSearch  │  │ PythonExec   │  │     Tracer        │ │
│  │   (BM25)     │  │  (Sandbox)   │  │   (JSONL Store)   │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### CLI Commands

```bash
# Run a new task
python run.py run "Your scientific question"
python run.py run --spec demo_data/tasks/task1.yaml --seed 42

# Replay a previous run
python run.py replay <run_id>
python run.py resume <run_id>   # Resume a waiting run from checkpoint

# Compare two runs
python run.py diff <run_id_a> <run_id_b> --save

# List all runs
python run.py list

# Show run details
python run.py show <run_id> --trace
```

### Example Questions (Related to Demo Corpus)

The demo corpus contains papers about thermal analysis, polymer degradation, and machine learning in materials science. **Recommended to try these queries:**

```bash
# Thermal analysis
python run.py run "What is the thermal degradation behavior of ABS polymer?" --seed 42

# Activation energy
python run.py run "How to calculate activation energy?" --seed 42

# Polymer pyrolysis
python run.py run "What happens during pyrolysis of polymer blends?" --seed 42

# Machine learning + materials
python run.py run "How is machine learning used in materials science?" --seed 42
```

### Streamlit UI

```bash
streamlit run app.py
```

The UI provides:
- **Run Task**: Execute new tasks with real-time progress
- **Replay**: Re-run previous executions
- **Diff**: Compare two runs side-by-side
- **History**: Browse all past runs and resume waiting runs

## Devpost Materials

- Gemini integration brief write-up: `docs/devpost_gemini_integration_200words.md`
- Testing checklist (live run, replay, OWL Full best-effort graph, WAIT/resume): `docs/testing.md`
- 3-minute demo flow script: `docs/demo_script.md`

## Project Structure

```
RAR-Demo/
├── README.md                 # This file
├── DESIGN.md                 # Architecture and design document
├── requirements.txt          # Python dependencies
├── run.py                    # CLI interface
├── app.py                    # Streamlit UI
├── rar/
│   ├── agents/               # Agent implementations
│   │   ├── base.py           # BaseAgent with BDI
│   │   ├── planner.py        # Task decomposition
│   │   ├── researcher.py     # Information retrieval
│   │   ├── executor.py       # Code execution
│   │   └── auditor.py        # Quality assurance
│   ├── protocols/            # Communication protocols
│   │   ├── acl.py            # ACL messages & speech acts
│   │   └── contract_net.py   # Contract Net Protocol
│   ├── tools/                # Tool implementations
│   │   ├── local_search.py   # BM25 search
│   │   └── python_exec.py    # Safe Python execution
│   ├── tracing/              # Trace infrastructure
│   │   ├── schema.py         # TraceEntry, EvidenceAnchor
│   │   ├── tracer.py         # Tracing interface
│   │   └── store.py          # JSONL storage
│   ├── replay/               # Replay engine
│   └── diff/                 # Diff comparison
├── demo_data/
│   ├── corpus/               # Sample papers (5-10 txt/md)
│   └── tasks/                # Task specifications (YAML)
└── runs/
    └── sample_run_1/         # Pre-generated sample run
```

## Key Concepts

### 1. BDI Architecture
Each agent maintains:
- **Beliefs**: Current knowledge about the world
- **Desires**: Goals to achieve
- **Intentions**: Committed plans

### 2. Contract Net Protocol
Task allocation flow:
1. Planner issues CFP (Call for Proposals)
2. Workers submit bids (cost, latency, success probability)
3. Planner awards contract to best bidder
4. Worker executes and reports result

### 3. ACL Speech Acts
Message types include:
- `INFORM`: Share information
- `REQUEST`: Ask for action
- `PROPOSE/ACCEPT/REJECT`: Negotiation
- `CHALLENGE`: Audit/quality check
- `CFP/BID/AWARD`: Contract Net

### 4. Evidence Chain
Every conclusion is linked to:
- Document ID and title
- Location (paragraph/line number)
- Content hash for verification
- Relevance score

## Demo Scenarios

1. **Activation Energy Analysis** (task1.yaml)
   - Research + Computation workflow
   - Searches literature, performs Arrhenius calculation

2. **Thermal Stability Comparison** (task2.yaml)
   - Multi-source research workflow
   - Cross-references multiple papers

3. **ML in Materials Science** (task3.yaml)
   - Research-only workflow
   - Explores ML methods and applications

## Pre-generated Results

The `runs/sample_run_1/` directory contains a pre-generated trace that can be viewed even without running the full system. This ensures the demo works on any machine.

## Requirements

- Python 3.9+
- See requirements.txt for dependencies

## MVP Scope

The current demo delivers an MVP closed loop (RunSpec/Trace/Replay/Diff + auditable collaboration), with room to extend Artifact Store (hot/cold tiering), dashboards, and Dung-style argument adjudication. See Section 8 in [DESIGN.md](DESIGN.md).

## License

MIT License
