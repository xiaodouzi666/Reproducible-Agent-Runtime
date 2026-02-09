# 3-Minute Demo Script (Devpost)

## 0:00 - 0:20 | Framing

- Open Streamlit app (public link).
- One-line pitch:
  `RAR-Gemini is a reproducible Marathon Agent Runtime for AI-for-Science: plan, tool-call, audit, self-correct, replay, and diff.`
- Show Run / Replay / Diff / History tabs.

## 0:20 - 1:30 | OWL DL Run (Finalize + Evidence)

- Run task in `owl_dl` with Gemini enabled.
- While running, show Trace Timeline:
  - LLM call/result entries
  - function calls
  - model + thinking level
- After completion, show:
  - Final answer section
  - Evidence chain with anchors/citations
  - Audit notes (approved or challenged)
- State clearly: finalize was called and result is structured.

## 1:30 - 2:10 | Replay (Reproducibility)

- Replay the just-finished run.
- Show:
  - replay completion
  - cache hit rate
  - answer comparison / match status
- Explain: replay can reuse cached LLM responses for reproducibility.

## 2:10 - 3:00 | OWL Full + Argument Graph (Best-effort Non-blocking)

- Run same (or similar) task in `owl_full`.
- Show `🧩 Argument Graph` tab:
  - if graph renders: highlight claim-support-attack structure.
  - if graph fails: show warning and `completed_with_warnings`.
- Key message:
  - Finalize-first invariant: optional modules are best-effort.
  - Even with optional-module failure, finalize is preserved and run remains auditable.

## Closing line

- “This is not baseline RAG. Gemini 3 drives orchestration via function calling, with reproducible traces, replay, and diff.”
