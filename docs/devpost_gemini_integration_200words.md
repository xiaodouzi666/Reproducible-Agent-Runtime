# Gemini 3 Integration (Devpost Brief Write-up)

RAR-Gemini is built as a runtime orchestrator, not a prompt wrapper. Gemini 3 is the central planner that drives a multi-step function-calling loop (`delegate_research`, `delegate_execute`, `delegate_audit`, `finalize`) and decides what to do next at each step. This enables explicit tool orchestration, evidence collection, and audit-driven self-correction.

The system maps OWL modes to Gemini reasoning profiles: OWL Lite uses fast/low-cost settings (Flash + minimal thinking), while OWL DL/Full use stricter settings (Pro + high thinking). This mode-to-thinking binding is reflected in run specs, metadata, traces, and diff reports, so reviewers can see exactly how configuration changes affect behavior.

Each LLM interaction records request/config/response hashes and function-call chains for replayability and diffability. Thought-signature continuity is preserved through history-based tool turns, while trace logs expose model, thinking level, function calls, cache hits, and response hashes. Replay can reuse cached LLM responses, improving reproducibility even when live API access is unavailable.

Optional enhancements (for example argument graph generation in OWL Full) are best-effort and never block finalize. Finalize-first semantics ensure the system can produce a structured, auditable answer with explicit uncertainty and warnings rather than failing silently.

Devpost submission note: include a public demo link (no login required) and this brief Gemini integration summary.
