#!/usr/bin/env python3
"""
RAR - Reproducible Agent Runtime
Streamlit UI

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import hashlib
import math
from pathlib import Path
from datetime import datetime
from typing import Optional
import plotly.graph_objects as go

from rar.config import get_mode_config, resolve_mode_config
from rar.orchestrator import Orchestrator
from rar.replay import ReplayEngine
from rar.diff import DiffEngine
from rar.tracing import TraceStore, TraceEventType


# Page config
st.set_page_config(
    page_title="RAR - Reproducible Agent Runtime",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "history_selected_run_id" not in st.session_state:
    st.session_state.history_selected_run_id = None


# Initialize components
@st.cache_resource
def get_store():
    return TraceStore("runs")


@st.cache_resource
def get_orchestrator():
    return Orchestrator(
        corpus_dir="demo_data/corpus",
        output_dir="runs"
    )


def _response_raw_from_entry(entry) -> dict:
    """Best-effort extraction of response_raw from a trace entry."""
    if not isinstance(getattr(entry, "content", None), dict):
        return {}
    raw = entry.content.get("response_raw")
    return raw if isinstance(raw, dict) else {}


def _extract_function_call_names(entry) -> str:
    """Extract function call names from LLM trace entry."""
    names = []

    def _append_name(value: str):
        name = str(value or "").strip()
        if name and name not in names:
            names.append(name)

    llm_meta = entry.llm if isinstance(entry.llm, dict) else {}
    fc_meta = llm_meta.get("function_calls")
    if isinstance(fc_meta, list):
        for call in fc_meta:
            if isinstance(call, dict):
                _append_name(call.get("name", ""))

    raw = _response_raw_from_entry(entry)
    calls = raw.get("function_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                _append_name(call.get("name", ""))

    candidates = raw.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                fc = part.get("function_call") or part.get("functionCall")
                if isinstance(fc, dict):
                    _append_name(fc.get("name", ""))

    summary = entry.content_summary or ""
    if summary.startswith("Function calls:"):
        suffix = summary.replace("Function calls:", "", 1).strip()
        for value in suffix.split(","):
            _append_name(value)

    return ", ".join(names) if names else "-"


def _iter_thought_signatures(data):
    """Yield thought-signature values from arbitrary nested JSON."""
    if isinstance(data, dict):
        for key, value in data.items():
            lower_key = str(key).lower()
            if lower_key in {"thought_signature", "thoughtsignature"} or (
                "thought" in lower_key and "signature" in lower_key
            ):
                yield value
            yield from _iter_thought_signatures(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_thought_signatures(item)


def _signature_hash(entry) -> str:
    """Return safe short hash of thought signature, never raw value."""
    llm_meta = entry.llm if isinstance(entry.llm, dict) else {}
    existing = llm_meta.get("thought_signature_hash") or llm_meta.get("signature_hash")
    if existing:
        return str(existing)[:8]

    content_llm = {}
    if isinstance(getattr(entry, "content", None), dict):
        raw_content_llm = entry.content.get("llm")
        if isinstance(raw_content_llm, dict):
            content_llm = raw_content_llm

    candidates = []
    candidates.extend(list(_iter_thought_signatures(_response_raw_from_entry(entry))))
    candidates.extend(list(_iter_thought_signatures(content_llm)))
    if not candidates:
        return "-"

    signature_payload = candidates[0] if len(candidates) == 1 else candidates
    try:
        normalized = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        normalized = str(signature_payload)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]


def _usage_tokens(entry) -> str:
    """Best-effort extraction of total usage tokens from llm metadata."""

    def _to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _extract_total(usage: dict) -> Optional[int]:
        if not isinstance(usage, dict):
            return None
        for key in ("total_token_count", "total_tokens", "totalTokenCount"):
            total = _to_int(usage.get(key))
            if total is not None:
                return total
        summed = 0
        for key in (
            "prompt_token_count",
            "input_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "output_token_count",
            "prompt_tokens",
            "completion_tokens",
        ):
            value = _to_int(usage.get(key))
            if value is not None:
                summed += value
        return summed if summed > 0 else None

    llm_meta = entry.llm if isinstance(entry.llm, dict) else {}
    usage = llm_meta.get("usage")
    total = _extract_total(usage if isinstance(usage, dict) else {})
    if total is not None:
        return str(total)

    raw = _response_raw_from_entry(entry)
    total = _extract_total(raw.get("usage_metadata") if isinstance(raw, dict) else {})
    if total is None and isinstance(raw, dict):
        total = _extract_total(raw.get("usageMetadata"))
    return str(total) if total is not None else "-"


def _cache_hit_value(entry):
    """Return cache-hit state for an entry: True/False/None."""
    llm_meta = entry.llm if isinstance(entry.llm, dict) else {}
    if "is_cache_hit" in llm_meta:
        return bool(llm_meta.get("is_cache_hit"))
    if entry.event_type == TraceEventType.LLM_CACHE_HIT:
        return True
    if entry.event_type in {TraceEventType.LLM_CALL, TraceEventType.LLM_RESULT}:
        return False
    return None


def _load_run_result_for_display(run_id: str, store: TraceStore) -> Optional[dict]:
    """Load a run summary payload that can be rendered by render_run_result."""
    metadata = store.get_metadata(run_id)
    if not metadata:
        return None

    answer = metadata.final_answer or ""
    evidence = []

    final_path = Path("runs") / run_id / "final.json"
    if final_path.exists():
        try:
            final_data = json.loads(final_path.read_text(encoding="utf-8"))
            answer = str(final_data.get("answer", answer) or "")
            raw_evidence = final_data.get("evidence_anchors", [])
            if isinstance(raw_evidence, list):
                evidence = raw_evidence
        except Exception:
            pass

    return {
        "success": metadata.status != "failed",
        "run_id": run_id,
        "final_answer": answer,
        "evidence": evidence,
    }


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    """Parse ISO datetime string to datetime object (best-effort)."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("🔬 RAR Demo")
    st.sidebar.markdown("*Reproducible Agent Runtime*")
    st.sidebar.divider()

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🚀 Run Task", "🔁 Replay", "📊 Compare (Diff)", "📋 Run History"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Quick stats
    store = get_store()
    runs = store.list_runs()
    st.sidebar.metric("Total Runs", len(runs))

    return page


def render_run_page():
    """Render the task execution page."""
    st.header("🚀 Run New Task")

    if "run_task_input" not in st.session_state:
        st.session_state.run_task_input = ""
    if "run_task_input_pending" in st.session_state:
        st.session_state.run_task_input = str(st.session_state.get("run_task_input_pending", "") or "")
        del st.session_state["run_task_input_pending"]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Task input
        st.text_area(
            "Task Description",
            placeholder="Enter your scientific question or task...\n\nExample: What is the activation energy for thermal decomposition of polymers?",
            height=100,
            key="run_task_input",
        )
        task = str(st.session_state.get("run_task_input", "") or "")

        # Example tasks
        st.markdown("**Quick examples:**")
        example_cols = st.columns(3)
        examples = [
            "What methods are used to calculate activation energy?",
            "Compare thermal decomposition mechanisms in different materials",
            "What factors affect polymer degradation temperature?"
        ]
        for i, (col, example) in enumerate(zip(example_cols, examples)):
            if col.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
                st.session_state.run_task_input_pending = example
                st.rerun()

    with col2:
        st.markdown("**Options**")
        enable_gemini = st.checkbox("Enable Gemini 3 Orchestrator", value=True)
        mode = st.selectbox(
            "Reasoning Mode",
            options=["owl_lite", "owl_dl", "owl_full"],
            format_func=lambda m: get_mode_config(m).label,
            index=0,
        )
        mode_cfg = get_mode_config(mode)
        st.caption(f"Model: `{mode_cfg.model}`")
        st.caption(f"Thinking: `{mode_cfg.thinking_level}`")
        st.caption(f"Evidence rules: {mode_cfg.evidence_rules}")
        st.caption(f"Audit level: `{mode_cfg.audit_level}` | Argumentation: `{mode_cfg.argumentation}`")

        advanced_override = st.checkbox("Advanced LLM override", value=False)
        model_override = ""
        thinking_override = ""
        if advanced_override:
            model_override = st.text_input("Model override", value=mode_cfg.model)
            thinking_override = st.text_input("Thinking override", value=mode_cfg.thinking_level)

        resolved_mode = resolve_mode_config(
            mode=mode,
            model_override=model_override or None,
            thinking_override=thinking_override or None,
        )

        seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
        verbose = st.checkbox("Show detailed trace", value=True)

    # Run button
    if st.button("▶️ Run Task", type="primary", disabled=not task.strip()):
        with st.spinner("Running multi-agent workflow..."):
            orchestrator = Orchestrator(
                corpus_dir="demo_data/corpus",
                output_dir="runs",
                seed=seed,
                llm_mode=enable_gemini,
                owl_mode=resolved_mode.mode,
                llm_provider="gemini" if enable_gemini else "",
                llm_model=resolved_mode.model,
                llm_thinking_level=resolved_mode.thinking_level,
            )
            result = orchestrator.run(task)

            st.session_state.current_run_id = result.get("run_id")
            st.session_state.run_history.append(result.get("run_id"))

        # Show result
        render_run_result(result, verbose)


def render_run_result(result: dict, verbose: bool = True):
    """Render the result of a run."""
    success = result.get("success", False)
    run_id = result.get("run_id")
    metadata = get_store().get_metadata(run_id) if run_id else None

    # Status header
    status_value = getattr(metadata, "status", "") if metadata else ""
    if status_value == "waiting":
        st.info(f"⏸️ Run is waiting for resume (ID: {result.get('run_id', 'N/A')})")
    elif status_value == "completed_with_warnings":
        st.warning(f"⚠️ Run completed with warnings (ID: {result.get('run_id', 'N/A')})")
    elif success:
        st.success(f"✅ Run completed successfully! (ID: {result.get('run_id', 'N/A')})")
    else:
        st.error(f"❌ Run failed: {result.get('error', 'Unknown error')}")

    if metadata:
        st.caption(
            f"Mode: `{metadata.owl_mode or 'N/A'}` | "
            f"Model: `{metadata.llm_model or 'N/A'}` | "
            f"Thinking: `{metadata.llm_thinking_level or 'N/A'}` | "
            f"LLM enabled: `{metadata.llm_mode}`"
        )
        if getattr(metadata, "next_run_at", ""):
            st.caption(f"Next run at: `{metadata.next_run_at}`")
        if getattr(metadata, "finalize_missing", False):
            st.warning("LLM did not call finalize; answer used fallback synthesis.")

    # Three-panel layout
    is_full_mode = bool(metadata and metadata.owl_mode == "owl_full")
    if is_full_mode:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📝 Final Answer", "🔍 Trace Timeline", "📊 Evidence Chain", "🧩 Argument Graph"]
        )
    else:
        tab1, tab2, tab3 = st.tabs(["📝 Final Answer", "🔍 Trace Timeline", "📊 Evidence Chain"])

    with tab1:
        st.markdown("### Final Answer")
        answer = result.get("final_answer", "No answer generated")
        st.markdown(answer)

        # Evidence citations
        evidence = result.get("evidence", [])
        if evidence:
            st.markdown("---")
            st.markdown("### Evidence Sources")
            for i, e in enumerate(evidence, 1):
                if isinstance(e, dict):
                    with st.expander(f"[{i}] {e.get('doc_title', e.get('doc_id', 'Unknown'))}"):
                        st.markdown(f"**Location:** {e.get('location', 'N/A')}")
                        st.markdown(f"**Relevance:** {e.get('relevance_score', 0):.2f}")
                        st.markdown("**Snippet:**")
                        st.text(e.get("snippet", "")[:500])

    with tab2:
        render_trace_timeline(result.get("run_id"))

    with tab3:
        render_evidence_chain(result)

    if is_full_mode:
        with tab4:
            render_argument_graph(run_id)


def render_argument_graph(run_id: str):
    """Render OWL Full argument graph."""
    if not run_id:
        st.info("No run selected")
        return

    run_dir = Path("runs") / run_id
    final_path = run_dir / "final.json"
    final_data = {}
    if final_path.exists():
        try:
            final_data = json.loads(final_path.read_text(encoding="utf-8"))
        except Exception:
            final_data = {}

    graph_ref = str(final_data.get("argument_graph_ref", "") or "")
    graph_status = str(final_data.get("argument_graph_status", "skipped") or "skipped")
    warnings = final_data.get("warnings", []) if isinstance(final_data.get("warnings"), list) else []
    graph_warnings = [w for w in warnings if isinstance(w, str) and w.startswith("argument_graph_failed")]

    st.caption(f"Graph status: `{graph_status}`")

    graph_path = run_dir / graph_ref if graph_ref else (run_dir / "argument_graph.json")
    if not graph_ref or not graph_path.exists():
        if graph_status == "failed":
            st.warning("Graph unavailable (best-effort). Finalize output is still valid.")
        elif graph_status == "skipped":
            st.info("Argument graph skipped for this run.")
        else:
            st.info("Argument graph not found.")
        if graph_warnings:
            st.markdown("**Warnings**")
            for warning in graph_warnings:
                st.markdown(f"- {warning}")
        return

    try:
        raw = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning("Graph unavailable (best-effort). Finalize output is still valid.")
        st.markdown(f"- argument_graph_failed: {e}")
        return

    data = _normalize_argument_graph_for_ui(raw)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    accepted = data.get("accepted_claim_ids", [])

    st.markdown(f"**Nodes:** {len(nodes)} | **Edges:** {len(edges)}")
    if accepted:
        st.markdown(f"**Accepted claims:** {', '.join(accepted)}")

    _render_argument_graph_plot(data)

    with st.expander("Graph JSON"):
        st.json(data)


def _normalize_argument_graph_for_ui(raw: dict) -> dict:
    """Normalize graph payload for UI rendering."""
    nodes_out = []
    edges_out = []

    nodes = raw.get("nodes", [])
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", node.get("node_id", "")) or "").strip()
            node_type = str(node.get("type", node.get("node_type", "")) or "").strip().lower()
            node_type = {
                "counter_claim": "counterclaim",
                "counterclaim": "counterclaim",
                "counter-claim": "counterclaim",
                "claim": "claim",
                "evidence": "evidence",
            }.get(node_type, node_type)
            text = str(node.get("text", "") or "").strip()
            if not node_id or not text:
                continue
            if node_type not in {"claim", "evidence", "counterclaim"}:
                continue
            nodes_out.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "text": text,
                    "source_anchor_id": str(node.get("source_anchor_id", "") or ""),
                }
            )

    edges = raw.get("edges", [])
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "") or "").strip()
            target = str(edge.get("target", "") or "").strip()
            relation = str(edge.get("relation", "") or "").strip().lower()
            relation = {
                "support": "supports",
                "supports": "supports",
                "attack": "attacks",
                "attacks": "attacks",
            }.get(relation, relation)
            if source and target and relation in {"supports", "attacks"}:
                edges_out.append({"source": source, "target": target, "relation": relation})

    accepted = raw.get("accepted_claim_ids", raw.get("winner_claims", []))
    if not isinstance(accepted, list):
        accepted = []
    accepted = [str(v) for v in accepted]

    rationale = raw.get("rationale", {})
    if not isinstance(rationale, dict):
        rationale = {}
    rationale = {str(k): str(v) for k, v in rationale.items()}

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "accepted_claim_ids": accepted,
        "rationale": rationale,
    }


def _render_argument_graph_plot(data: dict):
    """Render a lightweight interactive argument graph."""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not nodes:
        st.info("No graph nodes to render.")
        return

    groups = {"claim": [], "counterclaim": [], "evidence": []}
    for node in nodes:
        node_type = node.get("type", "claim")
        if node_type not in groups:
            groups["claim"].append(node)
        else:
            groups[node_type].append(node)

    y_map = {"claim": 1.0, "counterclaim": 0.05, "evidence": -1.0}
    pos = {}
    for node_type, group in groups.items():
        n = len(group)
        if n == 0:
            continue
        for i, node in enumerate(group):
            if n == 1:
                x = 0.0
            else:
                x = -1.2 + (2.4 * i / (n - 1))
            # Slight deterministic jitter by hash for overlap reduction.
            jitter = (int(hashlib.sha256(node["id"].encode("utf-8")).hexdigest()[:2], 16) / 255.0 - 0.5) * 0.12
            pos[node["id"]] = (x + jitter, y_map[node_type])

    fig = go.Figure()

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in pos or target not in pos:
            continue
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        relation = edge.get("relation", "supports")
        color = "#2e7d32" if relation == "supports" else "#c62828"
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": 2, "color": color},
                hoverinfo="text",
                text=[relation, relation],
                showlegend=False,
            )
        )

    palette = {"claim": "#1565c0", "counterclaim": "#ef6c00", "evidence": "#6a1b9a"}
    for node_type, group in groups.items():
        if not group:
            continue
        x_vals = []
        y_vals = []
        labels = []
        for node in group:
            x, y = pos.get(node["id"], (0.0, 0.0))
            x_vals.append(x)
            y_vals.append(y)
            labels.append(f"{node['id']}: {node['text'][:140]}")
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker={"size": 16, "color": palette[node_type]},
                text=[node["id"] for node in group],
                textposition="top center",
                hovertext=labels,
                hoverinfo="text",
                name=node_type,
            )
        )

    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=True,
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trace_timeline(run_id: str):
    """Render the trace timeline visualization."""
    if not run_id:
        st.info("No trace available")
        return

    store = get_store()
    entries = store.get_entries(run_id)

    if not entries:
        st.info("No trace entries found")
        return

    st.markdown(f"### Execution Trace ({len(entries)} steps)")

    # Create timeline data
    timeline_data = []
    for entry in entries:
        llm_meta = entry.llm if isinstance(entry.llm, dict) else {}
        cache_hit = _cache_hit_value(entry)
        function_calls = _extract_function_call_names(entry)
        signature_hash = _signature_hash(entry)
        usage_tokens = _usage_tokens(entry)
        tool_value = entry.tool_name or ("gemini" if llm_meta else "-")
        timeline_data.append({
            "Step": entry.step_id,
            "Time": entry.timestamp,
            "Event": entry.event_type.value,
            "Agent": entry.agent_id,
            "Tool": tool_value,
            "LLM Model": llm_meta.get("model", "-") or "-",
            "Thinking": llm_meta.get("thinking_level", "-") or "-",
            "FunctionCall": function_calls,
            "CacheHit": "-" if cache_hit is None else str(cache_hit),
            "SignatureHash": signature_hash,
            "UsageTokens": usage_tokens,
            "Summary": entry.content_summary[:50] if entry.content_summary else "",
            "Latency (ms)": f"{entry.latency_ms:.1f}" if entry.latency_ms else "-"
        })

    # Display as table
    df = pd.DataFrame(timeline_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Event type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Event Distribution**")
        event_counts = {}
        for entry in entries:
            event_type = entry.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        event_df = pd.DataFrame([
            {"Event Type": k, "Count": v}
            for k, v in sorted(event_counts.items(), key=lambda x: -x[1])
        ])
        st.bar_chart(event_df.set_index("Event Type"))

    with col2:
        st.markdown("**Agent Activity**")
        agent_counts = {}
        for entry in entries:
            if entry.agent_id:
                agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1

        if agent_counts:
            agent_df = pd.DataFrame([
                {"Agent": k, "Actions": v}
                for k, v in sorted(agent_counts.items(), key=lambda x: -x[1])
            ])
            st.bar_chart(agent_df.set_index("Agent"))

    # Detailed view
    st.markdown("---")
    st.markdown("**Detailed Trace**")

    for entry in entries:
        event_icon = {
            TraceEventType.AGENT_START: "🟢",
            TraceEventType.AGENT_END: "🔴",
            TraceEventType.MESSAGE_SENT: "💬",
            TraceEventType.TOOL_CALL: "🔧",
            TraceEventType.TOOL_RESULT: "📤",
            TraceEventType.CFP_ISSUED: "📢",
            TraceEventType.BID_SUBMITTED: "🎯",
            TraceEventType.CONTRACT_AWARDED: "🏆",
            TraceEventType.CHALLENGE_RAISED: "⚠️",
            TraceEventType.BELIEF_UPDATE: "🧠",
            TraceEventType.INTENTION_FORM: "📋",
            TraceEventType.LLM_CALL: "🤖",
            TraceEventType.LLM_RESULT: "🧾",
            TraceEventType.LLM_CACHE_HIT: "♻️",
            TraceEventType.ARG_GRAPH_OK: "🧩",
            TraceEventType.ARG_GRAPH_FAILED: "⚠️",
        }.get(entry.event_type, "•")

        with st.expander(
            f"{event_icon} [{entry.step_id:03d}] {entry.event_type.value} - {entry.agent_id or 'system'}"
        ):
            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Performative:** {entry.performative or 'N/A'}")
            col2.markdown(f"**Tool:** {entry.tool_name or 'N/A'}")
            col3.markdown(f"**Latency:** {entry.latency_ms:.1f}ms" if entry.latency_ms else "**Latency:** N/A")

            if entry.content_summary:
                st.markdown(f"**Summary:** {entry.content_summary}")

            content_llm = {}
            if isinstance(entry.content, dict):
                raw_content_llm = entry.content.get("llm")
                if isinstance(raw_content_llm, dict):
                    content_llm = raw_content_llm

            has_llm_info = bool(entry.llm) or bool(content_llm)
            if has_llm_info:
                llm_meta = {}
                if isinstance(entry.llm, dict):
                    llm_meta.update(entry.llm)
                for key, value in content_llm.items():
                    if key not in llm_meta:
                        llm_meta[key] = value

                function_calls = _extract_function_call_names(entry)
                signature_hash = _signature_hash(entry)
                usage_tokens = _usage_tokens(entry)
                cache_hit = _cache_hit_value(entry)

                st.markdown("**LLM Metadata:**")
                lcol1, lcol2, lcol3 = st.columns(3)
                lcol1.markdown(f"**Model:** `{llm_meta.get('model', 'N/A')}`")
                lcol2.markdown(f"**Thinking:** `{llm_meta.get('thinking_level', 'N/A')}`")
                if cache_hit is None:
                    lcol3.markdown("**Cache hit:** `N/A`")
                else:
                    lcol3.markdown(f"**Cache hit:** `{cache_hit}`")

                lcol4, lcol5, lcol6 = st.columns(3)
                lcol4.markdown(f"**Function call(s):** `{function_calls}`")
                lcol5.markdown(f"**Thought signature hash:** `{signature_hash}`")
                lcol6.markdown(f"**Usage tokens:** `{usage_tokens}`")

                llm_structured = dict(llm_meta)
                if function_calls != "-":
                    llm_structured["function_calls"] = [name.strip() for name in function_calls.split(",") if name.strip()]
                if signature_hash != "-":
                    llm_structured["thought_signature_hash"] = signature_hash
                if usage_tokens != "-":
                    llm_structured["usage_tokens"] = usage_tokens
                if cache_hit is not None:
                    llm_structured["cache_hit"] = cache_hit
                st.json(llm_structured)

            if entry.beliefs:
                st.markdown("**BDI State - Beliefs:**")
                st.json(entry.beliefs)

            if entry.intentions:
                st.markdown("**BDI State - Intentions:**")
                st.json(entry.intentions)

            if entry.evidence_anchors:
                st.markdown(f"**Evidence Anchors:** {len(entry.evidence_anchors)}")


def render_evidence_chain(result: dict):
    """Render the evidence chain visualization."""
    evidence = result.get("evidence", [])

    if not evidence:
        st.info("No evidence collected")
        return

    st.markdown(f"### Evidence Chain ({len(evidence)} sources)")

    # Group by document
    docs = {}
    for e in evidence:
        if isinstance(e, dict):
            doc_id = e.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "title": e.get("doc_title", doc_id),
                    "anchors": []
                }
            docs[doc_id]["anchors"].append(e)

    # Display by document
    for doc_id, doc_info in docs.items():
        with st.expander(f"📄 {doc_info['title']} ({len(doc_info['anchors'])} references)"):
            for i, anchor in enumerate(doc_info["anchors"], 1):
                st.markdown(f"**Reference {i}** - {anchor.get('location', 'N/A')}")
                st.markdown(f"*Relevance: {anchor.get('relevance_score', 0):.2f}*")
                st.text(anchor.get("snippet", "")[:300])
                st.markdown("---")


def render_replay_page():
    """Render the replay page."""
    st.header("🔁 Replay Previous Run")

    store = get_store()
    runs = store.list_runs()

    if not runs:
        st.info("No runs available for replay. Run a task first!")
        return

    # Select run to replay
    selected_run = st.selectbox(
        "Select run to replay",
        runs,
        format_func=lambda x: f"{x} - {store.get_metadata(x).task_description[:50] if store.get_metadata(x) else 'Unknown'}..."
    )

    if selected_run:
        metadata = store.get_metadata(selected_run)
        if metadata:
            st.markdown("**Original Run Details:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Seed", metadata.seed or "None")
            col2.metric("Steps", metadata.total_steps)
            col3.metric("Status", metadata.status)

            st.markdown(f"**Task:** {metadata.task_description}")

    if st.button("🔁 Replay Run", type="primary"):
        with st.spinner("Replaying run..."):
            engine = ReplayEngine(store=store)
            result = engine.replay(selected_run)

            st.session_state.current_run_id = result.get("run_id")

        # Show comparison
        if result.get("success"):
            st.success(f"✅ Replay completed! New Run ID: {result.get('run_id')}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Matches Original",
                    "Yes" if result.get("matches_original") else "No"
                )
            with col2:
                diff_summary = result.get("diff_summary", {})
                st.metric(
                    "Step Differences",
                    len(diff_summary.get("step_differences", []))
                )

            cache_stats = result.get("llm_cache_stats", {}) or {}
            if cache_stats:
                st.caption(
                    "LLM cache hit rate: "
                    f"{cache_stats.get('cache_hits', 0)}/{cache_stats.get('llm_calls', 0)} "
                    f"({cache_stats.get('cache_hit_rate', 0.0):.1%})"
                )

            # Show diff summary
            if not result.get("matches_original"):
                st.warning("Results differ from original run!")
                render_diff_summary(diff_summary)
        else:
            st.error(f"Replay failed: {result.get('error')}")


def render_diff_page():
    """Render the diff comparison page."""
    st.header("📊 Compare Two Runs")

    store = get_store()
    runs = store.list_runs()

    if len(runs) < 2:
        st.info("Need at least 2 runs to compare. Run more tasks first!")
        return

    col1, col2 = st.columns(2)

    with col1:
        run_a = st.selectbox(
            "Run A",
            runs,
            key="diff_a",
            format_func=lambda x: f"{x[:30]}..."
        )

    with col2:
        run_b = st.selectbox(
            "Run B",
            [r for r in runs if r != run_a],
            key="diff_b",
            format_func=lambda x: f"{x[:30]}..."
        )

    if st.button("📊 Compare Runs", type="primary"):
        with st.spinner("Comparing runs..."):
            engine = DiffEngine(store=store)
            report = engine.compare(run_a, run_b)

        render_diff_report(report)


def render_diff_summary(diff_summary: dict):
    """Render a brief diff summary."""
    st.markdown("**Diff Summary:**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Answer Match", "Yes" if diff_summary.get("answer_match") else "No")
    col2.metric("Steps Compared", diff_summary.get("steps_compared", 0))

    cost = diff_summary.get("cost_comparison", {})
    if cost:
        col3.metric(
            "Latency Diff",
            f"{cost.get('latency_diff_ms', 0):.1f}ms"
        )


def render_diff_report(report):
    """Render a full diff report."""
    # Overall status
    if report.identical:
        st.success("✅ Runs are IDENTICAL")
    else:
        st.warning("⚠️ Runs are DIFFERENT")

    st.metric("Similarity Score", f"{report.similarity_score:.1%}")

    st.markdown("**LLM / Mode Config**")
    if getattr(report, "llm_config_diffs", None):
        for diff in report.llm_config_diffs:
            st.markdown(
                f"- `{diff['field']}`: `{diff['value_a']}` -> `{diff['value_b']}`"
            )
    else:
        st.markdown("- No LLM/mode config differences detected.")

    if hasattr(report, "llm_calls_a"):
        st.caption(
            f"LLM calls A/B: {report.llm_calls_a}/{report.llm_calls_b} | "
            f"Cache hits A/B: {report.llm_cache_hits_a}/{report.llm_cache_hits_b}"
        )
    if getattr(report, "llm_call_diffs", None):
        st.markdown("**LLM Call Differences (cache_key level):**")
        for diff in report.llm_call_diffs[:10]:
            status = diff.get("status", "")
            if status == "changed":
                st.markdown(
                    f"- `{diff.get('cache_key','')}`: "
                    f"`{str(diff.get('hash_a',''))[:12]}` -> `{str(diff.get('hash_b',''))[:12]}` "
                    f"({diff.get('model_a','')}/{diff.get('thinking_a','')} -> "
                    f"{diff.get('model_b','')}/{diff.get('thinking_b','')})"
                )
            else:
                st.markdown(f"- `{diff.get('cache_key','')}`: {status}")

    # Tabs for different comparisons
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Answer Diff",
        "🔢 Step Diff",
        "💰 Cost Comparison",
        "🔧 Tool Usage"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Run A Answer:**")
            st.text(report.answer_a[:500] if report.answer_a else "No answer")
        with col2:
            st.markdown("**Run B Answer:**")
            st.text(report.answer_b[:500] if report.answer_b else "No answer")

        if report.answer_diff:
            st.markdown("**Unified Diff:**")
            diff_text = "\n".join(report.answer_diff[:50])
            st.code(diff_text, language="diff")

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Steps in A", report.steps_a)
        col2.metric("Steps in B", report.steps_b)
        col3.metric("Different Steps", report.different_steps)

        if report.step_diffs:
            st.markdown("**Step Differences:**")
            for diff in report.step_diffs[:10]:
                st.markdown(f"- Step {diff['step']}: {diff['status']}")
                if diff.get("details"):
                    st.json(diff["details"])

    with tab3:
        col1, col2, col3 = st.columns(3)
        col1.metric("Latency A", f"{report.latency_a_ms:.1f}ms")
        col2.metric("Latency B", f"{report.latency_b_ms:.1f}ms")
        col3.metric("Difference", f"{report.latency_diff_ms:.1f}ms ({report.latency_diff_percent:.1f}%)")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Run A Tool Usage:**")
            for tool, count in report.tools_a.items():
                st.markdown(f"- {tool}: {count}")
        with col2:
            st.markdown("**Run B Tool Usage:**")
            for tool, count in report.tools_b.items():
                st.markdown(f"- {tool}: {count}")

        if report.tool_diffs:
            st.markdown("**Differences:**")
            for diff in report.tool_diffs:
                st.markdown(f"- {diff['tool']}: {diff['count_a']} → {diff['count_b']} ({diff['diff']:+d})")


def render_history_page():
    """Render the run history page."""
    st.header("📋 Run History")

    store = get_store()
    runs = store.list_runs()

    if not runs:
        st.info("No runs yet. Start by running a task!")
        return

    # Filter and sort options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_options = ["completed", "completed_with_warnings", "waiting", "failed", "running"]
        status_filter = st.multiselect(
            "Filter by status",
            status_options,
            default=["completed", "completed_with_warnings", "waiting", "failed"]
        )
    with col2:
        limit = st.slider("Show runs", 5, 50, 20)
    with col3:
        sort_option = st.selectbox(
            "Sort runs by",
            options=[
                "Start Time (Newest)",
                "Start Time (Oldest)",
                "End Time (Newest)",
                "End Time (Oldest)",
                "Run ID (Newest)",
                "Run ID (Oldest)",
            ],
            index=0,
        )

    selected_run_id = st.session_state.get("history_selected_run_id")
    if selected_run_id:
        st.markdown("---")
        dcol1, dcol2 = st.columns([4, 1])
        with dcol1:
            st.markdown(f"### Selected Run Details: `{selected_run_id}`")
        with dcol2:
            if st.button("Clear Selection", key="clear_history_selection"):
                st.session_state.history_selected_run_id = None
                st.rerun()

        detail_result = _load_run_result_for_display(selected_run_id, store)
        if detail_result:
            render_run_result(detail_result, verbose=True)
        else:
            st.warning(f"Unable to load details for run `{selected_run_id}`.")
        st.markdown("---")

    # Build filtered list with metadata, then sort.
    run_items = []
    for run_id in runs:
        metadata = store.get_metadata(run_id)
        if not metadata:
            continue
        if status_filter and metadata.status not in status_filter:
            continue
        run_items.append((run_id, metadata))

    def _sort_key(item):
        run_id, metadata = item
        if sort_option.startswith("Start Time"):
            return _parse_iso_datetime(getattr(metadata, "start_time", "")) or datetime.min
        if sort_option.startswith("End Time"):
            return _parse_iso_datetime(getattr(metadata, "end_time", "")) or datetime.min
        return run_id

    reverse = sort_option.endswith("(Newest)")
    run_items.sort(key=_sort_key, reverse=reverse)

    # Display runs
    for run_id, metadata in run_items[:limit]:

        if metadata.status == "completed":
            status_icon = "✅"
        elif metadata.status == "completed_with_warnings":
            status_icon = "⚠️"
        elif metadata.status == "waiting":
            status_icon = "⏸️"
        elif metadata.status == "running":
            status_icon = "🔄"
        else:
            status_icon = "❌"
        with st.expander(
            f"{status_icon} {run_id}"
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", metadata.status)
            col2.metric("Steps", metadata.total_steps)
            col3.metric("Seed", metadata.seed or "None")
            col4.metric("Replay", "Yes" if metadata.is_replay else "No")

            st.markdown(f"**Task:** {metadata.task_description}")
            st.markdown(f"**Time:** {metadata.start_time} → {metadata.end_time}")
            if getattr(metadata, "next_run_at", ""):
                st.markdown(f"**Next Run At:** `{metadata.next_run_at}`")

            # Action buttons
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)

            if bcol1.button("View Details", key=f"view_{run_id}"):
                st.session_state.current_run_id = run_id
                st.session_state.history_selected_run_id = run_id
                st.rerun()

            if bcol2.button("Replay", key=f"replay_{run_id}"):
                with st.spinner(f"Replaying {run_id}..."):
                    replay_engine = ReplayEngine(store=store)
                    replay_result = replay_engine.replay(run_id)

                if replay_result.get("success"):
                    new_run_id = replay_result.get("run_id")
                    if new_run_id:
                        st.session_state.current_run_id = new_run_id
                        st.session_state.history_selected_run_id = new_run_id
                        if new_run_id not in st.session_state.run_history:
                            st.session_state.run_history.append(new_run_id)

                    st.success(f"Replay completed: {new_run_id or 'N/A'}")
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        st.metric(
                            "Matches Original",
                            "Yes" if replay_result.get("matches_original") else "No",
                        )
                    with rcol2:
                        diff_summary = replay_result.get("diff_summary", {}) or {}
                        st.metric(
                            "Step Differences",
                            len(diff_summary.get("step_differences", []) or []),
                        )

                    cache_stats = replay_result.get("llm_cache_stats", {}) or {}
                    if cache_stats:
                        st.caption(
                            "LLM cache hit rate: "
                            f"{cache_stats.get('cache_hits', 0)}/{cache_stats.get('llm_calls', 0)} "
                            f"({cache_stats.get('cache_hit_rate', 0.0):.1%})"
                        )
                else:
                    st.error(f"Replay failed: {replay_result.get('error', 'unknown error')}")

            checkpoint_exists = (Path("runs") / run_id / "checkpoint.json").exists()
            can_resume = metadata.status == "waiting" or checkpoint_exists
            if bcol3.button("Resume", key=f"resume_{run_id}", disabled=not can_resume):
                llm_kwargs = {}
                if metadata.llm_mode:
                    llm_kwargs = {
                        "llm_mode": True,
                        "owl_mode": metadata.owl_mode or "owl_lite",
                        "llm_provider": metadata.llm_provider or "gemini",
                        "llm_model": metadata.llm_model or "",
                        "llm_thinking_level": metadata.llm_thinking_level or "",
                    }
                else:
                    llm_kwargs = {
                        "llm_mode": False,
                        "owl_mode": metadata.owl_mode or "owl_lite",
                    }
                with st.spinner(f"Resuming {run_id}..."):
                    orchestrator = Orchestrator(
                        corpus_dir="demo_data/corpus",
                        output_dir="runs",
                        seed=metadata.seed,
                        marathon_context={
                            "enabled": True,
                            "resume": True,
                            "source_run_id": run_id,
                        },
                        **llm_kwargs,
                    )
                    resumed = orchestrator.run(metadata.task_description)
                if resumed.get("success"):
                    st.session_state.current_run_id = resumed.get("run_id")
                    if resumed.get("run_id"):
                        st.session_state.run_history.append(resumed.get("run_id"))
                    st.success(f"Resumed: {resumed.get('run_id', 'N/A')}")
                else:
                    st.error(f"Resume failed: {resumed.get('error', 'unknown error')}")

            if bcol4.button("Export", key=f"export_{run_id}"):
                # Export trace as JSON
                entries = store.get_entries(run_id)
                export_data = {
                    "metadata": metadata.to_dict(),
                    "entries": [e.to_dict() for e in entries]
                }
                st.download_button(
                    "Download JSON",
                    json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
                    file_name=f"{run_id}.json",
                    mime="application/json",
                    key=f"dl_{run_id}"
                )


def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if page == "🚀 Run Task":
        render_run_page()
    elif page == "🔁 Replay":
        render_replay_page()
    elif page == "📊 Compare (Diff)":
        render_diff_page()
    elif page == "📋 Run History":
        render_history_page()


if __name__ == "__main__":
    main()
