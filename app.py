#!/usr/bin/env python3
"""
RAR - Reproducible Agent Runtime
Streamlit UI

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

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

    col1, col2 = st.columns([2, 1])

    with col1:
        # Task input
        task = st.text_area(
            "Task Description",
            placeholder="Enter your scientific question or task...\n\nExample: What is the activation energy for thermal decomposition of polymers?",
            height=100
        )

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
                task = example
                st.rerun()

    with col2:
        st.markdown("**Options**")
        seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
        verbose = st.checkbox("Show detailed trace", value=True)

    # Run button
    if st.button("▶️ Run Task", type="primary", disabled=not task):
        with st.spinner("Running multi-agent workflow..."):
            orchestrator = Orchestrator(
                corpus_dir="demo_data/corpus",
                output_dir="runs",
                seed=seed
            )
            result = orchestrator.run(task)

            st.session_state.current_run_id = result.get("run_id")
            st.session_state.run_history.append(result.get("run_id"))

        # Show result
        render_run_result(result, verbose)


def render_run_result(result: dict, verbose: bool = True):
    """Render the result of a run."""
    success = result.get("success", False)

    # Status header
    if success:
        st.success(f"✅ Run completed successfully! (ID: {result.get('run_id', 'N/A')})")
    else:
        st.error(f"❌ Run failed: {result.get('error', 'Unknown error')}")

    # Three-panel layout
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
        timeline_data.append({
            "Step": entry.step_id,
            "Time": entry.timestamp,
            "Event": entry.event_type.value,
            "Agent": entry.agent_id,
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

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filter by status",
            ["completed", "failed", "running"],
            default=["completed", "failed"]
        )
    with col2:
        limit = st.slider("Show runs", 5, 50, 20)

    # Display runs
    for run_id in runs[:limit]:
        metadata = store.get_metadata(run_id)
        if not metadata:
            continue

        if status_filter and metadata.status not in status_filter:
            continue

        with st.expander(
            f"{'✅' if metadata.status == 'completed' else '❌'} {run_id}"
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", metadata.status)
            col2.metric("Steps", metadata.total_steps)
            col3.metric("Seed", metadata.seed or "None")
            col4.metric("Replay", "Yes" if metadata.is_replay else "No")

            st.markdown(f"**Task:** {metadata.task_description}")
            st.markdown(f"**Time:** {metadata.start_time} → {metadata.end_time}")

            # Action buttons
            bcol1, bcol2, bcol3 = st.columns(3)

            if bcol1.button("View Details", key=f"view_{run_id}"):
                st.session_state.current_run_id = run_id
                # Would navigate to detail view

            if bcol2.button("Replay", key=f"replay_{run_id}"):
                # Trigger replay
                pass

            if bcol3.button("Export", key=f"export_{run_id}"):
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
