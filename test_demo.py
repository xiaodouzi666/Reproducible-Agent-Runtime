#!/usr/bin/env python3
"""
Quick test script to verify RAR demo functionality.
Run this to ensure everything is working before demo.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from rar.protocols import ACLMessage, Performative, ContractNetProtocol, Bid
        from rar.tools import LocalSearchTool, PythonExecTool
        from rar.tracing import TraceStore, Tracer, TraceEntry
        from rar.agents import PlannerAgent, ResearcherAgent, ExecutorAgent, AuditorAgent
        from rar.orchestrator import Orchestrator
        from rar.replay import ReplayEngine
        from rar.diff import DiffEngine
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_search_tool():
    """Test the local search tool."""
    print("Testing search tool...")
    try:
        from rar.tools import LocalSearchTool
        search = LocalSearchTool("demo_data/corpus")
        result = search.execute(query="thermal analysis", top_k=3)
        if result.success and len(result.output) > 0:
            print(f"  ✓ Search found {len(result.output)} results")
            return True
        else:
            print("  ✗ Search returned no results")
            return False
    except Exception as e:
        print(f"  ✗ Search error: {e}")
        return False


def test_python_exec():
    """Test the Python execution tool."""
    print("Testing Python exec tool...")
    try:
        from rar.tools import PythonExecTool
        exec_tool = PythonExecTool(seed=42)
        code = "result = 2 + 2\nprint(f'Result: {result}')"
        result = exec_tool.execute(code=code)
        if result.success and "4" in result.output.get("stdout", ""):
            print("  ✓ Python execution works")
            return True
        else:
            print(f"  ✗ Unexpected output: {result.output}")
            return False
    except Exception as e:
        print(f"  ✗ Exec error: {e}")
        return False


def test_trace_store():
    """Test the trace store."""
    print("Testing trace store...")
    try:
        from rar.tracing import TraceStore
        store = TraceStore("runs")
        runs = store.list_runs()
        if "sample_run_1" in runs:
            metadata = store.get_metadata("sample_run_1")
            if metadata and metadata.status == "completed":
                print(f"  ✓ Found {len(runs)} runs, sample_run_1 is valid")
                return True
        print("  ✗ sample_run_1 not found or invalid")
        return False
    except Exception as e:
        print(f"  ✗ Store error: {e}")
        return False


def test_simple_workflow():
    """Test a simple workflow execution."""
    print("Testing simple workflow...")
    try:
        from rar.orchestrator import Orchestrator
        orchestrator = Orchestrator(
            corpus_dir="demo_data/corpus",
            output_dir="runs",
            seed=123
        )
        result = orchestrator.run("What is activation energy?")
        if result.get("success"):
            print(f"  ✓ Workflow completed: {result.get('run_id')}")
            return True
        else:
            print(f"  ✗ Workflow failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"  ✗ Workflow error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("RAR Demo Test Suite")
    print("=" * 50)
    print()

    tests = [
        test_imports,
        test_search_tool,
        test_python_exec,
        test_trace_store,
        test_simple_workflow,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("✓ All tests passed! Demo is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
