#!/usr/bin/env python3
"""
RAR - Reproducible Agent Runtime
Command Line Interface

Usage:
    python run.py run "Your task description"
    python run.py run --spec demo_data/tasks/task1.yaml
    python run.py replay <run_id>
    python run.py replay --spec demo_data/tasks/task1.yaml
    python run.py diff <run_id_a> <run_id_b>
    python run.py list
    python run.py show <run_id>
"""

import argparse
import sys
import json
from pathlib import Path

from rar.orchestrator import Orchestrator
from rar.replay import ReplayEngine
from rar.diff import DiffEngine, DiffReport
from rar.tracing import TraceStore


def cmd_run(args):
    """Run a new task."""
    orchestrator = Orchestrator(
        corpus_dir=args.corpus,
        output_dir=args.output,
        seed=args.seed
    )

    if args.spec:
        result = orchestrator.run_from_spec(args.spec)
    elif args.task:
        result = orchestrator.run(args.task)
    else:
        print("Error: Either --spec or a task description is required")
        return 1

    # Print result
    print("\n" + "=" * 60)
    print("RUN COMPLETED")
    print("=" * 60)
    print(f"Run ID: {result.get('run_id', 'N/A')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Trace Path: {result.get('trace_path', 'N/A')}")
    print("\n--- Final Answer ---")
    print(result.get('final_answer', result.get('error', 'No answer')))

    if args.verbose:
        print("\n--- Evidence ---")
        evidence = result.get('evidence', [])
        for i, e in enumerate(evidence, 1):
            if isinstance(e, dict):
                print(f"{i}. [{e.get('doc_id', 'unknown')}] {e.get('snippet', '')[:100]}...")

    return 0 if result.get('success') else 1


def cmd_replay(args):
    """Replay a previous run."""
    store = TraceStore(args.output)
    engine = ReplayEngine(store=store, output_dir=args.output)

    if args.spec:
        result = engine.replay_from_spec(args.spec)
    elif args.run_id:
        result = engine.replay(args.run_id)
    else:
        print("Error: Either --spec or a run_id is required")
        return 1

    # Print result
    print("\n" + "=" * 60)
    print("REPLAY COMPLETED")
    print("=" * 60)
    print(f"New Run ID: {result.get('run_id', 'N/A')}")
    print(f"Original Run ID: {result.get('original_run_id', 'N/A')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Matches Original: {result.get('matches_original', 'N/A')}")

    if result.get('diff_summary'):
        diff = result['diff_summary']
        print(f"\n--- Comparison ---")
        print(f"Identical: {diff.get('identical', False)}")
        print(f"Answer Match: {diff.get('answer_match', False)}")
        print(f"Step Differences: {len(diff.get('step_differences', []))}")

    return 0 if result.get('success') else 1


def cmd_diff(args):
    """Compare two runs."""
    store = TraceStore(args.output)
    engine = DiffEngine(store=store)

    report = engine.compare(args.run_id_a, args.run_id_b)

    # Print report
    print(engine.format_report(report))

    # Save report if requested
    if args.save:
        output_path = Path(args.output) / f"diff_{args.run_id_a}_{args.run_id_b}.txt"
        engine.save_report(report, str(output_path))
        print(f"\nReport saved to: {output_path}")

    return 0


def cmd_list(args):
    """List all runs."""
    store = TraceStore(args.output)
    runs = store.list_runs()

    if not runs:
        print("No runs found.")
        return 0

    print(f"\n{'Run ID':<45} {'Status':<12} {'Task'}")
    print("-" * 80)

    for run_id in runs[:args.limit]:
        metadata = store.get_metadata(run_id)
        if metadata:
            task = metadata.task_description[:30] + "..." if len(metadata.task_description) > 30 else metadata.task_description
            status = metadata.status
            if metadata.is_replay:
                status += " (replay)"
            print(f"{run_id:<45} {status:<12} {task}")

    if len(runs) > args.limit:
        print(f"\n... and {len(runs) - args.limit} more runs")

    return 0


def cmd_show(args):
    """Show details of a run."""
    store = TraceStore(args.output)
    metadata = store.get_metadata(args.run_id)

    if not metadata:
        print(f"Run not found: {args.run_id}")
        return 1

    print("\n" + "=" * 60)
    print(f"RUN: {args.run_id}")
    print("=" * 60)
    print(f"Task: {metadata.task_description}")
    print(f"Status: {metadata.status}")
    print(f"Start: {metadata.start_time}")
    print(f"End: {metadata.end_time}")
    print(f"Seed: {metadata.seed}")
    print(f"Is Replay: {metadata.is_replay}")
    if metadata.is_replay:
        print(f"Original Run: {metadata.original_run_id}")

    # Show trace entries
    if args.trace:
        entries = store.get_entries(args.run_id)
        print(f"\n--- Trace ({len(entries)} entries) ---")
        for entry in entries[:args.limit]:
            print(entry.format_for_display())
            print()

    # Show final answer
    print("\n--- Final Answer ---")
    print(metadata.final_answer or "(No answer)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RAR - Reproducible Agent Runtime CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py run "What is the activation energy for thermal decomposition?"
  python run.py run --spec demo_data/tasks/task1.yaml --seed 42
  python run.py replay run_20240101_120000_abc123
  python run.py diff run_a run_b --save
  python run.py list
  python run.py show run_20240101_120000_abc123 --trace
        """
    )

    # Global options
    parser.add_argument("--output", "-o", default="runs",
                        help="Output directory for runs (default: runs)")
    parser.add_argument("--corpus", "-c", default="demo_data/corpus",
                        help="Corpus directory (default: demo_data/corpus)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a new task")
    run_parser.add_argument("task", nargs="?", help="Task description")
    run_parser.add_argument("--spec", "-s", help="Path to spec YAML file")
    run_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    run_parser.add_argument("--verbose", "-v", action="store_true",
                            help="Show verbose output")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a previous run")
    replay_parser.add_argument("run_id", nargs="?", help="Run ID to replay")
    replay_parser.add_argument("--spec", "-s", help="Path to replay spec YAML file")

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two runs")
    diff_parser.add_argument("run_id_a", help="First run ID")
    diff_parser.add_argument("run_id_b", help="Second run ID")
    diff_parser.add_argument("--save", action="store_true",
                             help="Save diff report to file")

    # List command
    list_parser = subparsers.add_parser("list", help="List all runs")
    list_parser.add_argument("--limit", "-n", type=int, default=20,
                             help="Maximum number of runs to show")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("run_id", help="Run ID to show")
    show_parser.add_argument("--trace", "-t", action="store_true",
                             help="Show trace entries")
    show_parser.add_argument("--limit", "-n", type=int, default=50,
                             help="Maximum trace entries to show")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command
    commands = {
        "run": cmd_run,
        "replay": cmd_replay,
        "diff": cmd_diff,
        "list": cmd_list,
        "show": cmd_show,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
