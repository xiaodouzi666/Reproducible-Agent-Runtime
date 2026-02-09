#!/usr/bin/env python3
"""
RAR - Reproducible Agent Runtime
Command Line Interface

Usage:
    python run.py run "Your task description"
    python run.py run --spec demo_data/tasks/task1.yaml
    python run.py resume <run_id>
    python run.py replay <run_id>
    python run.py replay --spec demo_data/tasks/task1.yaml
    python run.py diff <run_id_a> <run_id_b>
    python run.py list
    python run.py show <run_id>
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path
import yaml

from rar.orchestrator import Orchestrator
from rar.replay import ReplayEngine
from rar.diff import DiffEngine, DiffReport
from rar.tracing import TraceStore, Tracer


def cmd_run(args):
    """Run a new task."""
    llm_kwargs = {}
    if getattr(args, "enable_gemini", False):
        try:
            from rar.llm import GeminiClient
            resolved = GeminiClient.resolve_mode_defaults(
                mode=args.mode,
                model_override=args.model,
                thinking_override=args.thinking_level,
            )
            llm_kwargs = {
                "llm_mode": True,
                "owl_mode": resolved["mode"],
                "llm_provider": "gemini",
                "llm_model": resolved["model"],
                "llm_thinking_level": resolved["thinking_level"],
            }
        except Exception as e:
            print(f"Warning: Gemini mode requested but unavailable: {e}")
            print("Falling back to non-LLM workflow.")

    orchestrator = Orchestrator(
        corpus_dir=args.corpus,
        output_dir=args.output,
        seed=args.seed,
        **llm_kwargs,
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

    cache_stats = result.get("llm_cache_stats", {}) or {}
    if cache_stats:
        print(
            "LLM Cache Hit Rate: "
            f"{cache_stats.get('cache_hits', 0)}/{cache_stats.get('llm_calls', 0)} "
            f"({cache_stats.get('cache_hit_rate', 0.0):.1%})"
        )

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
    print(f"LLM Mode: {metadata.llm_mode}")
    print(f"OWL Mode: {metadata.owl_mode or 'N/A'}")
    print(f"LLM Model: {metadata.llm_model or 'N/A'}")
    print(f"Thinking Level: {metadata.llm_thinking_level or 'N/A'}")
    print(f"LLM Finalize Called: {getattr(metadata, 'llm_finalize_called', False)}")
    print(f"Finalize Missing: {getattr(metadata, 'finalize_missing', False)}")
    print(f"Argument Graph Generated: {getattr(metadata, 'argument_graph_generated', False)}")
    print(f"Next Run At: {getattr(metadata, 'next_run_at', '') or 'N/A'}")
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


def cmd_llm_test(args):
    """Minimal Gemini connectivity test with trace logging."""
    try:
        from rar.llm import GeminiClient, GeminiClientError
    except Exception as e:
        print(f"Failed to import Gemini integration: {e}")
        return 1

    resolved = GeminiClient.resolve_mode_defaults(
        mode=args.mode,
        model_override=args.model,
        thinking_override=args.thinking_level,
    )
    mode = resolved["mode"]
    model = resolved["model"]
    thinking_level = resolved["thinking_level"]

    run_id = args.run_id or f"llm_test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    store = TraceStore(args.output)
    tracer = Tracer(run_id=run_id, store=store)
    tracer.start_run(
        task_description=args.prompt,
        seed=args.seed,
        llm_mode=True,
        owl_mode=mode,
        llm_provider="gemini",
        llm_model=model,
        llm_thinking_level=thinking_level,
    )

    try:
        client = GeminiClient(
            run_dir=str(run_dir),
            auto_fallback=True,
        )

        cache_key = client.compute_cache_key(
            prompt=args.prompt,
            model=model,
            thinking_level=thinking_level,
            system_prompt=args.system_prompt,
            response_format="text",
        )
        request_payload = {
            "provider": "gemini",
            "model": model,
            "thinking_level": thinking_level,
            "prompt": args.prompt,
            "system_prompt": args.system_prompt or "",
            "tools_schema": [],
            "response_format": "text",
        }
        config_payload = {
            "thinking_level": thinking_level,
            "system_prompt": args.system_prompt or "",
            "tool_schema_names": [],
            "response_mime_type": "text/plain",
        }

        tracer.log_llm_call(
            agent_id="llm_tester",
            model=model,
            thinking_level=thinking_level,
            cache_key=cache_key,
            prompt_summary=args.prompt[:200],
            tool_schema_names=[],
            request=request_payload,
            config=config_payload,
        )

        record = client.generate_text(
            prompt=args.prompt,
            model=model,
            thinking_level=thinking_level,
            system_prompt=args.system_prompt,
        )

        if record.is_cache_hit:
            tracer.log_llm_cache_hit(
                agent_id="llm_tester",
                model=record.model,
                thinking_level=record.thinking_level,
                cache_key=record.cache_key,
                response_hash=record.response_hash,
                usage=record.usage,
                latency_ms=record.latency_ms,
            )

        tracer.log_llm_result(
            agent_id="llm_tester",
            cache_key=record.cache_key,
            response_text_summary=(record.response_text or "")[:300],
            response_hash=record.response_hash,
            usage=record.usage,
            model=record.model,
            thinking_level=record.thinking_level,
            deterministic=record.is_cache_hit,
            latency_ms=record.latency_ms,
            response_text=record.response_text,
            response_raw=record.response_raw,
        )

        final_answer = record.response_text or "(empty response)"
        tracer.end_run(
            status="completed",
            final_answer=final_answer,
            evidence_summary=[],
        )

        # Save run spec/final summary for consistent run artifacts.
        run_spec = {
            "task": args.prompt,
            "run_id": run_id,
            "seed": args.seed,
            "owl_mode": mode,
            "llm_provider": "gemini",
            "llm_model": record.model,
            "llm_thinking_level": record.thinking_level,
        }
        with open(run_dir / "run_spec.yaml", "w", encoding="utf-8") as f:
            yaml.dump(run_spec, f, allow_unicode=True, default_flow_style=False)

        final = {
            "run_id": run_id,
            "task": args.prompt,
            "answer": final_answer,
            "status": "completed",
            "llm": {
                "provider": "gemini",
                "mode": mode,
                "model": record.model,
                "thinking_level": record.thinking_level,
                "cache_key": record.cache_key,
                "cache_hit": record.is_cache_hit,
                "response_hash": record.response_hash,
                "usage": record.usage,
                "fallback_from": record.fallback_from,
            },
        }
        with open(run_dir / "final.json", "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2, default=str)

        print("\n" + "=" * 60)
        print("LLM TEST COMPLETED")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        print(f"Mode: {mode}")
        print(f"Model: {record.model}")
        print(f"Thinking Level: {record.thinking_level}")
        print(f"Cache Hit: {record.is_cache_hit}")
        print(f"Response Hash: {record.response_hash[:16]}...")
        if record.fallback_from:
            print(f"Fallback Applied: {record.fallback_from} -> {record.model}")
        print("\n--- Response ---")
        print(final_answer)
        return 0

    except GeminiClientError as e:
        tracer.end_run(status="failed", final_answer=f"LLM error: {e}")
        print("\nGemini call failed.")
        print(f"Error: {e}")
        print("Hint: verify GEMINI_API_KEY and quota; retry with --mode owl_lite.")
        return 1
    except Exception as e:
        tracer.end_run(status="failed", final_answer=f"Unexpected error: {e}")
        print(f"\nUnexpected llm-test error: {e}")
        return 1


def cmd_resume(args):
    """Resume a waiting/paused run from checkpoint."""
    store = TraceStore(args.output)
    metadata = store.get_metadata(args.run_id)
    if not metadata:
        print(f"Run not found: {args.run_id}")
        return 1

    task_description = args.task or metadata.task_description
    llm_mode = bool(metadata.llm_mode or args.enable_gemini)
    llm_kwargs = {}

    if llm_mode:
        mode = args.mode or metadata.owl_mode or "owl_lite"
        model = args.model or metadata.llm_model or ""
        thinking = args.thinking_level or metadata.llm_thinking_level or ""
        try:
            from rar.llm import GeminiClient

            resolved = GeminiClient.resolve_mode_defaults(
                mode=mode,
                model_override=model or None,
                thinking_override=thinking or None,
            )
            llm_kwargs = {
                "llm_mode": True,
                "owl_mode": resolved["mode"],
                "llm_provider": "gemini",
                "llm_model": resolved["model"],
                "llm_thinking_level": resolved["thinking_level"],
            }
        except Exception as e:
            print(f"Warning: resume requested with Gemini but setup failed: {e}")
            print("Falling back to non-LLM resume path.")
            llm_kwargs = {
                "llm_mode": False,
                "owl_mode": metadata.owl_mode or "owl_lite",
            }
    else:
        llm_kwargs = {
            "llm_mode": False,
            "owl_mode": metadata.owl_mode or "owl_lite",
        }

    orchestrator = Orchestrator(
        corpus_dir=args.corpus,
        output_dir=args.output,
        seed=args.seed if args.seed is not None else metadata.seed,
        marathon_context={
            "enabled": True,
            "resume": True,
            "source_run_id": args.run_id,
        },
        **llm_kwargs,
    )

    result = orchestrator.run(task_description)

    print("\n" + "=" * 60)
    print("RESUME COMPLETED")
    print("=" * 60)
    print(f"Run ID: {result.get('run_id', 'N/A')}")
    print(f"Source Run ID: {args.run_id}")
    print(f"Success: {result.get('success', False)}")
    if result.get("run_status") == "waiting":
        print(f"Next Run At: {result.get('next_run_at', '') or 'N/A'}")
    print("\n--- Final Answer ---")
    print(result.get("final_answer", result.get("error", "No answer")))

    return 0 if result.get("success") else 1


def main():
    parser = argparse.ArgumentParser(
        description="RAR - Reproducible Agent Runtime CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py run "What is the activation energy for thermal decomposition?"
  python run.py run --spec demo_data/tasks/task1.yaml --seed 42
  python run.py resume run_20240101_120000_abc123
  python run.py replay run_20240101_120000_abc123
  python run.py diff run_a run_b --save
  python run.py llm-test --mode owl_lite --prompt "Say hi"
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
    run_parser.add_argument("--enable-gemini", action="store_true",
                            help="Store Gemini mode/model/thinking config in run metadata/spec")
    run_parser.add_argument("--mode", choices=["owl_lite", "owl_dl", "owl_full"],
                            default="owl_lite", help="OWL mode preset for Gemini metadata")
    run_parser.add_argument("--model", help="Optional Gemini model override")
    run_parser.add_argument("--thinking-level", help="Optional thinking level override")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a previous run")
    replay_parser.add_argument("run_id", nargs="?", help="Run ID to replay")
    replay_parser.add_argument("--spec", "-s", help="Path to replay spec YAML file")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a waiting run from checkpoint")
    resume_parser.add_argument("run_id", help="Run ID to resume from checkpoint")
    resume_parser.add_argument("--task", help="Optional task override")
    resume_parser.add_argument("--seed", type=int, help="Optional seed override")
    resume_parser.add_argument("--enable-gemini", action="store_true",
                               help="Force-enable Gemini resume path")
    resume_parser.add_argument("--mode", choices=["owl_lite", "owl_dl", "owl_full"],
                               help="Optional OWL mode override")
    resume_parser.add_argument("--model", help="Optional Gemini model override")
    resume_parser.add_argument("--thinking-level", help="Optional thinking level override")

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

    # LLM test command
    llm_parser = subparsers.add_parser("llm-test", help="Run a minimal Gemini call with tracing")
    llm_parser.add_argument("--prompt", default="Say hi in one concise sentence.",
                            help="Prompt sent to Gemini")
    llm_parser.add_argument("--mode", choices=["owl_lite", "owl_dl", "owl_full"],
                            default="owl_lite", help="OWL mode preset")
    llm_parser.add_argument("--model", help="Optional model override")
    llm_parser.add_argument("--thinking-level", help="Optional thinking level override")
    llm_parser.add_argument("--system-prompt", help="Optional system prompt")
    llm_parser.add_argument("--seed", type=int, default=42, help="Seed stored in run metadata")
    llm_parser.add_argument("--run-id", help="Optional explicit run ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command
    commands = {
        "run": cmd_run,
        "resume": cmd_resume,
        "replay": cmd_replay,
        "diff": cmd_diff,
        "list": cmd_list,
        "show": cmd_show,
        "llm-test": cmd_llm_test,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
