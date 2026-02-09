"""
Planner Agent - BDI-based task decomposition and coordination.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .base import BaseAgent
from ..config import resolve_mode_config, ModeConfig
from ..llm.schemas import PlanSchema, AuditSchema, FinalizeSchema
from ..marathon import MarathonRunner, CheckpointState, WaitDirective
from ..protocols.acl import Performative
from ..protocols.contract_net import ContractNetProtocol, TaskAnnouncement, Bid
from ..tracing.tracer import Tracer


class PlannerAgent(BaseAgent):
    """
    Planner agent responsible for:
    - Task decomposition and coordination
    - Contract Net based worker selection
    - Optional LLM-driven function-calling loop
    """

    role = "planner"
    description = "Task decomposition and multi-agent coordination"

    def __init__(
        self,
        agent_id: str = "planner",
        tracer: Optional[Tracer] = None,
        tools: dict = None,
        llm_client: Optional[object] = None,
    ):
        super().__init__(agent_id, tracer, tools)
        self.llm_client = llm_client
        self.contract_net = ContractNetProtocol(self.agent_id)
        self.subtask_counter = 0

    def process(self, task: dict) -> dict:
        """
        Process a task either via:
        - LLM tool-calling loop (when enabled)
        - Rule-based decomposition (fallback/offline)
        """
        self.activate()

        context = task.get("context", {})
        workers = task.get("workers", [])

        self.update_beliefs(
            {
                "task": task.get("description", ""),
                "context": context,
                "available_workers": [w.agent_id for w in workers],
                "completed_subtasks": [],
                "pending_subtasks": [],
            },
            reason="Task received",
        )
        self.set_desires([f"Complete task: {task.get('description', '')}"])

        llm_enabled = bool(context.get("llm_enabled")) and self.llm_client is not None

        if llm_enabled:
            try:
                result = self.process_llm(task)
            except Exception as e:
                # Hard fallback to baseline planner to keep demo robust offline.
                self.update_beliefs({"llm_error": str(e)}, reason="LLM planning failed; fallback to rule-based")
                result = self._process_rule_based(task)
                result["llm_error"] = str(e)
                mode_cfg = resolve_mode_config(context.get("owl_mode", "owl_lite"))
                result["llm_finalize_called"] = False
                result["finalize_missing"] = bool(mode_cfg.require_finalize)
                result["llm_loop"] = {
                    "steps": 0,
                    "ended_reason": "llm_exception_fallback",
                    "finalize_called": False,
                    "force_finalize_attempted": False,
                }
                # For DL/Full, missing finalize means run is not considered successful.
                if mode_cfg.require_finalize:
                    result["success"] = False
                    result["run_status"] = "failed"
                elif not result.get("run_status"):
                    result["run_status"] = "completed_with_warnings" if result.get("uncertainty") else "completed"
        else:
            result = self._process_rule_based(task)

        self.result = result
        self.deactivate()
        return self.result

    def process_llm(self, task: dict) -> dict:
        """LLM-driven planning loop using Gemini function calling."""
        if self.llm_client is None:
            raise RuntimeError("LLM client not configured")

        description = task.get("description", "")
        context = task.get("context", {})
        workers = task.get("workers", [])

        mode = str(context.get("owl_mode", "owl_lite"))
        model_override = str(context.get("llm_model", "") or "")
        thinking_override = str(context.get("llm_thinking_level", "") or "")
        max_steps = int(context.get("llm_max_steps", 10))
        max_steps = max(2, min(max_steps, 12))
        llm_use_cache = bool(context.get("llm_use_cache", True))
        marathon_context = context.get("marathon", {}) if isinstance(context.get("marathon", {}), dict) else {}
        marathon_enabled = bool(marathon_context.get("enabled", True))
        marathon_output_dir = str(marathon_context.get("output_dir", "runs") or "runs")
        resume_requested = bool(marathon_context.get("resume", False))
        resume_source_run_id = str(marathon_context.get("source_run_id", "") or "")
        checkpoint_runner = MarathonRunner(output_dir=marathon_output_dir)

        mode_cfg = resolve_mode_config(
            mode=mode,
            model_override=model_override or None,
            thinking_override=thinking_override or None,
        )
        mode = mode_cfg.mode
        model = mode_cfg.model
        thinking_level = mode_cfg.thinking_level
        max_replans = int(context.get("max_replans", mode_cfg.max_replans))
        max_replans = max(0, min(max_replans, 4))

        self.update_beliefs(
            {
                "llm_enabled": True,
                "owl_mode": mode,
                "llm_model": model,
                "llm_thinking_level": thinking_level,
                "audit_level": mode_cfg.audit_level,
                "argumentation": mode_cfg.argumentation,
            },
            reason="LLM planner enabled",
        )

        tools_schema = self._build_planner_tools_schema()
        tool_schema_names = self._extract_tool_names(tools_schema)
        self.set_intentions(
            [
                {
                    "action": "llm_tool_loop",
                    "model": model,
                    "thinking_level": thinking_level,
                    "max_steps": max_steps,
                }
            ]
        )
        structured_plan: dict = {}
        try:
            structured_plan = self._generate_structured_plan(
                task_description=description,
                mode_cfg=mode_cfg,
                model=model,
                thinking_level=thinking_level,
            )
            if structured_plan:
                self.update_beliefs(
                    {"structured_plan": structured_plan},
                    reason="Structured plan generated",
                )
        except Exception:
            # Plan schema generation is non-blocking for tool loop.
            structured_plan = {}

        subtask_results: dict[str, dict] = {}
        all_evidence: list = []
        executed_plan: list = []
        all_audit_challenges: list = []
        latest_audit_challenges: list = []
        audit_rounds: list[dict] = []
        final_payload: dict = {}
        replan_attempts = 0
        has_research = False
        has_audit = False
        last_audit_approved: Optional[bool] = None
        checkpoint_warnings: list[str] = []
        checkpoint_path = ""
        checkpoint_history: list[Any] = []
        checkpoint_wait: Optional[WaitDirective] = None

        def _short_warning(prefix: str, err: Any) -> str:
            text = str(err or "").strip().replace("\n", " ")
            if len(text) > 160:
                text = text[:160] + "..."
            return f"{prefix}: {text}" if text else prefix

        def _merge_warnings(*groups: list[Any]) -> list[str]:
            output: list[str] = []
            seen = set()
            for group in groups:
                if not isinstance(group, list):
                    continue
                for item in group:
                    text = str(item or "").strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    output.append(text)
            return output

        if resume_requested and resume_source_run_id:
            try:
                resume_state = checkpoint_runner.load_checkpoint(resume_source_run_id)
                subtask_results = (
                    resume_state.subtask_results
                    if isinstance(resume_state.subtask_results, dict)
                    else {}
                )
                restored_ids = []
                for key in subtask_results.keys():
                    key_str = str(key)
                    if key_str.startswith("subtask_"):
                        try:
                            restored_ids.append(int(key_str.split("_", 1)[1]))
                        except Exception:
                            pass
                if restored_ids:
                    self.subtask_counter = max(self.subtask_counter, max(restored_ids))
                all_evidence = (
                    [item for item in resume_state.evidence if isinstance(item, dict)]
                    if isinstance(resume_state.evidence, list)
                    else []
                )
                executed_plan = (
                    [item for item in resume_state.completed_actions if isinstance(item, dict)]
                    if isinstance(resume_state.completed_actions, list)
                    else []
                )
                audit_rounds = (
                    [item for item in resume_state.audit_rounds if isinstance(item, dict)]
                    if isinstance(resume_state.audit_rounds, list)
                    else []
                )
                latest_audit_challenges = (
                    [
                        item
                        for item in resume_state.latest_audit_challenges
                        if isinstance(item, dict)
                    ]
                    if isinstance(resume_state.latest_audit_challenges, list)
                    else []
                )
                all_audit_challenges = list(latest_audit_challenges)
                final_payload = (
                    resume_state.final_payload
                    if isinstance(resume_state.final_payload, dict)
                    else {}
                )
                has_research = bool(resume_state.has_research)
                has_audit = bool(resume_state.has_audit)
                last_audit_approved = resume_state.last_audit_approved
                replan_attempts = max(0, int(resume_state.cursor.round))
                checkpoint_history = (
                    resume_state.gemini_history
                    if isinstance(resume_state.gemini_history, list)
                    else []
                )
                checkpoint_wait = resume_state.wait
                self.update_beliefs(
                    {
                        "marathon_resume": True,
                        "resume_source_run_id": resume_source_run_id,
                        "restored_step": int(resume_state.cursor.step),
                    },
                    reason="Loaded checkpoint state for resume",
                )
            except Exception as resume_error:
                checkpoint_warnings.append(_short_warning("checkpoint_read_failed", resume_error))
                fallback = checkpoint_runner.build_fallback_resume_context(resume_source_run_id)
                summary = str(fallback.get("summary", "") or "").strip()
                if summary:
                    checkpoint_history = [
                        {
                            "role": "user",
                            "parts": [{"text": f"Task: {description}"}],
                        },
                        {
                            "role": "model",
                            "parts": [{"text": f"Previous run summary:\n{summary}"}],
                        },
                    ]
                self.update_beliefs(
                    {
                        "marathon_resume": True,
                        "resume_source_run_id": resume_source_run_id,
                        "checkpoint_read_failed": True,
                    },
                    reason="Checkpoint load failed; fallback context prepared",
                )

        def save_checkpoint_state(
            *,
            step_index: int,
            status: str,
            pending_actions: list[str],
            history: list[Any],
            wait: Optional[WaitDirective] = None,
        ) -> None:
            nonlocal checkpoint_path, checkpoint_warnings, checkpoint_history, checkpoint_wait
            if not marathon_enabled or self.tracer is None:
                return
            run_id = str(getattr(self.tracer, "run_id", "") or "")
            if not run_id:
                return

            checkpoint_history = history if isinstance(history, list) else checkpoint_history
            checkpoint_wait = wait or checkpoint_wait
            checkpoint_state = CheckpointState(
                run_id=run_id,
                source_run_id=resume_source_run_id if resume_requested else "",
                task_description=description,
                mode=mode_cfg.mode,
                model=model,
                thinking_level=thinking_level,
                status=status if status in {"running", "waiting", "completed"} else "running",
                cursor={"round": replan_attempts, "step": max(0, int(step_index))},
                pending_actions=[str(a) for a in pending_actions if str(a).strip()],
                completed_actions=executed_plan,
                subtask_results=subtask_results,
                evidence=all_evidence,
                audit_rounds=audit_rounds,
                latest_audit_challenges=latest_audit_challenges,
                has_research=has_research,
                has_audit=has_audit,
                last_audit_approved=last_audit_approved,
                final_payload=final_payload,
                gemini_history=checkpoint_history,
                wait=checkpoint_wait,
                next_run_at=checkpoint_wait.next_run_at if checkpoint_wait else "",
                wait_reason=checkpoint_wait.reason if checkpoint_wait else "",
                warnings=checkpoint_warnings,
            )
            try:
                checkpoint_path = str(checkpoint_runner.save_checkpoint(checkpoint_state))
            except Exception as checkpoint_error:
                warning = _short_warning("checkpoint_write_failed", checkpoint_error)
                if warning not in checkpoint_warnings:
                    checkpoint_warnings.append(warning)

        def delegate_research(args: dict) -> dict:
            nonlocal has_research
            query = str(args.get("query", "")).strip() or self._extract_search_query(description)
            try:
                top_k = int(args.get("top_k", 5))
            except (TypeError, ValueError):
                top_k = 5
            top_k = max(1, min(top_k, 20))

            subtask_id = self._next_subtask_id()
            subtask = {
                "id": subtask_id,
                "type": "research",
                "description": f"LLM research: {query}",
                "query": query,
                "top_k": top_k,
            }
            executed_plan.append({"action": "delegate_research", "subtask_id": subtask_id, "query": query})

            result = self._execute_subtask_via_contract(subtask, workers)
            subtask_results[subtask_id] = result
            evidence = result.get("evidence", [])
            all_evidence.extend(evidence)
            has_research = True

            return {
                "subtask_id": subtask_id,
                "status": "success" if result.get("success") else "error",
                "summary": self._short_output(result.get("output", result.get("error", ""))),
                "evidence_anchors": evidence,
            }

        def delegate_execute(args: dict) -> dict:
            python_code = str(args.get("python_code", "")).strip()
            purpose = str(args.get("purpose", "")).strip()
            if not python_code:
                python_code = self._generate_computation_code(description, context)

            subtask_id = self._next_subtask_id()
            subtask = {
                "id": subtask_id,
                "type": "execute",
                "description": purpose or f"LLM execute for: {description}",
                "code": python_code,
            }
            executed_plan.append({"action": "delegate_execute", "subtask_id": subtask_id, "purpose": purpose})

            result = self._execute_subtask_via_contract(subtask, workers)
            subtask_results[subtask_id] = result
            evidence = result.get("evidence", [])
            all_evidence.extend(evidence)

            return {
                "subtask_id": subtask_id,
                "status": "success" if result.get("success") else "error",
                "output": self._short_output(result.get("output", result.get("error", ""))),
                "artifacts": result.get("artifacts", {}),
                "evidence_anchors": evidence,
            }

        def delegate_audit(args: dict) -> dict:
            nonlocal has_audit, last_audit_approved, replan_attempts, latest_audit_challenges
            targets = args.get("target_subtasks", [])
            if not isinstance(targets, list) or not targets:
                targets = list(subtask_results.keys())
            targets = [str(t) for t in targets if str(t) in subtask_results]

            audit_input_results = {sid: subtask_results[sid] for sid in targets}
            policy = args.get("policy", {})
            mode_arg = str(args.get("mode", mode))
            mode_arg_cfg = resolve_mode_config(mode_arg)

            if not isinstance(policy, dict):
                policy = {"value": str(policy)}
            policy = {
                "audit_level": mode_arg_cfg.audit_level,
                "min_evidence_count": 1 if mode_arg_cfg.mode == "owl_lite" else (2 if mode_arg_cfg.mode == "owl_dl" else 3),
                "min_relevance_score": 0.15 if mode_arg_cfg.mode == "owl_lite" else (0.30 if mode_arg_cfg.mode == "owl_dl" else 0.35),
                "min_confidence": 0.45 if mode_arg_cfg.mode == "owl_lite" else (0.65 if mode_arg_cfg.mode == "owl_dl" else 0.70),
                "argumentation_required": mode_arg_cfg.argumentation,
                **policy,
            }

            subtask_id = self._next_subtask_id()
            subtask = {
                "id": subtask_id,
                "type": "audit",
                "description": f"LLM audit ({mode_arg})",
                "target_subtasks": targets,
                "results": audit_input_results,
                "policy": policy,
                "mode": mode_arg_cfg.mode,
            }
            executed_plan.append({"action": "delegate_audit", "subtask_id": subtask_id, "targets": targets})

            result = self._execute_subtask_via_contract(subtask, workers)
            subtask_results[subtask_id] = result
            has_audit = True
            last_audit_approved = bool(result.get("approved", False))

            challenges = self._coerce_audit_challenges(result.get("challenges", []))
            latest_audit_challenges = list(challenges)
            audit_rounds.append(
                {
                    "round": len(audit_rounds),
                    "audit_subtask_id": subtask_id,
                    "approved": bool(result.get("approved", False)),
                    "challenges": challenges,
                }
            )
            if challenges:
                all_audit_challenges.extend(challenges)
                if mode_cfg.require_audit and replan_attempts < max_replans:
                    replan_attempts += 1

            suggestions = []
            audit_details = result.get("audit_details", {})
            if isinstance(audit_details, dict):
                for target_id, detail in audit_details.items():
                    if not isinstance(detail, dict):
                        continue
                    recommendation = detail.get("recommendation")
                    if recommendation and recommendation != "Approved":
                        suggestions.append({"target": target_id, "recommendation": recommendation})

            structured_audit = self._normalize_audit_schema(
                task_description=description,
                mode=mode_arg_cfg.mode,
                raw_audit_result=result,
                model=model,
                thinking_level=thinking_level,
            )
            if isinstance(structured_audit, dict):
                # Prefer schema-normalized challenges/fix_suggestions for downstream display.
                schema_challenges = structured_audit.get("challenges")
                if isinstance(schema_challenges, list):
                    challenges = schema_challenges
                schema_fixes = structured_audit.get("fix_suggestions")
                if isinstance(schema_fixes, list):
                    for fix in schema_fixes:
                        if isinstance(fix, str) and fix.strip():
                            suggestions.append({"target": "schema", "recommendation": fix.strip()})

            return {
                "subtask_id": subtask_id,
                "approved": bool(result.get("approved", False)),
                "replan_required": bool(challenges) and mode_cfg.require_audit and replan_attempts <= max_replans,
                "challenges": challenges,
                "suggestions": suggestions,
                "summary": self._short_output(result.get("output", "")),
                "audit_level": mode_arg_cfg.audit_level,
                "structured_audit": structured_audit,
            }

        def request_more_evidence(args: dict) -> dict:
            # Optional helper, mapped to another research delegation.
            query = str(args.get("query", "")).strip() or self._extract_search_query(description)
            return delegate_research({"query": query, "top_k": args.get("top_k", 5)})

        def wait_seconds(args: dict) -> dict:
            nonlocal checkpoint_wait
            try:
                seconds = int(args.get("seconds", 60))
            except (TypeError, ValueError):
                seconds = 60
            seconds = max(1, min(seconds, 7 * 24 * 3600))
            reason = str(args.get("reason", "") or "").strip() or "Planner requested wait."
            next_run_at = (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
            checkpoint_wait = WaitDirective(
                wait_type="wait_seconds",
                reason=reason,
                seconds=seconds,
                until_iso="",
                next_run_at=next_run_at,
            )
            executed_plan.append(
                {
                    "action": "wait_seconds",
                    "seconds": seconds,
                    "reason": reason,
                    "next_run_at": next_run_at,
                }
            )
            return {
                "status": "waiting",
                "wait_requested": True,
                "wait_type": "wait_seconds",
                "seconds": seconds,
                "reason": reason,
                "next_run_at": next_run_at,
            }

        def wait_until(args: dict) -> dict:
            nonlocal checkpoint_wait
            raw_iso = str(args.get("iso_datetime", "") or args.get("until", "")).strip()
            reason = str(args.get("reason", "") or "").strip() or "Planner requested wait."
            if not raw_iso:
                return {
                    "status": "error",
                    "wait_requested": False,
                    "error": "iso_datetime is required",
                }
            try:
                normalized = raw_iso.replace("Z", "+00:00")
                target_dt = datetime.fromisoformat(normalized)
                if target_dt.tzinfo is None:
                    target_dt = target_dt.replace(tzinfo=timezone.utc)
                next_run_at = target_dt.astimezone(timezone.utc).isoformat()
            except Exception as parse_error:
                return {
                    "status": "error",
                    "wait_requested": False,
                    "error": f"invalid iso_datetime: {parse_error}",
                }

            checkpoint_wait = WaitDirective(
                wait_type="wait_until",
                reason=reason,
                seconds=0,
                until_iso=raw_iso,
                next_run_at=next_run_at,
            )
            executed_plan.append(
                {
                    "action": "wait_until",
                    "iso_datetime": raw_iso,
                    "reason": reason,
                    "next_run_at": next_run_at,
                }
            )
            return {
                "status": "waiting",
                "wait_requested": True,
                "wait_type": "wait_until",
                "iso_datetime": raw_iso,
                "reason": reason,
                "next_run_at": next_run_at,
            }

        def finalize(args: dict) -> dict:
            nonlocal final_payload
            answer_markdown = str(args.get("answer_markdown", "")).strip()
            claims = args.get("claims", [])
            evidence_used = args.get("evidence_used", [])
            confidence = args.get("confidence", None)
            uncertainty = args.get("uncertainty", [])

            if not isinstance(claims, list):
                claims = [claims]
            if not isinstance(evidence_used, list):
                evidence_used = [evidence_used]
            if not isinstance(uncertainty, list):
                uncertainty = [str(uncertainty)] if uncertainty is not None else []

            blockers = []
            warnings = []
            if mode_cfg.require_audit and not has_audit and subtask_results:
                try:
                    auto_targets = [
                        sid
                        for sid, result in subtask_results.items()
                        if not (isinstance(result, dict) and "approved" in result and "audit_details" in result)
                    ]
                    delegate_audit(
                        {
                            "target_subtasks": auto_targets,
                            "mode": mode_cfg.mode,
                            "policy": {"reason": "auto_audit_before_finalize"},
                        }
                    )
                    warnings.append("Audit was auto-triggered during finalize to satisfy mode policy.")
                except Exception as auto_audit_error:
                    blockers.append(f"Audit auto-trigger failed: {auto_audit_error}")

            if mode_cfg.require_audit and not has_audit:
                blockers.append("Audit is required in this mode before finalize.")
            if not has_research:
                warnings.append("No research subtask was completed.")
            if len(all_evidence) < mode_cfg.min_evidence_per_claim:
                warnings.append(
                    f"Insufficient evidence anchors collected ({len(all_evidence)} < {mode_cfg.min_evidence_per_claim})."
                )
            if mode_cfg.require_audit and has_audit and not bool(last_audit_approved):
                warnings.append(
                    "Latest audit did not approve all checks; finalizing with uncertainty and outstanding challenges."
                )
            if mode_cfg.argumentation:
                if not claims:
                    warnings.append("OWL Full prefers structured claims with counter-claims.")
                elif not self._has_counter_claim(claims):
                    warnings.append("OWL Full expects at least one counter-claim/attack relation.")

            if blockers:
                return {
                    "status": "error",
                    "finalized": False,
                    "issues": blockers,
                    "mode": mode_cfg.mode,
                    "next_action": "Run more research/execute/audit before finalizing.",
                }

            if not answer_markdown:
                answer_markdown = self._synthesize_answer(description, subtask_results)

            header = (
                f"Mode: {mode_cfg.label} | Model: {model} | Thinking: {thinking_level}\n\n"
                f"Evidence Rules: {mode_cfg.evidence_rules}\n"
            )
            if not answer_markdown.startswith("Mode: "):
                answer_markdown = header + "\n" + answer_markdown

            if confidence is None:
                confidence = 0.85 if bool(last_audit_approved) else 0.55
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.55
            confidence = max(0.0, min(1.0, confidence))
            if not uncertainty and latest_audit_challenges:
                uncertainty = [
                    str(ch.get("reason", ""))
                    for ch in latest_audit_challenges
                    if isinstance(ch, dict) and ch.get("reason")
                ]
            if warnings:
                uncertainty.extend(warnings)
            uncertainty = self._coerce_uncertainty_list(uncertainty)

            structured_finalize = self._normalize_finalize_schema(
                answer_markdown=answer_markdown,
                claims=claims,
                evidence_used=evidence_used,
                confidence=confidence,
                uncertainty=uncertainty,
                model=model,
                thinking_level=thinking_level,
            )
            normalized_answer = str(structured_finalize.get("final_answer_markdown", answer_markdown))
            normalized_claims = structured_finalize.get("claims", claims)
            normalized_evidence = structured_finalize.get("evidence_used", evidence_used)
            normalized_confidence = structured_finalize.get("confidence", confidence)
            normalized_uncertainty = self._coerce_uncertainty_list(
                structured_finalize.get("uncertainty", uncertainty)
            )
            uncertainty_details = self._build_uncertainty_details(
                raw_uncertainty=normalized_uncertainty,
                challenges=latest_audit_challenges if not bool(last_audit_approved) else [],
            )
            outstanding_challenges = latest_audit_challenges if not bool(last_audit_approved) else []

            final_payload = {
                "final_answer": normalized_answer,
                "claims": normalized_claims,
                "evidence_used": normalized_evidence,
                "confidence": normalized_confidence,
                "uncertainty": normalized_uncertainty,
                "uncertainty_details": uncertainty_details,
                "warnings": warnings,
                "outstanding_challenges": outstanding_challenges,
                "mode": mode_cfg.mode,
                "model": model,
                "thinking_level": thinking_level,
                "audit_level": mode_cfg.audit_level,
                "argumentation": mode_cfg.argumentation,
                "structured_finalize": structured_finalize,
            }
            executed_plan.append({"action": "finalize"})
            return {"status": "success", "finalized": True, **final_payload}

        handlers = {
            "delegate_research": delegate_research,
            "delegate_execute": delegate_execute,
            "delegate_audit": delegate_audit,
            "request_more_evidence": request_more_evidence,
            "wait_seconds": wait_seconds,
            "wait_until": wait_until,
            "finalize": finalize,
        }

        system_prompt = self._build_llm_system_prompt(mode_cfg)
        user_prompt = (
            f"Task: {description}\n\n"
            f"Available workers: {[w.agent_id for w in workers]}\n"
            "You must use tools to gather evidence, compute if needed, audit the result, "
            "then call finalize. For long-horizon work you may call wait_seconds/wait_until to pause."
        )

        latest_history_snapshot = checkpoint_history if isinstance(checkpoint_history, list) else []

        def on_llm_step(step: dict):
            nonlocal latest_history_snapshot
            self._trace_llm_step(step, tool_schema_names)
            history = step.get("history")
            if isinstance(history, list):
                latest_history_snapshot = history
            function_calls = step.get("function_calls", [])
            pending_actions = [
                str(fc.get("name", "")).strip()
                for fc in function_calls
                if isinstance(fc, dict) and str(fc.get("name", "")).strip()
            ]
            save_checkpoint_state(
                step_index=int(step.get("step", 0) or 0),
                status="running",
                pending_actions=pending_actions,
                history=latest_history_snapshot,
            )

        loop_result = self.llm_client.tool_call_chat_loop(
            user_prompt=user_prompt,
            model=model,
            thinking_level=thinking_level,
            tools_schema=tools_schema,
            tool_handlers=handlers,
            system_prompt=system_prompt,
            max_steps=max_steps,
            on_step=on_llm_step,
            use_cache=llm_use_cache,
            initial_history=checkpoint_history if resume_requested and checkpoint_history else None,
        )

        waiting_requested = bool(loop_result.get("waiting_requested", False))
        wait_result = loop_result.get("wait_result", {}) if isinstance(loop_result.get("wait_result"), dict) else {}
        loop_history = loop_result.get("history") if isinstance(loop_result.get("history"), list) else latest_history_snapshot
        unresolved_challenges = latest_audit_challenges if has_audit and not bool(last_audit_approved) else []

        if waiting_requested:
            wait_next_run_at = str(
                wait_result.get("next_run_at", checkpoint_wait.next_run_at if checkpoint_wait else "")
                or ""
            )
            wait_reason = str(
                wait_result.get("reason", checkpoint_wait.reason if checkpoint_wait else "")
                or ""
            )
            wait_type = str(
                wait_result.get("wait_type", checkpoint_wait.wait_type if checkpoint_wait else "wait_seconds")
                or "wait_seconds"
            )
            if checkpoint_wait is None:
                checkpoint_wait = WaitDirective(
                    wait_type="wait_until" if wait_type == "wait_until" else "wait_seconds",
                    reason=wait_reason,
                    seconds=int(wait_result.get("seconds", 0) or 0),
                    until_iso=str(wait_result.get("iso_datetime", "") or ""),
                    next_run_at=wait_next_run_at,
                )
            save_checkpoint_state(
                step_index=len(loop_result.get("steps", [])),
                status="waiting",
                pending_actions=[],
                history=loop_history,
                wait=checkpoint_wait,
            )

            partial_answer = final_payload.get("final_answer") or self._synthesize_answer(description, subtask_results)
            header = (
                f"Mode: {mode_cfg.label} | Model: {model} | Thinking: {thinking_level}\n\n"
                f"Evidence Rules: {mode_cfg.evidence_rules}\n"
            )
            if not partial_answer.startswith("Mode: "):
                partial_answer = header + "\n" + partial_answer
            partial_answer += (
                "\n\n## Marathon Wait\n"
                f"- Status: waiting\n"
                f"- Next run at: {wait_next_run_at or 'unspecified'}\n"
                f"- Reason: {wait_reason or 'Planner requested wait'}\n"
            )
            waiting_warnings = _merge_warnings(final_payload.get("warnings", []), checkpoint_warnings)

            return {
                "success": True,
                "run_status": "waiting",
                "output": partial_answer,
                "evidence": all_evidence,
                "plan": executed_plan,
                "subtask_results": subtask_results,
                "audit_challenges": unresolved_challenges,
                "all_audit_challenges": all_audit_challenges,
                "audit_rounds": audit_rounds,
                "audit_notes": self._build_audit_notes(audit_rounds),
                "uncertainty": bool(unresolved_challenges),
                "structured_plan": structured_plan,
                "claims": final_payload.get("claims", []),
                "evidence_used": final_payload.get("evidence_used", []),
                "confidence": final_payload.get("confidence"),
                "uncertainty_details": final_payload.get("uncertainty_details", []),
                "warnings": waiting_warnings,
                "outstanding_challenges": final_payload.get("outstanding_challenges", unresolved_challenges),
                "structured_finalize": final_payload.get("structured_finalize", {}),
                "finalize_missing": False,
                "llm_finalize_called": False,
                "checkpoint_path": checkpoint_path,
                "next_run_at": wait_next_run_at,
                "wait_reason": wait_reason,
                "llm_loop": {
                    "steps": len(loop_result.get("steps", [])),
                    "ended_reason": loop_result.get("ended_reason", "waiting_requested"),
                    "finalize_called": False,
                    "force_finalize_attempted": bool(loop_result.get("force_finalize_attempted", False)),
                    "waiting_requested": True,
                },
            }

        final_answer = final_payload.get("final_answer")
        if not final_answer:
            if loop_result.get("final_text"):
                final_answer = loop_result["final_text"]
            else:
                final_answer = self._synthesize_answer(description, subtask_results)

        finalize_called = bool(loop_result.get("finalize_called")) or bool(final_payload.get("final_answer"))
        finalize_missing = bool(mode_cfg.require_finalize and not finalize_called)
        save_checkpoint_state(
            step_index=len(loop_result.get("steps", [])),
            status="completed" if finalize_called else "running",
            pending_actions=[],
            history=loop_history,
            wait=None,
        )

        non_audit_results = [
            r
            for r in subtask_results.values()
            if not (isinstance(r, dict) and "approved" in r and "audit_details" in r)
        ]
        if non_audit_results:
            non_audit_success = all(r.get("success", False) for r in non_audit_results)
        else:
            non_audit_success = bool(final_payload.get("final_answer"))
        success = non_audit_success and (has_audit if mode_cfg.require_audit else True)
        if finalize_missing:
            success = False

        uncertainty_details = final_payload.get("uncertainty_details", [])
        if unresolved_challenges and not uncertainty_details:
            uncertainty_details = self._build_uncertainty_details([], unresolved_challenges)
        uncertainty = bool(unresolved_challenges or uncertainty_details)
        if uncertainty and not final_payload.get("uncertainty"):
            final_payload["uncertainty"] = [d.get("detail", "") for d in uncertainty_details if d.get("detail")]
        warnings = _merge_warnings(final_payload.get("warnings", []), checkpoint_warnings)
        run_status = (
            "completed_with_warnings"
            if (success and (uncertainty or bool(warnings)))
            else ("completed" if success else "failed")
        )

        return {
            "success": success,
            "run_status": run_status,
            "output": final_answer,
            "evidence": all_evidence,
            "plan": executed_plan,
            "subtask_results": subtask_results,
            "audit_challenges": unresolved_challenges,
            "all_audit_challenges": all_audit_challenges,
            "audit_rounds": audit_rounds,
            "audit_notes": self._build_audit_notes(audit_rounds),
            "uncertainty": uncertainty,
            "structured_plan": structured_plan,
            "claims": final_payload.get("claims", []),
            "evidence_used": final_payload.get("evidence_used", []),
            "confidence": final_payload.get("confidence"),
            "uncertainty_details": uncertainty_details,
            "warnings": warnings,
            "outstanding_challenges": final_payload.get("outstanding_challenges", unresolved_challenges),
            "structured_finalize": final_payload.get("structured_finalize", {}),
            "finalize_missing": finalize_missing,
            "llm_finalize_called": finalize_called,
            "checkpoint_path": checkpoint_path,
            "next_run_at": "",
            "wait_reason": "",
            "llm_loop": {
                "steps": len(loop_result.get("steps", [])),
                "ended_reason": loop_result.get("ended_reason", ""),
                "finalize_called": finalize_called,
                "force_finalize_attempted": bool(loop_result.get("force_finalize_attempted", False)),
                "waiting_requested": False,
            },
        }

    def _process_rule_based(self, task: dict) -> dict:
        """Deterministic planner with audit-driven replan loop."""
        description = task["description"]
        context = task.get("context", {})
        workers = task.get("workers", [])
        mode_cfg = resolve_mode_config(context.get("owl_mode", "owl_lite"))
        max_replans = int(context.get("max_replans", mode_cfg.max_replans))
        max_replans = max(0, min(max_replans, 4))

        base_subtasks = self._decompose_task(description, context)
        self.set_intentions([{"action": "execute_subtask", "subtask": st} for st in base_subtasks])

        results: dict[str, dict] = {}
        all_evidence: list = []
        executed_plan: list[dict] = []
        audit_rounds: list[dict] = []
        unresolved_challenges: list[dict] = []
        replan_count = 0
        latest_audit_id = ""

        # Round 0: execute initial non-audit subtasks, then audit.
        initial_non_audit = [st for st in base_subtasks if st.get("type") != "audit"]
        for subtask in initial_non_audit:
            subtask_result = self._execute_subtask_via_contract(subtask, workers)
            results[subtask["id"]] = subtask_result
            all_evidence.extend(subtask_result.get("evidence", []))
            executed_plan.append({"action": "execute_subtask", "subtask_id": subtask["id"], "type": subtask.get("type", "")})

        audit_result, audit_subtask = self._run_audit_subtask(
            description=description,
            workers=workers,
            mode_cfg=mode_cfg,
            current_results=results,
        )
        latest_audit_id = audit_subtask["id"]
        results[audit_subtask["id"]] = audit_result
        executed_plan.append({"action": "execute_subtask", "subtask_id": audit_subtask["id"], "type": "audit"})
        unresolved_challenges = list(audit_result.get("challenges", []) or [])
        audit_rounds.append(
            {
                "round": 0,
                "audit_subtask_id": audit_subtask["id"],
                "approved": bool(audit_result.get("approved", False)),
                "challenges": unresolved_challenges,
            }
        )

        # Replan rounds: challenge -> new subtasks -> audit again.
        while unresolved_challenges and replan_count < max_replans:
            replan_count += 1
            new_subtasks = self._build_replan_subtasks(
                challenges=unresolved_challenges,
                description=description,
                context=context,
            )
            if not new_subtasks:
                break

            if self.tracer:
                self.tracer.log_replan(
                    self.agent_id,
                    reason="; ".join(str(c.get("reason", "")) for c in unresolved_challenges if isinstance(c, dict)) or "Audit challenges",
                    new_plan=[
                        {"action": "execute_subtask", "subtask_id": st["id"], "type": st.get("type", "")}
                        for st in new_subtasks
                    ],
                )

            self.set_intentions([{"action": "execute_subtask", "subtask": st} for st in new_subtasks])
            for subtask in new_subtasks:
                subtask_result = self._execute_subtask_via_contract(subtask, workers)
                results[subtask["id"]] = subtask_result
                all_evidence.extend(subtask_result.get("evidence", []))
                executed_plan.append({"action": "execute_subtask", "subtask_id": subtask["id"], "type": subtask.get("type", "")})

            audit_result, audit_subtask = self._run_audit_subtask(
                description=f"{description} (replan round {replan_count})",
                workers=workers,
                mode_cfg=mode_cfg,
                current_results=results,
            )
            latest_audit_id = audit_subtask["id"]
            results[audit_subtask["id"]] = audit_result
            executed_plan.append({"action": "execute_subtask", "subtask_id": audit_subtask["id"], "type": "audit"})
            unresolved_challenges = list(audit_result.get("challenges", []) or [])
            audit_rounds.append(
                {
                    "round": replan_count,
                    "audit_subtask_id": audit_subtask["id"],
                    "approved": bool(audit_result.get("approved", False)),
                    "challenges": unresolved_challenges,
                }
            )

            if audit_result.get("approved", False):
                break

        final_answer = self._synthesize_answer(description, results)

        # Only evaluate success on non-audit subtasks + latest audit gate when required.
        non_audit_results = [
            r
            for sid, r in results.items()
            if sid != latest_audit_id and not (isinstance(r, dict) and "approved" in r and "audit_details" in r)
        ]
        non_audit_success = all(r.get("success", False) for r in non_audit_results) if non_audit_results else False
        final_audit_present = bool(latest_audit_id and latest_audit_id in results)
        success = non_audit_success and (final_audit_present if mode_cfg.require_audit else True)
        uncertainty = bool(unresolved_challenges)
        uncertainty_details = self._build_uncertainty_details([], unresolved_challenges)
        run_status = "completed_with_warnings" if (success and uncertainty) else ("completed" if success else "failed")

        return {
            "success": success,
            "run_status": run_status,
            "output": final_answer,
            "evidence": all_evidence,
            "plan": base_subtasks,
            "executed_plan": executed_plan,
            "subtask_results": results,
            "audit_challenges": unresolved_challenges,
            "audit_rounds": audit_rounds,
            "audit_notes": self._build_audit_notes(audit_rounds),
            "uncertainty": uncertainty,
            "confidence": 0.85 if not uncertainty else 0.55,
            "uncertainty_details": uncertainty_details,
            "warnings": [d.get("detail", "") for d in uncertainty_details if d.get("detail")],
            "outstanding_challenges": unresolved_challenges,
        }

    def _run_audit_subtask(
        self,
        description: str,
        workers: list,
        mode_cfg: ModeConfig,
        current_results: dict[str, dict],
    ) -> tuple[dict, dict]:
        """Run one auditor pass over all non-audit subtask results."""
        target_subtasks = [
            sid
            for sid, result in current_results.items()
            if not (isinstance(result, dict) and "approved" in result and "audit_details" in result)
        ]
        audit_subtask = {
            "id": self._next_subtask_id(),
            "type": "audit",
            "description": f"Validate findings and check evidence quality: {description}",
            "target_subtasks": target_subtasks,
            "results": {sid: current_results[sid] for sid in target_subtasks if sid in current_results},
            "mode": mode_cfg.mode,
            "policy": {
                "audit_level": mode_cfg.audit_level,
                "argumentation_required": mode_cfg.argumentation,
                "min_evidence_count": 1 if mode_cfg.mode == "owl_lite" else (2 if mode_cfg.mode == "owl_dl" else 3),
                "min_relevance_score": 0.15 if mode_cfg.mode == "owl_lite" else (0.30 if mode_cfg.mode == "owl_dl" else 0.35),
                "min_confidence": 0.45 if mode_cfg.mode == "owl_lite" else (0.65 if mode_cfg.mode == "owl_dl" else 0.70),
            },
        }
        audit_result = self._execute_subtask_via_contract(audit_subtask, workers)
        return audit_result, audit_subtask

    def _build_replan_subtasks(self, challenges: list, description: str, context: dict) -> list[dict]:
        """Generate targeted follow-up subtasks from audit challenges."""
        subtasks: list[dict] = []
        seen_queries: set[str] = set()
        reused_target_ids: set[str] = set()
        execute_added = False

        for challenge in challenges or []:
            if not isinstance(challenge, dict):
                continue
            reason = str(challenge.get("reason", ""))
            suggested_fix_query = str(challenge.get("suggested_fix_query", "")).strip()
            suggested_calc = str(challenge.get("suggested_calc", "")).strip()
            target_id = str(challenge.get("target_subtask_id", challenge.get("target_subtask", ""))).strip()
            reusable_target_id = ""
            if target_id.startswith("subtask_") and target_id not in reused_target_ids:
                reusable_target_id = target_id

            if suggested_fix_query and suggested_fix_query not in seen_queries:
                seen_queries.add(suggested_fix_query)
                subtask_id = reusable_target_id or self._next_subtask_id()
                if reusable_target_id:
                    reused_target_ids.add(reusable_target_id)
                    reusable_target_id = ""
                subtasks.append(
                    {
                        "id": subtask_id,
                        "type": "research",
                        "description": f"Replan evidence collection for {target_id or 'task'}",
                        "query": suggested_fix_query,
                    }
                )

            needs_execute = (
                bool(suggested_calc)
                or "unit" in reason.lower()
                or "inconsistent" in reason.lower()
                or "recompute" in reason.lower()
            )
            if needs_execute and not execute_added:
                subtask_id = reusable_target_id or self._next_subtask_id()
                if reusable_target_id:
                    reused_target_ids.add(reusable_target_id)
                subtasks.append(
                    {
                        "id": subtask_id,
                        "type": "execute",
                        "description": f"Replan computation check for {target_id or 'task'}",
                        "code": self._generate_computation_code(description, context),
                    }
                )
                execute_added = True

        if not subtasks:
            fallback_query = f"{self._extract_search_query(description)} uncertainty evidence consistency"
            subtasks.append(
                {
                    "id": self._next_subtask_id(),
                    "type": "research",
                    "description": "Replan fallback evidence collection",
                    "query": fallback_query,
                }
            )
        return subtasks

    def _build_audit_notes(self, audit_rounds: list[dict]) -> str:
        """Summarize audit-replan lifecycle for final answer rendering."""
        if not audit_rounds:
            return ""
        lines = []
        for item in audit_rounds:
            round_idx = item.get("round", 0)
            approved = bool(item.get("approved", False))
            challenge_count = len(item.get("challenges", []) or [])
            status = "approved" if approved else "challenged"
            lines.append(f"Round {round_idx}: {status} ({challenge_count} challenge(s))")
        if audit_rounds and not audit_rounds[-1].get("approved", False):
            lines.append("Uncertainty remains after max replans.")
        return "; ".join(lines)

    def _execute_subtask_via_contract(self, subtask: dict, workers: list) -> dict:
        """Run one subtask end-to-end through Contract Net."""
        task_id = subtask.get("id", self._next_subtask_id())
        subtask = dict(subtask)
        subtask["id"] = task_id

        announcement = TaskAnnouncement(
            task_id=task_id,
            task_type=subtask.get("type", ""),
            description=subtask.get("description", ""),
            requirements=subtask.get("requirements", {}),
        )
        self.contract_net.announce_task(announcement)
        if self.tracer:
            self.tracer.log_cfp(self.agent_id, task_id, subtask.get("type", ""), subtask.get("description", ""))

        bids = self._collect_bids(subtask, workers)
        if not bids:
            return {"success": False, "error": f"No worker available for task type: {subtask.get('type', '')}"}

        for bid in bids:
            self.contract_net.submit_bid(bid)
            if self.tracer:
                self.tracer.log_bid(
                    bid.bidder,
                    task_id,
                    bid.estimated_cost,
                    bid.estimated_latency,
                    bid.success_probability,
                )

        winner = self.contract_net.evaluate_bids(task_id)
        if winner is None:
            return {"success": False, "error": "No winning bid could be selected"}

        self.contract_net.award_contract(task_id, winner)
        if self.tracer:
            self.tracer.log_contract_award(self.agent_id, winner.bidder, task_id)

        winner_agent = next((w for w in workers if w.agent_id == winner.bidder), None)
        if winner_agent is None:
            return {"success": False, "error": f"Winning agent not found: {winner.bidder}"}

        self.send_message(
            receiver=winner.bidder,
            performative=Performative.REQUEST,
            content=subtask,
            protocol="contract-net",
        )

        self.contract_net.start_execution(task_id)
        result = winner_agent.process(subtask)

        if result.get("success"):
            self.contract_net.complete_task(task_id, result.get("output"), result.get("evidence", []))
        else:
            self.contract_net.fail_task(task_id, result.get("error", "Unknown error"))

        self.update_beliefs(
            {
                "completed_subtasks": sorted(
                    [
                        tid
                        for tid, status in self.contract_net.task_status.items()
                        if status.value == "completed"
                    ]
                ),
                f"result_{task_id}": self._short_output(result.get("output", result.get("error", ""))),
            },
            reason=f"Subtask {task_id} completed",
        )
        return result

    def _build_planner_tools_schema(self) -> list[dict]:
        """Gemini function declarations for planner actions."""
        return [
            {
                "function_declarations": [
                    {
                        "name": "delegate_research",
                        "description": "Delegate literature/corpus search to the researcher agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "delegate_execute",
                        "description": "Delegate scientific computation/code execution to the executor agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "python_code": {"type": "string"},
                                "purpose": {"type": "string"},
                            },
                            "required": ["python_code"],
                        },
                    },
                    {
                        "name": "delegate_audit",
                        "description": "Delegate quality/evidence audit to the auditor agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_subtasks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "mode": {"type": "string"},
                                "policy": {"type": "object"},
                            },
                            "required": ["target_subtasks"],
                        },
                    },
                    {
                        "name": "request_more_evidence",
                        "description": "Request extra evidence collection if current support is weak.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "default": 5},
                            },
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "wait_seconds",
                        "description": "Pause workflow for a fixed duration and resume later.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "seconds": {"type": "integer", "minimum": 1},
                                "reason": {"type": "string"},
                            },
                            "required": ["seconds"],
                        },
                    },
                    {
                        "name": "wait_until",
                        "description": "Pause workflow until an ISO datetime and resume later.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "iso_datetime": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["iso_datetime"],
                        },
                    },
                    {
                        "name": "finalize",
                        "description": "Finalize answer with claims and evidence mapping.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer_markdown": {"type": "string"},
                                "claims": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                                "evidence_used": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                                "confidence": {"type": "number"},
                                "uncertainty": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["answer_markdown"],
                        },
                    },
                ]
            }
        ]

    def _build_llm_system_prompt(self, mode_cfg: ModeConfig) -> str:
        """System prompt for planner function-calling loop."""
        if mode_cfg.mode == "owl_lite":
            rule = (
                "Fast mode: you may finalize after research if evidence is sufficient. "
                "Audit is optional but recommended."
            )
        elif mode_cfg.mode == "owl_dl":
            rule = (
                "DL mode: audit is mandatory. If audit fails, replan up to max replans "
                "before finalize."
            )
        else:
            rule = (
                "Full mode: audit is mandatory and argumentation is required. "
                "Produce claims with support and at least one counter-claim."
            )
        return (
            "You are the RAR Planner in a reproducible marathon runtime.\n"
            "Use function calls to drive execution: research -> optional execute -> audit -> finalize.\n"
            "For long-horizon workflows, you may request wait_seconds/wait_until and resume later.\n"
            "When uncertainty remains, state it explicitly in the final answer.\n"
            f"Mode: {mode_cfg.label}\n"
            f"Model: {mode_cfg.model}\n"
            f"Thinking: {mode_cfg.thinking_level}\n"
            f"Evidence rule: {mode_cfg.evidence_rules}\n"
            f"Mode policy: {rule}"
        )

    def _generate_structured_plan(
        self,
        task_description: str,
        mode_cfg: ModeConfig,
        model: str,
        thinking_level: str,
    ) -> dict:
        """Generate a structured high-level plan via JSON schema."""
        if self.llm_client is None:
            return {}
        prompt = (
            "Create a compact execution plan for this AI4S task.\n"
            f"Task: {task_description}\n"
            f"Mode: {mode_cfg.mode}\n"
            "Use step types from {research, execute, audit, synthesize}. "
            "Return only JSON."
        )
        record = self.llm_client.generate_json(
            prompt=prompt,
            model=model,
            thinking_level=thinking_level,
            json_schema=PlanSchema,
            system_prompt=(
                "You are RAR planner. Produce a deterministic structured plan. "
                "Output must follow JSON schema exactly."
            ),
            schema_name="plan",
            max_schema_retries=1,
            on_schema_violation=self._build_schema_violation_callback("plan"),
        )
        self._trace_llm_step({"record": record.to_dict(), "function_calls": []}, [])
        return record.response_json if isinstance(record.response_json, dict) else {}

    def _normalize_audit_schema(
        self,
        task_description: str,
        mode: str,
        raw_audit_result: dict,
        model: str,
        thinking_level: str,
    ) -> dict:
        """Normalize audit output to AuditSchema; use Gemini JSON mode as canonicalizer."""
        normalized_challenges = self._coerce_audit_challenges(raw_audit_result.get("challenges", []))
        severity = "low"
        if normalized_challenges:
            if any(ch.get("severity") == "high" for ch in normalized_challenges):
                severity = "high"
            elif any(ch.get("severity") == "medium" for ch in normalized_challenges):
                severity = "medium"
        baseline = {
            "approved": bool(raw_audit_result.get("approved", False)),
            "challenges": normalized_challenges,
            "severity": severity,
            "fix_suggestions": self._coerce_fix_suggestions(raw_audit_result.get("fix_suggestions", [])),
        }
        if not baseline["fix_suggestions"]:
            for challenge in baseline["challenges"]:
                recommendation = challenge.get("suggested_fix_query")
                if recommendation:
                    baseline["fix_suggestions"].append(str(recommendation))
                recommendation = challenge.get("suggested_calc")
                if recommendation:
                    baseline["fix_suggestions"].append(str(recommendation))

        try:
            normalized = AuditSchema.model_validate(baseline).model_dump(exclude_none=True)
        except Exception:
            normalized = baseline

        if self.llm_client is None:
            return normalized

        prompt = (
            "Normalize the following audit outcome into strict JSON schema.\n"
            f"Task: {task_description}\n"
            f"Mode: {mode}\n"
            f"Audit result JSON: {json.dumps(raw_audit_result, ensure_ascii=False, default=str)}\n"
            "Output only JSON."
        )
        try:
            record = self.llm_client.generate_json(
                prompt=prompt,
                model=model,
                thinking_level=thinking_level,
                json_schema=AuditSchema,
                system_prompt=(
                    "You are a strict JSON normalizer. Preserve semantic fidelity to provided audit result."
                ),
                schema_name="audit",
                max_schema_retries=1,
                on_schema_violation=self._build_schema_violation_callback("audit"),
            )
            self._trace_llm_step({"record": record.to_dict(), "function_calls": []}, [])
            if isinstance(record.response_json, dict):
                return record.response_json
        except Exception:
            pass
        return normalized

    def _normalize_finalize_schema(
        self,
        answer_markdown: str,
        claims: list,
        evidence_used: list,
        confidence: float,
        uncertainty: list,
        model: str,
        thinking_level: str,
    ) -> dict:
        """Normalize finalize payload to FinalizeSchema; retries on schema violations."""
        safe_confidence = 0.0
        try:
            safe_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            safe_confidence = 0.0
        baseline = {
            "final_answer_markdown": answer_markdown,
            "claims": self._coerce_finalize_claims(claims),
            "evidence_used": self._coerce_finalize_evidence(evidence_used),
            "confidence": safe_confidence,
            "uncertainty": self._coerce_uncertainty_list(uncertainty),
        }

        try:
            normalized = FinalizeSchema.model_validate(baseline).model_dump(exclude_none=True)
        except Exception:
            normalized = baseline

        if self.llm_client is None:
            return normalized

        prompt = (
            "Normalize this finalize payload into strict JSON schema.\n"
            f"Finalize payload: {json.dumps(baseline, ensure_ascii=False, default=str)}\n"
            "Output only JSON."
        )
        try:
            record = self.llm_client.generate_json(
                prompt=prompt,
                model=model,
                thinking_level=thinking_level,
                json_schema=FinalizeSchema,
                system_prompt=(
                    "You are a strict JSON normalizer for final answers. "
                    "Do not drop key information."
                ),
                schema_name="finalize",
                max_schema_retries=1,
                on_schema_violation=self._build_schema_violation_callback("finalize"),
            )
            self._trace_llm_step({"record": record.to_dict(), "function_calls": []}, [])
            if isinstance(record.response_json, dict):
                return record.response_json
        except Exception:
            pass
        return normalized

    def _build_schema_violation_callback(self, schema_name: str):
        """Create callback for GeminiClient schema-violation events."""

        def _callback(info: dict):
            if not self.tracer:
                return
            if not isinstance(info, dict):
                return
            self.tracer.log_schema_violation(
                agent_id=self.agent_id,
                schema_name=schema_name or str(info.get("schema_name", "schema")),
                attempt=int(info.get("attempt", 1)),
                max_attempts=int(info.get("max_attempts", 1)),
                error=str(info.get("error", "")),
                cache_key=str(info.get("cache_key", "")),
                model=str(info.get("model", "")),
                thinking_level=str(info.get("thinking_level", "")),
                response_hash=str(info.get("response_hash", "")),
                response_text_preview=str(info.get("response_text_preview", "")),
            )

        return _callback

    def _coerce_audit_challenges(self, raw_challenges: Any) -> list[dict]:
        """Coerce arbitrary auditor challenges into AuditSchema-compatible entries."""
        if not isinstance(raw_challenges, list):
            raw_challenges = [raw_challenges]
        normalized: list[dict] = []
        for item in raw_challenges:
            if isinstance(item, str):
                reason = item.strip()
                if not reason:
                    continue
                normalized.append(
                    {
                        "target_subtask_id": "",
                        "reason": reason,
                        "severity": "medium",
                        "suggested_fix_query": "",
                        "suggested_calc": "",
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason", "") or "").strip()
            if not reason:
                reason = str(item.get("message", "") or "Audit issue detected").strip()
            severity = str(item.get("severity", "medium") or "medium").lower()
            if severity not in {"low", "medium", "high"}:
                severity = "medium"
            normalized.append(
                {
                    "target_subtask_id": str(
                        item.get("target_subtask_id", item.get("target_subtask", ""))
                    ).strip(),
                    "reason": reason,
                    "severity": severity,
                    "suggested_fix_query": str(item.get("suggested_fix_query", "") or "").strip(),
                    "suggested_calc": str(item.get("suggested_calc", "") or "").strip(),
                }
            )
        return normalized

    def _coerce_fix_suggestions(self, raw_suggestions: Any) -> list[str]:
        """Coerce fix suggestions to list[str]."""
        if not isinstance(raw_suggestions, list):
            raw_suggestions = [raw_suggestions]
        output = []
        for item in raw_suggestions:
            text = str(item or "").strip()
            if text:
                output.append(text)
        return output

    def _coerce_finalize_claims(self, raw_claims: Any) -> list[dict]:
        """Coerce claims to FinalizeSchema claim shape."""
        if not isinstance(raw_claims, list):
            raw_claims = [raw_claims]
        normalized: list[dict] = []
        for item in raw_claims:
            if isinstance(item, str):
                claim_text = item.strip()
                if not claim_text:
                    continue
                normalized.append(
                    {
                        "claim": claim_text,
                        "status": "supported",
                        "evidence_anchor_ids": [],
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            claim_text = str(item.get("claim", item.get("text", "")) or "").strip()
            if not claim_text:
                continue
            status = str(item.get("status", "supported") or "supported").lower()
            if status not in {"supported", "contested", "uncertain"}:
                status = "supported"
            anchors = item.get("evidence_anchor_ids", item.get("evidence_ids", []))
            if not isinstance(anchors, list):
                anchors = [anchors] if anchors else []
            normalized.append(
                {
                    "claim": claim_text,
                    "status": status,
                    "evidence_anchor_ids": [str(a) for a in anchors if str(a).strip()],
                }
            )
        return normalized

    def _coerce_finalize_evidence(self, raw_evidence: Any) -> list[dict]:
        """Coerce evidence entries to FinalizeSchema evidence shape."""
        if not isinstance(raw_evidence, list):
            raw_evidence = [raw_evidence]
        normalized: list[dict] = []
        for item in raw_evidence:
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "doc_id": "",
                        "location": "",
                        "content_hash": "",
                        "note": text,
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "doc_id": str(item.get("doc_id", "") or ""),
                    "location": str(item.get("location", "") or ""),
                    "content_hash": str(item.get("content_hash", "") or ""),
                    "note": str(
                        item.get("note", item.get("snippet", item.get("summary", ""))) or ""
                    ),
                }
            )
        return normalized

    def _coerce_uncertainty_list(self, raw_uncertainty: Any) -> list[str]:
        """Coerce uncertainty field to list[str]."""
        if not isinstance(raw_uncertainty, list):
            raw_uncertainty = [raw_uncertainty]
        values = []
        for item in raw_uncertainty:
            text = str(item or "").strip()
            if text:
                values.append(text)
        return values

    def _build_uncertainty_details(self, raw_uncertainty: Any, challenges: list[dict]) -> list[dict]:
        """Build structured uncertainty details from LLM uncertainty + audit challenges."""
        details: list[dict] = []

        for text in self._coerce_uncertainty_list(raw_uncertainty):
            details.append(
                {
                    "type": "llm_declared",
                    "detail": text,
                    "severity": "info",
                }
            )

        for challenge in self._coerce_audit_challenges(challenges):
            reason = str(challenge.get("reason", "") or "").strip()
            if not reason:
                continue
            details.append(
                {
                    "type": "audit_challenge",
                    "detail": reason,
                    "severity": challenge.get("severity", "medium"),
                    "target_subtask_id": challenge.get("target_subtask_id", ""),
                }
            )

        deduped: list[dict] = []
        seen = set()
        for item in details:
            key = (
                str(item.get("type", "")),
                str(item.get("detail", "")),
                str(item.get("target_subtask_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _trace_llm_step(self, step: dict, tool_schema_names: list[str]):
        """Emit LLM call/result events for one loop step."""
        if not self.tracer:
            return
        record = step.get("record", {})
        if not isinstance(record, dict):
            return

        request = record.get("request", {}) if isinstance(record.get("request"), dict) else {}
        config = {
            "response_format": request.get("response_format", ""),
            "schema_name": request.get("schema_name", ""),
            "has_system_prompt": bool(request.get("system_prompt")),
            "tool_schema_names": tool_schema_names,
        }

        prompt_summary = self._short_output(request.get("prompt", ""))
        if not prompt_summary and request.get("contents"):
            prompt_summary = self._short_output(request.get("contents"))

        self.tracer.log_llm_call(
            agent_id=self.agent_id,
            model=record.get("model", ""),
            thinking_level=record.get("thinking_level", ""),
            cache_key=record.get("cache_key", ""),
            prompt_summary=prompt_summary,
            tool_schema_names=tool_schema_names,
            request=request,
            config=config,
        )

        if record.get("is_cache_hit"):
            self.tracer.log_llm_cache_hit(
                agent_id=self.agent_id,
                model=record.get("model", ""),
                thinking_level=record.get("thinking_level", ""),
                cache_key=record.get("cache_key", ""),
                response_hash=record.get("response_hash", ""),
                usage=record.get("usage", {}),
                latency_ms=float(record.get("latency_ms", 0.0) or 0.0),
            )

        function_calls = step.get("function_calls", [])
        if function_calls:
            call_names = [fc.get("name", "") for fc in function_calls if isinstance(fc, dict)]
            response_summary = f"Function calls: {', '.join([n for n in call_names if n])}"
        else:
            response_summary = self._short_output(record.get("response_text", ""))

        self.tracer.log_llm_result(
            agent_id=self.agent_id,
            cache_key=record.get("cache_key", ""),
            response_text_summary=response_summary,
            response_hash=record.get("response_hash", ""),
            usage=record.get("usage", {}),
            model=record.get("model", ""),
            thinking_level=record.get("thinking_level", ""),
            deterministic=bool(record.get("is_cache_hit")),
            latency_ms=float(record.get("latency_ms", 0.0) or 0.0),
            response_text=record.get("response_text", ""),
            response_raw=record.get("response_raw"),
        )

    def _extract_tool_names(self, tools_schema: list[dict]) -> list[str]:
        names = []
        for tool in tools_schema:
            if not isinstance(tool, dict):
                continue
            declarations = tool.get("function_declarations", [])
            if not isinstance(declarations, list):
                continue
            for decl in declarations:
                if isinstance(decl, dict) and decl.get("name"):
                    names.append(str(decl["name"]))
        return names

    def _next_subtask_id(self) -> str:
        self.subtask_counter += 1
        return f"subtask_{self.subtask_counter}"

    def _decompose_task(self, description: str, context: dict) -> list:
        """Rule-based fallback decomposition."""
        subtasks = []
        mode_cfg = resolve_mode_config(context.get("owl_mode", "owl_lite"))

        subtasks.append(
            {
                "id": self._next_subtask_id(),
                "type": "research",
                "description": f"Search for relevant information about: {description}",
                "query": self._extract_search_query(description),
            }
        )

        if self._needs_computation(description):
            subtasks.append(
                {
                    "id": self._next_subtask_id(),
                    "type": "execute",
                    "description": f"Perform calculations for: {description}",
                    "code": self._generate_computation_code(description, context),
                }
            )

        subtasks.append(
            {
                "id": self._next_subtask_id(),
                "type": "audit",
                "description": "Validate findings and check evidence quality",
                "target_subtasks": [st["id"] for st in subtasks],
                "mode": mode_cfg.mode,
                "policy": {
                    "audit_level": mode_cfg.audit_level,
                    "argumentation_required": mode_cfg.argumentation,
                },
            }
        )
        return subtasks

    def _extract_search_query(self, description: str) -> str:
        """Extract a lightweight search query from user task."""
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "why", "when", "where", "which"}
        words = description.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(keywords[:6])

    def _needs_computation(self, description: str) -> bool:
        compute_keywords = [
            "calculate",
            "compute",
            "fit",
            "plot",
            "graph",
            "analyze",
            "statistics",
            "regression",
            "curve",
            "data",
            "numeric",
            "temperature",
            "pressure",
            "rate",
            "coefficient",
            "equation",
        ]
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in compute_keywords)

    def _generate_computation_code(self, description: str, context: dict) -> str:
        """Generate fallback computation code snippets."""
        if "activation energy" in description.lower() or "arrhenius" in description.lower():
            return """
import numpy as np
from scipy import stats

T = np.array([300, 350, 400, 450, 500])
k = np.array([1e-3, 5e-3, 2e-2, 8e-2, 0.3])
R = 8.314
x = 1 / T
y = np.log(k)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
Ea = -slope * R / 1000
A = np.exp(intercept)
print(f"Activation Energy (Ea): {Ea:.2f} kJ/mol")
print(f"Pre-exponential factor (A): {A:.2e} s^-1")
print(f"R-squared: {r_value**2:.4f}")
result = {"Ea_kJ_mol": Ea, "A": A, "R_squared": r_value**2}
"""
        if "thermal" in description.lower() or "decomposition" in description.lower():
            return """
import numpy as np

T = np.linspace(300, 800, 100)
w = 1 - 0.5 * (1 + np.tanh((T - 550) / 50))
dw_dT = np.gradient(w, T)
peak_idx = np.argmin(dw_dT)
T_peak = T[peak_idx]
print(f"Peak decomposition temperature: {T_peak:.1f} K ({T_peak-273.15:.1f} °C)")
print(f"Final weight fraction: {w[-1]:.3f}")
result = {"T_peak_K": T_peak, "final_weight_fraction": w[-1]}
"""
        return """
import statistics
data = [1.2, 2.3, 3.1, 4.5, 5.2, 6.1]
mean_val = statistics.mean(data)
std_val = statistics.stdev(data)
print(f"Mean: {mean_val:.3f}")
print(f"Standard deviation: {std_val:.3f}")
result = {"mean": mean_val, "std": std_val}
"""

    def _collect_bids(self, subtask: dict, workers: list) -> list[Bid]:
        """Collect synthetic bids from workers."""
        bids = []
        task_type = subtask.get("type", "")

        for worker in workers:
            if task_type == "research" and worker.role == "researcher":
                bids.append(
                    Bid(
                        task_id=subtask["id"],
                        bidder=worker.agent_id,
                        estimated_cost=1.0,
                        estimated_latency=0.5,
                        success_probability=0.9,
                        capability_match=1.0,
                        strategy="BM25 search over local corpus",
                    )
                )
            elif task_type == "execute" and worker.role == "executor":
                bids.append(
                    Bid(
                        task_id=subtask["id"],
                        bidder=worker.agent_id,
                        estimated_cost=2.0,
                        estimated_latency=1.0,
                        success_probability=0.85,
                        capability_match=1.0,
                        strategy="Python code execution in sandbox",
                    )
                )
            elif task_type == "audit" and worker.role == "auditor":
                bids.append(
                    Bid(
                        task_id=subtask["id"],
                        bidder=worker.agent_id,
                        estimated_cost=0.5,
                        estimated_latency=0.3,
                        success_probability=0.95,
                        capability_match=1.0,
                        strategy="Evidence verification and quality check",
                    )
                )

        return bids

    def _synthesize_answer(self, task: str, results: dict) -> str:
        """Baseline synthesis from subtask outputs."""
        parts = [f"Task: {task}\n", "Findings:\n"]

        for task_id, result in results.items():
            if result.get("success"):
                output = result.get("output", "")
                if isinstance(output, dict):
                    output = json.dumps(output, ensure_ascii=False)
                parts.append(f"- {task_id}: {self._short_output(output, 260)}\n")
            else:
                parts.append(f"- {task_id}: Failed - {result.get('error', 'Unknown error')}\n")

        return "".join(parts)

    def _short_output(self, value: Any, limit: int = 220) -> str:
        """Compact human-readable output for trace/status fields."""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)
        return text[:limit] + ("..." if len(text) > limit else "")

    def _has_counter_claim(self, claims: list) -> bool:
        """Check whether claims include at least one counter-claim/attack relation."""
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            node_type = str(claim.get("type", "")).lower()
            if node_type in {"counter_claim", "counterclaim", "attack"}:
                return True
            if claim.get("counterclaim") or claim.get("counter_claim"):
                return True
            attacks = claim.get("attacks")
            if isinstance(attacks, list) and attacks:
                return True
        return False

    def handle_challenge(self, challenge: dict, workers: list) -> dict:
        """Handle audit challenge by triggering a retry action."""
        reason = challenge.get("reason", "")
        target_subtask = challenge.get("target_subtask_id", challenge.get("target_subtask", ""))

        if self.tracer:
            self.tracer.log_replan(
                self.agent_id,
                reason,
                [{"action": "retry_subtask", "subtask_id": target_subtask}],
            )

        return {"action": "retry", "subtask_id": target_subtask}
