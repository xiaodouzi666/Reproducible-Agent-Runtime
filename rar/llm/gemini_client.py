"""
Gemini SDK client wrapper with cache and reproducible call records.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import base64
import ast
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Callable

from .cache import JsonlCache
from ..config import resolve_mode_config

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - import errors handled at runtime
    genai = None
    types = None


class GeminiClientError(RuntimeError):
    """Raised when Gemini requests fail."""


@dataclass
class LLMCallRecord:
    """A normalized, cacheable record for one LLM call."""

    cache_key: str
    model: str
    thinking_level: str
    request: dict
    response_text: str
    response_raw: Any
    usage: dict
    response_hash: str
    request_hash: str = ""

    is_cache_hit: bool = False
    deterministic: bool = False
    latency_ms: float = 0.0
    error: str = ""
    fallback_from: str = ""
    response_json: Optional[dict] = None
    function_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMCallRecord":
        normalized = dict(data)
        normalized.setdefault("request_hash", normalized.get("cache_key", ""))
        return cls(**normalized)


class GeminiClient:
    """
    Gemini API wrapper:
    - deterministic request payload hashing
    - JSONL cache support
    - optional fallback from Pro to Flash on transient errors
    """

    FALLBACK_MODEL = "gemini-3-flash-preview"
    FALLBACK_THINKING_LEVEL = "minimal"

    def __init__(
        self,
        api_key: Optional[str] = None,
        run_dir: Optional[str] = None,
        run_cache_path: Optional[str] = None,
        cache: Optional[JsonlCache] = None,
        enable_cache: bool = True,
        global_cache_path: Optional[str] = "runs/_global_llm_cache.jsonl",
        readonly_cache_paths: Optional[list[str]] = None,
        cache_readonly: bool = False,
        allow_missing_api_key: bool = False,
        auto_fallback: bool = True,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.enable_cache = enable_cache
        self.allow_missing_api_key = allow_missing_api_key
        self.auto_fallback = auto_fallback
        self.cache = cache or JsonlCache(
            run_dir=run_dir,
            cache_path=run_cache_path,
            global_cache_path=global_cache_path,
            readonly_cache_paths=readonly_cache_paths,
            cache_readonly=cache_readonly,
        )

        if self.api_key:
            if genai is None:
                raise GeminiClientError(
                    "google-genai is not installed. Run `pip install -r requirements.txt`."
                )
            self.client = genai.Client(api_key=self.api_key)
        else:
            if not (self.allow_missing_api_key and self.enable_cache):
                raise GeminiClientError("Missing GEMINI_API_KEY.")
            self.client = None

    @classmethod
    def resolve_mode_defaults(
        cls,
        mode: str,
        model_override: Optional[str] = None,
        thinking_override: Optional[str] = None,
    ) -> dict:
        """Resolve model/thinking defaults from OWL mode."""
        resolved = resolve_mode_config(
            mode=mode or "owl_lite",
            model_override=model_override,
            thinking_override=thinking_override,
        )
        return {
            "mode": resolved.mode,
            "model": resolved.model,
            "thinking_level": resolved.thinking_level,
        }

    def generate_text(
        self,
        prompt: str,
        model: str,
        thinking_level: str,
        system_prompt: Optional[str] = None,
        tools_schema: Optional[list[dict]] = None,
    ) -> LLMCallRecord:
        """Generate plain text using Gemini."""
        request_payload = self._build_request_payload(
            prompt=prompt,
            model=model,
            thinking_level=thinking_level,
            system_prompt=system_prompt,
            tools_schema=tools_schema,
            json_schema=None,
            response_format="text",
        )
        return self._execute(request_payload=request_payload)

    def compute_cache_key(
        self,
        prompt: str,
        model: str,
        thinking_level: str,
        system_prompt: Optional[str] = None,
        tools_schema: Optional[list[dict]] = None,
        json_schema: Optional[Any] = None,
        response_format: str = "text",
    ) -> str:
        """Compute deterministic cache_key for a request without executing it."""
        payload = self._build_request_payload(
            prompt=prompt,
            model=model,
            thinking_level=thinking_level,
            system_prompt=system_prompt,
            tools_schema=tools_schema,
            json_schema=self._normalize_schema(json_schema) if json_schema else None,
            response_format=response_format,
        )
        return self._hash_payload(payload)

    def generate_json(
        self,
        prompt: str,
        model: str,
        thinking_level: str,
        json_schema: Any,
        system_prompt: Optional[str] = None,
        tools_schema: Optional[list[dict]] = None,
        max_schema_retries: int = 1,
        schema_name: str = "",
        on_schema_violation: Optional[Callable[[dict], None]] = None,
    ) -> LLMCallRecord:
        """
        Generate structured JSON using response_json_schema + schema validation.

        If output violates schema, retries with one repair prompt by default.
        """
        normalized_schema = self._normalize_schema(json_schema)
        schema_model = self._resolve_schema_model(json_schema)
        total_attempts = max(1, int(max_schema_retries) + 1)
        base_prompt = prompt
        last_error = ""
        last_record: Optional[LLMCallRecord] = None

        for attempt in range(1, total_attempts + 1):
            attempt_prompt = base_prompt
            if attempt > 1:
                attempt_prompt = self._build_schema_repair_prompt(base_prompt, last_error)

            request_payload = self._build_request_payload(
                prompt=attempt_prompt,
                model=model,
                thinking_level=thinking_level,
                system_prompt=system_prompt,
                tools_schema=tools_schema,
                json_schema=normalized_schema,
                response_format="json",
                schema_name=schema_name,
            )

            record = self._execute(request_payload=request_payload)
            last_record = record

            validated_json, validation_error = self._validate_json_output(
                response_text=record.response_text,
                schema_model=schema_model,
            )

            if validated_json is not None:
                record.response_json = validated_json
                return record

            last_error = validation_error or "Unknown schema validation error"
            if on_schema_violation:
                try:
                    on_schema_violation(
                        {
                            "schema_name": schema_name or self._schema_name(json_schema),
                            "attempt": attempt,
                            "max_attempts": total_attempts,
                            "error": last_error,
                            "model": record.model,
                            "thinking_level": record.thinking_level,
                            "cache_key": record.cache_key,
                            "response_hash": record.response_hash,
                            "response_text_preview": (record.response_text or "")[:500],
                        }
                    )
                except Exception:
                    # Schema violation callbacks are non-fatal.
                    pass

        schema_label = schema_name or self._schema_name(json_schema)
        raise GeminiClientError(
            f"Structured JSON generation failed for schema '{schema_label}' after {total_attempts} attempt(s): {last_error}"
        )

    def tool_call_chat_loop(
        self,
        user_prompt: str,
        model: str,
        thinking_level: str,
        tools_schema: list[dict],
        tool_handlers: dict[str, Callable[[dict], dict]],
        system_prompt: Optional[str] = None,
        max_steps: int = 10,
        on_step: Optional[Callable[[dict], None]] = None,
        use_cache: bool = False,
        force_finalize_retry: bool = True,
        initial_history: Optional[list[Any]] = None,
    ) -> dict:
        """
        Run a function-calling chat loop.

        The loop preserves model response content in history so Gemini can keep
        thought signatures consistent across tool-calling turns.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")

        if initial_history and isinstance(initial_history, list):
            if self.client is None:
                history = self._to_jsonable(initial_history)
            else:
                history = self._sanitize_history_items(initial_history)
            if not history:
                history = [self._make_user_text_content(user_prompt)]
        else:
            history = [self._make_user_text_content(user_prompt)]
        steps: list[dict] = []
        finalize_result: Optional[dict] = None
        wait_result: Optional[dict] = None
        final_text = ""
        ended_reason = "max_steps_reached"
        force_finalize_attempted = False

        def _run_step(step_number: int, append_finalize_prompt_on_no_call: bool = True):
            nonlocal finalize_result, wait_result, final_text, ended_reason, history
            request_payload = {
                "provider": "gemini",
                "model": model,
                "thinking_level": thinking_level,
                "contents": self._to_jsonable(history),
                "system_prompt": system_prompt or "",
                "tools_schema": tools_schema or [],
                "json_schema": {},
                "response_format": "tool_loop",
                "step_index": step_number,
            }
            cache_key = self._hash_payload(request_payload)

            record: LLMCallRecord
            function_calls: list[dict]
            response_obj = None

            if use_cache and self.enable_cache:
                cached = self.cache.get(cache_key)
            else:
                cached = None

            if cached:
                cache_record = dict(cached)
                cache_record.setdefault("request_hash", cache_record.get("cache_key", ""))
                if use_cache and self.enable_cache:
                    # Mirror cache-hit records into current run cache for replay artifacts.
                    self.cache.append(cache_record, write_global=False)
                record = LLMCallRecord.from_dict(cache_record)
                record.is_cache_hit = True
                function_calls = record.function_calls or self._extract_function_calls_from_raw(record.response_raw)
                cached_request = record.request if isinstance(record.request, dict) else {}
                cached_contents = cached_request.get("contents")
                if isinstance(cached_contents, list):
                    # Reset to original request history for deterministic replay key matching.
                    if self.client is None:
                        # Cache-only replay mode: preserve exact payload for stable cache keys.
                        history = self._to_jsonable(cached_contents)
                    else:
                        # Live mode: sanitize cached history to avoid SDK validation errors
                        # from stale/serialized thought-signature fields.
                        history = self._sanitize_history_items(cached_contents)
            else:
                start_time = time.time()
                try:
                    response_obj, used_model, used_thinking, fallback_from = self._call_with_fallback(
                        {
                            "model": model,
                            "thinking_level": thinking_level,
                            "contents": history,
                            "system_prompt": system_prompt or "",
                            "tools_schema": tools_schema or [],
                            "response_format": "tool_loop",
                        }
                    )
                except Exception as e:
                    raise GeminiClientError(f"Gemini tool loop failed at step {step_number}: {e}") from e

                latency_ms = (time.time() - start_time) * 1000
                response_text = self._extract_text(response_obj)
                response_raw = self._to_jsonable(response_obj)
                usage = self._extract_usage(response_obj)
                response_hash = self._compute_response_hash(response_text, response_raw)
                function_calls = self._extract_function_calls(response_obj)

                record = LLMCallRecord(
                    cache_key=cache_key,
                    model=used_model,
                    thinking_level=used_thinking,
                    request=request_payload,
                    response_text=response_text,
                    response_raw=response_raw,
                    usage=usage,
                    response_hash=response_hash,
                    request_hash=cache_key,
                    is_cache_hit=False,
                    deterministic=False,
                    latency_ms=latency_ms,
                    fallback_from=fallback_from,
                    function_calls=function_calls,
                )

                if use_cache and self.enable_cache:
                    self.cache.append(record.to_dict(), write_global=True)

            # Preserve model content in history to keep function-call context.
            if response_obj is not None:
                model_content = self._extract_model_content(response_obj)
                if model_content is not None:
                    history.append(model_content)
            else:
                # Cache path: reconstruct model content to preserve function-call history.
                model_content = self._extract_model_content_from_raw(
                    response_raw=record.response_raw,
                    response_text=record.response_text,
                    function_calls=function_calls,
                )
                if model_content is not None:
                    history.append(model_content)

            step_info = {
                "step": step_number,
                "record": record.to_dict(),
                "function_calls": function_calls,
                "tool_results": [],
                "history": self._to_jsonable(history),
            }

            if function_calls:
                for fc in function_calls:
                    fn_name = fc.get("name", "")
                    fn_args = fc.get("args", {}) if isinstance(fc.get("args"), dict) else {}

                    handler = tool_handlers.get(fn_name)
                    if not handler:
                        tool_result = {
                            "status": "error",
                            "error": f"Unknown function: {fn_name}",
                            "function": fn_name,
                        }
                    else:
                        try:
                            tool_result = handler(fn_args)
                        except Exception as e:
                            tool_result = {
                                "status": "error",
                                "error": str(e),
                                "function": fn_name,
                            }

                    step_info["tool_results"].append(
                        {
                            "name": fn_name,
                            "args": fn_args,
                            "result": tool_result,
                        }
                    )
                    history.append(self._make_function_response_content(fn_name, tool_result))

                    if fn_name == "finalize":
                        finalized = False
                        if isinstance(tool_result, dict):
                            finalized = bool(
                                tool_result.get("finalized")
                                or str(tool_result.get("status", "")).lower() == "success"
                            )
                        if finalized:
                            finalize_result = tool_result
                            ended_reason = "finalized"
                    if fn_name in {"wait_seconds", "wait_until"}:
                        waiting = False
                        if isinstance(tool_result, dict):
                            waiting = bool(
                                tool_result.get("wait_requested")
                                or str(tool_result.get("status", "")).lower() == "waiting"
                            )
                        if waiting and finalize_result is None:
                            wait_result = tool_result if isinstance(tool_result, dict) else {"status": "waiting"}
                            ended_reason = "waiting_requested"
            else:
                final_text = record.response_text or final_text
                if append_finalize_prompt_on_no_call and finalize_result is None and wait_result is None:
                    history.append(
                        self._make_user_text_content(
                            "You must finish by calling finalize(...) with final answer and evidence mapping."
                        )
                    )

            step_info["history"] = self._to_jsonable(history)
            if on_step:
                on_step(step_info)
            steps.append(step_info)

        for step_idx in range(max_steps):
            _run_step(step_idx + 1, append_finalize_prompt_on_no_call=True)
            if finalize_result is not None or wait_result is not None:
                break

        if finalize_result is None and wait_result is None and force_finalize_retry:
            force_finalize_attempted = True
            history.append(
                self._make_user_text_content(
                    "FINAL STEP: You must call finalize(...) now. "
                    "Do not call any other function. "
                    "If uncertainty remains, include it inside finalize and still call finalize."
                )
            )
            _run_step(max_steps + 1, append_finalize_prompt_on_no_call=False)
            if finalize_result is None:
                ended_reason = "force_finalize_failed"

        success = finalize_result is not None or wait_result is not None
        return {
            "success": success,
            "finalize_result": finalize_result,
            "wait_result": wait_result,
            "waiting_requested": wait_result is not None,
            "final_text": final_text,
            "steps": steps,
            "ended_reason": ended_reason,
            "history_length": len(history),
            "history": self._to_jsonable(history),
            "finalize_called": finalize_result is not None,
            "force_finalize_attempted": force_finalize_attempted,
        }

    def _execute(self, request_payload: dict) -> LLMCallRecord:
        cache_key = self._hash_payload(request_payload)

        if self.enable_cache:
            cached = self.cache.get(cache_key)
            if cached:
                cache_record = dict(cached)
                cache_record.setdefault("request_hash", cache_record.get("cache_key", ""))
                # Mirror cache-hit record into current run cache for replay artifact completeness.
                self.cache.append(cache_record, write_global=False)
                cached_record = LLMCallRecord.from_dict(cache_record)
                cached_record.is_cache_hit = True
                return cached_record

        start_time = time.time()
        response_obj, used_model, used_thinking, fallback_from = self._call_with_fallback(request_payload)
        latency_ms = (time.time() - start_time) * 1000

        response_text = self._extract_text(response_obj)
        response_raw = self._to_jsonable(response_obj)
        usage = self._extract_usage(response_obj)
        response_hash = self._compute_response_hash(response_text, response_raw)

        record = LLMCallRecord(
            cache_key=cache_key,
            model=used_model,
            thinking_level=used_thinking,
            request=request_payload,
            response_text=response_text,
            response_raw=response_raw,
            usage=usage,
            response_hash=response_hash,
            request_hash=cache_key,
            is_cache_hit=False,
            deterministic=False,
            latency_ms=latency_ms,
            fallback_from=fallback_from,
        )

        if self.enable_cache:
            self.cache.append(record.to_dict(), write_global=True)

        return record

    def _call_with_fallback(self, request_payload: dict):
        if self.client is None:
            raise GeminiClientError(
                "Gemini API client unavailable (missing GEMINI_API_KEY) and cache miss occurred."
            )
        model = request_payload["model"]
        try:
            return self._call_once(request_payload), model, request_payload.get("thinking_level", ""), ""
        except Exception as e:
            if not (self.auto_fallback and self._is_transient_error(e)):
                raise GeminiClientError(f"Gemini request failed: {e}") from e

            if model == self.FALLBACK_MODEL:
                raise GeminiClientError(f"Gemini request failed on fallback model: {e}") from e

            fallback_payload = dict(request_payload)
            fallback_payload["model"] = self.FALLBACK_MODEL
            fallback_payload["thinking_level"] = self.FALLBACK_THINKING_LEVEL

            try:
                return (
                    self._call_once(fallback_payload),
                    self.FALLBACK_MODEL,
                    fallback_payload["thinking_level"],
                    model,
                )
            except Exception as fallback_error:
                raise GeminiClientError(
                    "Gemini request failed on both primary and fallback models: "
                    f"{fallback_error}"
                ) from fallback_error

    def _call_once(self, request_payload: dict):
        contents = request_payload.get("contents", request_payload.get("prompt", ""))

        # Primary attempt: response_json_schema (Gemini 3 docs).
        config = self._build_config(
            thinking_level=request_payload.get("thinking_level"),
            system_prompt=request_payload.get("system_prompt"),
            json_schema=request_payload.get("json_schema"),
            response_format=request_payload.get("response_format", "text"),
            tools_schema=request_payload.get("tools_schema"),
            json_schema_field="response_json_schema",
        )
        try:
            return self.client.models.generate_content(
                model=request_payload["model"],
                contents=contents,
                config=config,
            )
        except Exception as first_error:
            should_fallback = (
                request_payload.get("response_format") == "json"
                and bool(request_payload.get("json_schema"))
                and self._looks_like_schema_field_error(first_error)
            )
            if not should_fallback:
                raise

            # Compatibility fallback for older SDK/server paths.
            fallback_config = self._build_config(
                thinking_level=request_payload.get("thinking_level"),
                system_prompt=request_payload.get("system_prompt"),
                json_schema=request_payload.get("json_schema"),
                response_format=request_payload.get("response_format", "text"),
                tools_schema=request_payload.get("tools_schema"),
                json_schema_field="response_schema",
            )
            return self.client.models.generate_content(
                model=request_payload["model"],
                contents=contents,
                config=fallback_config,
            )

    def _build_request_payload(
        self,
        prompt: str,
        model: str,
        thinking_level: str,
        system_prompt: Optional[str],
        tools_schema: Optional[list[dict]],
        json_schema: Optional[dict],
        response_format: str,
        schema_name: str = "",
    ) -> dict:
        payload = {
            "provider": "gemini",
            "model": model,
            "thinking_level": thinking_level,
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "tools_schema": tools_schema or [],
            "json_schema": json_schema or {},
            "schema_name": schema_name or "",
            "response_format": response_format,
        }
        return payload

    def _build_config(
        self,
        thinking_level: Optional[str],
        system_prompt: Optional[str],
        json_schema: Optional[dict],
        response_format: str,
        tools_schema: Optional[list[dict]],
        json_schema_field: str = "response_json_schema",
    ):
        config: dict[str, Any] = {}

        if system_prompt:
            config["system_instruction"] = system_prompt

        if thinking_level:
            config["thinking_config"] = {"thinking_level": thinking_level}

        if response_format == "json":
            config["response_mime_type"] = "application/json"
            if json_schema:
                config[json_schema_field] = json_schema

        if tools_schema:
            config["tools"] = tools_schema

        # Prefer typed config when available, then fall back to dict.
        if types is not None:
            try:
                return types.GenerateContentConfig(**config)
            except Exception:
                return config
        return config

    def _normalize_schema(self, schema: Any) -> dict:
        if schema is None:
            return {}
        if isinstance(schema, dict):
            return schema
        # Pydantic v2 model class.
        if hasattr(schema, "model_json_schema"):
            return schema.model_json_schema()
        # Pydantic model instance.
        if hasattr(schema, "__class__") and hasattr(schema.__class__, "model_json_schema"):
            return schema.__class__.model_json_schema()
        raise TypeError("json_schema must be a dict or a Pydantic model class.")

    def _resolve_schema_model(self, schema: Any):
        """Resolve Pydantic model class from schema input when available."""
        if schema is None:
            return None
        if inspect.isclass(schema) and hasattr(schema, "model_validate_json"):
            return schema
        if hasattr(schema, "__class__") and hasattr(schema.__class__, "model_validate_json"):
            return schema.__class__
        return None

    def _extract_text(self, response_obj: Any) -> str:
        parts = []
        candidates = getattr(response_obj, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            content_parts = getattr(content, "parts", None) or []
            for part in content_parts:
                value = getattr(part, "text", None)
                if value:
                    parts.append(str(value))

        if parts:
            return "\n".join(parts).strip()

        # Avoid calling response.text when response contains non-text parts
        # (e.g., function_call), which emits SDK warnings.
        return ""

    def _extract_usage(self, response_obj: Any) -> dict:
        usage = getattr(response_obj, "usage_metadata", None)
        if usage is None:
            return {}
        return self._to_jsonable(usage)

    def _extract_model_content(self, response_obj: Any):
        candidates = getattr(response_obj, "candidates", None) or []
        if not candidates:
            return None
        return getattr(candidates[0], "content", None)

    def _extract_model_content_from_raw(
        self,
        response_raw: Any,
        response_text: str,
        function_calls: list[dict],
    ):
        """
        Reconstruct model content from cached raw response for replay history.
        """
        if isinstance(response_raw, dict):
            candidates = response_raw.get("candidates")
            if isinstance(candidates, list) and candidates:
                first = candidates[0] if isinstance(candidates[0], dict) else None
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict):
                        sanitized = self._sanitize_content_dict(content, role_default="model")
                        if sanitized is not None:
                            return sanitized

        if function_calls:
            parts = []
            for fc in function_calls:
                if not isinstance(fc, dict):
                    continue
                name = str(fc.get("name", "") or "")
                args = fc.get("args", {})
                if not name:
                    continue
                parts.append({"function_call": {"name": name, "args": args if isinstance(args, dict) else {}}})
            if parts:
                return {"role": "model", "parts": parts}

        if response_text:
            return self._make_model_text_content(response_text)
        return None

    def _extract_function_calls(self, response_obj: Any) -> list[dict]:
        calls: list[dict] = []

        direct_calls = getattr(response_obj, "function_calls", None) or []
        for fc in direct_calls:
            calls.append(
                {
                    "name": str(getattr(fc, "name", "") or ""),
                    "args": self._coerce_args(getattr(fc, "args", {})),
                }
            )

        if calls:
            return calls

        candidates = getattr(response_obj, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc is None:
                    continue
                calls.append(
                    {
                        "name": str(getattr(fc, "name", "") or ""),
                        "args": self._coerce_args(getattr(fc, "args", {})),
                    }
                )

        return calls

    def _extract_function_calls_from_raw(self, response_raw: Any) -> list[dict]:
        if not isinstance(response_raw, dict):
            return []
        calls = response_raw.get("function_calls")
        if isinstance(calls, list):
            normalized = []
            for call in calls:
                if isinstance(call, dict):
                    normalized.append(
                        {
                            "name": str(call.get("name", "")),
                            "args": self._coerce_args(call.get("args", {})),
                        }
                    )
            return normalized
        return []

    def _coerce_args(self, args: Any) -> dict:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": args}
        if hasattr(args, "items"):
            try:
                return {str(k): self._to_jsonable(v) for k, v in args.items()}
            except Exception:
                return {"value": self._to_jsonable(args)}
        return {"value": self._to_jsonable(args)}

    def _make_user_text_content(self, text: str):
        if types is not None:
            try:
                return types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            except Exception:
                pass
        return {"role": "user", "parts": [{"text": text}]}

    def _make_model_text_content(self, text: str):
        if types is not None:
            try:
                return types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=text)],
                )
            except Exception:
                pass
        return {"role": "model", "parts": [{"text": text}]}

    def _make_function_response_content(self, name: str, response: dict):
        payload = response if isinstance(response, dict) else {"result": response}
        if types is not None:
            try:
                part = types.Part.from_function_response(name=name, response={"result": payload})
                return types.Content(role="user", parts=[part])
            except Exception:
                pass
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": name,
                        "response": {"result": payload},
                    }
                }
            ],
        }

    def _to_jsonable(self, value: Any):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return self._to_jsonable(value.model_dump(exclude_none=True))
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return self._to_jsonable(
                    {k: v for k, v in vars(value).items() if not k.startswith("_")}
                )
            except Exception:
                pass
        return str(value)

    def _sanitize_history_items(self, contents: list[Any]) -> list[Any]:
        """Sanitize cached history for live API calls while preserving turn structure."""
        sanitized: list[Any] = []
        for item in contents:
            cleaned = self._sanitize_history_item(item)
            if cleaned is not None:
                sanitized.append(cleaned)
        return sanitized

    def _sanitize_history_item(self, item: Any) -> Optional[Any]:
        if isinstance(item, dict):
            if "parts" in item or "role" in item:
                return self._sanitize_content_dict(item)
            return None
        if isinstance(item, str):
            return self._make_user_text_content(item)
        # Keep SDK-native content objects untouched.
        if item is not None and item.__class__.__name__ == "Content":
            return item
        return None

    def _sanitize_content_dict(self, content: dict, role_default: str = "user") -> Optional[dict]:
        role = str(content.get("role", role_default) or role_default)
        raw_parts = content.get("parts", [])
        if not isinstance(raw_parts, list):
            raw_parts = [raw_parts]

        clean_parts = []
        for part in raw_parts:
            clean = self._sanitize_part_dict(part)
            if clean is not None:
                clean_parts.append(clean)

        if not clean_parts:
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                clean_parts.append({"text": text})

        if not clean_parts:
            return None
        return {"role": role, "parts": clean_parts}

    def _sanitize_part_dict(self, part: Any) -> Optional[dict]:
        if not isinstance(part, dict):
            return None

        # Keep text parts.
        if isinstance(part.get("text"), str):
            return {"text": part.get("text", "")}

        # Keep function-call parts with normalized args.
        fc = part.get("function_call")
        if isinstance(fc, dict):
            name = str(fc.get("name", "") or "")
            if not name:
                return None
            args = self._coerce_args(fc.get("args", {}))
            output = {"function_call": {"name": name, "args": args}}
            thought_signature = self._normalize_thought_signature(
                part.get("thought_signature", fc.get("thought_signature"))
            )
            if thought_signature:
                output["thought_signature"] = thought_signature
            return output

        # Keep function-response parts with normalized response payload.
        fr = part.get("function_response")
        if isinstance(fr, dict):
            name = str(fr.get("name", "") or "")
            response = fr.get("response", {})
            if not isinstance(response, dict):
                response = {"result": self._to_jsonable(response)}
            return {
                "function_response": {
                    "name": name or "function_response",
                    "response": self._to_jsonable(response),
                }
            }

        # Drop thought-signature-only or unknown part kinds in cached history.
        return None

    def _normalize_thought_signature(self, value: Any) -> str:
        """
        Normalize cached thought signatures to base64 string accepted by SDK.
        Handles legacy serialized values like \"b'...\\x12...'\".
        """
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")

        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return ""

            # Legacy python-bytes-literal string.
            if (raw.startswith("b'") and raw.endswith("'")) or (raw.startswith('b"') and raw.endswith('"')):
                try:
                    literal = ast.literal_eval(raw)
                    if isinstance(literal, (bytes, bytearray)):
                        return base64.b64encode(bytes(literal)).decode("ascii")
                except Exception:
                    pass

            # Already base64? keep it.
            try:
                base64.b64decode(raw, validate=True)
                return raw
            except Exception:
                # Last resort: treat as utf-8 bytes and encode.
                return base64.b64encode(raw.encode("utf-8")).decode("ascii")

        # Fallback for unexpected value types.
        return base64.b64encode(str(value).encode("utf-8")).decode("ascii")

    def _safe_parse_json(self, text: str) -> Optional[dict]:
        if not text:
            return None
        cleaned = self._strip_json_fence(text)
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {"data": data}
        except json.JSONDecodeError:
            return None

    def _strip_json_fence(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned

    def _validate_json_output(self, response_text: str, schema_model: Any) -> tuple[Optional[dict], str]:
        """Validate generated JSON against schema model when available."""
        parsed = self._safe_parse_json(response_text)
        if schema_model is None:
            if parsed is not None:
                return parsed, ""
            return None, "Response is not valid JSON object."

        cleaned = self._strip_json_fence(response_text)
        try:
            if cleaned:
                validated = schema_model.model_validate_json(cleaned)
            elif parsed is not None:
                validated = schema_model.model_validate(parsed)
            else:
                raise ValueError("Empty JSON response")
            return self._to_jsonable(validated.model_dump(exclude_none=True)), ""
        except Exception as e:
            return None, str(e)

    def _build_schema_repair_prompt(self, base_prompt: str, validation_error: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "IMPORTANT: Your previous output did not match the required JSON schema.\n"
            f"Validation error: {validation_error}\n"
            "Return ONLY valid JSON. No markdown, no explanations."
        )

    def _schema_name(self, schema: Any) -> str:
        if inspect.isclass(schema):
            return getattr(schema, "__name__", "schema")
        return schema.__class__.__name__ if hasattr(schema, "__class__") else "schema"

    def _hash_payload(self, payload: dict) -> str:
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _compute_response_hash(self, response_text: str, response_raw: Any) -> str:
        """Stable response hash; uses raw payload when text is empty (e.g. function-call turns)."""
        if response_text:
            return self._hash_text(response_text)
        try:
            raw_json = json.dumps(response_raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        except Exception:
            return self._hash_text(str(response_raw))

    def _is_transient_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        transient_patterns = [
            r"\b429\b",
            r"resource_exhausted",
            r"\brate limit\b",
            r"\bquota\b",
            r"\btimeout\b",
            r"timed out",
            r"deadline_exceeded",
            r"\bunavailable\b",
            r"service unavailable",
        ]
        return any(re.search(pattern, message) for pattern in transient_patterns)

    def _looks_like_schema_field_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        schema_tokens = [
            "response_json_schema",
            "unexpected keyword argument",
            "unknown field",
            "invalid argument",
        ]
        return any(token in message for token in schema_tokens)
