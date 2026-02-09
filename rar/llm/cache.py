"""
JSONL cache for LLM requests/responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class JsonlCache:
    """
    A small JSONL cache with run-local and optional global cache files.

    Priority on read:
    1) run cache
    2) readonly cache sources (e.g., original run cache for replay)
    3) global cache
    """

    def __init__(
        self,
        run_dir: Optional[str] = None,
        cache_path: Optional[str] = None,
        global_cache_path: Optional[str] = "runs/_global_llm_cache.jsonl",
        readonly_cache_paths: Optional[list[str]] = None,
        cache_readonly: bool = False,
    ):
        if cache_path:
            self.cache_path = Path(cache_path)
        elif run_dir:
            self.cache_path = Path(run_dir) / "llm_cache.jsonl"
        else:
            self.cache_path = None

        self.global_cache_path = Path(global_cache_path) if global_cache_path else None
        self.readonly_cache_paths = [Path(p) for p in (readonly_cache_paths or []) if p]
        self.cache_readonly = cache_readonly
        self._index: dict[str, dict] = {}
        self._local_keys: set[str] = set()
        self._indexed = False

        if self.cache_path and not self.cache_readonly:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.touch(exist_ok=True)

        if self.global_cache_path and not self.cache_readonly:
            self.global_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.global_cache_path.touch(exist_ok=True)

    def _iter_records(self, path: Path):
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record

    def _ensure_index(self):
        if self._indexed:
            return
        self._index = {}
        self._local_keys = set()

        # Global first, then readonly caches, then run-level overrides them.
        if self.global_cache_path:
            for record in self._iter_records(self.global_cache_path):
                key = record.get("cache_key")
                if key:
                    self._index[key] = record

        for ro_path in self.readonly_cache_paths:
            for record in self._iter_records(ro_path):
                key = record.get("cache_key")
                if key:
                    self._index[key] = record

        if self.cache_path:
            for record in self._iter_records(self.cache_path):
                key = record.get("cache_key")
                if key:
                    self._index[key] = record
                    self._local_keys.add(key)

        self._indexed = True

    def get(self, cache_key: str) -> Optional[dict]:
        """Get cached record by cache_key."""
        self._ensure_index()
        return self._index.get(cache_key)

    def append(self, record: dict, write_global: bool = True):
        """Append a cache record to run cache and optionally global cache."""
        if not isinstance(record, dict):
            raise TypeError("record must be a dict")
        if "cache_key" not in record:
            raise ValueError("record missing cache_key")

        self._ensure_index()
        line = json.dumps(record, ensure_ascii=False, default=str)
        cache_key = str(record["cache_key"])

        if not self.cache_readonly and self.cache_path and cache_key not in self._local_keys:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._local_keys.add(cache_key)

        if not self.cache_readonly and write_global and self.global_cache_path:
            with open(self.global_cache_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        self._index[cache_key] = record
        self._indexed = True
