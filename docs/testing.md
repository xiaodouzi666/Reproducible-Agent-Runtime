# Testing Guide

## 1) Live Run (requires `GEMINI_API_KEY`)

```bash
export GEMINI_API_KEY="YOUR_KEY"
PROMPT="What methods are used to calculate activation energy from thermal analysis data? Provide citations and uncertainty."
python run.py run "$PROMPT" --enable-gemini --mode owl_dl --seed 42
```

Check expected completion:

```bash
RUN_ID=$(python - <<'PY'
import json
from pathlib import Path
best_id, best_t = "", ""
for p in Path("runs").iterdir():
    m = p / "metadata.json"
    if not m.exists():
        continue
    d = json.loads(m.read_text(encoding="utf-8"))
    t = d.get("start_time", "")
    if t > best_t:
        best_t, best_id = t, p.name
print(best_id)
PY
)
python run.py show "$RUN_ID"
```

Expected: `LLM Finalize Called: True`, `Finalize Missing: False`.

## 2) Replay (no API key required)

```bash
env -u GEMINI_API_KEY python run.py replay "$RUN_ID"
```

Expected:
- Replay completes without Gemini key.
- Output shows LLM cache hit stats.
- `Answer Match` should be stable for deterministic/cached paths.

## 3) OWL Full Argument Graph (best-effort)

```bash
python run.py run "$PROMPT" --enable-gemini --mode owl_full --seed 42
```

Inspect:

```bash
RUN_FULL=$(python - <<'PY'
import json
from pathlib import Path
best_id, best_t = "", ""
for p in Path("runs").iterdir():
    m = p / "metadata.json"
    if not m.exists():
        continue
    d = json.loads(m.read_text(encoding="utf-8"))
    t = d.get("start_time", "")
    if t > best_t:
        best_t, best_id = t, p.name
print(best_id)
PY
)
python run.py show "$RUN_FULL"
```

Semantics:
- If graph succeeds: `argument_graph.json` exists, status may be `completed` or `completed_with_warnings`.
- If graph fails: run should still finalize; status is `completed_with_warnings` with graph warning(s).
- Only missing finalize should produce `failed`.

## 4) Marathon WAIT / Resume

```bash
python run.py run "Start analysis, but do not finalize now. Call wait_seconds with 60 seconds and reason 'need new data window'." --enable-gemini --mode owl_full --seed 42
```

```bash
RUN_WAIT=$(python - <<'PY'
import json
from pathlib import Path
best_id, best_t = "", ""
for p in Path("runs").iterdir():
    m = p / "metadata.json"
    if not m.exists():
        continue
    d = json.loads(m.read_text(encoding="utf-8"))
    t = d.get("start_time", "")
    if t > best_t:
        best_t, best_id = t, p.name
print(best_id)
PY
)
python run.py show "$RUN_WAIT"
python run.py resume "$RUN_WAIT"
```

Expected:
- Waiting run: `Status: waiting`, `Next Run At` present.
- Resumed run: finalize succeeds (`LLM Finalize Called: True`, `Finalize Missing: False`).
