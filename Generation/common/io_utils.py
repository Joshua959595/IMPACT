"""Small filesystem helpers."""
import json as _json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON to a sibling .tmp file, re-read + re-parse it to make sure
    it landed intact, then os.replace() onto the final path. A crash mid-write
    leaves whatever was previously at `path` (if anything) untouched -- used
    so a killed run can never leave a half-written sample_plans.json
    behind that looks complete."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    _json.loads(tmp.read_text(encoding="utf-8"))  # sanity re-parse
    os.replace(tmp, path)
