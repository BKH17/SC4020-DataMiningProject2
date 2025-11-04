# src/utils/text_normalize.py
from __future__ import annotations
import json
from pathlib import Path

_BLANKS = {"", "nan", "na", "none", "null", "-", "_"}

def load_synonyms(path: Path | None) -> dict[str, str]:
    """Load {variant -> canonical} mapping; keys/values are normalized to lowercase."""
    if path is None or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}

def _canonicalize(token: str) -> str:
    """Lowercase; collapse spaces/hyphens to underscores; compress repeats."""
    t = str(token or "").strip().lower()
    t = t.replace("-", " ").replace("_", " ")
    t = " ".join(t.split())           
    t = t.replace(" ", "_")           
    return t

def normalize_token(s: str, synonyms: dict[str, str]) -> str:
    """Return canonical symptom (or '' for blanks). Applies synonyms after canonicalization."""
    t = _canonicalize(s)
    if t in _BLANKS:
        return ""
    return _canonicalize(synonyms.get(t, synonyms.get(str(s).strip().lower(), t)))
