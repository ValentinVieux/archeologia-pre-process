"""Rapport HTML statique auto-suffisant : template + audit.json embarqué, rendu client."""
from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = Path(__file__).with_name("template.html")


def render_report(audit: dict) -> str:
    # "<" n'apparaît que dans les chaînes JSON et < est un échappement JSON valide :
    # couvre </script>, <!-- et <script (états script-data escaped/double-escaped du
    # tokenizer HTML) pour des valeurs attributaires non fiables, sans perte au JSON.parse.
    payload = json.dumps(audit, ensure_ascii=False).replace("<", "\\u003c")
    return _TEMPLATE.read_text(encoding="utf-8").replace("__AUDIT_JSON__", payload)
