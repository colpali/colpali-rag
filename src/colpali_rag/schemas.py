"""Structured answer schema + a dependency-free validator.

The answer model returns `{answer, claims:[{text, cites, confidence}]}` where `cites`
are **bracket indices** ([1], [2], …) referring to the attached page images in prompt
order — NOT page numbers or ids. This is deliberate: models reliably echo `[1]` but
mangle `report.pdf::p3`, and mapping a bare "page 3" to an attach index silently cites
the wrong page. We resolve each index back to its real (doc, page, page_id) here, and
flag out-of-range indices as citation hallucinations.

Zero third-party deps so this imports in the CLI core. `json_repair` is used only if
it happens to be installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Strict-mode friendly: every field required, additionalProperties disabled, no unions.
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "cites": {"type": "array", "items": {"type": "integer"}},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "cites", "confidence"],
            },
        },
    },
    "required": ["answer", "claims"],
}


@dataclass
class Claim:
    text: str
    pages: list[str]                 # resolved page_ids the claim cites
    confidence: float | None = None  # model-reported; informational only (uncalibrated)


@dataclass
class ClaimsResult:
    answer: str
    claims: list[Claim]
    structured: bool                 # False => the model didn't return usable JSON
    mode: str = "prompt"             # which cascade tier produced it
    hallucinated_citations: list[int] = field(default_factory=list)  # out-of-range indices
    errors: list[str] = field(default_factory=list)


def extract_json(text: str) -> str | None:
    """Pull the first balanced {...} object out of prose / markdown fences. Pure Python."""
    if not text:
        return None
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def parse_json(text: str):
    """json.loads → balanced-brace extract → optional json_repair. Returns dict or None."""
    for candidate in (text, extract_json(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    try:
        import json_repair  # optional

        return json_repair.loads(text)
    except Exception:  # noqa: BLE001
        return None


def validate_answer_obj(obj, attached_page_ids: list[str], *, mode: str = "prompt") -> ClaimsResult:
    """Validate a parsed answer object and resolve bracket-index cites → page_ids.
    Raises ValueError on unusable shape (caller retries or falls back)."""
    if not isinstance(obj, dict) or "answer" not in obj:
        raise ValueError("answer object missing 'answer'")
    answer = str(obj.get("answer", ""))
    n = len(attached_page_ids)
    claims, hallucinated = [], []
    for c in obj.get("claims") or []:
        if not isinstance(c, dict):
            continue
        cites = c.get("cites")
        if isinstance(cites, (int, str)):     # model returned "cites": 1 instead of [1]
            cites = [cites]
        elif not isinstance(cites, list):
            cites = []
        pages = []
        for idx in cites:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= n:
                pid = attached_page_ids[i - 1]
                if pid not in pages:
                    pages.append(pid)
            else:
                hallucinated.append(i)          # cited an image that wasn't attached
        conf = c.get("confidence")
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = None
        claims.append(Claim(text=str(c.get("text", "")), pages=pages, confidence=conf))
    if not claims:
        raise ValueError("no valid claims")
    return ClaimsResult(answer=answer, claims=claims, structured=True, mode=mode,
                        hallucinated_citations=sorted(set(hallucinated)))


def single_free_text_claim(text: str, attached_page_ids: list[str]) -> ClaimsResult:
    """Graceful fallback: the whole answer as one claim citing all attached pages, so
    the API never 500s and the faithfulness judge can still check it."""
    return ClaimsResult(answer=text, structured=False, mode="freetext",
                        claims=[Claim(text=text, pages=list(attached_page_ids), confidence=None)])
