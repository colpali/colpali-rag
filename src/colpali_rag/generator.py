"""Optional answer generation over retrieved pages.

Vendor-neutral: this talks to ANY OpenAI-compatible chat/completions endpoint
(a self-hosted vLLM / Ollama / LM Studio / TGI server, or any hosted one) over
plain HTTP — no SDK, no provider names, no hard-coded model. Configure with
`VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL`; leave `VLM_BASE_URL` empty to disable
answer generation entirely (retrieval + heatmaps still work).

The model only ever sees the retrieved page images, so answers are grounded in the
documents, not the model's memory.
"""

from __future__ import annotations

import base64
import io
import json
import logging

log = logging.getLogger(__name__)


def _image_data_uri(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _post_chat(base_url, api_key, model, messages, *, response_format=None,
               max_tokens=800, timeout=90.0, temperature=0.0) -> dict:
    """Shared OpenAI-compatible /chat/completions POST. Raises on HTTP/transport error."""
    import httpx

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature}
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = httpx.post(base_url.rstrip("/") + "/chat/completions",
                      json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def answer(
    question: str,
    images: list,
    *,
    base_url: str,
    api_key: str | None = None,
    model: str = "vlm",
    labels: list[str] | None = None,
    max_tokens: int = 800,
    timeout: float = 90.0,
) -> str:
    """POST the question + page images to an OpenAI-compatible vision endpoint and
    return the text answer. Each image is preceded by its `labels[i]` (e.g. "Page 3 of
    report.pdf:") so the model can cite pages *verifiably* — without labels it cannot
    know which image is which page. Raises on HTTP/transport error (callers handle it)."""
    import httpx

    content = [
        {
            "type": "text",
            "text": (
                "Answer the question using ONLY the attached document page image(s). "
                "Each image is labelled with its source page; cite the exact page(s) you "
                "relied on. If the pages do not contain the answer, say so plainly."
                f"\n\nQuestion: {question}"
            ),
        }
    ]
    for i, img in enumerate(images):
        if labels and i < len(labels):
            content.append({"type": "text", "text": labels[i]})
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(img)}})

    data = _post_chat(base_url, api_key, model, [{"role": "user", "content": content}],
                      max_tokens=max_tokens, timeout=timeout)
    return data["choices"][0]["message"]["content"].strip()


# ---- structured, cited answers -------------------------------------------
_CAP_CACHE: dict = {}   # (base_url, model) -> working tier; in-memory only (no stale disk pin)
_TIERS = ["json_schema", "json_object", "prompt"]


def _response_format(tier):
    from colpali_rag.schemas import ANSWER_JSON_SCHEMA

    if tier == "json_schema":
        return {"type": "json_schema",
                "json_schema": {"name": "grounded_answer", "schema": ANSWER_JSON_SCHEMA, "strict": True}}
    if tier == "json_object":
        return {"type": "json_object"}
    return None


def answer_structured(question, images, *, attached_page_ids, base_url, api_key=None,
                      model="vlm", labels=None, mode="auto", max_retries=1,
                      max_tokens=1000, timeout=120.0):
    """Return a validated ClaimsResult {answer, claims:[{text, pages, confidence}]}.

    Cascade json_schema -> json_object -> prompt (demote only on an explicit 400/422 —
    a server rejecting the format). Parse defensively, one corrective retry, then fall
    back to a single free-text claim so /api/ask never 500s. `cites` are bracket indices
    resolved to real page_ids in schemas.validate_answer_obj."""
    import httpx

    from colpali_rag.schemas import parse_json, single_free_text_claim, validate_answer_obj

    instr = (
        "Answer the question using ONLY the attached page images. Cite the pages you use "
        "by their bracket number shown above each image, e.g. [1] or [2]. Return ONLY a "
        'JSON object: {"answer": "...", "claims": [{"text": "...", "cites": [1], '
        '"confidence": 0.0}]}. Every claim needs cites (bracket numbers) and a confidence '
        f"in [0,1].\n\nQuestion: {question}"
    )
    content = [{"type": "text", "text": instr}]
    for i, img in enumerate(images):
        content.append({"type": "text", "text": labels[i] if labels and i < len(labels) else f"[{i+1}]"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(img)}})
    messages = [{"role": "user", "content": content}]

    key = (base_url, model)
    if mode != "auto":
        tiers = [mode]
    elif key in _CAP_CACHE:                    # try the known-good tier first, but still
        cached = _CAP_CACHE[key]               # demote through the rest on a transient failure
        tiers = [cached] + [t for t in _TIERS if t != cached]
    else:
        tiers = list(_TIERS)
    raw, errs = "", []
    for tier in tiers:
        for attempt in range(max_retries + 1):
            msgs = messages if attempt == 0 else messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid. Return ONLY JSON matching the schema. " + "; ".join(errs[-2:])}]
            try:
                data = _post_chat(base_url, api_key, model, msgs,
                                  response_format=_response_format(tier),
                                  max_tokens=max_tokens, timeout=timeout)
            except httpx.HTTPStatusError as e:
                sc = e.response.status_code if e.response is not None else 0
                if sc in (400, 422):
                    errs.append(f"{tier}: HTTP {sc}")
                    break                       # server rejects this format -> next tier
                raise
            raw = (data["choices"][0]["message"].get("content") or "").strip()
            obj = parse_json(raw)
            if obj is not None:
                try:
                    res = validate_answer_obj(obj, attached_page_ids, mode=tier)
                    _CAP_CACHE[key] = tier
                    return res
                except Exception as ve:  # noqa: BLE001 - any shape/validate error -> retry/next tier
                    errs.append(f"{tier}: {ve}")
    res = single_free_text_claim(raw or "No answer could be produced.", attached_page_ids)
    res.errors = errs
    log.info("structured answer fell back to free text: %s", errs[-3:])
    return res
