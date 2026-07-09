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


def _image_data_uri(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def answer(
    question: str,
    images: list,
    *,
    base_url: str,
    api_key: str | None = None,
    model: str = "vlm",
    max_tokens: int = 800,
    timeout: float = 90.0,
) -> str:
    """POST the question + page images to an OpenAI-compatible vision endpoint and
    return the text answer. Raises on HTTP/transport error (callers handle it)."""
    import httpx

    content = [
        {
            "type": "text",
            "text": (
                "Answer the question using ONLY the attached document page image(s). "
                "Cite the page number(s) you relied on. If the pages do not contain "
                f"the answer, say so plainly.\n\nQuestion: {question}"
            ),
        }
    ]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(img)}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.post(base_url.rstrip("/") + "/chat/completions",
                      json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
