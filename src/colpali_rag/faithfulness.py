"""Faithfulness / hallucination checking for answers. Vendor-neutral, OFF by default.

Honest design for the corpus this targets — ColPali exists for *visual/scanned* docs
where extracted page text is empty or noisy, so a lexical "cheap tier" is noise, not a
signal. The real check is a VISION judge that re-reads the cited page image(s) for a
claim and rules supported / partial / unsupported. There is therefore no faithful
default that runs for free: faithfulness is simply **off unless you configure a judge
endpoint**, and turning it on costs judge calls.

Anti-circularity: the judge uses a SEPARATE endpoint (`JUDGE_BASE_URL`). Reusing the
generator as its own judge requires `JUDGE_ALLOW_SAME_ENDPOINT=true` and logs a warning
— a model grading its own homework is not a real check.

We never call a passed claim "verified": an entailment check shows the claim is
*consistent with* the cited page, not that the model causally used it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "unverified": 0.0}


@dataclass
class Verdict:
    claim_index: int
    verdict: str            # supported | partial | unsupported | unverified
    pages: list[str]
    why: str = ""


@dataclass
class FaithfulnessReport:
    verdicts: list[Verdict]
    faithfulness: float                 # mean support over claims
    citation_precision: float           # supported / claims-that-cite
    hallucinated_citations: list[int] = field(default_factory=list)
    unsupported: list[int] = field(default_factory=list)   # claim indices


def _judge_endpoint(settings):
    """Resolve (base_url, api_key, model) for the judge, or None if faithfulness is off."""
    if settings.judge_base_url:
        if settings.judge_base_url == settings.vlm_base_url and not settings.judge_allow_same_endpoint:
            log.warning("JUDGE_BASE_URL equals the generator endpoint (self-grading). Use a "
                        "different judge model, or set JUDGE_ALLOW_SAME_ENDPOINT=true to silence.")
        return (settings.judge_base_url, settings.judge_api_key,
                settings.judge_model or settings.vlm_model)
    if settings.judge_allow_same_endpoint and settings.vlm_base_url:
        log.warning("faithfulness judge is the SAME endpoint as the generator "
                    "(self-grading). Set JUDGE_BASE_URL to a different model for a real check.")
        return (settings.vlm_base_url, settings.vlm_api_key, settings.vlm_model)
    return None


def _judge_claim(claim_text, images, *, base_url, api_key, model):
    from colpali_rag.generator import _image_data_uri, _post_chat
    from colpali_rag.schemas import parse_json

    content = [{"type": "text", "text": (
        "You are a strict fact-checker. Using ONLY the attached page image(s) and no "
        "outside knowledge, judge whether they SUPPORT the claim. Reply ONLY JSON: "
        '{"verdict": "supported" | "partial" | "unsupported", "why": "..."}.\n\n'
        f"Claim: {claim_text}")}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _image_data_uri(img)}})
    data = _post_chat(base_url, api_key, model, [{"role": "user", "content": content}],
                      max_tokens=300, timeout=90.0)
    obj = parse_json(data["choices"][0]["message"].get("content") or "") or {}
    v = str(obj.get("verdict", "")).lower().strip()
    return (v if v in ("supported", "partial", "unsupported") else "unverified"), str(obj.get("why", ""))


def judge_answer(result, get_image, settings):
    """Judge each claim by re-reading ALL its cited page images. Returns a
    FaithfulnessReport, or None if no judge endpoint is configured. Never raises."""
    ep = _judge_endpoint(settings)
    if ep is None:
        return None
    base_url, api_key, model = ep
    verdicts, scores = [], []
    supported_ct, cited_ct = 0, 0
    for i, claim in enumerate(result.claims):
        if not claim.pages:
            verdicts.append(Verdict(i, "unverified", [], "no citation"))
            scores.append(0.0)
            continue
        cited_ct += 1
        try:
            imgs = [im for im in (get_image(pid) for pid in claim.pages) if im is not None]
            if not imgs:
                verdicts.append(Verdict(i, "unverified", claim.pages, "cited page image missing"))
                scores.append(0.0)
                continue
            v, why = _judge_claim(claim.text, imgs, base_url=base_url, api_key=api_key, model=model)
        except Exception as e:  # noqa: BLE001 - image fetch OR judge failure must never 500
            log.warning("faithfulness check failed for claim %d: %s: %s", i, type(e).__name__, e)
            verdicts.append(Verdict(i, "unverified", claim.pages, "check error"))
            scores.append(0.0)
            continue
        verdicts.append(Verdict(i, v, claim.pages, why))
        scores.append(_SCORE[v])
        if v == "supported":
            supported_ct += 1
    faith = round(sum(scores) / len(scores), 3) if scores else 0.0
    prec = round(supported_ct / cited_ct, 3) if cited_ct else 0.0
    unsupported = [v.claim_index for v in verdicts if v.verdict in ("unsupported", "partial")]
    return FaithfulnessReport(verdicts, faith, prec,
                              list(getattr(result, "hallucinated_citations", [])), unsupported)


def apply_gate(result, report, mode: str, min_score):
    """mode: off | flag | withhold. Returns (result, withheld: bool).

    - flag: nothing hidden (the caller attaches verdicts to the response).
    - withhold: if overall faithfulness < min_score (default 0.5), replace the WHOLE
      answer with a 'could not verify' message. We do not mask individual sentences out
      of a prose answer (that leaves the hallucinated text visible)."""
    if report is None or mode == "off":
        return result, False
    if mode == "withhold":
        threshold = 0.5 if min_score is None else min_score
        if report.faithfulness < threshold:
            result.answer = ("The retrieved pages do not sufficiently support an answer "
                             "to this question.")
            result.claims = []
            return result, True
    return result, False
