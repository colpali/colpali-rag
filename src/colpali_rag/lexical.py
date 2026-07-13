"""A lightweight lexical retrieval channel over the already-extracted page text — the
keyword / exact-identifier counterpart to the visual MaxSim ranking.

Character n-gram BM25 (not whole-word) because the identifiers this is meant to catch are full
of punctuation ('AX-1234', 'RS-232/A'); word tokenizers shatter them and substring matching is
brittle, but overlapping char n-grams over the normalized text keep the run contiguous and score
a near-exact hit highly. Pure-Python, zero dependencies. Off unless hybrid retrieval is enabled;
the fusion with the visual channel (Reciprocal Rank Fusion) lives in engine.retrieve.
"""

from __future__ import annotations

import math


def char_ngrams(text: str, lo: int, hi: int) -> list[str]:
    """Overlapping character n-grams (lo..hi) over whitespace-normalized, lowercased text.
    lo is clamped to >= 1 so a misconfigured 0 can't emit empty-string grams (which would match
    every document)."""
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    s = " ".join((text or "").lower().split())
    grams: list[str] = []
    for n in range(lo, hi + 1):
        if len(s) >= n:
            grams.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return grams


class LexicalIndex:
    """Char-n-gram BM25 over (id, text) pairs. Build once per corpus; `search` per query."""

    def __init__(self, docs, *, ngram=(3, 5), k1: float = 1.5, b: float = 0.75):
        self.lo = max(1, int(ngram[0]))
        self.hi = max(self.lo, int(ngram[1]))
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[dict] = []
        self.lens: list[int] = []
        df: dict[str, int] = {}
        for pid, text in docs:
            grams = char_ngrams(text, self.lo, self.hi)
            tf: dict[str, int] = {}
            for g in grams:
                tf[g] = tf.get(g, 0) + 1
            self.ids.append(pid)
            self.tfs.append(tf)
            self.lens.append(len(grams))
            for g in tf:
                df[g] = df.get(g, 0) + 1
        self.n = len(self.ids)
        self.avglen = (sum(self.lens) / self.n) if self.n else 0.0
        # BM25 idf in the log(1 + ...) form, which is always >= 0 (no negative-idf pathology).
        self.idf = {g: math.log(1.0 + (self.n - dfg + 0.5) / (dfg + 0.5)) for g, dfg in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return [(id, bm25_score)] for docs that share any query n-gram, best-first."""
        qgrams = set(char_ngrams(query, self.lo, self.hi))
        if not qgrams or not self.n:
            return []
        avg = self.avglen or 1.0
        scored: list[tuple[str, float]] = []
        for i, tf in enumerate(self.tfs):
            dl = self.lens[i] or 1
            s = 0.0
            for g in qgrams:
                f = tf.get(g)
                if not f:
                    continue
                s += self.idf.get(g, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / avg))
            if s > 0.0:
                scored.append((self.ids[i], s))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]
