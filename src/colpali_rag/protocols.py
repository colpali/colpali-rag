"""Structural typing seams (typing.Protocol) so backends are pluggable and tests
have a checked contract instead of duck-typing. Dependency-light: importing this
pulls in no torch / qdrant / httpx.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    model_id: str

    def embed_pages(self, images: list) -> list: ...
    def score(self, query: str, page_embs: list) -> list[float]: ...
    def similarity_maps(self, page_image, query: str): ...
    @property
    def dim(self) -> int: ...


@runtime_checkable
class PageStore(Protocol):
    def build_from(self, records, images, embs): ...
    def search(self, query: str, top_k: int = 12): ...
    def get_image(self, page_id: str): ...
    def __len__(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list, store, top_k: int) -> list: ...
