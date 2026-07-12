"""Page stores. Both backends expose the same shape:

    build_from(records, images, embs)   # index pages
    search(query, top_k) -> [(Page, score, page_id)]
    get_image(page_id) -> PIL.Image
    __len__

Page images are always persisted to `<data_dir>/images/` so the web UI can serve
them and the index survives a restart.

  * MemoryStore — brute-force MaxSim in Python; zero infrastructure. Persists
    embeddings to disk so `index` once, `serve` many.
  * QdrantStore — native multivector MAX_SIM; embedded on-disk or a real server via
    QDRANT_URL. Use when the corpus outgrows memory.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from colpali_rag.pdf import Page

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
SCHEMA_VERSION = 1


def page_id(doc: str, page: int) -> str:
    return f"{doc}::p{page}"


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)


def image_key(pid: str) -> str:
    """Storage key for a page image. `images/<safe>.png` — byte-identical to the
    pre-adapter on-disk path, so LocalArtifactStore(root=data_dir) keeps old indexes."""
    return f"images/{_safe(pid)}.png"


def _emb_dim(embs) -> int | None:
    """Best-effort embedding dim from the first page's multivector (for the identity
    guard). Robust to tensors, 2D list-of-lists, 1D vectors, and scalar test fakes."""
    if not embs:
        return None
    e0 = embs[0]
    try:
        return int(e0.shape[-1])         # tensor / ndarray (seq, dim)
    except AttributeError:
        pass
    try:
        return len(e0[0])                # 2D list-of-lists (seq, dim)
    except (TypeError, IndexError, KeyError):
        pass
    try:
        return len(e0)                   # 1D vector
    except TypeError:
        return None


def check_identity(meta: dict, embedder) -> None:
    """Guard: refuse to query an index built with a different model (whose page
    vectors are meaningless against another model's query vectors)."""
    from colpali_rag.errors import IndexModelMismatch

    built = meta.get("model")
    if built and built != embedder.model_id:
        raise IndexModelMismatch(
            f"index was built with model {built!r} but COLPALI_MODEL is {embedder.model_id!r}. "
            f"Scores would be meaningless. Re-index, or set COLPALI_MODEL={built}."
        )
    sv = meta.get("schema_version")
    if sv is not None and sv != SCHEMA_VERSION:
        raise IndexModelMismatch(
            f"index schema v{sv} != current v{SCHEMA_VERSION}. Re-index: colpali-rag index <dir>."
        )


class _Base:
    def __init__(self, embedder, data_dir: str, artifacts=None):
        from colpali_rag.artifact_store import LocalArtifactStore

        self.embedder = embedder
        self.data_dir = Path(data_dir)
        self.artifacts = artifacts or LocalArtifactStore(self.data_dir)
        self.records: list[Page] = []
        self.ids: list[str] = []

    def _persist_images(self, records: list[Page], images: list) -> None:
        import io

        for rec, im in zip(records, images):
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")
            self.artifacts.put(image_key(page_id(rec.doc, rec.page)), buf.getvalue(), "image/png")

    def get_image(self, pid: str):
        from colpali_rag.artifact_store import load_bytes_as_image

        return load_bytes_as_image(self.artifacts.get(image_key(pid)))

    def image_url(self, pid: str, expires_in: int = 900):
        return self.artifacts.url_for(image_key(pid), expires_in)

    def __len__(self) -> int:
        return len(self.records)


class MemoryStore(_Base):
    backend = "memory"

    def __init__(self, embedder, data_dir: str, artifacts=None):
        super().__init__(embedder, data_dir, artifacts)
        self._embs = None

    def build_from(self, records, images, embs):
        self.records, self._embs = list(records), list(embs)
        self.ids = [page_id(r.doc, r.page) for r in records]
        self._persist_images(records, images)
        self.save()
        return self

    def search(self, query: str, top_k: int = 12):
        scores = self.embedder.score(query, self._embs)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.records[i], float(scores[i]), self.ids[i]) for i in order]

    def save(self):
        import torch

        self.data_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._embs, self.data_dir / "embeddings.pt")
        (self.data_dir / "records.json").write_text(
            json.dumps({"records": [asdict(r) for r in self.records], "ids": self.ids,
                        "model": self.embedder.model_id, "dim": _emb_dim(self._embs),
                        "schema_version": SCHEMA_VERSION, "backend": "memory"}, indent=2)
        )

    @classmethod
    def load(cls, embedder, data_dir: str, artifacts=None):
        import torch

        d = Path(data_dir)
        meta = json.loads((d / "records.json").read_text())
        check_identity(meta, embedder)
        store = cls(embedder, data_dir, artifacts)
        store.records = [Page(**r) for r in meta["records"]]
        store.ids = meta["ids"]
        store._embs = torch.load(d / "embeddings.pt", weights_only=False)
        return store


class QdrantStore(_Base):
    backend = "qdrant"

    def __init__(self, embedder, data_dir: str, url: str | None = None,
                 api_key: str | None = None, collection: str = "documents", artifacts=None):
        super().__init__(embedder, data_dir, artifacts)
        from qdrant_client import QdrantClient

        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=str(self.data_dir / "qdrant"))

    def build_from(self, records, images, embs, recreate: bool = True):
        from qdrant_client import models as qm

        self.records = list(records)
        self.ids = [page_id(r.doc, r.page) for r in records]
        self._persist_images(records, images)
        dim = len(self.embedder.page_to_list(embs[0])[0]) if embs else self.embedder.dim
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={"original": qm.VectorParams(
                    size=dim, distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM))},
            )
        points = [
            qm.PointStruct(id=i, vector={"original": self.embedder.page_to_list(e)},
                           payload={"doc": r.doc, "page": r.page, "text": r.text,
                                    "page_id": pid})
            for i, (r, e, pid) in enumerate(zip(records, embs, self.ids))
        ]
        self.client.upsert(self.collection, points=points)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "records.json").write_text(
            json.dumps({"records": [asdict(r) for r in self.records], "ids": self.ids,
                        "model": self.embedder.model_id, "dim": dim,
                        "schema_version": SCHEMA_VERSION, "backend": "qdrant",
                        "collection": self.collection}, indent=2)
        )
        return self

    def search(self, query: str, top_k: int = 12):
        qv = self.embedder.embed_query_raw(query)
        res = self.client.query_points(self.collection, query=qv, using="original",
                                       limit=top_k, with_payload=True).points
        out = []
        for p in res:
            pl = p.payload or {}
            out.append((Page(doc=pl.get("doc", ""), page=int(pl.get("page", 0)), text=pl.get("text", "")),
                        float(p.score), pl.get("page_id", "")))
        return out

    def __len__(self) -> int:
        try:
            return self.client.count(self.collection).count
        except Exception:
            return len(self.records)


def build_store(settings, embedder):
    """Factory from Settings.store."""
    from colpali_rag.artifact_store import build_artifact_store

    artifacts = build_artifact_store(settings)
    if settings.store == "qdrant":
        return QdrantStore(embedder, settings.data_dir, url=settings.qdrant_url,
                           api_key=settings.qdrant_api_key, collection=settings.collection,
                           artifacts=artifacts)
    return MemoryStore(embedder, settings.data_dir, artifacts=artifacts)


def load_store(settings, embedder):
    """Re-open a persisted store for serving/searching without re-indexing."""
    from colpali_rag.artifact_store import build_artifact_store

    artifacts = build_artifact_store(settings)
    if settings.store == "qdrant":
        store = QdrantStore(embedder, settings.data_dir, url=settings.qdrant_url,
                            api_key=settings.qdrant_api_key, collection=settings.collection,
                            artifacts=artifacts)
        # records/ids/images already persisted; reload the record list for the UI
        rec_path = Path(settings.data_dir) / "records.json"
        if rec_path.exists():
            meta = json.loads(rec_path.read_text())
            check_identity(meta, embedder)
            store.records = [Page(**r) for r in meta["records"]]
            store.ids = meta["ids"]
        return store
    return MemoryStore.load(embedder, settings.data_dir, artifacts=artifacts)
