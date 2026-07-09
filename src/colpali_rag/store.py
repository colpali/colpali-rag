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


def page_id(doc: str, page: int) -> str:
    return f"{doc}::p{page}"


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)


class _Base:
    def __init__(self, embedder, data_dir: str):
        self.embedder = embedder
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.records: list[Page] = []
        self.ids: list[str] = []

    def _persist_images(self, records: list[Page], images: list) -> None:
        self.img_dir.mkdir(parents=True, exist_ok=True)
        for rec, im in zip(records, images):
            im.convert("RGB").save(self.img_dir / f"{_safe(page_id(rec.doc, rec.page))}.png")

    def get_image(self, pid: str):
        from PIL import Image

        p = self.img_dir / f"{_safe(pid)}.png"
        return Image.open(p) if p.exists() else None

    def __len__(self) -> int:
        return len(self.records)


class MemoryStore(_Base):
    backend = "memory"

    def __init__(self, embedder, data_dir: str):
        super().__init__(embedder, data_dir)
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
                        "model": self.embedder.model_id, "backend": "memory"}, indent=2)
        )

    @classmethod
    def load(cls, embedder, data_dir: str):
        import torch

        d = Path(data_dir)
        meta = json.loads((d / "records.json").read_text())
        store = cls(embedder, data_dir)
        store.records = [Page(**r) for r in meta["records"]]
        store.ids = meta["ids"]
        store._embs = torch.load(d / "embeddings.pt", weights_only=False)
        return store


class QdrantStore(_Base):
    backend = "qdrant"

    def __init__(self, embedder, data_dir: str, url: str | None = None,
                 api_key: str | None = None, collection: str = "documents"):
        super().__init__(embedder, data_dir)
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
                        "model": self.embedder.model_id, "backend": "qdrant",
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
    if settings.store == "qdrant":
        return QdrantStore(embedder, settings.data_dir, url=settings.qdrant_url,
                           api_key=settings.qdrant_api_key, collection=settings.collection)
    return MemoryStore(embedder, settings.data_dir)


def load_store(settings, embedder):
    """Re-open a persisted store for serving/searching without re-indexing."""
    if settings.store == "qdrant":
        store = QdrantStore(embedder, settings.data_dir, url=settings.qdrant_url,
                            api_key=settings.qdrant_api_key, collection=settings.collection)
        # records/ids/images already persisted; reload the record list for the UI
        rec_path = Path(settings.data_dir) / "records.json"
        if rec_path.exists():
            meta = json.loads(rec_path.read_text())
            store.records = [Page(**r) for r in meta["records"]]
            store.ids = meta["ids"]
        return store
    return MemoryStore.load(embedder, settings.data_dir)
