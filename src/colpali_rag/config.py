"""All configuration comes from environment variables (with sane defaults), so the
tool runs with zero setup and scales up by editing a `.env` — never the code.

Copy `.env.example` to `.env` and adjust. Nothing here hard-codes a model id,
endpoint, or key. Kept a plain dataclass on purpose (no extra dependency); values
are validated on read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {os.environ.get(name)!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_float_opt(name: str):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Minimal .env loader (no extra dependency). Does not override real env vars."""
    p = Path(path) if path else Path.cwd() / ".env"
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # --- retriever model ---
    model: str = "vidore/colSmol-500M"     # any ColVision id (colSmol / colqwen2-v1.0 / colnomic-7b …)
    family: str | None = None              # force a family for a new checkpoint (else auto from id)
    device: str = "cpu"                    # cpu | cuda | mps
    batch_size: int = 1

    # --- ingestion ---
    dpi: int = 150
    max_dim: int = 1600

    # --- store ---
    store: str = "memory"                  # memory | qdrant
    data_dir: str = "colpali_data"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "documents"

    # --- reranking (config-gated, OFF by default; needs the [rerank] extra + a GPU) ---
    rerank_enabled: bool = False
    rerank_backend: str = "monoqwen"       # none | monoqwen  (Apache-2.0 base)
    rerank_model: str = "lightonai/MonoQwen2-VL-v0.1"
    rerank_top_k: int = 10                 # how many first-stage candidates to rerank
    rerank_first_stage_n: int = 40         # how many the first stage retrieves
    rerank_device: str = "cuda"

    # --- optional answer generator (any OpenAI-compatible vision endpoint) ---
    vlm_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_model: str = "vlm"
    answer_top_k: int = 3
    answer_min_score: float | None = None  # gate: skip answering if top score below this

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            model=_env("COLPALI_MODEL", cls.model),
            family=os.environ.get("COLPALI_FAMILY") or None,
            device=_env("COLPALI_DEVICE", cls.device),
            batch_size=_env_int("COLPALI_BATCH_SIZE", cls.batch_size),
            dpi=_env_int("COLPALI_DPI", cls.dpi),
            max_dim=_env_int("COLPALI_MAX_DIM", cls.max_dim),
            store=_env("COLPALI_STORE", cls.store),
            data_dir=_env("COLPALI_DATA_DIR", cls.data_dir),
            qdrant_url=os.environ.get("QDRANT_URL") or None,
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=_env("COLPALI_COLLECTION", cls.collection),
            rerank_enabled=_env_bool("RERANK_ENABLED", cls.rerank_enabled),
            rerank_backend=_env("RERANK_BACKEND", cls.rerank_backend),
            rerank_model=_env("RERANK_MODEL", cls.rerank_model),
            rerank_top_k=_env_int("RERANK_TOP_K", cls.rerank_top_k),
            rerank_first_stage_n=_env_int("RERANK_FIRST_STAGE_N", cls.rerank_first_stage_n),
            rerank_device=_env("RERANK_DEVICE", cls.rerank_device),
            vlm_base_url=os.environ.get("VLM_BASE_URL") or None,
            vlm_api_key=os.environ.get("VLM_API_KEY") or None,
            vlm_model=_env("VLM_MODEL", cls.vlm_model),
            answer_top_k=_env_int("ANSWER_TOP_K", cls.answer_top_k),
            answer_min_score=_env_float_opt("ANSWER_MIN_SCORE"),
            host=_env("COLPALI_HOST", cls.host),
            port=_env_int("COLPALI_PORT", cls.port),
        )

    @property
    def vlm_enabled(self) -> bool:
        return bool(self.vlm_base_url)


def get_settings() -> Settings:
    return Settings.from_env()
