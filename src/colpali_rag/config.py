"""All configuration comes from environment variables (with sane defaults), so the
tool runs with zero setup and scales up by editing a `.env` — never the code.

Copy `.env.example` to `.env` and adjust. Nothing here hard-codes a model id,
endpoint, or key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


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
    model: str = "vidore/colSmol-500M"     # any ColVision id (colSmol / colqwen2-v1.0 / colnomic-*)
    device: str = "cpu"                    # cpu | cuda | mps
    batch_size: int = 1

    # --- store ---
    store: str = "memory"                  # memory | qdrant
    data_dir: str = "colpali_data"         # where the memory store persists its index
    qdrant_url: str | None = None          # e.g. http://localhost:6333 ; None => embedded
    qdrant_api_key: str | None = None
    collection: str = "documents"

    # --- optional answer generator (any OpenAI-compatible vision endpoint) ---
    # Vendor-neutral: point it at whatever chat/completions server you run.
    # Leave vlm_base_url empty to disable answer generation (search still works).
    vlm_base_url: str | None = None        # e.g. http://localhost:8000/v1
    vlm_api_key: str | None = None
    vlm_model: str = "vlm"                 # whatever model name your endpoint expects

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            model=_env("COLPALI_MODEL", cls.model),
            device=_env("COLPALI_DEVICE", cls.device),
            batch_size=int(_env("COLPALI_BATCH_SIZE", str(cls.batch_size))),
            store=_env("COLPALI_STORE", cls.store),
            data_dir=_env("COLPALI_DATA_DIR", cls.data_dir),
            qdrant_url=os.environ.get("QDRANT_URL") or None,
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=_env("COLPALI_COLLECTION", cls.collection),
            vlm_base_url=os.environ.get("VLM_BASE_URL") or None,
            vlm_api_key=os.environ.get("VLM_API_KEY") or None,
            vlm_model=_env("VLM_MODEL", cls.vlm_model),
            host=_env("COLPALI_HOST", cls.host),
            port=int(_env("COLPALI_PORT", str(cls.port))),
        )

    @property
    def vlm_enabled(self) -> bool:
        return bool(self.vlm_base_url)


def get_settings() -> Settings:
    return Settings.from_env()
