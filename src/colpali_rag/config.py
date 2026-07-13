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
        v = v.strip()
        if v[:1] not in ("'", '"'):
            # unquoted value: drop an inline comment introduced by whitespace + '#', so
            # `KEY=value   # note` parses to `value` (and `KEY=a#b` keeps the literal '#').
            for sep in (" #", "\t#"):
                if sep in v:
                    v = v.split(sep, 1)[0].rstrip()
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # --- retriever model ---
    model: str = "vidore/colSmol-500M"     # any ColVision id (colSmol / colqwen2-v1.0 / colnomic-7b …)
    family: str | None = None              # force a family for a new checkpoint (else auto from id)
    device: str = "cpu"                    # cpu | cuda | mps
    batch_size: int = 1
    adapter_path: str = ""                 # optional PEFT/LoRA adapter dir/id (a domain fine-tune)
    adapter_merge: bool = False            # merge the adapter into the base weights at load time
    norm_check: bool = True                # warn at index time if page embeddings aren't unit-norm
    norm_tol: float = 1e-3                 # tolerance for the unit-norm check (mean |‖E‖-1|)

    # --- ingestion ---
    dpi: int = 150
    max_dim: int = 1600

    # --- store ---
    store: str = "memory"                  # memory | qdrant
    data_dir: str = "colpali_data"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "documents"

    # --- artifact (page image) storage: local (default) | s3-compatible object storage ---
    storage_backend: str = "local"         # local | s3
    storage_bucket: str | None = None
    storage_endpoint_url: str | None = None  # for S3-compatible object stores
    storage_region: str = "auto"
    storage_access_key: str | None = None
    storage_secret_key: str | None = None
    storage_prefix: str = ""
    storage_addressing: str = "path"       # path | virtual
    storage_url_ttl: int = 900
    storage_serve_mode: str = "proxy"      # proxy (safe default) | presigned

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
    answer_structured: bool = False        # return {answer, claims:[{text,pages,confidence}]}
    answer_structured_mode: str = "auto"   # auto | json_schema | json_object | prompt
    answer_max_retries: int = 1
    query_rewrite: bool = False             # rewrite a follow-up into a standalone retrieval query

    # --- faithfulness check (optional; needs a SEPARATE judge endpoint) ---
    faithfulness_gate: str = "off"         # off | flag | withhold
    faithfulness_min_score: float | None = None
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_model: str = ""
    judge_allow_same_endpoint: bool = False

    # --- closed-vocabulary constraint for structured outputs (optional; OFF by default) ---
    # When an uploaded table carries the configured id column, its rows are compiled into a
    # closed vocabulary and every proposed node is projected onto it. All names/values are
    # injected at runtime — nothing here hard-codes a schema. Empty id column => feature off.
    catalog_gate: str = "off"              # off | flag | withhold
    catalog_id_col: str = ""               # header of the id column; empty => constraint OFF
    catalog_name_cols: str = ""            # comma-separated alias/label columns (match-only)
    catalog_required_col: str = ""         # column marking rows that must appear (=> Req)
    catalog_iface_cols: str = ""           # seam for edge feasibility (parsed, unused for now)
    catalog_match_threshold: float = 0.84  # accept a fuzzy match at/above this similarity
    catalog_match_margin: float = 0.08     # ...only if it beats the runner-up by this much
    catalog_withhold_max_drop: float = 0.5 # withhold if this fraction (or more) of nodes drop
    catalog_repair_max: int = 1            # times to re-prompt the model to fix catalog violations

    # --- tabular display caps (constraint channel is never capped; these bound the LLM view) ---
    tabular_max_preview_rows: int = 40
    tabular_max_cols: int = 24
    tabular_max_cell: int = 80

    # --- hybrid visual + lexical retrieval (optional; OFF by default) ---
    # Fuse the visual MaxSim ranking with a keyword ranking over extracted page text (RRF), so
    # exact identifiers aren't lost to a blurred page. Degrades to visual-only on scanned corpora.
    hybrid_enabled: bool = False
    hybrid_kappa: int = 60                  # RRF constant
    hybrid_fetch: int = 100                 # candidates pulled from each channel before fusion
    hybrid_min_coverage: float = 0.5        # skip the lexical channel if fewer pages have text
    hybrid_ngram_min: int = 3
    hybrid_ngram_max: int = 5

    # --- logging + per-generation run logs ---
    log_level: str = "INFO"                # DEBUG | INFO | WARNING | ERROR (the studio step trace is INFO)
    run_log_dir: str = ""                  # if set, write a JSON + text summary of each studio run here

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
            adapter_path=_env("COLPALI_ADAPTER_PATH", cls.adapter_path),
            adapter_merge=_env_bool("COLPALI_ADAPTER_MERGE", cls.adapter_merge),
            norm_check=_env_bool("COLPALI_NORM_CHECK", cls.norm_check),
            norm_tol=float(_env("COLPALI_NORM_TOL", str(cls.norm_tol))),
            dpi=_env_int("COLPALI_DPI", cls.dpi),
            max_dim=_env_int("COLPALI_MAX_DIM", cls.max_dim),
            store=_env("COLPALI_STORE", cls.store),
            data_dir=_env("COLPALI_DATA_DIR", cls.data_dir),
            qdrant_url=os.environ.get("QDRANT_URL") or None,
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=_env("COLPALI_COLLECTION", cls.collection),
            storage_backend=_env("STORAGE_BACKEND", cls.storage_backend),
            storage_bucket=os.environ.get("STORAGE_BUCKET") or None,
            storage_endpoint_url=os.environ.get("STORAGE_ENDPOINT_URL") or None,
            storage_region=_env("STORAGE_REGION", cls.storage_region),
            storage_access_key=os.environ.get("STORAGE_ACCESS_KEY") or None,
            storage_secret_key=os.environ.get("STORAGE_SECRET_KEY") or None,
            storage_prefix=_env("STORAGE_PREFIX", cls.storage_prefix),
            storage_addressing=_env("STORAGE_ADDRESSING", cls.storage_addressing),
            storage_url_ttl=_env_int("STORAGE_URL_TTL", cls.storage_url_ttl),
            storage_serve_mode=_env("STORAGE_SERVE_MODE", cls.storage_serve_mode),
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
            answer_structured=_env_bool("ANSWER_STRUCTURED", cls.answer_structured),
            answer_structured_mode=_env("ANSWER_STRUCTURED_MODE", cls.answer_structured_mode),
            answer_max_retries=_env_int("ANSWER_MAX_RETRIES", cls.answer_max_retries),
            query_rewrite=_env_bool("COLPALI_QUERY_REWRITE", cls.query_rewrite),
            faithfulness_gate=_env("FAITHFULNESS_GATE", cls.faithfulness_gate),
            faithfulness_min_score=_env_float_opt("FAITHFULNESS_MIN_SCORE"),
            judge_base_url=os.environ.get("JUDGE_BASE_URL") or None,
            judge_api_key=os.environ.get("JUDGE_API_KEY") or None,
            judge_model=_env("JUDGE_MODEL", cls.judge_model),
            judge_allow_same_endpoint=_env_bool("JUDGE_ALLOW_SAME_ENDPOINT", cls.judge_allow_same_endpoint),
            catalog_gate=_env("COLPALI_CATALOG_GATE", cls.catalog_gate),
            catalog_id_col=_env("CATALOG_ID_COL", cls.catalog_id_col),
            catalog_name_cols=_env("CATALOG_NAME_COLS", cls.catalog_name_cols),
            catalog_required_col=_env("CATALOG_REQUIRED_COL", cls.catalog_required_col),
            catalog_iface_cols=_env("CATALOG_IFACE_COLS", cls.catalog_iface_cols),
            catalog_match_threshold=float(_env("CATALOG_MATCH_THRESHOLD", str(cls.catalog_match_threshold))),
            catalog_match_margin=float(_env("CATALOG_MATCH_MARGIN", str(cls.catalog_match_margin))),
            catalog_withhold_max_drop=float(_env("CATALOG_WITHHOLD_MAX_DROP", str(cls.catalog_withhold_max_drop))),
            catalog_repair_max=_env_int("CATALOG_REPAIR_MAX", cls.catalog_repair_max),
            tabular_max_preview_rows=_env_int("TABULAR_MAX_PREVIEW_ROWS", cls.tabular_max_preview_rows),
            tabular_max_cols=_env_int("TABULAR_MAX_COLS", cls.tabular_max_cols),
            tabular_max_cell=_env_int("TABULAR_MAX_CELL", cls.tabular_max_cell),
            hybrid_enabled=_env_bool("COLPALI_HYBRID_ENABLED", cls.hybrid_enabled),
            hybrid_kappa=_env_int("COLPALI_HYBRID_KAPPA", cls.hybrid_kappa),
            hybrid_fetch=_env_int("COLPALI_HYBRID_FETCH", cls.hybrid_fetch),
            hybrid_min_coverage=float(_env("COLPALI_HYBRID_MIN_COVERAGE", str(cls.hybrid_min_coverage))),
            hybrid_ngram_min=_env_int("COLPALI_HYBRID_NGRAM_MIN", cls.hybrid_ngram_min),
            hybrid_ngram_max=_env_int("COLPALI_HYBRID_NGRAM_MAX", cls.hybrid_ngram_max),
            log_level=_env("COLPALI_LOG_LEVEL", cls.log_level),
            run_log_dir=_env("COLPALI_RUN_LOG_DIR", cls.run_log_dir),
            host=_env("COLPALI_HOST", cls.host),
            port=_env_int("COLPALI_PORT", cls.port),
        )

    @property
    def vlm_enabled(self) -> bool:
        return bool(self.vlm_base_url)


def setup_logging(level: str | None = None) -> None:
    """Configure logging once (idempotent). Level from the argument or COLPALI_LOG_LEVEL, default
    INFO — which surfaces the studio's per-generation step trace. Safe to call from every entrypoint."""
    import logging

    lvl = (level or os.environ.get("COLPALI_LOG_LEVEL") or "INFO").upper()
    num = getattr(logging, lvl, logging.INFO)
    logging.basicConfig(level=num, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("colpali_rag").setLevel(num)


def get_settings() -> Settings:
    return Settings.from_env()
