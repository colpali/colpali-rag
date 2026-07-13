"""Follow-up query rewriting — turn a context-dependent request into a standalone retrieval
query using session history. Model-free (the model call is stubbed)."""

import types


def test_rewrite_query_no_history_returns_original():
    from colpali_rag.generator import rewrite_query

    assert rewrite_query([], "what is X?", base_url="x", model="m") == "what is X?"
    assert rewrite_query(None, "q", base_url="x", model="m") == "q"
    assert rewrite_query(["prior"], "   ", base_url="x", model="m") == "   "   # empty query untouched


def test_rewrite_query_uses_model_output(monkeypatch):
    from colpali_rag import generator

    monkeypatch.setattr(generator, "_post_chat",
                        lambda *a, **k: {"choices": [{"message": {"content": '"standalone Q"'}}]})
    out = generator.rewrite_query(["max voltage?"], "and the current?", base_url="x", model="m")
    assert out == "standalone Q"                     # surrounding quotes stripped


def test_rewrite_query_falls_back_on_error(monkeypatch):
    from colpali_rag import generator

    def boom(*a, **k):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(generator, "_post_chat", boom)
    assert generator.rewrite_query(["prior"], "orig", base_url="x", model="m") == "orig"


def test_query_rewrite_wired_into_retrieval(monkeypatch):
    from colpali_rag.studio import generate as gen
    from colpali_rag.studio.spec import Block, DiagramSpec

    class _Store:
        records = [1]
        ids = ["p1"]

        def __init__(self):
            self.seen = []

        def __len__(self):
            return 1

        def search(self, q, top_k=12):
            self.seen.append(q)
            return []

        def get_image(self, pid):
            return None

    store = _Store()
    monkeypatch.setattr("colpali_rag.generator.rewrite_query",
                        lambda history, query, **k: "STANDALONE:" + query)
    monkeypatch.setattr(gen, "_llm_diagram",
                        lambda *a, **k: DiagramSpec(title="T", blocks=[Block("a", "A")], connections=[]))

    s = types.SimpleNamespace(vlm_enabled=True, query_rewrite=True, vlm_base_url="x",
                              vlm_api_key="", vlm_model="m", answer_structured_mode="auto",
                              answer_max_retries=1, catalog_id_col="", catalog_gate="off",
                              hybrid_enabled=False, run_log_dir="",
                              tabular_max_preview_rows=40, tabular_max_cols=24, tabular_max_cell=80)
    gen.generate_diagram("and the current?", store=store, settings=s, history=["max voltage?"])
    assert store.seen and store.seen[0].startswith("STANDALONE:")   # retrieval used the rewrite
