"""Native-Qdrant launcher: platform asset mapping + idempotent install (no network in tests)."""

import types

import pytest

from colpali_rag import qdrant_server as q


def _asset_for(monkeypatch, system, machine):
    monkeypatch.setattr(q.platform, "system", lambda: system)
    monkeypatch.setattr(q.platform, "machine", lambda: machine)
    return q._asset_name()


def test_asset_name_per_platform(monkeypatch):
    assert _asset_for(monkeypatch, "Windows", "AMD64") == ("qdrant-x86_64-pc-windows-msvc.zip", "qdrant.exe")

    mac_arm = _asset_for(monkeypatch, "Darwin", "arm64")
    assert mac_arm[1] == "qdrant" and "aarch64-apple-darwin" in mac_arm[0]
    assert "x86_64-apple-darwin" in _asset_for(monkeypatch, "Darwin", "x86_64")[0]

    assert "aarch64-unknown-linux-musl" in _asset_for(monkeypatch, "Linux", "aarch64")[0]
    assert "x86_64-unknown-linux-gnu" in _asset_for(monkeypatch, "Linux", "x86_64")[0]


def test_ensure_binary_skips_download_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(q.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(q.platform, "machine", lambda: "arm64")

    def boom(*a, **k):
        raise AssertionError("must not download when binary + web UI already present")

    monkeypatch.setattr(q, "_download", boom)
    (tmp_path / "qdrant").write_text("#!/bin/sh\n")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "index.html").write_text("<html></html>")

    exe = q.ensure_binary(tmp_path)
    assert exe == tmp_path / "qdrant"


def test_ensure_binary_downloads_and_lays_out_static(tmp_path, monkeypatch):
    # simulate the two downloads with local fixtures — verifies extract + dist->static rename
    import tarfile
    import zipfile

    monkeypatch.setattr(q.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(q.platform, "machine", lambda: "arm64")

    def fake_download(url, dest, progress):
        if url.endswith(".tar.gz"):
            (tmp_path / "_qdrant_bin").write_text("binary")
            with tarfile.open(dest, "w:gz") as t:
                t.add(tmp_path / "_qdrant_bin", arcname="qdrant")
        else:  # dist-qdrant.zip -> a top-level dist/ folder
            with zipfile.ZipFile(dest, "w") as z:
                z.writestr("dist/index.html", "<html></html>")
                z.writestr("dist/assets/app.js", "//")

    monkeypatch.setattr(q, "_download", fake_download)
    server = tmp_path / "srv"
    exe = q.ensure_binary(server)
    assert exe.exists() and exe.name == "qdrant"
    assert (server / "static" / "index.html").exists()      # dist/ renamed to static/
    assert not (server / "dist").exists()
    assert not (server / "dist-qdrant.zip").exists()         # temp archive cleaned up


def test_run_server_uses_data_dir_subfolder(tmp_path, monkeypatch):
    # run_server should install under <data_dir>/qdrant-server and exec there, not re-embed anything
    calls = {}
    monkeypatch.setattr(q, "ensure_binary", lambda d, p=None: (calls.setdefault("dir", d), d / "qdrant")[1])
    monkeypatch.setattr(q.subprocess, "run", lambda cmd, cwd: calls.setdefault("cwd", cwd))
    q.run_server(types.SimpleNamespace(data_dir=str(tmp_path)))
    assert calls["dir"] == tmp_path / "qdrant-server"
    assert calls["cwd"] == str(tmp_path / "qdrant-server")
