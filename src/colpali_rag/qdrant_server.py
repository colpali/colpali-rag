"""Run a native Qdrant server (with its web dashboard) WITHOUT Docker or npm.

`colpali-rag qdrant` downloads a pinned prebuilt Qdrant binary + the web-UI static bundle once
into `<data_dir>/qdrant-server/`, then launches the server in the foreground. Everything stays
local; the dashboard is at http://localhost:6333/dashboard and the API at http://localhost:6333.

The binary is the official release for your OS/arch. The web UI ships as a separate `dist-qdrant.zip`
(the raw server binary doesn't embed it), so we drop it into `./static/` next to the binary — which
is exactly where Qdrant looks for the dashboard when run from that directory.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Pinned so the download is reproducible. Bump deliberately.
QDRANT_VERSION = "v1.18.2"
WEBUI_VERSION = "v0.2.15"
_REL = "https://github.com/qdrant/qdrant/releases/download"
_WEBUI = "https://github.com/qdrant/qdrant-web-ui/releases/download"


def _asset_name() -> tuple[str, str]:
    """(release asset filename, executable name) for the current OS/arch."""
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        return "qdrant-x86_64-pc-windows-msvc.zip", "qdrant.exe"
    if system == "Darwin":
        arch = "aarch64" if arm else "x86_64"
        return f"qdrant-{arch}-apple-darwin.tar.gz", "qdrant"
    # Linux: aarch64 ships musl only; x86_64 has a gnu build.
    return (("qdrant-aarch64-unknown-linux-musl.tar.gz", "qdrant") if arm
            else ("qdrant-x86_64-unknown-linux-gnu.tar.gz", "qdrant"))


def _download(url: str, dest: Path, progress) -> None:
    progress(f"downloading {url}")
    ctx = None
    try:                                         # prefer certifi's CA bundle if present
        import ssl

        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = None
    try:
        with urllib.request.urlopen(url, context=ctx) as r, open(dest, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
    except Exception:
        # common on python.org macOS builds (no CA certs installed); curl uses the OS trust store
        if shutil.which("curl") is None:
            raise
        progress("  (urllib TLS failed — falling back to curl)")
        subprocess.run(["curl", "-fSL", "-o", str(dest), url], check=True)


def _extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(into, filter="data")   # py3.12+: safe extraction
            except TypeError:
                t.extractall(into)


def ensure_binary(server_dir: Path, progress=lambda m: None) -> Path:
    """Return the path to a ready-to-run Qdrant binary under `server_dir`, downloading the binary
    and the web-UI static bundle on first use. Idempotent: skips anything already present."""
    server_dir = Path(server_dir)
    asset, exe = _asset_name()
    exe_path = server_dir / exe

    if not exe_path.exists():
        server_dir.mkdir(parents=True, exist_ok=True)
        tmp = server_dir / asset
        _download(f"{_REL}/{QDRANT_VERSION}/{asset}", tmp, progress)
        _extract(tmp, server_dir)          # binary lands at server_dir/<exe>
        tmp.unlink(missing_ok=True)
        exe_path.chmod(0o755)
        progress(f"installed qdrant {QDRANT_VERSION} → {exe_path}")

    static = server_dir / "static"
    if not (static / "index.html").exists():
        zip_path = server_dir / "dist-qdrant.zip"
        _download(f"{_WEBUI}/{WEBUI_VERSION}/dist-qdrant.zip", zip_path, progress)
        _extract(zip_path, server_dir)     # yields server_dir/dist/
        dist = server_dir / "dist"
        if dist.exists():
            if static.exists():
                shutil.rmtree(static)
            dist.rename(static)            # Qdrant serves the dashboard from ./static
        zip_path.unlink(missing_ok=True)
        progress(f"installed web dashboard {WEBUI_VERSION} → {static}")

    return exe_path


def run_server(settings, progress=lambda m: None):
    """Download (once) and run the Qdrant server in the foreground. Blocks until Ctrl-C.

    Run from `server_dir` so Qdrant finds ./static (dashboard) and keeps its own storage there,
    separate from the ColPali index."""
    server_dir = Path(settings.data_dir) / "qdrant-server"
    exe = ensure_binary(server_dir, progress)
    progress("starting Qdrant  →  API http://localhost:6333   dashboard http://localhost:6333/dashboard")
    progress("(leave this running; in another terminal: colpali-rag migrate, then colpali-rag serve)")
    # cwd=server_dir so ./static and ./storage resolve there.
    return subprocess.run([str(exe)], cwd=str(server_dir))
