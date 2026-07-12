"""Pluggable artifact storage for page images — local filesystem (default, zero
deps) or S3-compatible object storage (AWS S3 / MinIO / R2 / … behind one client).

Vendor-neutral: only the common Put/Get/Head/Delete/presigned subset that behaves
identically everywhere. Vectors live in the vector store; this is only for the page
image bytes, so the app (and Qdrant) stay stateless while images live in storage.

Config uses generic names (STORAGE_*), never a cloud/vendor word. `local` is the
default and its on-disk layout is byte-identical to the pre-adapter path, so existing
indexes keep working with no migration.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol, runtime_checkable

from colpali_rag.errors import ArtifactStoreError


@runtime_checkable
class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    def get(self, key: str) -> bytes | None: ...        # None ONLY if the key is missing
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def url_for(self, key: str, expires_in: int = 900) -> str | None: ...  # None => serve via app


class LocalArtifactStore:
    """Default. Writes `root/<key>`; `url_for` returns None so the app proxies bytes."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        # keys are already sanitized by store.image_key(); reject traversal defensively.
        # Resolve and confirm the result stays under root — this catches ".." in any
        # separator, absolute keys (which pathlib would let escape root), and symlinks.
        root = self.root.resolve()
        p = (self.root / key).resolve()
        if p != root and root not in p.parents:
            raise ArtifactStoreError(f"unsafe key {key!r}")
        return p

    def put(self, key, data, content_type=None):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key):
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def exists(self, key):
        return self._path(key).exists()

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def url_for(self, key, expires_in=900):
        return None


class S3ArtifactStore:
    """S3-compatible object storage via boto3 (the `[s3]` extra). One class covers
    AWS S3, MinIO, and R2 via endpoint_url + path addressing + region 'auto'.

    Correctness: a genuinely missing object returns None; auth/network errors RAISE
    ArtifactStoreError (never masked as 'missing', which would hide misconfiguration)."""

    def __init__(self, bucket, endpoint_url=None, region="auto", access_key=None,
                 secret_key=None, addressing="path", prefix=""):
        try:
            import boto3
            from botocore.config import Config
        except Exception as e:  # noqa: BLE001
            raise ArtifactStoreError("S3 storage needs the [s3] extra: pip install '.[s3]'") from e
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            config=Config(s3={"addressing_style": addressing}),
        )

    def _k(self, key):
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key, data, content_type=None):
        try:
            kw = {"ContentType": content_type} if content_type else {}
            self._client.put_object(Bucket=self.bucket, Key=self._k(key), Body=data, **kw)
        except Exception as e:  # noqa: BLE001
            raise ArtifactStoreError(f"put {key} failed: {type(e).__name__}: {e}") from e

    def get(self, key):
        from botocore.exceptions import ClientError

        try:
            return self._client.get_object(Bucket=self.bucket, Key=self._k(key))["Body"].read()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None                      # genuinely missing
            raise ArtifactStoreError(f"get {key} failed ({code})") from e   # auth/net → raise
        except Exception as e:  # noqa: BLE001
            raise ArtifactStoreError(f"get {key} failed: {type(e).__name__}: {e}") from e

    def exists(self, key):
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise ArtifactStoreError(f"head {key} failed") from e

    def delete(self, key):
        try:
            self._client.delete_object(Bucket=self.bucket, Key=self._k(key))
        except Exception as e:  # noqa: BLE001
            raise ArtifactStoreError(f"delete {key} failed: {type(e).__name__}: {e}") from e

    def url_for(self, key, expires_in=900):
        try:
            return self._client.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": self._k(key)},
                ExpiresIn=expires_in)
        except Exception as e:  # noqa: BLE001
            raise ArtifactStoreError(f"presign {key} failed: {type(e).__name__}: {e}") from e


def build_artifact_store(settings) -> ArtifactStore:
    """Factory from Settings.storage_backend. `local` (default) mirrors the on-disk
    layout under data_dir, so existing indexes need no migration."""
    if getattr(settings, "storage_backend", "local") == "s3":
        return S3ArtifactStore(
            bucket=settings.storage_bucket, endpoint_url=settings.storage_endpoint_url,
            region=settings.storage_region, access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key, addressing=settings.storage_addressing,
            prefix=settings.storage_prefix)
    return LocalArtifactStore(root=settings.data_dir)


def load_bytes_as_image(data: bytes | None):
    from PIL import Image

    return Image.open(io.BytesIO(data)) if data else None
