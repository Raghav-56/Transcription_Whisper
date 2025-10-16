from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import importlib

DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset download fails."""


@dataclass(slots=True)
class DownloadResult:
    """Lightweight container for results returned by dataset downloaders."""

    dataset_path: Path
    details: Dict[str, Any] | None = field(default=None)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the result."""
        payload: Dict[str, Any] = {"dataset_path": str(self.dataset_path)}
        if self.details:
            payload["details"] = self.details
        return payload


def ensure_destination(path: Path, overwrite: bool = False) -> Path:
    """Ensure a clean destination directory for a download."""
    if path.exists():
        if not overwrite:
            raise DatasetDownloadError(
                (
                    f"Destination {path} already exists. "
                    "Pass overwrite=True to replace it."
                )
            )
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _load_requests() -> Any:
    try:
        return importlib.import_module("requests")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise DatasetDownloadError(
            "The 'requests' package is required for this download source. "
            "Install it with 'pip install requests'."
        ) from exc


def require_requests() -> Any:
    """Ensure the optional dependency ``requests`` is available."""
    return _load_requests()


def stream_response_to_file(response: Any, target_path: Path) -> Path:
    """Persist a streaming HTTP response to disk."""
    require_requests()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(DEFAULT_CHUNK_SIZE):
            if chunk:
                handle.write(chunk)
    return target_path
