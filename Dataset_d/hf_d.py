from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
)


class HuggingFaceDownloader:
    """Download datasets hosted on Hugging Face."""

    def __init__(self) -> None:
        self._hub = self._load_hub()

    def download(
        self,
        destination: Path,
        *,
        repo_id: str,
        repo_type: str = "dataset",
        revision: str = "main",
        token: Optional[str] = None,
        allow_patterns: Optional[Iterable[str]] = None,
        ignore_patterns: Optional[Iterable[str]] = None,
        overwrite: bool = False,
        local_dir_use_symlinks: bool = False,
    ) -> DownloadResult:
        ensure_destination(destination, overwrite=overwrite)
        logger.info(
            "Snapshotting %s (type=%s, revision=%s) to %s",
            repo_id,
            repo_type,
            revision,
            destination,
        )
        try:
            resolved_path = self._hub.snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                allow_patterns=(list(allow_patterns) if allow_patterns else None),
                ignore_patterns=(list(ignore_patterns) if ignore_patterns else None),
                local_dir=str(destination),
                local_dir_use_symlinks=local_dir_use_symlinks,
            )
        except Exception as exc:  # pragma: no cover - bubble up message
            raise DatasetDownloadError(
                f"Failed to download {repo_id} from Hugging Face: {exc}"
            ) from exc
        details = self._build_details(
            destination,
            repo_id,
            revision,
            resolved_path,
        )
        return DownloadResult(dataset_path=destination, details=details)

    def _build_details(
        self,
        destination: Path,
        repo_id: str,
        revision: str,
        resolved_path: str,
    ) -> Dict[str, Any]:
        file_count = sum(1 for _ in destination.rglob("*") if _.is_file())
        return {
            "repo_id": repo_id,
            "revision": revision,
            "resolved_path": resolved_path,
            "file_count": file_count,
        }

    def _load_hub(self) -> Any:
        try:
            return importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DatasetDownloadError(
                ("Install 'huggingface_hub' to download datasets from " "Hugging Face.")
            ) from exc
