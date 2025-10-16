from __future__ import annotations

import importlib
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
    stream_response_to_file,
)

CONFIRM_PATTERN = re.compile("download_warning[0-9A-Za-z_]+")


class GoogleDriveDownloader:
    """Download helper for datasets stored on Google Drive."""

    base_url = "https://docs.google.com/uc"

    def __init__(self, *, timeout: int = 120) -> None:
        self._timeout = timeout
        self._requests = self._load_requests()
        self._session = self._requests.Session()

    def download(
        self,
        destination: Path,
        *,
        file_id: str,
        file_name: Optional[str] = None,
        extract: bool = False,
        overwrite: bool = False,
        keep_archive: bool = False,
    ) -> DownloadResult:
        ensure_destination(destination, overwrite=overwrite)
        with self._fetch(file_id) as response:
            inferred_name = self._infer_filename(response) or f"{file_id}.bin"
            target_name = file_name or inferred_name
            target_path = destination / target_name
            logger.info(
                "Downloading Google Drive file %s to %s",
                file_id,
                target_path,
            )
            stream_response_to_file(response, target_path)
        details: Dict[str, Any] = {
            "file_id": file_id,
            "files": [str(target_path)],
        }
        if extract:
            extracted = self._maybe_extract(
                target_path,
                destination,
                keep_archive,
            )
            details.update(extracted)
        return DownloadResult(dataset_path=destination, details=details)

    def _fetch(self, file_id: str) -> Any:
        params = {"id": file_id, "export": "download"}
        response = self._session.get(
            self.base_url,
            params=params,
            stream=True,
            timeout=self._timeout,
        )
        token = self._confirm_token(response)
        if token:
            response.close()
            params["confirm"] = token
            response = self._session.get(
                self.base_url,
                params=params,
                stream=True,
                timeout=self._timeout,
            )
        self._raise_for_status(response, file_id)
        return response

    def _confirm_token(self, response: Any) -> Optional[str]:
        for key, value in response.cookies.items():
            if CONFIRM_PATTERN.match(key):
                return value
        return None

    def _infer_filename(self, response: Any) -> Optional[str]:
        disposition = response.headers.get("Content-Disposition")
        if not disposition:
            return None
        match = re.search('filename="?([^";]+)"?', disposition)
        if match:
            return match.group(1)
        return None

    def _maybe_extract(
        self,
        archive_path: Path,
        destination: Path,
        keep_archive: bool,
    ) -> Dict[str, Any]:
        suffix = archive_path.suffix.lower()
        extracted_files: list[str] = []
        if suffix == ".zip":
            with zipfile.ZipFile(archive_path) as zip_ref:
                zip_ref.extractall(destination)
                extracted_files = [
                    str(destination / name) for name in zip_ref.namelist()
                ]
        elif suffix in {".tar", ".gz", ".tgz", ".bz2"}:
            mode = "r:gz" if suffix in {".gz", ".tgz"} else "r"
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(destination)
                extracted_files = [
                    str(destination / member.name) for member in tar_ref.getmembers()
                ]
        else:
            logger.warning(
                "Extraction requested but unsupported archive type %s",
                suffix,
            )
            return {"extracted": False}
        if not keep_archive:
            archive_path.unlink()
        return {"extracted": True, "files": extracted_files or None}

    def _raise_for_status(self, response: Any, file_id: str) -> None:
        try:
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - propagate details
            snippet = response.text[:200]
            raise DatasetDownloadError(
                f"Failed to download Google Drive file {file_id}: {snippet}"
            ) from exc

    def _load_requests(self) -> Any:
        try:
            return importlib.import_module("requests")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DatasetDownloadError(
                (
                    "The 'requests' package is required for Google Drive "
                    "downloads. Install it with 'pip install requests'."
                )
            ) from exc

    def __enter__(self) -> "GoogleDriveDownloader":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self._session.close()
