from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
    require_requests,
    stream_response_to_file,
)


class HTTPDownloader:
    """Download datasets from generic HTTP(S) endpoints."""

    def __init__(self, *, timeout: int = 120) -> None:
        self._requests = require_requests()
        self._timeout = timeout
        self._session = self._requests.Session()

    def download(
        self,
        destination: Path,
        *,
        url: Optional[str] = None,
        urls: Optional[Iterable[str]] = None,
        filename: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        overwrite: bool = False,
        extract: bool = False,
        keep_archive: bool = False,
    ) -> DownloadResult:
        ensure_destination(destination, overwrite=overwrite)
        targets = self._normalise_urls(url, urls)
        saved_files: list[str] = []
        for index, target_url in enumerate(targets, start=1):
            logger.info("Downloading %s", target_url)
            inferred_name = self._infer_filename(target_url)
            target_name = self._pick_name(filename, inferred_name, index)
            file_path = destination / target_name
            self._stream_to_disk(target_url, headers, file_path)
            if extract:
                extracted = self._maybe_extract(file_path, destination, keep_archive)
                saved_files.extend(extracted)
            else:
                saved_files.append(str(file_path))
        details: Dict[str, Any] = {
            "urls": list(targets),
            "files": saved_files,
        }
        return DownloadResult(dataset_path=destination, details=details)

    def _normalise_urls(
        self,
        url: Optional[str],
        urls: Optional[Iterable[str]],
    ) -> list[str]:
        if url and urls:
            raise DatasetDownloadError(
                "Provide either 'url' or 'urls', not both, for HTTP downloads."
            )
        if url:
            return [url]
        if urls:
            normalised = [item for item in urls if item]
            if not normalised:
                raise DatasetDownloadError("No URLs provided.")
            return normalised
        raise DatasetDownloadError("A URL is required for HTTP downloads.")

    def _infer_filename(self, url: str) -> str:
        parsed = self._requests.utils.urlparse(url)
        name = os.path.basename(parsed.path) or "download.bin"
        return name

    def _pick_name(
        self,
        requested: Optional[str],
        inferred: str,
        index: int,
    ) -> str:
        if requested and requested.strip():
            if index == 1:
                return requested
            stem, suffix = os.path.splitext(requested)
            return f"{stem}_{index}{suffix}"
        if index == 1:
            return inferred
        stem, suffix = os.path.splitext(inferred)
        return f"{stem}_{index}{suffix}"

    def _stream_to_disk(
        self,
        url: str,
        headers: Optional[Mapping[str, str]],
        file_path: Path,
    ) -> None:
        with self._session.get(
            url,
            headers=headers,
            stream=True,
            timeout=self._timeout,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover
                snippet = response.text[:200]
                raise DatasetDownloadError(
                    f"HTTP download failed for {url}: {snippet}"
                ) from exc
            stream_response_to_file(response, file_path)

    def _maybe_extract(
        self,
        archive_path: Path,
        destination: Path,
        keep_archive: bool,
    ) -> list[str]:
        try:
            shutil.unpack_archive(str(archive_path), str(destination))
        except (shutil.ReadError, ValueError):
            return [str(archive_path)]
        if not keep_archive:
            archive_path.unlink(missing_ok=True)
        extracted: list[str] = [
            str(path) for path in destination.rglob("*") if path.is_file()
        ]
        return extracted or [str(destination)]

    def __enter__(self) -> "HTTPDownloader":  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        self._session.close()
