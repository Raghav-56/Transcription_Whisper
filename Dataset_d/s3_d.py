from __future__ import annotations

import shutil
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
)


class S3Downloader:
    """Download datasets stored in Amazon S3."""

    def __init__(
        self,
        *,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        self._boto3 = self._load_boto3()
        session_args: Dict[str, Any] = {}
        if profile_name:
            session_args["profile_name"] = profile_name
        if region_name:
            session_args["region_name"] = region_name
        self._session = self._boto3.session.Session(**session_args)
        self._client = self._session.client("s3")

    def download(
        self,
        destination: Path,
        *,
        bucket: str,
        key: str,
        version_id: Optional[str] = None,
        filename: Optional[str] = None,
        extract: bool = False,
        overwrite: bool = False,
        keep_archive: bool = False,
    ) -> DownloadResult:
        ensure_destination(destination, overwrite=overwrite)
        target_name = filename or Path(key).name or "s3_object"
        target_path = destination / target_name
        logger.info("Downloading s3://%s/%s to %s", bucket, key, target_path)
        extra_args = {"VersionId": version_id} if version_id else None
        try:
            self._client.download_file(
                bucket,
                key,
                str(target_path),
                ExtraArgs=extra_args,
            )
        except TypeError:  # ExtraArgs only when present
            self._client.download_file(bucket, key, str(target_path))
        except Exception as exc:  # pragma: no cover
            raise DatasetDownloadError(
                f"Failed to download s3://{bucket}/{key}: {exc}"
            ) from exc
        files = [str(target_path)]
        if extract:
            files = self._extract_archive(
                target_path,
                destination,
                keep_archive,
            )
        details = {
            "bucket": bucket,
            "key": key,
            "files": files,
        }
        return DownloadResult(dataset_path=destination, details=details)

    def _extract_archive(
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
        return [str(path) for path in destination.rglob("*") if path.is_file()]

    def _load_boto3(self) -> Any:
        try:
            return importlib.import_module("boto3")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise DatasetDownloadError(
                "Install 'boto3' to download datasets from Amazon S3."
            ) from exc
