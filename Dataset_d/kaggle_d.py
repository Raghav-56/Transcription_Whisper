from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
)


class KaggleDownloader:
    """Download datasets from Kaggle via the Kaggle CLI."""

    def __init__(self, *, kaggle_executable: Optional[str] = None) -> None:
        self._executable = kaggle_executable or shutil.which("kaggle")
        if not self._executable:
            raise DatasetDownloadError(
                (
                    "Kaggle CLI not found. Install the 'kaggle' package and "
                    "set up API credentials."
                )
            )

    def download(
        self,
        destination: Path,
        *,
        dataset: Optional[str] = None,
        competition: Optional[str] = None,
        files: Optional[Iterable[str]] = None,
        unzip: bool = True,
        overwrite: bool = False,
        keep_archive: bool = False,
        extra_args: Optional[Iterable[str]] = None,
    ) -> DownloadResult:
        ensure_destination(destination, overwrite=overwrite)
        command = self._build_command(
            dataset=dataset,
            competition=competition,
            files=files,
            unzip=unzip,
            destination=destination,
            extra_args=extra_args,
        )
        logger.info("Running Kaggle CLI: %s", " ".join(command))
        result = subprocess.run(  # noqa: S603,S607
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise DatasetDownloadError(f"Kaggle CLI failed: {message}")
        if unzip:
            self._cleanup_archives(destination, keep_archive)
        details = self._build_details(destination, command)
        return DownloadResult(dataset_path=destination, details=details)

    def _build_command(
        self,
        dataset: Optional[str],
        competition: Optional[str],
        files: Optional[Iterable[str]],
        unzip: bool,
        destination: Path,
        extra_args: Optional[Iterable[str]],
    ) -> list[str]:
        if bool(dataset) == bool(competition):
            raise DatasetDownloadError(
                (
                    "Specify exactly one of 'dataset' or 'competition' for "
                    "Kaggle downloads."
                )
            )
        command: list[str] = [self._executable or "kaggle"]
        if dataset:
            command.extend(["datasets", "download"])
            identifier = dataset
        else:
            command.extend(["competitions", "download"])
            identifier = competition  # type: ignore[assignment]
        command.extend(["-p", str(destination)])
        if unzip:
            command.append("--unzip")
        if files:
            for item in files:
                command.extend(["-f", item])
        if extra_args:
            command.extend(list(extra_args))
        command.append(identifier or "")
        return command

    def _cleanup_archives(self, destination: Path, keep_archive: bool) -> None:
        for archive in destination.glob("*.zip"):
            if keep_archive:
                continue
            archive.unlink(missing_ok=True)

    def _build_details(
        self,
        destination: Path,
        command: list[str],
    ) -> Dict[str, Any]:
        files = [str(path) for path in destination.iterdir() if path.is_file()]
        return {
            "command": command,
            "file_count": len(files),
            "files": files,
        }
