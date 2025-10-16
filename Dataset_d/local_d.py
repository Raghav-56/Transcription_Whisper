from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
)


class LocalDatasetImporter:
    """Copy or link datasets from a local directory or archive."""

    def download(
        self,
        destination: Path,
        *,
        source: Path,
        overwrite: bool = False,
        symlink: bool = False,
    ) -> DownloadResult:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise DatasetDownloadError(f"Source path {source_path} does not exist.")
        if symlink:
            self._prepare_for_symlink(destination, overwrite)
            self._symlink(source_path, destination)
        else:
            ensure_destination(destination, overwrite=overwrite)
            self._copy(source_path, destination)
        details: Dict[str, object] = {
            "source": str(source_path),
            "symlink": symlink,
        }
        return DownloadResult(dataset_path=destination, details=details)

    def _copy(self, source: Path, destination: Path) -> None:
        logger.info("Copying dataset from %s", source)
        if source.is_dir():
            for item in source.iterdir():
                dest = destination / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        else:
            shutil.copy2(source, destination / source.name)

    def _symlink(self, source: Path, destination: Path) -> None:
        logger.info("Linking dataset from %s", source)
        destination.symlink_to(source, target_is_directory=source.is_dir())

    def _prepare_for_symlink(self, destination: Path, overwrite: bool) -> None:
        if destination.exists() or destination.is_symlink():
            if not overwrite:
                raise DatasetDownloadError(
                    (
                        "Destination already exists. Use overwrite=True "
                        "to replace it with a symlink."
                    )
                )
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
