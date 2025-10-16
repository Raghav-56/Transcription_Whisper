from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Type

from config.logger_config import logger
from Dataset_d.common import DatasetDownloadError, DownloadResult
from Dataset_d.drive import GoogleDriveDownloader
from Dataset_d.gh_d import GitHubDownloader
from Dataset_d.hf_d import HuggingFaceDownloader
from Dataset_d.http_d import HTTPDownloader
from Dataset_d.kaggle_d import KaggleDownloader
from Dataset_d.local_d import LocalDatasetImporter
from Dataset_d.s3_d import S3Downloader

DATASETS_ROOT = Path(__file__).resolve().parent / "data"

DownloaderType = Type[Any]
DOWNLOADERS: Mapping[str, DownloaderType] = {
    "github": GitHubDownloader,
    "gh": GitHubDownloader,
    "huggingface": HuggingFaceDownloader,
    "hf": HuggingFaceDownloader,
    "google_drive": GoogleDriveDownloader,
    "gdrive": GoogleDriveDownloader,
    "kaggle": KaggleDownloader,
    "http": HTTPDownloader,
    "https": HTTPDownloader,
    "url": HTTPDownloader,
    "s3": S3Downloader,
    "aws": S3Downloader,
    "local": LocalDatasetImporter,
    "filesystem": LocalDatasetImporter,
}


def get_datasets_root() -> Path:
    """Return the default root directory used for dataset downloads."""
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    return DATASETS_ROOT


def available_sources() -> list[str]:
    """Return the list of supported download source keys."""
    return sorted(set(DOWNLOADERS.keys()))


def download_dataset(
    source: str,
    *,
    dataset_name: str | None = None,
    target_root: Path | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> DownloadResult:
    """Download a dataset from the requested source."""
    key = source.lower()
    downloader_cls = DOWNLOADERS.get(key)
    if not downloader_cls:
        raise DatasetDownloadError(f"Unknown dataset source '{source}'.")
    destination_root = (
        Path(target_root) if target_root is not None else get_datasets_root()
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / dataset_name if dataset_name else destination_root
    logger.info(
        "Preparing to download dataset via %s into %s",
        key,
        destination,
    )
    downloader = downloader_cls()
    result = downloader.download(
        destination,
        overwrite=overwrite,
        **kwargs,
    )
    logger.info("Dataset download complete: %s", result.dataset_path)
    return result
