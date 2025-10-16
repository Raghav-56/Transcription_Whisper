from __future__ import annotations

import importlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from config.logger_config import logger
from Dataset_d.common import (
    DatasetDownloadError,
    DownloadResult,
    ensure_destination,
    require_requests,
    stream_response_to_file,
)


class GitHubDownloader:
    """Handle dataset downloads stored on GitHub."""

    api_root = "https://api.github.com"

    def __init__(self, *, timeout: int = 60) -> None:
        require_requests()
        self._timeout = timeout
        self._requests = self._load_requests()
        self._session = self._requests.Session()

    def download(
        self,
        destination: Path,
        *,
        repo: str,
        ref: str = "main",
        subdir: Optional[str] = None,
        release_tag: Optional[str] = None,
        asset_name: Optional[str] = None,
        token: Optional[str] = None,
        overwrite: bool = False,
        extract: bool = True,
        keep_archive: bool = False,
    ) -> DownloadResult:
        """Download a dataset from a GitHub repo or release."""
        ensure_destination(destination, overwrite=overwrite)
        logger.info(
            "Downloading from GitHub repo=%s ref=%s release_tag=%s asset=%s",
            repo,
            ref,
            release_tag,
            asset_name,
        )
        headers = self._build_headers(token)
        if release_tag:
            if not asset_name:
                raise DatasetDownloadError(
                    "asset_name is required when release_tag is provided."
                )
            asset_path, original_name = self._download_release_asset(
                repo,
                release_tag,
                asset_name,
                headers,
            )
            return self._handle_downloaded_file(
                asset_path,
                destination,
                original_name,
                extract=extract,
                keep_archive=keep_archive,
            )
        archive_path = self._download_repo_archive(repo, ref, headers)
        archive_label = f"{Path(repo).name}-{ref}.zip"
        return self._extract_archive(
            archive_path,
            destination,
            subdir=subdir,
            keep_archive=keep_archive,
            extract=extract,
            archive_name=archive_label,
        )

    def _build_headers(self, token: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _download_repo_archive(
        self,
        repo: str,
        ref: str,
        headers: Dict[str, str],
    ) -> Path:
        url = f"{self.api_root}/repos/{repo}/zipball/{ref}"
        logger.debug("Fetching GitHub archive from %s", url)
        with self._session.get(
            url,
            headers=headers,
            stream=True,
            timeout=self._timeout,
        ) as response:
            self._raise_for_status(response, url)
            handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            tmp_path = Path(handle.name)
            handle.close()
            stream_response_to_file(response, tmp_path)
        return tmp_path

    def _download_release_asset(
        self,
        repo: str,
        release_tag: str,
        asset_name: str,
        headers: Dict[str, str],
    ) -> tuple[Path, str]:
        url = f"{self.api_root}/repos/{repo}/releases/tags/{release_tag}"
        logger.debug("Resolving GitHub release at %s", url)
        with self._session.get(
            url,
            headers=headers,
            timeout=self._timeout,
        ) as resp:
            self._raise_for_status(resp, url)
            data = resp.json()
        asset = next(
            (item for item in data.get("assets", []) if item.get("name") == asset_name),
            None,
        )
        if not asset:
            raise DatasetDownloadError(
                f"Asset '{asset_name}' not found in release {release_tag}."
            )
        asset_url = asset.get("url")
        if not asset_url:
            raise DatasetDownloadError("Release asset URL is missing.")
        download_headers = {
            **headers,
            "Accept": "application/octet-stream",
        }
        logger.debug("Downloading GitHub release asset from %s", asset_url)
        with self._session.get(
            asset_url,
            headers=download_headers,
            stream=True,
            timeout=self._timeout,
        ) as response:
            self._raise_for_status(response, asset_url)
            handle = tempfile.NamedTemporaryFile(
                suffix=f"_{asset_name}",
                delete=False,
            )
            tmp_path = Path(handle.name)
            handle.close()
            stream_response_to_file(response, tmp_path)
        return tmp_path, asset_name

    def _handle_downloaded_file(
        self,
        file_path: Path,
        destination: Path,
        original_name: str,
        *,
        extract: bool,
        keep_archive: bool,
    ) -> DownloadResult:
        if extract and file_path.suffix == ".zip":
            result = self._extract_archive(
                file_path,
                destination,
                subdir=None,
                keep_archive=keep_archive,
                extract=True,
                archive_name=original_name,
            )
            return result
        target = destination / original_name
        destination.mkdir(parents=True, exist_ok=True)
        if file_path != target:
            shutil.move(str(file_path), target)
        return DownloadResult(
            dataset_path=destination,
            details={"files": [str(target)]},
        )

    def _extract_archive(
        self,
        archive_path: Path,
        destination: Path,
        *,
        subdir: Optional[str],
        keep_archive: bool,
        extract: bool,
        archive_name: Optional[str] = None,
    ) -> DownloadResult:
        if not extract:
            final_name = archive_name or archive_path.name
            final_path = destination / final_name
            shutil.move(str(archive_path), final_path)
            return DownloadResult(
                dataset_path=destination,
                details={
                    "archive": str(final_path),
                    "extracted": False,
                },
            )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(tmp_dir)
            extracted_root = self._find_single_root(Path(tmp_dir))
            source_path = extracted_root
            if subdir:
                source_path = extracted_root / Path(subdir)
                if not source_path.exists():
                    raise DatasetDownloadError(
                        f"Sub-directory '{subdir}' not found in archive."
                    )
            ensure_destination(destination, overwrite=True)
            self._copy_contents(source_path, destination)
        archive_target = destination / (archive_name or archive_path.name)
        if keep_archive:
            shutil.move(str(archive_path), archive_target)
        elif archive_path.exists():
            archive_path.unlink()
        return DownloadResult(
            dataset_path=destination,
            details={"extracted": True},
        )

    def _copy_contents(self, source: Path, target: Path) -> None:
        for item in source.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    def _find_single_root(self, temp_root: Path) -> Path:
        candidates = [path for path in temp_root.iterdir() if path.is_dir()]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise DatasetDownloadError("Archive content is empty.")
        return temp_root

    def _raise_for_status(self, response: Any, url: str) -> None:
        try:
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - surface details upwards
            content = response.text[:200]
            raise DatasetDownloadError(
                f"GitHub request to {url} failed: {content}"
            ) from exc

    def _load_requests(self) -> Any:
        try:
            return importlib.import_module("requests")
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise DatasetDownloadError(
                "The 'requests' package is required for GitHub downloads. "
                "Install it with 'pip install requests'."
            ) from exc
