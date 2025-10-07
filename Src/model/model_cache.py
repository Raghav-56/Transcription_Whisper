from __future__ import annotations

from pathlib import Path
from typing import Sequence


_MODEL_DIR_NAMES: tuple[str, ...] = ("Models", "models")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_model_filename(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_") + ".nemo"


def candidate_dirs(extra_dirs: Sequence[Path] | None = None) -> list[Path]:
    base = [repo_root() / name for name in _MODEL_DIR_NAMES]
    if extra_dirs:
        base.extend(extra_dirs)
    return base


def find_local_checkpoint(
    model_name: str,
    *,
    search_dirs: Sequence[Path] | None = None,
) -> Path | None:
    safe_name = safe_model_filename(model_name)
    dirs = candidate_dirs(search_dirs)
    for directory in dirs:
        candidate = directory / safe_name
        if candidate.exists():
            return candidate
        if directory.exists():
            for path in directory.glob("*.nemo"):
                return path
    return None


def default_output_dir() -> Path:
    target = repo_root() / "models"
    target.mkdir(parents=True, exist_ok=True)
    return target


__all__ = [
    "candidate_dirs",
    "default_output_dir",
    "find_local_checkpoint",
    "repo_root",
    "safe_model_filename",
]
