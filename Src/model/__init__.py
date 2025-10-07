"""Convenience exports for model utilities."""

from __future__ import annotations

from importlib import import_module

model_cache = import_module(f"{__name__}.model_cache")
save_model = import_module(f"{__name__}.save_model")

candidate_dirs = model_cache.candidate_dirs
default_output_dir = model_cache.default_output_dir
find_local_checkpoint = model_cache.find_local_checkpoint
repo_root = model_cache.repo_root
safe_model_filename = model_cache.safe_model_filename

ensure_dir = save_model.ensure_dir
download_and_save = save_model.download_and_save
app = save_model.app

__all__ = [
    "model_cache",
    "save_model",
    "candidate_dirs",
    "default_output_dir",
    "find_local_checkpoint",
    "repo_root",
    "safe_model_filename",
    "ensure_dir",
    "download_and_save",
    "app",
]
