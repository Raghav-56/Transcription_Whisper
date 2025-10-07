"""Public API for the diarization helpers."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

_softformer = import_module(f"{__name__}.Softformer")

__all__ = [
    "DEFAULT_SORTFORMER_MODEL",
    "load_model",
    "diarize_files",
    "convert_results",
    "print_results",
    "validate_paths",
]

DEFAULT_SORTFORMER_MODEL = _softformer.DEFAULT_SORTFORMER_MODEL
load_model = _softformer.load_model
diarize_files = _softformer.diarize_files
convert_results = _softformer.convert_results
print_results = _softformer.print_results
validate_paths = _softformer.validate_paths

# Allow both `Softformer` and `softformer` module paths for compatibility.
sys.modules[f"{__name__}.Softformer"] = _softformer
sys.modules[f"{__name__}.softformer"] = _softformer


def __getattr__(name: str) -> Any:
    return getattr(_softformer, name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
