"""Public API for transcription helpers."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time guard for type checkers
    from .Parakeet import (
        main,
        print_results,
        transcribe_files,
        validate_paths,
        load_model,
    )

__all__ = [
    "validate_paths",
    "load_model",
    "transcribe_files",
    "print_results",
    "main",
    "Parakeet",
]

_parakeet: ModuleType | None = None


def _load() -> ModuleType:
    global _parakeet
    if _parakeet is None:
        _parakeet = import_module(f"{__name__}.Parakeet")
    return _parakeet


def __getattr__(name: str) -> Any:
    module = _load()
    if name == "Parakeet":
        return module
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - bubble up original error
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from exc


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
