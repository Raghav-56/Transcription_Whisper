"""Public entrypoints for the Src package.

This module keeps imports light by lazily exposing the main subpackages and
the legacy ``Parakeet`` module alias used by earlier callers.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["transcription", "diarization", "model", "Parakeet"]


def _load_submodule(name: str) -> ModuleType:
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __getattr__(name: str) -> ModuleType:
    if name in ("transcription", "diarization", "model"):
        return _load_submodule(name)
    if name == "Parakeet":
        module = import_module(f"{__name__}.transcription.Parakeet")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
