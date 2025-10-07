from __future__ import annotations

import importlib
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Tuple

from Transcription_parakeet.config.logger_config import logger


TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes -> 16-bit linear PCM
TARGET_EXTENSION = ".wav"


def _resolve_executables() -> Tuple[str, str]:
    """Return paths for (ffmpeg, ffprobe)."""
    try:
        static_ffmpeg = importlib.import_module("static_ffmpeg")
    except ImportError:
        return "ffmpeg", "ffprobe"

    try:
        static_ffmpeg.add_paths()
        ffmpeg, ffprobe = (
            static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        )
        return str(ffmpeg), str(ffprobe)
    except (AttributeError, RuntimeError, OSError) as exc:
        logger.debug("static_ffmpeg helper failed: %s", exc)
        return "ffmpeg", "ffprobe"


def _ffprobe_inspect(path: str) -> dict[str, Any] | None:
    """Return sample_rate/channels/sample_fmt for the first audio stream.

    Returns None when probing fails or ffprobe is unavailable.
    """
    _, ffprobe = _resolve_executables()

    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,channels,sample_rate,sample_fmt",
        "-of",
        "json",
        str(path),
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
        out = proc.stdout.decode("utf-8")
        data = json.loads(out)
        streams = data.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") == "audio":
                sr = stream.get("sample_rate")
                ch = stream.get("channels")
                fmt = stream.get("sample_fmt")
                return {
                    "sample_rate": int(sr) if sr else 0,
                    "channels": int(ch) if ch else 0,
                    "sample_fmt": fmt,
                }
        return None
    except subprocess.CalledProcessError as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
        return None


def _needs_conversion(path: str, info: dict[str, Any] | None) -> bool:
    """Decide if conversion is required based on probe info."""
    if info is None:
        return True
    extension = Path(path).suffix.lower()
    if extension != TARGET_EXTENSION:
        return True
    sample_rate = int(info.get("sample_rate", 0) or 0)
    if sample_rate != TARGET_SAMPLE_RATE:
        return True
    channels = int(info.get("channels", 0) or 0)
    if channels != TARGET_CHANNELS:
        return True
    sample_fmt = str(info.get("sample_fmt") or "")
    if not sample_fmt.startswith("s16"):
        return True
    return False


def _make_output_path(temp_dir: Path, original: Path, index: int) -> Path:
    base_name = original.stem or f"audio_{index:03d}"
    candidate = temp_dir / f"{base_name}_parakeet.wav"
    if candidate.exists():
        candidate = temp_dir / f"{base_name}_{index:03d}_parakeet.wav"
    return candidate


def _ffmpeg_convert(input_path: str, output_path: Path) -> None:
    """Convert file to target WAV using ffmpeg CLI and raise on failure."""
    ffmpeg, _ = _resolve_executables()

    cmd = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@contextlib.contextmanager
def prepare_audio_files(paths: Sequence[str]) -> Iterator[list[str]]:
    """Yield a list of audio paths encoded for NeMo models.

    Converted files are written to a temporary directory cleaned up when the
    context exits.
    """

    if not paths:
        yield []
        return

    temp_dir: Path | None = None
    processed: list[str] = []

    try:
        for idx, original_path in enumerate(paths):
            if not os.path.exists(original_path):
                logger.warning(
                    "Skipping missing audio file during preparation: %s",
                    original_path,
                )
                continue

            info = _ffprobe_inspect(original_path)
            if not _needs_conversion(original_path, info):
                processed.append(original_path)
                continue

            if temp_dir is None:
                temp_dir = Path(tempfile.mkdtemp(prefix="parakeet_audio_"))
                logger.debug(
                    "Created temporary directory for audio conversions: %s",
                    temp_dir,
                )

            output_path = _make_output_path(temp_dir, Path(original_path), idx)
            _ffmpeg_convert(original_path, output_path)
            logger.info(
                "Converted audio '%s' -> '%s' (mono %dkHz, 16-bit)",
                original_path,
                output_path,
                TARGET_SAMPLE_RATE // 1000,
            )
            processed.append(str(output_path))

        yield processed
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Removed temporary audio directory: %s", temp_dir)


__all__ = [
    "prepare_audio_files",
    "TARGET_SAMPLE_RATE",
    "TARGET_CHANNELS",
    "TARGET_SAMPLE_WIDTH",
    "TARGET_EXTENSION",
]
