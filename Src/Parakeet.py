import os
import sys
import subprocess
from pathlib import Path
from typing import Any

import static_ffmpeg
from static_ffmpeg import run

# Option A: add to PATH for the current process
static_ffmpeg.add_paths()  # downloads ffmpeg+ffprobe on first call

from pydub import AudioSegment

# Option B: get explicit paths and set pydub
# ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
# AudioSegment.converter = ffmpeg_path
# AudioSegment.ffprobe = ffprobe_path

# now run existing code that uses pydub
from nemo.collections.asr.models import ASRModel

from Transcription_parakeet.config.logger_config import logger


def validate_paths(paths: list[str]) -> list[str]:
    """Return existing paths from the provided list."""
    files: list[str] = []
    for pth in paths:
        if not os.path.exists(pth):
            logger.warning("path not found: %s", pth)
        else:
            files.append(pth)
    return files


def load_model(model_name: str = "nvidia/parakeet-tdt-0.6b-v2") -> ASRModel:
    """Load and return the NeMo ASR model."""
    logger.info("Looking for a local .nemo model for %s", model_name)

    def _find_local_nemo(name: str) -> Path | None:
        repo_root = Path(__file__).resolve().parents[1]
        # Prefer these folders (Windows is case-insensitive, support both)
        candidate_dirs = [repo_root / "Models", repo_root / "models"]
        safe = name.replace("/", "__").replace(" ", "_") + ".nemo"

        # check explicit filename first
        for d in candidate_dirs:
            p = d / safe
            if p.exists():
                return p

        # check for any .nemo in the candidate folders
        for d in candidate_dirs:
            if d.exists():
                for p in d.glob("*.nemo"):
                    return p

        return None

    # Try to find an existing local .nemo first
    local = _find_local_nemo(model_name)
    if local is not None:
        logger.info("Loading model from local file %s", local)
        return ASRModel.restore_from(restore_path=str(local))

    # No local model found: try to run the local save script (if present)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(__file__).resolve().parent / "save_model.py",
        repo_root / "Model" / "save_model.py",
        repo_root / "Src" / "save_model.py",
    ]

    save_script = None
    for c in candidates:
        if c.exists():
            save_script = c
            break

    if save_script is not None:
        out_dir = Path(__file__).resolve().parents[1] / "models"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(save_script),
                    "--model",
                    model_name,
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning("save_model script failed: %s", exc)
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.warning("Unable to run save script: %s", exc)

        # Attempt to find the created file
        local = _find_local_nemo(model_name)
        if local is not None:
            logger.info("Loading model from newly saved file %s", local)
            return ASRModel.restore_from(restore_path=str(local))

    # Last resort: load from the hub
    logger.info("Falling back to ASRModel.from_pretrained(%s)", model_name)
    return ASRModel.from_pretrained(model_name)


def transcribe_files(model: ASRModel, files: list[str], batch_size: int = 1):
    """Transcribe files with the given model and return results."""
    logger.info("Transcribing %d file(s)...", len(files))
    return model.transcribe(
        files, batch_size=batch_size, return_hypotheses=True, timestamps=True
    )


def _result_to_dict(file_path: str, result: Any) -> dict:
    """Convert a model result object into a structured dict.

    The shape is intentionally minimal and serializable:
    {
        "file": str,
        "text": str,
        "score": float | None,
        "timestamps": dict | None
    }
    """
    text = getattr(result, "text", "")
    # Some NeMo result objects may include confidence/score
    score = getattr(result, "score", None)
    timestamps = getattr(result, "timestamp", None)
    # If timestamps is not serializable, attempt to coerce
    try:
        if timestamps is not None:
            # convert mapping-like timestamp to a plain dict if possible
            timestamps = dict(timestamps)
    except Exception:
        timestamps = None
    return {
        "file": file_path,
        "text": text,
        "score": score,
        "timestamps": timestamps,
    }


def print_results(files: list[str], results) -> None:
    """Log a human-friendly transcription summary to the configured logger.

    This improves readability for console output while still keeping the
    structured results available from the pipeline.
    """
    for idx, r in enumerate(results):
        file_path = files[idx] if idx < len(files) else f"<unknown-{idx}>"
        logger.info("----")
        logger.info("File: %s", file_path)
        logger.info("Transcription:")
        # Log a short preview and the full text at DEBUG
        text = getattr(r, "text", "")
        preview = (text[:200] + "...") if len(text) > 200 else text
        logger.info("%s", preview)
        logger.debug("Full text for %s: %s", file_path, text)

        ts = getattr(r, "timestamp", None)
        if ts:
            try:
                keys = list(ts.keys())
            except Exception:
                keys = []
            logger.info("Timestamp keys: %s", keys)
        else:
            logger.info("No timestamp info available for this result.")


def main(argv: list[str] | None = None) -> int:
    """Backward-compatible entrypoint that mirrors previous CLI behaviour.

    argv: list of arguments (excluding program name). Returns exit code.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("No input files provided. Exiting.", file=sys.stderr)
        return 2

    model_name = "nvidia/parakeet-tdt-0.6b-v2"
    batch_size = 1

    # Simple parsing: treat any arg starting with --model= or --batch-size=
    paths: list[str] = []
    for a in argv:
        if a.startswith("--model="):
            model_name = a.split("=", 1)[1]
        elif a.startswith("--batch-size="):
            try:
                batch_size = int(a.split("=", 1)[1])
            except Exception:
                batch_size = 1
        else:
            paths.append(a)

    files = validate_paths(paths)
    if not files:
        print("No valid input files provided. Exiting.", file=sys.stderr)
        return 2

    model = load_model(model_name)
    results = transcribe_files(model, files, batch_size=batch_size)
    print_results(files, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
