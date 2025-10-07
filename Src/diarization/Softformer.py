from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nemo.collections.asr.models import SortformerEncLabelModel

from Transcription_parakeet.Src.model.model_cache import find_local_checkpoint
from Transcription_parakeet.Src.transcription.Parakeet import validate_paths
from Transcription_parakeet.config.logger_config import logger


DEFAULT_SORTFORMER_MODEL = "nvidia/diar_sortformer_4spk-v1"


def load_model(
    model_name: str = DEFAULT_SORTFORMER_MODEL,
) -> SortformerEncLabelModel:
    logger.info("Looking for a local diarization model for %s", model_name)
    local = find_local_checkpoint(model_name)
    if local is not None:
        logger.info("Loading Sortformer model from %s", local)
        model = SortformerEncLabelModel.restore_from(
            restore_path=str(local),
            strict=False,
        )
        if not hasattr(model, "diarize"):
            logger.warning("Local checkpoint not diarization: %s", local)
            logger.warning("Missing 'diarize'; will try pretrained model")
            try:
                model = SortformerEncLabelModel.from_pretrained(model_name)
            except Exception:
                logger.exception(
                    "Failed to load pretrained diarization model '%s' after "
                    "restoring local checkpoint %s.",
                    model_name,
                    local,
                )
                raise
        model.eval()
        return model

    logger.info(
        "Falling back to SortformerEncLabelModel.from_pretrained(%s)",
        model_name,
    )
    model = SortformerEncLabelModel.from_pretrained(model_name)
    model.eval()
    return model


def diarize_files(
    model: SortformerEncLabelModel,
    files: Sequence[str],
    *,
    batch_size: int = 1,
    include_tensor_outputs: bool = False,
    postprocessing_yaml: str | None = None,
    num_workers: int = 0,
) -> tuple[list[list[Any]], list[Any] | None]:
    logger.info("Running Sortformer diarization on %d file(s)", len(files))
    diarize_kwargs: dict[str, Any] = {
        "audio": list(files),
        "batch_size": batch_size,
        "include_tensor_outputs": include_tensor_outputs,
        "postprocessing_yaml": postprocessing_yaml,
        "num_workers": num_workers,
        "verbose": False,
    }

    output = model.diarize(**diarize_kwargs)
    if include_tensor_outputs:
        segments, tensors = output
        return list(segments), list(tensors)
    return list(output), None


def _parse_segment(entry: Any) -> tuple[float, float, int] | None:
    if isinstance(entry, str):
        parts = entry.strip().split()
        if len(parts) < 3:
            return None
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            return None
        speaker_token = parts[2]
        try:
            if speaker_token.lower().startswith("speaker_"):
                speaker_idx = int(speaker_token.split("_", 1)[1])
            else:
                speaker_idx = int(float(speaker_token))
        except ValueError:
            return None
        return start, end, speaker_idx

    if isinstance(entry, Sequence) and len(entry) >= 3:
        try:
            start = float(entry[0])
            end = float(entry[1])
            speaker_idx = int(entry[2])
        except (TypeError, ValueError):
            return None
        return start, end, speaker_idx
    return None


def _segments_to_dict(
    file_path: str,
    segments: Sequence[Any],
) -> dict[str, Any]:
    structured: list[dict[str, Any]] = []
    for entry in segments:
        parsed = _parse_segment(entry)
        if parsed is None:
            logger.debug(
                "Unexpected diarization entry for %s: %s",
                file_path,
                entry,
            )
            continue
        start, end, speaker_idx = parsed
        structured.append(
            {
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "speaker_index": speaker_idx,
                "speaker_label": f"SPEAKER_{speaker_idx:02d}",
            }
        )
    return {"file": file_path, "segments": structured}


def convert_results(
    paths: Sequence[str],
    segments: Sequence[Sequence[Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        results.append(_segments_to_dict(paths[idx], seg))
    return results


def print_results(
    paths: Sequence[str],
    segments: Sequence[Sequence[Any]],
) -> None:
    for idx, seg in enumerate(segments):
        file_path = paths[idx] if idx < len(paths) else f"<unknown-{idx}>"
        logger.info("----")
        logger.info("File: %s", file_path)
        logger.info("Detected speaker segments: %d", len(seg))
        # Use _parse_segment to safely handle different segment entry shapes
        preview_items: list[str] = []
        for entry in seg[:5]:
            parsed = _parse_segment(entry)
            if parsed is None:
                continue
            start, end, spk = parsed
            preview_items.append(
                f"[{float(start):.2f}-{float(end):.2f}] spk {int(spk)}"
            )
        preview = ", ".join(preview_items)
        if preview:
            logger.info("Preview: %s", preview)
        else:
            logger.info("No speech activity detected")


__all__ = [
    "DEFAULT_SORTFORMER_MODEL",
    "validate_paths",
    "load_model",
    "diarize_files",
    "convert_results",
    "print_results",
]
