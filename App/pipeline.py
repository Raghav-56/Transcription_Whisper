from __future__ import annotations

from typing import Any

from Transcription_parakeet.Src.transcription.Parakeet import (
    DEFAULT_PARAKEET_MODEL,
    _result_to_dict,
    load_model as load_transcription_model,
    print_results as print_transcription_results,
    transcribe_files,
    validate_paths,
)
from Transcription_parakeet.Src.diarization.Softformer import (
    DEFAULT_SORTFORMER_MODEL,
    convert_results as convert_diarization_results,
    diarize_files,
    load_model as load_diarization_model,
    print_results as print_diarization_results,
)
from Transcription_parakeet.Src.model.model_cache import (
    default_output_dir,
    find_local_checkpoint,
)
from Transcription_parakeet.Src.model.save_model import download_and_save
from Transcription_parakeet.config.logger_config import logger


def _ensure_local_model(model_name: str) -> None:
    local = find_local_checkpoint(model_name)
    if local is not None and local.exists():
        return

    out_dir = default_output_dir()
    try:
        path = download_and_save(model_name, out_dir)
        logger.info("Model cached at %s", path)
    except Exception as exc:  # pragma: no cover - best effort warning
        logger.warning("Unable to cache model %s: %s", model_name, exc)


def _merge_diarization(
    transcripts: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
) -> None:
    by_file = {entry["file"]: entry for entry in transcripts}
    for item in diarization:
        file_key = item.get("file")
        segments = item.get("segments")
        if file_key is None:
            continue
        target = by_file.get(file_key)
        if target is not None:
            target["speakers"] = segments or []
        else:
            new_entry = {
                "file": file_key,
                "text": "",
                "score": None,
                "timestamps": None,
                "speakers": segments or [],
            }
            transcripts.append(new_entry)
            by_file[file_key] = new_entry


def run_pipeline(
    files: list[str],
    model: str | None = None,
    batch_size: int = 1,
    *,
    diarize: bool = True,
    diarization_model: str | None = None,
    diarization_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    paths = validate_paths(files)
    if not paths:
        raise SystemExit(2)

    model_name = model or DEFAULT_PARAKEET_MODEL
    _ensure_local_model(model_name)
    m = load_transcription_model(model_name)
    results = transcribe_files(m, paths, batch_size=batch_size)
    print_transcription_results(paths, results)

    structured = [_result_to_dict(paths[i], r) for i, r in enumerate(results)]

    if not diarize:
        return structured

    diar_model_name = diarization_model or DEFAULT_SORTFORMER_MODEL
    _ensure_local_model(diar_model_name)
    diarizer = load_diarization_model(diar_model_name)
    diar_segments, _ = diarize_files(
        diarizer,
        paths,
        batch_size=diarization_batch_size or batch_size,
    )
    print_diarization_results(paths, diar_segments)
    diar_structured = convert_diarization_results(paths, diar_segments)
    _merge_diarization(structured, diar_structured)
    return structured
