from __future__ import annotations

from typing import Any
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from Transcription_parakeet.Src.transcription.Parakeet import (
    DEFAULT_PARAKEET_MODEL,
    _result_to_dict,
    load_model as load_transcription_model,
    print_results as print_transcription_results,
    transcribe_files,
    validate_paths,
)
from Transcription_parakeet.Src.preprocessing.file_format import (
    prepare_audio_files,
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


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _make_run_directory() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = _timestamp()
    run_dir = RESULTS_DIR / base_name
    counter = 1
    while run_dir.exists():
        run_dir = RESULTS_DIR / f"{base_name}_{counter:02d}"
        counter += 1
    run_dir.mkdir()
    return run_dir


def _write_combined_json(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    output_path = run_dir / "combined.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)
    logger.info("Saved combined JSON results to %s", output_path)


def _write_transcription_csv(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    fieldnames = ["AudioFileName", "Transcript", "Confidence"]
    output_path = run_dir / "ASR_00.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in entries:
            score = item.get("score")
            writer.writerow(
                {
                    "AudioFileName": Path(item.get("file", "")).name,
                    "Transcript": item.get("text", ""),
                    "Confidence": "" if score is None else score,
                }
            )
    logger.info("Saved transcription CSV results to %s", output_path)


def _format_ts(value: float) -> str:
    return f"{value:06.3f}"


def _write_diarization_csv(run_dir: Path, entries: list[dict[str, Any]]) -> None:
    fieldnames = ["AudioFileName", "Speaker", "Confidence", "StartTS", "EndTS"]
    rows: list[dict[str, Any]] = []
    for item in entries:
        speakers = item.get("speakers") or []
        if not speakers:
            continue
        filename = Path(item.get("file", "")).name
        for segment in speakers:
            start = segment.get("start")
            end = segment.get("end")
            speaker_label = segment.get("speaker_label") or segment.get("speaker_index")
            if start is None or end is None:
                continue
            rows.append(
                {
                    "AudioFileName": filename,
                    "Speaker": speaker_label,
                    "Confidence": segment.get("confidence", 100),
                    "StartTS": _format_ts(float(start)),
                    "EndTS": _format_ts(float(end)),
                }
            )

    if not rows:
        logger.info("No diarization segments to save; skipping SD_00.csv")
        return

    output_path = run_dir / "SD_00.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved diarization CSV results to %s", output_path)


def _persist_results(entries: list[dict[str, Any]]) -> None:
    run_dir = _make_run_directory()
    _write_combined_json(run_dir, entries)
    _write_transcription_csv(run_dir, entries)
    _write_diarization_csv(run_dir, entries)
    logger.info("Results saved under %s", run_dir)


def _ensure_local_model(model_name: str) -> None:
    local = find_local_checkpoint(model_name)
    if local is not None and local.exists():
        return

    out_dir = default_output_dir()
    try:
        path = download_and_save(model_name, out_dir)
        logger.info("Model cached at %s", path)
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:  # pragma: no cover - best effort
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

    # Prepare audio files to meet model expectations (mono 16kHz 16-bit WAV).
    try:
        with prepare_audio_files(paths) as prepared_paths:
            if not prepared_paths:
                raise SystemExit(2)

            model_name = model or DEFAULT_PARAKEET_MODEL
            _ensure_local_model(model_name)
            m = load_transcription_model(model_name)
            results = transcribe_files(
                m,
                prepared_paths,
                batch_size=batch_size,
            )

            # Build a list of original input paths that correspond to the
            # prepared list.
            originals_for_prepared = []
            for p in paths:
                if p and os.path.exists(p):
                    originals_for_prepared.append(p)

            print_transcription_results(originals_for_prepared, results)

            structured = []
            for i, r in enumerate(results):
                structured.append(_result_to_dict(originals_for_prepared[i], r))

            if not diarize:
                try:
                    _persist_results(structured)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("Unable to persist results: %s", exc)
                return structured

            diar_model_name = diarization_model or DEFAULT_SORTFORMER_MODEL
            _ensure_local_model(diar_model_name)
            diarizer = load_diarization_model(diar_model_name)
            diar_segments, _ = diarize_files(
                diarizer,
                prepared_paths,
                batch_size=diarization_batch_size or batch_size,
            )
            # Use original paths so results reference input files.
            print_diarization_results(originals_for_prepared, diar_segments)
            diar_structured = convert_diarization_results(
                originals_for_prepared, diar_segments
            )
            _merge_diarization(structured, diar_structured)
            try:
                _persist_results(structured)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Unable to persist results: %s", exc)
            return structured
    # pragma: no cover - bubble up friendly error
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.error("Audio preparation failed: %s", exc)
        raise SystemExit(3) from exc
