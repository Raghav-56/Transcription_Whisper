from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

from Transcription_parakeet.App.pipeline import (  # noqa: E402
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_SORTFORMER_MODEL,
    run_pipeline,
)
from Transcription_parakeet.config.logger_config import logger  # noqa: E402

app = FastAPI(title="Audio Transcription Pipeline", version="1.0.0")

VALID_MODES: set[str] = {"transcription", "diarization", "combined"}


class ProcessRequest(BaseModel):
    file_paths: list[str]
    mode: Literal["transcription", "diarization", "combined"] = "combined"
    model: str | None = DEFAULT_PARAKEET_MODEL
    batch_size: int = 1
    diarization_model: str | None = DEFAULT_SORTFORMER_MODEL
    diarization_batch_size: int | None = None


class ProcessResponse(BaseModel):
    success: bool
    files_processed: int
    input_files: list[str]
    result: Any
    processing_time: float
    mode: str


def _normalize_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        logger.warning("Invalid mode '%s', defaulting to 'combined'", mode)
        return "combined"
    return mode


def _shape_pipeline_output(
    mode: str,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if mode == "transcription":
        return [
            {
                "file": item.get("file"),
                "text": item.get("text"),
                "score": item.get("score"),
                "timestamps": item.get("timestamps"),
            }
            for item in entries
        ]
    if mode == "diarization":
        return [
            {
                "file": item.get("file"),
                "speakers": item.get("speakers", []),
            }
            for item in entries
        ]
    return entries


def _process_inputs(
    input_paths: list[str],
    mode: str,
    *,
    model: str | None,
    batch_size: int | None,
    diarization_model: str | None,
    diarization_batch_size: int | None,
) -> tuple[str, list[dict[str, Any]]]:
    effective_mode = _normalize_mode(mode)
    diarize = effective_mode != "transcription"
    effective_batch = batch_size or 1
    results = run_pipeline(
        input_paths,
        model=model or None,
        batch_size=effective_batch,
        diarize=diarize,
        diarization_model=diarization_model if diarize else None,
        diarization_batch_size=diarization_batch_size,
    )
    shaped = _shape_pipeline_output(effective_mode, results)
    return effective_mode, shaped


@app.post("/process_json", response_model=ProcessResponse)
async def process_audio_json(request: ProcessRequest):
    start_time = time.time()

    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No files provided.")

    try:
        mode, result = _process_inputs(
            input_paths=request.file_paths,
            mode=request.mode,
            model=request.model,
            batch_size=request.batch_size,
            diarization_model=request.diarization_model,
            diarization_batch_size=request.diarization_batch_size,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Error processing JSON request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc

    processing_time = time.time() - start_time

    return ProcessResponse(
        success=True,
        files_processed=len(request.file_paths),
        input_files=[os.path.basename(p) for p in request.file_paths],
        result=result,
        processing_time=processing_time,
        mode=mode,
    )


@app.get("/api")
async def root():
    return {
        "message": "Audio Transcription Pipeline (server mode)",
        "version": "1.0.0",
        "endpoints": ["/process", "/health"],
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    files: list[UploadFile] | None = File(None),
    file_paths: str | None = Form(None),
    mode: str = Form("combined"),
    model: str = Form(DEFAULT_PARAKEET_MODEL),
    batch_size: int = Form(1),
    diarization_model: str = Form(DEFAULT_SORTFORMER_MODEL),
    diarization_batch_size: int | None = Form(None),
):
    start_time = time.time()
    temp_files: list[str] = []
    temp_dir: str | None = None

    effective_mode = mode.strip() if mode else "combined"

    input_paths: list[str] = []

    if files:
        temp_dir = tempfile.mkdtemp()
        for file in files:
            if not file.filename:
                continue
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            input_paths.append(temp_path)
            temp_files.append(temp_path)
    elif file_paths:
        try:
            path_list = json.loads(file_paths)
            if isinstance(path_list, list):
                input_paths.extend(path_list)
            else:
                input_paths.append(str(path_list))
        except json.JSONDecodeError:
            input_paths.append(file_paths)

    if not input_paths:
        raise HTTPException(status_code=400, detail="No files provided.")

    cleaned_model = model.strip() if model else None
    cleaned_diar_model = diarization_model.strip() if diarization_model else None

    try:
        mode_value, result = _process_inputs(
            input_paths=input_paths,
            mode=effective_mode,
            model=cleaned_model,
            batch_size=batch_size,
            diarization_model=cleaned_diar_model,
            diarization_batch_size=diarization_batch_size,
        )
    except Exception as exc:  # pragma: no cover
        logger.error("Error processing audio: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:  # pragma: no cover
                logger.warning("Failed to cleanup temp file %s", temp_file)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    processing_time = time.time() - start_time

    return ProcessResponse(
        success=True,
        files_processed=len(input_paths),
        input_files=[os.path.basename(p) for p in input_paths],
        result=result,
        processing_time=processing_time,
        mode=mode_value,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "Transcription_parakeet.Interface.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
