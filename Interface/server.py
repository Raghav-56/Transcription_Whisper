import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Literal

sys.path.append(str(Path(__file__).parent.parent))

from Transcription_parakeet.config.logger_config import logger

app = FastAPI(title="Audio Transcription Pipeline", version="1.0.0")


class ProcessRequest(BaseModel):
    file_paths: list[str]
    mode: Literal["transcription", "diarization", "combined"] = "combined"
    model: str | None = None
    language: str | None = None
    device: str | None = None
    output_format: Literal["json", "txt", "srt", "vtt"] = "json"
    output: str | None = None


class ProcessResponse(BaseModel):
    success: bool
    files_processed: int
    input_files: list[str]
    result: Any
    processing_time: float
    mode: str


def _process_inputs(
    input_paths: list[str],
    mode: str,
    model: str | None,
    language: str | None,
    device: str,
    output_format: str,
    output: str | None,
):
    """Small helper that calls unified_api.process_audio and returns the result."""
    return unified_api.process_audio(
        inputs=input_paths,
        mode=mode,
        include_transcription=(mode != "diarization"),
        include_diarization=(mode != "transcription"),
        model=model,
        language=language,
        device=device,
        output_formats=[output_format],
        output_path=output,
    )


@app.post("/process_json", response_model=ProcessResponse)
async def process_audio_json(request: ProcessRequest):
    """Process audio files from a JSON request body using the Pydantic `ProcessRequest` model."""
    start_time = time.time()

    # Basic validation
    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No files provided.")

    effective_device = request.device or "cpu"
    if isinstance(effective_device, str) and effective_device.lower() == "auto":
        effective_device = "cpu"

    try:
        result = _process_inputs(
            input_paths=request.file_paths,
            mode=request.mode,
            model=request.model,
            language=request.language,
            device=effective_device,
            output_format=request.output_format,
            output=request.output,
        )
    except Exception as e:
        logger.error("Error processing JSON request: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

    processing_time = time.time() - start_time

    return ProcessResponse(
        success=True,
        files_processed=len(request.file_paths),
        input_files=[os.path.basename(p) for p in request.file_paths],
        result=result,
        processing_time=processing_time,
        mode=request.mode,
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
    file_paths: str | None = Form(None),  # JSON string of file paths
    mode: str = Form("combined"),
    model: str = Form("base"),
    language: str = Form("auto"),
    device: str = Form("auto"),
    output_format: str = Form("json"),
    output: str | None = Form(None),
):
    """Process audio files. This endpoint is a trimmed server alternative to the CLI.

    Notes:
    - Web UI serving is intentionally removed. This server only exposes API endpoints.
    - Files can be uploaded or a JSON string of file paths can be provided in `file_paths`.
    """
    start_time = time.time()
    temp_files: list[str] = []

    # Normalize device
    effective_device = device or "cpu"
    if isinstance(effective_device, str) and effective_device.lower() == "auto":
        effective_device = "cpu"
        logger.info("Device set to 'auto', defaulting to 'cpu'")

    valid_modes = ["transcription", "diarization", "combined"]
    if mode not in valid_modes:
        mode = "combined"
        logger.warning("Invalid mode, defaulting to 'combined'")

    valid_formats = ["json", "txt", "srt", "vtt"]
    if output_format not in valid_formats:
        output_format = "json"
        logger.warning("Invalid format, defaulting to 'json'")

    input_paths: list[str] = []

    # Handle uploaded files
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

    # Handle file paths from JSON or string
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

    # Process based on mode
    if mode == "transcription":
        include_transcription = True
        include_diarization = False
    elif mode == "diarization":
        include_transcription = False
        include_diarization = True
    else:  # combined
        include_transcription = True
        include_diarization = True

    try:
        # Call the unified API
        result = unified_api.process_audio(
            inputs=input_paths,
            mode=mode,
            include_transcription=include_transcription,
            include_diarization=include_diarization,
            model=model,
            language=language,
            device=effective_device,
            output_formats=[output_format],
            output_path=output,
        )

    except Exception as e:
        # Simplified error handling: log and expose a generic HTTP error
        logger.error("Error processing audio: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e

    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                logger.warning("Failed to cleanup temp file %s", temp_file)

    processing_time = time.time() - start_time

    return ProcessResponse(
        success=True,
        files_processed=len(input_paths),
        input_files=[os.path.basename(p) for p in input_paths],
        result=result,
        processing_time=processing_time,
        mode=mode,
    )


if __name__ == "__main__":
    # Run with: python -m interface.server
    import uvicorn

    uvicorn.run("Transcription_parakeet.Interface.server:app", host="0.0.0.0", port=8000, reload=True)
