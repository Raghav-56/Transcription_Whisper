from __future__ import annotations

import json

import typer

from Transcription_parakeet.App.pipeline import (
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_SORTFORMER_MODEL,
    run_pipeline,
)

app = typer.Typer()


@app.command()
def transcribe(
    files: list[str] = typer.Argument(..., help="Paths to audio files to process."),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Transcription model name or path. "
            f"Defaults to {DEFAULT_PARAKEET_MODEL}."
        ),
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Batch size for transcription.",
    ),
    diarize: bool = typer.Option(
        True,
        "--diarize/--no-diarize",
        help="Enable speaker diarization output.",
    ),
    diarization_model: str | None = typer.Option(
        None,
        "--diarization-model",
        help=(
            "Override diarization model. " f"Defaults to {DEFAULT_SORTFORMER_MODEL}."
        ),
    ),
    diarization_batch_size: int | None = typer.Option(
        None,
        "--diarization-batch-size",
        help=("Batch size for diarization " "(defaults to transcription batch size)."),
    ),
) -> None:
    """Run the transcription pipeline from the command line."""
    try:
        results = run_pipeline(
            files,
            model=model,
            batch_size=batch_size,
            diarize=diarize,
            diarization_model=diarization_model,
            diarization_batch_size=diarization_batch_size,
        )
    except SystemExit as exc:
        if getattr(exc, "code", None) == 2:
            typer.echo("No valid input files provided.")
            raise typer.Exit(code=2)
        raise

    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
