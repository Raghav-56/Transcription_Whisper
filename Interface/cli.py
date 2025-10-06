import typer
from typing import Optional

from Transcription_parakeet.App.pipeline import run_pipeline

app = typer.Typer()


@app.command()
def transcribe(
	files: list[str],
	model: Optional[str] = typer.Option(None, help="Pretrained model name or path"),
	batch_size: int = typer.Option(1, help="Batch size for transcription"),
):

	try:
		run_pipeline(files, model=model, batch_size=batch_size)
	except SystemExit as e:
		if getattr(e, 'code', None) == 2:
			typer.echo("No valid input files provided.")
			raise typer.Exit(code=2)
		raise


def main():
	app()


if __name__ == "__main__":
	main()
