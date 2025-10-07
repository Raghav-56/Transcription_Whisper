"""Download and save a NeMo ASR pretrained model to disk (.nemo).

This script loads the model via ``ASRModel.from_pretrained`` and calls
``model.save_to(path)`` to persist a ``.nemo`` package. It supports
``--force`` to overwrite an existing file.
"""

from __future__ import annotations

from pathlib import Path
import typer

from nemo.collections.asr.models import ASRModel


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_and_save(model_name: str, out_dir: Path, force: bool = False) -> Path:
    """Download the pretrained model and save it as a .nemo file.

    Returns the path to the saved .nemo file.
    """
    ensure_dir(out_dir)

    # Create a safe filename from the model name
    safe_name = model_name.replace("/", "__").replace(" ", "_")
    out_path = out_dir / f"{safe_name}.nemo"

    if out_path.exists() and not force:
        print(
            "Model file already exists at {}.\n"
            "Use --force to overwrite.".format(out_path)
        )
        return out_path

    print(
        "Loading pretrained model '{}' (this downloads if needed)...".format(model_name)
    )
    model = ASRModel.from_pretrained(model_name)

    print("Saving model to {} ...".format(out_path))
    # NeMo provides `save_to` to persist a model as a .nemo package
    model.save_to(str(out_path))
    print("Saved.")
    return out_path


app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = "nvidia/parakeet-tdt-0.6b-v2",
    out_dir: str = "./models",
    force: bool = False,
) -> int:
    """Download and save a NeMo model as a .nemo file.

    Example:
        python -m Src.save_model --model <name> --out-dir ./models --force
    """
    out_path = Path(out_dir)
    path = download_and_save(model, out_path, force=force)
    print("Model available at: {}".format(path))
    print(
        "To load this saved model later, use:\n"
        "  ASRModel.restore_from(restore_path='path/to/file.nemo')"
    )
    return 0


if __name__ == "__main__":
    app()
