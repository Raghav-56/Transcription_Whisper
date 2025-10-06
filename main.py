import sys
import subprocess
from pathlib import Path
from typing import Optional, Annotated
import typer


app = typer.Typer(
    name="transcription_agent",
    help="Transcription Agent Main Entry Point",
    no_args_is_help=True,
)


def run_script(script_name: str, extra_args: list[str]):
    # Execute the interface script as a module so package imports work.
    # When run as a subprocess, ensure the working directory is the repo root
    # (parent of this package) so `Transcription_parakeet` is importable.
    repo_root = Path(__file__).parent.parent
    module_name = f"Transcription_parakeet.Interface.{Path(script_name).stem}"
    cmd = [sys.executable, "-m", module_name] + extra_args
    subprocess.run(cmd, cwd=repo_root)


@app.command()
def cli(
    ctx: typer.Context,
    extra_args: Annotated[Optional[list[str]], typer.Argument()] = None,
):
    """Run CLI interface."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    run_script("cli.py", args)


@app.command()
def client(
    ctx: typer.Context,
    extra_args: Annotated[Optional[list[str]], typer.Argument()] = None,
):
    """Run API client."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    run_script("client.py", args)


@app.command()
def server(
    ctx: typer.Context,
    extra_args: Annotated[Optional[list[str]], typer.Argument()] = None,
):
    """Run API server."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    run_script("server.py", args)


@app.command()
def streamlit(
    ctx: typer.Context,
    extra_args: Annotated[Optional[list[str]], typer.Argument()] = None,
):
    """Run Streamlit app."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    streamlit_path = Path(__file__).parent / "interface" / "streamlit_app.py"
    cmd = ["streamlit", "run", str(streamlit_path)] + args
    subprocess.run(cmd)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Interactive mode when no command is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo("Select a mode to run:")
        typer.echo("1. CLI interface")
        typer.echo("2. API client")
        typer.echo("3. API server")
        typer.echo("4. Streamlit app")

        choice = typer.prompt("Enter choice [1-4]").strip()

        if choice == "1":
            ctx.invoke(cli)
        elif choice == "2":
            ctx.invoke(client)
        elif choice == "3":
            ctx.invoke(server)
        elif choice == "4":
            ctx.invoke(streamlit)
        else:
            typer.echo("Invalid choice. Exiting.")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
