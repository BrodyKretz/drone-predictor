"""Augur CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from augur import __version__
from augur.report import render

app = typer.Typer(add_completion=False, help="Multimodal drone property inference.")


@app.command()
def version():
    """Print the Augur version."""
    typer.echo(f"augur {__version__}")


@app.command()
def predict(
    audio: Path = typer.Option(None, exists=True, help="WAV flight recording"),
    verbal: Path = typer.Option(None, exists=True, help="JSON verbal spec"),
    image: list[Path] = typer.Option(None, help="Still image(s) [not yet wired]"),
    video: Path = typer.Option(None, help="Flight video [not yet wired]"),
    samples: int = typer.Option(8000, help="Monte Carlo sample count"),
    seed: int = typer.Option(0, help="RNG seed"),
):
    """Predict drone properties from any subset of inputs."""
    from augur.pipeline import predict as run_predict

    if not any([audio, verbal, image, video]):
        typer.echo("Provide at least one of --audio / --verbal / --image / --video", err=True)
        raise typer.Exit(code=1)

    report = run_predict(
        audio=audio, verbal=verbal,
        image=[str(p) for p in image] if image else None,
        video=video, n=samples, seed=seed,
    )
    typer.echo(render(report))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8000, help="Port"),
):
    """Run the HTTP API (requires the 'serve' extra)."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("serve needs the 'serve' extra: pip install -e '.[serve]'", err=True)
        raise typer.Exit(code=1) from None

    uvicorn.run("augur.api:app", host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
