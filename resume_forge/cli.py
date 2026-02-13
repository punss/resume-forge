import warnings

# Suppress non-critical warnings to keep CLI output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from resume_forge.llm import check_llm_status
from resume_forge.pipeline import tailor_resume_section
from resume_forge.vectorstore import ingest_vault

app = typer.Typer(help="Resume-Forge: Privacy-first AI Resume Tailor")
console = Console()

@app.command()
def ingest(
    vault_dir: Path = typer.Option(
        "./vault", "--vault-dir", "-v", help="Directory containing markdown files to ingest",
        exists=True, file_okay=False, dir_okay=True, readable=True
    )
):
    """
    Ingest markdown files from the vault directory into the vector store.
    """
    console.print(f"[bold blue]ingesting vault from:[/bold blue] {vault_dir}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Processing files...", total=None)
            count = ingest_vault(str(vault_dir))

        console.print(f"[bold green]Successfully ingested {count} chunks![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during ingestion:[/bold red] {e}")
        raise typer.Exit(code=1)

@app.command()
def tailor(
    jd: str = typer.Option(..., "--jd", help="Job Description string or path to a text file"),
    template: Path = typer.Option(..., "--template", "-t", help="Path to LaTeX template file", exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path for the tailored resume")
):
    """
    Tailor a resume based on the provided Job Description and LaTeX template.
    """
    # 0. Check LLM Status
    if not check_llm_status():
        console.print("[bold red]Error:[/bold red] LLM endpoint is not reachable or no models are loaded.")
        console.print("[dim]Please ensure LM Studio is running and a model is loaded in the 'Local Server' tab.[/dim]")
        raise typer.Exit(code=1)

    # 1. Resolve Job Description (File vs String)
    jd_text = ""
    try:
        jd_path = Path(jd)
        if jd_path.exists() and jd_path.is_file():
            console.print(f"[dim]Reading JD from file: {jd_path}[/dim]")
            jd_text = jd_path.read_text(encoding="utf-8")
        else:
            jd_text = jd
    except OSError:
        # If it's a very long string that happens to be an invalid path, treat as text
        jd_text = jd

    if not jd_text.strip():
        console.print("[bold red]Error:[/bold red] Job Description is empty.")
        raise typer.Exit(code=1)

    # 2. Read Template
    try:
        template_content = template.read_text(encoding="utf-8")
    except Exception as e:
        console.print(f"[bold red]Error reading template:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 3. Running RAG Pipeline
    console.print("[bold blue]Generating tailored content...[/bold blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Tailoring resume sections...", total=None)
            response = tailor_resume_section(jd_text, template_content)

    except Exception as e:
        console.print(f"[bold red]Error generating resume:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 4. Output
    if output:
        try:
            output.write_text(response, encoding="utf-8")
            console.print(f"[bold green]Tailored resume saved to:[/bold green] {output}")
        except Exception as e:
            console.print(f"[bold red]Error saving output:[/bold red] {e}")
            raise typer.Exit(code=1)
    else:
        # Print to stdout if no output file specified
        print(response)

if __name__ == "__main__":
    app()
