import typer
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from resume_forge.vectorstore import ingest_vault
from resume_forge.pipeline import build_rag_chain

from langchain.schema import HumanMessage, SystemMessage

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
        # If it's a very long string that happens to be an invalid path (too long filename), treat as text
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
        # Load chain locally to avoid overhead if just ingesting
        chain = build_rag_chain()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Thinking...", total=None)
            
            # We invoke the chain with the JD and the Template content
            # The prompt expects 'job_description' and 'section_template'
            # Here we are passing the FULL template as 'section_template' based on current implementation
            # Ideally we might want to split it by sections, but for now passing the whole thing 
            # and letting the LLM fill placeholders is the strategy.
            response = chain.invoke({
                "job_description": jd_text,
                "section_template": template_content
            })
            
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
