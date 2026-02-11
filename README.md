# Resume-Forge

A privacy-first CLI tool that uses RAG (Retrieval-Augmented Generation) to tailor resumes based on a job description.

## Prerequisites
1. **LM Studio** running locally with a model loaded (e.g., Gemma-8B)
2. **Local Server** enabled in LM Studio at `http://localhost:1234/v1`

## Installation

```bash
pip install .
```

## Usage

### 1. Ingest Your Vault
Place markdown files describing your projects and roles in `vault/` (or specify a custom directory).

```bash
resume-forge ingest --vault-dir ./vault
```

### 2. Tailor Your Resume
Provide a job description (as a string or file) and your LaTeX template.

```bash
resume-forge tailor --jd "We need a Python expert..." --template templates/resume.tex --output tailored.tex
```

Or using a file for JD:
```bash
resume-forge tailor --jd path/to/jd.txt --template templates/resume.tex
```

## Configuration
Prompts can be edited in `templates/prompts.yaml`.
