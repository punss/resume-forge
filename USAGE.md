# Resume-Forge Usage Guide

## Prerequisites
1. **LM Studio**: Must be running locally.
   - Server: Start the server at `http://localhost:1234/v1`.
   - Model: Load a chat model (e.g., Gemma-8B, Mistral, Llama 3).
3. **Hardware Acceleration**: On macOS with Apple Silicon (M1/M2/M3), the application automatically uses **MPS (Metal Performance Shaders)** for embeddings, providing similar performance gains to MLX. Ensure LM Studio also has Metal/GPU acceleration enabled in its settings for the best experience.

---

## 1. Setup Your Data Vault
Create a directory (default: `./vault`) and add Markdown files representing your professional history.

- **Projects**: `vault/project_name.md`
- **Roles**: `vault/role_company.md`
- **Education/Skills**: `vault/skills.md`

**Templates**: Use the provided `vault/_template_project.md` and `vault/_template_role.md` as a starting point. Files starting with `_` are automatically excluded from ingestion.

**Tip**: Use clear headers and metrics in your markdown files. The RAG system uses these to find relevant content.

---

## 2. Ingest Data
Before generating resumes, you must index your vault. Run this whenever you add or edit files.

### Default Ingestion
Indexes all `.md` files in `./vault`.
```bash
python3 -m resume_forge.cli ingest
```

### Custom Vault Directory
If your files are elsewhere:
```bash
python3 -m resume_forge.cli ingest --vault-dir /path/to/my/notes
```

---

## 3. Tailor Your Resume
This command generates the LaTeX content based on a Job Description (JD).

### Basic Usage (String Input)
Paste the JD directly into the command.
```bash
python3 -m resume_forge.cli tailor \
  --jd "We need a Python expert with AWS experience..." \
  --template templates/resume.tex \
  --output tailored.tex
```

### File Input (Recommended for Long JDs)
Save the JD to a text file first.
```bash
python3 -m resume_forge.cli tailor \
  --jd job_description.txt \
  --template templates/resume.tex \
  --output tailored.tex
```

### Output to Screen
Omit the `--output` flag to print the LaTeX code to the terminal (useful for piping).
```bash
python3 -m resume_forge.cli tailor --jd "JD..." --template templates/resume.tex
```

---

## 4. Customizing Prompts
You can tweak how the AI writes your resume by editing `templates/prompts.yaml`.

- **Style**: Change "Action-oriented" to "Formal" or "Creative".
- **Logic**: Adjust the "Strategic Skill Selection" instructions.
- **Tone**: Modify the list of strong verbs.

No code changes are required—just edit the YAML file and run `tailor` again.
