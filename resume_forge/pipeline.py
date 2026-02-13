import json
import os
import re
from typing import Dict, Any, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from resume_forge.config import settings
from resume_forge.llm import get_llm
from resume_forge.vectorstore import get_retriever
from resume_forge.prompts import load_prompts


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _load_action_words() -> dict:
    """Loads the action words JSON file from disk."""
    action_words_path = settings.ACTION_WORDS_FILE
    if os.path.exists(action_words_path):
        with open(action_words_path, "r") as f:
            return json.load(f)
    return {}


def _select_relevant_action_words(action_words: dict, section_name: str) -> str:
    """
    Selects a relevant subset of action words based on the section type,
    rather than dumping the entire 300-line JSON into the prompt.
    """
    section_category_map = {
        "EXPERIENCE": ["lead", "mgmt", "tech"],
        "PROJECTS": ["tech", "crea", "rsch"],
        "SKILLS": [],  # Skills section doesn't use action words
    }

    categories = section_category_map.get(section_name, list(action_words.keys()))

    if not categories:
        return "N/A (Skills section does not require action words)"

    subset = {}
    for cat in categories:
        if cat in action_words:
            subset[cat] = action_words[cat]

    return json.dumps(subset, indent=2)


def build_rag_chain():
    """
    Builds the RAG chain:
    Retriever -> Format Docs -> Prompt -> LLM -> Output Parser
    """
    prompts = load_prompts()
    system_prompt = prompts.get("system_prompt", "")
    user_prompt_template = prompts.get("user_prompt", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt_template)
    ])

    retriever = get_retriever()
    llm = get_llm()
    action_words = _load_action_words()

    chain = (
        {
            "context": (lambda x: x["job_description"]) | retriever | format_docs,
            "job_description": lambda x: x["job_description"],
            "section_name": lambda x: x["section_name"],
            "action_words": lambda x: _select_relevant_action_words(
                action_words, x["section_name"]
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def tailor_resume_section(job_description: str, template_content: str) -> str:
    """
    Detects placeholders (%% SECTION %%) in the template and fills them sequentially.
    """
    placeholders = ["SKILLS", "EXPERIENCE", "PROJECTS"]
    final_output = template_content
    chain = build_rag_chain()

    for section in placeholders:
        # Match %% SECTION %% with any amount of internal whitespace
        pattern = rf"%%\s+{section}\s+%%"
        if re.search(pattern, final_output):
            print(f"Generating section: {section}...")
            try:
                response = chain.invoke({
                    "job_description": job_description,
                    "section_name": section
                })

                # Clean up markdown fences
                cleaned_response = re.sub(r'^```(latex)?\n', '', response, flags=re.MULTILINE | re.IGNORECASE)
                cleaned_response = re.sub(r'\n```$', '', cleaned_response, flags=re.MULTILINE)
                cleaned_response = cleaned_response.strip()

                # Sanitize LaTeX (convert **bold** to \textbf{bold}, etc.)
                cleaned_response = sanitize_latex(cleaned_response)

                final_output = re.sub(pattern, lambda m: cleaned_response, final_output)
            except Exception as e:
                print(f"Error generating section {section}: {e}")
                final_output = re.sub(pattern, f"% Error generating {section}: {e}", final_output)

    return final_output.strip()


def sanitize_latex(text: str) -> str:
    """
    Cleans up common LLM output artifacts that break LaTeX compilation.
    """
    # 1. Convert **text** to \textbf{text}
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)

    # 2. Fix misclosed bold: \textbf{text** -> \textbf{text}
    text = re.sub(r'\\textbf\{(.*?)\*\*', r'\\textbf{\1}', text)

    # 3. Convert _text_ (markdown italic) to \textit{text}
    #    Only match single underscores not preceded/followed by word chars (avoid __dunder__)
    text = re.sub(r'(?<!\w)_(.*?)_(?!\w)', r'\\textit{\1}', text)

    # 4. Escape lone % signs that follow a number (e.g. "30%" -> "30\%")
    text = re.sub(r'(?<!\\)(\d+)%', r'\1\\%', text)

    # 5. Escape unescaped & signs (not already \&)
    text = re.sub(r'(?<!\\)&', r'\\&', text)

    # 6. Escape unescaped $ signs (not already \$ and not part of LaTeX math)
    text = re.sub(r'(?<!\\)\$(?![^$]*\\)', r'\\$', text)

    # 7. Remove stray markdown headers (# Header) that the LLM sometimes outputs
    text = re.sub(r'^#{1,3}\s+.*$', '', text, flags=re.MULTILINE)

    # 8. Remove hallucinated backslashes before capitalized words that aren't LaTeX commands
    #    Matches \Word where Word is not a known LaTeX command
    known_commands = (
        r'textbf|textit|textsc|emph|underline|href|hfill|vspace|hspace|begin|end|'
        r'item|itemsep|section|subsection|header|lineunder|contact|employer|school|'
        r'area|bull|cdot|input|pdfgentounicode|documentclass|usepackage|pagestyle|'
        r'raggedright|newcommand|newenvironment|def|renewcommand|tabular|topsep'
    )
    text = re.sub(rf'\\(?!(?:{known_commands})\b)([A-Z][a-zA-Z]*)', r'\1', text)

    return text
