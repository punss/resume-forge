from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from resume_forge.llm import get_llm
from resume_forge.vectorstore import get_retriever
from resume_forge.prompts import load_prompts

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

    # The chain needs to:
    # 1. Take {"job_description": "...", "section_template": "..."} as input
    # 2. Retrieve relevant docs based on job_description (or maybe a combination?)
    #    Let's use the JD to retrieve relevant context.
    
    chain = (
        {
            "context": (lambda x: x["job_description"]) | retriever | format_docs,
            "job_description": lambda x: x["job_description"],
            "section_template": lambda x: x["section_template"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def tailor_resume_section(job_description: str, section_template: str) -> str:
    """
    Runs the RAG chain to tailor a specific resume section.
    """
    chain = build_rag_chain()
    response = chain.invoke({
        "job_description": job_description,
        "section_template": section_template
    })

    # Clean up markdown fences if present
    import re
    cleaned_response = re.sub(r'^```(latex)?\n', '', response, flags=re.MULTILINE)
    cleaned_response = re.sub(r'\n```$', '', cleaned_response, flags=re.MULTILINE)
    
    return cleaned_response.strip()
