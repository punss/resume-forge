from setuptools import setup, find_packages

setup(
    name="resume-forge",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "typer[all]",
        "openai",
        "chromadb",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "sentence-transformers",
        "rich",
        "pyyaml",
        "pydantic-settings",
        "unstructured",
        "markdown"
    ],
    entry_points={
        "console_scripts": [
            "resume-forge=resume_forge.cli:app",
        ],
    },
)
