# 01_load_documents.py

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader


load_dotenv()

DOCS_PATH = "./docs"


def load_documents(docs_path: str):
    """
    Load all markdown files recursively from docs_path.
    """
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


if __name__ == "__main__":
    print("Loading documents...\n")
    documents = load_documents(DOCS_PATH)

    print(f"Total documents loaded: {len(documents)}")

    if documents:
        print("\nPreview of document 1")
        print(f"Source: {documents[0].metadata['source']}")
        print(f"Content (first 300 chars):\n{documents[0].page_content[:300]}")

    print("\nAll loaded files")
    for i, doc in enumerate(documents, 1):
        filename = os.path.basename(doc.metadata["source"])
        print(f"  {i:02d}. {filename}")
