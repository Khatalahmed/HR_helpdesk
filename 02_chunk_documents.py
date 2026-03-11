# 02_chunk_documents.py

import os
import pickle
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DOCS_PATH = "./docs"
CHUNKS_FILE = "chunks.pkl"

# Apply a secondary splitter to oversized heading chunks.
MAX_CHUNK_CHARS = 1200
SECONDARY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_documents(docs_path: str):
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def chunk_markdown_universal(document: Document) -> list[Document]:
    """
    Two-stage chunking:
    1) Split by numbered markdown headings.
    2) If a heading chunk is too large, split with recursive character chunks.
    """
    text = document.page_content
    source = document.metadata.get("source", "")
    filename = os.path.basename(source).replace(".md", "")

    heading_pattern = re.compile(r"(?=^\*\*\d+[\.\d]*[\.\s])", re.MULTILINE)
    raw_chunks = heading_pattern.split(text)

    chunks: list[Document] = []
    for i, raw in enumerate(raw_chunks):
        content = raw.strip()
        if not content:
            continue

        first_line = content.split("\n")[0].strip()
        base_metadata = {
            "source": source,
            "filename": filename,
            "heading": first_line,
        }

        if len(content) > MAX_CHUNK_CHARS:
            sub_texts = SECONDARY_SPLITTER.split_text(content)
            for j, sub in enumerate(sub_texts):
                sub = sub.strip()
                if not sub:
                    continue
                chunks.append(
                    Document(
                        page_content=sub,
                        metadata={**base_metadata, "chunk_id": f"{filename}_chunk_{i}_sub_{j}"},
                    )
                )
        else:
            chunks.append(
                Document(
                    page_content=content,
                    metadata={**base_metadata, "chunk_id": f"{filename}_chunk_{i}"},
                )
            )

    return chunks


def chunk_all_documents(documents: list[Document]) -> list[Document]:
    all_chunks: list[Document] = []
    for doc in documents:
        all_chunks.extend(chunk_markdown_universal(doc))
    return all_chunks


if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents(DOCS_PATH)
    print(f"Documents loaded: {len(documents)}\n")

    print("Chunking documents (heading split + recursive split)...")
    all_chunks = chunk_all_documents(documents)
    long_chunks = sum(1 for c in all_chunks if len(c.page_content) > MAX_CHUNK_CHARS)
    avg_len = sum(len(c.page_content) for c in all_chunks) // max(1, len(all_chunks))

    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Average chunk length: {avg_len} chars")
    print(f"Chunks > {MAX_CHUNK_CHARS} chars: {long_chunks}\n")

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"Saved {len(all_chunks)} chunks to {CHUNKS_FILE}\n")

    print("Chunks per document")
    chunk_counts: dict[str, int] = {}
    for chunk in all_chunks:
        fname = chunk.metadata["filename"]
        chunk_counts[fname] = chunk_counts.get(fname, 0) + 1

    for fname, count in sorted(chunk_counts.items()):
        print(f"  {count:3d} chunks <- {fname}")

    if all_chunks:
        print("\nPreview of chunk 1")
        sample = all_chunks[0]
        print(f"  filename: {sample.metadata['filename']}")
        print(f"  heading : {sample.metadata['heading']}")
        print(f"  chunk_id: {sample.metadata['chunk_id']}")
        print(f"  content:\n{sample.page_content[:300]}")

    print("\nchunks.pkl saved. Ready for 03_embed_and_store.py")
