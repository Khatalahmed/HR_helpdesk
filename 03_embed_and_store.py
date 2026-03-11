"""
Step 3: Embed chunks and store in pgvector.

Run:
    python 03_embed_and_store.py
"""

import os
import pickle
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "hr_helpdesk")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
COLLECTION_NAME = "hr_helpdesk"
CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
CHUNKS_FILE = "chunks.pkl"
BATCH_SIZE = 100
SLEEP_BETWEEN = 1


def load_chunks(filepath: str) -> list[Document]:
    if not Path(filepath).exists():
        print(f"{filepath} not found. Run 02_chunk_documents.py first.")
        sys.exit(1)

    with open(filepath, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks


def store_by_batch(chunks: list[Document], embedding, batch_size: int, sleep_secs: int):
    total = len(chunks)
    batches = [chunks[i : i + batch_size] for i in range(0, total, batch_size)]
    print(f"\nStoring {total} chunks in {len(batches)} batches of {batch_size}")

    vectorstore = None
    for idx, batch in enumerate(batches, 1):
        print(f"Batch {idx}/{len(batches)} ({len(batch)} chunks)...", end=" ", flush=True)
        t0 = time.time()

        if idx == 1:
            vectorstore = PGVector.from_documents(
                documents=batch,
                embedding=embedding,
                collection_name=COLLECTION_NAME,
                connection=CONNECTION_STRING,
                pre_delete_collection=True,
            )
        else:
            vectorstore.add_documents(batch)

        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")

        if idx < len(batches):
            time.sleep(sleep_secs)

    print(f"\nAll {total} chunks stored in collection '{COLLECTION_NAME}'")
    return vectorstore


def main():
    print("=" * 60)
    print("STEP 3: Embed and Store")
    print("=" * 60)

    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set in .env")
        sys.exit(1)

    print(f"API key prefix: {GOOGLE_API_KEY[:8]}...")
    print(f"DB connection: {CONNECTION_STRING}")

    chunks = load_chunks(CHUNKS_FILE)

    print("\nInitializing embedding model: models/gemini-embedding-001")
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    store_by_batch(chunks, embedding, BATCH_SIZE, SLEEP_BETWEEN)
    print("\nStep 3 complete. Ready for retrieval.")


if __name__ == "__main__":
    main()
