"""
Step 4: Retrieval pipeline test (uses shared API retrieval logic).

Run:
    python 04_retrieval_pipeline.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.service import RAGService


@dataclass
class RetrievalResult:
    document: Document
    score: float
    filename: str = field(init=False)
    heading: str = field(init=False)
    chunk_id: str = field(init=False)

    def __post_init__(self):
        metadata = self.document.metadata
        self.filename = metadata.get("filename", "Unknown")
        self.heading = metadata.get("heading", "")
        self.chunk_id = metadata.get("chunk_id", "")


class HRRetrievalPipeline:
    def __init__(self):
        self.service = RAGService(settings=get_settings())
        print("Retrieval pipeline ready (shared RAGService logic).\n")

    @staticmethod
    def normalize_query(query: str) -> str:
        return query.strip()

    def retrieve(self, query: str, top_k: int = 6) -> tuple[list[RetrievalResult], str | None]:
        normalized = self.normalize_query(query)
        pairs, route_name = self.service.retrieve_pairs(normalized, top_k_override=top_k)
        return [RetrievalResult(document=doc, score=score) for doc, score in pairs], route_name

    @staticmethod
    def print_results(query: str, route_name: str | None, results: list[RetrievalResult]):
        print("\n" + "-" * 90)
        print(f"Query   : {query}")
        print(f"Route   : {route_name or 'global'}")
        print(f"Results : {len(results)} chunks")
        print("-" * 90)

        for i, result in enumerate(results, 1):
            snippet = result.document.page_content[:220].replace("\n", " ")
            print(f"\n[{i}] {result.filename}")
            print(f"    heading: {result.heading[:80]}")
            print(f"    score  : {result.score:.4f} | chunk: {result.chunk_id}")
            print(f"    text   : {snippet}...")


TEST_QUERIES = [
    "What is the salary credit date?",
    "How do I raise a payroll discrepancy?",
    "What is the mediclaim sum insured for dependents?",
    "How does the promotion process work?",
    "What are the DEI commitments of the company?",
]


def main() -> int:
    try:
        pipeline = HRRetrievalPipeline()
    except Exception as exc:
        print(f"Failed to initialize retrieval pipeline: {exc}")
        return 1

    print("=" * 72)
    print("STEP 4: Retrieval Pipeline Test")
    print("=" * 72)

    for query in TEST_QUERIES:
        t0 = time.time()
        results, route_name = pipeline.retrieve(query)
        elapsed = time.time() - t0
        pipeline.print_results(query, route_name, results)
        print(f"\nRetrieved in {elapsed:.2f}s")

    print("\nAll test queries executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
