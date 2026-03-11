"""
Step 4: Retrieval pipeline test (MMR + pgvector).

Run:
    python 04_retrieval_pipeline.py
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field

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


@dataclass
class RetrievalConfig:
    top_k: int = 6
    lambda_mult: float = 0.35
    score_threshold: float = 1.6
    min_results: int = 3


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


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when",
    "how", "your", "are", "any", "there", "does", "into", "about",
}

POLICY_HINTS = [
    (
        ("salary", "salary credit", "payroll", "payslip", "discrepancy"),
        ("Payroll_and_Salary_Processing_Policy", "Compensation_and_Benefits_Policy"),
    ),
    (
        ("mediclaim", "insurance", "maternity", "top-up"),
        ("Group_Health_Insurance_and_Mediclaim_Policy", "Compensation_and_Benefits_Policy"),
    ),
    (
        ("promotion", "appraisal", "rating"),
        ("Employee_Promotion_Policy", "Performance_Review_Policy"),
    ),
    (
        ("referral", "bonus"),
        ("Employee_Referral_Policy", "Compensation_and_Benefits_Policy"),
    ),
    (
        ("resign", "resignation", "full and final", "settlement", "fnf", "exit"),
        ("Resignation_and_Exit_Policy",),
    ),
    (
        ("sexual harassment", "posh", "icc"),
        ("POSH_Policy", "Grievance_Redressal_Policy"),
    ),
    (
        ("working hours", "attendance", "office hours", "flexi", "core hours"),
        ("Attendance_Policy", "Work_From_Home_Policy"),
    ),
    (
        ("training", "upskilling", "learning", "development"),
        ("Training_and_Learning_Development_Policy",),
    ),
    (
        ("leave", "earned leave", "annual leave"),
        ("Leave_Policy",),
    ),
]


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def lexical_overlap_ratio(question_tokens: set[str], text: str) -> float:
    if not question_tokens:
        return 0.0
    doc_tokens = tokenize(text)
    if not doc_tokens:
        return 0.0
    return len(question_tokens.intersection(doc_tokens)) / len(question_tokens)


def detect_policy_hints(question: str) -> list[str]:
    question_lower = question.lower()
    matches: list[str] = []
    for keywords, policy_names in POLICY_HINTS:
        if any(keyword in question_lower for keyword in keywords):
            matches.extend(policy_names)
    return list(dict.fromkeys(matches))


class HRRetrievalPipeline:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()

        print("Loading embedding model: models/gemini-embedding-001")
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        print("Connecting to pgvector")
        self.vectorstore = PGVector(
            embeddings=self.embedding,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
        print("Retrieval pipeline ready\n")

    @staticmethod
    def normalize_query(query: str) -> str:
        return query.strip()

    def retrieve(self, query: str) -> list[RetrievalResult]:
        normalized = self.normalize_query(query)
        raw = self.vectorstore.max_marginal_relevance_search_with_score(
            query=normalized,
            k=self.config.top_k,
            lambda_mult=self.config.lambda_mult,
            fetch_k=self.config.top_k * 8,
        )

        filtered = [(doc, score) for doc, score in raw if score <= self.config.score_threshold]
        selected = list(filtered)

        hint_policies = detect_policy_hints(normalized)
        hint_policy_set = set(hint_policies)
        for policy_name in hint_policies:
            selected.extend(
                self.vectorstore.max_marginal_relevance_search_with_score(
                    query=normalized,
                    k=2,
                    fetch_k=8,
                    lambda_mult=0.2,
                    filter={"filename": policy_name},
                )
            )

        if hint_policies:
            selected = [pair for pair in selected if pair[0].metadata.get("filename") in hint_policy_set]

        if len(selected) < self.config.min_results and hint_policies:
            for policy_name in hint_policies:
                selected.extend(
                    self.vectorstore.similarity_search_with_score(
                        normalized,
                        k=2,
                        filter={"filename": policy_name},
                    )
                )

        if len(selected) < self.config.min_results:
            selected.extend(
                self.vectorstore.similarity_search_with_score(
                    normalized,
                    k=max(self.config.top_k, self.config.min_results * 2),
                )
            )

        # De-duplicate by chunk_id first, then by content prefix.
        deduped: list[tuple] = []
        seen: set[str] = set()
        for doc, score in selected:
            key = doc.metadata.get("chunk_id") or doc.page_content[:180]
            if key in seen:
                continue
            seen.add(key)
            deduped.append((doc, score))

        if len(deduped) < self.config.min_results:
            deduped = raw[: self.config.min_results]

        query_tokens = tokenize(normalized)
        ranked = sorted(
            deduped,
            key=lambda item: (
                item[1] - 0.55 * lexical_overlap_ratio(query_tokens, item[0].page_content),
                item[1],
            ),
        )
        if hint_policies:
            hinted = [pair for pair in ranked if pair[0].metadata.get("filename") in hint_policy_set]
            others = [pair for pair in ranked if pair[0].metadata.get("filename") not in hint_policy_set]
            hint_quota = min(4, len(hinted))
            selected = hinted[:hint_quota] + others[: max(0, self.config.top_k - hint_quota)]
            if len(selected) < self.config.top_k:
                selected.extend(hinted[hint_quota : hint_quota + (self.config.top_k - len(selected))])
            selected = selected[: self.config.top_k]
        else:
            selected = ranked[: self.config.top_k]

        return [RetrievalResult(document=doc, score=score) for doc, score in selected]

    def print_results(self, query: str, results: list[RetrievalResult]):
        print("\n" + "-" * 90)
        print(f"Query   : {query}")
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


def main():
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set in .env")
        sys.exit(1)

    print("=" * 72)
    print("STEP 4: Retrieval Pipeline Test")
    print("=" * 72)

    pipeline = HRRetrievalPipeline(RetrievalConfig())

    for query in TEST_QUERIES:
        t0 = time.time()
        results = pipeline.retrieve(query)
        elapsed = time.time() - t0
        pipeline.print_results(query, results)
        print(f"\nRetrieved in {elapsed:.2f}s")

    print("\nAll test queries executed.")


if __name__ == "__main__":
    main()
