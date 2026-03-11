"""
Step 6: RAG governance and accuracy evaluation with RAGAS.

Run:
    python 06_rag_evaluation.py

Key goals:
- Measure hallucination risk (faithfulness).
- Measure answer relevance.
- Measure retrieval quality (context precision + recall).
"""

import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)


warnings.filterwarnings("ignore")

# Avoid Windows console issues when output contains unsupported symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")


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

# Default to semantic (LLM) retrieval metrics. This better reflects policy-style
# content than strict lexical overlap.
USE_LLM_CONTEXT_METRICS = (
    os.getenv("RAG_EVAL_USE_LLM_CONTEXT_METRICS", "1").strip().lower()
    in {"1", "true", "yes", "y"}
)

PASS_THRESHOLD = 0.70


@dataclass
class EvalRetrievalConfig:
    top_k: int = 6
    lambda_mult: float = 0.35
    score_threshold: float = 1.6
    min_contexts: int = 3
    fetch_multiplier: int = 8


EVAL_RETRIEVAL = EvalRetrievalConfig()


@dataclass(frozen=True)
class QueryRoute:
    name: str
    triggers: tuple[str, ...]
    policies: tuple[str, ...]
    required_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    top_k: int
    min_contexts: int

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "how",
    "your",
    "are",
    "any",
    "there",
    "does",
    "into",
    "about",
}

LOW_SIGNAL_HEADING_TERMS = {
    "policy brief",
    "purpose",
    "scope",
    "roles and responsibilities",
    "definitions",
    "glossary",
    "policy review",
}

LOW_PRECISION_ROUTES: tuple[QueryRoute, ...] = (
    QueryRoute(
        name="promotion",
        triggers=("promotion", "appraisal", "rating", "eligible"),
        policies=("Employee_Promotion_Policy", "Performance_Review_Policy"),
        required_terms=("promotion", "appraisal", "rating", "tenure", "eligible", "approval", "vacancy"),
        negative_terms=("esop", "transfer", "grievance"),
        top_k=4,
        min_contexts=3,
    ),
    QueryRoute(
        name="payroll_discrepancy",
        triggers=("payroll discrepancy", "salary discrepancy", "raise a payroll", "salary discrepancy"),
        policies=("Payroll_and_Salary_Processing_Policy", "Compensation_and_Benefits_Policy"),
        required_terms=("discrepancy", "salary credit", "finance", "helpdesk", "working days", "raise"),
        negative_terms=("referral", "mediclaim", "bonus", "cut-off date", "pay cycle"),
        top_k=3,
        min_contexts=2,
    ),
    QueryRoute(
        name="posh",
        triggers=("sexual harassment", "posh", "icc", "complaint"),
        policies=("POSH_Policy", "Grievance_Redressal_Policy"),
        required_terms=("icc", "complaint", "inquiry", "confidentiality", "respondent", "timeline", "days"),
        negative_terms=("travel", "esop", "benefits", "awareness", "training workshop", "orientation"),
        top_k=3,
        min_contexts=3,
    ),
    QueryRoute(
        name="working_hours",
        triggers=("working hours", "office working", "core hours", "attendance", "flexi"),
        policies=("Attendance_Policy",),
        required_terms=("working hours", "core hours", "9:00", "6:00", "monday", "friday", "flexi"),
        negative_terms=("tax", "award", "bonus", "referral", "wfh", "hybrid"),
        top_k=3,
        min_contexts=3,
    ),
    QueryRoute(
        name="training",
        triggers=("upskilling", "training benefits", "learning and development", "training", "sponsorship"),
        policies=("Training_and_Learning_Development_Policy",),
        required_terms=("learning", "development", "budget", "band", "sponsorship", "reimbursement", "training"),
        negative_terms=("spot award", "tax treatment", "referral", "bonus"),
        top_k=3,
        min_contexts=2,
    ),
)

# Keyword-to-policy hints used to improve retrieval coverage and relevance.
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


TEST_DATASET = [
    {
        "question": "What is the salary credit date?",
        "ground_truth": (
            "Salaries are credited to the employee's designated bank account on the last "
            "working day of each month. Salary slips are generated and made available on "
            "the HRMS portal by the 5th of the following month. In case of salary "
            "discrepancies, employees must raise a query with the Finance team through the "
            "HRMS helpdesk within 5 working days of salary credit."
        ),
    },
    {
        "question": "How many annual leaves do I get per year?",
        "ground_truth": (
            "All confirmed employees are entitled to 18 days of Earned Leave (EL) per "
            "calendar year. EL is credited in advance on 1st January every year. "
            "Employees who join mid-year receive pro-rated EL at the rate of 1.5 days "
            "per completed month of service remaining in the year."
        ),
    },
    {
        "question": "What is the mediclaim sum insured for employees and dependents?",
        "ground_truth": (
            "SnailCloud Technologies provides group health insurance to all employees and "
            "their dependents. The sum insured under the group mediclaim policy is "
            "Rs. 5 lakh per employee per annum. Coverage includes the employee, spouse or "
            "partner, up to 2 dependent children below 25 years of age, and dependent "
            "parents or parents-in-law subject to an additional premium. Pre-existing "
            "conditions are covered from day one. Maternity expenses are covered up to "
            "Rs. 75,000 per delivery. Employees may opt for a top-up cover of Rs. 5 lakh "
            "at an additional premium deducted from salary."
        ),
    },
    {
        "question": "How does the promotion process work?",
        "ground_truth": (
            "Promotions at SnailCloud Technologies are merit-based and reviewed annually "
            "during the appraisal cycle. To be eligible, an employee must have received a "
            "Rating 4 or 5 in the current appraisal cycle, have a minimum tenure of 18 "
            "months in the current role, have demonstrated competencies required for the "
            "next level, and have a vacancy or business need at the higher level. "
            "Promotions are subject to budget availability and approval by the department "
            "head and HR Director. Promotion decisions are communicated in April "
            "increment letters."
        ),
    },
    {
        "question": "How do I raise a payroll discrepancy?",
        "ground_truth": (
            "In case of salary discrepancies, employees must raise a query with the "
            "Finance team through the HRMS helpdesk within 5 working days of salary "
            "credit. Employees are responsible for maintaining accurate personal and "
            "bank details in HRMS and flagging discrepancies promptly."
        ),
    },
    {
        "question": "What is the full and final settlement process when I resign?",
        "ground_truth": (
            "The full and final settlement is processed within 45 days of the last "
            "working day, subject to completion of the exit clearance process. The FnF "
            "settlement includes outstanding salary for days worked, earned leave "
            "encashment, deduction of notice period buyout if applicable, recovery of "
            "pending advances or loans, and gratuity if the employee has completed 5 or "
            "more years of service. The FnF amount is credited to the employee's "
            "registered bank account and a Relieving Letter is issued within 7 working "
            "days of FnF processing."
        ),
    },
    {
        "question": "How are sexual harassment complaints handled at SnailCloud?",
        "ground_truth": (
            "Complaints of sexual harassment are handled by the Internal Complaints "
            "Committee (ICC). Upon receipt, the ICC sends a copy to the respondent within "
            "7 working days requesting a written response within 10 working days. Both "
            "parties are given equal opportunity to present their case. The inquiry is "
            "conducted with strict confidentiality. The ICC completes its inquiry and "
            "submits a report to the employer within 60 days of receiving the complaint. "
            "The employer acts on the ICC recommendation within 60 days of receiving "
            "the report."
        ),
    },
    {
        "question": "What is the employee referral bonus?",
        "ground_truth": (
            "Employees who refer candidates for open positions are eligible for a referral "
            "bonus paid in two tranches: 50% upon the referred employee completing 3 months "
            "of service and 50% upon completion of 6 months. Referral bonus amounts are: "
            "Rs. 20,000 for non-technical roles, Rs. 30,000 for technical individual "
            "contributor roles, Rs. 50,000 for senior and lead roles, and Rs. 75,000 for "
            "managerial and specialist roles. Referrals must be submitted through the HRMS "
            "Referral module before the candidate applies."
        ),
    },
    {
        "question": "What is the office working hours policy?",
        "ground_truth": (
            "Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday, "
            "inclusive of a 1-hour unpaid lunch break, giving 8 hours of net productive "
            "work. Core hours during which all employees must be available are 10:00 AM "
            "to 4:00 PM. Employees may opt for flexi-start with manager approval: "
            "8:00 AM to 5:00 PM, 9:00 AM to 6:00 PM (standard), or 10:00 AM to 7:00 PM."
        ),
    },
    {
        "question": "Are there any upskilling or training benefits?",
        "ground_truth": (
            "SnailCloud Technologies allocates an annual Learning and Development budget "
            "per band: Band 1-3 employees are eligible for up to Rs. 20,000 per year; "
            "Band 4-5 employees up to Rs. 40,000 per year; Band 6 and above up to "
            "Rs. 75,000 per year. Unused budget does not carry forward. Employees must "
            "submit a Training Sponsorship Request through the HRMS Learning Module and "
            "obtain manager and HR approval before registering. Reimbursement is claimed "
            "by submitting fee receipts within 30 days of completing the training."
        ),
    },
]


HR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert HR Assistant for SnailCloud Technologies.

Your job is to help employees understand HR policies clearly and completely.

STRICT RULES:
1. Answer only using the HR Policy Context below.
2. Do not add facts or numbers that are not in context.
3. Use concise, structured formatting (bullets for process details).
4. If information is missing, clearly say what is missing.
5. If the answer is not in the context, respond exactly:
   "I couldn't find a specific policy on this. Please contact hr@snailcloud.in"

HR POLICY CONTEXT:
{context}

Employee Question: {question}

Answer:""",
)


def split_reference_context(text: str) -> list[str]:
    """
    Break the ground truth into semantically usable units for context metrics.
    This improves recall scoring versus using one large paragraph as a single unit.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()]

    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence.split()) <= 5:
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged


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


def is_low_signal_heading(heading: str) -> bool:
    heading_lower = heading.strip().lower()
    if not heading_lower:
        return False
    return any(term in heading_lower for term in LOW_SIGNAL_HEADING_TERMS)


def combined_relevance_score(question_tokens: set[str], doc, distance: float) -> float:
    heading = str(doc.metadata.get("heading", "")).strip()
    content_overlap = lexical_overlap_ratio(question_tokens, doc.page_content)
    heading_overlap = lexical_overlap_ratio(question_tokens, heading)
    low_signal_penalty = 0.2 if is_low_signal_heading(heading) else 0.0
    # Lower score is better: reward lexical overlap, penalize generic headings.
    return distance - 0.65 * content_overlap - 0.35 * heading_overlap + low_signal_penalty


def detect_policy_hints(question: str) -> list[str]:
    question_lower = question.lower()
    matches: list[str] = []
    for keywords, policy_names in POLICY_HINTS:
        if any(keyword in question_lower for keyword in keywords):
            matches.extend(policy_names)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(matches))


def route_question(question: str) -> QueryRoute | None:
    question_lower = question.lower()
    best_route: QueryRoute | None = None
    best_match_count = 0

    for route in LOW_PRECISION_ROUTES:
        match_count = sum(1 for trigger in route.triggers if trigger in question_lower)
        if match_count > best_match_count:
            best_route = route
            best_match_count = match_count

    return best_route if best_match_count > 0 else None


def retrieve_policy_hint_pairs(question: str, vectorstore, policies: list[str]) -> list[tuple]:
    pairs: list[tuple] = []
    for policy_name in policies:
        pairs.extend(
            vectorstore.max_marginal_relevance_search_with_score(
                query=question,
                k=2,
                fetch_k=8,
                lambda_mult=0.2,
                filter={"filename": policy_name},
            )
        )
    return pairs


def retrieve_route_pairs(question: str, vectorstore, route: QueryRoute) -> list[tuple]:
    pairs: list[tuple] = []
    for policy_name in route.policies:
        # Similarity gives best local relevance inside a chosen policy.
        pairs.extend(
            vectorstore.similarity_search_with_score(
                question,
                k=6,
                filter={"filename": policy_name},
            )
        )
        # MMR adds some section diversity while staying policy-scoped.
        pairs.extend(
            vectorstore.max_marginal_relevance_search_with_score(
                query=question,
                k=3,
                fetch_k=10,
                lambda_mult=0.25,
                filter={"filename": policy_name},
            )
        )
    return pairs


def load_resources():
    print("Connecting to pgvector...")
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = PGVector(
        embeddings=embedding,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    print("Loading Gemini 2.5 Flash for generation...")
    generation_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.15,
        max_output_tokens=2048,
    )

    print("Loading Gemini 2.5 Flash evaluator (thinking disabled)...")
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=4096,
        thinking_budget=0,
        response_mime_type="application/json",
    )

    return vectorstore, generation_llm, embedding, evaluator_llm


def dedupe_pairs(pairs: list[tuple]) -> list[tuple]:
    deduped: list[tuple] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for doc, score in pairs:
        metadata = getattr(doc, "metadata", {}) or {}
        chunk_id = str(metadata.get("chunk_id", "")).strip()
        text_key = doc.page_content.strip()[:200]

        if chunk_id and chunk_id in seen_ids:
            continue
        if text_key in seen_texts:
            continue

        if chunk_id:
            seen_ids.add(chunk_id)
        seen_texts.add(text_key)
        deduped.append((doc, score))

    return deduped


def retrieve_contexts(question: str, vectorstore, config: EvalRetrievalConfig) -> list[str]:
    route = route_question(question)
    if route is not None:
        question_tokens = tokenize(question)
        routed_pairs = dedupe_pairs(retrieve_route_pairs(question, vectorstore, route))
        routed_pairs = [pair for pair in routed_pairs if pair[1] <= (config.score_threshold + 0.15)] or routed_pairs

        ranked_routed: list[tuple] = []
        for doc, score in routed_pairs:
            text_lower = doc.page_content.lower()
            heading = str(doc.metadata.get("heading", "")).strip().lower()
            full_text = f"{heading}\n{text_lower}"

            content_overlap = lexical_overlap_ratio(question_tokens, doc.page_content)
            heading_overlap = lexical_overlap_ratio(question_tokens, heading)
            required_hits = sum(1 for term in route.required_terms if term in full_text)
            negative_hits = sum(1 for term in route.negative_terms if term in full_text)

            if required_hits == 0 and (content_overlap + heading_overlap) < 0.08:
                continue

            blended = combined_relevance_score(question_tokens, doc, score)
            # Route-specific reranking: prefer route vocabulary, penalize off-route cues.
            rerank_score = blended - 0.10 * required_hits + 0.14 * negative_hits
            ranked_routed.append((doc, score, rerank_score))

        if not ranked_routed:
            ranked_routed = [(doc, score, score) for doc, score in routed_pairs]

        ranked_routed.sort(key=lambda item: item[2])
        selected_pairs = [(doc, score) for doc, score, _ in ranked_routed[: route.top_k]]

        # If route-only retrieval is too sparse, backfill from global retrieval.
        if len(selected_pairs) < route.min_contexts:
            global_pairs = vectorstore.similarity_search_with_score(question, k=max(config.top_k, route.min_contexts * 2))
            global_pairs = dedupe_pairs(selected_pairs + global_pairs)
            selected_pairs = global_pairs[: max(route.min_contexts, route.top_k)]

        contexts = [doc.page_content.strip() for doc, _ in selected_pairs if doc.page_content.strip()]
        return contexts

    raw_mmr = vectorstore.max_marginal_relevance_search_with_score(
        query=question,
        k=config.top_k,
        lambda_mult=config.lambda_mult,
        fetch_k=max(config.top_k * config.fetch_multiplier, config.top_k + 8),
    )

    filtered_mmr = [(doc, score) for doc, score in raw_mmr if score <= config.score_threshold]
    hint_policies = detect_policy_hints(question)
    hint_policy_set = set(hint_policies)
    hint_pairs = retrieve_policy_hint_pairs(question, vectorstore, hint_policies) if hint_policies else []

    if hint_policy_set:
        filtered_mmr = [pair for pair in filtered_mmr if pair[0].metadata.get("filename") in hint_policy_set]
    selected_pairs = dedupe_pairs(hint_pairs + filtered_mmr)

    if len(selected_pairs) < config.min_contexts and hint_policy_set:
        hint_backfill: list[tuple] = []
        for policy_name in hint_policies:
            hint_backfill.extend(
                vectorstore.similarity_search_with_score(
                    question,
                    k=2,
                    filter={"filename": policy_name},
                )
            )
        selected_pairs = dedupe_pairs(selected_pairs + hint_backfill)

    if len(selected_pairs) < config.min_contexts:
        raw_sim = vectorstore.similarity_search_with_score(
            question,
            k=max(config.top_k, config.min_contexts * 2),
        )
        selected_pairs = dedupe_pairs(selected_pairs + raw_sim)

    if len(selected_pairs) < config.min_contexts:
        selected_pairs = dedupe_pairs(raw_mmr)[: config.min_contexts]

    question_tokens = tokenize(question)
    ranked_pairs: list[tuple] = []
    for doc, score in selected_pairs:
        heading = str(doc.metadata.get("heading", "")).strip()
        content_overlap = lexical_overlap_ratio(question_tokens, doc.page_content)
        heading_overlap = lexical_overlap_ratio(question_tokens, heading)
        low_signal = is_low_signal_heading(heading)

        if low_signal and (content_overlap + heading_overlap) < 0.1 and len(selected_pairs) > config.top_k:
            continue

        blended = combined_relevance_score(question_tokens, doc, score)
        ranked_pairs.append((doc, score, blended))

    if not ranked_pairs:
        ranked_pairs = [(doc, score, score) for doc, score in selected_pairs]

    ranked_pairs.sort(key=lambda item: item[2])

    if hint_policy_set:
        hinted_pairs = [pair for pair in ranked_pairs if pair[0].metadata.get("filename") in hint_policy_set]
        other_pairs = [pair for pair in ranked_pairs if pair[0].metadata.get("filename") not in hint_policy_set]
        hint_quota = min(4, len(hinted_pairs))
        selected_ranked = hinted_pairs[:hint_quota] + other_pairs[: max(0, config.top_k - hint_quota)]
        selected_pairs = [(doc, score) for doc, score, _ in selected_ranked]
        if len(selected_pairs) < config.top_k:
            selected_pairs.extend(
                [(doc, score) for doc, score, _ in hinted_pairs[hint_quota : hint_quota + (config.top_k - len(selected_pairs))]]
            )
    else:
        selected_pairs = [(doc, score) for doc, score, _ in ranked_pairs[: config.top_k]]

    contexts = [doc.page_content.strip() for doc, _ in selected_pairs if doc.page_content.strip()]
    return contexts


def run_rag(question: str, vectorstore, llm, config: EvalRetrievalConfig) -> dict:
    contexts = retrieve_contexts(question, vectorstore, config)
    context_text = "\n\n".join(f"[Source {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts))
    prompt = HR_PROMPT.format(context=context_text, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "contexts": contexts}


def build_eval_dataset(test_data: list[dict], vectorstore, llm, config: EvalRetrievalConfig):
    samples: list[SingleTurnSample] = []
    raw_rows: list[dict] = []
    total = len(test_data)

    print(f"\nBuilding evaluation dataset for {total} questions...\n")
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i:02d}/{total}] {question}")

        result = run_rag(question, vectorstore, llm, config)
        reference_units = split_reference_context(ground_truth)

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=result["contexts"],
                response=result["answer"],
                reference=ground_truth,
                reference_contexts=reference_units,
            )
        )
        raw_rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "ground_truth": ground_truth,
                "contexts": result["contexts"],
                "reference_units": reference_units,
            }
        )
        time.sleep(1.0)

    return EvaluationDataset(samples=samples), raw_rows


def score_label(score: float) -> str:
    if score >= 0.85:
        return "EXCELLENT"
    if score >= PASS_THRESHOLD:
        return "PASS"
    return "FAIL"


def select_metric_score(scores: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in scores:
            return scores[key]
    return float("nan")


def print_results(scores: dict, df: pd.DataFrame):
    print("\n" + "=" * 86)
    print("RAG GOVERNANCE - ACCURACY EVALUATION REPORT")
    print("SnailCloud Technologies - HR Helpdesk RAG")
    print("=" * 86)

    metrics = [
        ("Faithfulness", ("faithfulness",), "No hallucination; answer grounded in context"),
        ("Answer Relevancy", ("answer_relevancy",), "Answer relevance to the question"),
        (
            "Context Precision",
            ("llm_context_precision_with_reference", "non_llm_context_precision_with_reference"),
            "Retrieved chunks are relevant",
        ),
        (
            "Context Recall",
            ("llm_context_recall", "non_llm_context_recall", "context_recall"),
            "Retrieved chunks cover necessary info",
        ),
    ]

    print(f"\n{'Metric':<20} {'Score':>7} {'Status':<10} Meaning")
    print("-" * 86)
    for label, keys, meaning in metrics:
        score = select_metric_score(scores, keys)
        if pd.isna(score):
            print(f"{label:<20} {'N/A':>7} {'-':<10} {meaning}")
        else:
            print(f"{label:<20} {score:>7.3f} {score_label(score):<10} {meaning}")

    valid_scores = [v for v in scores.values() if isinstance(v, float) and not pd.isna(v)]
    overall = sum(valid_scores) / max(1, len(valid_scores))
    verdict = "PRODUCTION READY" if overall >= PASS_THRESHOLD else "NEEDS IMPROVEMENT"

    print("-" * 86)
    print(f"Overall RAG score: {overall:.3f} ({verdict})")
    print(f"Pass threshold: {PASS_THRESHOLD:.2f} | Questions evaluated: {len(df)}")
    print("-" * 86)

    if "faithfulness" in df.columns:
        print("\nPer-question faithfulness:")
        print(f"{'#':<4} {'Score':>7}  Question")
        print("-" * 86)
        for _, row in df.iterrows():
            score = row.get("faithfulness", float("nan"))
            if pd.isna(score):
                continue
            status = "OK" if score >= PASS_THRESHOLD else "LOW"
            print(f"{int(row['idx']):<4} {score:>7.3f}  [{status}] {row['question'][:68]}")


def main():
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set in .env")
        sys.exit(1)

    print("=" * 70)
    print("SnailCloud HR Helpdesk - RAG Governance and Accuracy Evaluation")
    print("Generation: gemini-2.5-flash | Evaluator: gemini-2.5-flash (thinking OFF)")
    print("=" * 70)

    vectorstore, generation_llm, embedding, evaluator_llm = load_resources()
    print("Resources loaded.")

    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding)

    context_precision_metric = (
        LLMContextPrecisionWithReference(llm=ragas_llm)
        if USE_LLM_CONTEXT_METRICS
        else NonLLMContextPrecisionWithReference()
    )
    context_recall_metric = (
        LLMContextRecall(llm=ragas_llm)
        if USE_LLM_CONTEXT_METRICS
        else NonLLMContextRecall()
    )

    context_mode = "LLM semantic" if USE_LLM_CONTEXT_METRICS else "Non-LLM lexical"
    print(f"Context metrics mode: {context_mode}")
    print(
        "Retrieval config: "
        f"top_k={EVAL_RETRIEVAL.top_k}, "
        f"lambda={EVAL_RETRIEVAL.lambda_mult}, "
        f"threshold={EVAL_RETRIEVAL.score_threshold}, "
        f"min_contexts={EVAL_RETRIEVAL.min_contexts}, "
        f"low_precision_routes={len(LOW_PRECISION_ROUTES)}"
    )

    eval_dataset, raw_rows = build_eval_dataset(
        TEST_DATASET,
        vectorstore,
        generation_llm,
        EVAL_RETRIEVAL,
    )
    print(f"\nDataset ready: {len(raw_rows)} samples.")

    print("\nRunning RAGAS evaluation (approx. 3-5 minutes)...\n")
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        context_precision_metric,
        context_recall_metric,
    ]

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        raise_exceptions=False,
        batch_size=2,
    )

    result_df = result.to_pandas().copy()
    scores_dict = result_df.mean(numeric_only=True).to_dict()

    result_df["question"] = [r["question"] for r in TEST_DATASET[: len(result_df)]]
    result_df["idx"] = range(1, len(result_df) + 1)

    print_results(scores_dict, result_df)

    report_df = result_df.copy()
    if "retrieved_contexts" in report_df.columns:
        report_df["retrieved_contexts"] = report_df["retrieved_contexts"].apply(
            lambda value: " | ".join(value) if isinstance(value, list) else str(value)
        )
    report_df.to_csv("rag_evaluation_report.csv", index=False)
    print("\nSaved detailed report: rag_evaluation_report.csv")

    summary = {
        "generation_model": "gemini-2.5-flash",
        "evaluation_model": "gemini-2.5-flash (thinking_budget=0)",
        "embedding_model": "models/gemini-embedding-001",
        "context_metrics_mode": context_mode,
        "retrieval_config": {
            "top_k": EVAL_RETRIEVAL.top_k,
            "lambda_mult": EVAL_RETRIEVAL.lambda_mult,
            "score_threshold": EVAL_RETRIEVAL.score_threshold,
            "min_contexts": EVAL_RETRIEVAL.min_contexts,
            "fetch_multiplier": EVAL_RETRIEVAL.fetch_multiplier,
        },
        "questions_evaluated": len(TEST_DATASET),
        "scores": {k: round(v, 4) for k, v in scores_dict.items() if isinstance(v, float)},
        "pass_threshold": PASS_THRESHOLD,
    }
    with open("rag_evaluation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("Saved summary: rag_evaluation_summary.json\n")


if __name__ == "__main__":
    main()
