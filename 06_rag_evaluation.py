"""
Step 6: RAG governance and accuracy evaluation with RAGAS.

Run:
    python 06_rag_evaluation.py
"""

from __future__ import annotations

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

from app.config import get_settings
from app.rag.service import RAGService


warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")


load_dotenv()

USE_LLM_CONTEXT_METRICS = (
    os.getenv("RAG_EVAL_USE_LLM_CONTEXT_METRICS", "1").strip().lower() in {"1", "true", "yes", "y"}
)
PASS_THRESHOLD = 0.70


@dataclass
class EvalRetrievalConfig:
    top_k: int = 6


EVAL_RETRIEVAL = EvalRetrievalConfig()


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


def load_resources():
    settings = get_settings()
    rag_service = RAGService(settings=settings)

    embedding = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
    generation_llm = ChatGoogleGenerativeAI(
        model=settings.generation_model,
        google_api_key=settings.google_api_key,
        temperature=0.15,
        max_output_tokens=2048,
    )
    evaluator_llm = ChatGoogleGenerativeAI(
        model=settings.generation_model,
        google_api_key=settings.google_api_key,
        temperature=0.0,
        max_output_tokens=4096,
        thinking_budget=0,
        response_mime_type="application/json",
    )
    return rag_service, generation_llm, embedding, evaluator_llm, settings


def run_rag(question: str, rag_service: RAGService, llm, config: EvalRetrievalConfig) -> dict:
    contexts, route_name = rag_service.retrieve_contexts(question, top_k_override=config.top_k)
    context_text = "\n\n".join(f"[Source {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts))
    prompt = HR_PROMPT.format(context=context_text, question=question)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer, "contexts": contexts, "route": route_name}


def build_eval_dataset(test_data: list[dict], rag_service: RAGService, llm, config: EvalRetrievalConfig):
    samples: list[SingleTurnSample] = []
    raw_rows: list[dict] = []
    total = len(test_data)

    print(f"\nBuilding evaluation dataset for {total} questions...\n")
    for i, item in enumerate(test_data, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i:02d}/{total}] {question}")

        result = run_rag(question, rag_service, llm, config)
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
                "route": result["route"],
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


def main():
    try:
        rag_service, generation_llm, embedding, evaluator_llm, settings = load_resources()
    except Exception as exc:
        print(f"Failed to initialize evaluation resources: {exc}")
        return 1

    print("=" * 70)
    print("SnailCloud HR Helpdesk - RAG Governance and Accuracy Evaluation")
    print(f"Generation: {settings.generation_model} | Evaluator: {settings.generation_model} (thinking OFF)")
    print("=" * 70)

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
    print(f"Retrieval config: top_k={EVAL_RETRIEVAL.top_k}")

    eval_dataset, raw_rows = build_eval_dataset(TEST_DATASET, rag_service, generation_llm, EVAL_RETRIEVAL)
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
        "generation_model": settings.generation_model,
        "evaluation_model": f"{settings.generation_model} (thinking_budget=0)",
        "embedding_model": settings.embedding_model,
        "context_metrics_mode": context_mode,
        "retrieval_config": {
            "top_k": EVAL_RETRIEVAL.top_k,
        },
        "questions_evaluated": len(TEST_DATASET),
        "scores": {k: round(v, 4) for k, v in scores_dict.items() if isinstance(v, float)},
        "pass_threshold": PASS_THRESHOLD,
    }
    with open("rag_evaluation_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print("Saved summary: rag_evaluation_summary.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
