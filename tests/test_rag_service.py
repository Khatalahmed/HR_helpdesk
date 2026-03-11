from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.config import Settings
from app.rag.service import RAGService


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict[str, Any]


class FakeVectorStore:
    def __init__(self, pairs: list[tuple[FakeDoc, float]]):
        self.pairs = pairs
        self.calls: list[tuple] = []

    def _apply_filter(self, pairs: list[tuple[FakeDoc, float]], metadata_filter: dict[str, Any] | None):
        if not metadata_filter:
            return pairs
        filename = metadata_filter.get("filename")
        if not filename:
            return pairs
        return [pair for pair in pairs if pair[0].metadata.get("filename") == filename]

    def similarity_search_with_score(self, query: str, k: int, filter: dict[str, Any] | None = None):
        self.calls.append(("similarity", query, k, filter))
        return self._apply_filter(self.pairs, filter)[:k]

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: dict[str, Any] | None = None,
    ):
        self.calls.append(("mmr", query, k, fetch_k, lambda_mult, filter))
        return self._apply_filter(self.pairs, filter)[:k]


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, answer: str = "Mock answer"):
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any):
        prompt = str(input)
        self.prompts.append(prompt)
        return FakeLLMResponse(self.answer)


@pytest.fixture()
def base_service():
    return RAGService(settings=Settings(google_api_key="test-key"))


def _wire_fakes(service: RAGService, pairs: list[tuple[FakeDoc, float]], llm_answer: str = "Mock answer"):
    vectorstore = FakeVectorStore(pairs)
    llm = FakeLLM(answer=llm_answer)
    service._embedding = object()
    service._vectorstore = vectorstore
    service._llm = llm
    return vectorstore, llm


def test_dedupe_pairs_removes_chunk_and_text_duplicates(base_service: RAGService):
    doc1 = FakeDoc("Same chunk content", {"chunk_id": "dup-1", "filename": "Leave_Policy"})
    doc2 = FakeDoc("Different text but same chunk id", {"chunk_id": "dup-1", "filename": "Leave_Policy"})
    doc3 = FakeDoc("Same chunk content", {"chunk_id": "dup-2", "filename": "Leave_Policy"})
    doc4 = FakeDoc("Unique content", {"chunk_id": "unique-1", "filename": "Leave_Policy"})

    deduped = base_service._dedupe_pairs([(doc1, 0.2), (doc2, 0.3), (doc3, 0.4), (doc4, 0.5)])

    assert len(deduped) == 2
    assert deduped[0][0].metadata["chunk_id"] == "dup-1"
    assert deduped[1][0].metadata["chunk_id"] == "unique-1"


def test_retrieve_pairs_route_keeps_only_routed_policies(base_service: RAGService):
    pairs = [
        (
            FakeDoc(
                "Promotion decisions depend on appraisal rating, tenure, vacancy and approval.",
                {"filename": "Employee_Promotion_Policy", "heading": "Promotion Eligibility", "chunk_id": "p1"},
            ),
            0.55,
        ),
        (
            FakeDoc(
                "Performance review ratings are used during promotion cycle eligibility checks.",
                {"filename": "Performance_Review_Policy", "heading": "Rating System", "chunk_id": "p2"},
            ),
            0.6,
        ),
        (
            FakeDoc(
                "Managers validate promotion readiness with business need and vacancy.",
                {"filename": "Employee_Promotion_Policy", "heading": "Approval Workflow", "chunk_id": "p3"},
            ),
            0.68,
        ),
        (
            FakeDoc(
                "ESOP vesting schedule depends on grant date and cliff period.",
                {"filename": "ESOP_Policy", "heading": "Vesting", "chunk_id": "e1"},
            ),
            0.12,
        ),
    ]
    _wire_fakes(base_service, pairs)

    selected_pairs, route_name = base_service._retrieve_pairs("How does the promotion process work?")

    assert route_name == "promotion"
    assert len(selected_pairs) >= 3
    assert {
        doc.metadata["filename"] for doc, _ in selected_pairs
    } <= {"Employee_Promotion_Policy", "Performance_Review_Policy"}


def test_retrieve_pairs_policy_hint_bias_for_salary_question(base_service: RAGService):
    pairs = [
        (
            FakeDoc(
                "Salary is credited on the last working day of every month.",
                {
                    "filename": "Payroll_and_Salary_Processing_Policy",
                    "heading": "Salary Credit Timeline",
                    "chunk_id": "sal-1",
                },
            ),
            0.45,
        ),
        (
            FakeDoc(
                "Payroll discrepancies should be raised within 3 working days via helpdesk.",
                {"filename": "Payroll_and_Salary_Processing_Policy", "heading": "Discrepancy", "chunk_id": "sal-2"},
            ),
            0.51,
        ),
        (
            FakeDoc(
                "Compensation components include fixed pay and variable pay details.",
                {"filename": "Compensation_and_Benefits_Policy", "heading": "Compensation", "chunk_id": "sal-3"},
            ),
            0.57,
        ),
        (
            FakeDoc(
                "Annual leave can be carried forward as per policy limits.",
                {"filename": "Leave_Policy", "heading": "Carry Forward", "chunk_id": "lv-1"},
            ),
            0.1,
        ),
    ]
    _wire_fakes(base_service, pairs)

    selected_pairs, route_name = base_service._retrieve_pairs("What is the salary credit date?")

    assert route_name is None
    assert len(selected_pairs) >= 3
    assert {
        doc.metadata["filename"] for doc, _ in selected_pairs
    } <= {"Payroll_and_Salary_Processing_Policy", "Compensation_and_Benefits_Policy"}


def test_retrieve_pairs_drops_title_only_chunks_when_enough_signal(base_service: RAGService):
    pairs = [
        (
            FakeDoc(
                "# Payroll and Salary Processing Policy",
                {
                    "filename": "Payroll_and_Salary_Processing_Policy",
                    "heading": "# Payroll and Salary Processing Policy",
                    "chunk_id": "pay-title",
                },
            ),
            0.12,
        ),
        (
            FakeDoc(
                "Salary for each calendar month is credited on or before the last working day.",
                {
                    "filename": "Payroll_and_Salary_Processing_Policy",
                    "heading": "Salary Credit Date",
                    "chunk_id": "pay-1",
                },
            ),
            0.2,
        ),
        (
            FakeDoc(
                "If the last working day is a holiday, salary is credited on the previous working day.",
                {
                    "filename": "Payroll_and_Salary_Processing_Policy",
                    "heading": "Holiday Handling",
                    "chunk_id": "pay-2",
                },
            ),
            0.24,
        ),
        (
            FakeDoc(
                "Salaries are credited to designated bank accounts and payslips are available on HRMS.",
                {
                    "filename": "Compensation_and_Benefits_Policy",
                    "heading": "Salary Disbursement",
                    "chunk_id": "pay-3",
                },
            ),
            0.29,
        ),
    ]
    _wire_fakes(base_service, pairs)

    selected_pairs, route_name = base_service._retrieve_pairs("What is the salary credit date?")

    assert route_name is None
    assert len(selected_pairs) >= 3
    assert "pay-title" not in {doc.metadata["chunk_id"] for doc, _ in selected_pairs}


def test_ask_returns_answer_and_respects_include_sources_false(base_service: RAGService):
    pairs = [
        (
            FakeDoc(
                "Salary is credited on the last working day of each month.",
                {
                    "filename": "Payroll_and_Salary_Processing_Policy",
                    "heading": "Salary Credit",
                    "chunk_id": "pay-1",
                },
            ),
            0.4,
        ),
        (
            FakeDoc(
                "If the credit date is a holiday, salary is processed on the previous working day.",
                {
                    "filename": "Payroll_and_Salary_Processing_Policy",
                    "heading": "Holiday Rule",
                    "chunk_id": "pay-2",
                },
            ),
            0.43,
        ),
        (
            FakeDoc(
                "Compensation policy defines monthly payroll cycle and statutory deductions.",
                {
                    "filename": "Compensation_and_Benefits_Policy",
                    "heading": "Payroll Cycle",
                    "chunk_id": "pay-3",
                },
            ),
            0.49,
        ),
    ]
    _, fake_llm = _wire_fakes(base_service, pairs, llm_answer="Salary is credited on the last working day.")

    result = base_service.ask("What is the salary credit date?", include_sources=False)

    assert result["answer"] == "Salary is credited on the last working day."
    assert result["sources"] == []
    assert {"retrieval", "generation", "total"} <= set(result["timings"].keys())
    assert fake_llm.prompts, "Expected the LLM to be called once."
    assert "What is the salary credit date?" in fake_llm.prompts[0]


def test_retrieve_debug_returns_route_and_sources(base_service: RAGService):
    pairs = [
        (
            FakeDoc(
                "Training budget is allocated annually by employee band.",
                {
                    "filename": "Training_and_Learning_Development_Policy",
                    "heading": "Budget Allocation",
                    "chunk_id": "tr-1",
                },
            ),
            0.52,
        ),
        (
            FakeDoc(
                "Employees can request sponsorship approval for certification programs.",
                {
                    "filename": "Training_and_Learning_Development_Policy",
                    "heading": "Sponsorship",
                    "chunk_id": "tr-2",
                },
            ),
            0.61,
        ),
    ]
    _wire_fakes(base_service, pairs)

    result = base_service.retrieve_debug("Are there any upskilling or training benefits?")

    assert result["route"] == "training"
    assert len(result["sources"]) >= 1
    assert result["sources"][0]["filename"] == "Training_and_Learning_Development_Policy"
