from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRoute:
    name: str
    triggers: tuple[str, ...]
    policies: tuple[str, ...]
    required_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    top_k: int
    min_contexts: int


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


def combined_relevance_score(question_tokens: set[str], text: str, heading: str, distance: float) -> float:
    content_overlap = lexical_overlap_ratio(question_tokens, text)
    heading_overlap = lexical_overlap_ratio(question_tokens, heading)
    low_signal_penalty = 0.2 if is_low_signal_heading(heading) else 0.0
    return distance - 0.65 * content_overlap - 0.35 * heading_overlap + low_signal_penalty


def detect_policy_hints(question: str) -> list[str]:
    question_lower = question.lower()
    matches: list[str] = []
    for keywords, policy_names in POLICY_HINTS:
        if any(keyword in question_lower for keyword in keywords):
            matches.extend(policy_names)
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

