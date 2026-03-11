from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail the build when RAG evaluation metrics fall below thresholds."
    )
    parser.add_argument(
        "--summary",
        default="rag_evaluation_summary.json",
        help="Path to rag evaluation summary JSON file.",
    )
    parser.add_argument("--faithfulness", type=float, default=0.90)
    parser.add_argument("--answer-relevancy", type=float, default=0.70)
    parser.add_argument("--context-precision", type=float, default=0.70)
    parser.add_argument("--context-recall", type=float, default=0.85)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pick_metric(scores: dict, *keys: str) -> float | None:
    for key in keys:
        value = scores.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)

    try:
        summary = load_summary(summary_path)
    except Exception as exc:
        print(f"[QUALITY GATE] ERROR: {exc}")
        return 1

    scores = summary.get("scores", {})
    if not isinstance(scores, dict):
        print("[QUALITY GATE] ERROR: 'scores' section missing or invalid in summary JSON.")
        return 1

    checks = [
        ("faithfulness", pick_metric(scores, "faithfulness"), args.faithfulness),
        ("answer_relevancy", pick_metric(scores, "answer_relevancy"), args.answer_relevancy),
        (
            "context_precision",
            pick_metric(scores, "llm_context_precision_with_reference", "context_precision"),
            args.context_precision,
        ),
        ("context_recall", pick_metric(scores, "context_recall"), args.context_recall),
    ]

    failures: list[str] = []
    print("[QUALITY GATE] Metric check:")
    for name, actual, minimum in checks:
        if actual is None:
            failures.append(f"{name}: missing")
            print(f"  - {name}: MISSING (required >= {minimum:.3f})")
            continue
        status = "PASS" if actual >= minimum else "FAIL"
        print(f"  - {name}: {actual:.4f} (required >= {minimum:.3f}) -> {status}")
        if actual < minimum:
            failures.append(f"{name}: {actual:.4f} < {minimum:.3f}")

    if failures:
        print("[QUALITY GATE] FAILED")
        for item in failures:
            print(f"  * {item}")
        return 1

    print("[QUALITY GATE] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
