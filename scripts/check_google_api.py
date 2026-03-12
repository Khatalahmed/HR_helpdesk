"""
Connectivity check for Google Generative AI API.

Run:
    python scripts/check_google_api.py
"""

from __future__ import annotations

import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    print(f"API key configured: {'Yes' if api_key else 'No'}")
    if not api_key:
        print("Set GOOGLE_API_KEY in .env first.")
        return 1

    genai.configure(api_key=api_key)  # type: ignore[arg-type]

    print("\nAvailable embedding models:\n")
    for model in genai.list_models():  # type: ignore[attr-defined]
        if "embedContent" in model.supported_generation_methods:
            print(f"  - {model.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
