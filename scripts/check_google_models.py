"""
List available generation and embedding models.

Run:
    python scripts/check_google_models.py
"""

from __future__ import annotations

import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("GOOGLE_API_KEY is not set in .env")
        return 1

    genai.configure(api_key=api_key)

    print("\nAvailable generation models:\n")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"  - {model.name}")

    print("\nAvailable embedding models:\n")
    for model in genai.list_models():
        if "embedContent" in model.supported_generation_methods:
            print(f"  - {model.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
