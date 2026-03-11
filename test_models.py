# test_models.py

import os

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("\nAvailable generation models:\n")
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"  - {model.name}")

print("\nAvailable embedding models:\n")
for model in genai.list_models():
    if "embedContent" in model.supported_generation_methods:
        print(f"  - {model.name}")
