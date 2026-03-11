# test_api.py

import os

import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API key found: {'Yes' if api_key else 'No - check your .env file'}")
print(f"API key starts with: {api_key[:10]}...\n" if api_key else "")

genai.configure(api_key=api_key)

print("Fetching available embedding models...\n")
for model in genai.list_models():
    if "embedContent" in model.supported_generation_methods:
        print(f"  - {model.name}")
