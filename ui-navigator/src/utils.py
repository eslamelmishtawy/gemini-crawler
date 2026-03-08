from google import genai
from src.config import config

# Shared Gemini client — single instance across all agents
gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM responses."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return text.strip()
