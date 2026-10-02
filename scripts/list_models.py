"""
Lists every Gemini model your API key currently has access to, and
which ones support generateContent (the method our generator.py uses).

Run this any time config.LLM_MODEL 404s — model availability changes
faster than this codebase will, so this is the reliable source of
truth over any hardcoded model name (including the one in config.py).

Usage:
    python scripts/list_models.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main():
    client = genai.Client()
    print("Models supporting generateContent:\n")
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(f"  {m.name}")


if __name__ == "__main__":
    main()
