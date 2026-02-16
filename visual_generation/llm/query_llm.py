# LLM API Call Script (Ollama Llama 3.1)

"""
This script sends user input and context to a local Ollama Llama 3.1 model via API.
It classifies the input and, if architectural, generates Mermaid code.
"""

import requests
import sys
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama endpoint
MODEL = "llama3.1:8b"

def send_prompt(prompt: str, model: str = None, base_url: str = OLLAMA_API_URL):
    """Send a raw prompt string to the Ollama API and return the unified text response.

    This function keeps model selection centralized (uses `MODEL` by default) and
    returns a concatenated text response to simplify callers.
    """
    use_model = model if model is not None else MODEL
    data = {
        "model": use_model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(base_url, json=data)
    response.raise_for_status()
    data = response.json()
    # Normalize common Ollama response shapes
    if isinstance(data, dict):
        if "response" in data:
            return data["response"]
        if "responses" in data and isinstance(data["responses"], list):
            parts = []
            for r in data["responses"]:
                content = r.get("content", [])
                for c in content:
                    if c.get("type") in ("output_text", "message") and "text" in c:
                        parts.append(c["text"])
            if parts:
                return "".join(parts)
    return json.dumps(data, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_llm.py <user_input>")
        sys.exit(1)
    user_input = sys.argv[1]
    print(send_prompt(user_input))
