# LLM API Call Script (Ollama Llama 3.1)

"""
This script sends user input and context to a local Ollama Llama 3.1 model via API.
It classifies the input and, if architectural, generates Mermaid code.
"""

import requests
import sys

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama endpoint
MODEL = "llama3.1:8b"

PROMPT_TEMPLATE = """
You are an assistant that classifies user requests for visual generation. 
If the request is for an architectural/system/process diagram, respond with:
Type: architecture\nMermaid:\n<mermaid code here>
If the request is for a general image, respond with:
Type: image
If neither, respond with:
Type: none
User input: {user_input}
"""

def query_llm(user_input: str):
    prompt = PROMPT_TEMPLATE.format(user_input=user_input)
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=data)
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_llm.py <user_input>")
        sys.exit(1)
    user_input = sys.argv[1]
    print(query_llm(user_input))
