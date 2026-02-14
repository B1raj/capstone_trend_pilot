# Unified Visual Generation Service (CLI)

"""
This script integrates LLM, diagram, and image generation modules.
It takes user input, queries the LLM, and routes to the appropriate generator.
"""

import sys
import subprocess
import os

LLM_SCRIPT = os.path.join(os.path.dirname(__file__), '../llm/query_llm.py')
MERMAID_SCRIPT = os.path.join(os.path.dirname(__file__), '../diagram/render_mermaid.py')
IMAGE_SCRIPT = os.path.join(os.path.dirname(__file__), '../image/generate_image.py')


def main():
    if len(sys.argv) < 2:
        print("Usage: python visual_service.py <user_input>")
        sys.exit(1)
    user_input = sys.argv[1]
    print("Querying LLM...")
    llm_result = subprocess.check_output([sys.executable, LLM_SCRIPT, user_input], encoding="utf-8")
    if llm_result.startswith("Type: architecture"):
        # Extract Mermaid code
        mermaid_code = llm_result.split('Mermaid:\n', 1)[-1].strip()
        with open("diagram.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print("Generating diagram...")
        subprocess.run([sys.executable, MERMAID_SCRIPT, "diagram.mmd", "diagram.svg"])
    elif llm_result.startswith("Type: image"):
        print("Generating image...")
        subprocess.run([sys.executable, IMAGE_SCRIPT, user_input, "image.png"])
    else:
        print("No visual output required.")

if __name__ == "__main__":
    main()
