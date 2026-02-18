# Mermaid Diagram Renderer

"""
This script takes Mermaid code as input and renders it to SVG or PNG using the Mermaid CLI.
Requires Node.js and Mermaid CLI installed locally.
"""

import subprocess
import sys
import os


def sanitize_mermaid_code(mermaid_code: str) -> str:
    """
    Sanitize Mermaid code string for rendering.
    - Converts escaped newlines (\\n) to real newlines.
    - Strips leading/trailing whitespace.
    - Ensures consistent line endings.
    """
    if not isinstance(mermaid_code, str):
        return ""
    code = mermaid_code.replace("\\n", "\n")
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.strip()
    return code

def render_mermaid(mermaid_code: str, output_path: str, format: str = "svg"):
    """
    Render Mermaid code to an image file using Mermaid CLI.
    Args:
        mermaid_code: Mermaid diagram code as a string.
        output_path: Path to save the rendered image.
        format: 'svg' or 'png'.
    """
    sanitized_code = sanitize_mermaid_code(mermaid_code)
    temp_input = "temp_diagram.mmd"
    with open(temp_input, "w", encoding="utf-8") as f:
        f.write(sanitized_code)
    try:
        # Try to use full path to mmdc if available
        mmdc_path = os.environ.get("MMDC_PATH")
        if not mmdc_path:
            # Default Windows npm global path 
            possible_paths = [
                os.path.expandvars(r"%APPDATA%/npm/mmdc.cmd"),
                os.path.expandvars(r"%APPDATA%/npm/mmdc"),
                "mmdc"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    mmdc_path = path
                    break
            else:
                mmdc_path = "mmdc"  # fallback
        cmd = [
            mmdc_path,
            "-i", temp_input,
            "-o", output_path,
            "-t", "default",
            "-b", "transparent",
            "-e", format
        ]
        subprocess.run(cmd, check=True)
        print(f"Diagram saved to {output_path}")
    except Exception as e:
        print(f"Error rendering diagram: {e}")
    # finally:
    #     if os.path.exists(temp_input):
    #         os.remove(temp_input)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python render_mermaid.py <input.mmd> <output.svg|png>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        code = f.read()
    render_mermaid(code, sys.argv[2], format=sys.argv[2].split(".")[-1])
