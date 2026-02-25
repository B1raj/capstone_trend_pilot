#!/usr/bin/env python3
"""
TrendPilot Architecture Diagram — Excalidraw Generator
-------------------------------------------------------
Run:    python generate_diagram.py
Output: trendpilot_diagram.excalidraw   (same folder as this script)

Open the output file at excalidraw.com by drag-and-drop,
or in the Excalidraw VS Code extension.
"""

import json
import os
import random

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level element builders
# ─────────────────────────────────────────────────────────────────────────────

def _uid() -> str:
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))


def _seed() -> int:
    return random.randint(100_000, 9_999_999)


def _base(tp, x, y, w, h, stroke="#000000", bg="transparent", sw=2, r=None):
    """Minimal Excalidraw element skeleton."""
    return {
        "id": _uid(), "type": tp,
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0,
        "strokeColor": stroke, "backgroundColor": bg,
        "fillStyle": "solid", "strokeWidth": sw, "strokeStyle": "solid",
        "roughness": 0, "opacity": 100,
        "groupIds": [], "frameId": None,
        "roundness": ({"type": 3, "value": r} if r is not None else None),
        "seed": _seed(), "version": 1, "versionNonce": _seed(),
        "isDeleted": False, "boundElements": None,
        "updated": 1, "link": None, "locked": False,
    }


def rect(x, y, w, h, bg="transparent", stroke="#000000", sw=2, r=10):
    """Rounded rectangle."""
    return _base("rectangle", x, y, w, h, stroke=stroke, bg=bg, sw=sw, r=r)


def circle(x, y, d, bg="transparent", stroke="#000000", sw=2):
    """Circle (ellipse with equal W/H).  x,y = top-left corner, d = diameter."""
    el = _base("ellipse", x, y, d, d, stroke=stroke, bg=bg, sw=sw)
    el["roundness"] = {"type": 2}
    return el


def txt(x, y, content, size=14, color="#000000", align="center", ff=2, w=None):
    """
    Text element.
    x,y  = top-left corner of bounding box
    ff   = fontFamily: 1=hand-drawn, 2=Helvetica, 3=monospace
    w    = explicit width (auto-calculated when None)
    """
    lines = content.split("\n")
    tw = w or max(len(l) * size * 0.60 + 12 for l in lines)
    th = size * 1.35 * len(lines) + 6
    el = _base("text", x, y, tw, th, stroke=color, bg="transparent", sw=1)
    el.update({
        "text": content, "originalText": content,
        "fontSize": size, "fontFamily": ff,
        "textAlign": align, "verticalAlign": "middle",
        "containerId": None, "roundness": None,
    })
    return el


def arrow(x1, y1, x2, y2, color="#94A3B8", sw=2):
    """Arrow from (x1,y1) → (x2,y2)."""
    el = _base("arrow", x1, y1, x2 - x1, y2 - y1,
                stroke=color, bg="transparent", sw=sw)
    el.update({
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "lastCommittedPoint": None,
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
        "roundness": {"type": 2},
    })
    return el


def hline(x1, x2, y, color="#CBD5E0", sw=1):
    """Horizontal rule (no arrowhead)."""
    el = arrow(x1, y, x2, y, color=color, sw=sw)
    el["endArrowhead"] = None
    return el


# ─────────────────────────────────────────────────────────────────────────────
# Content
# ─────────────────────────────────────────────────────────────────────────────

# (header_bg, body_bg, stroke/accent)
THEMES = [
    ("#2B6CB0", "#DBEEFF", "#2B6CB0"),   # 1 — blue
    ("#276749", "#E6F7EE", "#276749"),   # 2 — green
    ("#C05621", "#FFF3E0", "#C05621"),   # 3 — orange
    ("#6B46C1", "#F0EEFF", "#6B46C1"),   # 4 — purple
]

BOXES = [
    ("1", "Trend\nIdentification", [
        "GPT-4o + Google Trends",
        "Interactive topic selection",
        "Hallucination guards",
        "4 iterations completed",
    ]),
    ("2", "Post\nGeneration", [
        "GPT-4o-mini LLM engine",
        "Master system prompt",
        "Parameterized user prompts",
        "Multi-variant hook styles",
    ]),
    ("3", "Engagement\nPrediction", [
        "HGBR best \u2014 reactions & comments",
        "Reactions  Log R\u00b2=0.63  \u03c1=0.69",
        "Comments   Log R\u00b2=0.77  \u03c1=0.79",
        "1.6k rows of real LinkedIn posts",
        "84\u201386 features  \u2022  5 models",
    ]),
    ("4", "Visual\nGeneration", [
        "LLM classification router",
        "Graphviz diagram renderer",
        "qwen-plus  image-gen context",
        "z-image-turbo  image gen",
        "CLI orchestration layer",
    ]),
]

ACHIEVEMENTS = [
    ("4 Iterations",
     "BuzzSumo \u2192 SpaCy NER\n\u2192 GPT-4o pipeline"),
    ("Multi-Variant",
     "3 hook styles per topic\n100\u2013200 word posts"),
    ("HGBR Wins",
     "Reactions \u03c1=0.69  Comments \u03c1=0.79\nLog R\u00b2 up to 0.77"),
    ("Unified Service",
     "LLM + Graphviz + z-image-turbo\nCLI orchestration"),
]

FOOTER = (
    "Remaining:  Unit Testing  \u2502  "
    "Integration Testing  \u2502  "
    "Streamlit App Development"
)


# ─────────────────────────────────────────────────────────────────────────────
# Layout constants  (all values in Excalidraw canvas pixels)
# ─────────────────────────────────────────────────────────────────────────────

# Main column boxes
BX = [60, 360, 660, 960]   # left-x of each column
BY = 195                    # top-y of main boxes
BW = 270                    # box width
BH = 365                    # box height
HH = 112                    # coloured header height
D  = 48                     # number-circle diameter
GAP_COL = 30                # horizontal gap between boxes

# Derived
TOTAL_RIGHT = BX[3] + BW                          # right edge  = 1230
CX = (BX[0] + TOTAL_RIGHT) // 2                   # canvas centreX = 645


# ─────────────────────────────────────────────────────────────────────────────
# Build element list
# ─────────────────────────────────────────────────────────────────────────────

els = []


# ── Title & subtitle ──────────────────────────────────────────────────────────
els += [
    txt(CX - 130, 15, "TrendPilot",
        size=40, color="#1A202C", align="center", w=260),
    txt(CX - 310, 68,
        "Modular Agent-Based System for LinkedIn Content Automation",
        size=13, color="#718096", align="center", w=620),
]


# ── Pipeline banner label ─────────────────────────────────────────────────────
PY = 150
PW = 232
PILL_X = CX - PW // 2

els += [
    hline(BX[0], TOTAL_RIGHT, PY + 11),
    rect(PILL_X, PY, PW, 22, bg="#FFFFFF", stroke="#CBD5E0", sw=1, r=5),
    txt(PILL_X, PY + 1, "END-TO-END AUTOMATED PIPELINE",
        size=9, color="#718096", align="center", w=PW),
]


# ── Four main column boxes ────────────────────────────────────────────────────
for i, ((num, title, bullets), (hc, bc, sc)) in enumerate(zip(BOXES, THEMES)):
    bx = BX[i]

    # Card shell (body colour + border)
    els.append(rect(bx, BY, BW, BH, bg=bc, stroke=sc, sw=2, r=12))

    # Coloured header — rounded on top, patched square at bottom
    els.append(rect(bx, BY,        BW, HH,   bg=hc, stroke=hc, sw=0, r=12))
    els.append(rect(bx, BY+HH-12,  BW, 14,   bg=hc, stroke=hc, sw=0, r=0))

    # Number circle
    circle_x = bx + (BW - D) // 2   # centred horizontally
    circle_y = BY + 12
    els.append(circle(circle_x, circle_y, D, bg="#FFFFFF", stroke="#FFFFFF", sw=0))
    # Number text — vertically centred inside circle
    els.append(txt(circle_x, circle_y + D // 4, num,
                   size=22, color=hc, align="center", w=D))

    # Section title — white, below circle, still in header
    title_y = BY + 12 + D + 8
    els.append(txt(bx, title_y, title,
                   size=14, color="#FFFFFF", align="center", ff=2, w=BW))

    # Bullet points — in the body below header
    for j, bullet in enumerate(bullets):
        by_ = BY + HH + 20 + j * 38
        els.append(circle(bx + 15, by_ + 5, 9, bg=hc, stroke=hc, sw=0))
        els.append(txt(bx + 30, by_, bullet,
                       size=13, color="#2D3748", align="left", w=BW - 42))


# ── Horizontal connector arrows between boxes ─────────────────────────────────
ARROW_Y = BY + BH // 2
for i in range(3):
    els.append(arrow(BX[i] + BW + 5, ARROW_Y, BX[i+1] - 5, ARROW_Y,
                     color="#94A3B8", sw=2))


# ── KEY ACHIEVEMENTS section ──────────────────────────────────────────────────
ACH_Y = BY + BH + 55
ACH_W = BW        # cards align with boxes above
ACH_H = 108

els.append(txt(CX - 115, ACH_Y - 35, "KEY ACHIEVEMENTS",
               size=13, color="#1A202C", align="center", w=230))

for i, ((atitle, abody), (hc, bc, sc)) in enumerate(zip(ACHIEVEMENTS, THEMES)):
    ax = BX[i]
    els += [
        # Card
        rect(ax, ACH_Y, ACH_W, ACH_H, bg="#FFFFFF", stroke=sc, sw=1, r=8),
        # Coloured top bar
        rect(ax, ACH_Y, ACH_W, 6, bg=hc, stroke=hc, sw=0, r=3),
        # Achievement title
        txt(ax, ACH_Y + 18, atitle,
            size=13, color=sc, align="center", ff=2, w=ACH_W),
        # Body text
        txt(ax, ACH_Y + 52, abody,
            size=11, color="#4A5568", align="center", w=ACH_W),
    ]


# ── Footer ────────────────────────────────────────────────────────────────────
FY = ACH_Y + ACH_H + 30
FW = 600
els += [
    rect(CX - FW // 2, FY - 4, FW, 26, bg="#FFFFFF", stroke="#CBD5E0", sw=1, r=5),
    txt(CX - FW // 2, FY, FOOTER, size=10, color="#718096", align="center", w=FW),
]


# ─────────────────────────────────────────────────────────────────────────────
# Write .excalidraw file
# ─────────────────────────────────────────────────────────────────────────────

diagram = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": els,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trendpilot_diagram.excalidraw",
)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(diagram, f, ensure_ascii=False, indent=2)

print(f"\u2713 {len(els)} elements written \u2192 {out_path}")
print("  Drag & drop the file into https://excalidraw.com to open it.")
