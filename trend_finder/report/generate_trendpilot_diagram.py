"""
Generate a professional architecture/overview diagram for TrendPilot:
Modular agent-based system for LinkedIn content automation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
DARK_BLUE = '#0D2137'
MID_BLUE = '#1B4F72'
LIGHT_BLUE = '#2E86C1'
ORANGE = '#E67E22'
GREEN = '#27AE60'
RED = '#E74C3C'
PURPLE = '#8E44AD'
TEAL = '#17A589'
WHITE = '#FFFFFF'
DARK_TEXT = '#2C3E50'
SUBTLE_TEXT = '#7F8C8D'
LIGHT_BG = '#F8F9FA'

MODULE_COLORS = [LIGHT_BLUE, GREEN, ORANGE, PURPLE]
MODULE_BGS = ['#EBF5FB', '#EAFAF1', '#FEF9E7', '#F5EEF8']
MODULE_BORDERS = ['#2E86C1', '#27AE60', '#E67E22', '#8E44AD']

# ═══════════════════════════════════════
# TITLE
# ═══════════════════════════════════════
ax.text(10, 13.5, 'TrendPilot', fontsize=30, fontweight='bold',
        ha='center', va='center', color=DARK_BLUE, fontfamily='sans-serif')
ax.text(10, 12.95, 'Modular Agent-Based System for LinkedIn Content Automation',
        fontsize=13, ha='center', va='center', color=SUBTLE_TEXT,
        fontfamily='sans-serif', style='italic')

# ═══════════════════════════════════════
# PIPELINE FLOW ARROW (horizontal backbone)
# ═══════════════════════════════════════
pipe_y = 9.2
ax.annotate('', xy=(18.5, pipe_y), xytext=(1.5, pipe_y),
            arrowprops=dict(arrowstyle='->', color='#BDC3C7', lw=4, alpha=0.5))
ax.text(10, 9.6, 'END-TO-END AUTOMATED PIPELINE', fontsize=9, ha='center',
        va='center', color=SUBTLE_TEXT, fontweight='bold', fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=WHITE, edgecolor='#BDC3C7',
                  alpha=0.9))

# ═══════════════════════════════════════
# FOUR MODULES (main boxes)
# ═══════════════════════════════════════
modules = [
    {
        'num': '1',
        'title': 'Trend\nIdentification',
        'icon': 'M1',
        'details': [
            'GPT-4o + Google Trends',
            'Interactive topic selection',
            'Hallucination guards',
            '4 iterations completed',
        ],
    },
    {
        'num': '2',
        'title': 'Post\nGeneration',
        'icon': 'M2',
        'details': [
            'GPT-4o-mini LLM engine',
            'Master system prompt',
            'Parameterized user prompts',
            'Multi-variant hook styles',
        ],
    },
    {
        'num': '3',
        'title': 'Engagement\nPrediction',
        'icon': 'M3',
        'details': [
            'Random Forest  R\u00b2=0.59',
            'LightGBM  R\u00b2=0.53',
            '31,996 posts \u2022 85 features',
            '69 influencer profiles',
        ],
    },
    {
        'num': '4',
        'title': 'Visual\nGeneration',
        'icon': 'M4',
        'details': [
            'LLM classification router',
            'Mermaid diagram renderer',
            'Stable Diffusion images',
            'CLI orchestration layer',
        ],
    },
]

box_w = 3.8
box_h = 5.2
gap = 0.55
total_w = 4 * box_w + 3 * gap
start_x = (20 - total_w) / 2
box_top = pipe_y - 0.6
box_bot = box_top - box_h

for i, mod in enumerate(modules):
    x = start_x + i * (box_w + gap)
    col = MODULE_COLORS[i]
    bg = MODULE_BGS[i]
    border = MODULE_BORDERS[i]

    # Main box
    main_box = FancyBboxPatch((x, box_bot), box_w, box_h,
                               boxstyle="round,pad=0.15",
                               facecolor=bg, edgecolor=border, linewidth=2.2)
    ax.add_patch(main_box)

    # Colored header bar inside box
    bar_h = 1.55
    bar = FancyBboxPatch((x + 0.08, box_top - bar_h), box_w - 0.16, bar_h,
                          boxstyle="round,pad=0.08",
                          facecolor=col, edgecolor=col, linewidth=0, alpha=0.9)
    ax.add_patch(bar)

    # Module number badge
    badge_r = 0.28
    badge_cx = x + box_w / 2
    badge_cy = box_top - 0.35
    circle = plt.Circle((badge_cx, badge_cy), badge_r, facecolor=WHITE,
                         edgecolor=col, linewidth=2, zorder=6)
    ax.add_patch(circle)
    ax.text(badge_cx, badge_cy, mod['num'], fontsize=12, fontweight='bold',
            ha='center', va='center', color=col, fontfamily='sans-serif', zorder=7)

    # Module title (inside header)
    ax.text(badge_cx, box_top - 1.05, mod['title'], fontsize=11.5,
            fontweight='bold', ha='center', va='center', color=WHITE,
            fontfamily='sans-serif', linespacing=1.15)

    # Detail bullets
    detail_top = box_top - bar_h - 0.35
    for j, detail in enumerate(mod['details']):
        dy = detail_top - j * 0.6
        # bullet dot
        ax.plot(x + 0.35, dy, 'o', color=col, markersize=5)
        ax.text(x + 0.6, dy, detail, fontsize=8.5, ha='left', va='center',
                color=DARK_TEXT, fontfamily='sans-serif')

    # Connector arrows between modules
    if i < 3:
        arr_x_start = x + box_w + 0.02
        arr_x_end = x + box_w + gap - 0.02
        arr_y = box_bot + box_h / 2
        ax.annotate('', xy=(arr_x_end, arr_y), xytext=(arr_x_start, arr_y),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.5))

# ═══════════════════════════════════════
# KEY ACHIEVEMENTS STRIP (bottom)
# ═══════════════════════════════════════
strip_y = 2.2
ax.text(10, strip_y + 0.55, 'KEY ACHIEVEMENTS', fontsize=12, fontweight='bold',
        ha='center', va='center', color=DARK_BLUE, fontfamily='sans-serif')

achievements = [
    ('4 Iterations', 'BuzzSumo \u2192 spaCy NER\n\u2192 GPT-4o pipeline', LIGHT_BLUE),
    ('Multi-Variant', '3 hook styles per topic\n100\u2013200 word posts', GREEN),
    ('85 Features', '9 categories \u2022 31,996 posts\nR\u00b2 up to 0.59', ORANGE),
    ('Unified Service', 'LLM + Mermaid + SD\nCLI orchestration', PURPLE),
]

ach_w = 3.5
ach_h = 1.25
ach_gap = 0.45
ach_total = 4 * ach_w + 3 * ach_gap
ach_start = (20 - ach_total) / 2
ach_bot = strip_y - 1.1

for i, (title, desc, col) in enumerate(achievements):
    ax_x = ach_start + i * (ach_w + ach_gap)

    card = FancyBboxPatch((ax_x, ach_bot), ach_w, ach_h,
                           boxstyle="round,pad=0.1",
                           facecolor=WHITE, edgecolor=col, linewidth=2)
    ax.add_patch(card)

    # Thin top accent
    accent = FancyBboxPatch((ax_x + 0.05, ach_bot + ach_h - 0.12),
                             ach_w - 0.1, 0.08,
                             boxstyle="round,pad=0.02",
                             facecolor=col, edgecolor=col, linewidth=0)
    ax.add_patch(accent)

    cx = ax_x + ach_w / 2
    ax.text(cx, ach_bot + ach_h - 0.35, title, fontsize=10, fontweight='bold',
            ha='center', va='center', color=col, fontfamily='sans-serif')
    ax.text(cx, ach_bot + 0.35, desc, fontsize=8, ha='center', va='center',
            color=DARK_TEXT, fontfamily='sans-serif', linespacing=1.25)

# ── Remaining work footnote ──
ax.text(10, 0.35, 'Remaining:  Unit Testing  \u2502  Integration Testing  \u2502  Streamlit App Development',
        fontsize=8.5, ha='center', va='center', color=SUBTLE_TEXT,
        fontfamily='sans-serif', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDFEFE', edgecolor='#D5D8DC',
                  linewidth=1, alpha=0.8))

plt.tight_layout(pad=0.5)
output_path = os.path.join(os.path.dirname(__file__), 'TrendPilot_Architecture_Diagram.png')
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Diagram saved to: {output_path}')
