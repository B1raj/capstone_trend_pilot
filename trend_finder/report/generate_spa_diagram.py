"""
Generate a professional architecture/evolution diagram for:
"Single Page Applications: Are We Witnessing the Next Evolution in Modern Web Architecture"
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('white')

# Color palette
DARK_BLUE = '#0D2137'
MID_BLUE = '#1B4F72'
LIGHT_BLUE = '#2E86C1'
ORANGE = '#E67E22'
GREEN = '#27AE60'
RED = '#E74C3C'
WHITE = '#FFFFFF'
DARK_TEXT = '#2C3E50'
SUBTLE_TEXT = '#7F8C8D'
PURPLE = '#8E44AD'

# ═══════════════════════════════════════
# TITLE
# ═══════════════════════════════════════
ax.text(9, 11.5, 'Single Page Applications (SPA)', fontsize=24, fontweight='bold',
        ha='center', va='center', color=DARK_BLUE, fontfamily='sans-serif')
ax.text(9, 11.0, 'The Next Evolution in Modern Web Architecture', fontsize=14,
        ha='center', va='center', color=SUBTLE_TEXT, fontfamily='sans-serif', style='italic')

# ═══════════════════════════════════════
# TIMELINE (top section)
# ═══════════════════════════════════════
timeline_y = 10.0
ax.annotate('', xy=(16.0, timeline_y), xytext=(2.0, timeline_y),
            arrowprops=dict(arrowstyle='->', color=LIGHT_BLUE, lw=2.5))
ax.text(9, 10.3, 'EVOLUTION TIMELINE', fontsize=9, ha='center', va='center',
        color=LIGHT_BLUE, fontweight='bold', fontfamily='sans-serif')

eras = [
    (3.5,  'Traditional MPA',  '2000s',    '#E8D5B7', DARK_TEXT),
    (6.5,  'AJAX Enhanced',    '2005-10',  '#D4E6F1', MID_BLUE),
    (9.5,  'SPA Frameworks',   '2010-20',  '#A9DFBF', '#1A5632'),
    (12.5, 'SPA + SSR Hybrid', '2020s+',   '#F9E79F', '#7D6608'),
]

for x, label, year, bg, text_col in eras:
    # dot on timeline
    ax.plot(x, timeline_y, 'o', color=text_col, markersize=10, zorder=5)
    # era box below timeline
    box = FancyBboxPatch((x - 1.3, timeline_y - 0.9), 2.6, 0.65,
                         boxstyle="round,pad=0.1",
                         facecolor=bg, edgecolor=text_col, linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x, timeline_y - 0.5, label, fontsize=9, ha='center', va='center',
            color=text_col, fontweight='bold', fontfamily='sans-serif')
    ax.text(x, timeline_y - 1.1, year, fontsize=8, ha='center', va='center',
            color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════
# ARCHITECTURE PANEL (main section)
# ═══════════════════════════════════════
panel_top = 8.3
panel_bot = 2.8
panel = FancyBboxPatch((0.8, panel_bot), 16.4, panel_top - panel_bot,
                       boxstyle="round,pad=0.15",
                       facecolor='#F8F9FA', edgecolor='#D5D8DC', linewidth=1.2)
ax.add_patch(panel)

ax.text(9, 8.0, 'SPA ARCHITECTURE \u2014 HOW IT WORKS', fontsize=12, fontweight='bold',
        ha='center', va='center', color=DARK_BLUE, fontfamily='sans-serif')

# ── Client Side (left) ──
client_left = 1.3
client_w = 5.0
client_bot = 3.4
client_h = 4.1
client_box = FancyBboxPatch((client_left, client_bot), client_w, client_h,
                            boxstyle="round,pad=0.12",
                            facecolor='#EBF5FB', edgecolor=LIGHT_BLUE, linewidth=2)
ax.add_patch(client_box)
client_cx = client_left + client_w / 2  # 3.8
ax.text(client_cx, client_bot + client_h - 0.35, 'CLIENT (Browser)', fontsize=11,
        fontweight='bold', ha='center', va='center', color=MID_BLUE, fontfamily='sans-serif')

client_items = [
    (6.65, 'Single HTML Page'),
    (5.95, 'JavaScript App Bundle'),
    (5.25, 'Virtual DOM / Router'),
    (4.55, 'State Management'),
]
for y, label in client_items:
    bw = 3.8
    bx = client_cx - bw / 2
    box = FancyBboxPatch((bx, y - 0.22), bw, 0.44, boxstyle="round,pad=0.05",
                         facecolor='#D6EAF8', edgecolor=MID_BLUE, linewidth=1, alpha=0.85)
    ax.add_patch(box)
    ax.text(client_cx, y, label, fontsize=9, ha='center', va='center',
            color=MID_BLUE, fontweight='bold', fontfamily='sans-serif')

# ── Server Side (right) ──
server_left = 11.7
server_w = 5.0
server_bot = 3.4
server_h = 4.1
server_box = FancyBboxPatch((server_left, server_bot), server_w, server_h,
                            boxstyle="round,pad=0.12",
                            facecolor='#EAFAF1', edgecolor=GREEN, linewidth=2)
ax.add_patch(server_box)
server_cx = server_left + server_w / 2  # 14.2
ax.text(server_cx, server_bot + server_h - 0.35, 'SERVER (API Layer)', fontsize=11,
        fontweight='bold', ha='center', va='center', color='#1A5632', fontfamily='sans-serif')

server_items = [
    (6.65, 'REST / GraphQL API'),
    (5.95, 'Authentication (JWT)'),
    (5.25, 'Business Logic'),
    (4.55, 'Database Layer'),
]
for y, label in server_items:
    bw = 3.8
    bx = server_cx - bw / 2
    box = FancyBboxPatch((bx, y - 0.22), bw, 0.44, boxstyle="round,pad=0.05",
                         facecolor='#D5F5E3', edgecolor='#1A5632', linewidth=1, alpha=0.85)
    ax.add_patch(box)
    ax.text(server_cx, y, label, fontsize=9, ha='center', va='center',
            color='#1A5632', fontweight='bold', fontfamily='sans-serif')

# ── Communication arrows and labels (middle column) ──
arrow_left = client_left + client_w + 0.15   # 6.45
arrow_right = server_left - 0.15              # 11.55
mid_x = (arrow_left + arrow_right) / 2       # 9.0

# Request arrow (top)
req_y = 6.3
ax.annotate('', xy=(arrow_right, req_y), xytext=(arrow_left, req_y),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5,
                           connectionstyle='arc3,rad=0.08'))
ax.text(mid_x, req_y + 0.35, 'JSON Request', fontsize=9, ha='center', va='center',
        color=ORANGE, fontweight='bold', fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF9E7', edgecolor=ORANGE, alpha=0.9))

# Response arrow (bottom)
resp_y = 4.6
ax.annotate('', xy=(arrow_left, resp_y), xytext=(arrow_right, resp_y),
            arrowprops=dict(arrowstyle='->', color=LIGHT_BLUE, lw=2.5,
                           connectionstyle='arc3,rad=0.08'))
ax.text(mid_x, resp_y - 0.35, 'JSON Response', fontsize=9, ha='center', va='center',
        color=MID_BLUE, fontweight='bold', fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#EBF5FB', edgecolor=MID_BLUE, alpha=0.9))

# No Page Reload badge (centered between arrows)
badge_w = 3.6
badge_h = 0.5
badge_x = mid_x - badge_w / 2
badge_y = (req_y + resp_y) / 2 - badge_h / 2
badge = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                       boxstyle="round,pad=0.08",
                       facecolor=RED, edgecolor='#C0392B', linewidth=1.5, alpha=0.92)
ax.add_patch(badge)
ax.text(mid_x, badge_y + badge_h / 2, 'NO FULL PAGE RELOAD', fontsize=9,
        ha='center', va='center', color=WHITE, fontweight='bold', fontfamily='sans-serif')

# ═══════════════════════════════════════
# KEY ADVANTAGES (bottom section)
# ═══════════════════════════════════════
ax.text(9, 2.45, 'KEY ADVANTAGES', fontsize=11, fontweight='bold',
        ha='center', va='center', color=DARK_BLUE, fontfamily='sans-serif')

benefits = [
    (2.8,  'Faster UX',           'No page reloads,\ninstant transitions',   LIGHT_BLUE),
    (6.6,  'Rich Interactivity',   'Desktop-like feel\nin the browser',       GREEN),
    (10.4, 'Decoupled Frontend',   'Independent deploy\n& scaling',           ORANGE),
    (14.2, 'SEO via SSR',          'Server-side rendering\nfor discoverability', PURPLE),
]

ben_y = 1.3
for x, title, desc, color in benefits:
    bw = 3.2
    bh = 1.3
    box = FancyBboxPatch((x - bw / 2, ben_y - bh / 2), bw, bh,
                         boxstyle="round,pad=0.1",
                         facecolor=WHITE, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    # colored bar at top of card
    bar = FancyBboxPatch((x - bw / 2 + 0.05, ben_y + bh / 2 - 0.18), bw - 0.1, 0.13,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=color, linewidth=0)
    ax.add_patch(bar)
    ax.text(x, ben_y + 0.25, title, fontsize=9.5, ha='center', va='center',
            color=color, fontweight='bold', fontfamily='sans-serif')
    ax.text(x, ben_y - 0.2, desc, fontsize=8, ha='center', va='center',
            color=DARK_TEXT, fontfamily='sans-serif')

# ── Challenges footnote ──
ax.text(16.5, 0.25, 'Challenges:  Initial load time  |  SEO complexity  |  Memory management',
        fontsize=7.5, ha='right', va='center', color=SUBTLE_TEXT, fontfamily='sans-serif', style='italic')

plt.tight_layout(pad=0.5)
output_path = os.path.join(os.path.dirname(__file__), 'SPA_Architecture_Diagram.png')
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Diagram saved to: {output_path}')
