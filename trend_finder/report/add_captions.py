"""
Add table numbers and figure numbers to TrendPilot_Comprehensive_Report copy.docx
"""
from docx import Document
from docx.oxml.ns import qn
from lxml import etree
import os

INPUT = os.path.join(os.path.dirname(__file__), 'TrendPilot_Comprehensive_Report copy.docx')
OUTPUT = INPUT

doc = Document(INPUT)

table_captions = [
    'Technology Stack',
    'Data Sources',
    'Iteration Summary – Module 1 Development',
    'spaCy NER Disambiguation Failures',
    'Key Learnings – Module 1',
    'Data Quality Issues and Treatment',
    'Feature Engineering – 85 Features Across 9 Categories',
    'Reactions Prediction Model Performance',
    'Comments Prediction Model Performance',
    'Top Predictors of Reactions (Feature Importance)',
    'Top Predictors of Comments (Feature Importance)',
    'Feature Category Importance Comparison',
    'Data Flow Between Modules',
    'Current Status Summary',
]

figure_captions = [
    'TrendPilot System Architecture Overview',
]


def make_caption_para(text):
    """Create a standalone caption paragraph element (not attached to any tree)."""
    p = etree.Element(qn('w:p'))

    pPr = etree.SubElement(p, qn('w:pPr'))
    jc = etree.SubElement(pPr, qn('w:jc'))
    jc.set(qn('w:val'), 'center')
    spacing = etree.SubElement(pPr, qn('w:spacing'))
    spacing.set(qn('w:before'), '120')
    spacing.set(qn('w:after'), '80')

    run = etree.SubElement(p, qn('w:r'))
    rPr = etree.SubElement(run, qn('w:rPr'))
    etree.SubElement(rPr, qn('w:b'))
    etree.SubElement(rPr, qn('w:i'))
    sz = etree.SubElement(rPr, qn('w:sz'))
    sz.set(qn('w:val'), '20')  # 10pt
    color = etree.SubElement(rPr, qn('w:color'))
    color.set(qn('w:val'), '003366')
    rFonts = etree.SubElement(rPr, qn('w:rFonts'))
    rFonts.set(qn('w:ascii'), 'Calibri')
    rFonts.set(qn('w:hAnsi'), 'Calibri')

    t = etree.SubElement(run, qn('w:t'))
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

    return p


body = doc.element.body

# ── Add table captions (before each table) ──
table_elements = list(body.findall(qn('w:tbl')))
for i, tbl_el in enumerate(table_elements):
    caption = f'Table {i + 1}: {table_captions[i]}' if i < len(table_captions) else f'Table {i + 1}'
    cap_p = make_caption_para(caption)
    tbl_el.addprevious(cap_p)

# ── Add figure captions (after each image paragraph) ──
fig_num = 0
para_elements = list(body.findall(qn('w:p')))
for p_el in para_elements:
    has_image = p_el.findall('.//' + qn('wp:inline')) or p_el.findall('.//' + qn('wp:anchor'))
    if has_image:
        caption = f'Figure {fig_num + 1}: {figure_captions[fig_num]}' if fig_num < len(figure_captions) else f'Figure {fig_num + 1}'
        cap_p = make_caption_para(caption)
        p_el.addnext(cap_p)
        fig_num += 1

doc.save(OUTPUT)
print(f'Added {len(table_elements)} table captions and {fig_num} figure caption(s).')
print(f'Saved to: {OUTPUT}')
