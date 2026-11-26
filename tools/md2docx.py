"""Minimal Markdown -> .docx converter for the report (headings, paragraphs with
**bold**/*italic*/`code`, bullet & numbered lists, pipe tables, images, code
blocks, block quotes).

    python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
"""
import re
import sys
from pathlib import Path
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

INLINE = re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)')


def add_runs(par, text, size=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith('`') and part.endswith('`'):
            r = par.add_run(part[1:-1]); r.font.name = 'Consolas'; r.font.size = Pt(9.5)
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        if size: r.font.size = size


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_fill); tcPr.append(shd)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    header, body = cells[0], [r for r in cells[2:]]
    t = doc.add_table(rows=1 + len(body), cols=len(header))
    t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.cell(0, j); c.text = ''
        add_runs(c.paragraphs[0], h, Pt(8.5)); c.paragraphs[0].runs and setattr(c.paragraphs[0].runs[0], 'bold', True)
        for r in c.paragraphs[0].runs: r.bold = True
        shade(c, 'E8EEF7')
    for i, row in enumerate(body, 1):
        is_base = bool(row) and '*baseline*' in row[0]          # the reference row of a comparison table
        for j in range(len(header)):
            c = t.cell(i, j); c.text = ''
            txt = row[j] if j < len(row) else ''
            add_runs(c.paragraphs[0], txt, Pt(8.5))
            if is_base or '*(baseline)*' in txt:
                shade(c, 'FFF3D1')
    doc.add_paragraph()


def convert(md_path, out_path):
    md_path = Path(md_path)
    lines = md_path.read_text(encoding='utf-8').splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2); sec.top_margin = sec.bottom_margin = Cm(2.0)
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(12)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    for lvl, size in ((1, 16), (2, 14), (3, 12.5)):
        h = doc.styles[f'Heading {lvl}']; h.font.name = 'Times New Roman'; h.font.size = Pt(size)
        h.font.color.rgb = RGBColor(0x10, 0x2a, 0x55)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith('```'):
                buf.append(lines[j]); j += 1
            p = doc.add_paragraph(); r = p.add_run('\n'.join(buf)); r.font.name = 'Consolas'; r.font.size = Pt(9)
            p.paragraph_format.left_indent = Cm(0.5)
            i = j + 1; continue
        if line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|?\s*:?-{2,}', lines[i + 1]):
            j = i; rows = []
            while j < len(lines) and lines[j].startswith('|'):
                rows.append(lines[j]); j += 1
            add_table(doc, rows); i = j; continue
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            lvl = len(m.group(1))
            if lvl == 1 and i == 0:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(m.group(2)); r.bold = True; r.font.size = Pt(17)
            else:
                doc.add_heading(m.group(2), level=min(lvl, 3))
            i += 1; continue
        m = re.match(r'^!\[(.*?)\]\((.*?)\)', line)
        if m:
            img = (md_path.parent / m.group(2)).resolve()
            if img.is_file():
                doc.add_picture(str(img), width=Cm(16.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(m.group(1)); r.italic = True; r.font.size = Pt(10)
            i += 1; continue
        m = re.match(r'^(\s*)[-*]\s+(.*)', line)
        if m:
            p = doc.add_paragraph(style='List Bullet' if not m.group(1) else 'List Bullet 2')
            add_runs(p, m.group(2)); i += 1; continue
        m = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if m:  # keep the number written in the markdown (Word's List Number would continue across lists)
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.6); p.paragraph_format.space_after = Pt(3)
            p.add_run(m.group(1) + '. '); add_runs(p, m.group(2)); i += 1; continue
        if line.startswith('>'):
            p = doc.add_paragraph(); p.paragraph_format.left_indent = Cm(0.8)
            add_runs(p, line.lstrip('> ').strip()); [setattr(r, 'italic', True) for r in p.runs]
            i += 1; continue
        if not line.strip():
            i += 1; continue
        buf = [line.strip()]; j = i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r'^(#|\||!\[|```|>|\s*[-*]\s|\s*\d+\.\s)', lines[j]):
            buf.append(lines[j].strip()); j += 1
        p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, ' '.join(buf)); i = j
    doc.save(out_path)
    print('Saved', out_path)


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
