"""Build report/final_report.pdf from final_report.md using reportlab.

Reuses the same lightweight Markdown tokenizer as the docx generator
(headings, paragraphs, lists, fenced code, GFM tables, figure-image
lines, italic captions). Output is A4, two-column margins.
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, Preformatted,
)


SRC = Path("report/final_report.md")
OUT = Path("report/final_report.pdf")
FIG_DIR = Path("report/figures")


# ── Styles ──────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    NAVY = colors.HexColor("#1F3A5F")
    s = {}
    s["Title"] = ParagraphStyle(
        "Title", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=20, textColor=NAVY, alignment=1, spaceAfter=14,
    )
    s["H1"] = ParagraphStyle(
        "H1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, textColor=NAVY, spaceBefore=14, spaceAfter=8,
    )
    s["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=12.5, textColor=NAVY, spaceBefore=10, spaceAfter=6,
    )
    s["H3"] = ParagraphStyle(
        "H3", parent=base["Heading3"], fontName="Helvetica-Bold",
        fontSize=11, textColor=NAVY, spaceBefore=8, spaceAfter=4,
    )
    s["H4"] = ParagraphStyle(
        "H4", parent=base["Heading4"], fontName="Helvetica-Bold",
        fontSize=10.5, textColor=NAVY, spaceBefore=6, spaceAfter=3,
    )
    s["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica",
        fontSize=10.5, leading=14, alignment=4, spaceAfter=6,
    )
    s["Caption"] = ParagraphStyle(
        "Caption", parent=s["Body"], fontName="Helvetica-Oblique",
        fontSize=9.5, leading=12, alignment=1, spaceAfter=8,
        textColor=colors.HexColor("#444444"),
    )
    s["Bullet"] = ParagraphStyle(
        "Bullet", parent=s["Body"], leftIndent=20, bulletIndent=6,
        firstLineIndent=0, spaceAfter=3,
    )
    s["Code"] = ParagraphStyle(
        "Code", parent=s["Body"], fontName="Courier", fontSize=9, leading=11,
        leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=10,
        backColor=colors.HexColor("#F4F4F4"),
        borderColor=colors.HexColor("#DDDDDD"),
        borderWidth=0.5, borderPadding=6,
    )
    s["TableHead"] = ParagraphStyle(
        "TableHead", parent=s["Body"], fontName="Helvetica-Bold",
        fontSize=9.5, alignment=1, textColor=colors.white, leading=11,
    )
    s["TableCell"] = ParagraphStyle(
        "TableCell", parent=s["Body"], fontSize=9.5, leading=11,
        spaceAfter=0, alignment=0,
    )
    s["TableCellNum"] = ParagraphStyle(
        "TableCellNum", parent=s["TableCell"], alignment=2,
    )
    return s


# ── Markdown parser (mirrors make_docx.py) ──────────────────────────────

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
FIG_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*[:\-\s\|]+\s*$")


def _tokenize(md: str):
    lines = md.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("```"):
            block = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            yield ("code", "\n".join(block)); continue
        m = HEADING_RE.match(line)
        if m:
            yield ("h" + str(len(m.group(1))), m.group(2).strip())
            i += 1; continue
        if line.strip() == "---":
            yield ("hr", None); i += 1; continue
        m = FIG_RE.match(line)
        if m:
            yield ("img", (m.group(1), m.group(2))); i += 1; continue
        if ("|" in line and i + 1 < n
                and TABLE_SEP_RE.match(lines[i + 1])):
            header = [c.strip() for c in
                      line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in
                             lines[i].strip().strip("|").split("|")])
                i += 1
            yield ("table", (header, rows)); continue
        if line.lstrip().startswith(("* ", "- ")):
            items = []
            while i < n:
                lst = lines[i].lstrip()
                if lst.startswith(("* ", "- ")):
                    items.append(lst[2:].rstrip())
                    i += 1
                elif (lines[i].startswith(("  ", "\t"))
                      and lines[i].strip() and items):
                    # Continuation of the previous item.
                    items[-1] = items[-1] + " " + lines[i].strip()
                    i += 1
                else:
                    break
            yield ("ul", items); continue
        if not line.strip():
            yield ("blank", None); i += 1; continue
        para = [line]
        i += 1
        while i < n and lines[i].strip() and not (
            lines[i].startswith("#")
            or lines[i].startswith("![")
            or lines[i].startswith("```")
            or lines[i].lstrip().startswith(("* ", "- "))
            or lines[i].strip() == "---"
            or ("|" in lines[i] and i + 1 < n
                and TABLE_SEP_RE.match(lines[i + 1]))
        ):
            para.append(lines[i]); i += 1
        yield ("p", " ".join(s.strip() for s in para))


# ── Inline rich-text helpers ────────────────────────────────────────────

def _md_inline_to_rl(text: str) -> str:
    """**bold** → <b>…</b>, `code` → <font face=Courier>…</font>.
    Special characters '<', '>', '&' are escaped first."""
    out = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    # italic *…* — not matching the inner stars of any unmatched `**…**`
    out = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", out)
    out = re.sub(
        r"`([^`]+?)`",
        r"<font face='Courier' size='9'>\1</font>", out,
    )
    return out


# ── Renderer ────────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=22 * mm, bottomMargin=22 * mm,
        title="CE 4011 — Assignment 4 Final Report",
        author="Ali Utku Tekin",
    )
    s = _styles()
    story = []
    md = SRC.read_text(encoding="utf-8")
    tokens = list(_tokenize(md))

    first_h1_done = False
    for kind, payload in tokens:
        if kind == "h1":
            if not first_h1_done:
                first_h1_done = True
                story.append(Paragraph(_md_inline_to_rl(payload), s["Title"]))
            else:
                story.append(Paragraph(_md_inline_to_rl(payload), s["H1"]))
        elif kind == "h2":
            story.append(Paragraph(_md_inline_to_rl(payload), s["H2"]))
        elif kind == "h3":
            story.append(Paragraph(_md_inline_to_rl(payload), s["H3"]))
        elif kind == "h4":
            story.append(Paragraph(_md_inline_to_rl(payload), s["H4"]))
        elif kind == "p":
            if not first_h1_done:
                continue
            txt = payload
            # Treat "*Figure N. …*" / "*Table N. …*" paragraphs as captions:
            # centred italic Caption style. Any other italic / bold / code
            # is handled inline by _md_inline_to_rl.
            stripped = txt.lstrip()
            is_caption = (
                stripped.startswith("*Figure ")
                or stripped.startswith("*Table ")
                or stripped.startswith("*End of report*")
            )
            if is_caption:
                story.append(Paragraph(_md_inline_to_rl(txt), s["Caption"]))
            else:
                story.append(Paragraph(_md_inline_to_rl(txt), s["Body"]))
        elif kind == "ul":
            for it in payload:
                story.append(Paragraph(
                    _md_inline_to_rl(it),
                    style=s["Bullet"], bulletText="•",
                ))
            story.append(Spacer(1, 4))
        elif kind == "code":
            story.append(Preformatted(payload, s["Code"]))
        elif kind == "img":
            alt, path = payload
            img = Path(path)
            if not img.exists():
                # paths in the .md are relative to report/
                candidate = Path("report") / path
                if candidate.exists():
                    img = candidate
                else:
                    img = FIG_DIR / Path(path).name
            if img.exists():
                # Scale to text width but cap the height so two figures +
                # paragraphs can share a page. Aspect ratio is preserved
                # by shrinking the width proportionally when capped.
                from PIL import Image as PILImage
                with PILImage.open(img) as im:
                    iw, ih = im.size
                w = doc.width
                h = w * ih / iw
                max_h = 4.0 * inch                 # ≈ 10 cm
                if h > max_h:
                    h = max_h
                    w = h * iw / ih
                fig = Image(str(img), width=w, height=h, hAlign="CENTER")
                story.append(fig)
            else:
                story.append(Paragraph(
                    f"[missing image: {path}]", s["Body"]))
        elif kind == "table":
            header, rows = payload
            # Right-align numeric-looking cells, left otherwise.
            def cell_style(text):
                t = text.replace(",", "").replace("·", "")
                t = t.replace("kN", "").replace("·m", "").replace("m", "")
                t = t.replace("kip", "").replace("ft", "").replace("ft", "")
                t = t.strip()
                try:
                    float(t.replace("+", "").replace("−", "-")
                           .replace("×10", "e").split()[0])
                    return s["TableCellNum"]
                except (ValueError, IndexError):
                    return s["TableCell"]

            data = [[Paragraph(_md_inline_to_rl(h), s["TableHead"])
                     for h in header]]
            for row in rows:
                data.append([Paragraph(_md_inline_to_rl(c), cell_style(c))
                             for c in row])
            ncols = len(header)
            avail = doc.width
            # equal split unless first column is wider (description)
            col_w = [avail / ncols] * ncols
            tbl = Table(data, colWidths=col_w, repeatRows=1, hAlign="CENTER")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#1F3A5F")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F7F9FC")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))
        elif kind == "hr":
            story.append(Spacer(1, 8))
        elif kind == "blank":
            pass

    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    print(f"wrote {OUT}")


def _decorate(canvas, doc):
    """Page header / footer."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(22 * mm, 12 * mm,
                      "CE 4011 — Assignment 4 Final Report  ·  v0.40.2")
    canvas.drawRightString(
        A4[0] - 22 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    build()
