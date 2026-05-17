"""One-shot script to render docs/proposal.md to docs/proposal.pdf.

Uses python-markdown + WeasyPrint so no LaTeX or pandoc install is needed.
Run from the repository root:

    python docs/render_proposal_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


HERE = Path(__file__).resolve().parent
SRC = HERE / "proposal.md"
OUT = HERE / "proposal.pdf"

CSS_TEXT = """
@page {
    size: A4;
    margin: 22mm 20mm 22mm 20mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Helvetica", "Arial", sans-serif;
        font-size: 9pt;
        color: #666;
    }
}

html, body {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 10.5pt;
    line-height: 1.38;
    color: #111;
}

h1 {
    font-size: 17pt;
    margin-top: 0;
    margin-bottom: 4pt;
    border-bottom: 1px solid #444;
    padding-bottom: 4pt;
}
h2 {
    font-size: 12.5pt;
    margin-top: 14pt;
    margin-bottom: 4pt;
    color: #1a1a1a;
}
h3 {
    font-size: 11pt;
    margin-top: 10pt;
    margin-bottom: 3pt;
}

p { margin: 4pt 0; }

ul, ol {
    margin: 4pt 0 4pt 18pt;
    padding: 0;
}
li { margin: 2pt 0; }

hr {
    border: none;
    border-top: 1px solid #bbb;
    margin: 10pt 0;
}

code, pre {
    font-family: "Menlo", "Consolas", "Courier New", monospace;
    font-size: 9.5pt;
}
pre {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    padding: 6pt 8pt;
    border-radius: 3pt;
    white-space: pre-wrap;
    page-break-inside: avoid;
}
code { background: #f1f1f1; padding: 0 2pt; border-radius: 2pt; }
pre code { background: transparent; padding: 0; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 6pt 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #bbb;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
}
th { background: #ececec; }

strong { color: #000; }

/* keep the title block tight */
h1 + p { color: #444; margin-top: 2pt; }
"""


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1
    text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Proposal</title></head>
<body>
{html_body}
</body>
</html>
"""
    HTML(string=html_doc, base_url=str(HERE)).write_pdf(
        target=str(OUT),
        stylesheets=[CSS(string=CSS_TEXT)],
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
