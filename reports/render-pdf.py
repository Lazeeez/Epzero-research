"""Render a Markdown statement to a print-styled HTML file for Chrome's PDF printer.

Usage (needs the `markdown` package):
    python reports/render-pdf.py reports/longeval-research-statement.md statement.html
    chrome --headless=new --no-pdf-header-footer --print-to-pdf=reports/longeval-research-statement.pdf statement.html
"""
import re, sys
from pathlib import Path
import markdown

src, out = Path(sys.argv[1]), Path(sys.argv[2])
text = src.read_text(encoding='utf-8')
# Python-Markdown nests lists and list paragraphs only at 4-space indents; the source uses 2.
text = re.sub(r'(?m)^  (?=\S)', '    ', text)
body = markdown.markdown(text, extensions=['tables', 'sane_lists', 'smarty'], output_format='html5')
# Title block: first h1 and the byline paragraph after it.
body = re.sub(r'^<h1>(.*?)</h1>\s*<p>(.*?)</p>', r'<header><h1>\1</h1><p class="byline">\2</p></header>', body, count=1, flags=re.S)
# Smaller type for the reference list.
body = re.sub(r'(<h2>References</h2>\s*)<ul>', r'\1<ul class="refs">', body)
css = """
@page { size: Letter; margin: 0.62in 0.8in 0.66in 0.8in;
        @bottom-center { content: counter(page); font: 9pt 'Segoe UI', Arial, sans-serif; color: #666; } }
html { font-size: 9.7pt; }
body { font-family: Charter, 'Bitstream Charter', Georgia, 'Times New Roman', serif; line-height: 1.38;
       color: #111; margin: 0; text-align: left; hyphens: auto; }
header { border-bottom: 1.5px solid #1f3a5f; padding-bottom: 8pt; margin-bottom: 14pt; }
h1 { font-family: 'Segoe UI', Arial, sans-serif; font-size: 17pt; line-height: 1.25; color: #1f3a5f; margin: 0 0 6pt; font-weight: 600; }
.byline { margin: 0; color: #444; font-size: 10pt; }
h2 { font-family: 'Segoe UI', Arial, sans-serif; font-size: 12pt; color: #1f3a5f; margin: 11pt 0 4pt;
     font-weight: 600; break-after: avoid; }
p { margin: 0 0 6.5pt; orphans: 3; widows: 3; }
p:has(+ blockquote), p:has(+ table), p:has(+ ul), p:has(+ ol) { break-after: avoid; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin: 0 0 3pt; }
li > ul { margin-top: 3pt; }
li > p { margin: 3pt 0; }
blockquote { margin: 8pt 0 10pt; padding: 7pt 12pt; border-left: 3px solid #1f3a5f; background: #f2f5f9;
             font-style: italic; break-inside: avoid; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 9pt; font-size: 9.1pt; }
table { break-inside: avoid; }
td { white-space: nowrap; } td:first-child, td:last-child { white-space: normal; }
th, td { border: 0.75px solid #b8c2cf; padding: 3pt 7pt; vertical-align: top; text-align: left; }
th { background: #e6ecf3; font-family: 'Segoe UI', Arial, sans-serif; font-weight: 600; }
a { color: #1f4e8c; text-decoration: none; overflow-wrap: anywhere; }
strong { font-weight: 700; }
ul.refs { font-size: 8.3pt; line-height: 1.22; padding-left: 14pt; margin-bottom: 0; }
ul.refs li { margin-bottom: 1pt; }
"""
title = re.search(r'<h1>(.*?)</h1>', body, re.S).group(1)
out.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{re.sub("<.*?>", "", title)}</title>'
               f'<style>{css}</style></head><body>{body}</body></html>', encoding='utf-8')
print(out)
