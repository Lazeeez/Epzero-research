"""Convert manuscript-clef2027.md to LNCS LaTeX (Springer `llncs` class), for measuring and submitting.

    python research/longeval/paper/md2lncs.py [MANUSCRIPT.md] [OUTPUT.tex]

Handles the manuscript's own subset of Markdown: `#` title, `##`/`###` sections, **bold**, *italic*,
`code`, pipe tables introduced by a bold "Table N." caption line, bullet lists, the numbered reference
list, and the figure captions ("**Figure N** (`figures/x.pdf`). caption"), each placed as a float after
the paragraph that first cites it. The draft notes under the title are left out. Compile with XeLaTeX
(for example `tectonic`), which reads the Unicode text directly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIAL = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
           '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}


def escape(text):
    return ''.join(SPECIAL.get(c, c) for c in text)


def inline(text):
    """Bold, italic and code spans, with everything else escaped."""
    out, pos = [], 0
    for m in re.finditer(r'\*\*(.+?)\*\*|`([^`]+)`|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', text):
        out.append(escape(text[pos:m.start()]))
        if m.group(1) is not None:
            out.append(r'\textbf{' + inline(m.group(1)) + '}')
        elif m.group(2) is not None:
            out.append(r'\texttt{' + escape(m.group(2)) + '}')
        else:
            out.append(r'\emph{' + inline(m.group(3)) + '}')
        pos = m.end()
    out.append(escape(text[pos:]))
    return ''.join(out)


def table(lines, caption, label):
    rows = [[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    header, body = rows[0], [r for r in rows[2:]]
    spec = 'l' + 'c' * (len(header) - 1)
    out = [r'\begin{table}[t]', r'\centering', r'\caption{' + inline(caption) + '}', r'\label{' + label + '}',
           r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{' + spec + '}', r'\hline',
           ' & '.join(inline(c) for c in header) + r' \\', r'\hline']
    out += [' & '.join(inline(c) for c in row) + r' \\' for row in body]
    out += [r'\hline', r'\end{tabular}}', r'\end{table}']
    return '\n'.join(out)


def convert(md):
    lines = md.replace('\r\n', '\n').split('\n')
    title = lines[0].lstrip('# ').strip()
    figures, references = {}, []
    for line in lines:
        m = re.match(r'\*\*Figure (\d+)\*\* \(`([^`]+)`\)\. (.*)', line)
        if m:
            figures[m.group(1)] = (m.group(2), m.group(3))
    # Figure captions may continue on the following lines.
    for number in list(figures):
        start = next(i for i, l in enumerate(lines) if l.startswith(f'**Figure {number}**'))
        caption = [figures[number][1]]
        for l in lines[start + 1:]:
            if not l.strip() or l.startswith('**Figure') or l.startswith('#'):
                break
            caption.append(l.strip())
        figures[number] = (figures[number][0], ' '.join(caption))
    body, placed, i = [], set(), 1
    abstract, section = [], None
    paragraph = []

    def flush():
        if not paragraph:
            return
        text = ' '.join(l.strip() for l in paragraph)
        paragraph.clear()
        target = abstract if section == 'abstract' else body
        target.append(inline(text))
        for number in re.findall(r'Figure (\d+)', text):
            if number in figures and number not in placed and section != 'abstract':
                placed.add(number)
                path, caption = figures[number]
                body.append('\n'.join([r'\begin{figure}[t]', r'\centering',
                                       r'\includegraphics[width=\textwidth]{' + path + '}',
                                       r'\caption{' + inline(caption) + '}', r'\label{fig' + number + '}', r'\end{figure}']))
    while i < len(lines):
        line = lines[i]
        if line.startswith('**Draft for') or line.startswith('This version restructures') or i < 8 and section is None and not line.startswith('#'):
            i += 1
            continue
        if line.startswith('## '):
            flush()
            name = line[3:].strip()
            if name == 'Abstract':
                section = 'abstract'
            elif name == 'References':
                section = 'references'
            else:
                section = name
                body.append(r'\section{' + inline(re.sub(r'^\d+\s+', '', name)) + '}')
            i += 1
            continue
        if line.startswith('### '):
            flush()
            body.append(r'\subsection{' + inline(re.sub(r'^[\d.]+\s+', '', line[4:].strip())) + '}')
            i += 1
            continue
        if line.startswith('**Figure '):                    # placed as floats
            flush()
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        if section == 'references':
            m = re.match(r'(\d+)\. (.*)', line)
            if m:
                references.append(m.group(2))
            i += 1
            continue
        m = re.match(r'\*\*Table (\d+)\.\*\* (.*)', line)
        if m:
            flush()
            caption = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith('|'):
                caption.append(lines[i].strip())
                i += 1
            while i < len(lines) and not lines[i].startswith('|'):
                i += 1
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i])
                i += 1
            body.append(table(rows, ' '.join(caption), 'tab' + m.group(1)))
            continue
        if line.startswith('| '):                            # a table without a caption line
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i])
                i += 1
            body.append(table(rows, '', 'tab-x').replace(r'\caption{}', ''))
            continue
        if line.startswith('- '):
            flush()
            items = []
            while i < len(lines) and (lines[i].startswith('- ') or lines[i].startswith('  ')):
                if lines[i].startswith('- '):
                    items.append(lines[i][2:].strip())
                else:
                    items[-1] += ' ' + lines[i].strip()
                i += 1
            body.append('\\begin{itemize}\n' + '\n'.join(r'\item ' + inline(t) for t in items) + '\n\\end{itemize}')
            continue
        if line.startswith('  D_A = '):                    # the manuscript's one displayed formula
            flush()
            body.append(r'\[ D_A = [\mathrm{nDCG}(A, s_3) - \mathrm{nDCG}(\mathrm{exact}, s_3)]'
                        r' - [\mathrm{nDCG}(A, s_1) - \mathrm{nDCG}(\mathrm{exact}, s_1)] \]')
            i += 1
            continue
        if not line.strip():
            flush()
        elif not line.startswith('# '):
            paragraph.append(line)
        i += 1
    flush()
    refs = '\n'.join(r'\bibitem{r' + str(n) + '} ' + inline(text) for n, text in enumerate(references, 1))
    return '\n'.join([
        r'\documentclass[runningheads]{llncs}',
        r'\usepackage{graphicx}',
        r'\usepackage{fontspec}',
        r'\setmainfont{lmroman10}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold, ItalicFont=*-italic, '
        r'BoldItalicFont=*-bolditalic]',
        r'\setmonofont{lmmono10}[Extension=.otf, UprightFont=*-regular]',
        r'\begin{document}',
        r'\title{' + inline(title) + '}',
        r'\titlerunning{Does the Index Matter Over Time?}',
        r'\author{Author}',
        r'\authorrunning{Author}',
        r'\institute{Institution}',
        r'\maketitle',
        r'\begin{abstract}', '\n\n'.join(abstract), r'\end{abstract}',
        '\n\n'.join(body),
        r'\begin{thebibliography}{99}', refs, r'\end{thebibliography}',
        r'\end{document}', ''])


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'manuscript-clef2027.md'
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'clef2027.tex'
    target.write_text(convert(source.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    print(target)


if __name__ == '__main__':
    main()
