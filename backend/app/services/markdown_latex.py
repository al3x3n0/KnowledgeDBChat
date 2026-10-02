"""Markdown to a compilable LaTeX document.

`export_document` handed its assembled markdown straight to the LaTeX
compiler: no `\\documentclass`, `#` headings, unescaped `%` and `_`. This
converts the same content items the DOCX and PDF builders are given, so the
three formats are derived from one reading of the document.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.docx_builder import markdown_to_content_items

_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_SECTIONS = {1: "section", 2: "section", 3: "subsection"}


def escape(text: Any) -> str:
    return "".join(_SPECIALS.get(ch, ch) for ch in str(text or ""))


def _inline(text: Any) -> str:
    """Escaped text with **bold**, *italic* and `code` kept."""
    out: List[str] = []
    for part in re.split(r"(\*\*.+?\*\*|`.+?`|\*.+?\*)", str(text or "")):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(r"\textbf{" + escape(part[2:-2]) + "}")
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            out.append(r"\texttt{" + escape(part[1:-1]) + "}")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(r"\emph{" + escape(part[1:-1]) + "}")
        else:
            out.append(escape(part))
    return "".join(out)


def _list(environment: str, items: List[Any]) -> str:
    rows = "\n".join(r"  \item " + _inline(item) for item in items)
    return f"\\begin{{{environment}}}\n{rows}\n\\end{{{environment}}}"


def _table(item: Dict[str, Any]) -> str:
    headers = [str(h) for h in item.get("headers") or []]
    rows = [list(r) for r in item.get("rows") or []]
    width = max([len(headers)] + [len(r) for r in rows] + [1])
    lines = [r"\begin{tabular}{" + "l" * width + "}", r"\hline"]
    if headers:
        lines += [" & ".join(_inline(h) for h in headers) + r" \\", r"\hline"]
    lines += [" & ".join(_inline(c) for c in row) + r" \\" for row in rows]
    return "\n".join(lines + [r"\hline", r"\end{tabular}"])


def markdown_to_latex(markdown: str, title: str = "") -> str:
    body: List[str] = []
    for item in markdown_to_content_items(markdown or ""):
        kind = item.get("type")
        if kind == "heading":
            command = _SECTIONS.get(int(item.get("level") or 2), "subsubsection")
            body.append(f"\\{command}{{{_inline(item.get('text'))}}}")
        elif kind == "bullet_list":
            body.append(_list("itemize", item.get("items") or []))
        elif kind == "numbered_list":
            body.append(_list("enumerate", item.get("items") or []))
        elif kind == "code_block":
            body.append(
                "\\begin{verbatim}\n"
                + str(item.get("code") or "")
                + "\n\\end{verbatim}"
            )
        elif kind == "table":
            body.append(_table(item))
        elif kind == "quote":
            body.append(
                "\\begin{quote}\n" + _inline(item.get("text")) + "\n\\end{quote}"
            )
        elif kind == "horizontal_rule":
            body.append(r"\medskip\hrule\medskip")
        elif kind == "page_break":
            body.append(r"\newpage")
        elif item.get("text"):
            body.append(_inline(item.get("text")))
    preamble = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[margin=1in]{geometry}",
        r"\title{" + _inline(title) + "}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle" if title else "",
    ]
    return (
        "\n".join(p for p in preamble if p)
        + "\n\n"
        + "\n\n".join(body)
        + "\n\n\\end{document}\n"
    )
