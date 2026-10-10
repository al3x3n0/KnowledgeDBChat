"""The name a download is given, when the name comes from data.

Thirteen routes wrote a title or an upload's own filename into
`Content-Disposition` with an f-string. A header is latin-1: a name in any
other script raises when the response is built, and the download is a 500.
"""

from __future__ import annotations

import ast
from pathlib import Path
from urllib.parse import unquote

import pytest
from starlette.responses import Response, StreamingResponse

from app.utils.http_headers import content_disposition

APP = Path(__file__).resolve().parents[1] / "app"

AWKWARD = [
    "report.pptx",
    "отчёт за квартал.pdf",
    "résumé final.pdf",
    "报告.docx",
    'he said "no".txt',
    "line\r\nSet-Cookie: stolen=1.txt",
    "../../etc/passwd",
    "back\\slash.txt",
    "",
    None,
    "   ",
]


@pytest.mark.parametrize("name", AWKWARD)
def test_any_name_makes_a_header_a_response_accepts(name):
    value = content_disposition(name)

    value.encode("latin-1")
    assert "\r" not in value and "\n" not in value
    Response(content=b"x", headers={"Content-Disposition": value})
    StreamingResponse(iter([b"x"]), headers={"Content-Disposition": value})


def test_the_hand_built_form_is_what_fails():
    # The control: this is the line the routes had.
    with pytest.raises(UnicodeEncodeError):
        StreamingResponse(
            iter([b"x"]),
            headers={"Content-Disposition": 'attachment; filename="отчёт.pdf"'},
        )


def test_a_plain_name_is_sent_plainly():
    assert content_disposition("report.pptx") == 'attachment; filename="report.pptx"'
    assert content_disposition("a.png", inline=True) == 'inline; filename="a.png"'


def test_a_name_in_another_script_arrives_whole():
    value = content_disposition("отчёт за квартал.pdf")

    fallback, _, encoded = value.partition("; filename*=UTF-8''")
    assert unquote(encoded) == "отчёт за квартал.pdf"
    # Every client gets something with the right extension.
    assert fallback.startswith('attachment; filename="') and fallback.endswith('.pdf"')


def test_accents_fall_back_to_the_letters_under_them():
    assert content_disposition("résumé.pdf").startswith(
        'attachment; filename="resume.pdf"'
    )


def test_a_quote_cannot_end_the_value_and_a_newline_cannot_start_a_header():
    quoted = content_disposition('he said "no".txt')
    broken = content_disposition("a\r\nSet-Cookie: x=1.txt")

    assert quoted.count('"') == 2
    assert "\r" not in broken and "\n" not in broken


def test_a_path_is_not_a_filename():
    value = content_disposition("../../etc/passwd")

    assert "/" not in value and "\\" not in content_disposition("back\\slash.txt")


def test_no_name_is_still_a_name():
    for nothing in ("", None, "   ", "..."):
        assert content_disposition(nothing) == 'attachment; filename="download"'


def test_no_route_writes_the_header_by_hand():
    by_hand, uses = [], 0
    for path in sorted(APP.rglob("*.py")):
        if path.name == "http_headers.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == (
                "content_disposition"
            ):
                uses += 1
            # A formatted string that is itself a disposition value.
            if isinstance(node, ast.JoinedStr):
                literal = "".join(
                    part.value
                    for part in node.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
                if "filename=" in literal and (
                    "attachment" in literal or "inline" in literal
                ):
                    by_hand.append(f"{path.relative_to(APP.parent)}:{node.lineno}")

    assert uses >= 16
    assert by_hand == [], f"use app.utils.http_headers.content_disposition: {by_hand}"
