"""What the diagram service generates has to render.

The generators pasted a model's labels and identifiers into Mermaid and DOT
as they came, and nothing tested the result. Run through the real renderers,
10 of 16 ordinary diagrams failed: a label containing a parenthesis broke
every flowchart, an id with a space or a hyphen (or the word `end`) broke the
graph, a "dotted" edge was written as `...`, and a quote broke DOT.

`CASES` is that list. Two kinds of check:

* against real renderers -- Graphviz's `dot` wherever it is installed, and
  the project's Mermaid renderer when `MERMAID_RENDERER_URL` names one
  (`docker run -p 18017:8002 knowledge_db_mermaid_renderer`). The escaping
  forms were chosen by trying them there, not by reading the grammar;
* on the generated text, which runs everywhere and pins the forms that were
  verified.
"""

import os
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest

from app.services.diagram_service import (
    DiagramService,
    dot_string,
    mermaid_label,
    mermaid_name,
    mermaid_text,
)

pytestmark = pytest.mark.unit

AWKWARD = 'parse(x) [fast] "quoted" | a & b'
SHAPES = [
    "rectangle",
    "rounded",
    "stadium",
    "diamond",
    "hexagon",
    "parallelogram",
    "cylinder",
    "circle",
]

MERMAID_CASES = {
    "plain flowchart": (
        "flowchart",
        {
            "nodes": [{"id": "a", "label": "Start"}, {"id": "b", "label": "End"}],
            "edges": [{"source": "a", "target": "b"}],
        },
    ),
    "flowchart with an awkward label": (
        "flowchart",
        {
            "nodes": [{"id": "a", "label": AWKWARD}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b", "label": AWKWARD}],
        },
    ),
    "flowchart with a dotted edge": (
        "flowchart",
        {
            "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "edges": [{"source": "a", "target": "b", "style": "dotted"}],
        },
    ),
    "flowchart ids with a space and a hyphen": (
        "flowchart",
        {
            "nodes": [
                {"id": "load data", "label": "A"},
                {"id": "step-2", "label": "B"},
            ],
            "edges": [{"source": "load data", "target": "step-2"}],
        },
    ),
    "flowchart with a node called end": (
        "flowchart",
        {
            "nodes": [{"id": "start", "label": "A"}, {"id": "end", "label": "B"}],
            "edges": [{"source": "start", "target": "end"}],
        },
    ),
    "flowchart in every shape": (
        "flowchart",
        {
            "nodes": [
                {"id": f"n{i}", "label": "L (x)", "shape": shape}
                for i, shape in enumerate(SHAPES)
            ],
            "edges": [],
        },
    ),
    "sequence with punctuation": (
        "sequence",
        {
            "participants": [{"id": "api", "label": "API (v2)"}, "db"],
            "messages": [{"from": "api", "to": "db", "text": "SELECT * ; -- x: y"}],
        },
    ),
    "class with spaces in its names": (
        "class",
        {
            "classes": [
                {
                    "name": "Cache Line",
                    "attributes": [{"name": "size in bytes", "type": "unsigned int"}],
                    "methods": [{"name": "get", "params": ["the key"]}],
                }
            ],
            "relationships": [
                {"source": "Cache Line", "target": "Cache Line", "label": "next; prev"}
            ],
        },
    ),
    "er with a punctuated label": (
        "er",
        {
            "entities": [
                {
                    "name": "USER",
                    "attributes": [{"name": "id", "type": "int", "key": "pk"}],
                },
                {"name": "agent job", "attributes": []},
            ],
            "relationships": [
                {
                    "source": "USER",
                    "target": "agent job",
                    "cardinality": "||--o{",
                    "label": "runs (many)",
                },
                {"source": "USER", "target": "USER", "cardinality": "nonsense"},
            ],
        },
    ),
    "state with spaces": (
        "state",
        {
            "states": [
                {"name": "Idle"},
                {"name": "In progress", "description": "doing (work); now"},
            ],
            "transitions": [
                {"from": "[*]", "to": "Idle"},
                {"from": "Idle", "to": "In progress", "label": "start: go"},
            ],
        },
    ),
    "gantt with a colon in a name": (
        "gantt",
        {
            "sections": [
                {
                    "name": "Build: phase 1",
                    "tasks": [
                        {
                            "name": "Compile: step 1, fast",
                            "start": "2026-10-01",
                            "duration": "3d",
                        }
                    ],
                }
            ]
        },
    ),
    "pie with a quote and a zero": (
        "pie",
        {
            "slices": [
                {"label": 'A "b"', "value": 3},
                {"label": "C", "value": 1},
                {"label": "nothing", "value": 0},
                {"label": "bad", "value": "n/a"},
            ]
        },
    ),
    "mindmap with brackets": (
        "mindmap",
        {
            "root": "Caches (L1/L2)",
            "branches": [{"name": "Prefetch [stride]", "children": ["a(b)", "plain"]}],
        },
    ),
}

DOT_CASES = {
    "plain": {
        "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        "edges": [{"source": "a", "target": "b"}],
    },
    "quoted labels": {
        "nodes": [{"id": "a", "label": 'say "hi" \\ there'}, {"id": "b", "label": "B"}],
        "edges": [{"source": "a", "target": "b", "label": 'x "y"'}],
    },
    "ids with a hyphen and a space": {
        "nodes": [{"id": "load-data", "label": "A"}, "step 2"],
        "edges": [{"source": "load-data", "target": "step 2"}],
    },
}


def _mermaid(case):
    kind, data = MERMAID_CASES[case]
    return DiagramService().create_mermaid_diagram(kind, data, {"render": False})[
        "code"
    ]


def _dot(case, graph_type="digraph"):
    return DiagramService().create_graphviz_diagram(
        graph_type, DOT_CASES[case], {"render": False}
    )["code"]


# --------------------------------------------------------------------------
# real renderers
# --------------------------------------------------------------------------

RENDERER = os.environ.get("MERMAID_RENDERER_URL", "").rstrip("/")


@pytest.mark.skipif(not RENDERER, reason="MERMAID_RENDERER_URL is not set")
@pytest.mark.parametrize("case", list(MERMAID_CASES))
def test_mermaid_renders(case):
    request = urllib.request.Request(
        f"{RENDERER}/svg", data=_mermaid(case).encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read()
    except urllib.error.HTTPError as refused:
        pytest.fail(f"{case}: {refused.read().decode(errors='replace')[:300]}")
    assert b"<svg" in body and b"Syntax error" not in body


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz is not installed")
@pytest.mark.parametrize("case", list(DOT_CASES))
@pytest.mark.parametrize("graph_type", ["digraph", "graph"])
def test_graphviz_renders(case, graph_type):
    done = subprocess.run(
        ["dot", "-Tsvg"],
        input=_dot(case, graph_type).encode(),
        capture_output=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr.decode()[:300]
    assert b"<svg" in done.stdout


# --------------------------------------------------------------------------
# the forms that were verified, pinned on the text
# --------------------------------------------------------------------------


def test_a_flowchart_label_is_a_quoted_string_in_every_shape():
    code = _mermaid("flowchart in every shape")

    for wrapped in (
        'n0["L (x)"]',
        'n1("L (x)")',
        'n2(["L (x)"])',
        'n3{"L (x)"}',
        'n4{{"L (x)"}}',
        'n5[/"L (x)"/]',
        'n6[("L (x)")]',
        'n7(("L (x)"))',
    ):
        assert wrapped in code


def test_a_quote_inside_a_label_is_an_entity():
    assert mermaid_label('say "hi" #1') == '"say #quot;hi#quot; #35;1"'
    assert '|"parse(x) [fast] #quot;quoted#quot; | a & b"|' in _mermaid(
        "flowchart with an awkward label"
    )


def test_an_unsafe_id_is_replaced_the_same_way_everywhere():
    code = _mermaid("flowchart ids with a space and a hyphen")

    assert "load data" not in code and "step-2" not in code
    lines = [line.strip() for line in code.splitlines()]
    assert 'n1["A"]' in lines and 'n2["B"]' in lines
    assert "n1 --> n2" in lines


def test_a_reserved_word_is_not_used_as_an_id():
    lines = [
        line.strip()
        for line in _mermaid("flowchart with a node called end").splitlines()
    ]

    assert not any(line.startswith("end[") for line in lines)
    assert any(line.startswith("start[") for line in lines)


def test_a_generated_id_never_collides_with_one_the_caller_chose():
    code = DiagramService().create_mermaid_diagram(
        "flowchart",
        {
            "nodes": [{"id": "n1", "label": "mine"}, {"id": "a b", "label": "theirs"}],
            "edges": [{"source": "n1", "target": "a b"}],
        },
        {"render": False},
    )["code"]

    assert 'n1["mine"]' in code
    assert 'n2["theirs"]' in code and "n1 --> n2" in code


def test_dotted_and_dashed_edges_are_valid_links():
    code = _mermaid("flowchart with a dotted edge")

    assert "a -.-> b" in code and "..." not in code


def test_a_direction_that_is_not_one_is_not_written():
    code = DiagramService().create_mermaid_diagram(
        "flowchart",
        {"nodes": [{"id": "a", "label": "A"}], "edges": []},
        {"render": False, "direction": "LR\\n    click a call alert()"},
    )["code"]

    assert code.splitlines()[0].strip() == "flowchart TD"
    assert "click" not in code


def test_free_text_cannot_end_its_statement():
    assert mermaid_text("SELECT * ; -- x") == "SELECT * #59; -- x"
    assert mermaid_text("a\nb") == "a b"


def test_names_are_made_of_word_characters():
    assert mermaid_name("Cache Line") == "Cache_Line"
    assert mermaid_name("2nd stage") == "_2nd_stage"
    assert mermaid_name("  ", "Class") == "Class"


def test_a_pie_drops_slices_it_cannot_draw():
    code = _mermaid("pie with a quote and a zero")

    assert "\"A 'b'\" : 3" in code and '"C" : 1' in code
    assert "nothing" not in code and "bad" not in code


def test_a_dot_string_escapes_quotes_and_backslashes():
    assert dot_string('say "hi"') == '"say \\"hi\\""'
    assert dot_string("a\\b") == '"a\\\\b"'
    assert '"load-data" -> "step 2";' in _dot("ids with a hyphen and a space")


# --------------------------------------------------------------------------
# what reaches a command line is chosen from a list
# --------------------------------------------------------------------------


def test_the_layout_program_and_format_are_chosen_from_a_list(monkeypatch):
    """`layout` was the program that was run."""
    ran = []

    real_run = subprocess.run

    def run(cmd, **kwargs):
        if "-o" not in cmd:  # the service probing for its programs
            return real_run(cmd, **kwargs)
        ran.append(cmd)
        with open(cmd[cmd.index("-o") + 1], "wb") as handle:
            handle.write(b"<svg/>")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)

    DiagramService()._render_graphviz(
        "digraph G { a -> b }",
        {"layout": "/bin/sh", "output_format": "svg/../../x"},
    )
    DiagramService()._render_graphviz(
        "digraph G { a -> b }", {"layout": "neato", "output_format": "png"}
    )

    assert ran[0][0] == "dot" and ran[0][1] == "-Tsvg"
    assert ran[0][3].endswith(".svg") and ".." not in ran[0][3]
    assert ran[1][0] == "neato" and ran[1][1] == "-Tpng"
