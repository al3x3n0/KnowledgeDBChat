"""Tests for the shared runner path and LaTeX helpers.

safe_relpath is a traversal guard applied to model-supplied paths, so the
adversarial cases matter more than the happy ones.
"""

from app.services.agent_artifact_paths import (
    MAX_RELPATH_LENGTH,
    insert_before_end_document,
    safe_relpath,
)


def test_keeps_an_ordinary_relative_path():
    assert safe_relpath("src/app/main.py") == "src/app/main.py"


def test_strips_leading_slashes_so_a_path_cannot_be_absolute():
    assert safe_relpath("/etc/passwd") == "etc/passwd"
    assert safe_relpath("///etc/passwd") == "etc/passwd"


def test_drops_parent_segments_rather_than_resolving_them():
    # Dropping rather than resolving means a path that tried to escape loses
    # the attempt instead of landing somewhere else on disk.
    assert safe_relpath("../../etc/passwd") == "etc/passwd"
    assert safe_relpath("src/../../../secrets.env") == "src/secrets.env"
    assert safe_relpath("..") == ""


def test_normalizes_windows_separators_and_dot_prefixes():
    assert safe_relpath("src\\app\\main.py") == "src/app/main.py"
    assert safe_relpath("./src/main.py") == "src/main.py"
    assert safe_relpath("././src/main.py") == "src/main.py"


def test_collapses_empty_segments():
    assert safe_relpath("src//app///main.py") == "src/app/main.py"


def test_returns_empty_for_nothing_usable():
    for value in ("", "   ", "/", "..", "./", None):
        assert safe_relpath(value) == ""


def test_bounds_the_length():
    assert len(safe_relpath("a/" * 500)) <= MAX_RELPATH_LENGTH


def test_inserts_before_the_end_document_marker():
    source = "\\documentclass{article}\n\\begin{document}\nBody\n\\end{document}\n"
    result = insert_before_end_document(source, "\\section{Added}")

    assert result.index("\\section{Added}") < result.index("\\end{document}")
    assert result.count("\\end{document}") == 1
    assert "Body" in result


def test_appends_when_the_document_has_no_end_marker():
    # A fragment must still receive the content rather than silently dropping it.
    result = insert_before_end_document("Just a fragment", "\\section{Added}")
    assert result.strip().endswith("\\section{Added}")
    assert "Just a fragment" in result


def test_uses_the_last_end_document_when_several_appear():
    source = "\\end{document} in a verbatim block\ntext\n\\end{document}\n"
    result = insert_before_end_document(source, "ADDED")
    assert result.rindex("ADDED") < result.rindex("\\end{document}")


def test_tolerates_empty_input():
    assert insert_before_end_document("", "ADDED").strip() == "ADDED"
    assert "ADDED" in insert_before_end_document(None, "ADDED")
