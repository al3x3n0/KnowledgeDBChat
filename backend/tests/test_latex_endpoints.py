import pytest
from unittest.mock import AsyncMock, patch

from .factories import create_test_document_source, create_test_document


def test_latex_status(client, auth_headers):
    resp = client.get("/api/v1/latex/status", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "enabled" in data
    assert "available_engines" in data


def test_latex_compile_disabled_returns_503(client, admin_headers):
    resp = client.post(
        "/api/v1/latex/compile",
        json={"tex_source": "\\documentclass{article}\\begin{document}hi\\end{document}"},
        headers=admin_headers,
    )
    assert resp.status_code == 503


def test_latex_apply_unified_diff_applies_patch(client, auth_headers):
    # Create a project
    create = client.post(
        "/api/v1/latex/projects",
        json={
            "title": "Test Paper",
            "tex_source": "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n",
        },
        headers=auth_headers,
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    # Apply a minimal unified diff against paper.tex
    diff = (
        "--- a/paper.tex\n"
        "+++ b/paper.tex\n"
        "@@ -2,3 +2,3 @@\n"
        " \\begin{document}\n"
        "-Hello\n"
        "+Hello world\n"
        " \\end{document}\n"
    )
    resp = client.post(
        f"/api/v1/latex/projects/{project_id}/apply-unified-diff",
        json={"diff_unified": diff},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] is True
    assert "Hello world" in data["tex_source"]

    # Ensure project source is updated
    getp = client.get(f"/api/v1/latex/projects/{project_id}", headers=auth_headers)
    assert getp.status_code == 200
    assert "Hello world" in getp.json()["tex_source"]


def test_latex_apply_unified_diff_base_sha_mismatch_returns_409(client, auth_headers):
    create = client.post(
        "/api/v1/latex/projects",
        json={
            "title": "Test Paper",
            "tex_source": "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n",
        },
        headers=auth_headers,
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    diff = (
        "--- a/paper.tex\n"
        "+++ b/paper.tex\n"
        "@@ -2,3 +2,3 @@\n"
        " \\begin{document}\n"
        "-Hello\n"
        "+Hello world\n"
        " \\end{document}\n"
    )
    resp = client.post(
        f"/api/v1/latex/projects/{project_id}/apply-unified-diff",
        json={"diff_unified": diff, "expected_base_sha256": "deadbeef"},
        headers=auth_headers,
    )
    assert resp.status_code == 409


def test_latex_apply_unified_diff_multiple_files_rejected(client, auth_headers):
    create = client.post(
        "/api/v1/latex/projects",
        json={
            "title": "Test Paper",
            "tex_source": "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n",
        },
        headers=auth_headers,
    )
    assert create.status_code == 201
    project_id = create.json()["id"]

    diff = (
        "--- a/paper.tex\n"
        "+++ b/paper.tex\n"
        "@@ -1,1 +1,1 @@\n"
        "-\\documentclass{article}\n"
        "+\\documentclass{article}\n"
        "--- a/other.tex\n"
        "+++ b/other.tex\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    resp = client.post(
        f"/api/v1/latex/projects/{project_id}/apply-unified-diff",
        json={"diff_unified": diff},
        headers=auth_headers,
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_latex_copilot_section_returns_snippet(client, auth_headers, db_session):
    source = await create_test_document_source(db_session, name="Copilot Source", source_type="web")
    doc = await create_test_document(
        db_session,
        source,
        title="RAG Notes",
        content="This document describes reranking with cross-encoders for RAG pipelines.",
    )

    mock_json = (
        '{'
        '"tex_snippet":"\\\\section{Introduction}\\\\nThis is a test snippet.",'
        '"bibtex":"\\\\begin{thebibliography}{9}\\\\n\\\\bibitem{S1} RAG Notes\\\\n\\\\end{thebibliography}"'
        '}'
    )

    with patch("app.services.llm_service.LLMService.generate_response", new=AsyncMock(return_value=mock_json)):
        resp = client.post(
            "/api/v1/latex/copilot/section",
            json={
                "prompt": "Write an introduction grounded on the source.",
                "document_ids": [str(doc.id)],
                "use_vector_snippets": False,
                "max_sources": 1,
            },
            headers=auth_headers,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "tex_snippet" in data
    assert data["tex_snippet"].startswith("\\section")
    assert "bibtex" in data
    assert "thebibliography" in data["bibtex"]
