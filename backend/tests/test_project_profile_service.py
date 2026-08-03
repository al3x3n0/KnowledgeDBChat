"""Tests for project profile helper functions."""

from app.services.project_profile_service import (
    format_project_profile_for_prompt,
    infer_project_profile_from_paths,
)


def test_infer_project_profile_from_paths_detects_stack_and_commands():
    profile = infer_project_profile_from_paths(
        [
            "backend/pyproject.toml",
            "backend/app/main.py",
            "backend/tests/test_executor.py",
            "frontend/package.json",
            "frontend/src/App.tsx",
            "frontend/src/__tests__/App.test.tsx",
            "Makefile",
        ]
    )

    assert profile["sampled_files"] == 7
    assert "python" in profile["detected_stack"]
    assert "typescript" in profile["detected_stack"]
    assert any(
        cmd.startswith("cd backend &&") or cmd.startswith("python -m pytest")
        for cmd in profile["command_groups"]["test"]
    )
    assert "Makefile" in profile["marker_files"]


def test_format_project_profile_for_prompt_renders_expected_sections():
    text = format_project_profile_for_prompt(
        {
            "project_profile": {
                "source_name": "KnowledgeDBChat",
                "source_type": "repository",
                "detected_stack": ["python", "typescript"],
                "suggested_commands": ["python -m pytest -q", "npm test"],
                "command_groups": {
                    "install": ["poetry install"],
                    "test": ["python -m pytest -q"],
                    "test_fallback": ["pytest -q"],
                },
                "marker_files": ["backend/pyproject.toml", "frontend/package.json"],
                "bootstrap_notes": ["Install dependencies first."],
                "test_paths": ["backend/tests/test_executor.py"],
            }
        }
    )

    assert "PROJECT PROFILE:" in text
    assert "KnowledgeDBChat" in text
    assert "Detected stack: python, typescript" in text
    assert "Bootstrap: poetry install" in text
    assert "Preferred verification: python -m pytest -q" in text
    assert "backend/pyproject.toml" in text
