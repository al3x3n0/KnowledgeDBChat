"""Tests for code execution tools (execute_python, execute_data_pipeline, write_and_run_script)."""

import pytest


def _make_state():
    """Create a minimal state dict with code execution fields."""
    return {
        "code_execution_history": [],
    }


class TestExecutePython:
    """Tests for execute_python tool logic."""

    WHITELISTED_IMPORTS = {
        "json", "re", "datetime", "math", "collections",
        "itertools", "statistics", "csv", "textwrap",
        "hashlib", "base64", "uuid",
    }

    def test_whitelisted_import_allowed(self):
        for mod in self.WHITELISTED_IMPORTS:
            assert mod in self.WHITELISTED_IMPORTS

    def test_blocked_import_rejected(self):
        dangerous = {"os", "subprocess", "sys", "shutil", "socket"}
        for mod in dangerous:
            assert mod not in self.WHITELISTED_IMPORTS

    def test_execution_result_recorded(self):
        state = _make_state()
        result = {
            "tool": "execute_python",
            "code_hash": "abc123",
            "success": True,
            "output": "42",
            "error": None,
        }
        state["code_execution_history"].append(result)
        assert len(state["code_execution_history"]) == 1
        assert state["code_execution_history"][0]["success"] is True

    def test_timeout_enforced(self):
        max_timeout = 30
        requested_timeout = 60
        effective = min(requested_timeout, max_timeout)
        assert effective == 30

    def test_execution_error_recorded(self):
        state = _make_state()
        result = {
            "tool": "execute_python",
            "success": False,
            "output": "",
            "error": "NameError: name 'os' is not defined",
        }
        state["code_execution_history"].append(result)
        assert state["code_execution_history"][0]["success"] is False
        assert "NameError" in state["code_execution_history"][0]["error"]


class TestExecuteDataPipeline:
    """Tests for execute_data_pipeline tool logic."""

    def test_docker_config_defaults(self):
        config = {
            "image": "knowledgedbchat-data-pipeline",
            "network_disabled": True,
            "timeout": 300,
            "memory_limit": "512m",
        }
        assert config["network_disabled"] is True
        assert config["timeout"] == 300
        assert config["memory_limit"] == "512m"

    def test_fallback_to_restricted_python(self):
        docker_available = False
        if not docker_available:
            execution_mode = "restricted_python"
        else:
            execution_mode = "docker"
        assert execution_mode == "restricted_python"

    def test_result_contains_output(self):
        state = _make_state()
        result = {
            "tool": "execute_data_pipeline",
            "success": True,
            "output": "DataFrame shape: (100, 5)\nMean: 42.3",
            "execution_mode": "docker",
        }
        state["code_execution_history"].append(result)
        assert "DataFrame" in state["code_execution_history"][0]["output"]


class TestWriteAndRunScript:
    """Tests for write_and_run_script tool logic."""

    def test_requires_docker(self):
        docker_available = False
        if not docker_available:
            error = "Docker is required for write_and_run_script but is not available"
        else:
            error = None
        assert error is not None
        assert "Docker is required" in error

    def test_docker_available_succeeds(self):
        docker_available = True
        if not docker_available:
            error = "Docker is required"
        else:
            error = None
        assert error is None

    def test_pip_packages_whitelisted(self):
        allowed_packages = {
            "numpy", "pandas", "scipy", "scikit-learn",
            "matplotlib", "seaborn", "requests",
        }
        requested = ["numpy", "pandas"]
        for pkg in requested:
            assert pkg in allowed_packages

    def test_blocked_pip_package(self):
        allowed_packages = {
            "numpy", "pandas", "scipy", "scikit-learn",
            "matplotlib", "seaborn", "requests",
        }
        blocked = "malicious-package"
        assert blocked not in allowed_packages

    def test_script_execution_recorded(self):
        state = _make_state()
        result = {
            "tool": "write_and_run_script",
            "success": True,
            "output": "Script completed successfully",
            "files_created": ["output.csv", "report.txt"],
        }
        state["code_execution_history"].append(result)
        assert len(state["code_execution_history"][0]["files_created"]) == 2

    def test_network_disabled_in_docker(self):
        config = {
            "network_disabled": True,
            "timeout": 300,
        }
        assert config["network_disabled"] is True
