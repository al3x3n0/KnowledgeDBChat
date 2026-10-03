"""A tool's governance label agrees with what its handler does.

A spec that says nothing is classified ``effects: read`` and ``network:
none`` -- the safe-sounding defaults -- so a tool that writes or reaches the
internet but was never labelled looks harmless in the policy UI. Eight did:
tools that queue ingestion or summarisation, persist a report, or search
arXiv. This reads each handler's source for the marks of a write or of
egress, and fails on a read-only label beside them.

It is a heuristic over source text, so a false positive is listed in
NOT_A_WRITE with the reason rather than worked around.
"""

import ast
import re
from pathlib import Path

import pytest

from app.agent_core.tool_specs import all_specs

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"
HANDLER_MODULES = ("services/agent_tool_dispatch.py", "services/agent_service.py")

WRITES = re.compile(
    r"\.commit\(|\bdb\.add\(|\.delete\(|\.delay\(|apply_async|write_text\(|write_bytes\("
)
EGRESS = re.compile(
    r"httpx|aiohttp|urlopen|WebScraperService|ArxivSearchService|_arxiv_search\("
)

#: Matched by the pattern, and not what it looks like.
NOT_A_WRITE = {}


def _handler_sources():
    sources = {}
    for rel in HANDLER_MODULES:
        text = (APP / rel).read_text()
        tree = ast.parse(text)
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, []).append(
                    ast.get_source_segment(text, node)
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(value, ast.Name)
                        and value.id in functions
                    ):
                        sources.setdefault(key.value, []).extend(functions[value.id])
        for name, bodies in functions.items():
            if name.startswith("_tool_"):
                sources.setdefault(name[len("_tool_") :], []).extend(bodies)
    return {name: "\n".join(bodies) for name, bodies in sources.items()}


def test_the_scan_sees_the_handlers():
    sources = _handler_sources()
    assert len(sources) > 150, "too few handlers found; this guard would pass vacuously"


def test_no_tool_that_writes_is_labelled_read():
    sources = _handler_sources()
    mislabelled = [
        f"{spec.name}: {sorted(set(WRITES.findall(sources[spec.name])))}"
        for spec in all_specs()
        if spec.effects == "read"
        and spec.name in sources
        and spec.name not in NOT_A_WRITE
        and WRITES.search(sources[spec.name])
    ]
    assert (
        not mislabelled
    ), "Labelled effects=read, but the handler writes:\n" + "\n".join(mislabelled)


def test_no_tool_that_reaches_the_internet_is_labelled_offline():
    sources = _handler_sources()
    mislabelled = [
        spec.name
        for spec in all_specs()
        if spec.network == "none"
        and spec.name in sources
        and EGRESS.search(sources[spec.name])
    ]
    assert (
        not mislabelled
    ), "Labelled network=none, but the handler goes out:\n" + "\n".join(mislabelled)
