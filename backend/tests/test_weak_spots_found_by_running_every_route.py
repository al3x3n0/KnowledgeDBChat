"""The defects the first walk over every route found, each pinned."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from uuid import uuid4

import pytest

BACKEND = Path(__file__).resolve().parents[1]


# --- a model nobody registered ---------------------------------------------


def test_every_model_module_is_registered_with_the_package():
    # `import app.models` is how the schema drift check, and every test
    # database, learn which tables exist. The campaign models were defined
    # and never imported there: the drift check had never looked at their
    # tables, and no test could store a campaign.
    registered = set(
        re.findall(
            r"^from \.(\w+) import",
            (BACKEND / "app/models/__init__.py").read_text(),
            re.M,
        )
    )
    unregistered = []
    for path in sorted((BACKEND / "app/models").glob("*.py")):
        if path.name == "__init__.py":
            continue
        tables = [
            node.name
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ClassDef)
            and any(isinstance(b, ast.Name) and b.id == "Base" for b in node.bases)
        ]
        if tables and path.stem not in registered:
            unregistered.append(f"{path.name}: {', '.join(tables)}")

    assert unregistered == []


@pytest.mark.asyncio
async def test_a_campaign_can_be_stored_in_a_test_database(db_session, test_user):
    from app.models.research_campaign import ResearchCampaign

    campaign = ResearchCampaign(
        id=uuid4(), user_id=test_user.id, name="Prefetchers", goal="Find what helps"
    )
    db_session.add(campaign)
    await db_session.commit()

    assert (await db_session.get(ResearchCampaign, campaign.id)).name == "Prefetchers"


# --- the workflow editor's tool list ----------------------------------------


def test_a_tool_parameter_declared_with_several_types_has_one_for_the_editor():
    # JSON Schema allows `"type": ["array", "string"]`. The response model
    # wants a string, so one such built-in tool made the whole list a 500.
    from app.api.endpoints.workflows import _parameter_type

    assert _parameter_type("string") == "string"
    assert _parameter_type(["array", "string"]) == "array"
    assert _parameter_type(["null", "integer"]) == "integer"
    assert _parameter_type(["null"]) == "any"
    assert _parameter_type(None) == "any"
    assert _parameter_type([]) == "any"


def test_every_built_in_tool_can_be_listed_for_the_editor():
    from app.api.endpoints.workflows import _parameter_type
    from app.schemas.workflow import ToolParameterDetail
    from app.services.agent_tools import AGENT_TOOLS

    several = 0
    for tool in AGENT_TOOLS:
        schema = tool.get("parameters") or tool.get("input_schema") or {}
        for name, prop in (schema.get("properties") or {}).items():
            declared = prop.get("type", "any")
            several += isinstance(declared, list)
            ToolParameterDetail(name=name, type=_parameter_type(declared))

    assert several > 0  # the case this is about really occurs


# --- ids that are not ids ----------------------------------------------------


def test_knowledge_graph_routes_take_ids_as_ids():
    # Typed `str`, a malformed id reached the database as text and came back
    # as a 500; typed `UUID`, the framework answers 422 before the handler.
    tree = ast.parse((BACKEND / "app/api/endpoints/knowledge_graph.py").read_text())
    loose = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        in_path = {
            name
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call) and decorator.args
            for name in re.findall(
                r"\{(\w+)\}", str(getattr(decorator.args[0], "value", ""))
            )
        }
        for argument in node.args.args:
            if (
                argument.arg in in_path
                and argument.arg.endswith("_id")
                and ast.unparse(argument.annotation) != "UUID"
            ):
                loose.append(f"{node.name}({argument.arg})")

    assert loose == []


@pytest.mark.asyncio
async def test_deleting_an_entity_takes_an_id_in_either_form(db_session):
    from app.services.knowledge_graph_service import KnowledgeGraphService

    service = KnowledgeGraphService()
    for missing in (uuid4(), str(uuid4())):
        with pytest.raises(ValueError, match="Entity not found"):
            await service.delete_entity(db_session, missing)
