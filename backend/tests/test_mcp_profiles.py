import json
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agent_core import research_programs
from app.agent_core.research_programs import ResearchProgram
from app.agent_core.tool_specs.spec import ToolSpec
from app.api.endpoints import api_keys, mcp_config, research_notes
from app.core.database import get_db
from app.mcp.server import mcp_router
from app.mcp.tools import chat, search
from app.models.document import Document, DocumentSource
from app.models.research_note import ResearchNote
from app.models.retrieval_trace import RetrievalTrace
from app.models.tool_policy import ToolPolicy
from app.services.api_key_service import api_key_service
from app.services.auth_service import get_current_user
from app.services.rag_retrieval import Retrieved

pytestmark = pytest.mark.unit


@pytest.fixture
async def client(db_session, test_user, monkeypatch):
    spec = ToolSpec("measure", "Measure a sample", {"type": "object"})
    monkeypatch.setattr(
        research_programs,
        "installed",
        lambda: [
            ResearchProgram("chosen", version="1", tool_specs=(spec,)),
            ResearchProgram("other"),
        ],
    )
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v1")
    app.include_router(mcp_config.router, prefix="/api/v1/mcp-config")
    app.include_router(api_keys.router, prefix="/api/v1/api-keys")
    app.include_router(research_notes.router, prefix="/api/v1/research-notes")
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_user] = lambda: test_user
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def test_context_pack_becomes_a_cited_mcp_note(
    client, db_session, test_user, admin_user
):
    source = DocumentSource(name="Lab records", source_type="file", config={})
    db_session.add(source)
    await db_session.flush()
    document = Document(
        title="Compiler benchmark report",
        source_id=source.id,
        source_identifier="compiler-benchmark",
        content_hash="report",
    )
    db_session.add(document)
    await db_session.flush()
    chunk_id = str(uuid4())
    evidence = {
        "document_id": str(document.id),
        "chunk_id": chunk_id,
        "text": "The benchmark improved.",
    }
    pack = {
        "kg_context": "Compiler benchmark results",
        "entities": [{"evidence": [evidence]}],
        "relationships": [{**evidence, "evidence": evidence["text"]}],
    }
    owned = RetrievalTrace(
        user_id=test_user.id, query="What changed?", trace={"kg_context_pack": pack}
    )
    other = RetrievalTrace(
        user_id=admin_user.id, query="Private question", trace={"kg_context_pack": pack}
    )
    empty = RetrievalTrace(user_id=test_user.id, query="Empty", trace={})
    db_session.add_all([owned, other, empty])
    await db_session.commit()

    path = "/api/v1/research-notes/context-packs"
    assert (await client.post(f"{path}/{other.id}")).status_code == 404
    assert (await client.post(f"{path}/{empty.id}")).status_code == 422
    saved = await client.post(f"{path}/{owned.id}")
    assert saved.status_code == 200
    note = saved.json()
    assert note["source_document_ids"] == [str(document.id)]
    assert note["attribution"]["saved_from_context_pack"] == {
        "trace_id": str(owned.id),
        "citations": [
            {
                "key": "S1",
                "doc_id": str(document.id),
                "chunk_id": chunk_id,
                "excerpt": evidence["text"],
            }
        ],
    }
    assert "Compiler benchmark report" in note["content_markdown"]
    assert (
        f"The benchmark improved. [[S1]] (chunk `{chunk_id}`)"
        in note["content_markdown"]
    )
    edited = await client.patch(
        f"/api/v1/research-notes/{note['id']}", json={"title": "Reviewed results"}
    )
    assert edited.status_code == 200
    repeated = (await client.post(f"{path}/{owned.id}")).json()
    assert repeated["id"] == note["id"]
    assert repeated["title"] == "Reviewed results"

    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={"name": "Compiler context", "note_ids": [note["id"]]},
    )
    assert created.status_code == 201
    profile = created.json()
    general, _ = await api_key_service.create_api_key(
        db_session, test_user.id, "Legacy"
    )
    keys = (await client.get("/api/v1/api-keys/")).json()["api_keys"]
    assert {entry["id"]: entry["is_mcp_profile"] for entry in keys} == {
        profile["id"]: True,
        str(general.id): False,
    }
    response = await client.post(
        "/api/v1/mcp",
        headers={"Authorization": f"Bearer {profile['api_key']}"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {
                "uri": f"kdbc://research-notes/{note['id']}",
            },
        },
    )
    assert response.json()["result"]["contents"][0]["text"] == note["content_markdown"]


async def test_profile_key_connects_and_exposes_only_selected_context(
    client, db_session
):
    options = await client.get("/api/v1/mcp-config/profile-options")
    assert [program["name"] for program in options.json()["programs"]] == [
        "chosen",
        "other",
    ]

    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={"name": "Lab context", "programs": ["chosen"], "tools": ["search"]},
    )
    assert created.status_code == 201
    assert created.headers["Cache-Control"] == "no-store"
    key = created.json()
    assert await api_key_service.validate_api_key(db_session, key["api_key"]) is None
    headers = {
        "Authorization": f"Bearer {key['api_key']}",
        "Accept": "application/json, text/event-stream",
    }

    async def rpc(method, params=None, **extra):
        return await client.post(
            "/api/v1/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "method": method, "params": params or {}, **extra},
        )

    init = await rpc("initialize", {"protocolVersion": "2025-06-18"}, id=1)
    assert init.json()["result"]["protocolVersion"] == "2025-06-18"
    headers["MCP-Protocol-Version"] = init.json()["result"]["protocolVersion"]
    assert (await rpc("notifications/initialized")).status_code == 202
    listed = await rpc("tools/list", id=2)
    assert (await client.get("/api/v1/mcp", headers=headers)).status_code == 405
    assert {tool["name"] for tool in listed.json()["result"]["tools"]} == {
        "search",
        "get_research_programs",
    }
    context = await rpc("tools/call", {"name": "get_research_programs"}, id=3)
    content = json.loads(context.json()["result"]["content"][0]["text"])
    assert [program["name"] for program in content["programs"]] == ["chosen"]
    assert content["programs"][0]["tools"][0]["name"] == "measure"
    denied = await rpc("tools/call", {"name": "list_documents"}, id=4)
    assert denied.json()["error"]["code"] == -32602
    direct = await client.get("/api/v1/mcp/documents", headers=headers)
    assert direct.status_code == 403
    legacy = await client.post(
        "/api/v1/mcp/tools/call", headers=headers, json={"name": "list_documents"}
    )
    assert legacy.status_code == 403

    key_config = f"/api/v1/mcp-config/keys/{key['id']}"
    assert (
        await client.put(
            f"{key_config}/tools/search",
            json={"tool_name": "search", "is_enabled": False},
        )
    ).status_code == 200
    listed = await rpc("tools/list", id=5)
    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == [
        "get_research_programs"
    ]

    assert (
        await client.patch(f"{key_config}/config", json={"allowed_tools": []})
    ).status_code == 200
    assert (await rpc("tools/list", id=6)).json()["result"]["tools"] == []


@pytest.mark.parametrize(
    "selection", [{}, {"programs": ["missing"]}, {"tools": ["docker_execute"]}]
)
async def test_invalid_profiles_are_rejected(client, db_session, test_user, selection):
    response = await client.post(
        "/api/v1/mcp-config/profiles", json={"name": "Lab", **selection}
    )
    assert response.status_code == 400
    assert await api_key_service.list_api_keys(db_session, test_user.id) == []


async def test_profile_source_scope_limits_listing_fetching_and_chat(
    client, db_session, monkeypatch
):
    allowed = DocumentSource(name="Research", source_type="file", config={})
    denied = DocumentSource(name="Private", source_type="file", config={})
    db_session.add_all([allowed, denied])
    await db_session.flush()
    documents = [
        Document(
            title=source.name,
            source_id=source.id,
            source_identifier=source.name,
            content_hash=source.name,
            content=source.name,
        )
        for source in (allowed, denied)
    ]
    db_session.add_all(documents)
    await db_session.commit()

    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Scoped",
            "tools": [
                "list_sources",
                "list_documents",
                "get_document",
                "chat",
                "search",
            ],
            "source_ids": [str(allowed.id)],
        },
    )
    assert created.status_code == 201, created.text
    key = created.json()
    headers = {"Authorization": f"Bearer {key['api_key']}"}

    async def call(name, **arguments):
        response = await client.post(
            "/api/v1/mcp/tools/call",
            headers=headers,
            json={"name": name, "arguments": arguments},
        )
        assert response.status_code == 200
        return response.json()["content"][0]["text"]

    config = await client.get(f"/api/v1/mcp-config/keys/{key['id']}/config")
    assert config.json()["source_access_mode"] == "restricted"
    assert [source["source_id"] for source in config.json()["source_access"]] == [
        str(allowed.id)
    ]
    assert [source["name"] for source in (await call("list_sources"))["sources"]] == [
        "Research"
    ]
    assert [doc["title"] for doc in (await call("list_documents"))["documents"]] == [
        "Research"
    ]
    assert (await call("list_documents", source_id=str(denied.id)))["total"] == 0
    assert (await call("get_document", document_id=str(documents[0].id)))[
        "title"
    ] == "Research"
    assert (await call("get_document", document_id=str(documents[1].id)))[
        "error"
    ] == "Document not found"
    assert (await call("search", query="Private", source_ids=[str(denied.id)]))[
        "total"
    ] == 0

    async def retrieve(_question, **kwargs):
        assert kwargs["document_ids"] == [str(documents[0].id)]
        assert kwargs["include_knowledge_graph"] is False
        return Retrieved(
            search_query=_question, processed_query={}, results=[], context=""
        )

    async def search_chunks(**kwargs):
        assert kwargs["document_ids"] == [str(documents[0].id)]
        return [
            {"metadata": {"document_id": str(doc.id)}, "content": doc.content}
            for doc in documents
        ]

    async def initialize(**kwargs):
        pass

    monkeypatch.setattr(chat, "retrieve_for_answer", retrieve)
    monkeypatch.setattr(search.vector_store_service, "initialize", initialize)
    monkeypatch.setattr(search.vector_store_service, "search", search_chunks)
    result = await call("search", query="anything")
    assert [hit["content"] for hit in result["results"]] == ["Research"]
    assert (
        await call("chat", question="Private?", document_ids=[str(documents[1].id)])
    )["context_found"] is False
    assert "error" not in await call("chat", question="Research?")
    removed = await client.delete(
        f"/api/v1/mcp-config/keys/{key['id']}/sources/{allowed.id}"
    )
    assert removed.status_code == 204
    assert (await call("list_sources"))["sources"] == []
    assert (await call("search", query="Research"))["total"] == 0
    assert (await call("chat", question="Research?"))["context_found"] is False


async def test_unknown_source_is_rejected_before_a_key_is_created(
    client, db_session, test_user
):
    response = await client.post(
        "/api/v1/mcp-config/profiles",
        json={"name": "Scoped", "tools": ["search"], "source_ids": [str(uuid4())]},
    )
    assert response.status_code == 400
    assert await api_key_service.list_api_keys(db_session, test_user.id) == []


async def test_edit_profile_keeps_credential_and_permissions(
    client, db_session, test_user, admin_user
):
    first = DocumentSource(name="First", source_type="file", config={})
    second = DocumentSource(name="Second", source_type="file", config={})
    db_session.add_all([first, second])
    await db_session.commit()
    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Original",
            "programs": ["chosen"],
            "tools": ["search"],
            "source_ids": [str(first.id)],
        },
    )
    assert created.status_code == 201
    credential = created.json()
    key_id = credential["id"]
    source_url = f"/api/v1/mcp-config/keys/{key_id}/sources/{first.id}"
    assert (
        await client.put(
            source_url, json={"source_id": str(first.id), "can_chat": False}
        )
    ).status_code == 200

    profile_url = f"/api/v1/mcp-config/profiles/{key_id}"
    updated = await client.put(
        profile_url,
        json={
            "name": "Updated",
            "programs": ["other"],
            "tools": ["chat"],
            "source_ids": [str(first.id), str(second.id)],
        },
    )
    assert updated.status_code == 200, updated.text
    result = updated.json()
    assert result["api_key_id"] == key_id
    assert result["api_key_name"] == "Updated"
    assert result["mcp_programs"] == ["other"]
    assert result["allowed_tools"] == ["chat", "get_research_programs"]
    assert {
        source["source_name"]: source["can_chat"] for source in result["source_access"]
    } == {
        "First": False,
        "Second": True,
    }

    headers = {"Authorization": f"Bearer {credential['api_key']}"}
    listed = await client.get("/api/v1/mcp/tools", headers=headers)
    assert {tool["name"] for tool in listed.json()["tools"]} == {
        "chat",
        "get_research_programs",
    }
    assert len(await api_key_service.list_api_keys(db_session, test_user.id)) == 1

    renamed = await client.put(
        profile_url, json={"name": "Renamed", "programs": ["other"], "tools": ["chat"]}
    )
    assert renamed.status_code == 200
    assert renamed.json()["source_access_mode"] == "restricted"
    assert len(renamed.json()["source_access"]) == 2

    ordinary, _ = await api_key_service.create_api_key(
        db_session, test_user.id, "General"
    )
    somebody_elses, _ = await api_key_service.create_api_key(
        db_session,
        admin_user.id,
        "Other profile",
        mcp_programs=[],
        allowed_tools=["chat"],
    )
    for key in (ordinary, somebody_elses):
        response = await client.put(
            f"/api/v1/mcp-config/profiles/{key.id}",
            json={"name": "Denied", "tools": ["chat"]},
        )
        assert response.status_code == 404


async def test_rotation_replaces_the_secret_and_preserves_the_profile(
    client, db_session, test_user, admin_user
):
    source = DocumentSource(name="Lab", source_type="file", config={})
    db_session.add(source)
    await db_session.commit()
    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Lab profile",
            "programs": ["chosen"],
            "tools": ["search"],
            "source_ids": [str(source.id)],
        },
    )
    assert created.status_code == 201
    original = created.json()
    original_key, key_id = original["api_key"], original["id"]
    api_key = (await api_key_service.list_api_keys(db_session, test_user.id))[0]
    original_prefix = api_key.key_prefix
    management = f"/api/v1/mcp-config/keys/{key_id}"
    rotate = f"/api/v1/mcp-config/profiles/{key_id}/rotate"
    assert (
        await client.put(
            f"{management}/tools/search",
            json={"tool_name": "search", "is_enabled": False},
        )
    ).status_code == 200
    before = (await client.get(f"{management}/config")).json()

    async def tool_list(secret):
        return await client.get(
            "/api/v1/mcp/tools", headers={"Authorization": f"Bearer {secret}"}
        )

    assert (await tool_list(original_key)).status_code == 200
    rotated = await client.post(rotate, json={"key_prefix": original_prefix})
    assert rotated.status_code == 200
    assert rotated.headers["Cache-Control"] == "no-store"
    replacement = rotated.json()
    assert replacement["id"] == key_id
    assert replacement["key_prefix"] != original_prefix
    assert replacement["api_key"].startswith(replacement["key_prefix"])
    assert (await client.get(f"{management}/config")).json() == before
    assert (await tool_list(original_key)).status_code == 401
    tools = await tool_list(replacement["api_key"])
    assert [tool["name"] for tool in tools.json()["tools"]] == ["get_research_programs"]
    assert (
        await api_key_service.validate_api_key(db_session, replacement["api_key"])
        is None
    )

    retry = await client.post(rotate, json={"key_prefix": original_prefix})
    assert retry.status_code == 409
    assert (await tool_list(replacement["api_key"])).status_code == 200
    assert len(await api_key_service.list_api_keys(db_session, test_user.id)) == 1

    other, _ = await api_key_service.create_api_key(
        db_session, admin_user.id, "Private", mcp_programs=[], allowed_tools=["search"]
    )
    other_rotation = await client.post(
        f"/api/v1/mcp-config/profiles/{other.id}/rotate",
        json={"key_prefix": other.key_prefix},
    )
    assert other_rotation.status_code == 404


async def test_notes_profile_lists_and_reads_only_selected_owned_notes(
    client, db_session, test_user, admin_user
):
    selected = ResearchNote(
        user_id=test_user.id, title="Lab findings", content_markdown="# First draft"
    )
    unselected = ResearchNote(
        user_id=test_user.id, title="Lab private", content_markdown="Unshared"
    )
    other = ResearchNote(
        user_id=admin_user.id, title="Lab other", content_markdown="Other owner"
    )
    db_session.add_all([selected, unselected, other])
    await db_session.commit()

    candidates = await client.get(
        "/api/v1/mcp-config/profile-notes", params={"q": "LAB"}
    )
    assert {row["title"] for row in candidates.json()} == {
        "Lab findings",
        "Lab private",
    }
    lookup = "/api/v1/mcp-config/profile-notes"
    linked = await client.get(lookup, params={"note_id": str(selected.id)})
    assert linked.json() == [{"id": str(selected.id), "title": "Lab findings"}]
    unowned = await client.get(lookup, params={"note_id": str(other.id)})
    assert unowned.json() == []
    profile_url = "/api/v1/mcp-config/profiles"
    reject = await client.post(
        profile_url, json={"name": "Notes", "note_ids": [str(other.id)]}
    )
    assert reject.status_code == 400
    created = await client.post(
        profile_url, json={"name": "Notes", "note_ids": [str(selected.id)]}
    )
    assert created.status_code == 201
    key = created.json()
    config_url = f"/api/v1/mcp-config/keys/{key['id']}/config"
    config = await client.get(config_url)
    assert config.json()["research_notes"] == [
        {"id": str(selected.id), "title": "Lab findings"}
    ]
    headers = {"Authorization": f"Bearer {key['api_key']}"}
    available = await client.get("/api/v1/mcp/tools", headers=headers)
    assert [tool["name"] for tool in available.json()["tools"]] == [
        "get_research_notes"
    ]

    async def read(note_id=None):
        response = await client.post(
            "/api/v1/mcp/tools/call",
            headers=headers,
            json={
                "name": "get_research_notes",
                "arguments": {"note_id": note_id} if note_id else {},
            },
        )
        assert response.status_code == 200
        return response.json()

    assert [note["id"] for note in (await read())["content"][0]["text"]["notes"]] == [
        str(selected.id)
    ]
    assert (await read(str(selected.id)))["content"][0]["text"][
        "content_markdown"
    ] == "# First draft"
    for note in (unselected, other):
        assert (await read(str(note.id)))["isError"] is True

    selected.content_markdown = "# Updated findings"
    await db_session.commit()
    assert (await read(str(selected.id)))["content"][0]["text"][
        "content_markdown"
    ] == "# Updated findings"
    assert (
        await client.put(
            f"/api/v1/mcp-config/keys/{key['id']}/tools/get_research_notes",
            json={
                "tool_name": "get_research_notes",
                "config": {"note_ids": [str(other.id)]},
            },
        )
    ).status_code == 400
    updated = await client.put(
        f"{profile_url}/{key['id']}", json={"name": "Notes renamed"}
    )
    assert updated.status_code == 200
    assert updated.json()["research_notes"] == config.json()["research_notes"]

    detached = await client.put(
        f"{profile_url}/{key['id']}",
        json={"name": "Notes renamed", "tools": ["search"], "note_ids": []},
    )
    assert detached.status_code == 200
    assert detached.json()["allowed_tools"] == ["search"]
    assert detached.json()["research_notes"] == []


async def test_profile_notes_are_discoverable_as_policy_checked_resources(
    client, db_session, test_user
):
    note = ResearchNote(
        user_id=test_user.id, title="Shared note", content_markdown="# Findings"
    )
    private = ResearchNote(
        user_id=test_user.id, title="Private note", content_markdown="Private"
    )
    db_session.add_all([note, private])
    await db_session.commit()
    response = await client.post(
        "/api/v1/mcp-config/profiles",
        json={"name": "Published notes", "note_ids": [str(note.id)]},
    )
    assert response.status_code == 201
    profile = response.json()
    headers = {"Authorization": f"Bearer {profile['api_key']}"}
    uri = f"kdbc://research-notes/{note.id}"

    async def rpc(method, **params):
        response = await client.post(
            "/api/v1/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        assert response.status_code == 200
        return response.json()

    initialized = await rpc("initialize", protocolVersion="2025-11-25")
    assert initialized["result"]["capabilities"]["resources"] == {}
    listed = await rpc("resources/list")
    assert listed["result"] == {
        "resources": [{"uri": uri, "name": "Shared note", "mimeType": "text/markdown"}]
    }
    assert (await rpc("resources/templates/list"))["result"] == {
        "resourceTemplates": []
    }
    assert (await rpc("resources/list", cursor="unknown"))["error"]["code"] == -32602
    assert (await rpc("resources/read"))["error"]["code"] == -32602
    read = await rpc("resources/read", uri=uri)
    assert read["result"]["contents"] == [
        {"uri": uri, "mimeType": "text/markdown", "text": "# Findings"}
    ]
    for invalid in (
        f"kdbc://research-notes/{private.id}",
        "file:///private.md",
        f"{uri}?raw=1",
    ):
        assert (await rpc("resources/read", uri=invalid))["error"]["code"] == -32002

    note.content_markdown = "# Updated findings"
    await db_session.commit()
    assert (await rpc("resources/read", uri=uri))["result"]["contents"][0][
        "text"
    ] == "# Updated findings"

    tool_settings = f"/api/v1/mcp-config/keys/{profile['id']}/tools/get_research_notes"
    for enabled in (False, True):
        changed = await client.put(
            tool_settings,
            json={"tool_name": "get_research_notes", "is_enabled": enabled},
        )
        assert changed.status_code == 200
        assert bool((await rpc("resources/list"))["result"]["resources"]) == enabled
        if not enabled:
            assert (await rpc("resources/read", uri=uri))["error"]["code"] == -32002

    db_session.add(
        ToolPolicy(
            tool_name="mcp:get_research_notes", effect="allow", require_approval=True
        )
    )
    await db_session.commit()
    assert (await rpc("resources/list"))["result"]["resources"] == []
    assert (await rpc("resources/read", uri=uri))["error"]["code"] == -32002


async def test_program_resources_share_selected_manifests_and_obey_policy(
    client, db_session, test_user, monkeypatch
):
    spec = ToolSpec(
        "measure_sample",
        "Measure a sample",
        {"type": "object", "properties": {"sample": {"type": "string"}}},
    )
    selected = ResearchProgram(
        "RTL / Lab",
        version="2.1",
        description="Architecture context",
        tool_specs=(spec,),
    )
    installed = [selected, ResearchProgram("private")]
    monkeypatch.setattr(research_programs, "installed", lambda: installed)
    note = ResearchNote(
        user_id=test_user.id, title="Results", content_markdown="# Results"
    )
    db_session.add(note)
    await db_session.commit()
    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Mixed context",
            "programs": [selected.name],
            "note_ids": [str(note.id)],
        },
    )
    assert created.status_code == 201
    credential = created.json()
    headers = {"Authorization": f"Bearer {credential['api_key']}"}
    program_uri = "kdbc://research-programs/RTL%20%2F%20Lab.json"
    note_uri = f"kdbc://research-notes/{note.id}"

    async def rpc(method, **params):
        response = await client.post(
            "/api/v1/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        assert response.status_code == 200
        return response.json()

    resources = (await rpc("resources/list"))["result"]["resources"]
    assert {resource["uri"] for resource in resources} == {program_uri, note_uri}
    assert next(r for r in resources if r["uri"] == program_uri) == {
        "uri": program_uri,
        "name": "RTL / Lab",
        "description": "Architecture context",
        "mimeType": "application/json",
    }
    manifest = (await rpc("resources/read", uri=program_uri))["result"]["contents"][0]
    assert manifest["mimeType"] == "application/json"
    assert json.loads(manifest["text"]) == {
        "name": "RTL / Lab",
        "version": "2.1",
        "description": "Architecture context",
        "tools": [spec.schema()],
    }
    for uri in ("kdbc://research-programs/private.json", f"{program_uri}?version=2"):
        assert (await rpc("resources/read", uri=uri))["error"]["code"] == -32002

    installed.remove(selected)
    assert (await rpc("resources/read", uri=program_uri))["error"]["code"] == -32002
    installed.append(selected)

    db_session.add(
        ToolPolicy(
            subject_type="api_key",
            subject_id=UUID(credential["id"]),
            tool_name="mcp:get_research_programs",
            effect="deny",
        )
    )
    await db_session.commit()
    assert (await rpc("resources/read", uri=program_uri))["error"]["code"] == -32002
    assert [r["uri"] for r in (await rpc("resources/list"))["result"]["resources"]] == [
        note_uri
    ]


async def test_streamable_usage_counts_operations_without_recording_request_data(
    client, db_session
):
    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={"name": "Usage test", "programs": ["chosen"]},
    )
    credential = created.json()
    headers = {"Authorization": f"Bearer {credential['api_key']}"}
    requests = [
        ("initialize", {}),
        ("notifications/initialized", {}),
        ("tools/list", {}),
        (
            "tools/call",
            {"name": "get_research_programs", "arguments": {"private": "redact-this"}},
        ),
        ("tools/call", {"name": "redact-this"}),
        ("resources/list", {}),
        ("resources/read", {"uri": "kdbc://research-notes/redact-this"}),
        ("redact-this", {}),
    ]
    for method, params in requests:
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if method != "notifications/initialized":
            message["id"] = 1
        response = await client.post("/api/v1/mcp", headers=headers, json=message)
        assert response.status_code in (200, 202)
    malformed = await client.post("/api/v1/mcp", headers=headers, content=b"{")
    assert malformed.status_code == 400

    usage = await api_key_service.get_usage_stats(db_session, UUID(credential["id"]))
    assert usage["total_requests"] == usage["lifetime_requests"] == 9
    assert {entry["endpoint"]: entry["count"] for entry in usage["top_endpoints"]} == {
        f"/api/v1/mcp/rpc/{operation}": 1
        for operation in (
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call/get_research_programs",
            "tools/call",
            "resources/list",
            "resources/read",
            "unknown",
            "invalid",
        )
    }


async def test_mcp_prompts_follow_profile_resources_and_validate_arguments(
    client, db_session, test_user
):
    notes = [
        ResearchNote(
            user_id=test_user.id,
            title=f"Note {index}",
            content_markdown="Unembedded note body",
        )
        for index in range(3)
    ]
    db_session.add_all(notes)
    await db_session.commit()
    created = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Prompt context",
            "programs": ["chosen"],
            "note_ids": [str(note.id) for note in notes[:2]],
        },
    )
    assert created.status_code == 201
    key = created.json()
    headers = {"Authorization": f"Bearer {key['api_key']}"}

    async def rpc(method, **params):
        response = await client.post(
            "/api/v1/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        assert response.status_code == 200
        return response.json()

    initialized = await rpc("initialize", protocolVersion="2025-11-25")
    assert initialized["result"]["capabilities"]["prompts"] == {}
    prompts = (await rpc("prompts/list"))["result"]["prompts"]
    assert [prompt["name"] for prompt in prompts] == [
        "research_brief",
        "compare_research_notes",
    ]
    assert prompts[0]["arguments"][0]["required"] is True
    assert (await rpc("prompts/list", cursor="unknown"))["error"]["code"] == -32602

    brief = await rpc(
        "prompts/get", name="research_brief", arguments={"topic": "Cache behavior"}
    )
    [message] = brief["result"]["messages"]
    assert message["role"] == "user"
    assert message["content"]["type"] == "text"
    text = message["content"]["text"]
    assert "Cache behavior" in text
    assert "kdbc://research-programs/chosen.json" in text
    assert all(f"kdbc://research-notes/{note.id}" in text for note in notes[:2])
    assert str(notes[2].id) not in text and "Unembedded note body" not in text

    compare = await rpc(
        "prompts/get", name="compare_research_notes", arguments={"focus": "Latency"}
    )
    comparison_text = compare["result"]["messages"][0]["content"]["text"]
    assert "Latency" in comparison_text
    assert "kdbc://research-programs/" not in comparison_text
    for name, arguments in (
        ("research_brief", {}),
        ("research_brief", {"topic": "a" * 401}),
        ("research_brief", {"topic": {"nested": True}}),
        ("compare_research_notes", {"unexpected": "argument"}),
        ("missing", {}),
    ):
        assert (await rpc("prompts/get", name=name, arguments=arguments))["error"][
            "code"
        ] == -32602

    db_session.add(ToolPolicy(tool_name="mcp:get_research_notes", effect="deny"))
    await db_session.commit()
    prompts = (await rpc("prompts/list"))["result"]["prompts"]
    assert [prompt["name"] for prompt in prompts] == ["research_brief"]
    assert (await rpc("prompts/get", name="compare_research_notes"))["error"][
        "code"
    ] == -32602
    brief = await rpc(
        "prompts/get", name="research_brief", arguments={"topic": "Cache behavior"}
    )
    assert (
        "kdbc://research-notes/"
        not in brief["result"]["messages"][0]["content"]["text"]
    )

    db_session.add(
        ToolPolicy(
            tool_name="mcp:get_research_programs",
            effect="allow",
            require_approval=True,
        )
    )
    await db_session.commit()
    assert (await rpc("prompts/list"))["result"]["prompts"] == []


async def test_custom_prompts_are_profile_scoped_and_editable(client, db_session):
    custom = {
        "name": "write_summary",
        "title": "Write a summary",
        "instructions": "Record the key findings.",
        "include_resources": False,
        "use_inputs": False,
    }
    base = "/api/v1/mcp-config/profiles"
    created = await client.post(
        base, json={"name": "Instructions", "prompts": [custom]}
    )
    assert created.status_code == 201, created.text
    profile = created.json()
    other = await client.post(base, json={"name": "Other", "programs": ["chosen"]})
    assert other.status_code == 201

    async def rpc(method, key=profile["api_key"], **params):
        result = await client.post(
            "/api/v1/mcp",
            headers={"Authorization": f"Bearer {key}"},
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        return result.json()

    assert (await rpc("tools/list"))["result"]["tools"] == []
    listed = await rpc("prompts/list")
    assert listed["result"]["prompts"] == [
        {"name": "write_summary", "title": "Write a summary"}
    ]
    prompt = await rpc("prompts/get", name="write_summary")
    assert prompt["result"]["messages"][0]["content"]["text"] == custom["instructions"]
    assert (
        await rpc("prompts/get", key=other.json()["api_key"], name="write_summary")
    )["error"]["code"] == -32602
    assert (
        await rpc("prompts/get", name="write_summary", arguments={"topic": "invalid"})
    )["error"]["code"] == -32602

    updated = await client.put(
        f"{base}/{profile['id']}",
        json={"name": "Instructions renamed", "programs": ["chosen"]},
    )
    assert updated.status_code == 200
    assert updated.json()["prompts"] == [custom]
    prompt = await rpc("prompts/get", name="write_summary")
    assert prompt["result"]["messages"][0]["content"]["text"] == custom["instructions"]
    custom = {
        **custom,
        "instructions": "Summarize the measurements.",
        "include_resources": True,
    }
    updated = await client.put(
        f"{base}/{profile['id']}",
        json={
            "name": "Instructions renamed",
            "programs": ["chosen"],
            "prompts": [custom],
        },
    )
    assert updated.status_code == 200
    prompt = await rpc("prompts/get", name="write_summary")
    text = prompt["result"]["messages"][0]["content"]["text"]
    assert text.startswith(custom["instructions"])
    assert "kdbc://research-programs/chosen.json" in text

    db_session.add(ToolPolicy(tool_name="mcp:get_research_programs", effect="deny"))
    await db_session.commit()
    prompt = await rpc("prompts/get", name="write_summary")
    assert prompt["result"]["messages"][0]["content"]["text"] == custom["instructions"]
    removed = await client.put(
        f"{base}/{profile['id']}",
        json={"name": "Programs only", "programs": ["chosen"], "prompts": []},
    )
    assert removed.status_code == 200
    assert (await rpc("prompts/get", name="write_summary"))["error"]["code"] == -32602


@pytest.mark.parametrize(
    "names", [["research_brief"], ["invalid name"], ["same", "same"]]
)
async def test_custom_prompt_names_are_validated(client, names):
    result = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Invalid prompts",
            "prompts": [
                {"name": name, "title": "Title", "instructions": "Instructions"}
                for name in names
            ],
        },
    )
    assert result.status_code == 422


async def test_custom_prompt_inputs_are_discovered_and_replaced_once(
    client, db_session, test_user
):
    prompt = {
        "name": "brief_for_reader",
        "title": "Brief for reader",
        "instructions": "Write {{topic}} for {{ reader }}. Restate {{topic}}. Preserve {{{literal}}}.",
        "use_inputs": True,
        "include_resources": False,
    }
    overflow = {**prompt, "name": "overflow", "instructions": "{{topic}}" * 100}
    legacy = {
        "name": "literal",
        "title": "Literal",
        "instructions": "Keep {{placeholder}}.",
    }
    response = await client.post(
        "/api/v1/mcp-config/profiles",
        json={"name": "Prompt inputs", "prompts": [prompt, overflow, legacy]},
    )
    assert response.status_code == 201, response.text
    key = response.json()
    api_key = (await api_key_service.list_api_keys(db_session, test_user.id))[0]
    # Older stored rows lack the opt-in flag.
    api_key.mcp_prompts = [
        saved
        if saved["name"] != "literal"
        else {field: value for field, value in saved.items() if field != "use_inputs"}
        for saved in api_key.mcp_prompts
    ]
    await db_session.commit()

    async def rpc(method, **params):
        answer = await client.post(
            "/api/v1/mcp",
            headers={"Authorization": f"Bearer {key['api_key']}"},
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        return answer.json()

    listed = (await rpc("prompts/list"))["result"]["prompts"]
    assert listed[0]["arguments"] == [
        {"name": "topic", "description": "Topic", "required": True},
        {"name": "reader", "description": "Reader", "required": True},
    ]
    assert "arguments" not in listed[2]
    literal = await rpc("prompts/get", name="literal")
    assert literal["result"]["messages"][0]["content"]["text"] == legacy["instructions"]

    answer = await rpc(
        "prompts/get",
        name=prompt["name"],
        arguments={"topic": "Cache {{reader}}", "reader": "the lab"},
    )
    assert answer["result"]["messages"][0]["content"]["text"] == (
        "Write Cache {{reader}} for the lab. Restate Cache {{reader}}. Preserve {{{literal}}}."
    )
    for arguments in (
        {"topic": "Caches"},
        {"topic": "Caches", "reader": " "},
        {"topic": "Caches", "reader": 42},
        {"topic": "Caches", "reader": "a" * 401},
        {"topic": "Caches", "reader": "Lab", "extra": "x"},
    ):
        rejected = await rpc("prompts/get", name=prompt["name"], arguments=arguments)
        assert rejected["error"]["code"] == -32602
    rejected = await rpc("prompts/get", name="overflow", arguments={"topic": "a" * 400})
    assert rejected["error"]["code"] == -32602


@pytest.mark.parametrize(
    "instructions",
    ["Ordinary text.", " ".join("{{field_" + str(i) + "}}" for i in range(11))],
)
async def test_prompt_inputs_require_one_to_ten_unique_names(client, instructions):
    rejected = await client.post(
        "/api/v1/mcp-config/profiles",
        json={
            "name": "Invalid inputs",
            "prompts": [
                {
                    "name": "brief",
                    "title": "Brief",
                    "instructions": instructions,
                    "use_inputs": True,
                }
            ],
        },
    )
    assert rejected.status_code == 422
