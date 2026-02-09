from app.services.agent_tools import get_tool_by_name, validate_tool_params


def test_ingest_url_tool_is_registered():
    tool = get_tool_by_name("ingest_url")
    assert tool is not None
    assert tool["name"] == "ingest_url"


def test_ingest_url_tool_requires_url():
    ok, err = validate_tool_params("ingest_url", {})
    assert ok is False
    assert "Missing required parameter" in err

    ok, err = validate_tool_params("ingest_url", {"url": "https://example.com"})
    assert ok is True
    assert err == ""

