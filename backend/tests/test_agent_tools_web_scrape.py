from app.services.agent_tools import get_tool_by_name, validate_tool_params


def test_web_scrape_tool_is_registered():
    tool = get_tool_by_name("web_scrape")
    assert tool is not None
    assert tool["name"] == "web_scrape"


def test_web_scrape_tool_requires_url():
    ok, err = validate_tool_params("web_scrape", {})
    assert ok is False
    assert "Missing required parameter" in err

    ok, err = validate_tool_params("web_scrape", {"url": "https://example.com"})
    assert ok is True
    assert err == ""

