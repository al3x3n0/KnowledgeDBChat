"""`delegate_to_agent` reaches the model and runs the tools it asks for.

It called `LLMService.generate_chat_response`, which never existed. The
AttributeError was caught and every delegation returned "Delegation failed".
"""

from uuid import uuid4

import pytest

from app.models.agent_definition import AgentDefinition
from app.services.agent_service import AgentService
from app.services.llm_providers.base import LLMCompletion, LLMToolCall

pytestmark = pytest.mark.unit


class ScriptedLLM:
    def __init__(self, completions):
        self.completions, self.calls = list(completions), []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions.pop(0)


@pytest.fixture
async def specialist(db_session):
    agent = AgentDefinition(
        id=uuid4(),
        name="cache_expert",
        display_name="Cache Expert",
        system_prompt="You know caches.",
        capabilities=[],
        tool_whitelist=["search_documents"],
        is_active=True,
    )
    db_session.add(agent)
    await db_session.commit()
    return agent


def _service(llm):
    service = AgentService.__new__(AgentService)
    service.llm_service = llm
    service._chat_tools = lambda: [
        {"name": "search_documents", "description": "", "parameters": {}},
        {"name": "delegate_to_agent", "description": "", "parameters": {}},
        {"name": "web_scrape", "description": "", "parameters": {}},
    ]
    return service


async def test_a_delegated_task_is_answered(db_session, test_user, specialist):
    llm = ScriptedLLM([LLMCompletion(text="Use a stride prefetcher.")])
    service = _service(llm)

    result = await service._tool_delegate_to_agent(
        {"target_agent": "cache_expert", "task_description": "Pick a prefetcher"},
        test_user.id,
        db_session,
    )

    assert result["result"] == "Use a stride prefetcher."
    assert result["delegated_to"] == "cache_expert"
    # The whitelist decides the menu, and delegation is never on it.
    assert [t["name"] for t in llm.calls[0]["tools"]] == ["search_documents"]
    assert llm.calls[0]["messages"][0]["content"] == "You know caches."


async def test_a_delegate_that_calls_a_tool_gets_its_result(
    db_session, test_user, specialist
):
    llm = ScriptedLLM(
        [
            LLMCompletion(
                tool_calls=[
                    LLMToolCall(
                        id="1", name="search_documents", arguments={"query": "l2"}
                    )
                ]
            ),
            LLMCompletion(text="Two papers agree."),
        ]
    )
    service = _service(llm)
    executed = []

    async def _execute_tool(tool_call, user_id, db, *args, **kwargs):
        executed.append((tool_call.tool_name, tool_call.tool_input))
        tool_call.tool_output, tool_call.status = {"hits": 2}, "completed"
        return tool_call

    service._execute_tool = _execute_tool

    result = await service._tool_delegate_to_agent(
        {"target_agent": "cache_expert", "task_description": "What is known?"},
        test_user.id,
        db_session,
    )

    assert executed == [("search_documents", {"query": "l2"})]
    assert result["tools_used"] == ["search_documents"]
    assert result["result"] == "Two papers agree."
    assert llm.calls[1]["tools"] is None
    assert "hits" in llm.calls[1]["messages"][-1]["content"]
