"""Drafting an agent definition from a description.

The prose in a definition is the easy part. The two closed sets either side of
it are not, and both fail silently: a capability outside the router's
vocabulary means the agent is never routed to, and a tool whitelist naming
something that is not a tool leaves the agent with fewer tools than its author
believes. Neither raises, neither logs, and both look right on the page.

So the checks are what these tests pin, against the real vocabularies rather
than fixtures -- the whole value is whether a draft would survive the create
endpoint and then actually work.
"""

import pytest

from app.agent_core import tool_specs
from app.agent_core.routing import CAPABILITY_KEYWORDS
from app.services import agent_definition_author_service as author

GOOD = {
    "name": "ise_scout",
    "display_name": "ISE Scout",
    "description": "Finds instruction fusion candidates.",
    "system_prompt": "You find instruction fusion candidates and cost them.",
    "capabilities": ["code_analysis"],
    "tool_whitelist": ["search_arxiv"],
    "priority": 50,
}


class TestTheVocabularyIsReadNotRestated:
    def test_capabilities_come_from_the_router(self):
        # If these drift, drafts are refused for a reason the author cannot act
        # on, because the prompt and the checker would disagree.
        assert set(author.vocabulary()["capabilities"]) == set(CAPABILITY_KEYWORDS)

    def test_tools_come_from_the_catalog(self):
        assert set(author.vocabulary()["tools"]) == set(
            tool_specs.STATIC_CATALOG.spec_names()
        )

    def test_the_prompt_names_the_real_vocabulary(self):
        prompt = author._system_prompt()
        assert "code_analysis" in prompt and "search_arxiv" in prompt


class TestWhatItAccepts:
    def test_a_sound_definition_passes(self):
        definition, complaints = author.check(GOOD)
        assert complaints == []
        assert definition["name"] == "ise_scout"

    def test_omitting_the_whitelist_is_allowed(self):
        # None means every tool, which is right for a general agent.
        payload = {k: v for k, v in GOOD.items() if k != "tool_whitelist"}
        definition, complaints = author.check(payload)
        assert complaints == []
        assert definition["tool_whitelist"] is None


class TestTheSilentFailures:
    """Each of these would validate as prose and then quietly not work."""

    def test_a_capability_the_router_does_not_know(self):
        _, complaints = author.check(dict(GOOD, capabilities=["fusion_mining"]))
        (complaint,) = complaints
        assert "never be routed to" in complaint
        # And it says what would be right, so the next attempt can fix it.
        assert "code_analysis" in complaint

    def test_a_whitelist_naming_something_that_is_not_a_tool(self):
        _, complaints = author.check(dict(GOOD, tool_whitelist=["make_me_a_sandwich"]))
        (complaint,) = complaints
        assert "fewer tools than intended" in complaint

    def test_an_empty_whitelist_grants_nothing(self):
        # Not the same as omitting it, and the difference is total.
        _, complaints = author.check(dict(GOOD, tool_whitelist=[]))
        assert any("allows no tools at all" in c for c in complaints)


class TestItRefusesWhatCreateWouldRefuse:
    @pytest.mark.parametrize(
        "payload,field",
        [
            ({**GOOD, "name": "ISE Scout!"}, "name"),
            ({**GOOD, "system_prompt": "short"}, "system_prompt"),
            ({**GOOD, "priority": 900}, "priority"),
        ],
    )
    def test_the_schema_that_refuses_is_the_endpoint_s_own(self, payload, field):
        definition, complaints = author.check(payload)
        assert definition is None
        assert any(c.startswith(field) for c in complaints)


class TestDraftingItself:
    @pytest.mark.asyncio
    async def test_an_empty_description_asks_for_one_rather_than_guessing(self):
        out = await author.draft_definition("   ")
        assert out["definition"] is None
        assert out["notes"] == ["No description was given."]

    def test_a_reply_that_is_not_json_yields_nothing(self):
        assert author._payload("not json at all") == {}

    def test_a_reply_wrapped_in_a_data_key_is_still_read(self):
        # Providers differ in how they return structured output.
        assert author._payload({"data": GOOD})["name"] == "ise_scout"


class TestRefiningWhatIsAlreadyThere:
    """A second pass revises the form, it does not start over.

    The form is the source of truth rather than the model's own last answer: a
    person may have edited a field by hand between drafts, and refining from
    what the model said would silently discard that edit.
    """

    def test_the_revision_shows_the_model_what_it_is_changing(self):
        message = author._revision_message(
            "restrict it to the coding tools", dict(GOOD)
        )
        assert "ise_scout" in message and "code_analysis" in message

    def test_it_says_to_leave_untouched_things_alone(self):
        # Otherwise "make it narrower" comes back as a different agent.
        message = author._revision_message("narrow it", dict(GOOD))
        assert "Keep everything the change does not touch" in message

    def test_empty_fields_are_not_shown_as_if_they_were_set(self):
        # An empty whitelist and an absent one mean opposite things; showing
        # `[]` as the current state would invite the model to preserve it.
        message = author._revision_message(
            "add a tool", dict(GOOD, tool_whitelist=[], description="")
        )
        assert '"tool_whitelist"' not in message
        assert '"description"' not in message

    @pytest.mark.asyncio
    async def test_an_empty_instruction_says_so_in_the_right_words(self):
        out = await author.draft_definition("  ", current=dict(GOOD))
        assert out["notes"] == ["No change was described."]

    @pytest.mark.asyncio
    async def test_a_first_draft_still_says_no_description(self):
        out = await author.draft_definition("  ")
        assert out["notes"] == ["No description was given."]


class TestRefinementIsNotAnExcuse:
    def test_a_revision_is_checked_exactly_as_a_first_draft_is(self):
        # "Make it narrower" is no reason to accept an agent nothing can route
        # to; check() is the same function either way.
        _, complaints = author.check(dict(GOOD, capabilities=["narrower_thing"]))
        assert any("never be routed to" in c for c in complaints)
