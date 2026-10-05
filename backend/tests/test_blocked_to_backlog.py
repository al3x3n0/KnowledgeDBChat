"""A run blocked by the platform files the platform's problem.

A run blocked by its own bad input is the development loop: write code, read
the diagnostic, fix line 12. A run blocked because a *tool* cannot do what it
was asked is different in kind -- nothing the run writes will help, and every
later run meets the same wall. Those blockers arrive fully specified, because
the tools say what they need, and they used to sit in a paused job until a
person read the logs.

The discriminator is the one the escalation machinery already uses, so the two
cannot drift apart.
"""

from app.services import agent_blocked_to_backlog as filing

MNEMONIC = (
    "The sequence could not be costed. EMITFAIL sequence candidate-coster: "
    "unknown mnemonic 'uaddw': add it to operand_arity, since guessing its "
    "operand count emits assembly the assembler will reject"
)
SYNTAX = "Compilation failed: /work/prog.c:12:5: error: expected ';' after expression"


def _state(*failures):
    return {
        "actions_taken": [
            {
                "action": {"tool": tool},
                "result": {"success": False, "error": error},
            }
            for tool, error in failures
        ]
    }


class TestWhatItFiles:
    def test_a_repeated_tool_failure_is_a_blocker(self):
        found = filing.blocker(
            _state(
                ("cost_fusion_candidate", MNEMONIC), ("cost_fusion_candidate", MNEMONIC)
            )
        )
        assert found["tool"] == "cost_fusion_candidate"
        assert found["attempts"] == 2
        assert "uaddw" in found["error"]

    def test_the_worst_offender_wins_when_several_failed(self):
        found = filing.blocker(
            _state(
                ("a_tool", "the daemon is unreachable"),
                ("cost_fusion_candidate", MNEMONIC),
                ("cost_fusion_candidate", MNEMONIC),
                ("cost_fusion_candidate", MNEMONIC),
            )
        )
        assert found["tool"] == "cost_fusion_candidate"
        assert found["attempts"] == 3


class TestWhatItRefusesToFile:
    def test_the_run_s_own_broken_code_is_not_a_platform_gap(self):
        # Writing code that does not compile and fixing it is the loop working.
        # Filing those would bury the real ones.
        assert (
            filing.blocker(
                _state(("compile_c_snippet", SYNTAX), ("compile_c_snippet", SYNTAX))
            )
            is None
        )

    def test_one_failure_is_not_yet_a_wall(self):
        # Once may be a flake or a half-written argument the next iteration
        # fixes; twice with the same class is the run meeting a wall.
        assert filing.blocker(_state(("cost_fusion_candidate", MNEMONIC))) is None

    def test_a_run_that_succeeded_files_nothing(self):
        state = {
            "actions_taken": [
                {"action": {"tool": "t"}, "result": {"success": True, "data": 1}}
            ]
        }
        assert filing.blocker(state) is None

    def test_an_empty_run_files_nothing(self):
        assert filing.blocker({}) is None


class TestItUsesTheSameDiscriminatorAsTheEscalation:
    def test_it_defers_to_blames_the_submitted_code(self):
        # If these drift apart, one of them starts lying about whose fault a
        # failure is.
        from app.services import agent_failure_diagnosis as diagnosis

        assert diagnosis.blames_the_submitted_code(SYNTAX) is True
        assert diagnosis.blames_the_submitted_code(MNEMONIC) is False

    def test_a_missing_toolchain_is_the_platform_s_problem(self):
        # The docstring's own example: nothing about the source is wrong.
        missing = "Compilation failed: clang: not found"
        found = filing.blocker(
            _state(("compile_c_snippet", missing), ("compile_c_snippet", missing))
        )
        assert found is not None


class TestFilingIt:
    """Against a real session: what lands, and what must not land twice."""

    def _job(self, user_id):
        from types import SimpleNamespace
        from uuid import uuid4

        return SimpleNamespace(id=uuid4(), user_id=user_id)

    def _file(self, db, job, state):
        import asyncio

        async def _run():
            item = await filing.file_blocker(job, state, db)
            await db.commit()
            return item

        return asyncio.get_event_loop().run_until_complete(_run())

    def test_the_item_carries_the_tool_s_own_words(self, db_session, test_user):
        state = _state(
            ("cost_fusion_candidate", MNEMONIC), ("cost_fusion_candidate", MNEMONIC)
        )
        item = self._file(db_session, self._job(test_user.id), state)

        assert item is not None
        assert "uaddw" in item.error_output
        assert "cost_fusion_candidate" in item.title

    def test_it_is_never_applied_unattended(self, db_session, test_user):
        # This edits the platform the agents themselves run on. An item that
        # files itself and then applies itself is two decisions taken by
        # something that has earned one.
        state = _state(
            ("simulate_mechanism", "the daemon is unreachable"),
            ("simulate_mechanism", "the daemon is unreachable"),
        )
        item = self._file(db_session, self._job(test_user.id), state)

        assert item.auto_apply_enabled is False
        assert item.status == "draft"

    def test_the_same_wall_met_twice_is_one_piece_of_work(self, db_session, test_user):
        # Four runs meeting the same missing mnemonic is one task, and four
        # identical items is a backlog nobody reads.
        state = _state(
            ("cost_fusion_candidate", MNEMONIC), ("cost_fusion_candidate", MNEMONIC)
        )
        first = self._file(db_session, self._job(test_user.id), state)
        second = self._file(db_session, self._job(test_user.id), state)

        assert first is not None
        assert second is None

    def test_a_run_with_nothing_to_file_files_nothing(self, db_session, test_user):
        assert self._file(db_session, self._job(test_user.id), {}) is None


class TestStartingWorkIsSeparateFromRecordingIt:
    """Filing costs nothing; starting a run costs, so only starting is gated."""

    def _file(self, db, user_id, monkeypatch, enabled):
        import asyncio
        from types import SimpleNamespace
        from uuid import uuid4

        from app.core.config import settings

        monkeypatch.setattr(
            settings, "AGENT_BLOCKER_AUTO_CODING_ENABLED", enabled, raising=False
        )
        job = SimpleNamespace(id=uuid4(), user_id=user_id)
        state = _state(
            ("cost_fusion_candidate", MNEMONIC), ("cost_fusion_candidate", MNEMONIC)
        )

        async def _run():
            item = await filing.file_blocker(job, state, db)
            await db.commit()
            return item

        return asyncio.get_event_loop().run_until_complete(_run())

    def test_the_blocker_is_recorded_even_with_the_flag_off(
        self, db_session, test_user, monkeypatch
    ):
        item = self._file(db_session, test_user.id, monkeypatch, False)
        assert item is not None
        # Recorded, but nobody was sent to work on it.
        assert item.orchestrator_job_id is None

    def test_nothing_can_land_unattended_either_way(
        self, db_session, test_user, monkeypatch
    ):
        # The runner resolves auto_apply_enabled=False to proposal_only, so
        # even with the flag on the outcome is a proposal a person reviews.
        item = self._file(db_session, test_user.id, monkeypatch, True)
        assert item.auto_apply_enabled is False
