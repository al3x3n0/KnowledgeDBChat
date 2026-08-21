"""Finding instruction sequences worth fusing, from code that really ran.

The decoder is the foundation: a data-flow graph built from wrong defs and
uses produces confident nonsense, and every case below is a form the compiler
actually emitted or a rule whose absence lost a real candidate.
"""

from __future__ import annotations

from app.services import isa_candidate_mining as mining

# A vec3-normalize inner loop, the shape this exists to mine. Written with
# explicit tab escapes: the compiler separates mnemonic from operands with a
# real tab, and a raw one in this file trips the linter for the whole module.
NORMALIZE = [
    "\tldr\ts1, [x8, x9, lsl #2]",
    "\tldr\ts2, [x10, x9, lsl #2]",
    "\tfmul\ts3, s1, s1",
    "\tfmadd\ts3, s2, s2, s3",
    "\tfsqrt\ts4, s3",
    "\tfdiv\ts5, s0, s4",
    "\tfmul\ts6, s1, s5",
    "\tstr\ts6, [x11, x9, lsl #2]",
    "\tadd\tx9, x9, #1",
    "\tcmp\tx9, #256",
    "\tb.ne\t.LBB0_1",
]


def test_register_widths_and_views_are_one_register():
    """w3 and x3 are the same register; so are s0, d0, q0 and v0.

    Treating them as different loses data-flow edges silently, and does it in
    exactly the floating-point code this is aimed at.
    """
    assert mining.normalize_register("w3") == mining.normalize_register("x3")
    for name in ("s0", "d0", "q0", "b0", "h0", "v0.4s"):
        assert mining.normalize_register(name) == "v0", name
    assert mining.normalize_register("sp") is None
    assert mining.normalize_register("xzr") is None
    assert mining.normalize_register("#1024") is None


def test_both_assembly_layouts_decode():
    """objdump writes an address first; the compiler writes a leading tab.

    Reading the last tab-separated field works for one and returns the
    *operands* for the other, which made every mnemonic a register name.
    """
    from_compiler = mining.parse_instruction("\tadd\tx0, x0, #1")
    from_objdump = mining.parse_instruction("  4005a0:\t91000400 \tadd\tx0, x0, #1")

    assert from_compiler.mnemonic == "add" == from_objdump.mnemonic
    assert from_compiler.defs == from_objdump.defs == ("x0",)


def test_a_store_reads_its_first_operand_rather_than_writing_it():
    stored = mining.parse_instruction("str s6, [x11, x9, lsl #2]")

    assert stored.defs == ()
    assert set(stored.uses) == {"v6", "x11", "x9"}


def test_a_load_that_reuses_its_base_reads_and_writes_it():
    loaded = mining.parse_instruction("ldr x0, [x0, #8]")

    assert loaded.defs == ("x0",)
    assert loaded.uses == ("x0",)


def test_pair_forms_touch_two_registers():
    """`ldp` loads two, and crediting one leaves an edge missing."""
    pair = mining.parse_instruction("ldp x1, x2, [x8]")

    assert set(pair.defs) == {"x1", "x2"}
    assert pair.uses == ("x8",)


def test_writeback_is_credited_to_the_base_register():
    """`stp x29, x30, [sp, #-32]!` writes back sp, not x29."""
    pre = mining.parse_instruction("stp x29, x30, [sp, #-32]!")
    assert pre.defs == ()

    post = mining.parse_instruction("ldp x1, x2, [x8], #16")
    assert "x8" in post.defs, "post-indexed writeback carries no '!'"


def test_labels_and_directives_are_not_instructions():
    for line in (".LBB0_1:", "\t.text", "// a comment", "", "\t.cfi_def_cfa w29, -16"):
        assert mining.parse_instruction(line) is None, line


def test_the_graph_links_a_value_to_its_producer():
    decoded = mining.parse_block(NORMALIZE)
    graph = mining.build_dfg(decoded)
    by_mnemonic = {i.mnemonic: i.index for i in decoded}

    assert (by_mnemonic["fsqrt"], by_mnemonic["fdiv"]) in graph.edges
    assert (by_mnemonic["fmadd"], by_mnemonic["fsqrt"]) in graph.edges


def test_reciprocal_square_root_is_found():
    """The best known fusion in this kernel, and the one a stricter liveness
    rule threw away by counting the sqrt's intermediate as a second result."""
    ranked = mining.mine_blocks([{"instructions": NORMALIZE, "executions": 1_000_000}])
    patterns = [row["pattern"] for row in ranked]

    assert any(p.startswith("fsqrt fdiv") for p in patterns), patterns


def test_candidates_respect_the_encoding_budget():
    ranked = mining.mine_blocks(
        [{"instructions": NORMALIZE, "executions": 1_000_000}],
        max_inputs=2,
        max_outputs=1,
    )

    assert ranked
    for row in ranked:
        assert row["inputs"] <= 2, row
        assert row["outputs"] <= 1, row


def test_control_flow_is_never_part_of_a_candidate():
    """Fusing across a branch is not a question this can answer."""
    ranked = mining.mine_blocks([{"instructions": NORMALIZE, "executions": 10}])

    for row in ranked:
        assert "b.ne" not in row["mnemonics"]
        assert "ret" not in row["mnemonics"]


def test_a_non_convex_group_is_rejected():
    """If a value leaves the group and returns, one opcode cannot express it."""
    block = [
        "fmul s1, s0, s0",  # 0
        "fadd s2, s1, s1",  # 1  <- outside the group, between its members
        "fsub s3, s2, s1",  # 2
    ]
    decoded = mining.parse_block(block)
    graph = mining.build_dfg(decoded)
    descendants = mining._reachability(graph)

    assert mining.is_convex(frozenset({0, 1, 2}), descendants)
    assert not mining.is_convex(frozenset({0, 2}), descendants)


def test_the_same_shape_counts_together_whatever_registers_it_uses():
    first = ["fmul s3, s1, s1", "fadd s4, s3, s3"]
    second = ["fmul s9, s7, s7", "fadd s8, s9, s9"]

    ranked = mining.mine_blocks(
        [
            {"instructions": first, "executions": 100},
            {"instructions": second, "executions": 400},
        ]
    )
    fused = [r for r in ranked if r["mnemonics"] == ["fmul", "fadd"]]

    assert len(fused) == 1, "the same shape should be one candidate"
    assert fused[0]["static_occurrences"] == 2
    assert fused[0]["dynamic_occurrences"] == 500


def test_ranking_is_by_how_often_the_code_ran_not_how_often_it_appears():
    """A pattern in cold setup code is worth nothing, however often it is
    written; this is the difference between mining source and mining a run."""
    cold = ["fmul s3, s1, s1", "fadd s4, s3, s3"]
    hot = ["fsub s3, s1, s1", "fabs s4, s3"]

    ranked = mining.mine_blocks(
        [
            {"instructions": cold * 20, "executions": 2},
            {"instructions": hot, "executions": 5_000_000},
        ]
    )

    assert ranked[0]["mnemonics"] == ["fsub", "fabs"]
    assert ranked[0]["static_occurrences"] < ranked[1]["static_occurrences"]


def test_the_profilers_own_block_format_is_read():
    """`hot_blocks` puts its disassembly in `listing` and uses `instructions`
    for the instruction *count*, so reading that field gets an integer."""
    block = {
        "start": 0x4005A0,
        "instructions": 4,
        "executions": 2_048_000,
        "listing": [
            {"address": 1, "count": 2_048_000, "text": "fmul s3, s1, s1"},
            {"address": 2, "count": 2_048_000, "text": "fmadd s3, s2, s2, s3"},
            {"address": 3, "count": 2_048_000, "text": "fsqrt s4, s3"},
            {"address": 4, "count": 2_048_000, "text": "fdiv s5, s0, s4"},
        ],
    }

    assert len(mining.block_lines(block)) == 4
    ranked = mining.mine_blocks([block])
    assert ranked
    assert ranked[0]["dynamic_occurrences"] == 2_048_000


def test_a_block_too_short_to_fuse_yields_nothing():
    assert mining.mine_blocks([{"instructions": ["ret"], "executions": 10}]) == []
    assert mining.mine_blocks([]) == []


BLOCK = {
    "start": 0x4005A0,
    "instructions": 4,
    "executions": 65_536,
    "listing": [
        {"address": 1, "count": 65_536, "text": "fmul s3, s1, s1"},
        {"address": 2, "count": 65_536, "text": "fmadd s3, s2, s2, s3"},
        {"address": 3, "count": 65_536, "text": "fsqrt s4, s3"},
        {"address": 4, "count": 65_536, "text": "fdiv s5, s0, s4"},
    ],
}


def _context(state):
    from app.services.agent_tool_dispatch import AgentToolExecutionContext

    return AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id=None, job=None, state=state
    )


async def _call(params, state):
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    ctx = _context(state)
    provider = AutonomousAgentExecutor().tool_registry.resolve(
        "find_fusion_candidates", ctx
    )
    return await provider.execute("find_fusion_candidates", params, ctx)


def _profiled_state():
    return {
        "actions_taken": [
            {
                "action": {"tool": "profile_c_workload", "params": {}},
                "result": {"success": True, "data": {"hot_blocks": [BLOCK]}},
            }
        ]
    }


async def test_the_miner_picks_up_the_profile_this_run_already_produced():
    """Copying kilobytes of disassembly between two tool calls is work the run
    should not do by hand, and a truncated copy mines a different program."""
    result = await _call({}, _profiled_state())

    assert result["success"] is True
    assert result["data"]["candidates"], "should have mined the profiled blocks"
    assert result["data"]["candidates"][0]["dynamic_occurrences"] == 65_536


async def test_blocks_sent_as_json_text_are_parsed():
    """A model asked for a large structure sends it as text; a live run lost an
    iteration to 'field blocks should be array, got str'."""
    import json

    result = await _call({"blocks": json.dumps([BLOCK])}, {"actions_taken": []})

    assert result["success"] is True
    assert result["data"]["candidates"]


async def test_a_whole_profile_result_object_is_accepted():
    result = await _call({"blocks": {"hot_blocks": [BLOCK]}}, {"actions_taken": []})

    assert result["success"] is True
    assert result["data"]["candidates"]


async def test_with_no_blocks_and_no_profile_it_says_what_to_run():
    result = await _call({}, {"actions_taken": []})

    assert "error" in result
    assert "profile_c_workload" in result["error"]


async def test_prose_in_blocks_falls_back_to_the_profile():
    """A model that describes its blocks instead of omitting the field should
    still get the run's own profile mined, not a type error."""
    result = await _call(
        {"blocks": "the hot blocks from the profile above"}, _profiled_state()
    )

    assert result["success"] is True
    assert result["data"]["candidates"]


async def test_prose_with_no_profile_still_says_what_to_run():
    result = await _call({"blocks": "the hot blocks"}, {"actions_taken": []})

    assert "error" in result and "profile_c_workload" in result["error"]
