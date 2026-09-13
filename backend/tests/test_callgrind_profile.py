"""Reading a callgrind profile correctly.

Both compressions here were got wrong first time, and the failure was silent:
the numbers still looked like a profile, they were just orders of magnitude
off, which is the worst way for evidence to be wrong.
"""

from app.services import callgrind_profile as cg


def _profile(body: str):
    return cg.parse(body.strip("\n").splitlines())


def test_relative_positions_are_resolved_against_the_previous_line():
    profile = _profile(
        """
positions: instr line
events: Ir
fn=hot
0x1000 10 100
+4 11 200
+4 12 300
"""
    )

    assert profile.by_address == {0x1000: 100, 0x1004: 200, 0x1008: 300}
    assert profile.total == 600


def test_a_star_position_repeats_the_previous_address():
    profile = _profile(
        """
positions: instr line
events: Ir
fn=hot
0x2000 10 5
* 11 7
"""
    )

    assert profile.by_address == {0x2000: 12}


def test_negative_offsets_go_backwards():
    profile = _profile(
        """
positions: instr line
events: Ir
fn=hot
0x3010 10 1
-16 11 2
"""
    )

    assert profile.by_address == {0x3010: 1, 0x3000: 2}


def test_a_call_cost_is_not_charged_to_the_calling_instruction():
    """The line after calls= is the callee's inclusive cost."""
    profile = _profile(
        """
positions: instr line
events: Ir
fn=caller
0x1000 10 4
cfn=callee
calls=1 20
0x1000 10 999999
0x1004 11 6
"""
    )

    assert profile.by_address == {0x1000: 4, 0x1004: 6}
    assert profile.by_function["caller"] == 10


def test_costs_are_attributed_to_the_function_in_scope():
    profile = _profile(
        """
positions: instr line
events: Ir
fn=first
0x1000 1 10
fn=second
0x2000 1 90
"""
    )

    assert profile.hottest_functions() == [("second", 90), ("first", 10)]


def test_a_line_only_profile_still_totals_without_addresses():
    profile = _profile(
        """
positions: line
events: Ir
fn=hot
10 100
+1 200
"""
    )

    assert profile.by_address == {}
    assert profile.total == 300


def test_the_first_event_column_is_the_one_read():
    profile = _profile(
        """
positions: instr
events: Ir Dr Dw
fn=hot
0x1000 100 7 3
"""
    )

    assert profile.total == 100


def test_hot_blocks_group_contiguous_addresses_and_rank_by_cost():
    profile = cg.Profile(
        by_address={0x1000: 50, 0x1004: 50, 0x1008: 50, 0x2000: 10},
        by_function={},
    )
    listing = {0x1000: "ldr q0, [x1]", 0x1004: "fmla v0.4s", 0x1008: "b.ne .L1"}

    blocks = cg.hot_blocks(profile, listing)

    assert blocks[0]["start"] == 0x1000
    assert blocks[0]["instructions"] == 3
    assert blocks[0]["instruction_cost"] == 150
    # The block ran 50 times; summing its instructions would say 150.
    assert blocks[0]["executions"] == 50
    assert blocks[0]["listing"][1]["text"] == "fmla v0.4s"
    assert blocks[1]["start"] == 0x2000


def test_disassembly_parsing_maps_addresses_to_instructions():
    listing = cg.parse_disassembly(
        "00000000000008e8 <norm>:\n"
        "     8e8:\t3dc00020 \tldr\tq0, [x1]\n"
        "     8ec:\t4e20cc00 \tfmla\tv0.4s, v0.4s, v1.4s\n"
    )

    assert listing[0x8E8].startswith("ldr")
    assert "fmla" in listing[0x8EC]


def test_header_records_are_not_read_as_costs():
    """`pid: 703` parsed as a cost added 705 phantom instructions to 51M."""
    profile = _profile(
        """
version: 1
creator: callgrind-3.19.0
pid: 703
cmd:  /out/w
part: 1
positions: instr line
events: Ir
fn=hot
0x1000 10 100
"""
    )

    assert profile.total == 100


def test_compressed_function_names_are_resolved():
    profile = _profile(
        """
positions: instr line
events: Ir
fn=(443) norm
0x1000 1 90
fn=(444) acc
0x2000 1 10
fn=(443)
0x1004 2 5
"""
    )

    assert profile.hottest_functions() == [("norm", 95), ("acc", 10)]


def test_a_block_ends_where_the_execution_count_changes():
    """Every instruction in a basic block runs the same number of times."""
    profile = cg.Profile(
        by_address={0x1000: 1, 0x1004: 1, 0x1008: 2048000, 0x100C: 2048000},
        by_function={},
    )

    blocks = cg.hot_blocks(profile, {})

    hot = blocks[0]
    assert (hot["start"], hot["end"]) == (0x1008, 0x100C)
    assert hot["executions"] == 2048000
    assert hot["instructions"] == 2
    cold = [b for b in blocks if b["start"] == 0x1000][0]
    assert cold["executions"] == 1
