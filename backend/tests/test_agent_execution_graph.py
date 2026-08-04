"""Tests for execution-graph analytics.

The graph drives replanning and recovery decisions, so the properties that
matter are structural: a cycle must be detected rather than silently producing a
bogus critical path, and health must escalate on evidence rather than on noise.
"""

import random

from app.services import agent_execution_graph as graph


def _chain(length: int) -> tuple[list[dict], list[dict]]:
    nodes = [{"id": f"n{i}"} for i in range(length)]
    edges = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(length - 1)]
    return nodes, edges


def test_stats_on_an_empty_graph():
    stats = graph.build_stats([], [])
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0
    assert stats["has_cycle"] is False
    assert stats["critical_path_length"] == 0


def test_stats_count_roots_leaves_and_orphans():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "lonely"}]
    edges = [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]
    stats = graph.build_stats(nodes, edges)
    assert stats["total_nodes"] == 4
    assert stats["total_edges"] == 2
    assert stats["root_nodes"] == 2  # "a" and the orphan
    assert stats["leaf_nodes"] == 2  # "c" and the orphan
    assert stats["orphan_nodes"] == 1
    assert stats["critical_path_length"] == 3


def test_stats_detect_a_cycle_and_refuse_to_report_a_critical_path():
    nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    edges = [
        {"from": "a", "to": "b"},
        {"from": "b", "to": "c"},
        {"from": "c", "to": "a"},
    ]
    stats = graph.build_stats(nodes, edges)
    assert stats["has_cycle"] is True
    # A cyclic graph has no longest path; reporting one would be a lie.
    assert stats["critical_path_length"] == 0


def test_stats_ignore_self_edges_and_malformed_rows():
    nodes = [{"id": "a"}, {"id": ""}, "not a dict", {"id": "b"}]
    edges = [
        {"from": "a", "to": "a"},
        {"from": "", "to": "b"},
        "not a dict",
        {"from": "a", "to": "b"},
    ]
    stats = graph.build_stats(nodes, edges)
    assert stats["total_nodes"] == 2
    assert stats["total_edges"] == 1


def test_stats_deduplicate_repeated_edges():
    nodes = [{"id": "a"}, {"id": "b"}]
    edges = [{"from": "a", "to": "b"}] * 5
    assert graph.build_stats(nodes, edges)["total_edges"] == 1


def test_stats_track_node_outcomes_by_success_flag():
    nodes = [
        {"id": "a", "success": True, "type": "act"},
        {"id": "b", "success": False, "type": "verify"},
        {"id": "c", "type": "verify"},
    ]
    stats = graph.build_stats(nodes, [])
    assert stats["successful_nodes"] == 1
    assert stats["blocked_nodes"] == 1  # an absent flag is neither
    assert stats["node_type_counts"] == {"act": 1, "verify": 2}


def test_health_treats_a_cycle_as_critical():
    stats = graph.build_stats(
        [{"id": "a"}, {"id": "b"}],
        [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
    )
    health = graph.build_health(stats)
    assert health["status"] == "critical"
    assert "cycle_detected" in health["reasons"]
    assert health["severity_score"] >= 80


def test_health_escalates_with_the_blocked_ratio():
    def health_for(blocked: int, total: int) -> dict:
        nodes = [{"id": f"n{i}", "success": i >= blocked} for i in range(total)]
        return graph.build_health(graph.build_stats(nodes, []))

    assert health_for(0, 4)["status"] == "ok"
    assert "moderate_blocked_ratio" in health_for(1, 4)["reasons"]
    assert "high_blocked_ratio" in health_for(3, 4)["reasons"]


def test_health_flags_a_long_critical_path():
    long_chain = graph.build_health(graph.build_stats(*_chain(25)))
    medium_chain = graph.build_health(graph.build_stats(*_chain(14)))
    assert "long_critical_path" in long_chain["reasons"]
    assert "moderate_critical_path" in medium_chain["reasons"]


def test_health_is_unknown_for_an_empty_or_unusable_graph():
    empty = graph.build_health(graph.build_stats([], []))
    assert empty["status"] == "unknown"
    assert "empty_graph" in empty["reasons"]
    assert graph.build_health("junk")["status"] == "unknown"
    assert graph.build_health({})["status"] == "unknown"


def test_severity_is_clamped_to_a_hundred():
    nodes = [{"id": f"n{i}", "success": False} for i in range(30)]
    edges = [{"from": f"n{i}", "to": f"n{i + 1}"} for i in range(29)]
    edges.append({"from": "n29", "to": "n0"})  # close the loop
    health = graph.build_health(graph.build_stats(nodes, edges))
    assert health["severity_score"] == 100


def test_recommendations_are_specific_deduplicated_and_bounded():
    cyclic = graph.build_health(
        graph.build_stats(
            [{"id": "a"}, {"id": "b"}],
            [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
        )
    )
    recommendations = graph.build_recommendations(cyclic)
    assert any("cyclic dependencies" in rec for rec in recommendations)
    assert len(recommendations) == len(set(recommendations))
    assert len(recommendations) <= 6


def test_recommendations_always_say_something_actionable():
    for health in ({}, {"status": "ok"}, {"status": "unknown"}, "junk"):
        recommendations = graph.build_recommendations(health)
        if health == "junk":
            assert recommendations == []
        else:
            assert recommendations, "an operator must never see an empty hint list"


def test_runtime_snapshot_combines_counters_with_graph_diagnostics():
    snapshot = graph.build_runtime_snapshot(
        {
            "execution_graph_nodes": [{"id": "a"}, {"id": "b"}],
            "execution_graph_edges": [{"from": "a", "to": "b"}],
            "verification_attempts": 4,
            "verification_successes": 1,
            "summarization_attempts": 2,
            "summarization_successes": 2,
        }
    )
    assert snapshot["verification_attempts"] == 4
    assert snapshot["dag_stats"]["total_nodes"] == 2
    assert snapshot["graph_health"]["status"] == "ok"
    assert snapshot["recommended_actions"]


def test_runtime_snapshot_tolerates_missing_graph_state():
    snapshot = graph.build_runtime_snapshot({})
    assert snapshot["dag_stats"]["total_nodes"] == 0
    assert snapshot["graph_health"]["status"] == "unknown"


def test_recovery_pressure_triggers_on_verification_debt():
    assert (
        graph.has_recovery_pressure(
            {"verification_attempts": 4, "verification_successes": 1}
        )
        is True
    )
    assert (
        graph.has_recovery_pressure(
            {"verification_attempts": 1, "verification_successes": 1}
        )
        is False
    )


def test_recovery_pressure_triggers_on_graph_severity_alone():
    # No verification debt at all, but the graph itself is unhealthy.
    assert (
        graph.has_recovery_pressure(
            {
                "execution_graph_nodes": [{"id": "a"}, {"id": "b"}],
                "execution_graph_edges": [
                    {"from": "a", "to": "b"},
                    {"from": "b", "to": "a"},
                ],
            }
        )
        is True
    )


def test_critical_path_matches_a_brute_force_longest_path_on_random_dags():
    """Property check: the DP result must equal an independent computation."""
    for seed in range(25):
        rng = random.Random(seed)
        size = rng.randint(2, 14)
        nodes = [{"id": f"n{i}"} for i in range(size)]
        # Edges only ever point forward, so the graph is acyclic by construction.
        edges = [
            {"from": f"n{i}", "to": f"n{j}"}
            for i in range(size)
            for j in range(i + 1, size)
            if rng.random() < 0.3
        ]
        stats = graph.build_stats(nodes, edges)
        assert stats["has_cycle"] is False

        successors: dict[str, list[str]] = {f"n{i}": [] for i in range(size)}
        for edge in edges:
            successors[edge["from"]].append(edge["to"])

        memo: dict[str, int] = {}

        def longest_from(node: str) -> int:
            if node not in memo:
                memo[node] = 1 + max(
                    (longest_from(nxt) for nxt in successors[node]), default=0
                )
            return memo[node]

        expected = max(longest_from(f"n{i}") for i in range(size))
        assert stats["critical_path_length"] == expected, f"seed {seed}"
