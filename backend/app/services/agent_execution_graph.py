"""Execution-graph analytics for the autonomous agent runtime.

The runtime records each act/verify/summarize step as a node and its
dependencies as edges. These functions read that graph and answer three
questions the loop needs during planning: what shape is it (stats), is it
healthy (health), and what should change (recommendations).

Pure functions over plain dicts — no services, no database. Extracted from
``autonomous_agent_executor`` with behaviour preserved exactly.
"""

from __future__ import annotations

import heapq
from typing import Any


def build_stats(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build compact DAG-style statistics for execution graph telemetry."""
    valid_nodes = [n for n in nodes if isinstance(n, dict)]
    valid_edges = [e for e in edges if isinstance(e, dict)]

    node_ids: set[str] = set()
    node_type_counts: dict[str, int] = {}
    blocked_nodes = 0
    successful_nodes = 0
    for row in valid_nodes:
        nid = str(row.get("id") or "").strip()
        if not nid:
            continue
        node_ids.add(nid)
        ntype = str(row.get("type") or "unknown").strip() or "unknown"
        node_type_counts[ntype] = int(node_type_counts.get(ntype, 0) or 0) + 1
        success = row.get("success")
        if success is True:
            successful_nodes += 1
        elif success is False:
            blocked_nodes += 1

    edge_type_counts: dict[str, int] = {}
    adj: dict[str, set[str]] = {}
    indeg: dict[str, int] = {}
    for nid in node_ids:
        adj[nid] = set()
        indeg[nid] = 0

    for edge in valid_edges:
        src = str(edge.get("from") or "").strip()
        dst = str(edge.get("to") or "").strip()
        if not src or not dst or src == dst:
            continue
        etype = str(edge.get("type") or "edge").strip() or "edge"
        edge_type_counts[etype] = int(edge_type_counts.get(etype, 0) or 0) + 1
        if src not in adj:
            adj[src] = set()
            indeg[src] = indeg.get(src, 0)
        if dst not in adj:
            adj[dst] = set()
            indeg[dst] = indeg.get(dst, 0)
        node_ids.add(src)
        node_ids.add(dst)
        if dst not in adj[src]:
            adj[src].add(dst)
            indeg[dst] = int(indeg.get(dst, 0) or 0) + 1

    roots = [nid for nid in node_ids if int(indeg.get(nid, 0) or 0) == 0]
    leaves = [nid for nid in node_ids if len(adj.get(nid, set())) == 0]
    orphans = [
        nid
        for nid in node_ids
        if int(indeg.get(nid, 0) or 0) == 0 and len(adj.get(nid, set())) == 0
    ]

    # Kahn topological traversal for cycle detection and longest path estimate.
    # A heap keeps the deterministic name ordering the previous list-and-resort
    # loop produced, without its quadratic pop(0)/sort on every step.
    indeg_work = {k: int(v or 0) for k, v in indeg.items()}
    queue = [nid for nid in node_ids if indeg_work.get(nid, 0) == 0]
    heapq.heapify(queue)
    topo: list[str] = []
    while queue:
        cur = heapq.heappop(queue)
        topo.append(cur)
        for nxt in sorted(adj.get(cur, set())):
            indeg_work[nxt] = int(indeg_work.get(nxt, 0) or 0) - 1
            if indeg_work[nxt] == 0:
                heapq.heappush(queue, nxt)

    has_cycle = len(topo) != len(node_ids)
    critical_path_length = 0
    if not has_cycle and node_ids:
        dist: dict[str, int] = {}
        for nid in topo:
            base = dist.get(nid, 1)
            dist[nid] = max(1, base)
            for nxt in adj.get(nid, set()):
                dist[nxt] = max(int(dist.get(nxt, 1) or 1), int(dist[nid] or 1) + 1)
        critical_path_length = max(dist.values()) if dist else 1

    return {
        "total_nodes": len(node_ids),
        "total_edges": sum(len(v) for v in adj.values()),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
        "root_nodes": len(roots),
        "leaf_nodes": len(leaves),
        "orphan_nodes": len(orphans),
        "blocked_nodes": int(blocked_nodes),
        "successful_nodes": int(successful_nodes),
        "has_cycle": bool(has_cycle),
        "critical_path_length": int(critical_path_length),
    }


def build_health(dag_stats: dict[str, Any]) -> dict[str, Any]:
    """Classify graph runtime quality into compact UI-friendly health status."""
    if not isinstance(dag_stats, dict):
        return {
            "status": "unknown",
            "reasons": ["missing_dag_stats"],
            "severity_score": 0,
        }

    total_nodes = max(0, int(dag_stats.get("total_nodes", 0) or 0))
    blocked_nodes = max(0, int(dag_stats.get("blocked_nodes", 0) or 0))
    has_cycle = bool(dag_stats.get("has_cycle", False))
    critical_path = max(0, int(dag_stats.get("critical_path_length", 0) or 0))
    orphan_nodes = max(0, int(dag_stats.get("orphan_nodes", 0) or 0))

    blocked_ratio = (
        (float(blocked_nodes) / float(total_nodes)) if total_nodes > 0 else 0.0
    )
    reasons: list[str] = []
    severity = 0

    if has_cycle:
        reasons.append("cycle_detected")
        severity += 80
    if blocked_ratio >= 0.5 and blocked_nodes >= 2:
        reasons.append("high_blocked_ratio")
        severity += 35
    elif blocked_ratio >= 0.25 and blocked_nodes >= 1:
        reasons.append("moderate_blocked_ratio")
        severity += 20

    if critical_path >= 20:
        reasons.append("long_critical_path")
        severity += 20
    elif critical_path >= 12:
        reasons.append("moderate_critical_path")
        severity += 10

    if orphan_nodes >= 3:
        reasons.append("orphan_nodes_detected")
        severity += 10

    if total_nodes <= 0:
        reasons.append("empty_graph")
        status = "unknown"
        severity = max(severity, 5)
    elif has_cycle or severity >= 70:
        status = "critical"
    elif severity >= 20:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "reasons": reasons,
        "severity_score": min(100, max(0, int(severity))),
        "blocked_ratio": round(blocked_ratio, 4),
    }


def build_recommendations(health: dict[str, Any]) -> list[str]:
    """Create short remediation hints based on graph health signals."""
    if not isinstance(health, dict):
        return []
    status = str(health.get("status") or "").strip().lower()
    reasons = [str(x).strip() for x in (health.get("reasons") or []) if str(x).strip()]
    recs: list[str] = []

    if status == "unknown":
        recs.append(
            "Collect at least one act->verify->summarize cycle to initialize graph diagnostics."
        )

    if "cycle_detected" in reasons:
        recs.append(
            "Reset or re-plan execution steps to remove cyclic dependencies between nodes."
        )
        recs.append(
            "Pin deterministic step_id ordering and avoid referencing future steps in depends_on."
        )

    if "high_blocked_ratio" in reasons or "moderate_blocked_ratio" in reasons:
        recs.append(
            "Review failed/blocked nodes and tighten tool params before retrying affected steps."
        )
        recs.append(
            "Enable scoped recovery actions to gather missing evidence before write operations."
        )

    if "long_critical_path" in reasons or "moderate_critical_path" in reasons:
        recs.append(
            "Split large plan steps into smaller nodes to shorten the critical path."
        )

    if "orphan_nodes_detected" in reasons:
        recs.append("Attach orphan nodes to explicit predecessors using depends_on.")

    if not recs and status == "ok":
        recs.append("Graph health is stable; continue with current execution strategy.")
    if not recs:
        recs.append(
            "Inspect execution_graph.nodes and execution_graph.edges for anomalies."
        )

    deduped: list[str] = []
    for r in recs:
        if r not in deduped:
            deduped.append(r)
    return deduped[:6]


def build_runtime_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Build live execution-graph diagnostics for in-loop observation and planning."""
    nodes = (
        state.get("execution_graph_nodes")
        if isinstance(state.get("execution_graph_nodes"), list)
        else []
    )
    edges = (
        state.get("execution_graph_edges")
        if isinstance(state.get("execution_graph_edges"), list)
        else []
    )
    dag_stats = build_stats(nodes, edges)
    health = build_health(dag_stats)
    recommendations = build_recommendations(health)
    return {
        "verification_attempts": int(state.get("verification_attempts", 0) or 0),
        "verification_successes": int(state.get("verification_successes", 0) or 0),
        "summarization_attempts": int(state.get("summarization_attempts", 0) or 0),
        "summarization_successes": int(state.get("summarization_successes", 0) or 0),
        "dag_stats": dag_stats,
        "graph_health": health,
        "recommended_actions": recommendations,
    }


def has_recovery_pressure(
    state: dict[str, Any],
    *,
    verification_debt_threshold: int = 2,
    severity_threshold: int = 20,
) -> bool:
    """Return whether live execution-graph health indicates rescue/recovery pressure."""
    runtime = build_runtime_snapshot(state)
    graph_health = (
        runtime.get("graph_health")
        if isinstance(runtime.get("graph_health"), dict)
        else {}
    )
    verification_attempts = int(runtime.get("verification_attempts", 0) or 0)
    verification_successes = int(runtime.get("verification_successes", 0) or 0)
    verification_debt = max(0, verification_attempts - verification_successes)
    graph_severity = int(graph_health.get("severity_score", 0) or 0)
    return verification_debt >= max(
        1, int(verification_debt_threshold or 2)
    ) or graph_severity >= max(1, int(severity_threshold or 20))
