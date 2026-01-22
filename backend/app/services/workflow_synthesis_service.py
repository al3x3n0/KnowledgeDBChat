"""
Workflow synthesis service.

Generates workflow drafts from natural language descriptions using the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import UserPreferences
from app.models.workflow import UserTool
from app.schemas.workflow import WorkflowCreate
from app.services.agent_tools import AGENT_TOOLS
from app.services.llm_service import LLMService, UserLLMSettings


ALLOWED_NODE_TYPES = {"start", "end", "tool", "condition", "parallel", "loop", "wait"}


@dataclass
class ToolCatalog:
    builtin_tools: List[Dict[str, Any]]
    custom_tools: List[UserTool]


class WorkflowSynthesisService:
    """Synthesize workflow definitions from descriptions."""

    async def synthesize(
        self,
        *,
        description: str,
        name: Optional[str],
        trigger_config: Optional[Dict[str, Any]],
        is_active: Optional[bool],
        user_id,
        db: AsyncSession,
    ) -> Tuple[WorkflowCreate, List[str]]:
        catalog = await self._load_tool_catalog(db, user_id)
        prompt = self._build_prompt(
            description=description,
            name=name,
            trigger_config=trigger_config,
            catalog=catalog,
        )

        llm_service = LLMService()
        user_settings = await self._load_user_settings(db, user_id)

        response_text = await llm_service.generate_response(
            query=prompt,
            user_settings=user_settings,
            task_type="workflow_synthesis",
        )

        try:
            raw_data = self._extract_json(response_text)
        except Exception as exc:
            logger.error(f"Workflow synthesis JSON parse failed: {exc}")
            raise ValueError("LLM response did not contain valid JSON") from exc

        normalized, warnings = self._normalize_workflow(
            raw_data,
            catalog,
            fallback_name=name,
            fallback_description=description,
            fallback_trigger=trigger_config,
            fallback_is_active=is_active,
        )

        try:
            workflow = WorkflowCreate.model_validate(normalized)
        except ValidationError as exc:
            logger.error(f"Workflow synthesis validation error: {exc}")
            raise ValueError("Synthesized workflow failed validation") from exc
        return workflow, warnings

    async def _load_user_settings(self, db: AsyncSession, user_id) -> Optional[UserLLMSettings]:
        try:
            prefs_result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            user_prefs = prefs_result.scalar_one_or_none()
            if user_prefs:
                return UserLLMSettings.from_preferences(user_prefs)
        except Exception as exc:
            logger.warning(f"Could not load user LLM preferences: {exc}")
        return None

    async def _load_tool_catalog(self, db: AsyncSession, user_id) -> ToolCatalog:
        result = await db.execute(
            select(UserTool).where(UserTool.user_id == user_id)
        )
        custom_tools = result.scalars().all()
        builtin_tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            for tool in AGENT_TOOLS
        ]
        return ToolCatalog(builtin_tools=builtin_tools, custom_tools=custom_tools)

    def _build_prompt(
        self,
        *,
        description: str,
        name: Optional[str],
        trigger_config: Optional[Dict[str, Any]],
        catalog: ToolCatalog,
    ) -> str:
        builtin_json = json.dumps(catalog.builtin_tools, ensure_ascii=True, indent=2)
        custom_json = json.dumps(
            [
                {
                    "id": str(tool.id),
                    "name": tool.name,
                    "description": tool.description,
                    "tool_type": tool.tool_type,
                    "parameters_schema": tool.parameters_schema or {},
                }
                for tool in catalog.custom_tools
            ],
            ensure_ascii=True,
            indent=2,
        )

        preferred_name = name or ""
        trigger_hint = json.dumps(trigger_config or {"type": "manual"}, ensure_ascii=True)

        return (
            "You are a workflow synthesis engine. "
            "Create a workflow definition from the user description. "
            "Output ONLY a JSON object with the following shape:\n"
            "{\n"
            '  "name": string,\n'
            '  "description": string,\n'
            '  "is_active": boolean,\n'
            '  "trigger_config": { "type": "manual|schedule|event|webhook", ... },\n'
            '  "nodes": [\n'
            "    {\n"
            '      "node_id": string,\n'
            '      "node_type": "start|end|tool|condition|parallel|loop|wait",\n'
            '      "tool_id": string|null,\n'
            '      "builtin_tool": string|null,\n'
            '      "config": {\n'
            '        "input_mapping": { "param": "value or {{context.path}}" },\n'
            '        "output_key": "string",\n'
            '        "condition": { "type": "equals|not_equals|greater_than|less_than|contains|truthy|expression", "left": "...", "right": "..." },\n'
            '        "loop_source": "{{context.path}}",\n'
            '        "max_iterations": 50,\n'
            '        "wait_seconds": 30\n'
            "      },\n"
            '      "position_x": integer,\n'
            '      "position_y": integer\n'
            "    }\n"
            "  ],\n"
            '  "edges": [\n'
            "    {\n"
            '      "source_node_id": string,\n'
            '      "target_node_id": string,\n'
            '      "source_handle": "true|false|null",\n'
            '      "condition": object|null\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Include exactly one start node and at least one end node.\n"
            "- Use tool nodes for actions; set builtin_tool to a name from the builtin tools list.\n"
            "- Use tool_id only for custom tools and only if provided in the custom tools list.\n"
            "- Keep node_id unique and under 50 characters.\n"
            "- Use {{context.trigger_data}} for trigger inputs when needed.\n"
            "- For condition nodes, connect outgoing edges with source_handle 'true' and 'false'.\n"
            "- Keep the graph small and practical (roughly 4-10 nodes).\n"
            "- Output JSON only, no markdown or explanations.\n\n"
            f"Preferred name (if suitable): {preferred_name}\n"
            f"Trigger hint: {trigger_hint}\n\n"
            f"USER DESCRIPTION:\n{description}\n\n"
            f"BUILTIN TOOLS:\n{builtin_json}\n\n"
            f"CUSTOM TOOLS:\n{custom_json}\n"
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in response")

        payload = cleaned[start:end + 1]
        return json.loads(payload)

    def _normalize_workflow(
        self,
        data: Dict[str, Any],
        catalog: ToolCatalog,
        *,
        fallback_name: Optional[str],
        fallback_description: str,
        fallback_trigger: Optional[Dict[str, Any]],
        fallback_is_active: Optional[bool],
    ) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        builtin_names = {tool["name"] for tool in catalog.builtin_tools}
        custom_by_name = {tool.name.lower(): tool for tool in catalog.custom_tools}
        custom_by_id = {str(tool.id): tool for tool in catalog.custom_tools}

        is_active = data.get("is_active")
        if is_active is None:
            is_active = fallback_is_active if fallback_is_active is not None else True

        normalized: Dict[str, Any] = {
            "name": (data.get("name") or fallback_name or "Generated Workflow").strip(),
            "description": (data.get("description") or fallback_description or "").strip(),
            "is_active": is_active,
            "trigger_config": data.get("trigger_config") or fallback_trigger or {"type": "manual"},
            "nodes": [],
            "edges": [],
        }

        nodes_input = data.get("nodes") or []
        edges_input = data.get("edges") or []

        seen_ids = set()
        for idx, node in enumerate(nodes_input):
            node_id_raw = node.get("node_id") or node.get("id") or f"node_{idx + 1}"
            node_id = self._sanitize_node_id(str(node_id_raw))
            if node_id in seen_ids:
                suffix = 2
                new_id = f"{node_id}_{suffix}"
                while new_id in seen_ids:
                    suffix += 1
                    new_id = f"{node_id}_{suffix}"
                node_id = new_id
                warnings.append(f"Duplicate node_id '{node_id_raw}' renamed to '{node_id}'.")
            seen_ids.add(node_id)

            node_type = node.get("node_type") or node.get("type") or "tool"
            if node_type not in ALLOWED_NODE_TYPES:
                warnings.append(f"Unknown node_type '{node_type}' set to 'tool'.")
                node_type = "tool"

            config = node.get("config") or {}
            config = self._normalize_config(node, config)

            tool_id = node.get("tool_id")
            builtin_tool = node.get("builtin_tool")
            tool_name = node.get("tool_name") or node.get("tool")

            if node_type == "tool":
                if tool_name and not builtin_tool and not tool_id:
                    if tool_name in builtin_names:
                        builtin_tool = tool_name
                    else:
                        custom_match = custom_by_name.get(str(tool_name).lower())
                        if custom_match:
                            tool_id = str(custom_match.id)

                if builtin_tool and builtin_tool not in builtin_names:
                    warnings.append(f"Unknown builtin tool '{builtin_tool}' on node '{node_id}'.")
                    builtin_tool = None

                if tool_id and str(tool_id) not in custom_by_id:
                    warnings.append(f"Unknown custom tool id '{tool_id}' on node '{node_id}'.")
                    tool_id = None

                if not builtin_tool and not tool_id:
                    warnings.append(f"Tool node '{node_id}' has no valid tool reference.")

            normalized["nodes"].append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "tool_id": tool_id,
                    "builtin_tool": builtin_tool,
                    "config": config or {},
                    "position_x": node.get("position_x"),
                    "position_y": node.get("position_y"),
                }
            )

        edges = []
        for edge in edges_input:
            source = edge.get("source_node_id") or edge.get("source")
            target = edge.get("target_node_id") or edge.get("target")
            if not source or not target:
                warnings.append("Skipped edge with missing source or target.")
                continue
            source_handle = edge.get("source_handle") or edge.get("sourceHandle")
            edges.append(
                {
                    "source_node_id": source,
                    "target_node_id": target,
                    "source_handle": source_handle,
                    "condition": edge.get("condition"),
                }
            )

        normalized["edges"] = edges

        self._ensure_start_end_nodes(normalized, warnings)
        self._prune_invalid_edges(normalized, warnings)
        self._auto_connect_orphans(normalized, warnings)
        self._assign_positions(normalized)

        return normalized, warnings

    def _sanitize_node_id(self, node_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", node_id.strip())
        cleaned = cleaned.strip("_") or "node"
        return cleaned[:50]

    def _normalize_config(self, node: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(config)

        if "inputMapping" in normalized and "input_mapping" not in normalized:
            normalized["input_mapping"] = normalized.pop("inputMapping")
        if "outputKey" in normalized and "output_key" not in normalized:
            normalized["output_key"] = normalized.pop("outputKey")
        if "loopSource" in normalized and "loop_source" not in normalized:
            normalized["loop_source"] = normalized.pop("loopSource")
        if "maxIterations" in normalized and "max_iterations" not in normalized:
            normalized["max_iterations"] = normalized.pop("maxIterations")
        if "waitSeconds" in normalized and "wait_seconds" not in normalized:
            normalized["wait_seconds"] = normalized.pop("waitSeconds")

        if "input_mapping" not in normalized and "inputs" in node:
            normalized["input_mapping"] = node.get("inputs")
        if "condition" not in normalized and "condition" in node:
            normalized["condition"] = node.get("condition")

        return normalized

    def _ensure_start_end_nodes(self, normalized: Dict[str, Any], warnings: List[str]) -> None:
        nodes = normalized["nodes"]
        node_ids = {n["node_id"] for n in nodes}
        has_start = any(n["node_type"] == "start" for n in nodes)
        has_end = any(n["node_type"] == "end" for n in nodes)

        if not has_start:
            start_id = "start"
            if start_id in node_ids:
                start_id = "start_node"
            nodes.insert(
                0,
                {
                    "node_id": start_id,
                    "node_type": "start",
                    "tool_id": None,
                    "builtin_tool": None,
                    "config": {},
                    "position_x": None,
                    "position_y": None,
                },
            )
            warnings.append("Added missing start node.")

        if not has_end:
            end_id = "end"
            if end_id in node_ids:
                end_id = "end_node"
            nodes.append(
                {
                    "node_id": end_id,
                    "node_type": "end",
                    "tool_id": None,
                    "builtin_tool": None,
                    "config": {},
                    "position_x": None,
                    "position_y": None,
                },
            )
            warnings.append("Added missing end node.")

    def _prune_invalid_edges(self, normalized: Dict[str, Any], warnings: List[str]) -> None:
        node_ids = {n["node_id"] for n in normalized["nodes"]}
        valid_edges = []
        for edge in normalized["edges"]:
            if edge["source_node_id"] not in node_ids or edge["target_node_id"] not in node_ids:
                warnings.append(
                    f"Removed edge from '{edge['source_node_id']}' to '{edge['target_node_id']}' (unknown node)."
                )
                continue
            valid_edges.append(edge)
        normalized["edges"] = valid_edges

    def _auto_connect_orphans(self, normalized: Dict[str, Any], warnings: List[str]) -> None:
        nodes = normalized["nodes"]
        edges = normalized["edges"]
        edge_keys = {(e["source_node_id"], e["target_node_id"], e.get("source_handle")) for e in edges}

        if not edges and len(nodes) >= 2:
            ordered = self._ordered_nodes(nodes)
            for prev_node, next_node in zip(ordered, ordered[1:]):
                key = (prev_node["node_id"], next_node["node_id"], None)
                if key not in edge_keys:
                    edges.append(
                        {
                            "source_node_id": prev_node["node_id"],
                            "target_node_id": next_node["node_id"],
                            "source_handle": None,
                            "condition": None,
                        }
                    )
            warnings.append("Generated linear edges between nodes.")
            return

        incoming = {n["node_id"]: 0 for n in nodes}
        outgoing = {n["node_id"]: 0 for n in nodes}
        for edge in edges:
            incoming[edge["target_node_id"]] += 1
            outgoing[edge["source_node_id"]] += 1

        start_nodes = [n for n in nodes if n["node_type"] == "start"]
        end_nodes = [n for n in nodes if n["node_type"] == "end"]
        start_id = start_nodes[0]["node_id"] if start_nodes else None
        end_id = end_nodes[0]["node_id"] if end_nodes else None

        if start_id:
            roots = [
                n["node_id"]
                for n in nodes
                if n["node_id"] != start_id and incoming.get(n["node_id"], 0) == 0
            ]
            for root in roots:
                key = (start_id, root, None)
                if key not in edge_keys:
                    edges.append(
                        {
                            "source_node_id": start_id,
                            "target_node_id": root,
                            "source_handle": None,
                            "condition": None,
                        }
                    )
                    edge_keys.add(key)
                    warnings.append(f"Connected start node to '{root}'.")

        if end_id:
            sinks = [
                n["node_id"]
                for n in nodes
                if n["node_id"] != end_id and outgoing.get(n["node_id"], 0) == 0
            ]
            for sink in sinks:
                key = (sink, end_id, None)
                if key not in edge_keys:
                    edges.append(
                        {
                            "source_node_id": sink,
                            "target_node_id": end_id,
                            "source_handle": None,
                            "condition": None,
                        }
                    )
                    edge_keys.add(key)
                    warnings.append(f"Connected '{sink}' to end node.")

    def _ordered_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        starts = [n for n in nodes if n["node_type"] == "start"]
        ends = [n for n in nodes if n["node_type"] == "end"]
        middle = [n for n in nodes if n["node_type"] not in ("start", "end")]
        return starts + middle + ends

    def _assign_positions(self, normalized: Dict[str, Any]) -> None:
        nodes = normalized["nodes"]
        ordered = self._ordered_nodes(nodes)
        x = 250
        y = 50
        gap = 160

        for index, node in enumerate(ordered):
            if node.get("position_x") is None:
                node["position_x"] = x
            if node.get("position_y") is None:
                node["position_y"] = y + index * gap
