"""Routing primitives extracted from the app service layer."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .types import AgentSpec, agent_spec_from_model

CAPABILITY_KEYWORDS = {
    "paper_search": [
        "arxiv",
        "paper",
        "papers",
        "preprint",
        "publication",
        "literature",
        "scientific",
        "research paper",
        "find papers",
        "search papers",
    ],
    "document_search": [
        "search",
        "find",
        "look for",
        "locate",
        "where is",
        "documents about",
        "files about",
        "what documents",
        "find documents",
    ],
    "document_crud": [
        "delete",
        "remove",
        "create document",
        "upload",
        "add document",
        "edit document",
        "update document",
    ],
    "document_compare": ["compare", "difference", "similar", "diff between", "versus"],
    "tag_management": ["tag", "tags", "categorize", "label", "organize"],
    "rag_qa": [
        "what is",
        "how does",
        "explain",
        "why",
        "when",
        "who",
        "tell me about",
        "describe",
        "what do you know",
        "based on",
    ],
    "summarization": [
        "summarize",
        "summary",
        "tldr",
        "overview",
        "key points",
        "main points",
    ],
    "knowledge_synthesis": [
        "analyze",
        "insight",
        "pattern",
        "trend",
        "relationship between",
        "connection",
        "how are",
        "related",
    ],
    "workflow_exec": ["workflow", "automate", "run workflow", "execute workflow"],
    "template_fill": [
        "template",
        "fill template",
        "generate from template",
        "document template",
    ],
    "diagram_gen": [
        "diagram",
        "chart",
        "visualization",
        "visualize",
        "draw",
        "mermaid",
    ],
    "automation": ["automate", "schedule", "recurring", "batch process"],
    "code_analysis": [
        "code",
        "function",
        "class",
        "method",
        "implementation",
        "algorithm",
        "logic",
        "debug",
        "refactor",
        "code structure",
        "code pattern",
        "code review",
    ],
    "code_explanation": [
        "explain code",
        "how does this code",
        "what does this code",
        "understand code",
        "walk through code",
        "code walkthrough",
    ],
}


class AgentRouter:
    """Routes user requests to the best matching agent."""

    def __init__(self, llm_service: Optional[Any] = None):
        self.llm_service = llm_service
        self._agent_cache: Dict[str, Any] = {}

    def set_agents(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        self._agent_cache = dict(agents)
        return self._agent_cache

    def get_agents(self) -> Dict[str, Any]:
        return self._agent_cache

    async def analyze_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        message_lower = message.lower()
        capabilities_found: List[str] = []
        confidence = 0.0

        for capability, keywords in CAPABILITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if capability not in capabilities_found:
                        capabilities_found.append(capability)
                    confidence = max(confidence, 0.7)

        if confidence >= 0.7 and capabilities_found:
            return {
                "intents": capabilities_found,
                "capabilities_needed": capabilities_found,
                "confidence": confidence,
                "method": "keyword",
            }

        if use_llm and confidence < 0.7 and self.llm_service is not None:
            try:
                llm_result = await self._analyze_intent_with_llm(message, history)
                if llm_result and llm_result.get("capabilities_needed"):
                    return {**llm_result, "method": "llm"}
            except Exception as exc:
                logger.warning(f"LLM intent analysis failed, using fallback: {exc}")

        if not capabilities_found:
            capabilities_found = ["general"]
            confidence = 0.3

        return {
            "intents": capabilities_found,
            "capabilities_needed": capabilities_found,
            "confidence": confidence,
            "method": "fallback",
        }

    async def _analyze_intent_with_llm(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        available_capabilities = list(CAPABILITY_KEYWORDS.keys()) + ["general"]
        prompt = f"""Analyze the user's message and determine what capabilities are needed to handle it.

Available capabilities:
- document_search: Finding and searching documents
- document_crud: Creating, updating, deleting documents
- document_compare: Comparing documents
- tag_management: Managing document tags and categories
- rag_qa: Answering questions using knowledge base (RAG)
- summarization: Summarizing content
- knowledge_synthesis: Analyzing patterns and relationships
- workflow_exec: Running automated workflows
- template_fill: Filling document templates
- diagram_gen: Generating diagrams and visualizations
- automation: Scheduling and batch processing
- code_analysis: Analyzing code structure, patterns, and architecture
- code_explanation: Explaining code logic and functionality
- general: General assistance (fallback)

User message: "{message}"

Respond with a JSON object:
{{
    "capabilities_needed": ["capability1", "capability2"],
    "confidence": 0.8,
    "reasoning": "Brief explanation"
}}

Only include capabilities that are clearly needed. If unsure, include "general"."""

        response = await self.llm_service.generate_text(
            prompt=prompt,
            system_prompt="You are an intent classifier. Respond only with valid JSON.",
            temperature=0.1,
            max_tokens=200,
        )
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            return None
        result = json.loads(json_match.group())
        valid_caps = [
            c
            for c in result.get("capabilities_needed", [])
            if c in available_capabilities
        ]
        if not valid_caps:
            return None
        return {
            "capabilities_needed": valid_caps,
            "confidence": result.get("confidence", 0.6),
            "reasoning": result.get("reasoning", ""),
        }

    async def select_agent(
        self,
        intent_analysis: Dict[str, Any],
        available_agents: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, str]:
        agents = available_agents or self._agent_cache
        if not agents:
            raise ValueError("No agents available for routing")

        capabilities_needed = intent_analysis.get("capabilities_needed", ["general"])
        agent_scores: List[Tuple[Any, float, str]] = []

        for agent in agents.values():
            spec = (
                agent if isinstance(agent, AgentSpec) else agent_spec_from_model(agent)
            )
            if not spec.is_active:
                continue

            agent_caps = set(spec.capabilities or [])
            needed_caps = set(capabilities_needed)
            if "general" in agent_caps:
                match_score = 0.3
                reason = "General fallback agent"
            elif needed_caps & agent_caps:
                match_score = len(needed_caps & agent_caps) / len(needed_caps)
                reason = f"Matched capabilities: {', '.join(sorted(needed_caps & agent_caps))}"
            else:
                continue

            priority_score = spec.priority / 100.0
            final_score = (match_score * 0.7) + (priority_score * 0.3)
            agent_scores.append((agent, final_score, reason))

        if not agent_scores:
            generalist = agents.get("generalist")
            if generalist:
                return generalist, "No specialist matched, using generalist"
            return list(agents.values())[0], "Fallback to first available agent"

        agent_scores.sort(key=lambda item: item[1], reverse=True)
        selected, score, reason = agent_scores[0]
        logger.info(
            f"Selected agent '{getattr(selected, 'name', 'unknown')}' with score {score:.2f}: {reason}"
        )
        return selected, reason

    async def should_handoff(
        self,
        current_agent: Any,
        message: str,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Tuple[Any, str]]:
        if not tool_results:
            return None

        tool_names = [result.get("tool_name", "") for result in tool_results if result]
        qa_tools = {"answer_question", "summarize_document"}
        doc_tools = {"search_documents", "get_document_details", "delete_document"}

        if (
            getattr(current_agent, "name", "") == "document_expert"
            and set(tool_names) & qa_tools
            and not set(tool_names) & doc_tools
        ):
            qa_agent = self._agent_cache.get("qa_specialist")
            if qa_agent and getattr(qa_agent, "is_active", True):
                return qa_agent, "Detected Q&A intent after document search"

        if (
            getattr(current_agent, "name", "") == "qa_specialist"
            and set(tool_names) & doc_tools
            and not set(tool_names) & qa_tools
        ):
            doc_agent = self._agent_cache.get("document_expert")
            if doc_agent and getattr(doc_agent, "is_active", True):
                return doc_agent, "Detected document management intent"
        return None
