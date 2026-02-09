"""
Agent Router Service.

Routes user requests to appropriate specialized agents based on:
- Intent analysis using LLM
- Capability matching
- Agent priority
"""

from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import json
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.models.agent_definition import AgentDefinition
from app.services.llm_service import LLMService


# Capability to intent keyword mapping for faster matching
CAPABILITY_KEYWORDS = {
    "paper_search": [
        "arxiv", "paper", "papers", "preprint", "publication",
        "literature", "scientific", "research paper", "find papers", "search papers"
    ],
    "document_search": [
        "search", "find", "look for", "locate", "where is", "documents about",
        "files about", "what documents", "find documents"
    ],
    "document_crud": [
        "delete", "remove", "create document", "upload", "add document",
        "edit document", "update document"
    ],
    "document_compare": [
        "compare", "difference", "similar", "diff between", "versus"
    ],
    "tag_management": [
        "tag", "tags", "categorize", "label", "organize"
    ],
    "rag_qa": [
        "what is", "how does", "explain", "why", "when", "who",
        "tell me about", "describe", "what do you know", "based on"
    ],
    "summarization": [
        "summarize", "summary", "tldr", "overview", "key points", "main points"
    ],
    "knowledge_synthesis": [
        "analyze", "insight", "pattern", "trend", "relationship between",
        "connection", "how are", "related"
    ],
    "workflow_exec": [
        "workflow", "automate", "run workflow", "execute workflow"
    ],
    "template_fill": [
        "template", "fill template", "generate from template", "document template"
    ],
    "diagram_gen": [
        "diagram", "chart", "visualization", "visualize", "draw", "mermaid"
    ],
    "automation": [
        "automate", "schedule", "recurring", "batch process"
    ],
    "code_analysis": [
        "code", "function", "class", "method", "implementation",
        "algorithm", "logic", "debug", "refactor", "code structure",
        "code pattern", "code review"
    ],
    "code_explanation": [
        "explain code", "how does this code", "what does this code",
        "understand code", "walk through code", "code walkthrough"
    ],
}


class AgentRouter:
    """
    Routes user requests to appropriate specialized agents.

    Uses a combination of:
    1. Keyword-based quick matching
    2. LLM-based intent analysis (for complex cases)
    3. Capability matching to agent definitions
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self._agent_cache: Dict[str, AgentDefinition] = {}

    async def load_agents(self, db: AsyncSession) -> Dict[str, AgentDefinition]:
        """Load active agent definitions from database."""
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.is_active == True)
        )
        agents = result.scalars().all()

        self._agent_cache = {agent.name: agent for agent in agents}
        logger.info(f"Loaded {len(self._agent_cache)} active agents")

        return self._agent_cache

    def get_agents(self) -> Dict[str, AgentDefinition]:
        """Get cached agents."""
        return self._agent_cache

    async def analyze_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze user message to determine intent and required capabilities.

        Args:
            message: User's message
            history: Conversation history for context
            use_llm: Whether to use LLM for complex analysis

        Returns:
            Dict with intents, capabilities_needed, and confidence
        """
        message_lower = message.lower()

        # Step 1: Quick keyword-based matching
        capabilities_found = []
        confidence = 0.0

        for capability, keywords in CAPABILITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if capability not in capabilities_found:
                        capabilities_found.append(capability)
                    confidence = max(confidence, 0.7)

        # If we have high confidence from keywords, skip LLM
        if confidence >= 0.7 and capabilities_found:
            return {
                "intents": capabilities_found,
                "capabilities_needed": capabilities_found,
                "confidence": confidence,
                "method": "keyword"
            }

        # Step 2: LLM-based intent analysis for ambiguous cases
        if use_llm and confidence < 0.7:
            try:
                llm_result = await self._analyze_intent_with_llm(message, history)
                if llm_result and llm_result.get("capabilities_needed"):
                    return {
                        **llm_result,
                        "method": "llm"
                    }
            except Exception as e:
                logger.warning(f"LLM intent analysis failed, using fallback: {e}")

        # Step 3: Fallback to general if no specific intent found
        if not capabilities_found:
            capabilities_found = ["general"]
            confidence = 0.3

        return {
            "intents": capabilities_found,
            "capabilities_needed": capabilities_found,
            "confidence": confidence,
            "method": "fallback"
        }

    async def _analyze_intent_with_llm(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze intent for complex messages."""
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

        try:
            response = await self.llm_service.generate_text(
                prompt=prompt,
                system_prompt="You are an intent classifier. Respond only with valid JSON.",
                temperature=0.1,
                max_tokens=200
            )

            # Parse JSON response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Validate capabilities
                valid_caps = [c for c in result.get("capabilities_needed", [])
                              if c in available_capabilities]
                if valid_caps:
                    return {
                        "capabilities_needed": valid_caps,
                        "confidence": result.get("confidence", 0.6),
                        "reasoning": result.get("reasoning", "")
                    }

        except Exception as e:
            logger.warning(f"Failed to parse LLM intent response: {e}")

        return None

    async def select_agent(
        self,
        intent_analysis: Dict[str, Any],
        available_agents: Optional[Dict[str, AgentDefinition]] = None
    ) -> Tuple[AgentDefinition, str]:
        """
        Select the best agent based on intent analysis.

        Args:
            intent_analysis: Result from analyze_intent()
            available_agents: Dict of agent name -> AgentDefinition

        Returns:
            Tuple of (selected_agent, routing_reason)
        """
        agents = available_agents or self._agent_cache
        if not agents:
            raise ValueError("No agents available for routing")

        capabilities_needed = intent_analysis.get("capabilities_needed", ["general"])

        # Score each agent based on capability match and priority
        agent_scores: List[Tuple[AgentDefinition, float, str]] = []

        for agent in agents.values():
            if not agent.is_active:
                continue

            # Calculate capability match score
            agent_caps = set(agent.capabilities or [])
            needed_caps = set(capabilities_needed)

            if "general" in agent_caps:
                # Generalist can handle anything but with lower base score
                match_score = 0.3
                reason = "General fallback agent"
            elif needed_caps & agent_caps:
                # Has some matching capabilities
                match_score = len(needed_caps & agent_caps) / len(needed_caps)
                matched = list(needed_caps & agent_caps)
                reason = f"Matched capabilities: {', '.join(matched)}"
            else:
                # No match
                continue

            # Combine match score with priority
            # Priority is 0-100, normalize to 0-1 and weight it
            priority_score = agent.priority / 100.0
            final_score = (match_score * 0.7) + (priority_score * 0.3)

            agent_scores.append((agent, final_score, reason))

        if not agent_scores:
            # No agent matched, use generalist if available
            generalist = agents.get("generalist")
            if generalist:
                return generalist, "No specialist matched, using generalist"
            # Return first available agent
            return list(agents.values())[0], "Fallback to first available agent"

        # Sort by score descending
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        selected, score, reason = agent_scores[0]
        logger.info(
            f"Selected agent '{selected.name}' with score {score:.2f}: {reason}"
        )

        return selected, reason

    async def should_handoff(
        self,
        current_agent: AgentDefinition,
        message: str,
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Tuple[AgentDefinition, str]]:
        """
        Determine if the current agent should hand off to another.

        This is called after the agent processes to check if a different
        agent would be better suited for follow-up.

        Args:
            current_agent: The agent that just processed
            message: The user's message
            tool_results: Results from tools the agent called

        Returns:
            Tuple of (new_agent, handoff_reason) or None if no handoff needed
        """
        if not tool_results:
            return None

        # Check if any tool results suggest a different domain
        tool_names = [tr.get("tool_name", "") for tr in tool_results if tr]

        # Detect domain shifts based on tool usage patterns
        qa_tools = {"answer_question", "summarize_document"}
        doc_tools = {"search_documents", "get_document_details", "delete_document"}
        workflow_tools = {"run_workflow", "start_template_fill", "generate_diagram"}

        used_qa = bool(set(tool_names) & qa_tools)
        used_doc = bool(set(tool_names) & doc_tools)
        used_workflow = bool(set(tool_names) & workflow_tools)

        # If current agent is document_expert but used QA tools, consider handoff
        if current_agent.name == "document_expert" and used_qa and not used_doc:
            qa_agent = self._agent_cache.get("qa_specialist")
            if qa_agent and qa_agent.is_active:
                return qa_agent, "Detected Q&A intent after document search"

        # If current is QA but heavy doc operations, consider handoff
        if current_agent.name == "qa_specialist" and used_doc and not used_qa:
            doc_agent = self._agent_cache.get("document_expert")
            if doc_agent and doc_agent.is_active:
                return doc_agent, "Detected document management intent"

        return None

    async def get_agent_by_name(
        self,
        name: str,
        db: Optional[AsyncSession] = None
    ) -> Optional[AgentDefinition]:
        """Get a specific agent by name."""
        if name in self._agent_cache:
            return self._agent_cache.get(name)

        if db:
            result = await db.execute(
                select(AgentDefinition).where(
                    AgentDefinition.name == name,
                    AgentDefinition.is_active == True
                )
            )
            agent = result.scalar_one_or_none()
            if agent:
                self._agent_cache[name] = agent
            return agent

        return None

    async def get_generalist_agent(
        self,
        db: Optional[AsyncSession] = None
    ) -> Optional[AgentDefinition]:
        """Get the generalist fallback agent."""
        return await self.get_agent_by_name("generalist", db)
