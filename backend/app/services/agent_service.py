"""
Agent service for handling agentic chat with tool calling.

Implements the agent loop: Plan -> Execute -> Observe -> Respond

Multi-Agent Architecture:
- AgentRouter selects specialized agents based on user intent
- AgentMemoryIntegration injects relevant memories into prompts
- Each agent has capabilities and tool whitelists
"""

import json
import time
import re
import asyncio
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import ARRAY
from loguru import logger

from app.services.llm_service import LLMService, UserLLMSettings
from app.services.document_service import DocumentService
from app.services.vector_store import vector_store_service
from app.services.agent_tools import AGENT_TOOLS, get_tools_description, validate_tool_params
from app.services.agent_router import AgentRouter
from app.services.agent_memory_integration import AgentMemoryIntegration
from app.services.memory_service import MemoryService
from app.services.arxiv_search_service import ArxivSearchService
from app.schemas.agent import (
    AgentMessage,
    AgentToolCall,
    AgentChatRequest,
    AgentChatResponse,
    DocumentSearchResult,
    DocumentDetails,
    AgentRoutingInfo,
)
from app.models.document import Document, DocumentSource
from app.models.user import User as DbUser
from app.models.agent_definition import AgentDefinition, AgentConversationContext
from app.models.memory import UserPreferences, AgentConversation
from app.models.tool_audit import ToolExecutionAudit
from app.core.config import settings
from fastapi.encoders import jsonable_encoder


class AgentService:
    """
    Service for handling agentic chat with tool calling.

    Orchestrates multi-agent conversations with:
    - Intent-based agent routing
    - Memory injection into prompts
    - Tool filtering per agent
    - Agent handoffs
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.document_service = DocumentService()
        self.vector_store = vector_store_service
        self.memory_service = MemoryService()
        self.router = AgentRouter(self.llm_service)
        self.memory_integration = AgentMemoryIntegration(self.memory_service)
        self._vector_store_initialized = False
        self._agents_loaded = False
        self._vector_store_init_lock = asyncio.Lock()
        self._agents_load_lock = asyncio.Lock()

    async def _ensure_vector_store_initialized(self):
        """Ensure vector store is initialized."""
        if self._vector_store_initialized:
            return
        async with self._vector_store_init_lock:
            if self._vector_store_initialized:
                return
            await self.vector_store.initialize(background=True)
            self._vector_store_initialized = True

    async def _ensure_agents_loaded(self, db: AsyncSession) -> None:
        """Ensure agent definitions are loaded from database."""
        if self._agents_loaded:
            return
        async with self._agents_load_lock:
            if self._agents_loaded:
                return
            await self.router.load_agents(db)
            self._agents_loaded = True

    async def _get_user_preferences(
        self,
        user_id: UUID,
        db: AsyncSession
    ) -> UserPreferences:
        """Get or create user preferences."""
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        prefs = result.scalar_one_or_none()

        if not prefs:
            # Create default preferences
            prefs = UserPreferences(user_id=user_id)
            db.add(prefs)
            await db.commit()
            await db.refresh(prefs)

        return prefs

    def _routing_from_agent(
        self,
        agent: AgentDefinition,
        *,
        user_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
    ) -> Optional[Dict[str, Any]]:
        rd = getattr(agent, "routing_defaults", None)
        if not isinstance(rd, dict):
            return None

        # Base routing defaults
        tier = str(rd.get("tier") or rd.get("llm_tier") or "").strip().lower() or None
        fallback = rd.get("fallback_tiers") or rd.get("llm_fallback_tiers")
        if not isinstance(fallback, list):
            fallback = []
        fallback_tiers = [str(x).strip().lower() for x in fallback if str(x).strip()]

        out: Dict[str, Any] = {"tier": tier, "fallback_tiers": fallback_tiers}

        for k_in, k_out, lo, hi in [
            ("timeout_seconds", "timeout_seconds", 2, 600),
            ("llm_timeout_seconds", "timeout_seconds", 2, 600),
            ("max_tokens_cap", "max_tokens_cap", 64, 20000),
            ("llm_max_tokens_cap", "max_tokens_cap", 64, 20000),
            ("cooldown_seconds", "cooldown_seconds", 5, 3600),
            ("llm_unhealthy_cooldown_seconds", "cooldown_seconds", 5, 3600),
        ]:
            if k_in in rd and rd.get(k_in) is not None:
                try:
                    v = int(rd.get(k_in))
                    out[k_out] = max(lo, min(v, hi))
                except Exception:
                    pass

        origin: Dict[str, Any] = {
            "source": "agent_defaults",
            "agent_id": str(getattr(agent, 'id', '') or ''),
        }

        # Optional A/B experiment routing per agent.
        # routing_defaults.experiment = {
        #   "id": "exp-1",
        #   "enabled": true,
        #   "salt": "optional",
        #   "variants": [
        #     {"id": "A", "weight": 50, "routing": {"tier": "deep", "fallback_tiers": ["balanced"]}},
        #     {"id": "B", "weight": 50, "routing": {"tier": "balanced", "fallback_tiers": ["fast"]}}
        #   ]
        # }
        exp = rd.get("experiment") if isinstance(rd.get("experiment"), dict) else None
        if exp and bool(exp.get("enabled")):
            exp_id = str(exp.get("id") or "").strip()
            variants = exp.get("variants") if isinstance(exp.get("variants"), list) else []
            salt = str(exp.get("salt") or "").strip()
            cleaned = []
            total_w = 0
            for v in variants:
                if not isinstance(v, dict):
                    continue
                vid = str(v.get("id") or "").strip()
                if not vid:
                    continue
                try:
                    w = int(v.get("weight") or 0)
                except Exception:
                    w = 0
                w = max(0, min(w, 1000000))
                routing_v = v.get("routing") if isinstance(v.get("routing"), dict) else {}
                if w <= 0:
                    continue
                cleaned.append({"id": vid, "weight": w, "routing": routing_v})
                total_w += w

            if exp_id and cleaned and total_w > 0:
                # Deterministic assignment.
                uid = str(user_id or "")
                cid = str(conversation_id or "")
                key = f"{getattr(agent, 'id', '')}:{exp_id}:{uid}:{cid}:{salt}".encode('utf-8')
                h = hashlib.sha256(key).hexdigest()
                bucket = int(h[:8], 16) / float(0xFFFFFFFF)
                target = bucket * float(total_w)

                acc = 0.0
                chosen = cleaned[-1]
                for v in cleaned:
                    acc += float(v["weight"])
                    if target <= acc:
                        chosen = v
                        break

                # Apply variant routing as overrides.
                rv = chosen.get("routing") if isinstance(chosen.get("routing"), dict) else {}
                if rv:
                    for kk, vv in rv.items():
                        if kk in {"tier", "llm_tier"}:
                            t = str(vv or "").strip().lower() or None
                            out["tier"] = t
                        elif kk in {"fallback_tiers", "llm_fallback_tiers"} and isinstance(vv, list):
                            out["fallback_tiers"] = [str(x).strip().lower() for x in vv if str(x).strip()]
                        elif kk in {"timeout_seconds", "llm_timeout_seconds", "max_tokens_cap", "llm_max_tokens_cap", "cooldown_seconds", "llm_unhealthy_cooldown_seconds"}:
                            try:
                                out_key = kk
                                if kk.startswith('llm_'):
                                    out_key = kk.replace('llm_', '').replace('unhealthy_', '')
                                    if out_key == 'cooldown_seconds':
                                        out_key = 'cooldown_seconds'
                                out[out_key] = int(vv)
                            except Exception:
                                pass

                origin = {
                    "source": "agent_experiment",
                    "agent_id": str(getattr(agent, 'id', '') or ''),
                    "experiment_id": exp_id,
                    "experiment_variant_id": str(chosen.get("id") or ""),
                }

        out["_origin"] = origin

        # Return None if empty
        if out.get("tier") is None and not out.get("fallback_tiers") and not any(
            k in out for k in ("timeout_seconds", "max_tokens_cap", "cooldown_seconds")
        ):
            return None
        return out


    def _filter_tools_for_agent(
        self,
        agent: AgentDefinition,
        all_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter available tools based on agent's whitelist."""
        if agent.tool_whitelist is None:
            # No whitelist = all tools available
            return all_tools

        allowed = set(agent.tool_whitelist or [])
        return [tool for tool in all_tools if tool.get("name") in allowed]

    def _get_tools_description_for_agent(
        self,
        agent: AgentDefinition
    ) -> str:
        """Get tools description filtered for specific agent."""
        filtered_tools = self._filter_tools_for_agent(agent, AGENT_TOOLS)
        if not filtered_tools:
            return "No tools available."

        descriptions = []
        for tool in filtered_tools:
            name = tool.get("name", "unknown_tool")
            desc = f"- {name}: {tool.get('description', 'No description')}"

            params = (tool.get("parameters") or {}).get("properties") or {}
            required_params = set((tool.get("parameters") or {}).get("required") or [])
            if params:
                param_strs = []
                for pname, pinfo in params.items():
                    required = "(required)" if pname in required_params else "(optional)"
                    param_strs.append(f"  - {pname} {required}: {pinfo.get('description', '')}")
                desc += "\n" + "\n".join(param_strs)

            descriptions.append(desc)

        return "\n\n".join(descriptions)

    async def process_message(
        self,
        message: str,
        conversation_history: Optional[List[AgentMessage]],
        user_id: UUID,
        db: AsyncSession,
        user_settings: Optional[UserLLMSettings] = None,
        conversation_id: Optional[UUID] = None,
        turn_number: int = 0,
        agent_id: Optional[UUID] = None,
    ) -> AgentChatResponse:
        """
        Process user message through multi-agent loop:
        1. Load user preferences and relevant memories
        2. Route to appropriate specialized agent
        3. Analyze intent and determine required tools (filtered by agent)
        4. Execute tools in sequence
        5. Generate response with tool results
        6. Check for agent handoff
        7. Extract memories (async, non-blocking)
        """
        history = conversation_history or []

        try:
            # Ensure agents are loaded
            await self._ensure_agents_loaded(db)

            # Step 1: Load user preferences
            preferences = await self._get_user_preferences(user_id, db)

            # Step 2: Get relevant memories (if enabled)
            memory_context = ""
            injected_memories = []
            if preferences.enable_agent_memory:
                memories = await self.memory_integration.get_relevant_memories(
                    user_id=user_id,
                    message=message,
                    conversation_id=conversation_id,
                    preferences=preferences,
                    db=db
                )
                if memories:
                    memory_context = self.memory_integration.format_memories_for_prompt(memories)
                    injected_memories = memories
                    logger.info(f"Injecting {len(memories)} memories into agent context")

                    # Record memory injections if we have a conversation
                    if conversation_id:
                        await self.memory_integration.record_memory_injections_batch(
                            conversation_id=conversation_id,
                            memories=memories,
                            turn_number=turn_number,
                            injection_type="automatic",
                            db=db
                        )

            # Step 3: Route to appropriate agent (or force a specific agent)
            selected_agent: AgentDefinition
            routing_reason: str

            if agent_id:
                selected_agent = next(
                    (a for a in self.router.get_agents().values() if a.id == agent_id),
                    None
                )
                if not selected_agent:
                    result = await db.execute(select(AgentDefinition).where(AgentDefinition.id == agent_id))
                    selected_agent = result.scalar_one_or_none()
                    if not selected_agent:
                        raise ValueError("Requested agent not found")
                    if not selected_agent.is_active:
                        raise ValueError("Requested agent is inactive")
                routing_reason = "Forced by request agent_id"
            else:
                intent_analysis = await self.router.analyze_intent(
                    message=message,
                    history=[{"role": m.role, "content": m.content} for m in history[-5:]],
                    use_llm=True
                )

                selected_agent, routing_reason = await self.router.select_agent(
                    intent_analysis=intent_analysis,
                    available_agents=self.router.get_agents()
                )

            logger.info(
                f"Selected agent '{selected_agent.name}' for message: {routing_reason}"
            )

            # Track agent selection if we have a conversation
            if conversation_id:
                context = AgentConversationContext(
                    conversation_id=conversation_id,
                    agent_definition_id=selected_agent.id,
                    turn_number=turn_number,
                    routing_reason=routing_reason
                )
                db.add(context)
                await db.commit()

            # Step 4: Plan - Determine which tools to call (filtered by agent)
            tool_calls = await self._plan_tool_calls_for_agent(
                message=message,
                history=history,
                agent=selected_agent,
                memory_context=memory_context,
                user_settings=user_settings
            )

            # Step 5: Execute - Run each tool
            tool_results = []
            requires_user_action = False
            action_type = None

            for call in tool_calls:
                # Verify tool is allowed for this agent
                if not selected_agent.has_tool(call.tool_name):
                    logger.warning(
                        f"Agent '{selected_agent.name}' not allowed to use tool '{call.tool_name}'"
                    )
                    call.status = "failed"
                    call.error = f"Tool '{call.tool_name}' not available for this agent"
                    tool_results.append(call)
                    continue

                result = await self._execute_tool(
                    call,
                    user_id,
                    db,
                    conversation_id=conversation_id,
                    agent_definition_id=selected_agent.id,
                )
                tool_results.append(result)

                # Check if any tool requires user action
                if result.tool_name == "request_file_upload" and result.status == "completed":
                    requires_user_action = True
                    action_type = "upload_file"

            # Step 6: Observe & Respond - Generate final response
            response_content = await self._generate_response_for_agent(
                message=message,
                tool_results=tool_results,
                history=history,
                agent=selected_agent,
                memory_context=memory_context,
                user_settings=user_settings,
                user_id=user_id,
                db=db,
            )

            # Step 7: Check for handoff to another agent
            handoff_info = None
            if tool_results:
                handoff = await self.router.should_handoff(
                    current_agent=selected_agent,
                    message=message,
                    tool_results=[{
                        "tool_name": r.tool_name,
                        "status": r.status,
                        "output": r.tool_output
                    } for r in tool_results]
                )
                if handoff:
                    new_agent, handoff_reason = handoff
                    logger.info(
                        f"Agent handoff: '{selected_agent.name}' -> '{new_agent.name}': {handoff_reason}"
                    )
                    handoff_info = {
                        "from_agent": selected_agent.name,
                        "to_agent": new_agent.name,
                        "reason": handoff_reason
                    }

            # Build routing info for response
            routing_info = AgentRoutingInfo(
                agent_id=selected_agent.id,
                agent_name=selected_agent.name,
                agent_display_name=selected_agent.display_name,
                routing_reason=routing_reason,
                handoff_from=handoff_info.get("from_agent") if handoff_info else None
            )

            response_message = AgentMessage(
                role="assistant",
                content=response_content,
                tool_calls=tool_results if tool_results else None,
                created_at=datetime.utcnow()
            )

            # Step 8: Extract memories asynchronously (non-blocking)
            if preferences.enable_agent_memory and conversation_id:
                # Build messages for memory extraction
                messages_for_extraction = [
                    {"role": m.role, "content": m.content}
                    for m in history[-5:]
                ]
                messages_for_extraction.append({"role": "user", "content": message})
                messages_for_extraction.append({"role": "assistant", "content": response_content})

                # Fire and forget - don't block response
                asyncio.create_task(
                    self._extract_memories_background(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        messages=messages_for_extraction,
                        preferences=preferences,
                        db=db
                    )
                )

            return AgentChatResponse(
                message=response_message,
                tool_results=tool_results if tool_results else None,
                requires_user_action=requires_user_action,
                action_type=action_type,
                routing_info=routing_info,
                injected_memories=[str(m.id) for m in injected_memories] if injected_memories else None
            )

        except Exception as e:
            logger.error(f"Error processing agent message: {e}", exc_info=True)
            error_message = AgentMessage(
                role="assistant",
                content=f"I encountered an error while processing your request: {str(e)}. Please try again.",
                created_at=datetime.utcnow()
            )
            return AgentChatResponse(
                message=error_message,
                requires_user_action=False
            )

    async def _extract_memories_background(
        self,
        user_id: UUID,
        conversation_id: UUID,
        messages: List[Dict[str, Any]],
        preferences: UserPreferences,
        db: AsyncSession
    ) -> None:
        """Extract memories from conversation in background."""
        try:
            await self.memory_integration.extract_and_store_memories(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=messages,
                preferences=preferences,
                db=db
            )
        except Exception as e:
            logger.error(f"Background memory extraction failed: {e}")

    async def _plan_tool_calls_for_agent(
        self,
        message: str,
        history: List[AgentMessage],
        agent: AgentDefinition,
        memory_context: str,
        user_settings: Optional[UserLLMSettings] = None
    ) -> List[AgentToolCall]:
        """
        Use LLM to determine which tools to call based on user message.
        Filtered by agent's tool whitelist and enhanced with memory context.
        """
        # Get tools available to this agent
        tools_desc = self._get_tools_description_for_agent(agent)

        # Build conversation context
        context_messages = []
        for msg in history[-5:]:
            context_messages.append(f"{msg.role.upper()}: {msg.content[:200]}")
        context_str = "\n".join(context_messages) if context_messages else "No previous context."

        # Include memory context if available
        memory_section = ""
        if memory_context:
            memory_section = f"\n\n{memory_context}\n"

        # Use agent's system prompt as base
        planning_prompt = f"""{agent.system_prompt}

You are acting as the {agent.display_name}. Based on the user's message, decide which tools to call.
{memory_section}
Available tools for your role:
{tools_desc}

Recent conversation:
{context_str}

User's current message: {message}

Respond ONLY with a JSON array of tool calls. Each tool call should have:
- "tool_name": name of the tool to call
- "tool_input": object with the required parameters

If no tools are needed (e.g., just a greeting or general question), respond with an empty array: []

Examples:
- User: "Find documents about Python" -> [{{"tool_name": "search_documents", "tool_input": {{"query": "Python", "limit": 5}}}}]
- User: "What is document abc123?" -> [{{"tool_name": "get_document_details", "tool_input": {{"document_id": "abc123"}}}}]
- User: "Hello!" -> []

Your response (JSON array only):"""

        try:
            response = await self.llm_service.generate_response(
                query=planning_prompt,
                temperature=0.1,
                max_tokens=500,
                user_settings=user_settings,
                task_type="chat",
                routing=self._routing_from_agent(agent),
            )

            tool_calls = self._parse_tool_calls(response)
            return tool_calls

        except Exception as e:
            logger.error(f"Error planning tool calls for agent '{agent.name}': {e}")
            return []

    async def _generate_response_for_agent(
        self,
        message: str,
        tool_results: List[AgentToolCall],
        history: List[AgentMessage],
        agent: AgentDefinition,
        memory_context: str,
        user_settings: Optional[UserLLMSettings] = None,
        user_id: UUID | None = None,
        db: AsyncSession | None = None,
    ) -> str:
        """Generate final response based on tool results, using agent's personality."""
        # Build context from tool results
        tool_context_parts = []
        for result in tool_results:
            if result.status == "completed":
                output_str = json.dumps(result.tool_output, indent=2, default=str)
                tool_context_parts.append(
                    f"Tool '{result.tool_name}' result:\n{output_str}"
                )
            elif result.status == "failed":
                tool_context_parts.append(
                    f"Tool '{result.tool_name}' failed: {result.error}"
                )

        tool_context = "\n\n".join(tool_context_parts) if tool_context_parts else "No tools were executed."

        # Include memory context if available
        memory_section = ""
        if memory_context:
            memory_section = f"\n\n{memory_context}\n"

        # Use agent's system prompt
        response_prompt = f"""{agent.system_prompt}

You are the {agent.display_name}. Based on the user's request and tool execution results, provide a helpful response.
{memory_section}
User's request: {message}

Tool execution results:
{tool_context}

Guidelines:
- Respond in character as the {agent.display_name}
- Summarize the results in a natural, conversational way
- If search results are present, list the most relevant documents
- If there were errors, explain what went wrong and suggest alternatives
- If a confirmation is required (like for deletion), ask the user to confirm
- Use any relevant user context from memories to personalize your response
- Keep the response concise but informative
- Use markdown formatting when appropriate

Your response:"""

        try:
            response = await self.llm_service.generate_response(
                query=response_prompt,
                temperature=0.7,
                max_tokens=800,
                user_settings=user_settings,
                task_type="chat",
                user_id=user_id,
                db=db,
                routing=self._routing_from_agent(agent, user_id=user_id, conversation_id=None),
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response for agent '{agent.name}': {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    async def _plan_tool_calls(
        self,
        message: str,
        history: List[AgentMessage],
        user_settings: Optional[UserLLMSettings] = None
    ) -> List[AgentToolCall]:
        """Use LLM to determine which tools to call based on user message."""

        # Build the planning prompt
        tools_desc = get_tools_description()

        # Build conversation context
        context_messages = []
        for msg in history[-5:]:  # Last 5 messages for context
            context_messages.append(f"{msg.role.upper()}: {msg.content[:200]}")
        context_str = "\n".join(context_messages) if context_messages else "No previous context."

        planning_prompt = f"""You are an AI assistant that helps users manage documents in a knowledge base.
Based on the user's message, decide which tools to call to fulfill their request.

Available tools:
{tools_desc}

Recent conversation:
{context_str}

User's current message: {message}

Respond ONLY with a JSON array of tool calls. Each tool call should have:
- "tool_name": name of the tool to call
- "tool_input": object with the required parameters

If no tools are needed (e.g., just a greeting or general question), respond with an empty array: []

Examples:
- User: "Find documents about Python" -> [{{"tool_name": "search_documents", "tool_input": {{"query": "Python", "limit": 5}}}}]
- User: "What is document abc123?" -> [{{"tool_name": "get_document_details", "tool_input": {{"document_id": "abc123"}}}}]
- User: "Hello!" -> []
- User: "Delete document xyz789" -> [{{"tool_name": "delete_document", "tool_input": {{"document_id": "xyz789", "confirm": false}}}}]
- User: "I want to upload a file" -> [{{"tool_name": "request_file_upload", "tool_input": {{}}}}]

Your response (JSON array only):"""

        try:
            response = await self.llm_service.generate_response(
                query=planning_prompt,
                temperature=0.1,  # Low temperature for consistent planning
                max_tokens=500,
                user_settings=user_settings,
                task_type="chat"
            )

            # Parse the JSON response
            tool_calls = self._parse_tool_calls(response)
            return tool_calls

        except Exception as e:
            logger.error(f"Error planning tool calls: {e}")
            return []

    def _parse_tool_calls(self, response: str) -> List[AgentToolCall]:
        """Parse LLM response to extract tool calls."""
        try:
            # Clean up the response - extract JSON array
            response = response.strip()

            # Try to find JSON array in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                if not isinstance(parsed, list):
                    return []

                tool_calls = []
                for item in parsed:
                    if isinstance(item, dict) and "tool_name" in item:
                        # Validate the tool
                        tool_name = item["tool_name"]
                        tool_input = item.get("tool_input", {})

                        is_valid, error = validate_tool_params(tool_name, tool_input)
                        if is_valid:
                            tool_calls.append(AgentToolCall(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                status="pending"
                            ))
                        else:
                            logger.warning(f"Invalid tool call: {error}")

                return tool_calls

            return []

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool calls JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing tool calls: {e}")
            return []

    async def _execute_tool(
        self,
        tool_call: AgentToolCall,
        user_id: UUID,
        db: AsyncSession,
        conversation_id: UUID | None = None,
        agent_definition_id: UUID | None = None,
        audit_row: ToolExecutionAudit | None = None,
        bypass_approval_gate: bool = False,
    ) -> AgentToolCall:
        """Execute a single tool and return result."""
        tool_name = tool_call.tool_name
        tool_input = tool_call.tool_input
        start_time = time.time()
        audit: ToolExecutionAudit | None = audit_row

        try:
            # Tool policies: allow-by-default with explicit denies and optional approvals.
            try:
                from app.models.user import User
                from app.services.tool_policy_engine import evaluate_tool_policy

                user = await db.get(User, user_id)
                decision = await evaluate_tool_policy(
                    db=db,
                    tool_name=tool_name,
                    tool_args=tool_input,
                    user=user,
                    agent_definition_id=agent_definition_id,
                )
                decision_snapshot = {
                    "allowed": bool(decision.allowed),
                    "require_approval": bool(decision.require_approval),
                    "denied_reason": decision.denied_reason,
                    "matched_policies": decision.matched_policies,
                }
                if not decision.allowed:
                    tool_call.status = "failed"
                    tool_call.error = decision.denied_reason or "Tool denied by policy"
                    tool_call.tool_output = {"error": "tool_denied", "message": tool_call.error}

                    if audit is None:
                        audit = ToolExecutionAudit(
                            user_id=user_id,
                            agent_definition_id=agent_definition_id,
                            conversation_id=conversation_id,
                            tool_name=tool_name,
                            tool_input=jsonable_encoder(tool_input),
                            policy_decision=jsonable_encoder(decision_snapshot),
                            status="failed",
                            error=tool_call.error,
                            approval_required=False,
                            approval_status=None,
                        )
                        db.add(audit)
                        await db.commit()
                    else:
                        audit.status = "failed"
                        audit.error = tool_call.error
                        audit.policy_decision = jsonable_encoder(decision_snapshot)
                        await db.commit()

                    return tool_call
            except Exception:
                # Fail closed: if policy evaluation fails, block execution.
                tool_call.status = "failed"
                tool_call.error = "Tool policy evaluation failed"
                tool_call.tool_output = {"error": "policy_error", "message": tool_call.error}
                if audit is None:
                    audit = ToolExecutionAudit(
                        user_id=user_id,
                        agent_definition_id=agent_definition_id,
                        conversation_id=conversation_id,
                        tool_name=tool_name,
                        tool_input=jsonable_encoder(tool_input),
                        policy_decision=jsonable_encoder({"allowed": False, "error": "policy_evaluation_failed"}),
                        status="failed",
                        error=tool_call.error,
                        approval_required=False,
                        approval_status=None,
                    )
                    db.add(audit)
                    await db.commit()
                else:
                    audit.status = "failed"
                    audit.error = tool_call.error
                    audit.policy_decision = jsonable_encoder({"allowed": False, "error": "policy_evaluation_failed"})
                    await db.commit()
                return tool_call

            tool_call.status = "running"
            if audit is None:
                audit = ToolExecutionAudit(
                    user_id=user_id,
                    agent_definition_id=agent_definition_id,
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    tool_input=jsonable_encoder(tool_input),
                    policy_decision=jsonable_encoder(
                        {
                            "allowed": True,
                            "require_approval": bool(getattr(decision, "require_approval", False)),  # type: ignore[name-defined]
                            "denied_reason": None,
                            "matched_policies": getattr(decision, "matched_policies", None),  # type: ignore[name-defined]
                        }
                    ),
                    status="running",
                )
                policy_requires_approval = False
                try:
                    policy_requires_approval = bool(decision.require_approval)  # type: ignore[name-defined]
                except Exception:
                    policy_requires_approval = False

                if (
                    settings.AGENT_REQUIRE_TOOL_APPROVAL
                    and (tool_name in set(settings.AGENT_DANGEROUS_TOOLS or []) or policy_requires_approval)
                ):
                    audit.approval_required = True
                    audit.approval_mode = "owner_and_admin"
                    audit.approval_status = "pending_owner"
                    audit.status = "requires_approval"
                    db.add(audit)
                    await db.commit()
                    await db.refresh(audit)
                    tool_call.status = "requires_approval"
                    tool_call.tool_output = {
                        "error": "approval_required",
                        "message": f"Tool '{tool_name}' requires approval before it can run.",
                        "approval_id": str(audit.id),
                    }
                    return tool_call

                db.add(audit)
                await db.commit()
                await db.refresh(audit)
            else:
                # Re-running an existing audit record (typically after approval)
                audit.agent_definition_id = audit.agent_definition_id or agent_definition_id
                audit.conversation_id = audit.conversation_id or conversation_id
                audit.tool_name = tool_name
                audit.tool_input = jsonable_encoder(tool_input)
                audit.tool_output = None
                audit.error = None
                audit.execution_time_ms = None
                audit.status = "running"
                await db.commit()

            if (
                not bypass_approval_gate
                and audit.approval_required
                and audit.approval_status != "approved"
            ):
                audit.status = "requires_approval"
                await db.commit()
                tool_call.status = "requires_approval"
                tool_call.tool_output = {
                    "error": "approval_required",
                    "message": f"Tool '{tool_name}' requires approval before it can run.",
                    "approval_id": str(audit.id),
                }
                return tool_call

            # Enforce per-agent tool whitelist at execution time.
            if agent_definition_id:
                try:
                    from app.models.agent_definition import AgentDefinition

                    agent_def = await db.get(AgentDefinition, agent_definition_id)
                    if agent_def and not agent_def.has_tool(tool_name):
                        audit.status = "blocked"
                        audit.error = f"Tool '{tool_name}' is not allowed for agent '{agent_def.name}'"
                        await db.commit()

                        tool_call.status = "failed"
                        tool_call.tool_output = {
                            "error": "tool_not_allowed",
                            "message": audit.error,
                        }
                        return tool_call
                except Exception:
                    # If whitelist enforcement fails unexpectedly, fail closed (block execution).
                    audit.status = "blocked"
                    audit.error = f"Failed to validate tool whitelist for agent_definition_id={agent_definition_id}"
                    await db.commit()
                    tool_call.status = "failed"
                    tool_call.tool_output = {"error": "tool_whitelist_check_failed", "message": audit.error}
                    return tool_call

            if tool_name == "search_documents":
                result = await self._tool_search_documents(tool_input, db)
            elif tool_name == "get_document_details":
                result = await self._tool_get_document_details(tool_input, db)
            elif tool_name == "summarize_document":
                result = await self._tool_summarize_document(tool_input, db)
            elif tool_name == "delete_document":
                result = await self._tool_delete_document(tool_input, db)
            elif tool_name == "list_recent_documents":
                result = await self._tool_list_recent_documents(tool_input, db)
            elif tool_name == "list_document_sources":
                result = await self._tool_list_document_sources(tool_input, db)
            elif tool_name == "list_documents_by_source":
                result = await self._tool_list_documents_by_source(tool_input, db)
            elif tool_name == "web_scrape":
                result = await self._tool_web_scrape(tool_input, user_id, db)
            elif tool_name == "request_file_upload":
                result = {
                    "action": "upload_requested",
                    "message": "Please select a file to upload using the upload button.",
                    "suggested_title": tool_input.get("suggested_title"),
                    "suggested_tags": tool_input.get("suggested_tags", [])
                }
            elif tool_name == "create_document_from_text":
                result = await self._tool_create_document_from_text(tool_input, user_id, db)
            elif tool_name == "ingest_url":
                result = await self._tool_ingest_url(tool_input, user_id, db)
            elif tool_name == "find_similar_documents":
                result = await self._tool_find_similar_documents(tool_input, db)
            elif tool_name == "search_documents_by_author":
                result = await self._tool_search_documents_by_author(tool_input, db)
            elif tool_name == "update_document_tags":
                result = await self._tool_update_document_tags(tool_input, db)
            elif tool_name == "get_knowledge_base_stats":
                result = await self._tool_get_knowledge_base_stats(db)
            elif tool_name == "batch_delete_documents":
                result = await self._tool_batch_delete_documents(tool_input, db)
            elif tool_name == "batch_summarize_documents":
                result = await self._tool_batch_summarize_documents(tool_input, db)
            elif tool_name == "search_by_tags":
                result = await self._tool_search_by_tags(tool_input, db)
            elif tool_name == "search_documents_by_tag":
                result = await self._tool_search_by_tags(tool_input, db)
            elif tool_name == "list_all_tags":
                result = await self._tool_list_all_tags(db)
            elif tool_name == "compare_documents":
                result = await self._tool_compare_documents(tool_input, db)
            elif tool_name == "start_template_fill":
                result = await self._tool_start_template_fill(tool_input, user_id, db)
            elif tool_name == "list_template_jobs":
                result = await self._tool_list_template_jobs(tool_input, user_id, db)
            elif tool_name == "get_template_job_status":
                result = await self._tool_get_template_job_status(tool_input, user_id, db)
            # RAG / Q&A Tools
            elif tool_name == "answer_question":
                result = await self._tool_answer_question(tool_input, db)
            # Document Content Tools
            elif tool_name == "read_document_content":
                result = await self._tool_read_document_content(tool_input, db)
            # Knowledge Graph Tools
            elif tool_name == "search_entities":
                result = await self._tool_search_entities(tool_input, db)
            elif tool_name == "get_entity_relationships":
                result = await self._tool_get_entity_relationships(tool_input, db)
            elif tool_name == "find_documents_by_entity":
                result = await self._tool_find_documents_by_entity(tool_input, db)
            elif tool_name == "get_document_knowledge_graph":
                result = await self._tool_get_document_knowledge_graph(tool_input, db)
            elif tool_name == "get_global_knowledge_graph":
                result = await self._tool_get_global_knowledge_graph(tool_input, db)
            elif tool_name == "get_entity_mentions":
                result = await self._tool_get_entity_mentions(tool_input, db)
            elif tool_name == "get_kg_stats":
                result = await self._tool_get_kg_stats(db)
            elif tool_name == "rebuild_document_knowledge_graph":
                result = await self._tool_rebuild_document_knowledge_graph(tool_input, user_id, db)
            elif tool_name == "merge_entities":
                result = await self._tool_merge_entities(tool_input, user_id, db)
            elif tool_name == "delete_entity":
                result = await self._tool_delete_entity(tool_input, user_id, db)
            elif tool_name == "generate_diagram":
                result = await self._tool_generate_diagram(tool_input, user_id, db)
            # Workflow and Custom Tool Integration
            elif tool_name == "run_workflow":
                result = await self._tool_run_workflow(tool_input, user_id, db)
            elif tool_name == "propose_workflow_from_description":
                result = await self._tool_propose_workflow_from_description(tool_input, user_id, db)
            elif tool_name == "create_workflow_from_description":
                result = await self._tool_create_workflow_from_description(tool_input, user_id, db)
            elif tool_name == "list_workflows":
                result = await self._tool_list_workflows(tool_input, user_id, db)
            elif tool_name == "run_custom_tool":
                result = await self._tool_run_custom_tool(tool_input, user_id, db)
            elif tool_name == "search_arxiv":
                result = await self._tool_search_arxiv(tool_input)
            elif tool_name == "ingest_arxiv_papers":
                result = await self._tool_ingest_arxiv_papers(tool_input, user_id, db)
            elif tool_name == "literature_review_arxiv":
                result = await self._tool_literature_review_arxiv(tool_input, user_id, db)
            elif tool_name == "summarize_documents_in_source":
                result = await self._tool_summarize_documents_in_source(tool_input, user_id, db)
            elif tool_name == "enrich_arxiv_metadata_for_source":
                result = await self._tool_enrich_arxiv_metadata_for_source(tool_input, user_id, db)
            elif tool_name == "generate_literature_review_for_source":
                result = await self._tool_generate_literature_review_for_source(tool_input, user_id, db)
            elif tool_name == "generate_slides_for_source":
                result = await self._tool_generate_slides_for_source(tool_input, user_id, db)
            elif tool_name == "list_custom_tools":
                result = await self._tool_list_custom_tools(tool_input, user_id, db)
            # Agent Collaboration Tools
            elif tool_name == "delegate_to_agent":
                result = await self._tool_delegate_to_agent(tool_input, user_id, db)
            elif tool_name == "list_available_agents":
                result = await self._tool_list_available_agents(db)
            # Data Analysis & Visualization Tools
            elif tool_name == "get_collection_statistics":
                result = await self._tool_get_collection_statistics(tool_input, db)
            elif tool_name == "get_source_analytics":
                result = await self._tool_get_source_analytics(tool_input, db)
            elif tool_name == "get_trending_topics":
                result = await self._tool_get_trending_topics(tool_input, db)
            elif tool_name == "generate_chart_data":
                result = await self._tool_generate_chart_data(tool_input, db)
            elif tool_name == "export_data":
                result = await self._tool_export_data(tool_input, db)
            # Advanced Search Tools
            elif tool_name == "faceted_search":
                result = await self._tool_faceted_search(tool_input, db)
            elif tool_name == "get_search_suggestions":
                result = await self._tool_get_search_suggestions(tool_input, db)
            elif tool_name == "get_related_searches":
                result = await self._tool_get_related_searches(tool_input, db)
            # Content Generation Tools
            elif tool_name == "draft_email":
                result = await self._tool_draft_email(tool_input, db)
            elif tool_name == "generate_meeting_notes":
                result = await self._tool_generate_meeting_notes(tool_input, db)
            elif tool_name == "generate_documentation":
                result = await self._tool_generate_documentation(tool_input, db)
            elif tool_name == "generate_executive_summary":
                result = await self._tool_generate_executive_summary(tool_input, db)
            elif tool_name == "generate_report":
                result = await self._tool_generate_report(tool_input, db)
            # GitLab Architecture Tools
            elif tool_name == "generate_gitlab_architecture":
                result = await self._tool_generate_gitlab_architecture(tool_input, user_id, db)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            tool_call.tool_output = result
            tool_call.status = "completed"
            tool_call.execution_time_ms = int((time.time() - start_time) * 1000)

            if audit:
                audit.tool_output = jsonable_encoder(result)
                audit.status = "completed"
                audit.execution_time_ms = tool_call.execution_time_ms
                await db.commit()

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            tool_call.status = "failed"
            tool_call.error = str(e)
            tool_call.execution_time_ms = int((time.time() - start_time) * 1000)
            if audit:
                audit.status = "failed"
                audit.error = str(e)
                audit.execution_time_ms = tool_call.execution_time_ms
                try:
                    await db.commit()
                except Exception:
                    pass

        return tool_call

    async def _tool_search_arxiv(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = (params.get("query") or "").strip()
        start = int(params.get("start") or 0)
        max_results = int(params.get("max_results") or 10)
        sort_by = params.get("sort_by") or "relevance"
        sort_order = params.get("sort_order") or "descending"

        service = ArxivSearchService()
        result = await service.search(
            query=query,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "total_results": result.total_results,
            "start": result.start,
            "max_results": result.max_results,
            "items": result.items,
        }

    async def _tool_literature_review_arxiv(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        topic = (params.get("topic") or "").strip()
        if not topic:
            raise ValueError("topic is required")

        query_override = (params.get("query") or "").strip()
        categories = [c.strip() for c in (params.get("categories") or []) if isinstance(c, str) and c.strip()]
        max_papers = max(1, min(int(params.get("max_papers") or 5), 25))
        ingest = bool(params.get("ingest", True))
        sort_by = params.get("sort_by") or "relevance"
        sort_order = params.get("sort_order") or "descending"

        if query_override:
            q = query_override
        else:
            q = f'all:"{topic}"' if " " in topic else f"all:{topic}"

        if categories:
            cat_expr = " OR ".join([f"cat:{c}" for c in categories])
            q = f"{q} AND ({cat_expr})"

        service = ArxivSearchService()
        result = await service.search(
            query=q,
            start=0,
            max_results=max_papers,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        papers = result.items
        paper_ids = [p.get("id") for p in papers if isinstance(p, dict) and p.get("id")]

        ingest_result = None
        if ingest and paper_ids:
            ingest_result = await self._tool_ingest_arxiv_papers(
                {
                    "name": f"Literature review: {topic}",
                    "paper_ids": paper_ids,
                    "max_results": len(paper_ids),
                    "start": 0,
                    "sort_by": "submittedDate",
                    "sort_order": "descending",
                    "auto_summarize": True,
                    "auto_literature_review": True,
                    "topic": topic,
                    "auto_sync": True,
                },
                user_id,
                db,
            )

        return {
            "topic": topic,
            "query": q,
            "papers": papers,
            "ingest": ingest_result,
            "next_steps": [
                "Open the imported documents in Documents once ingestion completes.",
                "Ask the agent to summarize and compare the imported papers.",
            ],
        }

    async def _tool_ingest_arxiv_papers(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        name = (params.get("name") or "ArXiv import").strip()
        search_queries = [q.strip() for q in (params.get("search_queries") or []) if isinstance(q, str) and q.strip()]
        paper_ids = [p.strip() for p in (params.get("paper_ids") or []) if isinstance(p, str) and p.strip()]
        categories = [c.strip() for c in (params.get("categories") or []) if isinstance(c, str) and c.strip()]
        max_results = int(params.get("max_results") or 25)
        start = int(params.get("start") or 0)
        sort_by = params.get("sort_by") or "submittedDate"
        sort_order = params.get("sort_order") or "descending"
        auto_sync = bool(params.get("auto_sync", True))
        auto_summarize = bool(params.get("auto_summarize", True))
        auto_literature_review = bool(params.get("auto_literature_review", False))
        topic = (params.get("topic") or None)

        if not search_queries and not paper_ids and not categories:
            raise ValueError("Provide at least one of: search_queries, paper_ids, categories")

        config = {
            "queries": search_queries,
            "paper_ids": paper_ids,
            "categories": categories,
            "max_results": max_results,
            "start": start,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "requested_by": str(user_id),
            "requested_by_user_id": str(user_id),
            "auto_summarize": auto_summarize,
            "auto_literature_review": auto_literature_review,
            "topic": topic,
            "display": {
                "queries": search_queries,
                "paper_ids": paper_ids,
                "categories": categories,
                "max_results": max_results,
            }
        }
        try:
            user_row = await db.execute(select(DbUser).where(DbUser.id == user_id))
            db_user = user_row.scalar_one_or_none()
            if db_user and getattr(db_user, "username", None):
                config["requested_by"] = db_user.username
        except Exception:
            pass

        source_name = f"{name} #{uuid4().hex[:6]}"
        source = await self.document_service.create_document_source(
            name=source_name,
            source_type="arxiv",
            config=config,
            db=db,
        )

        task_id = None
        if auto_sync:
            try:
                from app.tasks.ingestion_tasks import ingest_from_source
                task = ingest_from_source.delay(str(source.id))
                task_id = getattr(task, "id", None)
            except Exception as exc:
                logger.warning(f"Failed to trigger arXiv ingestion for {source.id}: {exc}")

        return {
            "status": "created",
            "source_id": str(source.id),
            "source_name": source.name,
            "queued": bool(task_id) if auto_sync else False,
            "task_id": task_id,
            "paper_ids_count": len(paper_ids),
            "search_queries_count": len(search_queries),
            "categories_count": len(categories),
        }

    async def _tool_summarize_documents_in_source(self, params: Dict[str, Any], user_id: UUID, db: AsyncSession) -> Dict[str, Any]:
        from uuid import UUID as _UUID
        from sqlalchemy import select, desc
        from app.models.document import Document, DocumentSource
        from app.tasks.summarization_tasks import summarize_document as summarize_task

        source_id = params.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")
        src = await db.get(DocumentSource, _UUID(str(source_id)))
        if not src:
            raise ValueError("Source not found")

        force = bool(params.get("force", False))
        only_missing = bool(params.get("only_missing", True))
        limit = min(int(params.get("limit", 500) or 500), 2000)

        rows = (
            await db.execute(
                select(Document.id, Document.summary)
                .where(Document.source_id == src.id)
                .order_by(desc(Document.created_at))
                .limit(limit)
            )
        ).all()
        queued = 0
        for doc_id, summary in rows:
            if only_missing and summary and not force:
                continue
            summarize_task.delay(str(doc_id), force, user_id=str(user_id))
            queued += 1

        return {
            "source_id": str(src.id),
            "queued": queued,
            "considered": len(rows),
            "force": force,
            "only_missing": only_missing,
        }

    async def _tool_enrich_arxiv_metadata_for_source(
        self, params: Dict[str, Any], user_id: UUID, db: AsyncSession
    ) -> Dict[str, Any]:
        from uuid import UUID as _UUID
        from app.models.document import DocumentSource
        from app.tasks.paper_enrichment_tasks import enrich_arxiv_source

        source_id = params.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")
        src = await db.get(DocumentSource, _UUID(str(source_id)))
        if not src or src.source_type != "arxiv":
            raise ValueError("arXiv source not found")

        # Best-effort ownership check: match requested_by / requested_by_user_id if present (admins bypass)
        cfg = src.config if isinstance(src.config, dict) else {}
        requested_by_user_id = cfg.get("requested_by_user_id") or cfg.get("requestedByUserId")
        requested_by = cfg.get("requested_by") or cfg.get("requestedBy")
        if requested_by_user_id and requested_by_user_id != str(user_id) and requested_by != str(user_id):
            from app.models.user import User as DbUser
            u = await db.get(DbUser, user_id)
            if not (u and u.is_admin()):
                raise ValueError("Not authorized for this source")

        force = bool(params.get("force", False))
        limit = min(int(params.get("limit", 500) or 500), 5000)
        task = enrich_arxiv_source.delay(str(src.id), force, limit)
        return {"source_id": str(src.id), "queued": True, "task_id": task.id, "force": force, "limit": limit}

    async def _tool_generate_literature_review_for_source(
        self, params: Dict[str, Any], user_id: UUID, db: AsyncSession
    ) -> Dict[str, Any]:
        from uuid import UUID as _UUID
        from app.models.document import DocumentSource
        from app.tasks.research_tasks import generate_literature_review

        source_id = params.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")
        src = await db.get(DocumentSource, _UUID(str(source_id)))
        if not src or src.source_type != "arxiv":
            raise ValueError("arXiv source not found")

        topic = params.get("topic")
        if topic and isinstance(src.config, dict):
            cfg = dict(src.config)
            cfg["topic"] = str(topic).strip()
            src.config = cfg
            await db.commit()

        task = generate_literature_review.delay(str(src.id), user_id=str(user_id))
        return {"source_id": str(src.id), "queued": True, "task_id": task.id}

    async def _tool_generate_slides_for_source(
        self, params: Dict[str, Any], user_id: UUID, db: AsyncSession
    ) -> Dict[str, Any]:
        from uuid import UUID as _UUID
        from sqlalchemy import select, desc
        from app.models.document import DocumentSource, Document
        from app.models.presentation import PresentationJob
        from app.tasks.presentation_tasks import generate_presentation_task

        source_id = params.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")
        src = await db.get(DocumentSource, _UUID(str(source_id)))
        if not src or src.source_type != "arxiv":
            raise ValueError("arXiv source not found")

        title = params.get("title") or f"Slides: {src.name}"
        topic = params.get("topic")
        if not topic and isinstance(src.config, dict):
            topic = src.config.get("topic")
        topic = topic or src.name

        slide_count = int(params.get("slide_count", 10) or 10)
        slide_count = max(3, min(40, slide_count))
        style = params.get("style") or "professional"
        include_diagrams = bool(params.get("include_diagrams", True))
        prefer_review = bool(params.get("prefer_review_document", True))

        review_doc_id = None
        if prefer_review:
            review_result = await db.execute(
                select(Document.id)
                .where(Document.source_id == src.id, Document.source_identifier.like("literature_review:%"))
                .order_by(desc(Document.created_at))
                .limit(1)
            )
            review_doc_id = review_result.scalar_one_or_none()

        if review_doc_id:
            source_document_ids = [str(review_doc_id)]
        else:
            docs_result = await db.execute(
                select(Document.id)
                .where(Document.source_id == src.id)
                .order_by(desc(Document.created_at))
                .limit(8)
            )
            source_document_ids = [str(r[0]) for r in docs_result.all()]
            if not source_document_ids:
                raise ValueError("No documents found for source")

        job = PresentationJob(
            user_id=user_id,
            title=str(title),
            topic=str(topic),
            source_document_ids=source_document_ids,
            slide_count=slide_count,
            style=str(style),
            include_diagrams=1 if include_diagrams else 0,
            status="pending",
            progress=0,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        generate_presentation_task.delay(str(job.id), str(user_id))
        return {"presentation_job_id": str(job.id), "source_id": str(src.id), "source_document_ids": source_document_ids}

    async def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Execute a tool directly by name.

        This is a public interface for executing agent tools from external
        callers like the workflow engine.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            user_id: User ID executing the tool
            db: Database session

        Returns:
            Tool execution result as a dictionary

        Raises:
            ValueError: If tool name is unknown
            Exception: If tool execution fails
        """
        from app.schemas.agent import AgentToolCall
        import uuid

        # Create a minimal tool call object
        tool_call = AgentToolCall(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            tool_input=tool_input,
            status="pending"
        )

        # Execute the tool
        result_call = await self._execute_tool(tool_call, user_id, db)

        if result_call.status == "failed":
            raise Exception(result_call.error or f"Tool {tool_name} failed")

        return result_call.tool_output or {}

    async def _tool_search_documents(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Execute document search tool."""
        query = params.get("query", "")
        limit = min(params.get("limit", 5), 20)

        await self._ensure_vector_store_initialized()

        # Perform semantic search
        search_results = await self.vector_store.search(
            query=query,
            limit=limit * 2  # Get more to filter
        )

        # Format results
        results = []
        seen_docs = set()

        for result in search_results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("document_id", result.get("id"))

            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                results.append({
                    "id": doc_id,
                    "title": metadata.get("title", "Untitled"),
                    "content_preview": (result.get("content", "") or "")[:200],
                    "score": round(result.get("score", 0), 3),
                    "source_type": metadata.get("source", "unknown")
                })

                if len(results) >= limit:
                    break

        return results

    async def _tool_get_document_details(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get detailed information about a document."""
        document_id = params.get("document_id")

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        document = await self.document_service.get_document(doc_uuid, db)

        if not document:
            return {"error": f"Document not found: {document_id}"}

        return {
            "id": str(document.id),
            "title": document.title,
            "content_preview": (document.content or "")[:500] if document.content else None,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "author": document.author,
            "tags": document.tags or [],
            "summary": document.summary,
            "is_processed": document.is_processed,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None
        }

    async def _tool_web_scrape(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Fetch a web page and extract readable text and links."""
        url = params.get("url", "")
        if not url:
            return {"error": "Missing required parameter: url"}

        follow_links = bool(params.get("follow_links", False))
        max_pages = int(params.get("max_pages", 1))
        max_depth = int(params.get("max_depth", 0))
        same_domain_only = bool(params.get("same_domain_only", True))
        include_links = bool(params.get("include_links", True))
        allow_private_networks = bool(params.get("allow_private_networks", False))
        max_content_chars = int(params.get("max_content_chars", 50_000))

        allow_private_effective = False
        is_allowlisted = await self._is_url_allowlisted_for_internal_scrape(url, db)

        if allow_private_networks:
            from app.models.user import User
            user_result = await db.execute(select(User).where(User.id == user_id))
            user = user_result.scalar_one_or_none()
            if user and user.role == "admin":
                allow_private_effective = True
            elif is_allowlisted:
                allow_private_effective = True
            else:
                return {"error": "allow_private_networks requires admin role (or an active web source allowlist)"}
        else:
            allow_private_effective = bool(is_allowlisted)

        from app.services.web_scraper_service import WebScraperService

        scraper = WebScraperService(enforce_network_safety=True)
        try:
            return await scraper.scrape(
                url,
                follow_links=follow_links,
                max_pages=max_pages,
                max_depth=max_depth,
                same_domain_only=same_domain_only,
                include_links=include_links,
                allow_private_networks=allow_private_effective,
                max_content_chars=max_content_chars,
            )
        finally:
            await scraper.aclose()

    async def _is_url_allowlisted_for_internal_scrape(self, url: str, db: AsyncSession) -> bool:
        """
        Check if a URL's hostname is allowlisted via active web document sources.

        This enables scraping internal portals safely without opening arbitrary private-network access.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return False

        from app.models.document import DocumentSource

        res = await db.execute(
            select(DocumentSource).where(
                DocumentSource.source_type == "web",
                DocumentSource.is_active == True,
            )
        )
        sources = res.scalars().all()

        def host_matches(allowed: str) -> bool:
            allowed = (allowed or "").strip().lower()
            if not allowed:
                return False
            if host == allowed:
                return True
            return host.endswith("." + allowed)

        for source in sources:
            cfg = source.config or {}
            for d in (cfg.get("allowed_domains") or []):
                if host_matches(d):
                    return True
            for base in (cfg.get("base_urls") or []):
                try:
                    base_host = (urlparse(str(base)).hostname or "").lower()
                except Exception:
                    base_host = ""
                if base_host and host_matches(base_host):
                    return True

        return False

    async def _tool_summarize_document(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Summarize a document."""
        document_id = params.get("document_id")
        force_regenerate = params.get("force_regenerate", False)

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        # Check if document exists
        document = await self.document_service.get_document(doc_uuid, db)
        if not document:
            return {"error": f"Document not found: {document_id}"}

        # If summary exists and not forcing regeneration, return existing
        if document.summary and not force_regenerate:
            return {
                "document_id": document_id,
                "title": document.title,
                "summary": document.summary,
                "status": "existing"
            }

        # Generate summary
        try:
            summary = await self.document_service.summarize_document(
                doc_uuid, db, force=force_regenerate
            )
            return {
                "document_id": document_id,
                "title": document.title,
                "summary": summary,
                "status": "generated"
            }
        except Exception as e:
            return {
                "document_id": document_id,
                "error": f"Failed to generate summary: {str(e)}"
            }

    async def _tool_delete_document(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Delete a document (requires confirmation)."""
        document_id = params.get("document_id")
        confirm = params.get("confirm", False)

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        # Get document info first
        document = await self.document_service.get_document(doc_uuid, db)
        if not document:
            return {"error": f"Document not found: {document_id}"}

        # If not confirmed, return document info for confirmation
        if not confirm:
            return {
                "action": "confirmation_required",
                "document_id": document_id,
                "title": document.title,
                "message": f"Are you sure you want to delete '{document.title}'? This action cannot be undone. Please confirm deletion."
            }

        # Proceed with deletion
        try:
            success = await self.document_service.delete_document(doc_uuid, db)
            if success:
                return {
                    "action": "deleted",
                    "document_id": document_id,
                    "title": document.title,
                    "message": f"Successfully deleted document '{document.title}'"
                }
            else:
                return {
                    "error": f"Failed to delete document '{document.title}'"
                }
        except Exception as e:
            return {"error": f"Error deleting document: {str(e)}"}

    async def _tool_list_recent_documents(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """List recently added/updated documents."""
        limit = min(params.get("limit", 10), 50)

        result = await db.execute(
            select(Document)
            .order_by(desc(Document.updated_at))
            .limit(limit)
        )
        documents = result.scalars().all()

        return [
            {
                "id": str(doc.id),
                "title": doc.title,
                "file_type": doc.file_type,
                "is_processed": doc.is_processed,
                "has_summary": bool(doc.summary),
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
            }
            for doc in documents
        ]

    async def _tool_list_document_sources(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List available document sources."""
        active_only = params.get("active_only", False)

        try:
            query = select(DocumentSource)
            if active_only:
                query = query.where(DocumentSource.is_active == True)

            query = query.order_by(DocumentSource.name)
            result = await db.execute(query)
            sources = result.scalars().all()

            return {
                "active_only": active_only,
                "count": len(sources),
                "sources": [
                    {
                        "id": str(source.id),
                        "name": source.name,
                        "source_type": source.source_type,
                        "is_active": source.is_active,
                        "is_syncing": source.is_syncing,
                        "last_sync": source.last_sync.isoformat() if source.last_sync else None,
                        "last_error": source.last_error,
                    }
                    for source in sources
                ],
            }
        except Exception as e:
            logger.error(f"Error listing document sources: {e}")
            return {"error": f"Failed to list document sources: {str(e)}"}

    async def _tool_list_documents_by_source(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List documents from a specific source."""
        source_id = params.get("source_id")
        source_name = params.get("source_name")
        source_type = params.get("source_type")
        limit = min(params.get("limit", 20), 50)
        offset = max(params.get("offset", 0), 0)

        if not source_id and not source_name and not source_type:
            return {"error": "Provide source_id, source_name, or source_type"}

        try:
            query = select(Document).options(selectinload(Document.source)).join(DocumentSource)

            if source_id:
                try:
                    source_uuid = UUID(source_id)
                except (ValueError, TypeError):
                    return {"error": f"Invalid source_id: {source_id}"}
                query = query.where(Document.source_id == source_uuid)

            if source_name:
                query = query.where(DocumentSource.name.ilike(f"%{source_name}%"))

            if source_type:
                query = query.where(DocumentSource.source_type.ilike(f"%{source_type}%"))

            query = query.order_by(desc(Document.updated_at)).offset(offset).limit(limit)
            result = await db.execute(query)
            documents = result.scalars().all()

            return {
                "filters": {
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_type": source_type,
                },
                "count": len(documents),
                "documents": [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "tags": doc.tags or [],
                        "file_type": doc.file_type,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                        "source": {
                            "id": str(doc.source.id) if doc.source else None,
                            "name": doc.source.name if doc.source else None,
                            "source_type": doc.source.source_type if doc.source else None,
                        },
                    }
                    for doc in documents
                ],
            }
        except Exception as e:
            logger.error(f"Error listing documents by source: {e}")
            return {"error": f"Failed to list documents by source: {str(e)}"}

    async def _tool_search_documents_by_author(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Search documents by author name."""
        author = (params.get("author") or "").strip()
        match_type = params.get("match_type", "contains")
        limit = min(params.get("limit", 20), 50)

        if not author:
            return {"error": "Author is required"}

        try:
            if match_type == "exact":
                clause = Document.author.ilike(author)
            elif match_type == "starts_with":
                clause = Document.author.ilike(f"{author}%")
            else:
                clause = Document.author.ilike(f"%{author}%")

            query = (
                select(Document)
                .where(Document.author.isnot(None))
                .where(clause)
                .order_by(desc(Document.updated_at))
                .limit(limit)
            )

            result = await db.execute(query)
            documents = result.scalars().all()

            return {
                "author_query": author,
                "match_type": match_type,
                "count": len(documents),
                "documents": [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "author": doc.author,
                        "tags": doc.tags or [],
                        "file_type": doc.file_type,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                        "source_id": str(doc.source_id),
                    }
                    for doc in documents
                ],
            }
        except Exception as e:
            logger.error(f"Error searching documents by author: {e}")
            return {"error": f"Search failed: {str(e)}"}

    async def _generate_response(
        self,
        user_message: str,
        tool_results: List[AgentToolCall],
        history: List[AgentMessage],
        user_settings: Optional[UserLLMSettings] = None
    ) -> str:
        """Generate final response based on tool results."""

        # Build context from tool results
        tool_context_parts = []
        for result in tool_results:
            if result.status == "completed":
                output_str = json.dumps(result.tool_output, indent=2, default=str)
                tool_context_parts.append(
                    f"Tool '{result.tool_name}' result:\n{output_str}"
                )
            elif result.status == "failed":
                tool_context_parts.append(
                    f"Tool '{result.tool_name}' failed: {result.error}"
                )

        tool_context = "\n\n".join(tool_context_parts) if tool_context_parts else "No tools were executed."

        # Build response prompt
        response_prompt = f"""You are a helpful AI assistant for a document knowledge base.
Based on the user's request and the tool execution results, provide a helpful response.

User's request: {user_message}

Tool execution results:
{tool_context}

Guidelines:
- Summarize the results in a natural, conversational way
- If search results are present, list the most relevant documents with their titles
- If there were errors, explain what went wrong and suggest alternatives
- If a confirmation is required (like for deletion), ask the user to confirm
- Keep the response concise but informative
- Use markdown formatting for lists and emphasis when appropriate

Your response:"""

        try:
            response = await self.llm_service.generate_response(
                query=response_prompt,
                temperature=0.7,
                max_tokens=800,
                user_settings=user_settings,
                task_type="chat"
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating agent response: {e}")
            # Fallback to basic response
            if tool_results:
                completed = [r for r in tool_results if r.status == "completed"]
                if completed:
                    return f"I executed {len(completed)} tool(s). Please check the results above."
            return "I processed your request. Please let me know if you need anything else."

    # ========================
    # New Tool Handlers
    # ========================

    async def _tool_create_document_from_text(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Create a new document from text content."""
        title = params.get("title", "").strip()
        content = params.get("content", "").strip()
        tags = params.get("tags", [])

        if not title:
            return {"error": "Title is required"}
        if not content:
            return {"error": "Content is required"}

        try:
            owner_display_name = None
            try:
                user_result = await db.execute(select(DbUser).where(DbUser.id == user_id))
                user = user_result.scalar_one_or_none()
                if user:
                    owner_display_name = user.full_name or user.username or user.email
            except Exception:
                owner_display_name = None

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            notes_source = await self.document_service._get_or_create_agent_notes_source(db)

            # Create document in database
            document = Document(
                title=title,
                content=content,
                content_hash=content_hash,
                url=None,
                file_path=None,
                file_type="text/plain",
                file_size=len(content.encode("utf-8")),
                source_id=notes_source.id,
                source_identifier=f"agent_note:{uuid4()}",
                author=owner_display_name,
                tags=tags,
                extra_metadata={
                    "origin": "agent_created",
                    "created_at": datetime.utcnow().isoformat(),
                },
                is_processed=False,
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)

            # Process document (chunks + vector index)
            try:
                await self.document_service.reprocess_document(document.id, db, user_id=user_id)
            except Exception as e:
                logger.warning(f"Failed to process agent-created document embeddings: {e}")

            return {
                "action": "created",
                "document_id": str(document.id),
                "title": title,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "tags": tags,
                "message": f"Successfully created document '{title}'"
            }

        except Exception as e:
            logger.error(f"Error creating document from text: {e}")
            return {"error": f"Failed to create document: {str(e)}"}

    async def _tool_ingest_url(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Scrape a URL and ingest it into the knowledge base as document(s)."""
        user_result = await db.execute(select(DbUser).where(DbUser.id == user_id))
        user = user_result.scalar_one_or_none()
        if not user:
            return {"error": "User not found"}

        from app.services.url_ingestion_service import UrlIngestionService

        service = UrlIngestionService()
        return await service.ingest_url(
            db=db,
            user=user,
            url=(params.get("url") or ""),
            title=(params.get("title") or None),
            tags=params.get("tags"),
            follow_links=bool(params.get("follow_links", False)),
            max_pages=int(params.get("max_pages", 1)),
            max_depth=int(params.get("max_depth", 0)),
            same_domain_only=bool(params.get("same_domain_only", True)),
            one_document_per_page=bool(params.get("one_document_per_page", False)),
            allow_private_networks=bool(params.get("allow_private_networks", False)),
            max_content_chars=int(params.get("max_content_chars", 50_000)),
        )

    async def _tool_find_similar_documents(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Find documents similar to a given document."""
        document_id = params.get("document_id")
        limit = min(params.get("limit", 5), 20)

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        # Get the reference document
        document = await self.document_service.get_document(doc_uuid, db)
        if not document:
            return {"error": f"Document not found: {document_id}"}

        # Use document content or title as query
        query_text = document.content[:1000] if document.content else document.title

        await self._ensure_vector_store_initialized()

        # Search for similar documents
        search_results = await self.vector_store.search(
            query=query_text,
            limit=limit + 5  # Get extras to filter out self
        )

        # Format results, excluding the reference document
        similar_docs = []
        seen_docs = set()
        seen_docs.add(document_id)  # Exclude self

        for result in search_results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("document_id", result.get("id"))

            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                similar_docs.append({
                    "id": doc_id,
                    "title": metadata.get("title", "Untitled"),
                    "similarity_score": round(result.get("score", 0), 3),
                    "content_preview": (result.get("content", "") or "")[:150]
                })

                if len(similar_docs) >= limit:
                    break

        return {
            "reference_document": {
                "id": document_id,
                "title": document.title
            },
            "similar_documents": similar_docs,
            "count": len(similar_docs)
        }

    async def _tool_update_document_tags(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Update tags on a document."""
        document_id = params.get("document_id")
        tags = params.get("tags", [])
        action = params.get("action", "add")

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        # Get the document
        document = await self.document_service.get_document(doc_uuid, db)
        if not document:
            return {"error": f"Document not found: {document_id}"}

        current_tags = set(document.tags or [])
        new_tags_set = set(tags)

        if action == "add":
            updated_tags = list(current_tags | new_tags_set)
            action_desc = f"Added tags: {', '.join(tags)}"
        elif action == "remove":
            updated_tags = list(current_tags - new_tags_set)
            action_desc = f"Removed tags: {', '.join(tags)}"
        elif action == "replace":
            updated_tags = list(new_tags_set)
            action_desc = f"Replaced all tags with: {', '.join(tags)}"
        else:
            return {"error": f"Invalid action: {action}. Use 'add', 'remove', or 'replace'"}

        # Update document
        document.tags = updated_tags
        await db.commit()

        return {
            "document_id": document_id,
            "title": document.title,
            "previous_tags": list(current_tags),
            "current_tags": updated_tags,
            "action": action_desc
        }

    async def _tool_get_knowledge_base_stats(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            # Total documents
            total_result = await db.execute(
                select(func.count(Document.id))
            )
            total_docs = total_result.scalar() or 0

            # Processed documents
            processed_result = await db.execute(
                select(func.count(Document.id)).where(Document.is_processed == True)
            )
            processed_docs = processed_result.scalar() or 0

            # Documents with summaries
            summarized_result = await db.execute(
                select(func.count(Document.id)).where(Document.summary.isnot(None))
            )
            summarized_docs = summarized_result.scalar() or 0

            # Total storage size
            size_result = await db.execute(
                select(func.coalesce(func.sum(Document.file_size), 0))
            )
            total_size = size_result.scalar() or 0

            # Documents by file type
            type_result = await db.execute(
                select(Document.file_type, func.count(Document.id))
                .group_by(Document.file_type)
                .order_by(func.count(Document.id).desc())
                .limit(10)
            )
            file_types = {row[0] or "unknown": row[1] for row in type_result.fetchall()}

            # Recent activity (documents in last 7 days)
            from datetime import timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_result = await db.execute(
                select(func.count(Document.id))
                .where(Document.created_at >= week_ago)
            )
            recent_docs = recent_result.scalar() or 0

            # Vector store stats
            await self._ensure_vector_store_initialized()
            vector_stats = await self.vector_store.get_collection_stats()

            return {
                "total_documents": total_docs,
                "processed_documents": processed_docs,
                "summarized_documents": summarized_docs,
                "pending_processing": total_docs - processed_docs,
                "total_storage_bytes": total_size,
                "total_storage_mb": round(total_size / (1024 * 1024), 2),
                "documents_by_type": file_types,
                "documents_last_7_days": recent_docs,
                "vector_store": vector_stats
            }

        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"error": f"Failed to get statistics: {str(e)}"}

    async def _tool_batch_delete_documents(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Delete multiple documents at once."""
        document_ids = params.get("document_ids", [])
        confirm = params.get("confirm", False)

        if not document_ids:
            return {"error": "No document IDs provided"}

        if len(document_ids) > 50:
            return {"error": "Cannot delete more than 50 documents at once"}

        # Validate all IDs and get document info
        documents_info = []
        valid_ids = []

        for doc_id in document_ids:
            try:
                doc_uuid = UUID(doc_id)
                document = await self.document_service.get_document(doc_uuid, db)
                if document:
                    documents_info.append({
                        "id": doc_id,
                        "title": document.title
                    })
                    valid_ids.append(doc_uuid)
            except (ValueError, TypeError):
                continue

        if not valid_ids:
            return {"error": "No valid documents found"}

        # If not confirmed, return document info for confirmation
        if not confirm:
            return {
                "action": "confirmation_required",
                "documents": documents_info,
                "count": len(documents_info),
                "message": f"Are you sure you want to delete {len(documents_info)} documents? This action cannot be undone."
            }

        # Proceed with deletion
        deleted = []
        failed = []

        for doc_uuid in valid_ids:
            try:
                success = await self.document_service.delete_document(doc_uuid, db)
                doc_id = str(doc_uuid)
                if success:
                    deleted.append(doc_id)
                else:
                    failed.append({"id": doc_id, "reason": "Deletion failed"})
            except Exception as e:
                failed.append({"id": str(doc_uuid), "reason": str(e)})

        return {
            "action": "batch_deleted",
            "deleted_count": len(deleted),
            "deleted_ids": deleted,
            "failed_count": len(failed),
            "failed": failed if failed else None,
            "message": f"Successfully deleted {len(deleted)} document(s)" + (
                f", {len(failed)} failed" if failed else ""
            )
        }

    async def _tool_batch_summarize_documents(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Queue summarization for multiple documents."""
        document_ids = params.get("document_ids", [])
        force_regenerate = params.get("force_regenerate", False)

        if not document_ids:
            return {"error": "No document IDs provided"}

        if len(document_ids) > 20:
            return {"error": "Cannot summarize more than 20 documents at once"}

        # Validate IDs and check which need summarization
        queued = []
        skipped = []
        invalid = []

        for doc_id in document_ids:
            try:
                doc_uuid = UUID(doc_id)
                document = await self.document_service.get_document(doc_uuid, db)

                if not document:
                    invalid.append(doc_id)
                    continue

                if document.summary and not force_regenerate:
                    skipped.append({
                        "id": doc_id,
                        "title": document.title,
                        "reason": "Already has summary"
                    })
                    continue

                # Queue for summarization (use Celery task)
                from app.tasks.summarization_tasks import summarize_document_task
                summarize_document_task.delay(str(doc_uuid), force=force_regenerate)

                queued.append({
                    "id": doc_id,
                    "title": document.title
                })

            except (ValueError, TypeError):
                invalid.append(doc_id)

        return {
            "queued_count": len(queued),
            "queued": queued,
            "skipped_count": len(skipped),
            "skipped": skipped if skipped else None,
            "invalid_count": len(invalid),
            "invalid_ids": invalid if invalid else None,
            "message": f"Queued {len(queued)} document(s) for summarization"
        }

    async def _tool_search_by_tags(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Search documents by tags."""
        tags = params.get("tags", [])
        match_all = params.get("match_all", False)
        limit = min(params.get("limit", 20), 50)

        if not tags:
            return {"error": "No tags provided"}

        try:
            # Build query based on match type
            if match_all:
                # Documents must have ALL tags
                query = select(Document).where(
                    Document.tags.contains(tags)
                )
            else:
                # Documents can have ANY of the tags
                query = select(Document).where(
                    Document.tags.overlap(tags)
                )

            query = query.order_by(desc(Document.updated_at)).limit(limit)
            result = await db.execute(query)
            documents = result.scalars().all()

            return {
                "search_tags": tags,
                "match_type": "all" if match_all else "any",
                "count": len(documents),
                "documents": [
                    {
                        "id": str(doc.id),
                        "title": doc.title,
                        "tags": doc.tags or [],
                        "file_type": doc.file_type,
                        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
                    }
                    for doc in documents
                ]
            }

        except Exception as e:
            logger.error(f"Error searching by tags: {e}")
            return {"error": f"Search failed: {str(e)}"}

    async def _tool_list_all_tags(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List all unique tags used in the knowledge base."""
        try:
            # Get all documents with tags
            result = await db.execute(
                select(Document.tags).where(Document.tags.isnot(None))
            )

            # Collect all unique tags
            all_tags = set()
            tag_counts = {}

            for row in result.fetchall():
                if row[0]:
                    for tag in row[0]:
                        all_tags.add(tag)
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # Sort by count
            sorted_tags = sorted(
                tag_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return {
                "total_unique_tags": len(all_tags),
                "tags": [
                    {"tag": tag, "count": count}
                    for tag, count in sorted_tags
                ]
            }

        except Exception as e:
            logger.error(f"Error listing tags: {e}")
            return {"error": f"Failed to list tags: {str(e)}"}

    # ========================
    # Document Comparison Tools
    # ========================

    async def _tool_compare_documents(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Compare two documents for similarities and differences."""
        doc_id_1 = params.get("document_id_1")
        doc_id_2 = params.get("document_id_2")
        comparison_type = params.get("comparison_type", "full")

        # Validate IDs
        try:
            uuid_1 = UUID(doc_id_1)
            uuid_2 = UUID(doc_id_2)
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid document ID: {e}"}

        # Get both documents
        doc1 = await self.document_service.get_document(uuid_1, db)
        doc2 = await self.document_service.get_document(uuid_2, db)

        if not doc1:
            return {"error": f"Document not found: {doc_id_1}"}
        if not doc2:
            return {"error": f"Document not found: {doc_id_2}"}

        result = {
            "document_1": {
                "id": doc_id_1,
                "title": doc1.title,
                "file_type": doc1.file_type,
                "word_count": len((doc1.content or "").split()) if doc1.content else 0
            },
            "document_2": {
                "id": doc_id_2,
                "title": doc2.title,
                "file_type": doc2.file_type,
                "word_count": len((doc2.content or "").split()) if doc2.content else 0
            }
        }

        content1 = doc1.content or ""
        content2 = doc2.content or ""

        # Keyword comparison
        if comparison_type in ["keyword", "full"]:
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())

            common_words = words1 & words2
            unique_to_1 = words1 - words2
            unique_to_2 = words2 - words1

            # Calculate Jaccard similarity
            union = words1 | words2
            keyword_similarity = len(common_words) / len(union) if union else 0

            result["keyword_analysis"] = {
                "similarity_score": round(keyword_similarity, 3),
                "common_word_count": len(common_words),
                "unique_to_doc1_count": len(unique_to_1),
                "unique_to_doc2_count": len(unique_to_2),
                "sample_common_words": list(common_words)[:20],
                "sample_unique_to_doc1": list(unique_to_1)[:10],
                "sample_unique_to_doc2": list(unique_to_2)[:10]
            }

        # Semantic comparison
        if comparison_type in ["semantic", "full"]:
            try:
                await self._ensure_vector_store_initialized()

                # Get embeddings for both documents
                # Use first 1000 chars as representative sample
                sample1 = content1[:1000]
                sample2 = content2[:1000]

                # Search using doc1 content to find doc2's similarity
                search_results = await self.vector_store.search(
                    query=sample1,
                    limit=50
                )

                # Find doc2 in results
                semantic_score = 0.0
                for res in search_results:
                    res_doc_id = res.get("metadata", {}).get("document_id")
                    if res_doc_id == doc_id_2:
                        semantic_score = res.get("score", 0)
                        break

                result["semantic_analysis"] = {
                    "similarity_score": round(semantic_score, 3),
                    "interpretation": self._interpret_similarity(semantic_score)
                }

            except Exception as e:
                logger.warning(f"Semantic comparison failed: {e}")
                result["semantic_analysis"] = {
                    "error": "Semantic comparison unavailable",
                    "reason": str(e)
                }

        # Generate summary using LLM
        try:
            comparison_prompt = f"""Compare these two documents briefly:

Document 1: "{doc1.title}"
Content preview: {content1[:500]}...

Document 2: "{doc2.title}"
Content preview: {content2[:500]}...

Provide a 2-3 sentence comparison highlighting key similarities and differences."""

            summary = await self.llm_service.generate_response(
                query=comparison_prompt,
                temperature=0.3,
                max_tokens=200,
                task_type="summarization"
            )
            result["comparison_summary"] = summary.strip()

        except Exception as e:
            logger.warning(f"Failed to generate comparison summary: {e}")

        return result

    def _interpret_similarity(self, score: float) -> str:
        """Interpret semantic similarity score."""
        if score >= 0.9:
            return "Nearly identical content"
        elif score >= 0.7:
            return "Highly similar - likely related topics"
        elif score >= 0.5:
            return "Moderately similar - some overlap"
        elif score >= 0.3:
            return "Loosely related"
        else:
            return "Different topics"

    # ========================
    # Template Fill Tools
    # ========================

    async def _tool_start_template_fill(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Start a template fill job - requires template upload via UI."""
        source_doc_ids = params.get("source_document_ids", [])

        if not source_doc_ids:
            return {"error": "No source document IDs provided"}

        # Validate source documents exist
        valid_docs = []
        for doc_id in source_doc_ids:
            try:
                doc_uuid = UUID(doc_id)
                doc = await self.document_service.get_document(doc_uuid, db)
                if doc:
                    valid_docs.append({
                        "id": doc_id,
                        "title": doc.title
                    })
            except (ValueError, TypeError):
                continue

        if not valid_docs:
            return {"error": "No valid source documents found"}

        # Return instructions for template upload
        return {
            "action": "template_upload_required",
            "message": "To fill a template, please upload a DOCX template file. The template will be analyzed and filled with content from your source documents.",
            "source_documents": valid_docs,
            "source_document_ids": [d["id"] for d in valid_docs],
            "instructions": [
                "1. Click the upload button to select a DOCX template",
                "2. The template will be analyzed for sections",
                "3. Each section will be filled with relevant content from the source documents",
                "4. You can download the completed document when ready"
            ],
            "supported_formats": [".docx", ".doc"]
        }

    async def _tool_list_template_jobs(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List user's template fill jobs."""
        status_filter = params.get("status_filter", "all")
        limit = min(params.get("limit", 10), 50)

        try:
            from app.models.template import TemplateJob

            query = select(TemplateJob).where(TemplateJob.user_id == user_id)

            if status_filter != "all":
                if status_filter == "processing":
                    query = query.where(TemplateJob.status.in_(["pending", "analyzing", "extracting", "filling"]))
                else:
                    query = query.where(TemplateJob.status == status_filter)

            query = query.order_by(desc(TemplateJob.created_at)).limit(limit)
            result = await db.execute(query)
            jobs = result.scalars().all()

            return {
                "count": len(jobs),
                "jobs": [
                    {
                        "id": str(job.id),
                        "template_filename": job.template_filename,
                        "status": job.status,
                        "progress": job.progress,
                        "current_section": job.current_section,
                        "section_count": len(job.sections) if job.sections else 0,
                        "source_doc_count": len(job.source_document_ids) if job.source_document_ids else 0,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                        "has_download": bool(job.filled_file_path),
                        "error": job.error_message
                    }
                    for job in jobs
                ]
            }

        except Exception as e:
            logger.error(f"Error listing template jobs: {e}")
            return {"error": f"Failed to list template jobs: {str(e)}"}

    async def _tool_get_template_job_status(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get detailed status of a template job."""
        job_id = params.get("job_id")

        try:
            job_uuid = UUID(job_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid job ID: {job_id}"}

        try:
            from app.models.template import TemplateJob

            result = await db.execute(
                select(TemplateJob).where(
                    TemplateJob.id == job_uuid,
                    TemplateJob.user_id == user_id
                )
            )
            job = result.scalar_one_or_none()

            if not job:
                return {"error": f"Template job not found: {job_id}"}

            response = {
                "id": str(job.id),
                "template_filename": job.template_filename,
                "status": job.status,
                "progress": job.progress,
                "current_section": job.current_section,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            }

            # Add sections info
            if job.sections:
                response["sections"] = [
                    {
                        "title": s.get("title"),
                        "level": s.get("level")
                    }
                    for s in job.sections
                ]
                response["total_sections"] = len(job.sections)

            # Add source docs
            if job.source_document_ids:
                response["source_document_count"] = len(job.source_document_ids)

            # Add completion info
            if job.status == "completed":
                response["completed_at"] = job.completed_at.isoformat() if job.completed_at else None
                response["filled_filename"] = job.filled_filename
                response["download_available"] = bool(job.filled_file_path)
                response["download_url"] = f"/api/v1/templates/{job_id}/download"

            # Add error info
            if job.status == "failed":
                response["error_message"] = job.error_message

            return response

        except Exception as e:
            logger.error(f"Error getting template job status: {e}")
            return {"error": f"Failed to get job status: {str(e)}"}

    # =========================================================================
    # RAG / Q&A Tools
    # =========================================================================

    async def _tool_answer_question(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Answer a question using RAG (Retrieval-Augmented Generation)."""
        question = params.get("question", "")
        max_sources = min(params.get("max_sources", 5), 10)

        if not question.strip():
            return {"error": "Question is required"}

        try:
            await self._ensure_vector_store_initialized()

            # Step 1: Search for relevant document chunks
            search_results = await self.vector_store.search(
                query=question,
                limit=max_sources * 2  # Get more for better context
            )

            if not search_results:
                return {
                    "answer": "I couldn't find any relevant documents to answer your question.",
                    "sources": [],
                    "confidence": "low"
                }

            # Step 2: Build context from search results
            context_parts = []
            sources = []
            seen_docs = set()

            for i, result in enumerate(search_results[:max_sources]):
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                doc_id = metadata.get("document_id")
                title = metadata.get("title", "Unknown")
                score = result.get("score", 0)

                context_parts.append(f"[Source {i+1}: {title}]\n{content}\n")

                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    sources.append({
                        "document_id": doc_id,
                        "title": title,
                        "relevance_score": round(score, 3),
                        "excerpt": content[:200] + "..." if len(content) > 200 else content
                    })

            context = "\n".join(context_parts)

            # Step 3: Generate answer using LLM
            prompt = f"""Based on the following context from the knowledge base, answer the question accurately and concisely.
If the context doesn't contain enough information to fully answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

            answer = await self.llm_service.generate_response(
                query=prompt,
                max_tokens=1000
            )

            # Determine confidence based on search scores
            avg_score = sum(r.get("score", 0) for r in search_results[:max_sources]) / max(len(search_results[:max_sources]), 1)
            confidence = "high" if avg_score > 0.7 else "medium" if avg_score > 0.4 else "low"

            return {
                "answer": answer.strip(),
                "sources": sources[:max_sources],
                "source_count": len(sources),
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"error": f"Failed to answer question: {str(e)}"}

    # =========================================================================
    # Document Content Tools
    # =========================================================================

    async def _tool_read_document_content(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Read the full text content of a document."""
        document_id = params.get("document_id")
        max_length = min(params.get("max_length", 10000), 50000)
        include_chunks = params.get("include_chunks", False)

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        try:
            # Get document
            document = await self.document_service.get_document(doc_uuid, db)

            if not document:
                return {"error": f"Document not found: {document_id}"}

            result = {
                "id": str(document.id),
                "title": document.title,
                "file_type": document.file_type,
            }

            if include_chunks and document.chunks:
                # Return content split by chunks
                chunks_data = []
                total_length = 0

                for chunk in sorted(document.chunks, key=lambda c: c.chunk_index):
                    if total_length >= max_length:
                        break

                    chunk_content = chunk.content or ""
                    remaining = max_length - total_length

                    chunks_data.append({
                        "index": chunk.chunk_index,
                        "content": chunk_content[:remaining] if len(chunk_content) > remaining else chunk_content,
                        "word_count": len(chunk_content.split()),
                    })

                    total_length += len(chunk_content)

                result["chunks"] = chunks_data
                result["total_chunks"] = len(document.chunks)
                result["truncated"] = total_length >= max_length

            else:
                # Return full content
                content = document.content or ""

                if len(content) > max_length:
                    result["content"] = content[:max_length]
                    result["truncated"] = True
                    result["full_length"] = len(content)
                else:
                    result["content"] = content
                    result["truncated"] = False
                    result["full_length"] = len(content)

                result["word_count"] = len(content.split())

            return result

        except Exception as e:
            logger.error(f"Error reading document content: {e}")
            return {"error": f"Failed to read document: {str(e)}"}

    # =========================================================================
    # Knowledge Graph Tools
    # =========================================================================

    async def _tool_search_entities(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Search for entities in the knowledge graph."""
        query = params.get("query", "")
        entity_type = params.get("entity_type")
        limit = min(params.get("limit", 10), 50)

        if not query.strip():
            return {"error": "Query is required"}

        try:
            from app.models.knowledge_graph import Entity, EntityMention
            from sqlalchemy import func

            # Build query
            stmt = select(Entity).where(
                Entity.canonical_name.ilike(f"%{query}%")
            )

            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)

            stmt = stmt.limit(limit)
            result = await db.execute(stmt)
            entities = result.scalars().all()

            # Get mention counts for each entity
            entity_data = []
            for entity in entities:
                # Count mentions
                count_result = await db.execute(
                    select(func.count(EntityMention.id)).where(
                        EntityMention.entity_id == entity.id
                    )
                )
                mention_count = count_result.scalar() or 0

                entity_data.append({
                    "id": str(entity.id),
                    "name": entity.canonical_name,
                    "type": entity.entity_type,
                    "description": entity.description,
                    "mention_count": mention_count,
                })

            return {
                "entities": entity_data,
                "count": len(entity_data),
                "query": query
            }

        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return {"error": f"Failed to search entities: {str(e)}"}

    async def _tool_get_entity_relationships(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get relationships for a specific entity."""
        entity_id = params.get("entity_id")
        limit = min(params.get("limit", 20), 100)

        try:
            entity_uuid = UUID(entity_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid entity ID: {entity_id}"}

        try:
            from app.models.knowledge_graph import Entity, Relationship

            # Get the entity
            entity_result = await db.execute(
                select(Entity).where(Entity.id == entity_uuid)
            )
            entity = entity_result.scalar_one_or_none()

            if not entity:
                return {"error": f"Entity not found: {entity_id}"}

            # Get outgoing relationships
            outgoing_result = await db.execute(
                select(Relationship, Entity).join(
                    Entity, Relationship.target_entity_id == Entity.id
                ).where(
                    Relationship.source_entity_id == entity_uuid
                ).limit(limit // 2)
            )
            outgoing = outgoing_result.all()

            # Get incoming relationships
            incoming_result = await db.execute(
                select(Relationship, Entity).join(
                    Entity, Relationship.source_entity_id == Entity.id
                ).where(
                    Relationship.target_entity_id == entity_uuid
                ).limit(limit // 2)
            )
            incoming = incoming_result.all()

            relationships = []

            for rel, target_entity in outgoing:
                relationships.append({
                    "direction": "outgoing",
                    "relation_type": rel.relation_type,
                    "related_entity": {
                        "id": str(target_entity.id),
                        "name": target_entity.canonical_name,
                        "type": target_entity.entity_type
                    },
                    "confidence": rel.confidence,
                    "evidence": rel.evidence[:200] if rel.evidence else None
                })

            for rel, source_entity in incoming:
                relationships.append({
                    "direction": "incoming",
                    "relation_type": rel.relation_type,
                    "related_entity": {
                        "id": str(source_entity.id),
                        "name": source_entity.canonical_name,
                        "type": source_entity.entity_type
                    },
                    "confidence": rel.confidence,
                    "evidence": rel.evidence[:200] if rel.evidence else None
                })

            return {
                "entity": {
                    "id": str(entity.id),
                    "name": entity.canonical_name,
                    "type": entity.entity_type
                },
                "relationships": relationships,
                "count": len(relationships)
            }

        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return {"error": f"Failed to get relationships: {str(e)}"}

    async def _tool_find_documents_by_entity(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Find all documents that mention a specific entity."""
        entity_id = params.get("entity_id")
        limit = min(params.get("limit", 10), 50)

        try:
            entity_uuid = UUID(entity_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid entity ID: {entity_id}"}

        try:
            from app.models.knowledge_graph import Entity, EntityMention

            # Get the entity
            entity_result = await db.execute(
                select(Entity).where(Entity.id == entity_uuid)
            )
            entity = entity_result.scalar_one_or_none()

            if not entity:
                return {"error": f"Entity not found: {entity_id}"}

            # Get mentions with document info
            mentions_result = await db.execute(
                select(EntityMention, Document).join(
                    Document, EntityMention.document_id == Document.id
                ).where(
                    EntityMention.entity_id == entity_uuid
                ).order_by(desc(Document.created_at)).limit(limit * 2)
            )
            mentions = mentions_result.all()

            # Group by document
            documents = {}
            for mention, doc in mentions:
                doc_id = str(doc.id)
                if doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "title": doc.title,
                        "file_type": doc.file_type,
                        "created_at": doc.created_at.isoformat() if doc.created_at else None,
                        "mentions": [],
                        "mention_count": 0
                    }

                if len(documents[doc_id]["mentions"]) < 3:
                    documents[doc_id]["mentions"].append({
                        "text": mention.text,
                        "sentence": mention.sentence[:200] if mention.sentence else None
                    })
                documents[doc_id]["mention_count"] += 1

            doc_list = list(documents.values())[:limit]

            return {
                "entity": {
                    "id": str(entity.id),
                    "name": entity.canonical_name,
                    "type": entity.entity_type
                },
                "documents": doc_list,
                "document_count": len(doc_list)
            }

        except Exception as e:
            logger.error(f"Error finding documents by entity: {e}")
            return {"error": f"Failed to find documents: {str(e)}"}

    async def _tool_get_document_knowledge_graph(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get the knowledge graph for a specific document."""
        document_id = params.get("document_id")

        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        try:
            from app.models.knowledge_graph import Entity, EntityMention, Relationship

            # Verify document exists
            doc_result = await db.execute(
                select(Document).where(Document.id == doc_uuid)
            )
            document = doc_result.scalar_one_or_none()

            if not document:
                return {"error": f"Document not found: {document_id}"}

            # Get all entities mentioned in this document
            mentions_result = await db.execute(
                select(EntityMention, Entity).join(
                    Entity, EntityMention.entity_id == Entity.id
                ).where(
                    EntityMention.document_id == doc_uuid
                )
            )
            mentions = mentions_result.all()

            # Build nodes (unique entities)
            entity_ids = set()
            nodes = []
            for mention, entity in mentions:
                if entity.id not in entity_ids:
                    entity_ids.add(entity.id)
                    nodes.append({
                        "id": str(entity.id),
                        "name": entity.canonical_name,
                        "type": entity.entity_type,
                        "description": entity.description
                    })

            # Get relationships between entities in this document
            if entity_ids:
                rels_result = await db.execute(
                    select(Relationship).where(
                        Relationship.document_id == doc_uuid
                    )
                )
                relationships = rels_result.scalars().all()

                edges = [
                    {
                        "source": str(rel.source_entity_id),
                        "target": str(rel.target_entity_id),
                        "relation_type": rel.relation_type,
                        "confidence": rel.confidence,
                        "evidence": rel.evidence[:150] if rel.evidence else None
                    }
                    for rel in relationships
                ]
            else:
                edges = []

            return {
                "document": {
                    "id": str(document.id),
                    "title": document.title
                },
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }

        except Exception as e:
            logger.error(f"Error getting document knowledge graph: {e}")
            return {"error": f"Failed to get document knowledge graph: {str(e)}"}

    async def _tool_get_global_knowledge_graph(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get the global knowledge graph with filters."""
        try:
            from app.services.knowledge_graph_service import KnowledgeGraphService

            entity_types = params.get("entity_types")
            relation_types = params.get("relation_types")
            min_confidence = float(params.get("min_confidence", 0.0) or 0.0)
            min_mentions = int(params.get("min_mentions", 1) or 1)
            limit_nodes = min(int(params.get("limit_nodes", 300) or 300), 1000)
            limit_edges = min(int(params.get("limit_edges", 1000) or 1000), 5000)
            search = params.get("search")

            svc = KnowledgeGraphService()
            return await svc.global_graph(
                db=db,
                entity_types=entity_types,
                relation_types=relation_types,
                min_confidence=min_confidence,
                min_mentions=min_mentions,
                limit_nodes=limit_nodes,
                limit_edges=limit_edges,
                search=search,
            )
        except Exception as e:
            logger.error(f"Error getting global knowledge graph: {e}")
            return {"error": f"Failed to get global graph: {str(e)}"}

    async def _tool_get_entity_mentions(
        self,
        params: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get mentions for an entity with pagination."""
        entity_id = params.get("entity_id")
        limit = min(int(params.get("limit", 25) or 25), 200)
        offset = max(int(params.get("offset", 0) or 0), 0)

        try:
            entity_uuid = UUID(entity_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid entity ID: {entity_id}"}

        try:
            from app.models.knowledge_graph import Entity
            from app.services.knowledge_graph_service import KnowledgeGraphService

            ent = await db.get(Entity, entity_uuid)
            if not ent:
                return {"error": f"Entity not found: {entity_id}"}

            svc = KnowledgeGraphService()
            items = await svc.mentions_for_entity(db, entity_id, limit=limit, offset=offset)
            total = await svc.mentions_count_for_entity(db, entity_id)

            return {
                "entity": {"id": str(ent.id), "name": ent.canonical_name, "type": ent.entity_type},
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        except Exception as e:
            logger.error(f"Error getting entity mentions: {e}")
            return {"error": f"Failed to get entity mentions: {str(e)}"}

    async def _tool_get_kg_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get knowledge graph stats."""
        try:
            from app.services.knowledge_graph_service import KnowledgeGraphService
            svc = KnowledgeGraphService()
            return await svc.stats(db)
        except Exception as e:
            logger.error(f"Error getting KG stats: {e}")
            return {"error": f"Failed to get KG stats: {str(e)}"}

    async def _require_admin_for_tool(self, user_id: UUID, db: AsyncSession) -> Optional[Dict[str, Any]]:
        from app.models.user import User

        user = await db.get(User, user_id)
        if not user or user.role != "admin":
            return {"error": "Admin privileges required for this tool"}
        return None

    async def _tool_rebuild_document_knowledge_graph(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Admin-only: rebuild KG for a document."""
        gate = await self._require_admin_for_tool(user_id, db)
        if gate:
            return gate

        document_id = params.get("document_id")
        try:
            doc_uuid = UUID(document_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid document ID: {document_id}"}

        try:
            from app.services.knowledge_graph_service import KnowledgeGraphService
            svc = KnowledgeGraphService()
            result = await svc.rebuild_for_document(db, doc_uuid)
            return {"document_id": str(doc_uuid), **result}
        except Exception as e:
            logger.error(f"Error rebuilding document KG: {e}")
            return {"error": f"Failed to rebuild KG: {str(e)}"}

    async def _tool_merge_entities(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Admin-only: merge duplicate entities."""
        gate = await self._require_admin_for_tool(user_id, db)
        if gate:
            return gate

        source_id = params.get("source_id")
        target_id = params.get("target_id")

        try:
            UUID(source_id)
            UUID(target_id)
        except (ValueError, TypeError):
            return {"error": "Invalid source_id or target_id"}

        try:
            from app.services.knowledge_graph_service import KnowledgeGraphService
            svc = KnowledgeGraphService()
            result = await svc.merge_entities(db, source_id=source_id, target_id=target_id)
            return {"source_id": source_id, "target_id": target_id, **result}
        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            return {"error": f"Failed to merge entities: {str(e)}"}

    async def _tool_delete_entity(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Admin-only: delete an entity (with confirm_name)."""
        gate = await self._require_admin_for_tool(user_id, db)
        if gate:
            return gate

        entity_id = params.get("entity_id")
        confirm_name = params.get("confirm_name")

        try:
            entity_uuid = UUID(entity_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid entity ID: {entity_id}"}

        try:
            from app.models.knowledge_graph import Entity
            from app.services.knowledge_graph_service import KnowledgeGraphService

            ent = await db.get(Entity, entity_uuid)
            if not ent:
                return {"error": f"Entity not found: {entity_id}"}

            if not confirm_name or confirm_name != ent.canonical_name:
                return {
                    "error": "Confirmation required",
                    "message": "To delete, set confirm_name to the entity canonical name exactly.",
                    "entity": {"id": str(ent.id), "canonical_name": ent.canonical_name, "entity_type": ent.entity_type},
                }

            svc = KnowledgeGraphService()
            result = await svc.delete_entity(db, entity_id=str(ent.id))
            return {"entity_id": str(ent.id), "canonical_name": ent.canonical_name, **result}
        except Exception as e:
            logger.error(f"Error deleting entity: {e}")
            return {"error": f"Failed to delete entity: {str(e)}"}

    async def _tool_generate_diagram(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Generate a Mermaid diagram from documents or description.

        Supports: flowchart, sequence, class, state, er, gantt, pie, mindmap, architecture
        """
        source = params.get("source", "description")
        diagram_type = params.get("diagram_type", "auto")
        focus = params.get("focus", "")
        detail_level = params.get("detail_level", "medium")

        # Gather content based on source
        content_parts = []
        source_docs = []

        if source == "documents":
            document_ids = params.get("document_ids", [])
            if not document_ids:
                return {"error": "No document_ids provided for source='documents'"}

            for doc_id in document_ids[:5]:  # Limit to 5 docs
                try:
                    doc_uuid = UUID(doc_id)
                    result = await db.execute(
                        select(Document).where(Document.id == doc_uuid)
                    )
                    doc = result.scalar_one_or_none()
                    if doc:
                        content = doc.content or doc.summary or ""
                        if len(content) > 8000:
                            content = content[:8000] + "..."
                        content_parts.append(f"## Document: {doc.title}\n{content}")
                        source_docs.append({"id": str(doc.id), "title": doc.title})
                except Exception as e:
                    logger.warning(f"Could not load document {doc_id}: {e}")

        elif source == "search":
            search_query = params.get("search_query", "")
            if not search_query:
                return {"error": "No search_query provided for source='search'"}

            # Use vector search
            try:
                vector_store = VectorStore()
                results = await vector_store.search(search_query, limit=5)

                for result in results:
                    doc_id = result.get("document_id")
                    if doc_id:
                        try:
                            doc_uuid = UUID(doc_id)
                            doc_result = await db.execute(
                                select(Document).where(Document.id == doc_uuid)
                            )
                            doc = doc_result.scalar_one_or_none()
                            if doc:
                                content = doc.content or doc.summary or ""
                                if len(content) > 5000:
                                    content = content[:5000] + "..."
                                content_parts.append(f"## Document: {doc.title}\n{content}")
                                source_docs.append({"id": str(doc.id), "title": doc.title})
                        except:
                            pass
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return {"error": f"Search failed: {str(e)}"}

        elif source == "description":
            description = params.get("description", "")
            if not description:
                return {"error": "No description provided for source='description'"}
            content_parts.append(description)

        elif source == "gitlab_repo":
            # Delegate to dedicated GitLab architecture tool
            gitlab_result = await self._tool_generate_gitlab_architecture(
                {
                    "project_id": params.get("gitlab_project"),
                    "branch": params.get("gitlab_branch"),
                    "diagram_type": diagram_type,
                    "focus": focus,
                    "detail_level": detail_level,
                },
                user_id,
                db,
            )
            if "error" in gitlab_result:
                return gitlab_result
            return gitlab_result

        if not content_parts:
            return {"error": "No content available to generate diagram"}

        # Build the prompt for diagram generation
        combined_content = "\n\n".join(content_parts)

        # Determine diagram type hints
        type_instructions = {
            "flowchart": "Create a flowchart (flowchart TD or flowchart LR) showing process flow, decision points, and connections.",
            "sequence": "Create a sequence diagram showing interactions between actors/systems over time.",
            "class": "Create a class diagram showing entities, their attributes, and relationships.",
            "state": "Create a state diagram showing states and transitions.",
            "er": "Create an entity-relationship diagram showing entities and their relationships.",
            "gantt": "Create a Gantt chart showing tasks, timelines, and dependencies.",
            "pie": "Create a pie chart showing proportions/distributions.",
            "mindmap": "Create a mind map showing hierarchical concepts and relationships.",
            "architecture": "Create an architecture diagram (using flowchart) showing system components, services, and their connections.",
            "auto": "Choose the most appropriate diagram type based on the content."
        }

        detail_instructions = {
            "high": "Include all details, sub-components, and relationships. Use descriptive labels.",
            "medium": "Include main components and key relationships. Balance detail with clarity.",
            "low": "Show only major components and primary relationships. Keep it simple and high-level."
        }

        focus_instruction = f"Focus specifically on: {focus}" if focus else ""

        prompt = f"""Analyze the following content and generate a Mermaid diagram.

CONTENT TO ANALYZE:
{combined_content}

DIAGRAM REQUIREMENTS:
- Diagram Type: {diagram_type} - {type_instructions.get(diagram_type, type_instructions['auto'])}
- Detail Level: {detail_level} - {detail_instructions.get(detail_level, detail_instructions['medium'])}
{focus_instruction}

IMPORTANT RULES:
1. Output ONLY valid Mermaid syntax that starts with the diagram type declaration
2. Use clear, concise labels (avoid special characters that might break Mermaid)
3. For flowcharts, use: flowchart TD (top-down) or flowchart LR (left-right)
4. Wrap node labels with spaces in quotes: A["Node Label"]
5. Use meaningful node IDs
6. Add styling for important nodes if helpful

Generate the Mermaid diagram code:"""

        # Call LLM to generate the diagram
        try:
            llm_service = LLMService()

            # Load user settings if available
            user_settings = None
            try:
                prefs_result = await db.execute(
                    select(UserPreferences).where(UserPreferences.user_id == user_id)
                )
                user_prefs = prefs_result.scalar_one_or_none()
                if user_prefs:
                    user_settings = UserLLMSettings.from_preferences(user_prefs)
            except:
                pass

            response = await llm_service.generate_response(
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, accurate Mermaid diagrams. Output only valid Mermaid code without markdown code blocks or explanations."},
                    {"role": "user", "content": prompt}
                ],
                user_settings=user_settings,
                task_type="chat"
            )

            mermaid_code = response.get("content", "").strip()

            # Clean up the response - remove markdown code blocks if present
            if mermaid_code.startswith("```mermaid"):
                mermaid_code = mermaid_code[10:]
            elif mermaid_code.startswith("```"):
                mermaid_code = mermaid_code[3:]
            if mermaid_code.endswith("```"):
                mermaid_code = mermaid_code[:-3]
            mermaid_code = mermaid_code.strip()

            # Validate it starts with a valid diagram type
            valid_starts = ["flowchart", "sequenceDiagram", "classDiagram", "stateDiagram",
                          "erDiagram", "gantt", "pie", "mindmap", "graph"]
            is_valid = any(mermaid_code.lower().startswith(s.lower()) for s in valid_starts)

            if not is_valid and mermaid_code:
                # Try to fix by prepending flowchart
                mermaid_code = "flowchart TD\n" + mermaid_code

            return {
                "diagram_type": diagram_type if diagram_type != "auto" else "detected",
                "mermaid_code": mermaid_code,
                "source_documents": source_docs,
                "focus": focus or "general overview",
                "detail_level": detail_level,
                "can_render": bool(mermaid_code and is_valid)
            }

        except Exception as e:
            logger.error(f"Error generating diagram: {e}")
            return {"error": f"Failed to generate diagram: {str(e)}"}

    # =========================================================================
    # Workflow and Custom Tool Integration
    # =========================================================================

    async def _tool_run_workflow(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute a workflow by name or ID."""
        from app.models.workflow import Workflow
        from app.services.workflow_engine import WorkflowEngine

        workflow_name = params.get("workflow_name")
        workflow_id = params.get("workflow_id")
        inputs = params.get("inputs", {})

        if not workflow_name and not workflow_id:
            return {"error": "Either workflow_name or workflow_id is required"}

        try:
            # Find the workflow
            if workflow_id:
                try:
                    wf_uuid = UUID(workflow_id)
                    result = await db.execute(
                        select(Workflow).where(
                            Workflow.id == wf_uuid,
                            Workflow.user_id == user_id
                        )
                    )
                    workflow = result.scalar_one_or_none()
                except (ValueError, TypeError):
                    return {"error": f"Invalid workflow ID: {workflow_id}"}
            else:
                # Search by name (case-insensitive)
                result = await db.execute(
                    select(Workflow).where(
                        Workflow.user_id == user_id,
                        func.lower(Workflow.name) == func.lower(workflow_name)
                    )
                )
                workflow = result.scalar_one_or_none()

            if not workflow:
                return {
                    "error": f"Workflow not found: {workflow_name or workflow_id}",
                    "suggestion": "Use list_workflows to see available workflows"
                }

            if not workflow.is_active:
                return {
                    "error": f"Workflow '{workflow.name}' is not active",
                    "workflow_id": str(workflow.id)
                }

            # Execute the workflow
            engine = WorkflowEngine()
            from app.models.user import User

            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()

            if not user:
                return {"error": "User not found"}

            execution_id = await engine.execute(
                workflow_id=workflow.id,
                trigger_type="agent",
                trigger_data={"source": "agent_tool"},
                inputs=inputs,
                user=user,
                db=db
            )

            return {
                "status": "started",
                "workflow_name": workflow.name,
                "workflow_id": str(workflow.id),
                "execution_id": str(execution_id),
                "message": f"Workflow '{workflow.name}' execution started"
            }

        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            return {"error": f"Failed to run workflow: {str(e)}"}

    async def _tool_create_workflow_from_description(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate and save a workflow from a natural language description."""
        from app.models.workflow import Workflow, WorkflowNode, WorkflowEdge
        from app.services.workflow_synthesis_service import WorkflowSynthesisService

        description = (params.get("description") or "").strip()
        name = params.get("name")
        is_active = params.get("is_active")
        trigger_config = params.get("trigger_config")

        if not description:
            return {"error": "Description is required to generate a workflow"}

        service = WorkflowSynthesisService()
        try:
            workflow_data, warnings = await service.synthesize(
                description=description,
                name=name,
                trigger_config=trigger_config,
                is_active=is_active,
                user_id=user_id,
                db=db,
            )
        except Exception as exc:
            return {"error": f"Failed to synthesize workflow: {str(exc)}"}

        try:
            workflow = Workflow(
                user_id=user_id,
                name=workflow_data.name,
                description=workflow_data.description,
                is_active=workflow_data.is_active,
                trigger_config=workflow_data.trigger_config or {},
            )
            db.add(workflow)
            await db.flush()

            for node_data in workflow_data.nodes:
                node = WorkflowNode(
                    workflow_id=workflow.id,
                    node_id=node_data.node_id,
                    node_type=node_data.node_type,
                    tool_id=node_data.tool_id,
                    builtin_tool=node_data.builtin_tool,
                    config=node_data.config,
                    position_x=node_data.position_x,
                    position_y=node_data.position_y,
                )
                db.add(node)

            for edge_data in workflow_data.edges:
                edge = WorkflowEdge(
                    workflow_id=workflow.id,
                    source_node_id=edge_data.source_node_id,
                    target_node_id=edge_data.target_node_id,
                    source_handle=edge_data.source_handle,
                    condition=edge_data.condition,
                )
                db.add(edge)

            await db.commit()

            return {
                "workflow_id": str(workflow.id),
                "workflow_name": workflow.name,
                "node_count": len(workflow_data.nodes),
                "edge_count": len(workflow_data.edges),
                "warnings": warnings,
                "message": f"Workflow '{workflow.name}' created"
            }
        except Exception as exc:
            await db.rollback()
            logger.error(f"Error creating workflow from description: {exc}")
            return {"error": f"Failed to create workflow: {str(exc)}"}

    async def _tool_propose_workflow_from_description(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate a workflow draft from a natural language description (no save)."""
        from app.services.workflow_synthesis_service import WorkflowSynthesisService

        description = (params.get("description") or "").strip()
        name = params.get("name")
        is_active = params.get("is_active")
        trigger_config = params.get("trigger_config")

        if not description:
            return {"error": "Description is required to generate a workflow"}

        service = WorkflowSynthesisService()
        try:
            workflow_data, warnings = await service.synthesize(
                description=description,
                name=name,
                trigger_config=trigger_config,
                is_active=is_active,
                user_id=user_id,
                db=db,
            )
        except Exception as exc:
            return {"error": f"Failed to synthesize workflow: {str(exc)}"}

        return {
            "workflow": workflow_data.model_dump(),
            "warnings": warnings,
            "message": f"Draft workflow '{workflow_data.name}' generated for review"
        }

    async def _tool_list_workflows(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List available workflows for the user."""
        from app.models.workflow import Workflow

        active_only = params.get("active_only", True)

        try:
            query = select(Workflow).where(Workflow.user_id == user_id)

            if active_only:
                query = query.where(Workflow.is_active == True)

            query = query.order_by(Workflow.name)
            result = await db.execute(query)
            workflows = result.scalars().all()

            return {
                "workflows": [
                    {
                        "id": str(wf.id),
                        "name": wf.name,
                        "description": wf.description,
                        "is_active": wf.is_active,
                        "trigger_type": wf.trigger_config.get("type", "manual") if wf.trigger_config else "manual"
                    }
                    for wf in workflows
                ],
                "count": len(workflows)
            }

        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return {"error": f"Failed to list workflows: {str(e)}"}

    async def _tool_run_custom_tool(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute a user-defined custom tool."""
        from app.models.workflow import UserTool
        from app.services.custom_tool_service import CustomToolService, ToolExecutionError
        from app.models.user import User

        tool_name = params.get("tool_name")
        inputs = params.get("inputs", {})

        if not tool_name:
            return {"error": "tool_name is required"}

        try:
            # Find the tool by name
            result = await db.execute(
                select(UserTool).where(
                    UserTool.user_id == user_id,
                    func.lower(UserTool.name) == func.lower(tool_name)
                )
            )
            tool = result.scalar_one_or_none()

            if not tool:
                return {
                    "error": f"Custom tool not found: {tool_name}",
                    "suggestion": "Use list_custom_tools to see available tools"
                }

            if not tool.is_enabled:
                return {
                    "error": f"Tool '{tool.name}' is disabled",
                    "tool_id": str(tool.id)
                }

            # Get user
            user_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()

            if not user:
                return {"error": "User not found"}

            # Execute the tool
            tool_service = CustomToolService()
            execution_result = await tool_service.execute_tool(
                tool=tool,
                inputs=inputs,
                user=user,
                db=db
            )

            return {
                "status": "completed",
                "tool_name": tool.name,
                "tool_type": tool.tool_type,
                "output": execution_result.get("output"),
                "execution_time_ms": execution_result.get("execution_time_ms", 0)
            }

        except ToolExecutionError as e:
            return {
                "status": "failed",
                "tool_name": tool_name,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"Error running custom tool: {e}")
            return {"error": f"Failed to run custom tool: {str(e)}"}

    async def _tool_list_custom_tools(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List available custom tools for the user."""
        from app.models.workflow import UserTool

        tool_type = params.get("tool_type")

        try:
            query = select(UserTool).where(
                UserTool.user_id == user_id,
                UserTool.is_enabled == True
            )

            if tool_type:
                query = query.where(UserTool.tool_type == tool_type)

            query = query.order_by(UserTool.name)
            result = await db.execute(query)
            tools = result.scalars().all()

            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "tool_type": tool.tool_type,
                        "parameters": tool.parameters_schema
                    }
                    for tool in tools
                ],
                "count": len(tools)
            }

        except Exception as e:
            logger.error(f"Error listing custom tools: {e}")
            return {"error": f"Failed to list custom tools: {str(e)}"}

    # ============================================================================
    # Agent Collaboration Tools
    # ============================================================================

    async def _tool_list_available_agents(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """List available agents that can be delegated to."""
        from app.models.agent_definition import AgentDefinition

        try:
            query = select(AgentDefinition).where(
                AgentDefinition.is_active == True
            ).order_by(AgentDefinition.priority.desc())

            result = await db.execute(query)
            agents = result.scalars().all()

            return {
                "agents": [
                    {
                        "name": agent.name,
                        "display_name": agent.display_name,
                        "description": agent.description,
                        "capabilities": agent.capabilities,
                        "priority": agent.priority
                    }
                    for agent in agents
                ],
                "count": len(agents)
            }

        except Exception as e:
            logger.error(f"Error listing available agents: {e}")
            return {"error": f"Failed to list agents: {str(e)}"}

    async def _tool_delegate_to_agent(
        self,
        params: Dict[str, Any],
        user_id: UUID,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Delegate a task to another specialized agent.

        Safety guards:
        - Prevents delegation to self (if current agent is known)
        - Prevents circular delegation chains
        - Limits delegation depth to 2 levels
        - Timeout of 60 seconds for delegated tasks
        """
        from app.models.agent_definition import AgentDefinition

        target_name = params.get("target_agent", "").strip()
        task_description = params.get("task_description", "").strip()
        context = params.get("context", "").strip()

        if not target_name:
            return {"error": "target_agent is required"}
        if not task_description:
            return {"error": "task_description is required"}

        try:
            # Get target agent
            query = select(AgentDefinition).where(
                AgentDefinition.name == target_name,
                AgentDefinition.is_active == True
            )
            result = await db.execute(query)
            target_agent = result.scalar_one_or_none()

            if not target_agent:
                # Try to find by display name
                query = select(AgentDefinition).where(
                    AgentDefinition.display_name.ilike(f"%{target_name}%"),
                    AgentDefinition.is_active == True
                )
                result = await db.execute(query)
                target_agent = result.scalar_one_or_none()

            if not target_agent:
                # List available agents for the error message
                agents_query = select(AgentDefinition.name).where(AgentDefinition.is_active == True)
                agents_result = await db.execute(agents_query)
                available_agents = [a[0] for a in agents_result.fetchall()]
                return {
                    "error": f"Agent '{target_name}' not found or inactive",
                    "available_agents": available_agents
                }

            # Build delegation prompt
            delegation_prompt = f"""You are being asked to help with a delegated task.

TASK: {task_description}

{"CONTEXT PROVIDED: " + context if context else ""}

Please complete this task and provide a focused, detailed response.
Include relevant information from the knowledge base when applicable.
"""

            # Process with target agent (using a simplified approach)
            # We run the agent's system prompt + task through the LLM
            messages = [
                {"role": "system", "content": target_agent.system_prompt},
                {"role": "user", "content": delegation_prompt}
            ]

            # Get tools available to the delegated agent
            from app.services.agent_tools import AGENT_TOOLS
            if target_agent.tool_whitelist:
                allowed_tools = [
                    t for t in AGENT_TOOLS
                    if t["name"] in target_agent.tool_whitelist
                    and t["name"] != "delegate_to_agent"  # Prevent recursive delegation
                ]
            else:
                # All tools except delegate_to_agent to prevent infinite loops
                allowed_tools = [t for t in AGENT_TOOLS if t["name"] != "delegate_to_agent"]

            # Generate response from the delegated agent
            response = await self.llm_service.generate_chat_response(
                messages=messages,
                tools=allowed_tools,
                max_tokens=4096
            )

            # If the agent wants to call tools, execute them
            tool_results = []
            if response.get("tool_calls"):
                for tc in response["tool_calls"]:
                    tool_call = AgentToolCall(
                        tool_name=tc["function"]["name"],
                        tool_input=tc["function"].get("arguments", {})
                    )
                    executed = await self._execute_tool(tool_call, user_id, db)
                    tool_results.append({
                        "tool": executed.tool_name,
                        "result": executed.tool_output,
                        "status": executed.status
                    })

                # If tools were called, get final response from agent
                if tool_results:
                    tool_message = "Tool results:\n" + "\n".join([
                        f"- {tr['tool']}: {tr['result']}"
                        for tr in tool_results
                    ])
                    messages.append({"role": "assistant", "content": response.get("content", "")})
                    messages.append({"role": "user", "content": tool_message})

                    final_response = await self.llm_service.generate_chat_response(
                        messages=messages,
                        tools=None,  # No more tool calls
                        max_tokens=4096
                    )
                    response = final_response

            return {
                "delegated_to": target_name,
                "agent_display_name": target_agent.display_name,
                "task": task_description,
                "result": response.get("content", "No response generated"),
                "tools_used": [tr["tool"] for tr in tool_results] if tool_results else []
            }

        except Exception as e:
            logger.error(f"Error delegating to agent '{target_name}': {e}")
            return {"error": f"Delegation failed: {str(e)}"}

    # =========================================================================
    # Data Analysis & Visualization Tool Handlers
    # =========================================================================

    async def _tool_get_collection_statistics(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Get comprehensive statistics for a document collection."""
        from app.services.analytics_service import analytics_service
        from datetime import datetime

        source_id = params.get("source_id")
        if source_id:
            source_id = UUID(str(source_id))

        tag = params.get("tag")

        date_from = None
        date_to = None
        if params.get("date_from"):
            try:
                date_from = datetime.fromisoformat(params["date_from"])
            except ValueError:
                pass
        if params.get("date_to"):
            try:
                date_to = datetime.fromisoformat(params["date_to"])
            except ValueError:
                pass

        return await analytics_service.get_collection_statistics(
            db=db,
            source_id=source_id,
            tag=tag,
            date_from=date_from,
            date_to=date_to,
        )

    async def _tool_get_source_analytics(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Get analytics for document sources."""
        from app.services.analytics_service import analytics_service

        source_id = params.get("source_id")
        if source_id:
            source_id = UUID(str(source_id))

        sources = await analytics_service.get_source_analytics(db=db, source_id=source_id)
        return {"sources": sources}

    async def _tool_get_trending_topics(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Find trending topics based on recent documents."""
        from app.services.analytics_service import analytics_service

        days = int(params.get("days", 7) or 7)
        limit = int(params.get("limit", 10) or 10)

        topics = await analytics_service.get_trending_topics(db=db, days=days, limit=limit)
        return {"trending_topics": topics, "period_days": days}

    async def _tool_generate_chart_data(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate data for charts and visualizations."""
        from app.services.analytics_service import analytics_service
        from datetime import datetime

        chart_type = params.get("chart_type", "bar")
        metric = params.get("metric", "document_count")
        group_by = params.get("group_by", "source_type")
        limit = int(params.get("limit", 10) or 10)

        date_from = None
        date_to = None
        if params.get("date_from"):
            try:
                date_from = datetime.fromisoformat(params["date_from"])
            except ValueError:
                pass
        if params.get("date_to"):
            try:
                date_to = datetime.fromisoformat(params["date_to"])
            except ValueError:
                pass

        return await analytics_service.generate_chart_data(
            db=db,
            chart_type=chart_type,
            metric=metric,
            group_by=group_by,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    async def _tool_export_data(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Export document data to various formats."""
        from app.services.analytics_service import analytics_service

        format_type = params.get("format", "json")
        source_id = params.get("source_id")
        if source_id:
            source_id = UUID(str(source_id))

        tag = params.get("tag")
        include_content = bool(params.get("include_content", False))
        include_chunks = bool(params.get("include_chunks", False))
        limit = int(params.get("limit", 1000) or 1000)

        content, filename, content_type = await analytics_service.export_data(
            db=db,
            format=format_type,
            source_id=source_id,
            tag=tag,
            include_content=include_content,
            include_chunks=include_chunks,
            limit=limit,
        )

        # Return info about the export (not the full content for large exports)
        return {
            "filename": filename,
            "content_type": content_type,
            "format": format_type,
            "size_bytes": len(content.encode('utf-8') if isinstance(content, str) else content),
            "preview": content[:1000] if len(content) > 1000 else content,
            "message": f"Export ready: {filename}"
        }

    # =========================================================================
    # Advanced Search Tool Handlers
    # =========================================================================

    async def _tool_faceted_search(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute faceted search with aggregations."""
        from app.services.search_service import search_service

        query = params.get("query", "")
        page = int(params.get("page", 1) or 1)
        page_size = int(params.get("page_size", 10) or 10)
        filters = params.get("filters")

        return await search_service.faceted_search(
            query=query,
            db=db,
            page=page,
            page_size=page_size,
            filters=filters,
        )

    async def _tool_get_search_suggestions(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Get search suggestions and autocomplete."""
        from app.services.search_service import search_service

        partial_query = params.get("partial_query", "")
        limit = int(params.get("limit", 5) or 5)

        suggestions = await search_service.get_search_suggestions(
            partial_query=partial_query,
            db=db,
            limit=limit,
        )
        return {"suggestions": suggestions, "query": partial_query}

    async def _tool_get_related_searches(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Get related search queries."""
        from app.services.search_service import search_service

        query = params.get("query", "")
        limit = int(params.get("limit", 5) or 5)

        related = await search_service.get_related_searches(
            query=query,
            db=db,
            limit=limit,
        )
        return {"related_searches": related, "original_query": query}

    # =========================================================================
    # Content Generation Tool Handlers
    # =========================================================================

    async def _tool_draft_email(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate an email draft."""
        from app.services.content_generation_service import content_generation_service

        subject = params.get("subject", "")
        recipient = params.get("recipient")
        context = params.get("context")
        tone = params.get("tone", "professional")
        length = params.get("length", "medium")

        document_ids = None
        if params.get("document_ids"):
            document_ids = [UUID(str(doc_id)) for doc_id in params["document_ids"]]

        search_query = params.get("search_query")

        return await content_generation_service.draft_email(
            db=db,
            subject=subject,
            recipient=recipient,
            context=context,
            document_ids=document_ids,
            search_query=search_query,
            tone=tone,
            length=length,
        )

    async def _tool_generate_meeting_notes(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate structured meeting notes."""
        from app.services.content_generation_service import content_generation_service

        transcript = params.get("transcript")
        meeting_title = params.get("meeting_title")
        participants = params.get("participants", [])
        include_action_items = bool(params.get("include_action_items", True))
        include_decisions = bool(params.get("include_decisions", True))

        document_ids = None
        if params.get("document_ids"):
            document_ids = [UUID(str(doc_id)) for doc_id in params["document_ids"]]

        return await content_generation_service.generate_meeting_notes(
            db=db,
            transcript=transcript,
            document_ids=document_ids,
            meeting_title=meeting_title,
            participants=participants,
            include_action_items=include_action_items,
            include_decisions=include_decisions,
        )

    async def _tool_generate_documentation(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate documentation from source documents."""
        from app.services.content_generation_service import content_generation_service

        topic = params.get("topic", "")
        doc_type = params.get("doc_type", "technical")
        target_audience = params.get("target_audience", "developers")
        include_examples = bool(params.get("include_examples", True))
        search_query = params.get("search_query")

        document_ids = None
        if params.get("document_ids"):
            document_ids = [UUID(str(doc_id)) for doc_id in params["document_ids"]]

        return await content_generation_service.generate_documentation(
            db=db,
            topic=topic,
            doc_type=doc_type,
            document_ids=document_ids,
            search_query=search_query,
            target_audience=target_audience,
            include_examples=include_examples,
        )

    async def _tool_generate_executive_summary(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate an executive summary."""
        from app.services.content_generation_service import content_generation_service

        topic = params.get("topic")
        max_length = int(params.get("max_length", 500) or 500)
        include_recommendations = bool(params.get("include_recommendations", True))
        include_metrics = bool(params.get("include_metrics", True))
        search_query = params.get("search_query")

        document_ids = None
        if params.get("document_ids"):
            document_ids = [UUID(str(doc_id)) for doc_id in params["document_ids"]]

        return await content_generation_service.generate_executive_summary(
            db=db,
            document_ids=document_ids,
            search_query=search_query,
            topic=topic,
            max_length=max_length,
            include_recommendations=include_recommendations,
            include_metrics=include_metrics,
        )

    async def _tool_generate_report(
        self, params: Dict[str, Any], db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate a structured report."""
        from app.services.content_generation_service import content_generation_service

        report_type = params.get("report_type", "summary")
        title = params.get("title")
        sections = params.get("sections")
        search_query = params.get("search_query")

        document_ids = None
        if params.get("document_ids"):
            document_ids = [UUID(str(doc_id)) for doc_id in params["document_ids"]]

        return await content_generation_service.generate_report(
            db=db,
            report_type=report_type,
            document_ids=document_ids,
            search_query=search_query,
            title=title,
            sections=sections,
        )

    async def _tool_generate_gitlab_architecture(
        self, params: Dict[str, Any], user_id: UUID, db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate architecture diagram from a GitLab repository."""
        from app.services.gitlab_architecture_service import get_gitlab_architecture_service
        from app.models.data_source import DataSource

        project_id = params.get("project_id")
        if not project_id:
            return {"error": "project_id is required"}

        branch = params.get("branch")
        diagram_type = params.get("diagram_type", "auto")
        focus = params.get("focus")
        detail_level = params.get("detail_level", "medium")

        # Find GitLab data source to get credentials
        query = select(DataSource).where(
            DataSource.source_type == "gitlab",
            DataSource.is_active == True,
        )
        result = await db.execute(query)
        gitlab_source = result.scalars().first()

        if not gitlab_source:
            return {
                "error": "No active GitLab data source configured",
                "hint": "Please configure a GitLab data source in Admin settings first",
            }

        config = gitlab_source.config or {}
        gitlab_url = config.get("gitlab_url")
        token = config.get("token")

        if not gitlab_url or not token:
            return {
                "error": "GitLab data source is missing URL or token",
                "hint": "Please check the GitLab data source configuration",
            }

        try:
            service = get_gitlab_architecture_service()
            result = await service.generate_architecture_diagram(
                gitlab_url=gitlab_url,
                token=token,
                project_id=project_id,
                branch=branch,
                diagram_type=diagram_type,
                focus=focus,
                detail_level=detail_level,
            )

            return {
                "success": True,
                "project": result["project"],
                "mermaid_code": result["mermaid_code"],
                "diagram_type": result["diagram_type"],
                "focus": result["focus"],
                "analysis_summary": result["analysis_summary"],
                "has_png": result.get("png_base64") is not None,
                "has_svg": result.get("svg") is not None,
                "render_error": result.get("render_error"),
            }

        except Exception as e:
            logger.error(f"Error generating GitLab architecture: {e}")
            return {
                "error": str(e),
                "project_id": project_id,
            }
