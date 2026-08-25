"""Workflows, scheduling, delegation, notifications and agent messaging.

Generated from the literals these tools were declared as; the descriptions
and parameter schemas are the original text.
"""

from __future__ import annotations

from app.agent_core.tool_specs.spec import ToolSpec

SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="run_workflow",
        description="Execute a saved workflow by name or ID. Workflows are user-defined automation sequences that can perform multiple operations.",
        parameters={
            "type": "object",
            "properties": {
                "workflow_name": {
                    "type": "string",
                    "description": "Name of the workflow to execute (case-insensitive search)",
                },
                "workflow_id": {
                    "type": "string",
                    "description": "UUID of the workflow to execute (alternative to name)",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters to pass to the workflow",
                },
            },
            "required": [],
        },
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="propose_workflow_from_description",
        description="Generate a workflow draft from a natural language description WITHOUT saving it. Use this to propose a workflow for the user to review/approve before saving.",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the workflow to generate",
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the workflow",
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the workflow should be active (default: true)",
                    "default": True,
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Optional trigger configuration (manual, schedule, event, webhook)",
                },
                "synthesize_custom_tools": {
                    "type": "boolean",
                    "description": "Allow generating custom tool drafts alongside the workflow (including docker_container tools)",
                    "default": False,
                },
                "preferred_tool_type": {
                    "type": "string",
                    "enum": [
                        "webhook",
                        "transform",
                        "python",
                        "llm_prompt",
                        "docker_container",
                    ],
                    "description": "Bias synthesized custom tools toward this type",
                },
                "expose_workflow_as_tool": {
                    "type": "boolean",
                    "description": "Also generate a workflow_runner tool draft wrapping this workflow",
                    "default": False,
                },
                "workflow_tool_name": {
                    "type": "string",
                    "description": "Optional custom name for the synthesized workflow_runner tool",
                },
            },
            "required": ["description"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="create_workflow_from_description",
        description="Generate and save a workflow from a natural language description. Returns the new workflow ID and summary.",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Natural language description of the workflow to generate",
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the workflow",
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the workflow should be active (default: true)",
                    "default": True,
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Optional trigger configuration (manual, schedule, event, webhook)",
                },
                "synthesize_custom_tools": {
                    "type": "boolean",
                    "description": "Generate and persist custom tools from the description (supports docker_container)",
                    "default": False,
                },
                "preferred_tool_type": {
                    "type": "string",
                    "enum": [
                        "webhook",
                        "transform",
                        "python",
                        "llm_prompt",
                        "docker_container",
                    ],
                    "description": "Bias synthesized custom tools toward this type",
                },
                "expose_workflow_as_tool": {
                    "type": "boolean",
                    "description": "Create a workflow_runner custom tool for the saved workflow",
                    "default": False,
                },
                "workflow_tool_name": {
                    "type": "string",
                    "description": "Optional name for the created workflow_runner tool",
                },
            },
            "required": ["description"],
        },
        effects="write",
        job_types=(),
    ),
    ToolSpec(
        name="list_workflows",
        description="List available workflows that can be executed.",
        parameters={
            "type": "object",
            "properties": {
                "active_only": {
                    "type": "boolean",
                    "description": "Only list active workflows (default: true)",
                    "default": True,
                }
            },
            "required": [],
        },
        job_types=(),
    ),
    ToolSpec(
        name="delegate_to_agent",
        description="Delegate a specific subtask to another specialized agent. Use when the task requires expertise outside your specialty. The other agent will process the request and return results. Available agents: qa_specialist (answering questions), document_expert (document operations), code_expert (code analysis), research_assistant (deep research), data_analyst (insights and visualizations), report_generator (creating reports), workflow_assistant (automation).",
        parameters={
            "type": "object",
            "properties": {
                "target_agent": {
                    "type": "string",
                    "description": "Name of the agent to delegate to (e.g., 'qa_specialist', 'code_expert', 'research_assistant')",
                },
                "task_description": {
                    "type": "string",
                    "description": "Clear description of what you need the other agent to do",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant context from your current analysis to pass to the other agent (optional)",
                },
            },
            "required": ["target_agent", "task_description"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="list_available_agents",
        description="List all available specialized agents that can be delegated to, including their capabilities and descriptions.",
        parameters={"type": "object", "properties": {}, "required": []},
        job_types=(),
    ),
    ToolSpec(
        name="delegate_subtask",
        description="Spawn a child agent job to work on a specific subtask. The child runs asynchronously as a background task.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the subtask job"},
                "goal": {
                    "type": "string",
                    "description": "Goal for the child agent job",
                },
                "job_type": {
                    "type": "string",
                    "enum": ["research", "analysis", "synthesis", "custom"],
                    "description": "Type of job for the child agent",
                },
                "config": {
                    "type": "object",
                    "description": "Job-specific config to pass to child",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max iterations for child (capped at parent's remaining)",
                },
                "share_findings": {
                    "type": "boolean",
                    "description": "Share parent findings with child (default: true)",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, poll for completion (blocks up to 60s)",
                },
            },
            "required": ["name", "goal"],
        },
        effects="write",
        cost_tier="medium",
    ),
    ToolSpec(
        name="wait_for_subtask",
        description="Check status or wait for a delegated subtask to complete. Returns current status and results if available.",
        parameters={
            "type": "object",
            "properties": {
                "subtask_job_id": {
                    "type": "string",
                    "description": "Job ID of the delegated subtask",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "How long to poll (max 120, default 30)",
                },
            },
            "required": ["subtask_job_id"],
        },
    ),
    ToolSpec(
        name="share_findings",
        description="Push findings to sibling agent jobs (those sharing the same parent). Used for coordination in multi-agent workflows.",
        parameters={
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                        },
                    },
                    "description": "Findings to share with siblings",
                },
                "target_job_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific sibling job IDs (optional, shares with all siblings if empty)",
                },
            },
            "required": ["findings"],
        },
        effects="write",
    ),
    ToolSpec(
        name="request_review",
        description="Ask another agent job or a human operator to review the current work. Creates a review checkpoint.",
        parameters={
            "type": "object",
            "properties": {
                "review_type": {
                    "type": "string",
                    "enum": ["peer_agent", "human"],
                    "description": "Whether to request review from a peer agent or human",
                },
                "content_to_review": {
                    "type": "string",
                    "description": "The content or summary to be reviewed",
                },
                "review_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Criteria for the reviewer to evaluate",
                },
                "reviewer_job_id": {
                    "type": "string",
                    "description": "Specific sibling job to request review from (for peer_agent type)",
                },
            },
            "required": ["content_to_review"],
        },
        effects="write",
        cost_tier="medium",
    ),
    ToolSpec(
        name="list_available_workflows",
        description="List DAG workflows available to the current user. Returns workflow IDs, names, and descriptions.",
        parameters={
            "type": "object",
            "properties": {
                "is_active": {
                    "type": "boolean",
                    "description": "Filter by active status (default true)",
                },
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="execute_workflow",
        description="Launch a DAG workflow by ID. The workflow executes its node graph and returns an execution ID for status tracking.",
        parameters={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "UUID of the workflow to execute",
                },
                "trigger_data": {
                    "type": "object",
                    "description": "Data to pass as trigger context",
                },
                "inputs": {
                    "type": "object",
                    "description": "Initial context variables for the workflow",
                },
            },
            "required": ["workflow_id"],
        },
        effects="write",
        cost_tier="medium",
    ),
    ToolSpec(
        name="get_workflow_status",
        description="Check the status of a workflow execution by its execution ID.",
        parameters={
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "UUID of the workflow execution to check",
                },
            },
            "required": ["execution_id"],
        },
    ),
    ToolSpec(
        name="enqueue_external_agent_call",
        description="Durably enqueue a capability-scoped call to a configured external "
            "agent through the transactional outbox. Delivery occurs only after "
            "the current agent checkpoint commits.",
        parameters={
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "string",
                    "description": "UUID of an enabled external-agent connection",
                },
                "capability": {
                    "type": "string",
                    "description": "Capability declared by the connection manifest",
                },
                "payload": {
                    "type": "object",
                    "description": "Bounded JSON request payload",
                },
                "idempotency_key": {
                    "type": "string",
                    "description": (
                        "Optional stable key; the current journal key is used by "
                        "default"
                    ),
                },
                "max_attempts": {
                    "type": "integer",
                    "description": "Delivery attempts before dead-lettering (1-8)",
                },
            },
            "required": ["tool_id", "capability", "payload"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="get_external_call_status",
        description="Read delivery, retry, dead-letter, or response state for an "
            "external-agent outbox request created by this job.",
        parameters={
            "type": "object",
            "properties": {
                "outbox_id": {
                    "type": "string",
                    "description": "UUID returned by enqueue_external_agent_call",
                },
            },
            "required": ["outbox_id"],
        },
        job_types=(),
    ),
    ToolSpec(
        name="send_message_to_agent",
        description="Send a message to another agent job. The target agent can read it via read_agent_messages. Works across any jobs owned by the same user.",
        parameters={
            "type": "object",
            "properties": {
                "target_job_id": {
                    "type": "string",
                    "description": "UUID of the target agent job",
                },
                "message": {"type": "string", "description": "Message content to send"},
                "category": {
                    "type": "string",
                    "description": "Optional category tag (e.g. 'question', 'finding', 'request')",
                },
            },
            "required": ["target_job_id", "message"],
        },
        effects="write",
    ),
    ToolSpec(
        name="read_agent_messages",
        description="Read messages sent to this agent by other agent jobs. Returns messages from the specified index onward.",
        parameters={
            "type": "object",
            "properties": {
                "since_index": {
                    "type": "integer",
                    "description": "Start reading from this message index (default 0)",
                },
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="send_notification",
        description="Send an in-app notification to the job owner. Delivered via WebSocket push and visible in the notification bell.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title"},
                "message": {
                    "type": "string",
                    "description": "Notification message body",
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level: low, normal, high, urgent (default normal)",
                },
                "action_url": {
                    "type": "string",
                    "description": "Optional URL for click-through action",
                },
            },
            "required": ["title", "message"],
        },
        effects="write",
    ),
    ToolSpec(
        name="send_email_alert",
        description="Send an email alert to the job owner. Falls back to in-app notification if SMTP is not configured.",
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
                "priority": {
                    "type": "string",
                    "description": "Priority level: low, normal, high, urgent (default normal)",
                },
            },
            "required": ["subject", "body"],
        },
        effects="write",
    ),
    ToolSpec(
        name="schedule_job",
        description="Schedule a new agent job for future execution. Supports one-time runs at a specific datetime or recurring runs with a cron expression. The scheduled job will be picked up automatically by the scheduler.",
        parameters={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal/task description for the scheduled job",
                },
                "job_type": {
                    "type": "string",
                    "description": "Job type: research, monitor, analysis, synthesis, coding (default research)",
                },
                "schedule_type": {
                    "type": "string",
                    "description": "Schedule type: once (run at specific time) or recurring (cron-based)",
                },
                "run_at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (required if schedule_type=once)",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression for recurring execution (required if schedule_type=recurring), e.g. '0 9 * * 1' for every Monday at 9am",
                },
                "config": {
                    "type": "object",
                    "description": "Optional job configuration (max_iterations, tool overrides, etc.)",
                },
            },
            "required": ["goal", "schedule_type"],
        },
        effects="write",
    ),
    ToolSpec(
        name="cancel_scheduled_job",
        description="Cancel a scheduled or recurring agent job. Prevents future executions and marks the job as cancelled.",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "UUID of the scheduled job to cancel",
                },
            },
            "required": ["job_id"],
        },
        effects="write",
    ),
    ToolSpec(
        name="create_handoff",
        description="Create a structured handoff to spawn a child agent job with a typed contract specifying what the child should produce. The child will see the contract in its system prompt. Use instead of delegate_subtask when you need structured output expectations.",
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The child agent's goal"},
                "job_type": {
                    "type": "string",
                    "enum": ["research", "analysis", "synthesis", "custom"],
                    "description": "Job type for the child (default research)",
                },
                "context": {
                    "type": "string",
                    "description": "Situation briefing — what the child needs to know about the current state",
                },
                "expected_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What the child must produce (e.g., summary, key_findings, recommendations)",
                },
                "share_findings": {
                    "type": "boolean",
                    "description": "Share current findings with the child (default true)",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum iterations for the child job (default 10, max 20)",
                },
            },
            "required": ["goal", "expected_outputs"],
        },
        effects="write",
        cost_tier="medium",
    ),
    ToolSpec(
        name="get_sibling_status",
        description="Check status, progress, and optionally findings of sibling agent jobs (jobs with the same parent). Use to coordinate with peer agents running in parallel.",
        parameters={
            "type": "object",
            "properties": {
                "include_findings": {
                    "type": "boolean",
                    "description": "Also return finding titles from siblings (default false)",
                },
            },
            "required": [],
        },
    ),
    ToolSpec(
        name="broadcast_to_siblings",
        description="Send a message to all sibling agent jobs at once. Use for coordination announcements, status updates, or sharing discoveries with all peer agents.",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to broadcast",
                },
                "category": {
                    "type": "string",
                    "description": "Message category (default broadcast)",
                },
            },
            "required": ["message"],
        },
        effects="write",
    ),
)
