"""
Celery tasks for workflow execution.

Handles:
- Asynchronous workflow execution
- Scheduled workflow triggers
- Event-based workflow triggers
"""

import asyncio
from uuid import UUID
from datetime import datetime
from celery import shared_task
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.workflow import Workflow, WorkflowExecution
from app.models.user import User


def run_async(coro):
    """Run an async coroutine in a sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="app.tasks.workflow.execute_workflow")
def execute_workflow_task(self, execution_id: str):
    """
    Execute a workflow asynchronously via Celery.

    Args:
        execution_id: The WorkflowExecution ID to run
    """
    logger.info(f"Starting workflow execution task: {execution_id}")

    try:
        run_async(_execute_workflow_async(execution_id))
        logger.info(f"Workflow execution completed: {execution_id}")
    except Exception as e:
        logger.error(f"Workflow execution failed: {execution_id} - {e}")
        # Update execution status on failure
        run_async(_mark_execution_failed(execution_id, str(e)))
        raise


async def _execute_workflow_async(execution_id: str):
    """Async implementation of workflow execution."""
    from app.services.workflow_engine import WorkflowEngine

    session_factory = create_celery_session()
    async with session_factory() as db:
        # Load the execution with workflow
        result = await db.execute(
            select(WorkflowExecution)
            .options(selectinload(WorkflowExecution.workflow))
            .where(WorkflowExecution.id == UUID(execution_id))
        )
        execution = result.scalar_one_or_none()

        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        if execution.status != "pending":
            logger.warning(f"Execution {execution_id} is not pending (status: {execution.status})")
            return

        # Load user
        user_result = await db.execute(
            select(User).where(User.id == execution.user_id)
        )
        user = user_result.scalar_one_or_none()

        if not user:
            raise ValueError(f"User {execution.user_id} not found")

        # Execute the workflow
        engine = WorkflowEngine(db, user)

        try:
            await engine.execute_existing_execution(execution)

        except Exception as e:
            # The engine should have already updated the execution status
            # but we re-raise for the Celery task to handle
            raise


async def _mark_execution_failed(execution_id: str, error: str):
    """Mark an execution as failed."""
    session_factory = create_celery_session()
    async with session_factory() as db:
        result = await db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == UUID(execution_id))
        )
        execution = result.scalar_one_or_none()

        if execution and execution.status not in ["completed", "failed", "cancelled"]:
            execution.status = "failed"
            execution.error = error
            execution.completed_at = datetime.utcnow()
            await db.commit()


@celery_app.task(name="app.tasks.workflow.trigger_scheduled_workflows")
def trigger_scheduled_workflows():
    """
    Celery Beat task to check and trigger scheduled workflows.

    Should be run every minute via Celery Beat.
    """
    logger.debug("Checking for scheduled workflows to trigger")

    try:
        run_async(_trigger_scheduled_workflows_async())
    except Exception as e:
        logger.error(f"Error triggering scheduled workflows: {e}")


async def _trigger_scheduled_workflows_async():
    """Check and trigger scheduled workflows."""
    from croniter import croniter

    session_factory = create_celery_session()
    async with session_factory() as db:
        # Find active workflows with schedule triggers
        result = await db.execute(
            select(Workflow)
            .where(
                Workflow.is_active == True,
                Workflow.trigger_config["type"].astext == "schedule"
            )
        )
        workflows = result.scalars().all()

        now = datetime.utcnow()

        for workflow in workflows:
            try:
                schedule = workflow.trigger_config.get("schedule")
                if not schedule:
                    continue

                # Parse cron expression
                cron = croniter(schedule, now)
                prev_run = cron.get_prev(datetime)
                next_run = cron.get_next(datetime)

                # Check if we should run (within last minute)
                time_since_prev = (now - prev_run).total_seconds()

                if time_since_prev < 60:  # Within last minute
                    # Check if we already ran recently
                    recent_result = await db.execute(
                        select(WorkflowExecution)
                        .where(
                            WorkflowExecution.workflow_id == workflow.id,
                            WorkflowExecution.trigger_type == "schedule",
                            WorkflowExecution.created_at > prev_run
                        )
                    )

                    if recent_result.scalar_one_or_none():
                        continue  # Already triggered

                    # Create execution
                    execution = WorkflowExecution(
                        workflow_id=workflow.id,
                        user_id=workflow.user_id,
                        trigger_type="schedule",
                        trigger_data={"schedule": schedule, "scheduled_time": prev_run.isoformat()},
                        status="pending",
                        progress=0,
                        context={}
                    )
                    db.add(execution)
                    await db.commit()
                    await db.refresh(execution)

                    # Queue the execution
                    execute_workflow_task.delay(str(execution.id))
                    logger.info(f"Triggered scheduled workflow: {workflow.name}")

            except Exception as e:
                logger.error(f"Error checking schedule for workflow {workflow.id}: {e}")


@celery_app.task(name="app.tasks.workflow.trigger_event_workflow")
def trigger_event_workflow(event_name: str, event_data: dict, user_id: str):
    """
    Trigger workflows based on an event.

    Args:
        event_name: The event type (e.g., "document.uploaded")
        event_data: Data associated with the event
        user_id: The user who triggered the event
    """
    logger.info(f"Processing event trigger: {event_name} for user {user_id}")

    try:
        run_async(_trigger_event_workflow_async(event_name, event_data, user_id))
    except Exception as e:
        logger.error(f"Error processing event trigger: {e}")


async def _trigger_event_workflow_async(event_name: str, event_data: dict, user_id: str):
    """Find and trigger workflows matching the event."""
    session_factory = create_celery_session()
    async with session_factory() as db:
        # Find active workflows with matching event triggers
        result = await db.execute(
            select(Workflow)
            .where(
                Workflow.is_active == True,
                Workflow.user_id == UUID(user_id),
                Workflow.trigger_config["type"].astext == "event",
                Workflow.trigger_config["event"].astext == event_name
            )
        )
        workflows = result.scalars().all()

        for workflow in workflows:
            try:
                # Create execution
                execution = WorkflowExecution(
                    workflow_id=workflow.id,
                    user_id=workflow.user_id,
                    trigger_type="event",
                    trigger_data={"event": event_name, "event_data": event_data},
                    status="pending",
                    progress=0,
                    context={"event": event_data}
                )
                db.add(execution)
                await db.commit()
                await db.refresh(execution)

                # Queue the execution
                execute_workflow_task.delay(str(execution.id))
                logger.info(f"Triggered event workflow: {workflow.name} for event {event_name}")

            except Exception as e:
                logger.error(f"Error triggering workflow {workflow.id} for event {event_name}: {e}")


# =============================================================================
# Event Publisher Helper
# =============================================================================

def publish_workflow_event(event_name: str, event_data: dict, user_id: str):
    """
    Publish an event that may trigger workflows.

    Call this from other parts of the application when events occur.

    Example:
        publish_workflow_event(
            "document.uploaded",
            {"document_id": str(doc.id), "title": doc.title},
            str(user.id)
        )
    """
    trigger_event_workflow.delay(event_name, event_data, user_id)
