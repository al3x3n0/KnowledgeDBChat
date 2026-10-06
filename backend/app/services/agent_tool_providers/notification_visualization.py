"""Autonomous-job tools: the ``notification_visualization`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)

MAX_NOTIFICATIONS_PER_RUN = 20


def build_autonomous_notification_visualization_provider(
    executor: Any,
) -> FunctionToolProvider:
    """Notification and standalone visualization tools for AutonomousAgentExecutor."""

    async def _send_notification(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.notification_service import NotificationService

        job = ctx.job
        notif_title = str(params.get("title", "")).strip()
        notif_message = str(params.get("message", "")).strip()
        if not notif_title:
            return {"error": "title is required"}
        if not notif_message:
            return {"error": "message is required"}
        try:
            from sqlalchemy import func

            from app.models.notification import Notification

            sent = (
                await ctx.db.execute(
                    select(func.count(Notification.id)).where(
                        Notification.related_entity_id == job.id,
                        Notification.notification_type == "agent_job_alert",
                    )
                )
            ).scalar() or 0
            if sent >= MAX_NOTIFICATIONS_PER_RUN:
                return {
                    "error": f"This run has already sent {sent} notifications "
                    f"(limit {MAX_NOTIFICATIONS_PER_RUN})."
                }
            ns = NotificationService()
            priority = str(params.get("priority", "normal")).strip().lower()
            if priority not in {"low", "normal", "high", "urgent"}:
                priority = "normal"
            notification = await ns.create_notification(
                db=ctx.db,
                user_id=job.user_id,
                notification_type="agent_job_alert",
                title=notif_title[:200],
                message=notif_message[:2000],
                priority=priority,
                related_entity_type="agent_job",
                related_entity_id=job.id,
                data={"source_job_id": str(job.id), "source_job_name": job.name or ""},
                action_url=str(params.get("action_url", "")).strip()[:500] or None,
                commit=False,
            )
            if notification is None:
                # The service swallows its own failures and returns None;
                # reporting success here told the run a person had been told.
                return {"error": "The notification could not be stored"}
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "notification_id": str(notification.id) if notification else None,
                    "delivered": notification is not None,
                    "priority": priority,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to send notification: {exc}"}

    async def _send_email_alert(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from loguru import logger

        from app.services.notification_service import NotificationService

        job = ctx.job
        subject = str(params.get("subject", "")).strip()
        body = str(params.get("body", "")).strip()
        if not subject:
            return {"error": "subject is required"}
        if not body:
            return {"error": "body is required"}
        try:
            logger.info(
                f"Email alert requested by job {job.id} (no SMTP configured), falling back to notification"
            )
            ns = NotificationService()
            priority = str(params.get("priority", "normal")).strip().lower()
            if priority not in {"low", "normal", "high", "urgent"}:
                priority = "normal"
            notification = await ns.create_notification(
                db=ctx.db,
                user_id=job.user_id,
                notification_type="agent_job_alert",
                title=f"[Email] {subject[:180]}",
                message=body[:2000],
                priority=priority,
                related_entity_type="agent_job",
                related_entity_id=job.id,
                data={"intended_delivery": "email", "source_job_id": str(job.id)},
                commit=False,
            )
            if notification is None:
                # The service swallows its own failures and returns None;
                # reporting success here told the run a person had been told.
                return {"error": "The notification could not be stored"}
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "notification_id": str(notification.id) if notification else None,
                    "delivered": notification is not None,
                    "delivery_method": "in_app_notification",
                    "note": "SMTP not configured; delivered as in-app notification",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to send email alert: {exc}"}

    async def _create_chart(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import base64 as b64
        from uuid import uuid4 as _uuid4

        from loguru import logger

        from app.services.storage_service import storage_service
        from app.services.visualization_service import VisualizationService

        job = ctx.job
        chart_type = str(params.get("chart_type", "")).strip().lower()
        data = params.get("data")
        if not chart_type:
            return {"error": "chart_type is required"}
        if not data or not isinstance(data, dict):
            return {"error": "data is required and must be an object"}
        if chart_type not in {
            "bar",
            "line",
            "pie",
            "scatter",
            "histogram",
            "heatmap",
            "box",
            "area",
        }:
            return {
                "error": f"Invalid chart_type: {chart_type}. Must be bar, line, pie, scatter, histogram, heatmap, box, or area"
            }
        try:
            vs = VisualizationService()
            fmt = str(params.get("format", "png")).strip().lower()
            if fmt not in {"png", "svg"}:
                fmt = "png"
            config = {"format": fmt}
            for key in ("title", "x_label", "y_label"):
                val = str(params.get(key, "")).strip()
                if val:
                    config[key] = val
            chart_result = vs.create_chart(
                chart_type=chart_type, data=data, config=config
            )
            image_bytes = b64.b64decode(chart_result["image_base64"])
            object_path = f"agent_artifacts/{job.id}/charts/{_uuid4()}.{fmt}"
            await storage_service.initialize()
            # The service builds its type as "image/<format>", which for svg
            # is not a registered type and browsers will not render it.
            await storage_service.upload_to_path(
                object_path,
                image_bytes,
                "image/svg+xml" if fmt == "svg" else "image/png",
            )
            url = await storage_service.get_presigned_download_url(object_path)
            return {
                "success": True,
                "data": {
                    "chart_type": chart_type,
                    "url": url,
                    "format": fmt,
                    "size_bytes": len(image_bytes),
                },
            }
        except Exception as exc:
            logger.error(f"create_chart failed: {exc}")
            return {"error": f"Failed to create chart: {exc}"}

    async def _render_diagram(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import base64 as b64
        from uuid import uuid4 as _uuid4

        from loguru import logger

        from app.services.storage_service import storage_service

        job = ctx.job
        diagram_code = str(params.get("diagram_code", "")).strip()
        if not diagram_code:
            return {"error": "diagram_code is required"}
        try:
            diagram_type = str(params.get("diagram_type") or "mermaid").strip().lower()
            if diagram_type not in {"mermaid", "graphviz"}:
                # Anything else used to be rendered as Mermaid and reported
                # back under the name the caller had asked for.
                return {
                    "error": f"Invalid diagram_type: {diagram_type}. "
                    "Must be mermaid or graphviz"
                }
            fmt = str(params.get("format", "png")).strip().lower()
            if fmt not in {"png", "svg"}:
                fmt = "png"
            mime = f"image/{fmt}" if fmt == "png" else "image/svg+xml"
            if diagram_type == "graphviz":
                from app.services.diagram_service import DiagramService

                ds = DiagramService()
                image_bytes = b64.b64decode(
                    ds._render_graphviz(diagram_code, {"output_format": fmt})
                )
            else:
                from app.services.mermaid_renderer import MermaidRenderer

                renderer = MermaidRenderer()
                if fmt == "svg":
                    svg_str = await renderer.render_to_svg(diagram_code)
                    image_bytes = (
                        svg_str.encode("utf-8") if isinstance(svg_str, str) else svg_str
                    )
                else:
                    image_bytes = await renderer.render_to_png(diagram_code)
            object_path = f"agent_artifacts/{job.id}/diagrams/{_uuid4()}.{fmt}"
            await storage_service.initialize()
            await storage_service.upload_to_path(object_path, image_bytes, mime)
            url = await storage_service.get_presigned_download_url(object_path)
            return {
                "success": True,
                "data": {
                    "url": url,
                    "diagram_type": diagram_type,
                    "format": fmt,
                    "size_bytes": len(image_bytes),
                },
            }
        except Exception as exc:
            logger.error(f"render_diagram failed: {exc}")
            return {"error": f"Failed to render diagram: {exc}"}

    return FunctionToolProvider(
        name="autonomous_notification_visualization_tools",
        modes={"autonomous"},
        handlers={
            "send_notification": _send_notification,
            "send_email_alert": _send_email_alert,
            "create_chart": _create_chart,
            "render_diagram": _render_diagram,
        },
    )
