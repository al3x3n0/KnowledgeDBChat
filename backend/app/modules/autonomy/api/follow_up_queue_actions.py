"""HTTP error boundary for follow-up queue action dispatch."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import HTTPException

from app.modules.autonomy.application.follow_up_queue_dispatcher import (
    FollowUpQueueDispatcherDependencies,
    dispatch_follow_up_queue_action,
)
from app.modules.autonomy.application.follow_up_queue_inbox import (
    FollowUpQueueActionError,
)

Dispatcher = Callable[..., Awaitable[Any]]
DependenciesFactory = Callable[[], FollowUpQueueDispatcherDependencies]


@dataclass(frozen=True)
class FollowUpQueueActionApi:
    perform_follow_up_queue_action: Callable[..., Awaitable[Any]]


def build_follow_up_queue_action_api(
    *,
    dependencies_factory: DependenciesFactory,
    dispatcher: Dispatcher = dispatch_follow_up_queue_action,
) -> FollowUpQueueActionApi:
    """Build an adapter that translates application errors into HTTP errors."""

    async def perform_follow_up_queue_action(**kwargs) -> Any:
        try:
            return await dispatcher(
                **kwargs,
                deps=dependencies_factory(),
            )
        except FollowUpQueueActionError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.detail,
            ) from error

    return FollowUpQueueActionApi(
        perform_follow_up_queue_action=perform_follow_up_queue_action,
    )
