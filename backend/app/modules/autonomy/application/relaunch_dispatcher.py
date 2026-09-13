"""Dispatch persisted quick-start jobs to their matching relaunch workflow."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

RequestBuilder = Callable[..., Any]
RequestLauncher = Callable[..., Awaitable[Any]]
RecoveryExtractor = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class RelaunchRoute:
    """Builder and launcher pair for one persisted launch mode."""

    builder: RequestBuilder
    launcher: RequestLauncher
    builder_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelaunchOutcome:
    """Result of dispatching a quick-start relaunch."""

    job: Any
    launch_mode: str
    recovery_strategy: Optional[str] = None
    recovery: Optional[dict[str, Any]] = None


class QuickStartRelaunchDispatcher:
    """Select and invoke quick-start relaunch builders without HTTP coupling."""

    def __init__(
        self,
        *,
        routes: dict[str, RelaunchRoute],
        refined_repo_route: RelaunchRoute,
        recovery_extractor: RecoveryExtractor,
    ) -> None:
        self._routes = dict(routes)
        self._refined_repo_route = refined_repo_route
        self._recovery_extractor = recovery_extractor

    @staticmethod
    def launch_mode(job: Any) -> str:
        config = getattr(job, "config", None)
        config = config if isinstance(config, dict) else {}
        return str(config.get("launch_mode") or "").strip().lower()

    async def relaunch(
        self,
        job: Any,
        *,
        db: Any,
        current_user: Any,
    ) -> Optional[RelaunchOutcome]:
        """Build and launch the matching clean quick-start relaunch."""
        launch_mode = self.launch_mode(job)
        route = self._routes.get(launch_mode)
        if route is None:
            return None
        request = route.builder(job, **route.builder_kwargs)
        if request is None:
            return None
        new_job = await route.launcher(request, db, current_user)
        return RelaunchOutcome(
            job=new_job,
            launch_mode=launch_mode,
            recovery_strategy=(
                "clean_relaunch"
                if launch_mode == "quick_start_repo_bug_triage"
                else None
            ),
        )

    async def refined_repo_retry(
        self,
        job: Any,
        *,
        db: Any,
        current_user: Any,
    ) -> Optional[RelaunchOutcome]:
        """Launch a refined retry for a persisted repository-triage job."""
        launch_mode = self.launch_mode(job)
        if launch_mode != "quick_start_repo_bug_triage":
            return None
        route = self._refined_repo_route
        request = route.builder(job, **route.builder_kwargs)
        if request is None:
            return None
        new_job = await route.launcher(request, db, current_user)
        return RelaunchOutcome(
            job=new_job,
            launch_mode=launch_mode,
            recovery_strategy="refined_retry",
            recovery=self._recovery_extractor(job),
        )
