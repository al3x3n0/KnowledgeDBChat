interface SchedulerState {
  queue_reason?: string | null;
  last_run_status?: string | null;
  failure_streak?: number | null;
  last_scheduled_at?: string | null;
  last_dispatched_at?: string | null;
  current_run_started_at?: string | null;
  last_completed_run_at?: string | null;
  backoff_until?: string | null;
  backoff_seconds?: number | null;
}

interface RecoveryAuditPanelProps {
  latestAction?: string;
  latestOutcome?: string;
  latestOutcomeReason?: string;
  recoveryReason?: string;
  nextStep?: string;
  schedulerState?: SchedulerState | null;
  className?: string;
  textClassName?: string;
}

function cleanValue(value: unknown): string {
  return String(value ?? '').trim();
}

export default function RecoveryAuditPanel({
  latestAction,
  latestOutcome,
  latestOutcomeReason,
  recoveryReason,
  nextStep,
  schedulerState,
  className = '',
  textClassName = 'text-xs',
}: RecoveryAuditPanelProps) {
  const hasSchedulerState = Boolean(
    schedulerState
    && typeof schedulerState === 'object'
    && (
      cleanValue(schedulerState.queue_reason)
      || cleanValue(schedulerState.last_run_status)
      || typeof schedulerState.failure_streak === 'number'
      || cleanValue(schedulerState.last_scheduled_at)
      || cleanValue(schedulerState.last_dispatched_at)
      || cleanValue(schedulerState.current_run_started_at)
      || cleanValue(schedulerState.last_completed_run_at)
      || cleanValue(schedulerState.backoff_until)
      || typeof schedulerState.backoff_seconds === 'number'
    ),
  );

  if (!(latestAction || latestOutcome || latestOutcomeReason || recoveryReason || nextStep || hasSchedulerState)) {
    return null;
  }

  const schedulerRows: Array<[string, string]> = [];
  if (hasSchedulerState && schedulerState) {
    const queueReason = cleanValue(schedulerState.queue_reason);
    const lastRunStatus = cleanValue(schedulerState.last_run_status);
    const failureStreak = typeof schedulerState.failure_streak === 'number' ? String(schedulerState.failure_streak) : '';
    const lastScheduledAt = cleanValue(schedulerState.last_scheduled_at);
    const lastDispatchedAt = cleanValue(schedulerState.last_dispatched_at);
    const currentRunStartedAt = cleanValue(schedulerState.current_run_started_at);
    const lastCompletedRunAt = cleanValue(schedulerState.last_completed_run_at);
    const backoffUntil = cleanValue(schedulerState.backoff_until);
    const backoffSeconds = typeof schedulerState.backoff_seconds === 'number' ? String(schedulerState.backoff_seconds) : '';

    if (queueReason) schedulerRows.push(['Queue reason', queueReason.replace(/_/g, ' ')]);
    if (lastRunStatus) schedulerRows.push(['Last run', lastRunStatus.replace(/_/g, ' ')]);
    if (failureStreak) schedulerRows.push(['Failure streak', failureStreak]);
    if (lastScheduledAt) schedulerRows.push(['Last scheduled', lastScheduledAt]);
    if (lastDispatchedAt) schedulerRows.push(['Last dispatched', lastDispatchedAt]);
    if (currentRunStartedAt) schedulerRows.push(['Current run', currentRunStartedAt]);
    if (lastCompletedRunAt) schedulerRows.push(['Last completed', lastCompletedRunAt]);
    if (backoffUntil) schedulerRows.push(['Backoff until', backoffUntil]);
    if (backoffSeconds) schedulerRows.push(['Backoff seconds', backoffSeconds]);
  }

  return (
    <div className={`rounded-lg border border-amber-200 bg-white p-3 ${className}`.trim()}>
      <div className="text-xs font-medium text-amber-900 mb-2">Recovery Audit</div>
      <div className={`space-y-1 ${textClassName}`.trim()}>
        {latestAction ? (
          <div className="text-amber-900">
            <span className="font-medium">Latest action:</span> {latestAction}
          </div>
        ) : null}
        {latestOutcome ? (
          <div className="text-orange-700">
            <span className="font-medium">Outcome:</span> {latestOutcome}
          </div>
        ) : null}
        {latestOutcomeReason ? (
          <div className="text-orange-700">
            <span className="font-medium">Outcome reason:</span> {latestOutcomeReason}
          </div>
        ) : null}
        {recoveryReason ? (
          <div className="text-rose-700">
            <span className="font-medium">Recovery reason:</span> {recoveryReason}
          </div>
        ) : null}
        {nextStep ? (
          <div className="text-blue-700">
            <span className="font-medium">Next step:</span> {nextStep}
          </div>
        ) : null}
        {schedulerRows.length > 0 ? (
          <div className="rounded-md border border-amber-100 bg-amber-50/50 p-2 text-amber-900">
            <div className="mb-1 font-medium">Scheduler state</div>
            <div className="space-y-0.5">
              {schedulerRows.map(([label, value]) => (
                <div key={`${label}:${value}`}>
                  <span className="font-medium">{label}:</span> {value}
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
