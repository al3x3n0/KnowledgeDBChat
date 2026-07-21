import type { AgentJob, AgentJobProgressUpdate, AgentJobStatus } from '../types';

export const TERMINAL_JOB_STATUSES = new Set(['completed', 'failed', 'cancelled']);

export function mergeProgressUpdateIntoJob(job: AgentJob, update: AgentJobProgressUpdate): AgentJob {
  const results = (job.results && typeof job.results === 'object') ? { ...(job.results as any) } : {};
  const executionStrategy =
    results.execution_strategy && typeof results.execution_strategy === 'object'
      ? { ...results.execution_strategy }
      : {};

  if (update.execution_graph_runtime && typeof update.execution_graph_runtime === 'object') {
    const prevGraph =
      executionStrategy.execution_graph && typeof executionStrategy.execution_graph === 'object'
        ? { ...executionStrategy.execution_graph }
        : {};
    executionStrategy.execution_graph = {
      ...prevGraph,
      ...update.execution_graph_runtime,
    };
  }

  if (update.scope_observability_runtime && typeof update.scope_observability_runtime === 'object') {
    const prevScope =
      executionStrategy.scope_observability && typeof executionStrategy.scope_observability === 'object'
        ? { ...executionStrategy.scope_observability }
        : {};
    executionStrategy.scope_observability = {
      ...prevScope,
      ...update.scope_observability_runtime,
    };
  }

  if (Object.keys(executionStrategy).length > 0) {
    results.execution_strategy = executionStrategy;
  }

  return {
    ...job,
    progress: Number(update.progress ?? job.progress ?? 0),
    status: (update.status as AgentJobStatus) || job.status,
    iteration: Number(update.iteration ?? job.iteration ?? 0),
    current_phase: String(update.phase || job.current_phase || ''),
    phase_details: typeof update.phase_details === 'string' ? update.phase_details : job.phase_details,
    error: typeof update.error === 'string' ? update.error : job.error,
    results,
  };
}
