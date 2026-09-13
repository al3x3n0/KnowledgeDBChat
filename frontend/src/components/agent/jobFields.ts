/**
 * Reading a field off an agent job, where the shape varies by where it came
 * from.
 *
 * A job's experiment runs arrive either as a typed column or inside the
 * untyped `results` blob, depending on which path wrote them. The reading was
 * a closure inside AutonomousAgentsPage; it is a pure function of the job, so
 * it is out here where components lifted out of that page can use it too.
 */

import type { AgentJob, AgentJobExperimentRun } from '../../types';

/** The most recent experiment run attached to a job, wherever it was stored. */
export const getLatestExperimentRun = (job: AgentJob): AgentJobExperimentRun | null => {
  const candidates: AgentJobExperimentRun[] = [];
  if (Array.isArray(job.experiment_runs)) {
    candidates.push(
      ...job.experiment_runs.filter((row): row is AgentJobExperimentRun =>
        Boolean(row && typeof row === 'object')
      )
    );
  } else {
    const hist = (job.results as any)?.experiment_runs;
    if (Array.isArray(hist)) {
      candidates.push(
        ...hist.filter((row: unknown): row is AgentJobExperimentRun =>
          Boolean(row && typeof row === 'object')
        )
      );
    }
  }
  if (job.experiment_run && typeof job.experiment_run === 'object') {
    candidates.push(job.experiment_run);
  } else {
    const cur = (job.results as any)?.experiment_run;
    if (cur && typeof cur === 'object') candidates.push(cur as AgentJobExperimentRun);
  }
  return candidates.length > 0 ? candidates[candidates.length - 1] : null;
};
