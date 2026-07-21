import type { AgentJobExperimentRun, AgentJobOperatorIntervention } from '../types';

export interface ExperimentRunSummary {
  verificationCommands: string[];
  bootstrapCommands: string[];
  fallbackCommands: string[];
  phases: string[];
  failedCommands: string[];
  finalPhase: string;
  sourceId: string;
  sourceName: string;
  detectedStack: string[];
}

export interface ExperimentRecoveryGuidance {
  reasons: string[];
  recommendedActions: string[];
}

export interface OperatorInterventionSummary {
  count: number;
  latestLabel: string;
  latestNote: string;
  latestOutcome: string;
  latestOutcomeReason: string;
  recentItems: string[];
}

export function summarizeExperimentRun(run: AgentJobExperimentRun | null | undefined): ExperimentRunSummary {
  const experimentRun = run && typeof run === 'object' ? run : null;
  const inferredProfile =
    experimentRun?.inferred_project_profile && typeof experimentRun.inferred_project_profile === 'object'
      ? experimentRun.inferred_project_profile
      : null;

  return {
    verificationCommands: Array.isArray(experimentRun?.verification_commands)
      ? (experimentRun?.verification_commands || [])
      : Array.isArray(experimentRun?.commands)
        ? (experimentRun?.commands || [])
        : [],
    bootstrapCommands: Array.isArray(experimentRun?.bootstrap_commands) ? (experimentRun?.bootstrap_commands || []) : [],
    fallbackCommands: Array.isArray(experimentRun?.fallback_commands) ? (experimentRun?.fallback_commands || []) : [],
    phases: Array.isArray(experimentRun?.phases) ? (experimentRun?.phases || []) : [],
    failedCommands: Array.isArray(experimentRun?.failed_commands) ? (experimentRun?.failed_commands || []) : [],
    finalPhase: String(experimentRun?.final_phase || '').trim(),
    sourceId: String(experimentRun?.source_id || '').trim(),
    sourceName: String(experimentRun?.source_name || '').trim(),
    detectedStack: Array.isArray(inferredProfile?.detected_stack) ? (inferredProfile?.detected_stack || []) : [],
  };
}

export function isExperimentRecoveryOpen(
  run: AgentJobExperimentRun | null | undefined,
  summary: ExperimentRunSummary = summarizeExperimentRun(run)
): boolean {
  return Boolean(summary.failedCommands.length > 0 && run?.fallback_attempted && !run?.fallback_ok);
}

export function getExperimentRecoveryPriority(
  run: AgentJobExperimentRun | null | undefined,
  summary: ExperimentRunSummary = summarizeExperimentRun(run)
): number {
  if (!run) return 0;
  if (isExperimentRecoveryOpen(run, summary)) return 5;
  if (run.fallback_attempted && run.fallback_ok) return 4;
  if (run.fallback_attempted) return 3;
  if (run.bootstrap_attempted && run.bootstrap_ok) return 2;
  if (run.bootstrap_attempted) return 1;
  return 0;
}

export function summarizeExperimentRecoveryGuidance(
  executionGraph: Record<string, any> | null | undefined
): ExperimentRecoveryGuidance {
  const graph = executionGraph && typeof executionGraph === 'object' ? executionGraph : null;
  const graphHealth = graph?.graph_health && typeof graph.graph_health === 'object' ? graph.graph_health : null;

  return {
    reasons: Array.isArray(graphHealth?.reasons)
      ? graphHealth.reasons.map((reason: unknown) => String(reason || '').trim()).filter(Boolean)
      : [],
    recommendedActions: Array.isArray(graph?.recommended_actions)
      ? (graph?.recommended_actions || []).map((action: unknown) => String(action || '').trim()).filter(Boolean)
      : [],
  };
}

export function summarizeOperatorInterventions(
  entries: AgentJobOperatorIntervention[] | null | undefined
): OperatorInterventionSummary {
  const rows = Array.isArray(entries) ? entries.filter((entry) => entry && typeof entry === 'object') : [];
  if (rows.length === 0) {
    return { count: 0, latestLabel: '', latestNote: '', latestOutcome: '', latestOutcomeReason: '', recentItems: [] };
  }

  const latest = rows[rows.length - 1];
  const recentItems = rows.slice(-3).map((entry) => {
    const actionLabel = String(entry?.action || '').trim().replace(/_/g, ' ');
    const before = String(entry?.job_status_before || '').trim();
    const after = String(entry?.job_status_after || '').trim();
    const note = String(entry?.note || '').trim();
    const outcome = String(entry?.outcome_status || '').trim().replace(/_/g, ' ');
    let line = actionLabel;
    if (before || after) {
      line += ` (${before || '?'} -> ${after || '?'})`;
    }
    if (note) {
      line += `: ${note}`;
    }
    if (outcome) {
      line += ` [${outcome}]`;
    }
    return line.trim();
  }).filter(Boolean);
  const action = String(latest?.action || '').trim().replace(/_/g, ' ');
  const statusBefore = String(latest?.job_status_before || '').trim();
  const statusAfter = String(latest?.job_status_after || '').trim();
  let latestLabel = action;
  if (statusBefore || statusAfter) {
    latestLabel += ` (${statusBefore || '?'} -> ${statusAfter || '?'})`;
  }

  return {
    count: rows.length,
    latestLabel: latestLabel.trim(),
    latestNote: String(latest?.note || '').trim(),
    latestOutcome: String(latest?.outcome_status || '').trim().replace(/_/g, ' '),
    latestOutcomeReason: String(latest?.outcome_reason || '').trim(),
    recentItems,
  };
}
