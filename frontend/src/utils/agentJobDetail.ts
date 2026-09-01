/**
 * Small pure helpers about an agent run's detail view.
 *
 * All of these were module-scope constants inside AutonomousAgentsPage. None
 * of them touches component state; they were only there because that is where
 * the one caller lived. JobDetailPanel moved out of that file and needed them
 * too, so they are here rather than duplicated or reached for across a
 * 19,000-line page module.
 */

import type {
  AgentJob,
  AgentJobCodePatchExecution,
  AgentJobCodePatchRecovery,
  AgentJobExecutionGraph,
} from '../types';

export const formatSchedulerTimestamp = (value: unknown): string | null => {
  const text = String(value || '').trim();
  if (!text) return null;
  const parsed = new Date(text);
  return Number.isNaN(parsed.getTime()) ? text : parsed.toLocaleString();
};

export const slugifyText = (value: string): string => {
  const text = String(value || '').trim().toLowerCase();
  return text
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48);
};

export const summarizeSchedulerState = (state: Record<string, any> | null | undefined): string[] => {
  if (!state || typeof state !== 'object') return [];
  const items: string[] = [];
  const lastRunStatus = String(state.last_run_status || '').trim();
  const failureStreak = Number(state.failure_streak || 0);
  const queueReason = String(state.queue_reason || '').trim();
  const lastScheduledAt = formatSchedulerTimestamp(state.last_scheduled_at);
  const lastDispatchedAt = formatSchedulerTimestamp(state.last_dispatched_at);
  const currentRunStartedAt = formatSchedulerTimestamp(state.current_run_started_at);
  const lastSuccessfulRunAt = formatSchedulerTimestamp(state.last_successful_run_at);
  const lastCompletedRunAt = formatSchedulerTimestamp(state.last_completed_run_at);
  const lastFailureAt = formatSchedulerTimestamp(state.last_failure_at);
  const backoffUntil = formatSchedulerTimestamp(state.backoff_until);
  const backoffSeconds = Number(state.backoff_seconds || 0);

  if (lastRunStatus) items.push(`Last run ${lastRunStatus}`);
  if (Number.isFinite(failureStreak) && failureStreak > 0) items.push(`Failure streak ${failureStreak}`);
  if (queueReason) items.push(`Queue reason ${queueReason.replace(/_/g, ' ')}`);
  if (lastScheduledAt) items.push(`Scheduled ${lastScheduledAt}`);
  if (lastDispatchedAt) items.push(`Dispatched ${lastDispatchedAt}`);
  if (currentRunStartedAt) items.push(`Run started ${currentRunStartedAt}`);
  if (lastSuccessfulRunAt) items.push(`Success ${lastSuccessfulRunAt}`);
  if (lastCompletedRunAt) items.push(`Completed ${lastCompletedRunAt}`);
  if (lastFailureAt) items.push(`Failed ${lastFailureAt}`);
  if (backoffUntil) items.push(`Backoff until ${backoffUntil}`);
  if (Number.isFinite(backoffSeconds) && backoffSeconds > 0) items.push(`Backoff ${backoffSeconds}s`);
  return items;
};

export type DomainResearchPromotionDraft = {
  title: string;
  interval_minutes: string;
  target_mode: 'profile_only' | 'profile_with_portfolio';
  portfolio_mode: 'existing' | 'new';
  portfolio_id: string;
  portfolio_title: string;
  start_profile_now: boolean;
  run_portfolio_now: boolean;
};

export const buildDomainResearchPromotionDraft = (
  job?: Pick<AgentJob, 'name' | 'config'> | null
): DomainResearchPromotionDraft => {
  const cfg = ((job?.config || {}) as Record<string, any>) || {};
  const domain = String(cfg.domain || '').trim();
  return {
    title: String(job?.name || '').trim() || (domain ? `${domain} Monitor` : 'Domain Research Monitor'),
    interval_minutes: String(cfg.interval_minutes ?? 1440),
    target_mode: 'profile_only',
    portfolio_mode: 'new',
    portfolio_id: '',
    portfolio_title: domain ? `${domain} Fleet` : 'Research Fleet',
    start_profile_now: true,
    run_portfolio_now: false,
  };
};

export const humanizeSwarmOutcome = (value?: string | null) =>
  String(value || '').trim().replace(/_/g, ' ') || 'unknown';

export const swarmOutcomeBadgeClass = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'verified_fix') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'repair_failed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'backlog_routed') return 'bg-amber-100 text-amber-800';
  if (normalized === 'stalled_after_handoff') return 'bg-cyan-100 text-cyan-800';
  return 'bg-slate-100 text-slate-700';
};

/**
 * Everything the execution-graph and scope views read off a job.
 *
 * These fourteen values were computed inline in JobDetailPanel, in three
 * clusters scattered across four hundred lines of a four-thousand-line
 * component body, and six of them were read by other sections too. That is
 * what makes the panel hard to split: its sections are not independent, they
 * share derivations.
 *
 * Pulling the derivations out is the precondition for pulling the sections
 * out — and unlike the sections, this is a pure function of a job, so it can
 * be tested without rendering anything.
 */
export interface ExecutionGraphView {
  executionGraph: AgentJobExecutionGraph | null;
  scopeObservability: Record<string, any> | null;
  graphHealth: Record<string, any> | null;
  dagStats: Record<string, any> | null;
  graphHealthStatus: string;
  graphHealthBadgeClass: string;
  graphRecommendedActions: string[];
  graphVerificationActions: Array<Record<string, any>>;
  graphSummarizationActions: Array<Record<string, any>>;
  scopeResolvedId: string;
  scopeSource: string;
  scopeEvents: Array<Record<string, any>>;
  recentScopeEvents: Array<Record<string, any>>;
  scopeGuardBlocks: number;
}

export const executionGraphView = (job: AgentJob): ExecutionGraphView => {
  const strategy = (job.results as any)?.execution_strategy;

  const executionGraph =
    strategy?.execution_graph && typeof strategy.execution_graph === 'object'
      ? (strategy.execution_graph as AgentJobExecutionGraph)
      : null;
  const scopeObservability =
    strategy?.scope_observability && typeof strategy.scope_observability === 'object'
      ? (strategy.scope_observability as Record<string, any>)
      : null;

  const graphHealth =
    (executionGraph as any)?.graph_health &&
    typeof (executionGraph as any).graph_health === 'object'
      ? ((executionGraph as any).graph_health as Record<string, any>)
      : null;
  const dagStats =
    (executionGraph as any)?.dag_stats && typeof (executionGraph as any).dag_stats === 'object'
      ? ((executionGraph as any).dag_stats as Record<string, any>)
      : null;
  const graphHealthStatus = String(graphHealth?.status || '').toLowerCase();
  const graphHealthBadgeClass =
    graphHealthStatus === 'critical'
      ? 'bg-red-50 text-red-700 border-red-200'
      : graphHealthStatus === 'warning'
        ? 'bg-amber-50 text-amber-700 border-amber-200'
        : graphHealthStatus === 'ok'
          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
          : 'bg-gray-50 text-gray-700 border-gray-200';

  const graphRecommendedActions = Array.isArray((executionGraph as any)?.recommended_actions)
    ? ((executionGraph as any).recommended_actions as any[])
        .filter((x: any) => String(x || '').trim())
        .slice(0, 6)
    : [];
  const graphVerificationActions = Array.isArray((executionGraph as any)?.verification_actions)
    ? ((executionGraph as any).verification_actions as Array<Record<string, any>>)
    : [];
  const graphSummarizationActions = Array.isArray((executionGraph as any)?.summarization_actions)
    ? ((executionGraph as any).summarization_actions as Array<Record<string, any>>)
    : [];

  const scopeEvents = Array.isArray(scopeObservability?.events)
    ? (scopeObservability?.events as Array<Record<string, any>>)
    : [];

  return {
    executionGraph,
    scopeObservability,
    graphHealth,
    dagStats,
    graphHealthStatus,
    graphHealthBadgeClass,
    graphRecommendedActions,
    graphVerificationActions,
    graphSummarizationActions,
    scopeResolvedId: String(scopeObservability?.resolved_scope_id || '').trim(),
    scopeSource: String(scopeObservability?.scope_source || '').trim(),
    scopeEvents,
    // The four most recent, newest first — what the panel shows.
    recentScopeEvents: scopeEvents
      .slice(-4)
      .reverse()
      .filter((event) => event && typeof event === 'object'),
    scopeGuardBlocks: scopeEvents.filter(
      (event) => String(event?.type || '').trim() === 'scope_guard_blocked'
    ).length,
  };
};


/**
 * Everything the code-patch views read off a job.
 *
 * Sixteen values, computed in four clusters spread over a hundred and fifty
 * lines of JobDetailPanel, six of them as separate useMemos over the same
 * `job.results`. They are the bulk of what makes the customer-research
 * section look immovable: that section reads 44 values from the panel, and
 * these are most of them.
 *
 * Pure, like the graph view: derived from a job and nothing else, so the
 * panel can memoise the whole thing once instead of six times, and so the
 * shapes can be checked without rendering a patch panel.
 */
export interface CodePatchProposalRef {
  proposal_id: string;
  title: string;
  summary: string;
}

export interface CodePatchView {
  codePatchProposal: CodePatchProposalRef | null;
  codePatchExecution: AgentJobCodePatchExecution | null;
  codePatchWorkspace: any;
  codePatchVerificationPlan: any;
  codePatchExecutionPlan: any[];
  codePatchRecovery: AgentJobCodePatchRecovery | null;
  codePatchDetectedStack: string[];
  codePatchVerificationCommands: any[];
  codePatchBootstrapCommands: any[];
  codePatchFallbackCommands: any[];
  codePatchFailedCommands: any[];
  codePatchSuggestedActions: any[];
  codingRecoveryState: string;
  codePatchProposals: CodePatchProposalRef[];
  codePatchApply: any;
  codePatchKbApply: any;
}

export const codePatchView = (job: AgentJob): CodePatchView => {
  const codePatchProposal = ((): any => {
    const fromResults = (job.results as any)?.code_patch;
    if (fromResults?.proposal_id) {
      return {
        proposal_id: String(fromResults.proposal_id),
        title: String(fromResults.title || 'Code Patch Proposal'),
        summary: fromResults.summary ? String(fromResults.summary) : '',
      };
    }
    const arts = (job.output_artifacts as any[]) || [];
    const art = arts.find((a) => a?.type === 'code_patch_proposal' && a?.id);
    if (art?.id) {
      return { proposal_id: String(art.id), title: String(art.title || 'Code Patch Proposal'), summary: '' };
    }
    return null;
  })();
  const codePatchExecution = ((): any => {
    const payload = (job.results as any)?.code_patch_execution;
    if (!payload || typeof payload !== 'object') return null;
    return payload as AgentJobCodePatchExecution;
  })();
  const codePatchWorkspace = codePatchExecution?.workspace || null;
  const codePatchVerificationPlan = codePatchExecution?.verification_plan || null;
  const codePatchExecutionPlan = Array.isArray(codePatchExecution?.execution_plan)
    ? (codePatchExecution?.execution_plan || [])
    : [];
  const codePatchRecovery = (codePatchExecution?.recovery || null) as AgentJobCodePatchRecovery | null;
  const codePatchDetectedStack = Array.isArray((codePatchExecution?.inferred_project_profile as any)?.detected_stack)
    ? ((codePatchExecution?.inferred_project_profile as any)?.detected_stack as any[])
        .map((item) => String(item || '').trim())
        .filter(Boolean)
    : [];
  const codePatchVerificationCommands = Array.isArray(codePatchVerificationPlan?.commands)
    ? (codePatchVerificationPlan?.commands || [])
    : [];
  const codePatchBootstrapCommands = Array.isArray(codePatchVerificationPlan?.bootstrap_commands)
    ? (codePatchVerificationPlan?.bootstrap_commands || [])
    : [];
  const codePatchFallbackCommands = Array.isArray(codePatchVerificationPlan?.fallback_commands)
    ? (codePatchVerificationPlan?.fallback_commands || [])
    : [];
  const codePatchFailedCommands = Array.isArray(codePatchRecovery?.last_failed_commands)
    ? (codePatchRecovery?.last_failed_commands || [])
    : [];
  const codePatchSuggestedActions = Array.isArray(codePatchRecovery?.suggested_operator_actions)
    ? (codePatchRecovery?.suggested_operator_actions || [])
    : [];
  const codingRecoveryState = String(codePatchRecovery?.recovery_state || '').trim().toLowerCase();
  const codePatchProposals = ((): any => {
    const seen = new Set<string>();
    const out: Array<{ proposal_id: string; title: string; summary: string }> = [];
    const hist = (job.results as any)?.code_patches;
    if (Array.isArray(hist)) {
      for (const p of hist) {
        const id = String(p?.proposal_id || '').trim();
        if (!id || seen.has(id)) continue;
        seen.add(id);
        out.push({
          proposal_id: id,
          title: String(p?.title || 'Code Patch Proposal'),
          summary: p?.summary ? String(p.summary) : '',
        });
      }
    }
    const cur = (job.results as any)?.code_patch;
    if (cur?.proposal_id) {
      const id = String(cur.proposal_id).trim();
      if (id && !seen.has(id)) {
        out.push({
          proposal_id: id,
          title: String(cur?.title || 'Code Patch Proposal'),
          summary: cur?.summary ? String(cur.summary) : '',
        });
      }
    }
    return out;
  })();
  const codePatchApply = ((): any => {
    const v = (job.results as any)?.code_patch_apply;
    if (v && typeof v === 'object') return v as any;
    return null;
  })();

  const codePatchKbApply = ((): any => {
    const v = (job.results as any)?.code_patch_kb_apply;
    if (v && typeof v === 'object') return v as any;
    return null;
  })();

  return {
    codePatchProposal,
    codePatchExecution,
    codePatchWorkspace,
    codePatchVerificationPlan,
    codePatchExecutionPlan,
    codePatchRecovery,
    codePatchDetectedStack,
    codePatchVerificationCommands,
    codePatchBootstrapCommands,
    codePatchFallbackCommands,
    codePatchFailedCommands,
    codePatchSuggestedActions,
    codingRecoveryState,
    codePatchProposals,
    codePatchApply,
    codePatchKbApply,
  };
};
