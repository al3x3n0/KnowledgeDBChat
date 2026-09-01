/**
 * Small pure helpers about an agent run's detail view.
 *
 * All of these were module-scope constants inside AutonomousAgentsPage. None
 * of them touches component state; they were only there because that is where
 * the one caller lived. JobDetailPanel moved out of that file and needed them
 * too, so they are here rather than duplicated or reached for across a
 * 19,000-line page module.
 */

import type { AgentJob } from '../types';

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
