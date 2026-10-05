/**
 * Autonomy controls shared by the domain-profiles and research-fleet tabs.
 *
 * A domain research profile and a research portfolio are different things with
 * the same controls: an automation policy, a review mode, a metric grid, and
 * the opportunity/review lists underneath. Three of these were already named
 * `Shared*` on the page, which is where they stayed only because both readers
 * happened to live in the same file. Extracting either tab on its own would
 * have forced a copy.
 *
 * (What they do NOT share is a lifecycle -- see the campaigns page. A portfolio
 * accumulates; a campaign concludes. Same controls, different nouns.)
 */

import React from 'react';
import Button from '../common/Button';
import { humanizeDecisionTraceValue } from '../../utils/agentJobDetail';

export type DomainResearchProfilePolicyDraft = ResearchPortfolioPolicyDraft;

export const SharedAutonomyMetricGrid: React.FC<{
  columns?: string;
  items: Array<{ label: string; value: React.ReactNode; detail?: React.ReactNode }>;
}> = ({ columns = 'grid-cols-4', items }) => (
  <div className={`grid ${columns} gap-2`}>
    {items.map((item) => (
      <AutonomyStatCard key={item.label} label={item.label} value={item.value} detail={item.detail} />
    ))}
  </div>
);

export const SharedPortfolioLikeAutonomyControls: React.FC<{
  draft: ResearchPortfolioPolicyDraft;
  applyLabel: string;
  disabled?: boolean;
  onApply: () => void;
  onFieldChange: (field: keyof ResearchPortfolioPolicyDraft, value: any) => void;
}> = ({ draft, applyLabel, disabled, onApply, onFieldChange }) => (
  <div className="bg-white border border-gray-200 rounded p-2">
    <div className="flex items-center justify-between gap-2">
      <div className="font-medium text-gray-800">Autonomy controls</div>
      <Button size="sm" variant="secondary" onClick={onApply} disabled={disabled}>
        {applyLabel}
      </Button>
    </div>
    <div className="mt-2 grid grid-cols-2 gap-2">
      <label className="text-gray-600">
        Autonomy profile
        <select
          className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
          value={draft.automation_profile}
          onChange={(e) => onFieldChange('automation_profile', e.target.value as 'balanced' | 'max_autonomy')}
        >
          <option value="balanced">balanced</option>
          <option value="max_autonomy">max autonomy</option>
        </select>
      </label>
      <label className="text-gray-600">
        Review mode
        <select
          className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
          value={draft.follow_up_review_mode}
          onChange={(e) => onFieldChange('follow_up_review_mode', e.target.value as 'auto_launch_safe' | 'queue_for_approval' | 'manual_only')}
        >
          <option value="auto_launch_safe">auto launch safe</option>
          <option value="queue_for_approval">queue for approval</option>
          <option value="manual_only">manual only</option>
        </select>
      </label>
      <label className="text-gray-600">
        Confidence threshold
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.confidence_threshold} onChange={(e) => onFieldChange('confidence_threshold', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Readiness threshold
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.experiment_readiness_threshold} onChange={(e) => onFieldChange('experiment_readiness_threshold', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Follow-up cap
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_auto_follow_up_launches} onChange={(e) => onFieldChange('max_auto_follow_up_launches', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Validation concurrency
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_concurrent_validation_runs} onChange={(e) => onFieldChange('max_concurrent_validation_runs', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Duplicate window
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.duplicate_window_items} onChange={(e) => onFieldChange('duplicate_window_items', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Runtime minutes
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_validation_runtime_minutes} onChange={(e) => onFieldChange('max_validation_runtime_minutes', e.target.value)} />
      </label>
      <label className="text-gray-600">
        Budget per run
        <input className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs" value={draft.max_validation_budget_per_run} onChange={(e) => onFieldChange('max_validation_budget_per_run', e.target.value)} />
      </label>
    </div>
    <div className="mt-2 flex flex-wrap gap-3 text-gray-600">
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={draft.auto_create_experiment_plans} onChange={(e) => onFieldChange('auto_create_experiment_plans', e.target.checked)} />
        Auto-create plans
      </label>
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={draft.auto_launch_follow_up} onChange={(e) => onFieldChange('auto_launch_follow_up', e.target.checked)} />
        Auto-launch follow-up
      </label>
      <label className="inline-flex items-center gap-2">
        <input type="checkbox" checked={draft.auto_launch_experiment_runs} onChange={(e) => onFieldChange('auto_launch_experiment_runs', e.target.checked)} />
        Auto-launch validation
      </label>
    </div>
  </div>
);

export const SharedAutonomyReviewLists: React.FC<{
  sections: Array<{
    title: string;
    rows?: Array<Record<string, any>> | null;
    formatter?: (row: Record<string, any>) => React.ReactNode;
    renderRow?: (row: Record<string, any>, idx: number) => React.ReactNode;
    limit?: number;
  }>;
}> = ({ sections }) => (
  <>
    {sections
      .filter((section) => Array.isArray(section.rows) && section.rows.length > 0)
      .map((section) => (
        <div key={section.title} className="bg-white border border-gray-200 rounded p-2">
          <div className="font-medium text-gray-800">{section.title}</div>
          <div className="mt-1 space-y-1">
            {(section.rows || []).slice(0, section.limit || 4).map((row, idx) => {
              const key = `${String(row.opportunity_id || row.canonical_key || idx)}`;
              if (section.renderRow) {
                return (
                  <React.Fragment key={key}>
                    {section.renderRow(row, idx)}
                  </React.Fragment>
                );
              }
              return (
                <div key={key} className="text-gray-600">
                  {section.formatter
                    ? section.formatter(row)
                    : `${String(row.title || row.canonical_key || 'Opportunity')} · ${String(row.reason_code || row.review_type || 'review').replaceAll('_', ' ')}`}
                </div>
              );
            })}
          </div>
        </div>
      ))}
  </>
);

export const AUTONOMY_FOCUS_ROW_CLASS = 'border-cyan-300 bg-cyan-50 ring-2 ring-cyan-200';

export type ResearchPortfolioPolicyDraft = {
  automation_profile: 'balanced' | 'max_autonomy';
  follow_up_review_mode: 'auto_launch_safe' | 'queue_for_approval' | 'manual_only';
  confidence_threshold: string;
  experiment_readiness_threshold: string;
  max_auto_follow_up_launches: string;
  max_concurrent_validation_runs: string;
  max_validation_runtime_minutes: string;
  max_validation_budget_per_run: string;
  duplicate_window_items: string;
  auto_create_experiment_plans: boolean;
  auto_launch_follow_up: boolean;
  auto_launch_experiment_runs: boolean;
};

export const AutonomyStatCard: React.FC<{
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
}> = ({ label, value, detail }) => (
  <div className="bg-white border border-gray-200 rounded p-2">
    <div className="text-gray-500">{label}</div>
    <div className="mt-1 font-medium text-gray-900">{value}</div>
    {detail ? <div className="text-gray-500">{detail}</div> : null}
  </div>
);

export const AUTONOMY_FOCUS_CARD_CLASS = 'border-cyan-300 ring-2 ring-cyan-200';

export const researchOpportunityStageClass = (value?: string | null) => {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'blocked' || normalized === 'suppressed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'planned' || normalized === 'accepted') return 'bg-blue-100 text-blue-700';
  if (normalized === 'validating') return 'bg-amber-100 text-amber-800';
  return 'bg-gray-200 text-gray-700';
};

export const renderOpportunityFollowUpOutcomeMeta = (row: Record<string, any>) => {
  const outcomeStatus = String(row.follow_up_outcome_status || '').trim();
  if (!outcomeStatus) return null;
  const outcomeSummary = String(row.follow_up_outcome_summary || '').trim();
  const recordedAt = String(row.follow_up_outcome_recorded_at || '').trim();
  const childJobId = String(row.follow_up_last_job_id || (Array.isArray(row.child_job_ids) ? row.child_job_ids[0] : '') || '').trim();
  const badgeClass = outcomeStatus === 'completed'
    ? 'bg-emerald-100 text-emerald-700'
    : outcomeStatus === 'failed' || outcomeStatus === 'cancelled'
      ? 'bg-rose-100 text-rose-700'
      : 'bg-gray-200 text-gray-700';
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-600">
      <span className={`px-2 py-0.5 rounded ${badgeClass}`}>
        Outcome {humanizeDecisionTraceValue(outcomeStatus)}
      </span>
      {recordedAt ? <span>{new Date(recordedAt).toLocaleString()}</span> : null}
      {childJobId ? <span>Job {childJobId}</span> : null}
      {outcomeSummary ? <span>{outcomeSummary}</span> : null}
    </div>
  );
};

export const renderOpportunityReevaluationReviewMeta = (
  row: Record<string, any>,
  onNavigate?: (url: string) => void,
) => {
  const outcomeStatus = String(row.last_reevaluation_review_outcome || '').trim();
  if (!outcomeStatus) return null;
  const recordedAt = String(row.last_reevaluation_reviewed_at || '').trim();
  const reviewJobId = String(row.last_reevaluation_review_job_id || '').trim();
  const reviewNote = String(row.last_reevaluation_review_note || '').trim();
  const sourceNoteId = String(row.last_reevaluation_review_source_note_id || '').trim();
  const targetNoteId = String(row.last_reevaluation_review_target_note_id || '').trim();
  const openUrl = (url: string) => {
    if (!url || !onNavigate) return;
    onNavigate(url);
  };
  const badgeClass = outcomeStatus === 'dismissed'
    ? 'bg-gray-200 text-gray-700'
    : 'bg-violet-100 text-violet-700';
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-600">
      <span className={`px-2 py-0.5 rounded ${badgeClass}`}>
        Reevaluation {humanizeDecisionTraceValue(outcomeStatus)}
      </span>
      {recordedAt ? <span>{new Date(recordedAt).toLocaleString()}</span> : null}
      {reviewNote ? <span>{reviewNote}</span> : null}
      {reviewJobId ? (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => openUrl(`/synthesis?job=${encodeURIComponent(reviewJobId)}`)}
        >
          Open reevaluation job
        </Button>
      ) : null}
      {sourceNoteId ? (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => openUrl(`/research-notes?note=${encodeURIComponent(sourceNoteId)}`)}
        >
          Open source note
        </Button>
      ) : null}
      {targetNoteId && targetNoteId !== sourceNoteId ? (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => openUrl(`/research-notes?note=${encodeURIComponent(targetNoteId)}`)}
        >
          Open saved note
        </Button>
      ) : null}
    </div>
  );
};

export const canRelaunchOpportunityRow = (row: Record<string, any>) => {
  const outcomeStatus = String(row.follow_up_outcome_status || '').trim().toLowerCase();
  const lastJobId = String(row.follow_up_last_job_id || '').trim();
  return ['failed', 'cancelled'].includes(outcomeStatus) && Boolean(lastJobId);
};

export const formatAutonomyLabel = (value?: string | null) => String(value || 'balanced').replace(/_/g, ' ');

export const formatReviewModeLabel = (value?: string | null) => String(value || 'auto_launch_safe').replace(/_/g, ' ');
