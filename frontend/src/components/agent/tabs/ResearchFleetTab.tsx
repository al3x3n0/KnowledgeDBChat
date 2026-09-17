import type {
  AnyMutation,
  AnyVoidMutation,
  BuildRunsUrl,
  NavigateFunction,
  QueryClient,
  Refetch,
} from '../propTypes';
import {
  AUTONOMY_FOCUS_CARD_CLASS,
  AUTONOMY_FOCUS_ROW_CLASS,
  SharedAutonomyMetricGrid,
  SharedAutonomyReviewLists,
  SharedPortfolioLikeAutonomyControls,
  canRelaunchOpportunityRow,
  formatReviewModeLabel,
  renderOpportunityFollowUpOutcomeMeta,
  renderOpportunityReevaluationReviewMeta,
  researchOpportunityStageClass,
} from '../autonomyShared';
import Button from '../../../components/common/Button';
import {
  DEFAULT_VALIDATION_POLICY,
} from '../../../pages/autonomousAgentQuickStarts';
import type {
  ScientificSandboxProfile,
  DomainResearchProfile,
  ResearchOpportunity,
  ResearchPortfolio,
  ResearchPortfolioCreate,
  ResearchPortfolioUpdate,
} from '../../../types';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import {
  RefreshCw,
} from 'lucide-react';
import type {
  ResearchPortfolioPolicyDraft,
} from '../autonomyShared';
import { apiClient } from '../../../services/api';
import toast from 'react-hot-toast';
import React, { useCallback, useMemo } from 'react';
import { useMutation } from 'react-query';

const buildResearchPortfolioPolicyDraft = (portfolio?: Partial<ResearchPortfolio> | null): ResearchPortfolioPolicyDraft => {
  const policy = ((portfolio?.effective_policy || portfolio?.automation_policy || {}) as Record<string, any>) || {};
  const automationProfile = String(portfolio?.automation_profile || 'balanced').trim().toLowerCase() === 'max_autonomy'
    ? 'max_autonomy'
    : 'balanced';
  return {
    automation_profile: automationProfile,
    follow_up_review_mode: (['auto_launch_safe', 'queue_for_approval', 'manual_only'].includes(String(policy.follow_up_review_mode || '').trim())
      ? String(policy.follow_up_review_mode).trim()
      : 'auto_launch_safe') as 'auto_launch_safe' | 'queue_for_approval' | 'manual_only',
    confidence_threshold: String(policy.confidence_threshold ?? (automationProfile === 'max_autonomy' ? 0.68 : 0.72)),
    experiment_readiness_threshold: String(policy.experiment_readiness_threshold ?? (automationProfile === 'max_autonomy' ? 0.72 : 0.8)),
    max_auto_follow_up_launches: String(policy.max_auto_follow_up_launches ?? (automationProfile === 'max_autonomy' ? 4 : 2)),
    max_concurrent_validation_runs: String(policy.max_concurrent_validation_runs ?? (automationProfile === 'max_autonomy' ? 2 : 1)),
    max_validation_runtime_minutes: String(policy.max_validation_runtime_minutes ?? (automationProfile === 'max_autonomy' ? 30 : 20)),
    max_validation_budget_per_run: String(policy.max_validation_budget_per_run ?? (automationProfile === 'max_autonomy' ? 50 : 25)),
    duplicate_window_items: String(policy.duplicate_window_items ?? (automationProfile === 'max_autonomy' ? 120 : 60)),
    auto_create_experiment_plans: Boolean(policy.auto_create_experiment_plans ?? true),
    auto_launch_follow_up: Boolean(policy.auto_launch_follow_up ?? true),
    auto_launch_experiment_runs: Boolean(policy.auto_launch_experiment_runs ?? (automationProfile === 'max_autonomy')),
  };
};

const buildResearchPortfolioUpdatePayload = (draft: ResearchPortfolioPolicyDraft): ResearchPortfolioUpdate => ({
  automation_profile: draft.automation_profile,
  automation_policy: {
    follow_up_review_mode: draft.follow_up_review_mode,
    confidence_threshold: Number(draft.confidence_threshold || 0),
    experiment_readiness_threshold: Number(draft.experiment_readiness_threshold || 0),
    max_auto_follow_up_launches: Number(draft.max_auto_follow_up_launches || 0),
    max_concurrent_validation_runs: Number(draft.max_concurrent_validation_runs || 0),
    max_validation_runtime_minutes: Number(draft.max_validation_runtime_minutes || 0),
    max_validation_budget_per_run: Number(draft.max_validation_budget_per_run || 0),
    duplicate_window_items: Number(draft.duplicate_window_items || 0),
    auto_create_experiment_plans: draft.auto_create_experiment_plans,
    auto_launch_follow_up: draft.auto_launch_follow_up,
    auto_launch_experiment_runs: draft.auto_launch_experiment_runs,
    auto_execute_validation_runs: draft.auto_launch_experiment_runs,
  },
});

export interface ResearchFleetTabProps {
  domainProfilesData: any;
  refetchResearchPortfolios: Refetch;
  researchPortfoliosData: any;
  researchPortfoliosLoading: boolean;
  beginOpportunityRelaunch: any;
  beginOpportunitySuppression: any;
  buildAutonomousAgentsUrl: BuildRunsUrl;
  buildAutonomyCardKey: any;
  buildAutonomyOpportunityRowKey: any;
  buildAutonomyReviewRowKey: any;
  buildResearchNoteExperimentUrl: any;
  cancelOpportunityAction: any;
  createScientificResearchPackMutation: AnyVoidMutation;
  expandedPortfolioIds: Record<string, boolean>;
  setExpandedPortfolioIds: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  highlightedAutonomyCardKey: string;
  highlightedAutonomyRowKey: string;
  navigate: NavigateFunction;
  opportunityNoteDraft: { mode: 'suppress' | 'launch' | 'relaunch'; surface: 'domain' | 'fleet'; ownerId: string; opportunityId: string; value: string } | null;
  setOpportunityNoteDraft: React.Dispatch<React.SetStateAction<{ mode: 'suppress' | 'launch' | 'relaunch'; surface: 'domain' | 'fleet'; ownerId: string; opportunityId: string; value: string } | null>>;
  portfolioAvailableSandboxProfiles: ScientificSandboxProfile[];
  portfolioObjective: string;
  setPortfolioObjective: React.Dispatch<React.SetStateAction<string>>;
  portfolioPolicyDrafts: Record<string, ResearchPortfolioPolicyDraft>;
  setPortfolioPolicyDrafts: React.Dispatch<React.SetStateAction<Record<string, ResearchPortfolioPolicyDraft>>>;
  portfolioProfileSelection: Record<string, boolean>;
  setPortfolioProfileSelection: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  portfolioSandboxProfileId: string;
  setPortfolioSandboxProfileId: React.Dispatch<React.SetStateAction<string>>;
  portfolioTitle: string;
  setPortfolioTitle: React.Dispatch<React.SetStateAction<string>>;
  queryClient: QueryClient;
  registerAutonomyCardRef: any;
  registerAutonomyRowRef: any;
  renderAutonomySummaryRow: any;
  renderBulkFollowUpControls: any;
  renderInlineFollowUpApprovalRow: any;
  renderInlineManualRecommendationRow: any;
  renderInlineSuppressedRelaunchRow: any;
  renderOpportunityExplainabilityPanel: any;
  renderScientificSandboxManagementPanel: any;
  renderScientificValidationRuns: any;
  researchPortfolioOpportunityActionMutation: AnyMutation;
  resolveOpportunityContextRow: any;
  resolveSandboxProfileId: any;
  submitOpportunityAction: any;
  updateResearchPortfolioMutation: AnyMutation;
}

export const ResearchFleetTab: React.FC<ResearchFleetTabProps> = ({
  domainProfilesData,
  refetchResearchPortfolios,
  researchPortfoliosData,
  researchPortfoliosLoading,
  beginOpportunityRelaunch,
  beginOpportunitySuppression,
  buildAutonomousAgentsUrl,
  buildAutonomyCardKey,
  buildAutonomyOpportunityRowKey,
  buildAutonomyReviewRowKey,
  buildResearchNoteExperimentUrl,
  cancelOpportunityAction,
  createScientificResearchPackMutation,
  expandedPortfolioIds,
  setExpandedPortfolioIds,
  highlightedAutonomyCardKey,
  highlightedAutonomyRowKey,
  navigate,
  opportunityNoteDraft,
  setOpportunityNoteDraft,
  portfolioAvailableSandboxProfiles,
  portfolioObjective,
  setPortfolioObjective,
  portfolioPolicyDrafts,
  setPortfolioPolicyDrafts,
  portfolioProfileSelection,
  setPortfolioProfileSelection,
  portfolioSandboxProfileId,
  setPortfolioSandboxProfileId,
  portfolioTitle,
  setPortfolioTitle,
  queryClient,
  registerAutonomyCardRef,
  registerAutonomyRowRef,
  renderAutonomySummaryRow,
  renderBulkFollowUpControls,
  renderInlineFollowUpApprovalRow,
  renderInlineManualRecommendationRow,
  renderInlineSuppressedRelaunchRow,
  renderOpportunityExplainabilityPanel,
  renderScientificSandboxManagementPanel,
  renderScientificValidationRuns,
  researchPortfolioOpportunityActionMutation,
  resolveOpportunityContextRow,
  resolveSandboxProfileId,
  submitOpportunityAction,
  updateResearchPortfolioMutation,
}) => {
  const selectedPortfolioProfileIds = useMemo(
    () => Object.entries(portfolioProfileSelection).filter(([, enabled]) => enabled).map(([id]) => id),
    [portfolioProfileSelection]
  );

  const createResearchPortfolioMutation = useMutation(
    (data: ResearchPortfolioCreate) => apiClient.createResearchPortfolio(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Research fleet portfolio created');
        setPortfolioTitle('');
        setPortfolioObjective('');
        setPortfolioProfileSelection({});
        setPortfolioSandboxProfileId(resolveSandboxProfileId('generic'));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create research portfolio');
      },
    }
  );

  const researchPortfolioActionMutation = useMutation(
    ({
      portfolioId,
      action,
    }: {
      portfolioId: string;
      action: 'start' | 'pause' | 'resume' | 'cancel' | 'run_now';
    }) => apiClient.performResearchPortfolioAction(portfolioId, { action }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Research portfolio action failed');
      },
    }
  );

  const updatePortfolioPolicyDraftField = useCallback(
    (portfolio: ResearchPortfolio, field: keyof ResearchPortfolioPolicyDraft, value: string | boolean) => {
      const portfolioId = String(portfolio.id || '');
      if (!portfolioId) return;
      setPortfolioPolicyDrafts((prev) => {
        const current = prev[portfolioId] || buildResearchPortfolioPolicyDraft(portfolio);
        return {
          ...prev,
          [portfolioId]: {
            ...current,
            [field]: value,
          },
        };
      });
    },
    [setPortfolioPolicyDrafts]
  );

  const submitPortfolioPolicyDraft = useCallback(
    (portfolio: ResearchPortfolio) => {
      const portfolioId = String(portfolio.id || '');
      const draft = portfolioPolicyDrafts[portfolioId] || buildResearchPortfolioPolicyDraft(portfolio);
      updateResearchPortfolioMutation.mutate({
        portfolioId,
        data: buildResearchPortfolioUpdatePayload(draft),
      });
    },
    [portfolioPolicyDrafts, updateResearchPortfolioMutation]
  );

  return (
    <div className="w-full flex flex-col min-h-0 gap-4">
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-1 bg-white border border-gray-200 rounded-lg p-4 space-y-3">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Research Fleet</h2>
            <p className="text-sm text-gray-500">Coordinate multiple domain profiles into a mostly automatic experiment portfolio.</p>
          </div>
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            placeholder="Portfolio title"
            value={portfolioTitle}
            onChange={(e) => setPortfolioTitle(e.target.value)}
          />
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            rows={4}
            placeholder="Portfolio objective"
            value={portfolioObjective}
            onChange={(e) => setPortfolioObjective(e.target.value)}
          />
          <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
            <div className="text-xs font-medium text-gray-800 mb-2">Linked domain profiles</div>
            <div className="space-y-2 max-h-56 overflow-auto">
              {(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).map((profile) => (
                <label key={profile.id} className="flex items-start gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={Boolean(portfolioProfileSelection[profile.id])}
                    onChange={(e) => setPortfolioProfileSelection((prev) => ({ ...prev, [profile.id]: e.target.checked }))}
                  />
                  <span>
                    <span className="font-medium text-gray-900">{profile.title}</span>
                    <span className="block text-xs text-gray-500">{profile.domain}</span>
                  </span>
                </label>
              ))}
              {!(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).length ? (
                <div className="text-xs text-gray-500">Create domain profiles first.</div>
              ) : null}
            </div>
          </div>
          {renderScientificSandboxManagementPanel()}
          <select
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={portfolioSandboxProfileId}
            onChange={(e) => setPortfolioSandboxProfileId(e.target.value)}
          >
            {portfolioAvailableSandboxProfiles.map((profile) => (
              <option key={String(profile.id)} value={String(profile.id)}>
                {String(profile.name)} ({String(profile.track_type || 'generic')})
              </option>
            ))}
          </select>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              disabled={createScientificResearchPackMutation.isLoading}
              onClick={() => createScientificResearchPackMutation.mutate()}
            >
              {createScientificResearchPackMutation.isLoading ? 'Seeding pack...' : 'Seed Scientific Pack'}
            </Button>
            <Button
              variant="primary"
              disabled={createResearchPortfolioMutation.isLoading || !portfolioTitle.trim() || !portfolioObjective.trim() || selectedPortfolioProfileIds.length === 0}
              onClick={() =>
                createResearchPortfolioMutation.mutate({
                  title: portfolioTitle.trim(),
                  objective: portfolioObjective.trim(),
                  linked_profile_ids: selectedPortfolioProfileIds,
                  automation_profile: 'balanced',
                  automation_policy: {
                    ...DEFAULT_VALIDATION_POLICY,
                    duplicate_window_items: 120,
                  },
                  sandbox_profile_id: portfolioSandboxProfileId || resolveSandboxProfileId('compiler'),
                  start_immediately: true,
                })
              }
            >
              Start Fleet
            </Button>
            <Button variant="ghost" onClick={() => refetchResearchPortfolios()}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>
        <div className="col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
          {researchPortfoliosLoading ? (
            <div className="flex justify-center items-center h-48"><LoadingSpinner /></div>
          ) : (
            <div className="space-y-3">
              {(((researchPortfoliosData as any)?.items || []) as ResearchPortfolio[]).map((portfolio) => {
                const summary = (portfolio.latest_summary || {}) as Record<string, any>;
                const autonomyMode = String(summary.autonomy_mode || portfolio.automation_profile || 'balanced');
                const autonomySummary = (summary.autonomy_summary || {}) as Record<string, any>;
                const effectivePolicy = ((portfolio.effective_policy || summary.effective_policy || portfolio.automation_policy || {}) as Record<string, any>) || {};
                const policyDraft = portfolioPolicyDrafts[String(portfolio.id)] || buildResearchPortfolioPolicyDraft(portfolio);
                const opportunities = Array.isArray(portfolio.opportunities) ? (portfolio.opportunities as ResearchOpportunity[]) : [];
                const stageCounts = (summary.stage_counts || {}) as Record<string, any>;
                const autonomyStateCounts = (summary.autonomy_state_counts || {}) as Record<string, any>;
                const linkedProfiles = Array.isArray(portfolio.linked_profile_ids) ? portfolio.linked_profile_ids.length : 0;
                const plansCount = Array.isArray(portfolio.latest_experiment_plan_ids) ? portfolio.latest_experiment_plan_ids.length : 0;
                const validationCount = Array.isArray(portfolio.latest_validation_run_ids) ? portfolio.latest_validation_run_ids.length : 0;
                const validationRuns = Array.isArray(portfolio.latest_validation_runs) ? portfolio.latest_validation_runs : [];
                const recentValidationStats = validationRuns.reduce(
                  (acc, run) => {
                    const key = String(run.status || '').trim().toLowerCase();
                    if (!key) return acc;
                    acc[key] = Number(acc[key] || 0) + 1;
                    return acc;
                  },
                  {} as Record<string, number>
                );
                const childCount = Array.isArray(portfolio.child_job_ids) ? portfolio.child_job_ids.length : 0;
                const queuedReviewsCount = Number(summary.queued_operator_reviews_count || 0);
                const queuedReviewsByType = (summary.queued_operator_reviews_by_type || {}) as Record<string, any>;
                const schedulerSummary = (summary.scheduler_summary || {}) as Record<string, any>;
                const portfolioCardKey = buildAutonomyCardKey('fleet', String(portfolio.id));
                const isPortfolioExpanded = Boolean(expandedPortfolioIds[String(portfolio.id)]);
                return (
                  <div
                    key={portfolio.id}
                    ref={registerAutonomyCardRef(portfolioCardKey)}
                    className={`border border-gray-200 rounded-lg p-4 transition-colors ${highlightedAutonomyCardKey === portfolioCardKey ? AUTONOMY_FOCUS_CARD_CLASS : ''}`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h3 className="section-heading">{portfolio.title}</h3>
                          <span className="text-xs px-2 py-0.5 rounded bg-gray-200 text-gray-700">{portfolio.status}</span>
                          <span className={`text-xs px-2 py-0.5 rounded ${autonomyMode === 'max_autonomy' ? 'bg-amber-100 text-amber-800' : 'bg-blue-100 text-blue-700'}`}>
                            {autonomyMode === 'max_autonomy' ? 'max autonomy' : autonomyMode}
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">
                            {opportunities.length} opportunities
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 whitespace-pre-wrap">{portfolio.objective}</div>
                        <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-3">
                          <span>Profiles {linkedProfiles}</span>
                          <span>Plans {plansCount}</span>
                          <span>Validations {validationCount}</span>
                          <span>Follow-ups {childCount}</span>
                          {portfolio.last_run_at ? <span>Last run {new Date(portfolio.last_run_at).toLocaleString()}</span> : null}
                        </div>
                      </div>
                      <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                        {['draft', 'completed', 'cancelled'].includes(portfolio.status) ? (
                          <Button size="sm" variant="primary" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'start' })}>
                            Start
                          </Button>
                        ) : null}
                        {portfolio.status === 'running' ? (
                          <Button size="sm" variant="secondary" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'pause' })}>
                            Pause
                          </Button>
                        ) : null}
                        {portfolio.status === 'paused' ? (
                          <Button size="sm" variant="secondary" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'resume' })}>
                            Resume
                          </Button>
                        ) : null}
                        <Button size="sm" variant="ghost" onClick={() => researchPortfolioActionMutation.mutate({ portfolioId: portfolio.id, action: 'run_now' })}>
                          Run Now
                        </Button>
                      </div>
                    </div>
                    <details
                      className="mt-3 bg-gray-50 border border-gray-100 rounded-lg p-3"
                      open={isPortfolioExpanded}
                      onToggle={(e) => {
                        const nextOpen = (e.currentTarget as HTMLDetailsElement).open;
                        setExpandedPortfolioIds((prev) => ({ ...prev, [String(portfolio.id)]: nextOpen }));
                      }}
                    >
                      <summary className="cursor-pointer text-xs font-medium text-gray-800">Portfolio state</summary>
                      <div className="mt-3 space-y-3 text-xs text-gray-700">
                        <SharedAutonomyMetricGrid
                          columns="grid-cols-5"
                          items={[
                            { label: 'Discovered', value: Number(stageCounts.discovered || 0) },
                            { label: 'Planned', value: Number(stageCounts.planned || 0) },
                            { label: 'Validating', value: Number(stageCounts.validating || 0) },
                            { label: 'Validation runs', value: validationCount, detail: `Run ${Number(recentValidationStats.running || 0)} · Blocked ${Number(recentValidationStats.blocked || 0)}` },
                            { label: 'Suppressed', value: Number(stageCounts.suppressed || 0) },
                          ]}
                        />
                        <SharedAutonomyMetricGrid
                          items={[
                            { label: 'Blocked', value: Number(autonomySummary.blocked_opportunities_count || 0) },
                            { label: 'Dupes suppressed', value: Number(autonomySummary.suppressed_duplicates_count || 0) },
                            { label: 'Plans launched', value: Number(autonomySummary.created_experiment_plan_count || 0) },
                            { label: 'Follow-ups launched', value: Number(autonomySummary.launched_follow_up_job_count || 0) },
                          ]}
                        />
                        <SharedAutonomyMetricGrid
                          items={[
                            { label: 'Eligible now', value: Number(autonomyStateCounts.eligible || 0) },
                            { label: 'Cooling down', value: Number(autonomyStateCounts.cooldown || 0) },
                            { label: 'Waiting on change', value: Number(autonomyStateCounts.completed_waiting_change || 0) },
                            { label: 'Structurally blocked', value: Number(autonomyStateCounts.blocked_structural || 0) },
                          ]}
                        />
                        <SharedAutonomyMetricGrid
                          items={[
                            { label: 'Queued reviews', value: queuedReviewsCount },
                            { label: 'Follow-up approvals', value: Number(queuedReviewsByType.follow_up_recommendation || 0) },
                            { label: 'Policy reviews', value: Number(queuedReviewsByType.policy_review || 0) },
                            { label: 'Budget reviews', value: Number(queuedReviewsByType.budget_review || 0) },
                          ]}
                        />
                        <SharedAutonomyMetricGrid
                          items={[
                            { label: 'Next run', value: schedulerSummary.next_run_at ? new Date(String(schedulerSummary.next_run_at)).toLocaleString() : 'n/a' },
                            { label: 'Pending approvals', value: Number(schedulerSummary.pending_follow_up_approvals_count || 0) },
                            { label: 'Manual recommendations', value: Number(schedulerSummary.manual_follow_up_recommendations_count || 0) },
                            { label: 'Suppressed relaunches', value: Number(schedulerSummary.suppressed_relaunches_count || 0) },
                          ]}
                        />
                        <SharedPortfolioLikeAutonomyControls
                          draft={policyDraft}
                          applyLabel="Apply settings"
                          disabled={updateResearchPortfolioMutation.isLoading}
                          onApply={() => submitPortfolioPolicyDraft(portfolio)}
                          onFieldChange={(field, value) => updatePortfolioPolicyDraftField(portfolio, field, value)}
                        />
                        <div className="text-gray-500">
                          Effective policy: confidence {Number(effectivePolicy.confidence_threshold || 0).toFixed(2)}
                          {' '}· readiness {Number(effectivePolicy.experiment_readiness_threshold || 0).toFixed(2)}
                          {' '}· validation {effectivePolicy.auto_launch_experiment_runs ? 'on' : 'off'}
                          {' '}· review {formatReviewModeLabel(effectivePolicy.follow_up_review_mode || 'auto_launch_safe')}
                        </div>
                        {renderBulkFollowUpControls(
                          'fleet',
                          String(portfolio.id),
                          summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                          summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                          summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                          opportunities as Array<Record<string, any>> | undefined,
                        )}
                        <SharedAutonomyReviewLists
                          sections={[
                            { title: 'Queued operator reviews', rows: summary.queued_operator_reviews as Array<Record<string, any>> | undefined },
                            {
                              title: 'Pending follow-up approvals',
                              rows: summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                              renderRow: (row, idx) => renderInlineFollowUpApprovalRow('fleet', String(portfolio.id), row, idx),
                            },
                            {
                              title: 'Manual follow-up recommendations',
                              rows: summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                              renderRow: (row, idx) => renderInlineManualRecommendationRow(
                                'fleet',
                                String(portfolio.id),
                                row,
                                idx,
                                opportunities as Array<Record<string, any>> | undefined,
                              ),
                            },
                            {
                              title: 'Suppressed relaunches',
                              rows: summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                              renderRow: (row, idx) => renderInlineSuppressedRelaunchRow(
                                'fleet',
                                String(portfolio.id),
                                row,
                                idx,
                                opportunities as Array<Record<string, any>> | undefined,
                              ),
                            },
                          ]}
                        />
                        {Array.isArray(summary.auto_launch_decisions) && summary.auto_launch_decisions.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Automatic actions</div>
                            <div className="mt-1 space-y-1">
                              {summary.auto_launch_decisions.slice(0, 6).map((row: Record<string, any>, idx: number) => (
                                <div key={`${String(row.type || 'action')}-${idx}`}>
                                  {String(row.type || 'action').replace(/_/g, ' ')}
                                  {row.plan_id ? ` · Plan ${String(row.plan_id)}` : ''}
                                  {row.job_id ? ` · Job ${String(row.job_id)}` : ''}
                                  {row.reason_code ? ` · ${String(row.reason_code)}` : ''}
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : null}
                        {Array.isArray(summary.blocked_opportunities) && summary.blocked_opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Blocked opportunities</div>
                            <div className="mt-1 space-y-1">
                              {summary.blocked_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                return renderAutonomySummaryRow(
                                  'fleet',
                                  String(portfolio.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Blocked opportunity')}{row.last_blocked_reason_code ? ` · ${String(row.last_blocked_reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                  </>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}
                        {Array.isArray(summary.completed_waiting_change_opportunities) && summary.completed_waiting_change_opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Waiting on evidence change</div>
                            <div className="mt-1 space-y-1">
                              {summary.completed_waiting_change_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                return renderAutonomySummaryRow(
                                  'fleet',
                                  String(portfolio.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Completed opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                  </>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}
                        {Array.isArray(summary.cooldown_opportunities) && summary.cooldown_opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Cooldown opportunities</div>
                            <div className="mt-1 space-y-1">
                              {summary.cooldown_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                return renderAutonomySummaryRow(
                                  'fleet',
                                  String(portfolio.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Cooldown opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                  </>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}
                        {Array.isArray(summary.skipped_opportunities) && summary.skipped_opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Skipped opportunities</div>
                            <div className="mt-1 space-y-1">
                              {summary.skipped_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                return renderAutonomySummaryRow(
                                  'fleet',
                                  String(portfolio.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Skipped opportunity')}{row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('fleet', String(portfolio.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                  </>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}
                        {validationRuns.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Recent validation runs</div>
                            <div className="mt-2">{renderScientificValidationRuns(validationRuns as any)}</div>
                          </div>
                        ) : null}
                        {opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Top opportunities</div>
                            <div className="mt-2 space-y-2">
                              {opportunities.slice(0, 6).map((row) => {
                                const opportunityRowKey = buildAutonomyOpportunityRowKey('fleet', String(portfolio.id), String(row.opportunity_id || row.canonical_key || row.title));
                                const opportunityNoteId = String((Array.isArray(row.source_note_ids) && row.source_note_ids.length > 0
                                  ? row.source_note_ids[0]
                                  : (Array.isArray(portfolio.latest_note_ids) && portfolio.latest_note_ids.length > 0 ? portfolio.latest_note_ids[0] : '')) || '').trim();
                                return (
                                <div
                                  key={String(row.opportunity_id || row.canonical_key || row.title)}
                                  ref={registerAutonomyRowRef(opportunityRowKey)}
                                  className={`border border-gray-100 rounded p-2 transition-colors ${highlightedAutonomyRowKey === opportunityRowKey ? AUTONOMY_FOCUS_ROW_CLASS : ''}`}
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <div className="font-medium text-gray-900">{String(row.title || row.canonical_key)}</div>
                                    <span className={`text-[11px] px-2 py-0.5 rounded ${researchOpportunityStageClass(row.stage)}`}>
                                      {String(row.stage || 'discovered')}
                                    </span>
                                  </div>
                                  <div className="mt-1 text-gray-500">
                                    Confidence {Number(row.confidence || 0).toFixed(2)}
                                    {' '}· Novelty {Number(row.novelty || 0).toFixed(2)}
                                    {' '}· Readiness {Number(row.readiness || 0).toFixed(2)}
                                  </div>
                                  {row.operator_note ? (
                                    <div className="mt-1 text-gray-500">Note: {row.operator_note}</div>
                                  ) : null}
                                  <div className="mt-2 text-gray-500">
                                    Plans {Array.isArray(row.linked_experiment_plan_ids) ? row.linked_experiment_plan_ids.length : 0}
                                    {' '}· Runs {Array.isArray(row.linked_validation_run_ids) ? row.linked_validation_run_ids.length : 0}
                                    {' '}· Jobs {Array.isArray(row.child_job_ids) ? row.child_job_ids.length : 0}
                                  </div>
                                  {String(row.latest_experiment_plan_id || row.latest_validation_run_id || row.latest_validation_job_id || '').trim() ? (
                                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                                      {row.latest_experiment_plan_id ? <span>Latest plan {String(row.latest_experiment_plan_id).slice(0, 8)}</span> : null}
                                      {row.latest_validation_run_id ? <span>Run {String(row.latest_validation_run_id).slice(0, 8)}</span> : null}
                                      {row.latest_validation_status ? <span>Status {String(row.latest_validation_status).replace(/_/g, ' ')}</span> : null}
                                      {row.latest_validation_blocked_reason_code ? <span>Blocked {String(row.latest_validation_blocked_reason_code).replace(/_/g, ' ')}</span> : null}
                                      {row.latest_experiment_plan_id && opportunityNoteId ? (
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          className="!px-2 !py-1 !h-auto text-xs"
                                          onClick={() => navigate(buildResearchNoteExperimentUrl(opportunityNoteId, { plan: String(row.latest_experiment_plan_id) }))}
                                        >
                                          Open plan
                                        </Button>
                                      ) : null}
                                      {row.latest_validation_run_id && opportunityNoteId ? (
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          className="!px-2 !py-1 !h-auto text-xs"
                                          onClick={() => navigate(buildResearchNoteExperimentUrl(opportunityNoteId, { run: String(row.latest_validation_run_id) }))}
                                        >
                                          Open run
                                        </Button>
                                      ) : null}
                                      {row.latest_validation_job_id ? (
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          className="!px-2 !py-1 !h-auto text-xs"
                                          onClick={() => navigate(buildAutonomousAgentsUrl(String(row.latest_validation_job_id)), { replace: true })}
                                        >
                                          Open validation job
                                        </Button>
                                      ) : null}
                                    </div>
                                  ) : null}
                                  <div className="mt-1 text-gray-500">
                                    Autonomy {String(row.autonomy_state || 'eligible').replace(/_/g, ' ')}
                                    {row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : ''}
                                    {row.next_eligible_at ? ` · Next eligible ${new Date(row.next_eligible_at).toLocaleString()}` : ''}
                                  </div>
                                  {renderOpportunityReevaluationReviewMeta(row, (url) => navigate(url))}
                                  {renderOpportunityFollowUpOutcomeMeta(row)}
                                  {renderOpportunityExplainabilityPanel(opportunityRowKey, row, { surface: 'fleet', ownerId: String(portfolio.id) })}
                                  <div className="mt-2 flex flex-wrap gap-2">
                                    {row.decision_state !== 'accepted' ? (
                                      <Button
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'accept' })}
                                      >
                                        Accept
                                      </Button>
                                    ) : null}
                                    {row.decision_state !== 'suppressed' ? (
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => beginOpportunitySuppression('fleet', portfolio.id, row)}
                                      >
                                        Suppress
                                      </Button>
                                    ) : (
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'reopen' })}
                                      >
                                        Reopen
                                      </Button>
                                    )}
                                    {row.decision_state === 'accepted' ? (
                                      <Button
                                        size="sm"
                                        variant="primary"
                                        onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'materialize_experiment', startImmediately: true })}
                                      >
                                        Run Experiment
                                      </Button>
                                    ) : null}
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={Array.isArray(row.linked_experiment_plan_ids) && row.linked_experiment_plan_ids.length > 0}
                                      onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'create_plan' })}
                                    >
                                      Create Plan
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={Array.isArray(row.linked_validation_run_ids) && row.linked_validation_run_ids.length > 0}
                                      onClick={() => researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'launch_validation' })}
                                    >
                                      Launch Validation
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={canRelaunchOpportunityRow(row) ? false : Array.isArray(row.child_job_ids) && row.child_job_ids.length > 0}
                                      onClick={() => (
                                        canRelaunchOpportunityRow(row)
                                          ? beginOpportunityRelaunch('fleet', String(portfolio.id), row)
                                          : researchPortfolioOpportunityActionMutation.mutate({ portfolioId: portfolio.id, opportunityId: row.opportunity_id, action: 'launch_follow_up' })
                                      )}
                                    >
                                      {canRelaunchOpportunityRow(row) ? 'Relaunch Follow-up' : 'Follow-up'}
                                    </Button>
                                  </div>
                                  {opportunityNoteDraft?.surface === 'fleet'
                                  && String(opportunityNoteDraft.ownerId) === String(portfolio.id)
                                  && String(opportunityNoteDraft.opportunityId) === String(row.opportunity_id) ? (
                                    <div className={`mt-2 rounded p-2 ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200 bg-rose-50' : 'border border-emerald-200 bg-emerald-50'}`}>
                                      <div className={`text-[11px] font-medium ${opportunityNoteDraft.mode === 'suppress' ? 'text-rose-700' : 'text-emerald-700'}`}>
                                        {opportunityNoteDraft.mode === 'suppress' ? 'Suppression note' : 'Relaunch note'}
                                      </div>
                                      <textarea
                                        aria-label={opportunityNoteDraft.mode === 'suppress' ? 'Fleet suppression note' : 'Fleet relaunch note'}
                                        className={`mt-2 w-full rounded px-2 py-1 text-xs ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200' : 'border border-emerald-200'}`}
                                        rows={3}
                                        value={opportunityNoteDraft.value}
                                        onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                                      />
                                      <div className="mt-2 flex gap-2">
                                        <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                                          {opportunityNoteDraft.mode === 'suppress' ? 'Save suppression' : 'Relaunch follow-up'}
                                        </Button>
                                        <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                                          Cancel
                                        </Button>
                                      </div>
                                    </div>
                                  ) : null}
                                </div>
                              );})}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </details>
                  </div>
                );
              })}
              {!(((researchPortfoliosData as any)?.items || []) as ResearchPortfolio[]).length ? (
                <div className="text-sm text-gray-500">No research portfolios yet.</div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResearchFleetTab;
