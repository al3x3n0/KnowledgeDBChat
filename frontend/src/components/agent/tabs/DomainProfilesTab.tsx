import React, { useCallback, useMemo } from 'react';
import Button from '../../../components/common/Button';
import {
  DEFAULT_VALIDATION_POLICY,
  DOMAIN_SOURCE_SCOPE_OPTIONS,
  DOMAIN_TRACK_OPTIONS,
  splitUniqueLines,
} from '../../../pages/autonomousAgentQuickStarts';
import type {
  ScientificSandboxProfile,
  DomainResearchProfile,
  DomainResearchProfileCreate,
  DomainResearchProfileUpdate,
} from '../../../types';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import {
  RefreshCw,
} from 'lucide-react';
import { apiClient } from '../../../services/api';
import {
} from '../../../utils/agentJobDetail';
import toast from 'react-hot-toast';
import { useMutation } from 'react-query';
import {
  AUTONOMY_FOCUS_CARD_CLASS,
  AUTONOMY_FOCUS_ROW_CLASS,
  SharedAutonomyMetricGrid,
  SharedAutonomyReviewLists,
  SharedPortfolioLikeAutonomyControls,
  canRelaunchOpportunityRow,
  formatAutonomyLabel,
  formatReviewModeLabel,
  renderOpportunityFollowUpOutcomeMeta,
  renderOpportunityReevaluationReviewMeta,
  researchOpportunityStageClass,
} from '../autonomyShared';
import type {
  DomainResearchProfilePolicyDraft,
} from '../autonomyShared';


const buildDomainResearchProfilePolicyDraft = (profile?: Partial<DomainResearchProfile> | null): DomainResearchProfilePolicyDraft => {
  const policy = ((profile?.effective_policy || profile?.automation_policy || {}) as Record<string, any>) || {};
  const automationProfile = String(profile?.automation_profile || 'balanced').trim().toLowerCase() === 'max_autonomy'
    ? 'max_autonomy'
    : 'balanced';
  return {
    automation_profile: automationProfile,
    follow_up_review_mode: (['auto_launch_safe', 'queue_for_approval', 'manual_only'].includes(String(policy.follow_up_review_mode || '').trim())
      ? String(policy.follow_up_review_mode).trim()
      : 'auto_launch_safe') as 'auto_launch_safe' | 'queue_for_approval' | 'manual_only',
    confidence_threshold: String(policy.confidence_threshold ?? profile?.confidence_threshold ?? (automationProfile === 'max_autonomy' ? 0.68 : 0.72)),
    experiment_readiness_threshold: String(policy.experiment_readiness_threshold ?? (automationProfile === 'max_autonomy' ? 0.72 : 0.8)),
    max_auto_follow_up_launches: String(policy.max_auto_follow_up_launches ?? (automationProfile === 'max_autonomy' ? 4 : 2)),
    max_concurrent_validation_runs: String(policy.max_concurrent_validation_runs ?? (automationProfile === 'max_autonomy' ? 2 : 1)),
    max_validation_runtime_minutes: String(policy.max_validation_runtime_minutes ?? (automationProfile === 'max_autonomy' ? 30 : 20)),
    max_validation_budget_per_run: String(policy.max_validation_budget_per_run ?? (automationProfile === 'max_autonomy' ? 50 : 25)),
    duplicate_window_items: String(policy.duplicate_window_items ?? (automationProfile === 'max_autonomy' ? 120 : 60)),
    auto_create_experiment_plans: Boolean(policy.auto_create_experiment_plans ?? profile?.auto_create_experiment_plans ?? true),
    auto_launch_follow_up: Boolean(policy.auto_launch_follow_up ?? profile?.auto_launch_follow_up ?? true),
    auto_launch_experiment_runs: Boolean(policy.auto_launch_experiment_runs ?? policy.auto_execute_validation_runs ?? (automationProfile === 'max_autonomy')),
  };
};

const buildDomainResearchProfileUpdatePayload = (draft: DomainResearchProfilePolicyDraft): DomainResearchProfileUpdate => ({
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

export interface DomainProfilesTabProps {
  domainProfilesData?: { items?: DomainResearchProfile[] };
  domainProfilesLoading: any;
  refetchDomainProfiles: any;
  beginOpportunityRelaunch: any;
  beginOpportunitySuppression: any;
  buildAutonomousAgentsUrl: any;
  buildAutonomyCardKey: any;
  buildAutonomyOpportunityRowKey: any;
  buildAutonomyReviewRowKey: any;
  buildResearchNoteExperimentUrl: any;
  cancelOpportunityAction: any;
  codeSources: any;
  createScientificResearchPackMutation: any;
  domainAvailableSandboxProfiles: ScientificSandboxProfile[];
  domainOpportunityActionMutation: any;
  domainProfileBenchmarkQueriesText: string;
  setDomainProfileBenchmarkQueriesText: React.Dispatch<React.SetStateAction<string>>;
  domainProfileCadenceMinutes: string;
  setDomainProfileCadenceMinutes: React.Dispatch<React.SetStateAction<string>>;
  domainProfileObjective: string;
  setDomainProfileObjective: React.Dispatch<React.SetStateAction<string>>;
  domainProfilePolicyDrafts: Record<string, DomainResearchProfilePolicyDraft>;
  setDomainProfilePolicyDrafts: React.Dispatch<React.SetStateAction<Record<string, DomainResearchProfilePolicyDraft>>>;
  domainProfileQueriesText: string;
  setDomainProfileQueriesText: React.Dispatch<React.SetStateAction<string>>;
  domainProfileRepoSelection: Record<string, boolean>;
  setDomainProfileRepoSelection: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  domainProfileSandboxProfileId: string;
  setDomainProfileSandboxProfileId: React.Dispatch<React.SetStateAction<string>>;
  domainProfileSourceScope: 'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo';
  setDomainProfileSourceScope: React.Dispatch<React.SetStateAction<'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo'>>;
  domainProfileTitle: string;
  setDomainProfileTitle: React.Dispatch<React.SetStateAction<string>>;
  domainProfileTopic: string;
  setDomainProfileTopic: React.Dispatch<React.SetStateAction<string>>;
  domainProfileTrackType: 'compiler' | 'microarchitecture' | 'generic';
  setDomainProfileTrackType: React.Dispatch<React.SetStateAction<'compiler' | 'microarchitecture' | 'generic'>>;
  expandedDomainProfileIds: Record<string, boolean>;
  setExpandedDomainProfileIds: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  highlightedAutonomyCardKey: string;
  highlightedAutonomyRowKey: string;
  navigate: any;
  opportunityNoteDraft: { mode: 'suppress' | 'launch' | 'relaunch'; surface: 'domain' | 'fleet'; ownerId: string; opportunityId: string; value: string } | null;
  setOpportunityNoteDraft: React.Dispatch<React.SetStateAction<{ mode: 'suppress' | 'launch' | 'relaunch'; surface: 'domain' | 'fleet'; ownerId: string; opportunityId: string; value: string } | null>>;
  queryClient: any;
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
  resolveOpportunityContextRow: any;
  resolveSandboxProfileId: any;
  scientificSandboxProfileById: any;
  submitOpportunityAction: any;
  updateDomainProfileMutation: any;
}

export const DomainProfilesTab: React.FC<DomainProfilesTabProps> = ({
  domainProfilesData,
  domainProfilesLoading,
  refetchDomainProfiles,
  beginOpportunityRelaunch,
  beginOpportunitySuppression,
  buildAutonomousAgentsUrl,
  buildAutonomyCardKey,
  buildAutonomyOpportunityRowKey,
  buildAutonomyReviewRowKey,
  buildResearchNoteExperimentUrl,
  cancelOpportunityAction,
  codeSources,
  createScientificResearchPackMutation,
  domainAvailableSandboxProfiles,
  domainOpportunityActionMutation,
  domainProfileBenchmarkQueriesText,
  setDomainProfileBenchmarkQueriesText,
  domainProfileCadenceMinutes,
  setDomainProfileCadenceMinutes,
  domainProfileObjective,
  setDomainProfileObjective,
  domainProfilePolicyDrafts,
  setDomainProfilePolicyDrafts,
  domainProfileQueriesText,
  setDomainProfileQueriesText,
  domainProfileRepoSelection,
  setDomainProfileRepoSelection,
  domainProfileSandboxProfileId,
  setDomainProfileSandboxProfileId,
  domainProfileSourceScope,
  setDomainProfileSourceScope,
  domainProfileTitle,
  setDomainProfileTitle,
  domainProfileTopic,
  setDomainProfileTopic,
  domainProfileTrackType,
  setDomainProfileTrackType,
  expandedDomainProfileIds,
  setExpandedDomainProfileIds,
  highlightedAutonomyCardKey,
  highlightedAutonomyRowKey,
  navigate,
  opportunityNoteDraft,
  setOpportunityNoteDraft,
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
  resolveOpportunityContextRow,
  resolveSandboxProfileId,
  scientificSandboxProfileById,
  submitOpportunityAction,
  updateDomainProfileMutation,
}) => {
  const selectedDomainProfileRepoSourceIds = useMemo(
    () => Object.entries(domainProfileRepoSelection).filter(([, enabled]) => enabled).map(([id]) => id),
    [domainProfileRepoSelection]
  );

  const createDomainProfileMutation = useMutation(
    (data: DomainResearchProfileCreate) => apiClient.createDomainResearchProfile(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Domain profile created');
        setDomainProfileTitle('');
        setDomainProfileTopic('');
        setDomainProfileObjective('');
        setDomainProfileTrackType('compiler');
        setDomainProfileSourceScope('kb_plus_arxiv_plus_repo');
        setDomainProfileQueriesText('');
        setDomainProfileBenchmarkQueriesText('');
        setDomainProfileCadenceMinutes('1440');
        setDomainProfileRepoSelection({});
        setDomainProfileSandboxProfileId(resolveSandboxProfileId('compiler'));
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create domain profile');
      },
    }
  );

  const domainProfileActionMutation = useMutation(
    ({
      profileId,
      action,
    }: {
      profileId: string;
      action: 'start' | 'pause' | 'resume' | 'cancel' | 'run_now';
    }) => apiClient.performDomainResearchProfileAction(profileId, { action }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Domain profile action failed');
      },
    }
  );

  const updateDomainProfilePolicyDraftField = useCallback(
    (profile: DomainResearchProfile, field: keyof DomainResearchProfilePolicyDraft, value: string | boolean) => {
      const profileId = String(profile.id || '');
      if (!profileId) return;
      setDomainProfilePolicyDrafts((prev) => {
        const current = prev[profileId] || buildDomainResearchProfilePolicyDraft(profile);
        return {
          ...prev,
          [profileId]: {
            ...current,
            [field]: value,
          },
        };
      });
    },
    [setDomainProfilePolicyDrafts]
  );

  const submitDomainProfilePolicyDraft = useCallback(
    (profile: DomainResearchProfile) => {
      const profileId = String(profile.id || '');
      const draft = domainProfilePolicyDrafts[profileId] || buildDomainResearchProfilePolicyDraft(profile);
      updateDomainProfileMutation.mutate({
        profileId,
        data: buildDomainResearchProfileUpdatePayload(draft),
      });
    },
    [domainProfilePolicyDrafts, updateDomainProfileMutation]
  );

  return (
    <div className="w-full flex flex-col min-h-0 gap-4">
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-1 bg-white border border-gray-200 rounded-lg p-4 space-y-3">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Domain Profiles</h2>
            <p className="text-sm text-gray-500">Saved R&D research monitors that persist notes, delta summaries, and experiment plans.</p>
          </div>
          <Button
            variant="secondary"
            disabled={createScientificResearchPackMutation.isLoading}
            onClick={() => createScientificResearchPackMutation.mutate()}
          >
            {createScientificResearchPackMutation.isLoading ? 'Seeding scientific pack...' : 'Seed Compiler + Microarch Pack'}
          </Button>
          {renderScientificSandboxManagementPanel()}
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            placeholder="Profile title"
            value={domainProfileTitle}
            onChange={(e) => setDomainProfileTitle(e.target.value)}
          />
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            placeholder="Domain or topic"
            value={domainProfileTopic}
            onChange={(e) => setDomainProfileTopic(e.target.value)}
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <select
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={domainProfileTrackType}
              onChange={(e) => setDomainProfileTrackType(e.target.value as any)}
            >
              {DOMAIN_TRACK_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label} track</option>
              ))}
            </select>
            <select
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={domainProfileSourceScope}
              onChange={(e) => setDomainProfileSourceScope(e.target.value as any)}
            >
              {DOMAIN_SOURCE_SCOPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>
          <select
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={domainProfileSandboxProfileId}
            onChange={(e) => setDomainProfileSandboxProfileId(e.target.value)}
          >
            {domainAvailableSandboxProfiles.map((profile) => (
              <option key={String(profile.id)} value={String(profile.id)}>
                {String(profile.name)} ({String(profile.track_type || 'generic')})
              </option>
            ))}
          </select>
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            rows={4}
            placeholder="Research objective"
            value={domainProfileObjective}
            onChange={(e) => setDomainProfileObjective(e.target.value)}
          />
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            rows={3}
            placeholder="Monitor queries, one per line"
            value={domainProfileQueriesText}
            onChange={(e) => setDomainProfileQueriesText(e.target.value)}
          />
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            rows={3}
            placeholder="Benchmark queries, one per line"
            value={domainProfileBenchmarkQueriesText}
            onChange={(e) => setDomainProfileBenchmarkQueriesText(e.target.value)}
          />
          {codeSources.length > 0 ? (
            <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
              <div className="text-xs font-medium text-gray-800 mb-2">Repository evidence sources</div>
              <div className="space-y-2 max-h-36 overflow-auto">
                {codeSources.map((source: any) => (
                  <label key={String(source.id)} className="flex items-start gap-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      checked={Boolean(domainProfileRepoSelection[String(source.id)])}
                      onChange={(e) => setDomainProfileRepoSelection((prev) => ({ ...prev, [String(source.id)]: e.target.checked }))}
                    />
                    <span>
                      <span className="font-medium text-gray-900">{String(source.name || source.id)}</span>
                      <span className="block text-xs text-gray-500">{String(source.source_type || '').toLowerCase()}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          ) : null}
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            placeholder="Cadence in minutes"
            value={domainProfileCadenceMinutes}
            onChange={(e) => setDomainProfileCadenceMinutes(e.target.value)}
          />
          <div className="flex gap-2">
            <Button
              variant="primary"
              disabled={createDomainProfileMutation.isLoading || !domainProfileTitle.trim() || !domainProfileTopic.trim() || !domainProfileObjective.trim()}
              onClick={() =>
                createDomainProfileMutation.mutate({
                  title: domainProfileTitle.trim(),
                  domain: domainProfileTopic.trim(),
                  objective: domainProfileObjective.trim(),
                  track_type: domainProfileTrackType,
                  source_scope: domainProfileSourceScope,
                  research_mode: 'literature_to_hypothesis',
                  monitor_queries: splitUniqueLines(domainProfileQueriesText, 12),
                  repo_source_ids: selectedDomainProfileRepoSourceIds.length ? selectedDomainProfileRepoSourceIds : undefined,
                  benchmark_queries: splitUniqueLines(domainProfileBenchmarkQueriesText, 16),
                  sandbox_profile_id: domainProfileSandboxProfileId || resolveSandboxProfileId(domainProfileTrackType),
                  scoring_policy: {
                    minimum_subscore: 0.6,
                    minimum_supporting_sources: 2,
                    weights: { novelty: 0.4, evidence: 0.35, testability: 0.25 },
                  },
                  selection_policy: { max_candidates: 10, max_hypotheses: 3 },
                  automation_profile: 'balanced',
                  automation_policy: DEFAULT_VALIDATION_POLICY,
                  interval_minutes: Number(domainProfileCadenceMinutes) > 0 ? Number(domainProfileCadenceMinutes) : 1440,
                  persist_artifacts: true,
                  auto_launch_follow_up: true,
                  auto_create_experiment_plans: true,
                  start_immediately: true,
                })
              }
            >
              Start Monitor
            </Button>
            <Button variant="ghost" onClick={() => refetchDomainProfiles()}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>
        <div className="col-span-2 bg-white border border-gray-200 rounded-lg p-4 min-h-0">
          {domainProfilesLoading ? (
            <div className="flex justify-center items-center h-48"><LoadingSpinner /></div>
          ) : (
            <div className="space-y-3">
              {(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).map((profile) => {
                const summary = (profile.latest_summary || {}) as Record<string, any>;
                const ideaTitles = Array.isArray(summary.ranked_opportunities) ? summary.ranked_opportunities.slice(0, 3) : [];
                const opportunities = Array.isArray(profile.opportunities) ? profile.opportunities : [];
                const autonomyMode = String(summary.autonomy_mode || profile.automation_profile || 'balanced');
                const effectivePolicy = ((profile.effective_policy || summary.effective_policy || profile.automation_policy || {}) as Record<string, any>) || {};
                const autonomyStateCounts = (summary.autonomy_state_counts || {}) as Record<string, any>;
                const schedulerSummary = (summary.scheduler_summary || {}) as Record<string, any>;
                const queuedReviewsCount = Number(summary.queued_operator_reviews_count || 0);
                const policyDraft = domainProfilePolicyDrafts[String(profile.id)] || buildDomainResearchProfilePolicyDraft(profile);
                const notesCount = Array.isArray(profile.latest_note_ids) ? profile.latest_note_ids.length : 0;
                const plansCount = Array.isArray(profile.latest_experiment_plan_ids) ? profile.latest_experiment_plan_ids.length : 0;
                const validationRuns = Array.isArray(profile.latest_validation_runs) ? profile.latest_validation_runs : [];
                const delta = (summary.delta_since_last_run || {}) as Record<string, any>;
                const profileCardKey = buildAutonomyCardKey('domain', String(profile.id));
                const isProfileExpanded = Boolean(expandedDomainProfileIds[String(profile.id)]);
                return (
                  <div
                    key={profile.id}
                    ref={registerAutonomyCardRef(profileCardKey)}
                    className={`border border-gray-200 rounded-lg p-4 transition-colors ${highlightedAutonomyCardKey === profileCardKey ? AUTONOMY_FOCUS_CARD_CLASS : ''}`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <h3 className="section-heading">{profile.title}</h3>
                          <span className="text-xs px-2 py-0.5 rounded bg-gray-200 text-gray-700">{profile.status}</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700">{profile.domain}</span>
                          <span className="text-xs px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">
                            {String(profile.track_type || 'generic').replaceAll('_', ' ')}
                          </span>
                          {plansCount > 0 ? (
                            <span className="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">
                              {plansCount} experiment plan{plansCount === 1 ? '' : 's'}
                            </span>
                          ) : null}
                        </div>
                        <div className="text-sm text-gray-600 whitespace-pre-wrap">{profile.objective}</div>
                        <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-3">
                          <span>Cadence {profile.interval_minutes}m</span>
                          <span>Mode {String(profile.research_mode || 'literature_to_hypothesis').replaceAll('_', ' ')}</span>
                          <span>Scope {String(profile.source_scope || 'kb_plus_arxiv').replaceAll('_', ' ')}</span>
                          <span>Notes {notesCount}</span>
                          <span>Plans {plansCount}</span>
                          <span>Validations {Array.isArray(profile.latest_validation_run_ids) ? profile.latest_validation_run_ids.length : 0}</span>
                          {profile.last_run_at ? <span>Last run {new Date(profile.last_run_at).toLocaleString()}</span> : null}
                        </div>
                        {summary.domain_summary ? (
                          <div className="mt-2 text-xs text-gray-600">{String(summary.domain_summary)}</div>
                        ) : null}
                        {Number(delta.new_signal_count || 0) > 0 ? (
                          <div className="mt-2 text-xs text-emerald-700">
                            New signals {Number(delta.new_signal_count || 0)}
                            {Array.isArray(delta.new_idea_titles) && delta.new_idea_titles.length > 0 ? ` · ${delta.new_idea_titles.slice(0, 2).join(', ')}` : ''}
                          </div>
                        ) : null}
                      </div>
                      <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                        {['draft', 'completed', 'cancelled'].includes(profile.status) ? (
                          <Button size="sm" variant="primary" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'start' })}>
                            Start
                          </Button>
                        ) : null}
                        {profile.status === 'running' ? (
                          <Button size="sm" variant="secondary" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'pause' })}>
                            Pause
                          </Button>
                        ) : null}
                        {profile.status === 'paused' ? (
                          <Button size="sm" variant="secondary" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'resume' })}>
                            Resume
                          </Button>
                        ) : null}
                        <Button size="sm" variant="ghost" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'run_now' })}>
                          Run Now
                        </Button>
                        {profile.status !== 'cancelled' ? (
                          <Button size="sm" variant="ghost" onClick={() => domainProfileActionMutation.mutate({ profileId: profile.id, action: 'cancel' })}>
                            Cancel
                          </Button>
                        ) : null}
                      </div>
                    </div>
                    <details
                      className="mt-3 bg-gray-50 border border-gray-100 rounded-lg p-3"
                      open={isProfileExpanded}
                      onToggle={(e) => {
                        const nextOpen = (e.currentTarget as HTMLDetailsElement).open;
                        setExpandedDomainProfileIds((prev) => ({ ...prev, [String(profile.id)]: nextOpen }));
                      }}
                    >
                      <summary className="cursor-pointer text-xs font-medium text-gray-800">Latest research ops state</summary>
                      <div className="mt-3 space-y-3 text-xs text-gray-700">
                        <SharedAutonomyMetricGrid
                          items={[
                            {
                              label: 'Fresh evidence',
                              value: `Docs ${Array.isArray(delta.new_document_ids) ? delta.new_document_ids.length : 0} · Repo ${Array.isArray(delta.new_repo_document_ids) ? delta.new_repo_document_ids.length : 0} · Papers ${Array.isArray(delta.new_paper_ids) ? delta.new_paper_ids.length : 0}`,
                            },
                            {
                              label: 'Novel ideas',
                              value: `${Number((summary.novelty_summary || {}).new_idea_count || 0)} new`,
                              detail: `Repeated ${Number((summary.novelty_summary || {}).repeated_idea_count || 0)}`,
                            },
                            {
                              label: 'Automation',
                              value: `${formatAutonomyLabel(autonomyMode)} · review ${formatReviewModeLabel(effectivePolicy.follow_up_review_mode || 'auto_launch_safe')}`,
                              detail: `Confidence ${Number(effectivePolicy.confidence_threshold ?? profile.confidence_threshold ?? 0.7).toFixed(2)} · Sandbox ${String(scientificSandboxProfileById[String(profile.sandbox_profile_id || '')]?.name || profile.sandbox_profile_id || 'default')}`,
                            },
                            {
                              label: 'Autonomy state',
                              value: `Eligible ${Number(autonomyStateCounts.eligible || 0)} · Active ${Number(autonomyStateCounts.active || 0)}`,
                              detail: `Waiting change ${Number(autonomyStateCounts.completed_waiting_change || 0)} · Structural blocked ${Number(autonomyStateCounts.blocked_structural || 0)}`,
                            },
                          ]}
                        />
                        <SharedAutonomyMetricGrid
                          items={[
                            { label: 'Next run', value: schedulerSummary.next_run_at ? new Date(String(schedulerSummary.next_run_at)).toLocaleString() : 'Not scheduled' },
                            { label: 'Pending approvals', value: Number(schedulerSummary.pending_follow_up_approvals_count || 0) },
                            { label: 'Manual recommendations', value: Number(schedulerSummary.manual_follow_up_recommendations_count || 0) },
                            { label: 'Suppressed relaunches', value: Number(schedulerSummary.suppressed_relaunches_count || 0) },
                          ]}
                        />
                        <SharedPortfolioLikeAutonomyControls
                          draft={policyDraft}
                          applyLabel="Save"
                          disabled={updateDomainProfileMutation.isLoading}
                          onApply={() => submitDomainProfilePolicyDraft(profile)}
                          onFieldChange={(field, value) => updateDomainProfilePolicyDraftField(profile, field, value)}
                        />
                        <div className="text-[11px] text-gray-500">
                          Queued reviews {queuedReviewsCount}
                        </div>
                        {renderBulkFollowUpControls(
                          'domain',
                          String(profile.id),
                          summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                          summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                          summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                          opportunities as Array<Record<string, any>> | undefined,
                        )}
                        {Array.isArray(summary.blocked_opportunities) && summary.blocked_opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Blocked opportunities</div>
                            <div className="mt-1 space-y-1">
                              {summary.blocked_opportunities.slice(0, 4).map((row: Record<string, any>, idx: number) => {
                                const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
                                return renderAutonomySummaryRow(
                                  'domain',
                                  String(profile.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Blocked opportunity')}{row.last_blocked_reason_code ? ` · ${String(row.last_blocked_reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
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
                                  'domain',
                                  String(profile.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Completed opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
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
                                  'domain',
                                  String(profile.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Cooldown opportunity')}{row.last_decision_reason_code ? ` · ${String(row.last_decision_reason_code)}` : row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
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
                                  'domain',
                                  String(profile.id),
                                  'suppressed',
                                  row,
                                  idx,
                                  <>
                                    <div>{String(row.title || row.canonical_key || 'Skipped opportunity')}{row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
                                    {renderOpportunityExplainabilityPanel(buildAutonomyReviewRowKey('domain', String(profile.id), 'suppressed', String(row.opportunity_id || row.canonical_key || idx)), resolvedRow, { surface: 'domain', ownerId: String(profile.id) })}
                                  </>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}
                        {summary.evidence_mix ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Evidence mix</div>
                            <div className="mt-1 text-gray-600">
                              KB {Number((summary.evidence_mix as any)?.documents || 0)}
                              {' '}· Repo {Number((summary.evidence_mix as any)?.repo_documents || 0)}
                              {' '}· Papers {Number((summary.evidence_mix as any)?.papers || 0)}
                            </div>
                          </div>
                        ) : null}
                        {ideaTitles.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Latest top ideas</div>
                            <div className="mt-1 space-y-1">
                              {ideaTitles.map((idea) => (
                                <div key={String(idea)}>{String(idea)}</div>
                              ))}
                            </div>
                          </div>
                        ) : null}
                        {opportunities.length > 0 ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Opportunity queue</div>
                            <div className="mt-2 space-y-2">
                              {opportunities.slice(0, 6).map((row) => {
                                const opportunityRowKey = buildAutonomyOpportunityRowKey('domain', String(profile.id), String(row.opportunity_id));
                                const opportunityNoteId = String((Array.isArray(row.source_note_ids) && row.source_note_ids.length > 0
                                  ? row.source_note_ids[0]
                                  : (Array.isArray(profile.latest_note_ids) && profile.latest_note_ids.length > 0 ? profile.latest_note_ids[0] : '')) || '').trim();
                                return (
                                <div
                                  key={row.opportunity_id}
                                  ref={registerAutonomyRowRef(opportunityRowKey)}
                                  className={`border border-gray-100 rounded p-2 transition-colors ${highlightedAutonomyRowKey === opportunityRowKey ? AUTONOMY_FOCUS_ROW_CLASS : ''}`}
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <div className="font-medium text-gray-900">{row.title}</div>
                                    <span className={`text-[11px] px-2 py-0.5 rounded ${researchOpportunityStageClass(row.stage)}`}>
                                      {row.stage}
                                    </span>
                                  </div>
                                  <div className="mt-1 text-gray-500">
                                    Confidence {Number(row.confidence || 0).toFixed(2)}
                                    {' '}· Novelty {Number(row.novelty || 0).toFixed(2)}
                                    {' '}· Readiness {Number(row.readiness || 0).toFixed(2)}
                                  </div>
                                  {row.operator_note ? <div className="mt-1 text-gray-500">Note: {row.operator_note}</div> : null}
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
                                  {renderOpportunityExplainabilityPanel(opportunityRowKey, row, { surface: 'domain', ownerId: String(profile.id) })}
                                  <div className="mt-2 flex flex-wrap gap-2">
                                    {row.decision_state !== 'accepted' ? (
                                      <Button
                                        size="sm"
                                        variant="secondary"
                                        onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'accept' })}
                                      >
                                        Accept
                                      </Button>
                                    ) : null}
                                    {row.decision_state !== 'suppressed' ? (
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => beginOpportunitySuppression('domain', profile.id, row)}
                                      >
                                        Suppress
                                      </Button>
                                    ) : (
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'reopen' })}
                                      >
                                        Reopen
                                      </Button>
                                    )}
                                    {row.decision_state === 'accepted' ? (
                                      <Button
                                        size="sm"
                                        variant="primary"
                                        onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'materialize_experiment', startImmediately: true })}
                                      >
                                        Run Experiment
                                      </Button>
                                    ) : null}
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={Array.isArray(row.linked_experiment_plan_ids) && row.linked_experiment_plan_ids.length > 0}
                                      onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'create_plan' })}
                                    >
                                      Create Plan
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={Array.isArray(row.linked_validation_run_ids) && row.linked_validation_run_ids.length > 0}
                                      onClick={() => domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'launch_validation' })}
                                    >
                                      Launch Validation
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      disabled={canRelaunchOpportunityRow(row) ? false : Array.isArray(row.child_job_ids) && row.child_job_ids.length > 0}
                                      onClick={() => (
                                        canRelaunchOpportunityRow(row)
                                          ? beginOpportunityRelaunch('domain', String(profile.id), row)
                                          : domainOpportunityActionMutation.mutate({ profileId: profile.id, opportunityId: row.opportunity_id, action: 'launch_follow_up' })
                                      )}
                                    >
                                      {canRelaunchOpportunityRow(row) ? 'Relaunch Follow-up' : 'Follow-up'}
                                    </Button>
                                  </div>
                                  {opportunityNoteDraft?.surface === 'domain'
                                  && String(opportunityNoteDraft.ownerId) === String(profile.id)
                                  && String(opportunityNoteDraft.opportunityId) === String(row.opportunity_id) ? (
                                    <div className={`mt-2 rounded p-2 ${opportunityNoteDraft.mode === 'suppress' ? 'border border-rose-200 bg-rose-50' : 'border border-emerald-200 bg-emerald-50'}`}>
                                      <div className={`text-[11px] font-medium ${opportunityNoteDraft.mode === 'suppress' ? 'text-rose-700' : 'text-emerald-700'}`}>
                                        {opportunityNoteDraft.mode === 'suppress' ? 'Suppression note' : 'Relaunch note'}
                                      </div>
                                      <textarea
                                        aria-label={opportunityNoteDraft.mode === 'suppress' ? 'Domain suppression note' : 'Domain relaunch note'}
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
                        <SharedAutonomyReviewLists
                          sections={[
                            { title: 'Queued operator reviews', rows: summary.queued_operator_reviews as Array<Record<string, any>> | undefined },
                            {
                              title: 'Pending approvals',
                              rows: summary.pending_follow_up_approvals as Array<Record<string, any>> | undefined,
                              renderRow: (row, idx) => renderInlineFollowUpApprovalRow('domain', String(profile.id), row, idx),
                            },
                            {
                              title: 'Manual recommendations',
                              rows: summary.manual_follow_up_recommendations as Array<Record<string, any>> | undefined,
                              renderRow: (row, idx) => renderInlineManualRecommendationRow(
                                'domain',
                                String(profile.id),
                                row,
                                idx,
                                opportunities as Array<Record<string, any>> | undefined,
                              ),
                            },
                            {
                              title: 'Suppressed relaunches',
                              rows: summary.suppressed_relaunches as Array<Record<string, any>> | undefined,
                              renderRow: (row, idx) => renderInlineSuppressedRelaunchRow(
                                'domain',
                                String(profile.id),
                                row,
                                idx,
                                opportunities as Array<Record<string, any>> | undefined,
                              ),
                            },
                          ]}
                        />
                        {(Array.isArray(profile.latest_note_ids) && profile.latest_note_ids.length > 0) || (Array.isArray(profile.latest_experiment_plan_ids) && profile.latest_experiment_plan_ids.length > 0) || validationRuns.length > 0 || (Array.isArray(profile.latest_validation_run_ids) && profile.latest_validation_run_ids.length > 0) ? (
                          <div className="bg-white border border-gray-200 rounded p-2">
                            <div className="font-medium text-gray-800">Artifacts</div>
                            {Array.isArray(profile.latest_note_ids) && profile.latest_note_ids.length > 0 ? (
                              <div className="mt-1 text-gray-600">Research notes: {profile.latest_note_ids.join(', ')}</div>
                            ) : null}
                            {Array.isArray(profile.latest_experiment_plan_ids) && profile.latest_experiment_plan_ids.length > 0 ? (
                              <div className="mt-1 text-gray-600">Experiment plans: {profile.latest_experiment_plan_ids.join(', ')}</div>
                            ) : null}
                            {validationRuns.length > 0 ? (
                              <div className="mt-2">{renderScientificValidationRuns(validationRuns as any, { ownerProfile: profile })}</div>
                            ) : Array.isArray(profile.latest_validation_run_ids) && profile.latest_validation_run_ids.length > 0 ? (
                              <div className="mt-1 text-gray-600">Validation runs: {profile.latest_validation_run_ids.join(', ')}</div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    </details>
                  </div>
                );
              })}
              {!(((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).length ? (
                <div className="text-sm text-gray-500">No domain profiles yet.</div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DomainProfilesTab;
