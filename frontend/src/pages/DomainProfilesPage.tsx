/**
 * Domain research profiles: a standing line of enquiry with an automation
 * policy, and the opportunities it turns up.
 *
 * A profile is not a research portfolio, though they render the same controls.
 * A profile is a question you keep asking; a portfolio is a programme that
 * accumulates. What they have in common is useOpportunitySurface.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { invalidateAgentRunQueries } from '../utils/agentRunQueries';
import type {
  DomainResearchProfilePolicyDraft,
} from '../components/agent/autonomyShared';
import type {
  DomainResearchProfileUpdate,
} from '../types';
import DomainProfilesTab from '../components/agent/tabs/DomainProfilesTab';
import toast from 'react-hot-toast';

import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../services/api';
import { useOpportunitySurface } from '../components/agent/useOpportunitySurface';
import { useFollowUpQueueActionMutation } from '../components/agent/agentJobMutations';

/** Links back to Runs: the jobs an opportunity launches live there. */
const buildRunsUrl = (jobId?: string, extras?: Record<string, string | null | undefined>) => {
  const next = new URLSearchParams();
  if (jobId && String(jobId).trim()) next.set('job', String(jobId).trim());
  Object.entries(extras || {}).forEach(([key, value]) => {
    const text = String(value ?? '').trim();
    if (text) next.set(key, text);
  });
  const query = next.toString();
  return query ? `/autonomous-agents?${query}` : '/autonomous-agents';
};

const DomainProfilesPage: React.FC = () => {

  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';

  const { data: documentSources } = useQuery(
    ['document-sources', 'all'],
    () => apiClient.getDocumentSources(),
    { staleTime: 30000 }
  );
  const codeSources = useMemo(() => {
    const items = (documentSources || []) as any[];
    return items.filter((s) => ['github', 'gitlab'].includes(
      String(s?.source_type || s?.sourceType || '').toLowerCase()));
  }, [documentSources]);

  const [activeFollowUpReviewKey, setActiveFollowUpReviewKey] = useState<string>('');
  const [followUpReviewNoteDrafts, setFollowUpReviewNoteDrafts] = useState<Record<string, string>>({});
  const followUpQueueActionMutation = useFollowUpQueueActionMutation({
    setActiveFollowUpReviewKey,
    setFollowUpReviewNoteDrafts,
    onFollowUpLaunched: (followUpJobId) => {
      navigate(buildRunsUrl(followUpJobId));
      return true;
    },
  });

  // Deep links still focus a card or an opportunity row and flash it, the way
  // they did when this was a tab: ?profileId=...&opportunityId=...
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const focusTarget = useMemo(() => {
    const ownerId = String(params.get('profileId') || '').trim();
    if (!ownerId) return null;
    const opportunityId = String(params.get('opportunityId') || '').trim();
    return { scope: 'domain' as const, ownerId, opportunityId: opportunityId || undefined };
  }, [params]);

  const surface = useOpportunitySurface({
    navigate,
    queryClient,
    isAdmin,
    codeSources,
    buildAutonomousAgentsUrl: buildRunsUrl,
    followUpQueueActionMutation,
    goToFleet: () => navigate('/research/fleet'),
    activeFollowUpReviewKey,
    setActiveFollowUpReviewKey,
    followUpReviewNoteDrafts,
    setFollowUpReviewNoteDrafts,
    focusTarget,
    expandTarget: (ownerId) => setExpandedDomainProfileIds((prev) => (prev[ownerId] ? prev : { ...prev, [ownerId]: true })),
  });
  const {
    beginOpportunityRelaunch,
    beginOpportunitySuppression,
    buildAutonomyCardKey,
    buildAutonomyOpportunityRowKey,
    buildAutonomyReviewRowKey,
    buildResearchNoteExperimentUrl,
    cancelOpportunityAction,
    createScientificResearchPackMutation,
    domainOpportunityActionMutation,
    domainProfilesData,
    domainProfilesLoading,
    highlightedAutonomyCardKey,
    highlightedAutonomyRowKey,
    opportunityNoteDraft,
    refetchDomainProfiles,
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
    setOpportunityNoteDraft,
    submitOpportunityAction,
    visibleScientificSandboxProfiles,
  } = surface;

  const [domainProfileTitle, setDomainProfileTitle] = useState('');
  const [domainProfileTopic, setDomainProfileTopic] = useState('');
  const [domainProfileObjective, setDomainProfileObjective] = useState('');
  const [domainProfileSourceScope, setDomainProfileSourceScope] = useState<'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo'>('kb_plus_arxiv_plus_repo');
  const [domainProfileQueriesText, setDomainProfileQueriesText] = useState('');
  const [domainProfileBenchmarkQueriesText, setDomainProfileBenchmarkQueriesText] = useState('');
  const [domainProfileCadenceMinutes, setDomainProfileCadenceMinutes] = useState('1440');
  const [domainProfileRepoSelection, setDomainProfileRepoSelection] = useState<Record<string, boolean>>({});
  const [domainProfilePolicyDrafts, setDomainProfilePolicyDrafts] = useState<Record<string, DomainResearchProfilePolicyDraft>>({});
  const [expandedDomainProfileIds, setExpandedDomainProfileIds] = useState<Record<string, boolean>>({});
  const updateDomainProfileMutation = useMutation(
    ({ profileId, data }: { profileId: string; data: DomainResearchProfileUpdate }) =>
      apiClient.updateDomainResearchProfile(profileId, data),
    {
      onSuccess: () => {
        invalidateAgentRunQueries(queryClient, [
          'domain-research-profiles',
        ]);
        toast.success('Domain profile settings updated');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update domain profile');
      },
    }
  );

  const [domainProfileTrackType, setDomainProfileTrackType] = useState<'compiler' | 'microarchitecture' | 'generic'>('compiler');
  const [domainProfileSandboxProfileId, setDomainProfileSandboxProfileId] = useState('');
  const domainAvailableSandboxProfiles = useMemo(
    () =>
      visibleScientificSandboxProfiles.filter((profile) => {
        const track = String(profile.track_type || '').trim().toLowerCase();
        return profile.enabled && (track === String(domainProfileTrackType).toLowerCase() || track === 'generic');
      }),
    [domainProfileTrackType, visibleScientificSandboxProfiles]
  );

  useEffect(() => {
    if (!domainAvailableSandboxProfiles.length) {
      setDomainProfileSandboxProfileId(resolveSandboxProfileId(domainProfileTrackType));
      return;
    }
    if (
      !domainProfileSandboxProfileId ||
      !domainAvailableSandboxProfiles.some((profile) => String(profile.id) === String(domainProfileSandboxProfileId))
    ) {
      setDomainProfileSandboxProfileId(resolveSandboxProfileId(domainProfileTrackType));
    }
  }, [domainAvailableSandboxProfiles, domainProfileSandboxProfileId, domainProfileTrackType, resolveSandboxProfileId]);

  return (
      <DomainProfilesTab
        domainProfilesData={domainProfilesData}
        domainProfilesLoading={domainProfilesLoading}
        refetchDomainProfiles={refetchDomainProfiles}
        beginOpportunityRelaunch={beginOpportunityRelaunch}
        beginOpportunitySuppression={beginOpportunitySuppression}
        buildAutonomousAgentsUrl={buildRunsUrl}
        buildAutonomyCardKey={buildAutonomyCardKey}
        buildAutonomyOpportunityRowKey={buildAutonomyOpportunityRowKey}
        buildAutonomyReviewRowKey={buildAutonomyReviewRowKey}
        buildResearchNoteExperimentUrl={buildResearchNoteExperimentUrl}
        cancelOpportunityAction={cancelOpportunityAction}
        codeSources={codeSources}
        createScientificResearchPackMutation={createScientificResearchPackMutation}
        domainAvailableSandboxProfiles={domainAvailableSandboxProfiles}
        domainOpportunityActionMutation={domainOpportunityActionMutation}
        domainProfileBenchmarkQueriesText={domainProfileBenchmarkQueriesText}
        setDomainProfileBenchmarkQueriesText={setDomainProfileBenchmarkQueriesText}
        domainProfileCadenceMinutes={domainProfileCadenceMinutes}
        setDomainProfileCadenceMinutes={setDomainProfileCadenceMinutes}
        domainProfileObjective={domainProfileObjective}
        setDomainProfileObjective={setDomainProfileObjective}
        domainProfilePolicyDrafts={domainProfilePolicyDrafts}
        setDomainProfilePolicyDrafts={setDomainProfilePolicyDrafts}
        domainProfileQueriesText={domainProfileQueriesText}
        setDomainProfileQueriesText={setDomainProfileQueriesText}
        domainProfileRepoSelection={domainProfileRepoSelection}
        setDomainProfileRepoSelection={setDomainProfileRepoSelection}
        domainProfileSandboxProfileId={domainProfileSandboxProfileId}
        setDomainProfileSandboxProfileId={setDomainProfileSandboxProfileId}
        domainProfileSourceScope={domainProfileSourceScope}
        setDomainProfileSourceScope={setDomainProfileSourceScope}
        domainProfileTitle={domainProfileTitle}
        setDomainProfileTitle={setDomainProfileTitle}
        domainProfileTopic={domainProfileTopic}
        setDomainProfileTopic={setDomainProfileTopic}
        domainProfileTrackType={domainProfileTrackType}
        setDomainProfileTrackType={setDomainProfileTrackType}
        expandedDomainProfileIds={expandedDomainProfileIds}
        setExpandedDomainProfileIds={setExpandedDomainProfileIds}
        highlightedAutonomyCardKey={highlightedAutonomyCardKey}
        highlightedAutonomyRowKey={highlightedAutonomyRowKey}
        navigate={navigate}
        opportunityNoteDraft={opportunityNoteDraft}
        setOpportunityNoteDraft={setOpportunityNoteDraft}
        queryClient={queryClient}
        registerAutonomyCardRef={registerAutonomyCardRef}
        registerAutonomyRowRef={registerAutonomyRowRef}
        renderAutonomySummaryRow={renderAutonomySummaryRow}
        renderBulkFollowUpControls={renderBulkFollowUpControls}
        renderInlineFollowUpApprovalRow={renderInlineFollowUpApprovalRow}
        renderInlineManualRecommendationRow={renderInlineManualRecommendationRow}
        renderInlineSuppressedRelaunchRow={renderInlineSuppressedRelaunchRow}
        renderOpportunityExplainabilityPanel={renderOpportunityExplainabilityPanel}
        renderScientificSandboxManagementPanel={renderScientificSandboxManagementPanel}
        renderScientificValidationRuns={renderScientificValidationRuns}
        resolveOpportunityContextRow={resolveOpportunityContextRow}
        resolveSandboxProfileId={resolveSandboxProfileId}
        scientificSandboxProfileById={scientificSandboxProfileById}
        submitOpportunityAction={submitOpportunityAction}
        updateDomainProfileMutation={updateDomainProfileMutation}
      />
  );
};

export default DomainProfilesPage;
