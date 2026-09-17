/**
 * Research portfolios: standing programmes, and what they have accumulated.
 *
 * A portfolio is not a campaign -- a campaign is a bounded enquiry that
 * concludes with evidence and gaps, a portfolio has no ending -- and not a
 * domain research profile either, though it shares every control with one.
 * See useOpportunitySurface for the machinery they share.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import type {
  ResearchPortfolioPolicyDraft,
} from '../components/agent/autonomyShared';
import type {
  ResearchPortfolioUpdate,
} from '../types';
import ResearchFleetTab from '../components/agent/tabs/ResearchFleetTab';
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

const ResearchFleetPage: React.FC = () => {

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
  // they did when this was a tab: ?fleetId=...&opportunityId=...
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const focusTarget = useMemo(() => {
    const ownerId = String(params.get('fleetId') || '').trim();
    if (!ownerId) return null;
    const opportunityId = String(params.get('opportunityId') || '').trim();
    return { scope: 'fleet' as const, ownerId, opportunityId: opportunityId || undefined };
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
    expandTarget: (ownerId) => setExpandedPortfolioIds((prev) => (prev[ownerId] ? prev : { ...prev, [ownerId]: true })),
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
    domainProfilesData,
    highlightedAutonomyCardKey,
    highlightedAutonomyRowKey,
    opportunityNoteDraft,
    refetchResearchPortfolios,
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
    researchPortfoliosData,
    researchPortfoliosLoading,
    resolveOpportunityContextRow,
    resolveSandboxProfileId,
    setOpportunityNoteDraft,
    submitOpportunityAction,
    visibleScientificSandboxProfiles,
  } = surface;

  const [portfolioTitle, setPortfolioTitle] = useState('');
  const [portfolioObjective, setPortfolioObjective] = useState('');
  const [portfolioProfileSelection, setPortfolioProfileSelection] = useState<Record<string, boolean>>({});
  const [portfolioPolicyDrafts, setPortfolioPolicyDrafts] = useState<Record<string, ResearchPortfolioPolicyDraft>>({});
  const [expandedPortfolioIds, setExpandedPortfolioIds] = useState<Record<string, boolean>>({});
  const updateResearchPortfolioMutation = useMutation(
    ({ portfolioId, data }: { portfolioId: string; data: ResearchPortfolioUpdate }) =>
      apiClient.updateResearchPortfolio(portfolioId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-portfolios']);
        toast.success('Research fleet settings updated');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update research fleet');
      },
    }
  );

  const [portfolioSandboxProfileId, setPortfolioSandboxProfileId] = useState('');
  const portfolioAvailableSandboxProfiles = useMemo(
    () => visibleScientificSandboxProfiles.filter((profile) => profile.enabled),
    [visibleScientificSandboxProfiles]
  );

  useEffect(() => {
    if (!portfolioAvailableSandboxProfiles.length) {
      setPortfolioSandboxProfileId(resolveSandboxProfileId('compiler'));
      return;
    }
    if (
      !portfolioSandboxProfileId ||
      !portfolioAvailableSandboxProfiles.some((profile) => String(profile.id) === String(portfolioSandboxProfileId))
    ) {
      setPortfolioSandboxProfileId(resolveSandboxProfileId('compiler'));
    }
  }, [portfolioAvailableSandboxProfiles, portfolioSandboxProfileId, resolveSandboxProfileId]);

  return (
      <ResearchFleetTab
        domainProfilesData={domainProfilesData}
        refetchResearchPortfolios={refetchResearchPortfolios}
        researchPortfoliosData={researchPortfoliosData}
        researchPortfoliosLoading={researchPortfoliosLoading}
        beginOpportunityRelaunch={beginOpportunityRelaunch}
        beginOpportunitySuppression={beginOpportunitySuppression}
        buildAutonomousAgentsUrl={buildRunsUrl}
        buildAutonomyCardKey={buildAutonomyCardKey}
        buildAutonomyOpportunityRowKey={buildAutonomyOpportunityRowKey}
        buildAutonomyReviewRowKey={buildAutonomyReviewRowKey}
        buildResearchNoteExperimentUrl={buildResearchNoteExperimentUrl}
        cancelOpportunityAction={cancelOpportunityAction}
        createScientificResearchPackMutation={createScientificResearchPackMutation}
        expandedPortfolioIds={expandedPortfolioIds}
        setExpandedPortfolioIds={setExpandedPortfolioIds}
        highlightedAutonomyCardKey={highlightedAutonomyCardKey}
        highlightedAutonomyRowKey={highlightedAutonomyRowKey}
        navigate={navigate}
        opportunityNoteDraft={opportunityNoteDraft}
        setOpportunityNoteDraft={setOpportunityNoteDraft}
        portfolioAvailableSandboxProfiles={portfolioAvailableSandboxProfiles}
        portfolioObjective={portfolioObjective}
        setPortfolioObjective={setPortfolioObjective}
        portfolioPolicyDrafts={portfolioPolicyDrafts}
        setPortfolioPolicyDrafts={setPortfolioPolicyDrafts}
        portfolioProfileSelection={portfolioProfileSelection}
        setPortfolioProfileSelection={setPortfolioProfileSelection}
        portfolioSandboxProfileId={portfolioSandboxProfileId}
        setPortfolioSandboxProfileId={setPortfolioSandboxProfileId}
        portfolioTitle={portfolioTitle}
        setPortfolioTitle={setPortfolioTitle}
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
        researchPortfolioOpportunityActionMutation={researchPortfolioOpportunityActionMutation}
        resolveOpportunityContextRow={resolveOpportunityContextRow}
        resolveSandboxProfileId={resolveSandboxProfileId}
        submitOpportunityAction={submitOpportunityAction}
        updateResearchPortfolioMutation={updateResearchPortfolioMutation}
      />
  );
};

export default ResearchFleetPage;
