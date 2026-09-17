/**
 * Research Inbox: opportunities the monitors surfaced, and what to do with them.
 *
 * Second-largest of the thirteen tabs -- 987 lines of body over 24 declarations
 * that nothing else on the page touches (eight mutations, the two bulk
 * follow-up state machines, and the paper-algorithm entrypoint settings).
 *
 * The filters are the exception and stay on the page: seven of them
 * (`inboxSearch`, `inboxStatusFilter`, `inboxTypeFilter`, `inboxJobFilter`,
 * `inboxCustomerFilter`, and the two drilldowns) are written by the health
 * tab's "show me this customer's inbox" handlers, so they are shared state, not
 * inbox state. Moving them here would have split each one in two -- the page
 * writing one copy, this component rendering another. See
 * tabs/__tests__/splitState.test.ts.
 *
 * This tab is destined for the Library door, beside Papers and Reading Lists:
 * it is triage of papers and documents, not a view of an agent run.
 */

import React, { useMemo, useState } from 'react';
import {
  Activity,
  Inbox,
  Link2,
  RefreshCw,
  RotateCcw,
  Search,
  Settings,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { useMutation } from 'react-query';

import Button from '../../common/Button';
import LoadingSpinner from '../../common/LoadingSpinner';
import { apiClient } from '../../../services/api';
import { invalidateAgentRunQueries } from '../../../utils/agentRunQueries';
import type { ResearchInboxItem, ResearchInboxItemStatus } from '../../../types';
import {
  formatInboxHealthDrilldownLabel,
  formatInboxPolicyDrilldownLabel,
} from '../drilldowns';
import type { InboxHealthDrilldown, InboxPolicyDrilldown } from '../drilldowns';

export interface ResearchInboxTabProps {
  chainsData: any;
  inboxLoading: any;
  inboxStats: any;
  myPreferences: any;
  refetchInbox: any;
  setActiveTab: any;
  setShowInboxMonitorModal: any;
  setShowMonitorProfilesModal: any;
  activeFollowUpReviewKey: string;
  buildAutonomousAgentsUrl: any;
  createFromChainMutation: any;
  createMutation: any;
  followUpQueueActionMutation: any;
  followUpReviewNoteDrafts: Record<string, string>;
  setFollowUpReviewNoteDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  healthCustomers: string[];
  inboxBulkFollowUpNote: string;
  setInboxBulkFollowUpNote: React.Dispatch<React.SetStateAction<string>>;
  inboxBulkRejectReason: string;
  setInboxBulkRejectReason: React.Dispatch<React.SetStateAction<string>>;
  inboxCustomerFilter: string;
  setInboxCustomerFilter: React.Dispatch<React.SetStateAction<string>>;
  inboxHealthDrilldown: InboxHealthDrilldown;
  setInboxHealthDrilldown: React.Dispatch<React.SetStateAction<InboxHealthDrilldown>>;
  inboxJobFilter: string;
  setInboxJobFilter: React.Dispatch<React.SetStateAction<string>>;
  inboxPolicyDrilldown: InboxPolicyDrilldown;
  setInboxPolicyDrilldown: React.Dispatch<React.SetStateAction<InboxPolicyDrilldown>>;
  inboxRejectReasonDrafts: Record<string, string>;
  setInboxRejectReasonDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  inboxSearch: string;
  setInboxSearch: React.Dispatch<React.SetStateAction<string>>;
  inboxStatusFilter: ResearchInboxItemStatus | '';
  setInboxStatusFilter: React.Dispatch<React.SetStateAction<ResearchInboxItemStatus | ''>>;
  inboxTypeFilter: string;
  setInboxTypeFilter: React.Dispatch<React.SetStateAction<string>>;
  location: { pathname: string; search: string };
  navigate: any;
  paperRepoSelectionDrafts: Record<string, string>;
  setPaperRepoSelectionDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  queryClient: any;
  selectedInboxIds: Record<string, boolean>;
  setSelectedInboxIds: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  selectedInboxItems: ResearchInboxItem[];
  unsafeExecBadge: { status: 'ready' | 'blocked' | 'off'; label: string; title: string; color: string };
  upsertMonitorProfileMutation: any;
  visibleInboxItems: ResearchInboxItem[];
}

export const ResearchInboxTab: React.FC<ResearchInboxTabProps> = ({
  chainsData,
  inboxLoading,
  inboxStats,
  myPreferences,
  refetchInbox,
  setActiveTab,
  setShowInboxMonitorModal,
  setShowMonitorProfilesModal,
  activeFollowUpReviewKey,
  buildAutonomousAgentsUrl,
  createFromChainMutation,
  createMutation,
  followUpQueueActionMutation,
  followUpReviewNoteDrafts,
  setFollowUpReviewNoteDrafts,
  healthCustomers,
  inboxBulkFollowUpNote,
  setInboxBulkFollowUpNote,
  inboxBulkRejectReason,
  setInboxBulkRejectReason,
  inboxCustomerFilter,
  setInboxCustomerFilter,
  inboxHealthDrilldown,
  setInboxHealthDrilldown,
  inboxJobFilter,
  setInboxJobFilter,
  inboxPolicyDrilldown,
  setInboxPolicyDrilldown,
  inboxRejectReasonDrafts,
  setInboxRejectReasonDrafts,
  inboxSearch,
  setInboxSearch,
  inboxStatusFilter,
  setInboxStatusFilter,
  inboxTypeFilter,
  setInboxTypeFilter,
  location,
  navigate,
  paperRepoSelectionDrafts,
  setPaperRepoSelectionDrafts,
  queryClient,
  selectedInboxIds,
  setSelectedInboxIds,
  selectedInboxItems,
  unsafeExecBadge,
  upsertMonitorProfileMutation,
  visibleInboxItems,
}) => {
  const [inboxResearchGoalDraft, setInboxResearchGoalDraft] = useState<string>(
    'Deep-dive on the selected Research Inbox items and propose concrete next steps (hypotheses + experiment plan).'
  );

  const [inboxMuteTokenDrafts, setInboxMuteTokenDrafts] = useState<Record<string, string>>({});

  const [inboxMutePhraseDrafts, setInboxMutePhraseDrafts] = useState<Record<string, string>>({});

  const deepLinkedInboxId = useMemo(() => new URLSearchParams(location.search).get('inbox'), [location.search]);

  const updateInboxItemMutation = useMutation(
    ({ itemId, data }: { itemId: string; data: { status?: ResearchInboxItemStatus; feedback?: string; metadata_patch?: Record<string, any> } }) =>
      apiClient.updateResearchInboxItem(itemId, data),
    {
      onSuccess: (_res, vars) => {
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
        ]);
        if (vars?.data?.status === 'rejected') {
          setInboxRejectReasonDrafts((current) => {
            const next = { ...current };
            delete next[String(vars.itemId || '')];
            return next;
          });
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Update failed');
      },
    }
  );

  const bulkUpdateInboxMutation = useMutation(
    ({ itemIds, data }: { itemIds: string[]; data: { status?: ResearchInboxItemStatus; feedback?: string } }) =>
      apiClient.bulkUpdateResearchInboxItems({ item_ids: itemIds, ...data }),
    {
      onSuccess: (res) => {
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
        ]);
        setSelectedInboxIds({});
        setInboxBulkRejectReason('');
        toast.success(`Updated ${res.updated} items`);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Bulk update failed');
      },
    }
  );

  const extractReposMutation = useMutation(
    (itemId: string) => apiClient.extractReposForInboxItem(itemId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Repo links extracted');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to extract repos');
      },
    }
  );

  const relaunchInboxFollowUpMutation = useMutation(
    ({ itemId, operatorNote }: { itemId: string; operatorNote?: string }) =>
      apiClient.relaunchInboxFollowUp(itemId, operatorNote ? { operator_note: operatorNote } : {}),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        toast.success('Follow-up relaunched');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to relaunch follow-up');
      },
    }
  );

  const ingestRepoMutation = useMutation(
    (payload: { provider: 'github' | 'gitlab'; repo: string }) =>
      apiClient.requestGitRepository({
        provider: payload.provider,
        repositories: [payload.repo],
        include_files: true,
        include_issues: false,
        include_pull_requests: false,
        include_wiki: false,
        incremental_files: true,
        use_gitignore: true,
        max_pages: 5,
        auto_sync: true,
      }),
    {
      onSuccess: (src) => {
        toast.success(`Repo ingestion started: ${src.name}`);
        // show in documents sources list
        navigate(`/documents`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error.message || 'Failed to ingest repo');
      },
    }
  );

  const runPaperRepoCodeAgent = async (item: ResearchInboxItem, chosenRepoOverride?: string) => {
    try {
      // Ensure repos are extracted
      let repos = (item.metadata as any)?.repos;
      if (!Array.isArray(repos) || repos.length === 0) {
        const res = await apiClient.extractReposForInboxItem(item.id);
        repos = res.repos;
      }
      const githubRepos = (repos || []).filter((r: any) => String(r?.provider) === 'github');
      if (githubRepos.length === 0) {
        toast.error('No GitHub repos found for this paper yet');
        return;
      }
      const defaultRepo = String(githubRepos[0].repo || '').trim();
      const chosenRepo = String(chosenRepoOverride || paperRepoSelectionDrafts[item.id] || defaultRepo).trim();
      if (!chosenRepo) {
        toast.error('Select a GitHub repo first');
        return;
      }

      const goal = `Implement or integrate the paper's repository changes relevant to our product. Start from the ingested repo and produce a minimal patch.\n\nPaper: ${item.title}\n\nAbstract:\n${item.summary || ''}`.slice(
        0,
        1600
      );

      const chains = ((chainsData as any)?.chains || []) as any[];
      const chain = chains.find((c: any) => c?.name === 'arxiv_repo_code_patch_chain');
      if (!chain?.id) {
        toast.error('Chain definition not found (arxiv_repo_code_patch_chain)');
        return;
      }

      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: `Paper→Repo→Code — ${new Date().toLocaleDateString()}`,
        variables: {
          inbox_item_id: item.id,
          provider: 'github',
          repo: chosenRepo,
          goal,
        },
        config_overrides: {
          inbox_item_id: item.id,
          provider: 'github',
          repo: chosenRepo,
          // help code patch proposer pick relevant files
          search_query: `${item.title}\n${item.summary || ''}`.slice(0, 500),
        },
        start_immediately: true,
      });
      setActiveTab('jobs');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to start chain');
    }
  };

  const runPaperAlgorithmProject = async (item: ResearchInboxItem, requestedBehavioralCheck: boolean, entrypoint: string) => {
    try {
      const chains = ((chainsData as any)?.chains || []) as any[];
      let preferredChainName = 'arxiv_algorithm_project_chain';
      let repos = (item.metadata as any)?.repos;
      if (!Array.isArray(repos) || repos.length === 0) {
        try {
          const res = await apiClient.extractReposForInboxItem(item.id);
          repos = res.repos;
        } catch {
          repos = repos || [];
        }
      }
      const hasGithubRepo = Array.isArray(repos) && repos.some((r: any) => String(r?.provider || '').toLowerCase() === 'github');
      if (hasGithubRepo) preferredChainName = 'arxiv_repo_algorithm_project_chain';

      const chain = chains.find((c: any) => c?.name === preferredChainName);
      if (!chain?.id) {
        toast.error(`Chain definition not found (${preferredChainName})`);
        return;
      }
      const allowBehavioral = !!requestedBehavioralCheck && unsafeExecBadge.status === 'ready';
      if (requestedBehavioralCheck && !allowBehavioral) {
        toast('Behavioral demo run requested, but server is not ready (see badge)');
      }
      const ep = String(entrypoint || 'demo.py').trim() || 'demo.py';
      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: `Paper→Algorithm — ${new Date().toLocaleDateString()}`,
        variables: { inbox_item_id: item.id },
        config_overrides: {
          inbox_item_id: item.id,
          language: 'python',
          include_tests: true,
          behavioral_check: allowBehavioral,
          entrypoint: ep,
        },
        start_immediately: true,
      });
      setActiveTab('jobs');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to start algorithm implementation');
    }
  };

  const inboxBulkFollowUpState = useMemo(() => {
    if (selectedInboxItems.length === 0) {
      return {
        enabled: false,
        disabledReason: 'Select one or more inbox items to use bulk follow-up actions.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    if (selectedInboxItems.some((item) => String(item.item_type || '').trim() !== 'follow_up_recommendation')) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions only support follow-up recommendations.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    const followUpRows = selectedInboxItems.map((item) => {
      const sourceKind = String(item.origin_source_kind || '').trim().toLowerCase();
      const ownerKind = sourceKind === 'profile'
        ? 'domain'
        : sourceKind === 'portfolio'
          ? 'fleet'
          : '';
      return {
        ownerKind,
        ownerId: String(item.origin_source_id || '').trim(),
        opportunityId: String(item.origin_opportunity_id || '').trim(),
        pendingApproval: String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval',
      };
    });
    if (followUpRows.some((row) => !row.pendingApproval)) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions only support pending approvals.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    if (followUpRows.some((row) => !row.ownerKind || !row.ownerId || !row.opportunityId)) {
      return {
        enabled: false,
        disabledReason: 'Selected inbox follow-up items are missing owner or opportunity identifiers.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    const ownerKinds = Array.from(new Set(followUpRows.map((row) => row.ownerKind)));
    if (ownerKinds.length !== 1) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions cannot mix domain and fleet owners.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    const ownerIds = Array.from(new Set(followUpRows.map((row) => row.ownerId)));
    if (ownerIds.length !== 1) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk follow-up actions must stay within one domain profile or research fleet.',
        ownerKind: '' as '' | 'domain' | 'fleet',
        ownerId: '',
        opportunityIds: [] as string[],
      };
    }
    return {
      enabled: true,
      disabledReason: '',
      ownerKind: ownerKinds[0] as 'domain' | 'fleet',
      ownerId: ownerIds[0],
      opportunityIds: Array.from(new Set(followUpRows.map((row) => row.opportunityId))),
    };
  }, [selectedInboxItems]);

  const inboxBulkRelaunchState = useMemo(() => {
    if (selectedInboxItems.length === 0) {
      return {
        enabled: false,
        disabledReason: 'Select one or more inbox items to use bulk relaunch.',
        itemIds: [] as string[],
      };
    }
    if (selectedInboxItems.some((item) => String(item.item_type || '').trim() !== 'follow_up_recommendation')) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk relaunch only supports follow-up recommendations.',
        itemIds: [] as string[],
      };
    }
    if (
      selectedInboxItems.some(
        (item) =>
          String(item.follow_up_launch_status || '').trim().toLowerCase() !== 'launched'
          || !['failed', 'cancelled'].includes(String(item.follow_up_outcome_status || '').trim().toLowerCase())
      )
    ) {
      return {
        enabled: false,
        disabledReason: 'Inbox bulk relaunch only supports failed or cancelled launched follow-ups.',
        itemIds: [] as string[],
      };
    }
    return {
      enabled: true,
      disabledReason: '',
      itemIds: selectedInboxItems.map((item) => String(item.id)),
    };
  }, [selectedInboxItems]);

  const bulkInboxFollowUpActionMutation = useMutation(
    ({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }: {
      domain_research_profile_id?: string;
      profile_opportunity_ids?: string[];
      portfolio_id?: string;
      portfolio_opportunity_ids?: string[];
      action: 'approve_launch' | 'reject_launch';
      operator_note?: string;
    }) => apiClient.bulkActionAgentCheckpointQueueFollowUp({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }),
    {
      onSuccess: (response) => {
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
          'agent-decision-trace',
          'agent-decision-trace-analytics',
          'notifications',
          'notifications-unread-count',
        ]);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setSelectedInboxIds((prev) => {
            const next = { ...prev };
            selectedInboxItems.forEach((item) => {
              const opportunityId = String(item.origin_opportunity_id || '').trim();
              if (successfulIds.has(opportunityId)) {
                delete next[String(item.id)];
              }
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setInboxBulkFollowUpNote('');
          toast.success(
            `Bulk follow-up ${response.applied === 1 ? 'action' : 'actions'} applied to ${response.applied} item${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk follow-up action failed');
      },
    }
  );

  const bulkInboxRelaunchMutation = useMutation(
    ({ item_ids, operator_note }: { item_ids: string[]; operator_note?: string }) =>
      apiClient.bulkRelaunchInboxFollowUp({ item_ids, operator_note }),
    {
      onSuccess: (response) => {
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
          'agent-decision-trace',
          'agent-decision-trace-analytics',
          'notifications',
          'notifications-unread-count',
        ]);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.item_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setSelectedInboxIds((prev) => {
            const next = { ...prev };
            selectedInboxItems.forEach((item) => {
              if (successfulIds.has(String(item.id))) {
                delete next[String(item.id)];
              }
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setInboxBulkFollowUpNote('');
          toast.success(
            `Bulk relaunch applied to ${response.applied} item${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.item_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk relaunch failed');
      },
    }
  );

  const [paperAlgoRunDemo, setPaperAlgoRunDemo] = useState<Record<string, boolean>>({});

  const [paperAlgoEntrypoint, setPaperAlgoEntrypoint] = useState<Record<string, string>>({});

  const [paperAlgoEntrypointSavedAt, setPaperAlgoEntrypointSavedAt] = useState<Record<string, string>>({});

  const [paperAlgoEntrypointSaving, setPaperAlgoEntrypointSaving] = useState<Record<string, boolean>>({});

  const [paperAlgoEntrypointError, setPaperAlgoEntrypointError] = useState<Record<string, string>>({});

  const updateMyPreferencesMutation = useMutation((updates: any) => apiClient.updateMyPreferences(updates), {
    onSuccess: () => {
      queryClient.invalidateQueries(['me-preferences']);
      toast.success('Preferences updated');
    },
    onError: (e: any) => {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to update preferences');
    },
  });

  const paperAlgoDefaultRunDemoCheck = (myPreferences as any)?.paper_algo_default_run_demo_check === true;

  const paperAlgoDefaultToggleTitle =
    unsafeExecBadge.status === 'ready'
      ? 'Set the default for new items in this session'
      : 'Server not ready for demo checks (see badge)';

  const normalizeEntrypoint = (raw: string): { ok: boolean; value: string; error?: string } => {
    let v = String(raw || '').trim();
    if (!v) return { ok: true, value: 'demo.py' };
    v = v.replace(/\\/g, '/');
    while (v.startsWith('./')) v = v.slice(2);
    if (v.startsWith('/') || v.startsWith('~') || v.includes(':')) return { ok: false, value: v, error: 'Absolute paths not allowed' };
    if (v.split('/').some((p) => p === '..')) return { ok: false, value: v, error: "'..' not allowed" };
    if (/\s/.test(v)) return { ok: false, value: v, error: 'Whitespace not allowed' };
    if (!v.endsWith('.py')) return { ok: false, value: v, error: 'Must end with .py' };
    if (!/^[A-Za-z0-9._/\\-]+$/.test(v)) return { ok: false, value: v, error: 'Invalid characters' };
    return { ok: true, value: v };
  };

  return (
    <div className="w-full flex flex-col min-h-0">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3 text-sm text-gray-600">
          <span className="font-medium text-gray-900">Research Inbox</span>
          <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded">Total: {inboxStats?.total ?? '-'}</span>
          <span className="bg-primary-100 text-primary-700 px-2 py-1 rounded">New: {inboxStats?.new ?? '-'}</span>
          <span className="bg-green-100 text-green-700 px-2 py-1 rounded">Accepted: {inboxStats?.accepted ?? '-'}</span>
          <span className="bg-red-100 text-red-700 px-2 py-1 rounded">Rejected: {inboxStats?.rejected ?? '-'}</span>
        </div>
        <div className="flex gap-2">
          <label className="flex items-center gap-2 text-xs text-gray-600 select-none" title={paperAlgoDefaultToggleTitle}>
            <input
              type="checkbox"
              className="h-3 w-3"
              checked={paperAlgoDefaultRunDemoCheck}
              disabled={updateMyPreferencesMutation.isLoading}
              onChange={(e) => updateMyPreferencesMutation.mutate({ paper_algo_default_run_demo_check: e.target.checked })}
            />
            <span>Default: Run demo check</span>
          </label>
          <Button variant="secondary" onClick={() => setShowInboxMonitorModal(true)}>
            <Activity className="w-4 h-4 mr-2" />
            Create Monitor
          </Button>
          <Button variant="secondary" onClick={() => setShowMonitorProfilesModal(true)}>
            <Settings className="w-4 h-4 mr-2" />
            Profiles
          </Button>
          <Button variant="ghost" onClick={() => refetchInbox()}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="flex gap-3 mb-4">
        <select
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={inboxStatusFilter}
          onChange={(e) => setInboxStatusFilter(e.target.value as any)}
        >
          <option value="">All Status</option>
          <option value="new">New</option>
          <option value="accepted">Accepted</option>
          <option value="rejected">Rejected</option>
        </select>
        <select
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={inboxTypeFilter}
          onChange={(e) => setInboxTypeFilter(e.target.value)}
        >
          <option value="">All Types</option>
          <option value="document">Document</option>
          <option value="arxiv">arXiv</option>
        </select>
        <select
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={inboxCustomerFilter}
          onChange={(e) => setInboxCustomerFilter(e.target.value)}
        >
          <option value="">All Customers</option>
          {healthCustomers.map((customer) => (
            <option key={customer} value={customer}>
              {customer}
            </option>
          ))}
        </select>
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              className="w-full border border-gray-300 rounded-lg pl-9 pr-3 py-2 text-sm"
              placeholder="Search inbox items…"
              value={inboxSearch}
              onChange={(e) => setInboxSearch(e.target.value)}
            />
        </div>
      </div>

      {inboxCustomerFilter ? (
        <div className="flex items-center gap-2 mb-4 text-xs">
          <span className="bg-amber-50 text-amber-800 border border-amber-200 px-2 py-1 rounded">
            Customer filter: {inboxCustomerFilter}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setInboxCustomerFilter('');
            }}
          >
            Clear customer filter
          </Button>
        </div>
      ) : null}

      {inboxJobFilter ? (
        <div className="flex items-center gap-2 mb-4 text-xs">
          <span className="bg-sky-50 text-sky-800 border border-sky-200 px-2 py-1 rounded">
            Monitor filter: {inboxJobFilter}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setInboxJobFilter('');
              const params = new URLSearchParams(location.search);
              params.delete('inbox_job');
              params.delete('inbox');
              navigate(`${location.pathname}${params.toString() ? `?${params.toString()}` : ''}`, { replace: true });
            }}
          >
            Clear monitor filter
          </Button>
        </div>
      ) : null}

      {inboxHealthDrilldown ? (
        <div className="flex items-center gap-2 mb-4 text-xs">
          <span className="bg-emerald-50 text-emerald-800 border border-emerald-200 px-2 py-1 rounded">
            Showing accepted follow-ups
            {inboxCustomerFilter ? ` for ${inboxCustomerFilter}` : ''}
            {inboxJobFilter ? ` · ${inboxJobFilter}` : ''}
            {` · ${formatInboxHealthDrilldownLabel(inboxHealthDrilldown)}`}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setInboxHealthDrilldown('');
            }}
          >
            Clear drilldown
          </Button>
        </div>
      ) : null}

      {inboxPolicyDrilldown ? (
        <div className="flex items-center gap-2 mb-4 text-xs">
          <span className="bg-violet-50 text-violet-800 border border-violet-200 px-2 py-1 rounded">
            Showing accepted signals
            {inboxJobFilter ? ` for ${inboxJobFilter}` : ''}
            {` · ${formatInboxPolicyDrilldownLabel(inboxPolicyDrilldown)}`}
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setInboxPolicyDrilldown('');
            }}
          >
            Clear drilldown
          </Button>
        </div>
      ) : null}

      {(() => {
        const items = visibleInboxItems;
        const selectedIds = selectedInboxItems.map((item) => String(item.id));
        const allSelected = items.length > 0 && selectedIds.length === items.length;
        if (items.length === 0) return null;
        return (
          <div className="flex items-center justify-between mb-3 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={(e) => {
                  const next: Record<string, boolean> = {};
                  if (e.target.checked) {
                    items.forEach((it) => (next[it.id] = true));
                  }
                  setSelectedInboxIds(next);
                }}
              />
              <span className="text-sm text-gray-700">
                Selected: {selectedIds.length}/{items.length}
              </span>
            </div>
            <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={selectedIds.length === 0 || createMutation.isLoading || createFromChainMutation.isLoading}
              onClick={() => {
                  const selectedItems = selectedInboxItems;
                  if (selectedItems.length === 0) return;

                  const goal = inboxResearchGoalDraft.trim();
                  if (!goal.trim()) return;

                  const docItems = selectedItems.filter((it) => it.item_type === 'document');
                  const paperItems = selectedItems.filter((it) => it.item_type === 'arxiv');

                  const top_documents = docItems.slice(0, 20).map((d) => ({
                    id: d.item_key,
                    title: d.title,
                    url: d.url,
                    score: null,
                    source: 'inbox',
                  }));
                  const top_papers = paperItems.slice(0, 20).map((p) => ({
                    id: p.item_key,
                    title: p.title,
                    url: p.url,
                    score: null,
                    source: 'inbox',
                  }));

                  const parent_findings = selectedItems.slice(0, 50).map((it) => ({
                    type: it.item_type === 'arxiv' ? 'paper' : 'document',
                    title: it.title,
                    id: it.item_key,
                    url: it.url,
                    snippet: it.summary,
                  }));

                  const customers = Array.from(new Set(selectedItems.map((it) => it.customer).filter(Boolean))) as string[];
                  const customerContextHint =
                    customers.length === 1 ? `Customer: ${customers[0]}` : customers.length > 1 ? `Customers: ${customers.join(', ')}` : '';

                  const chains = ((chainsData as any)?.chains || []) as any[];
                  const deepDiveChain =
                    chains.find((c: any) => c?.name === 'customer_research_scout_deep_dive_chain') || null;

                  if (deepDiveChain?.id) {
                    createFromChainMutation.mutate({
                      chain_definition_id: deepDiveChain.id,
                      name_prefix: `Inbox Research — ${new Date().toLocaleDateString()}`,
                      variables: { goal: goal.trim() },
                      config_overrides: {
                        customer_context: customerContextHint,
                        prefer_sources: ['documents', 'arxiv'],
                        max_documents: 12,
                        max_papers: 8,
                        persist_artifacts: false,
                        reading_list_name: 'Customer Research',
                        inherited_data: {
                          parent_results: {
                            summary: `Seeded from ${selectedItems.length} Research Inbox items.`,
                            research_bundle: {
                              top_documents,
                              top_papers,
                              insights: [],
                              next_steps: [],
                              artifacts: [],
                            },
                            inbox_items: selectedItems,
                          },
                          parent_findings,
                        },
                      },
                      start_immediately: true,
                    });
                  } else {
                    // Fallback: single research job
                    createMutation.mutate({
                      name: `Inbox Research — ${new Date().toLocaleDateString()}`,
                      job_type: 'research',
                      goal: goal.trim(),
                      config: {
                        customer_context: customerContextHint,
                        prefer_sources: ['documents', 'arxiv'],
                        max_documents: 12,
                        max_papers: 8,
                        persist_artifacts: false,
                        reading_list_name: 'Customer Research',
                        inherited_data: {
                          parent_results: {
                            summary: `Seeded from ${selectedItems.length} Research Inbox items.`,
                            research_bundle: {
                              top_documents,
                              top_papers,
                              insights: [],
                              next_steps: [],
                              artifacts: [],
                            },
                            inbox_items: selectedItems,
                          },
                          parent_findings,
                        },
                      },
                      start_immediately: true,
                    });
                  }
                  setActiveTab('jobs');
                }}
              >
                <Sparkles className="w-4 h-4 mr-1" />
                Research Selected
              </Button>
              <Button
                size="sm"
                variant="primary"
                disabled={!inboxBulkFollowUpState.enabled || bulkInboxFollowUpActionMutation.isLoading}
                onClick={() => {
                  if (!inboxBulkFollowUpState.enabled) return;
                  bulkInboxFollowUpActionMutation.mutate({
                    domain_research_profile_id: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.ownerId : undefined,
                    profile_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.opportunityIds : undefined,
                    portfolio_id: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.ownerId : undefined,
                    portfolio_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.opportunityIds : undefined,
                    action: 'approve_launch',
                    operator_note: inboxBulkFollowUpNote.trim() || undefined,
                  });
                }}
              >
                <ThumbsUp className="w-4 h-4 mr-1" />
                Approve Follow-ups
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={!inboxBulkFollowUpState.enabled || bulkInboxFollowUpActionMutation.isLoading}
                onClick={() => {
                  if (!inboxBulkFollowUpState.enabled) return;
                  bulkInboxFollowUpActionMutation.mutate({
                    domain_research_profile_id: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.ownerId : undefined,
                    profile_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'domain' ? inboxBulkFollowUpState.opportunityIds : undefined,
                    portfolio_id: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.ownerId : undefined,
                    portfolio_opportunity_ids: inboxBulkFollowUpState.ownerKind === 'fleet' ? inboxBulkFollowUpState.opportunityIds : undefined,
                    action: 'reject_launch',
                    operator_note: inboxBulkFollowUpNote.trim() || undefined,
                  });
                }}
              >
                <ThumbsDown className="w-4 h-4 mr-1" />
                Reject Follow-ups
              </Button>
              <Button
                size="sm"
                variant="secondary"
                disabled={!inboxBulkRelaunchState.enabled || bulkInboxRelaunchMutation.isLoading}
                onClick={() => {
                  if (!inboxBulkRelaunchState.enabled) return;
                  bulkInboxRelaunchMutation.mutate({
                    item_ids: inboxBulkRelaunchState.itemIds,
                    operator_note: inboxBulkFollowUpNote.trim() || undefined,
                  });
                }}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Relaunch Follow-ups
              </Button>
              <Button
                size="sm"
                variant="secondary"
                disabled={selectedIds.length === 0 || bulkUpdateInboxMutation.isLoading}
                onClick={() => bulkUpdateInboxMutation.mutate({ itemIds: selectedIds, data: { status: 'accepted' } })}
              >
                <ThumbsUp className="w-4 h-4 mr-1" />
                Accept Selected
              </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={selectedIds.length === 0 || bulkUpdateInboxMutation.isLoading}
              onClick={() => {
                  bulkUpdateInboxMutation.mutate({
                    itemIds: selectedIds,
                    data: { status: 'rejected', feedback: inboxBulkRejectReason.trim() || undefined },
                  });
                  setInboxBulkRejectReason('');
                }}
            >
              <ThumbsDown className="w-4 h-4 mr-1" />
              Reject Selected
            </Button>
          </div>
          <textarea
            className="mt-3 w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            rows={3}
            placeholder="Follow-up research goal"
            value={inboxResearchGoalDraft}
            onChange={(e) => setInboxResearchGoalDraft(e.target.value)}
          />
          <div className="mt-3 grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
            <input
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="Bulk follow-up note (optional)"
              value={inboxBulkFollowUpNote}
              onChange={(e) => setInboxBulkFollowUpNote(e.target.value)}
            />
            <div className="text-[11px] text-gray-500 self-center">
              {inboxBulkFollowUpState.enabled
                ? 'Applies to selected pending follow-up approvals'
                : inboxBulkRelaunchState.enabled
                  ? 'Applies to selected failed or cancelled follow-ups'
                  : inboxBulkFollowUpState.disabledReason || inboxBulkRelaunchState.disabledReason}
            </div>
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
            <input
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="Bulk reject reason (optional)"
              value={inboxBulkRejectReason}
              onChange={(e) => setInboxBulkRejectReason(e.target.value)}
            />
            <div className="text-[11px] text-gray-500 self-center">
              Applies when rejecting selected inbox items
            </div>
          </div>
        </div>
        );
      })()}

      {inboxLoading ? (
        <div className="flex justify-center items-center flex-1">
          <LoadingSpinner />
        </div>
      ) : visibleInboxItems.length === 0 ? (
        <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
          <Inbox className="w-12 h-12 mb-3 text-gray-400" />
          <p className="text-lg font-medium">Inbox is empty</p>
          <p className="text-sm">Create a monitor or run customer research to discover items</p>
        </div>
      ) : (
        <div className="space-y-3 overflow-y-auto flex-1 pr-1">
          {visibleInboxItems.map((item: ResearchInboxItem) => (
            <div
              key={item.id}
              className={`bg-white border rounded-lg p-4 ${
                deepLinkedInboxId && String(deepLinkedInboxId) === String(item.id)
                  ? 'border-emerald-400 ring-1 ring-emerald-200'
                  : 'border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={!!selectedInboxIds[item.id]}
                      onChange={(e) => setSelectedInboxIds((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                    />
                    <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                      {item.item_type}
                    </span>
                    <span
                      className={`text-xs px-2 py-1 rounded ${
                        item.status === 'new'
                          ? 'bg-primary-100 text-primary-700'
                          : item.status === 'accepted'
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {item.status}
                    </span>
                    {item.customer ? (
                      <span className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded">
                        {item.customer}
                      </span>
                    ) : null}
                    {item.follow_up_launch_status ? (
                      <span
                        className={`text-xs px-2 py-1 rounded ${
                          item.follow_up_launch_status === 'launched'
                            ? 'bg-emerald-100 text-emerald-700'
                            : item.follow_up_launch_status === 'pending_approval'
                              ? 'bg-amber-100 text-amber-800'
                              : item.follow_up_launch_status === 'failed'
                                ? 'bg-rose-100 text-rose-700'
                                : 'bg-gray-200 text-gray-700'
                        }`}
                      >
                        {item.follow_up_launch_status.replace(/_/g, ' ')}
                      </span>
                    ) : null}
                    {item.follow_up_outcome_status ? (
                      <span
                        className={`text-xs px-2 py-1 rounded ${
                          item.follow_up_outcome_status === 'completed'
                            ? 'bg-blue-100 text-blue-700'
                            : item.follow_up_outcome_status === 'failed'
                              ? 'bg-rose-100 text-rose-700'
                              : 'bg-gray-200 text-gray-700'
                        }`}
                      >
                        outcome: {item.follow_up_outcome_status.replace(/_/g, ' ')}
                      </span>
                    ) : null}
                  </div>
                  <h3 className="section-heading mt-2 truncate">{item.title}</h3>
                  {item.summary ? (
                    <p className="text-sm text-gray-600 mt-1 line-clamp-2">{item.summary}</p>
                  ) : null}
                  {(item.metadata as any)?.query ? (
                    <p className="text-xs text-gray-500 mt-2">
                      Why: matched query “{String((item.metadata as any).query).slice(0, 140)}”
                      {(item.metadata as any)?.bias?.source ? (
                        <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-0.5 rounded">
                          bias: {String((item.metadata as any).bias.source)}
                        </span>
                      ) : null}
                    </p>
                  ) : null}
                  <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-x-4 gap-y-1">
                    <span>Discovered: {new Date(item.discovered_at).toLocaleString()}</span>
                    {item.published_at ? (
                      <span>Published: {new Date(item.published_at).toLocaleDateString()}</span>
                    ) : null}
                    {item.follow_up_policy_mode ? (
                      <span>Policy: {item.follow_up_policy_mode.replace(/_/g, ' ')}</span>
                    ) : null}
                    {typeof (item.metadata as any)?.discovery_score === 'number' ? (
                      <span>Discovery score: {Number((item.metadata as any).discovery_score)}</span>
                    ) : null}
                    {item.follow_up_launched_at ? (
                      <span>Launched: {new Date(item.follow_up_launched_at).toLocaleString()}</span>
                    ) : null}
                    {item.follow_up_outcome_recorded_at ? (
                      <span>Outcome: {new Date(item.follow_up_outcome_recorded_at).toLocaleString()}</span>
                    ) : null}
                    {item.follow_up_operator_acted_at ? (
                      <span>Operator acted: {new Date(item.follow_up_operator_acted_at).toLocaleString()}</span>
                    ) : null}
                  </div>
                  {item.follow_up_block_reason ? (
                    <p className="text-xs text-gray-500 mt-2">
                      Follow-up: {item.follow_up_block_reason}
                    </p>
                  ) : null}
                  {item.follow_up_budget_decision || item.follow_up_budget_throttle_state ? (
                    <p className="text-xs text-amber-700 mt-2">
                      Budget: {String(item.follow_up_budget_decision || item.follow_up_budget_throttle_state || '').replace(/_/g, ' ')}
                      {item.follow_up_budget_reason ? ` — ${item.follow_up_budget_reason}` : ''}
                    </p>
                  ) : null}
                  {item.follow_up_customer_budget_decision || item.follow_up_customer_budget_throttle_state ? (
                    <p className="text-xs text-rose-700 mt-2">
                      Customer budget: {String(item.follow_up_customer_budget_decision || item.follow_up_customer_budget_throttle_state || '').replace(/_/g, ' ')}
                      {item.follow_up_customer_budget_reason ? ` — ${item.follow_up_customer_budget_reason}` : ''}
                    </p>
                  ) : null}
                  {Array.isArray((item.metadata as any)?.discovery_reasons) && (item.metadata as any).discovery_reasons.length > 0 ? (
                    <p className="text-xs text-gray-500 mt-2">
                      Discovery why: {((item.metadata as any).discovery_reasons as string[]).slice(0, 3).join(', ')}
                    </p>
                  ) : null}
                  {item.follow_up_operator_decision ? (
                    <p className="text-xs text-gray-500 mt-2">
                      Operator: {item.follow_up_operator_decision.replace(/_/g, ' ')}
                      {item.follow_up_operator_note ? ` — ${item.follow_up_operator_note}` : ''}
                    </p>
                  ) : null}
                  {item.follow_up_outcome_summary ? (
                    <p className="text-xs text-gray-500 mt-2">
                      Outcome summary: {item.follow_up_outcome_summary}
                    </p>
                  ) : null}
                  {item.status === 'accepted'
                  && item.item_type === 'follow_up_recommendation'
                  && String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval' ? (
                    <input
                      aria-label={`Inbox follow-up note for ${String(item.title || item.id)}`}
                      className="mt-3 border border-gray-300 rounded-lg px-3 py-2 text-sm w-full"
                      placeholder="Follow-up note (optional)"
                      value={followUpReviewNoteDrafts[`inbox:${String(item.id)}`] || ''}
                      disabled={followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === `inbox:${String(item.id)}`}
                      onChange={(e) => setFollowUpReviewNoteDrafts((prev) => ({ ...prev, [`inbox:${String(item.id)}`]: e.target.value }))}
                    />
                  ) : null}
                  {item.origin_source_kind && item.origin_source_id && item.origin_opportunity_id ? (
                    <p className="text-xs text-gray-500 mt-2">
                      Target: {String(item.origin_source_kind).trim().toLowerCase() === 'profile' ? 'Domain profile' : 'Research fleet'}
                    </p>
                  ) : null}
                </div>
                <div className="flex flex-col gap-2 shrink-0">
                  {item.origin_source_kind && item.origin_source_id && item.origin_opportunity_id ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        // Both surfaces are destinations of their own now. Link
                        // straight there rather than to the tab URL that merely
                        // redirects: a link that needs a redirect is a stale one.
                        const sourceKind = String(item.origin_source_kind || '').trim().toLowerCase();
                        const isProfile = sourceKind === 'profile';
                        const params = new URLSearchParams({
                          [isProfile ? 'profileId' : 'fleetId']: String(item.origin_source_id || ''),
                          opportunityId: String(item.origin_opportunity_id || ''),
                        });
                        navigate(
                          `${isProfile ? '/settings/domain-profiles' : '/research/fleet'}?${params.toString()}`
                        );
                      }}
                    >
                      Open Target
                    </Button>
                  ) : null}
                  {item.follow_up_job_id ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setActiveTab('jobs');
                        navigate(buildAutonomousAgentsUrl(item.follow_up_job_id), { replace: true });
                      }}
                    >
                      Open Follow-up
                    </Button>
                  ) : null}
                  {item.follow_up_launch_status === 'launched' && ['failed', 'cancelled'].includes(String(item.follow_up_outcome_status || '').trim().toLowerCase()) ? (
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={relaunchInboxFollowUpMutation.isLoading}
                      onClick={() => relaunchInboxFollowUpMutation.mutate({ itemId: item.id })}
                    >
                      <RotateCcw className="w-4 h-4 mr-1" />
                      Relaunch Follow-up
                    </Button>
                  ) : null}
                  {item.status === 'accepted'
                  && item.item_type === 'follow_up_recommendation'
                  && String(item.follow_up_launch_status || '').trim().toLowerCase() === 'pending_approval' ? (
                    <>
                      <Button
                        size="sm"
                        variant="primary"
                        disabled={followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === `inbox:${String(item.id)}`}
                        onClick={() => followUpQueueActionMutation.mutate({
                          inbox_item_id: String(item.id),
                          action: 'approve_launch',
                          operator_note: followUpReviewNoteDrafts[`inbox:${String(item.id)}`]?.trim() || undefined,
                          navigateOnLaunch: false,
                          reviewRowKey: `inbox:${String(item.id)}`,
                        })}
                      >
                        <ThumbsUp className="w-4 h-4 mr-1" />
                        Approve Follow-up
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === `inbox:${String(item.id)}`}
                        onClick={() => followUpQueueActionMutation.mutate({
                          inbox_item_id: String(item.id),
                          action: 'reject_launch',
                          operator_note: followUpReviewNoteDrafts[`inbox:${String(item.id)}`]?.trim() || undefined,
                          navigateOnLaunch: false,
                          reviewRowKey: `inbox:${String(item.id)}`,
                        })}
                      >
                        <ThumbsDown className="w-4 h-4 mr-1" />
                        Reject Follow-up
                      </Button>
                    </>
                  ) : null}
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={item.status === 'accepted' || updateInboxItemMutation.isLoading}
                    onClick={() => updateInboxItemMutation.mutate({ itemId: item.id, data: { status: 'accepted' } })}
                  >
                    <ThumbsUp className="w-4 h-4 mr-1" />
                    Accept
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={item.status === 'rejected' || updateInboxItemMutation.isLoading}
                    onClick={() => {
                      updateInboxItemMutation.mutate({
                        itemId: item.id,
                        data: { status: 'rejected', feedback: inboxRejectReasonDrafts[item.id]?.trim() || undefined },
                      });
                    }}
                  >
                    <ThumbsDown className="w-4 h-4 mr-1" />
                    Reject
                  </Button>
                  <div className="grid gap-2">
                    <input
                      className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      placeholder="Reject reason (optional)"
                      value={inboxRejectReasonDrafts[item.id] || ''}
                      onChange={(e) => setInboxRejectReasonDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                    />
                    <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
                      <input
                        className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                        placeholder="Mute token"
                        value={inboxMuteTokenDrafts[item.id] ?? ((item.title || '').split(/[^a-zA-Z0-9_-]+/).find((t) => t && t.length >= 4) || '')}
                        onChange={(e) => setInboxMuteTokenDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                      />
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={upsertMonitorProfileMutation.isLoading}
                        onClick={() => {
                          const token = String(inboxMuteTokenDrafts[item.id] || '').trim().toLowerCase();
                          if (!token) {
                            toast.error('Enter a mute token first');
                            return;
                          }
                          upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_tokens: [token], merge_lists: true });
                        }}
                      >
                        Mute token
                      </Button>
                    </div>
                    <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
                      <input
                        className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                        placeholder="Mute phrase"
                        value={inboxMutePhraseDrafts[item.id] ?? (item.title || '')}
                        onChange={(e) => setInboxMutePhraseDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                      />
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={upsertMonitorProfileMutation.isLoading}
                        onClick={() => {
                          const phrase = String(inboxMutePhraseDrafts[item.id] || '').trim();
                          if (!phrase) {
                            toast.error('Enter a mute phrase first');
                            return;
                          }
                          upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_patterns: [phrase], merge_lists: true });
                        }}
                      >
                        Mute phrase
                      </Button>
                    </div>
                  </div>
                  {(item.metadata as any)?.query ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={upsertMonitorProfileMutation.isLoading}
                      onClick={() => {
                        const q = String((item.metadata as any).query || '').trim();
                        if (!q) return;
                        upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_patterns: [q], merge_lists: true });
                      }}
                    >
                      Mute query
                    </Button>
                  ) : null}
                      {item.item_type === 'arxiv' ? (
                        <>
                      {Array.isArray((item.metadata as any)?.repos) && (item.metadata as any).repos.length > 0 ? (
                        <>
                          <div className="text-xs text-gray-500 mt-1">Repos</div>
                          {((item.metadata as any).repos as any[]).slice(0, 2).map((r: any, idx: number) => (
                            <Button
                              key={idx}
                              size="sm"
                              variant="secondary"
                              disabled={ingestRepoMutation.isLoading || String(r?.provider) !== 'github'}
                              title={String(r?.provider) === 'github' ? 'Ingest this repo' : 'GitLab ingestion requires a token (use Documents → Repos)'}
                              onClick={() => ingestRepoMutation.mutate({ provider: 'github', repo: String(r.repo) })}
                            >
                              Ingest {String(r?.provider || 'repo')}
                            </Button>
                          ))}
                        </>
                      ) : (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={extractReposMutation.isLoading}
                          onClick={() => extractReposMutation.mutate(item.id)}
                        >
                          Find repos
                        </Button>
                      )}
                      {Array.isArray((item.metadata as any)?.repos) && (item.metadata as any).repos.filter((r: any) => String(r?.provider || '').toLowerCase() === 'github').length > 1 ? (
                        <select
                          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          value={
                            String(
                              paperRepoSelectionDrafts[item.id] ??
                                String(((item.metadata as any).repos as any[]).find((r: any) => String(r?.provider || '').toLowerCase() === 'github')?.repo || '')
                            )
                          }
                          onChange={(e) => setPaperRepoSelectionDrafts((current) => ({ ...current, [item.id]: e.target.value }))}
                        >
                          {((item.metadata as any).repos as any[])
                            .filter((r: any) => String(r?.provider || '').toLowerCase() === 'github')
                            .slice(0, 12)
                            .map((r: any) => (
                              <option key={String(r.repo)} value={String(r.repo)}>
                                {String(r.repo)}
                              </option>
                            ))}
                        </select>
                      ) : null}
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={createFromChainMutation.isLoading}
                        onClick={() =>
                          runPaperRepoCodeAgent(
                            item,
                            paperRepoSelectionDrafts[item.id] ||
                              String(
                                Array.isArray((item.metadata as any)?.repos)
                                  ? ((item.metadata as any).repos as any[]).find((r: any) => String(r?.provider || '').toLowerCase() === 'github')?.repo || ''
                                  : ''
                              )
                          )
                        }
                      >
                        Code Agent on Repo
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={createFromChainMutation.isLoading}
                        onClick={() => {
                          const persistedEp = String((item.metadata as any)?.paper_algo_entrypoint || '').trim();
                          const ep = (paperAlgoEntrypoint[item.id] ?? persistedEp ?? 'demo.py') || 'demo.py';
                          runPaperAlgorithmProject(item, paperAlgoRunDemo[item.id] ?? paperAlgoDefaultRunDemoCheck, ep);
                        }}
                      >
                        <span className="inline-flex items-center gap-2">
                          <span>Implement Algorithm</span>
                          <span className="inline-flex items-center gap-1" title={unsafeExecBadge.title}>
                            <span className={`inline-block w-2 h-2 rounded-full ${unsafeExecBadge.color}`} />
                            <span className="text-[10px] text-gray-600">{unsafeExecBadge.label}</span>
                          </span>
                        </span>
                      </Button>
                      <label
                        className="flex items-center gap-1 text-xs text-gray-600 select-none"
                        title={
                          unsafeExecBadge.status === 'ready'
                            ? 'Run a sandboxed demo.py check after generating the project'
                            : unsafeExecBadge.title
                        }
                      >
                        {(() => {
                          const persisted = (item.metadata as any)?.paper_algo_run_demo_check;
                          const checked =
                            typeof persisted === 'boolean'
                              ? persisted
                              : paperAlgoRunDemo[item.id] ?? paperAlgoDefaultRunDemoCheck;
                          return (
                        <input
                          type="checkbox"
                          className="h-3 w-3"
                          checked={checked}
                          disabled={unsafeExecBadge.status !== 'ready' || updateInboxItemMutation.isLoading}
                          onChange={(e) => {
                            const v = e.target.checked;
                            setPaperAlgoRunDemo((prev) => ({ ...prev, [item.id]: v }));
                            updateInboxItemMutation.mutate({
                              itemId: item.id,
                              data: { metadata_patch: { paper_algo_run_demo_check: v } },
                            });
                          }}
                        />
                          );
                        })()}
                        <span>Run demo check</span>
                      </label>
                      <input
                        className={`border rounded px-2 py-1 text-xs w-36 ${
                          paperAlgoEntrypointError[item.id] ? 'border-red-400' : 'border-gray-200'
                        }`}
                        placeholder="demo.py"
                        value={
                          paperAlgoEntrypoint[item.id] ??
                          (String((item.metadata as any)?.paper_algo_entrypoint || '').trim() || 'demo.py')
                        }
                        onChange={(e) => {
                          const raw = e.target.value;
                          setPaperAlgoEntrypoint((prev) => ({ ...prev, [item.id]: raw }));
                          const check = normalizeEntrypoint(raw);
                          setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: check.ok ? '' : String(check.error || 'Invalid') }));
                        }}
                        onBlur={async () => {
                          const raw =
                            paperAlgoEntrypoint[item.id] ??
                            (String((item.metadata as any)?.paper_algo_entrypoint || '').trim() || 'demo.py');
                          const check = normalizeEntrypoint(raw);
                          setPaperAlgoEntrypoint((prev) => ({ ...prev, [item.id]: check.value }));
                          setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: check.ok ? '' : String(check.error || 'Invalid') }));
                          if (!check.ok) {
                            toast.error(`Invalid entrypoint: ${check.error || 'Invalid'}`);
                            return;
                          }
                          setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: true }));
                          try {
                            await apiClient.updateResearchInboxItem(item.id, {
                              metadata_patch: { paper_algo_entrypoint: check.value },
                            } as any);
                            queryClient.invalidateQueries(['research-inbox']);
                            setPaperAlgoEntrypointSavedAt((prev) => ({ ...prev, [item.id]: new Date().toISOString() }));
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to save entrypoint');
                          } finally {
                            setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: false }));
                          }
                        }}
                        title={
                          paperAlgoEntrypointError[item.id]
                            ? `Entrypoint invalid: ${paperAlgoEntrypointError[item.id]}`
                            : 'Demo entrypoint path (persisted per paper)'
                        }
                      />
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={paperAlgoEntrypointSaving[item.id] || updateInboxItemMutation.isLoading}
                        title="Reset entrypoint override to default (demo.py)"
                        onClick={async () => {
                          setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: true }));
                          try {
                            await apiClient.updateResearchInboxItem(item.id, {
                              metadata_patch: { paper_algo_entrypoint: null },
                            } as any);
                            setPaperAlgoEntrypoint((prev) => {
                              const next = { ...prev };
                              delete next[item.id];
                              return next;
                            });
                            setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: '' }));
                            setPaperAlgoEntrypointSavedAt((prev) => ({ ...prev, [item.id]: new Date().toISOString() }));
                            queryClient.invalidateQueries(['research-inbox']);
                            toast.success('Entrypoint reset to default');
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to reset entrypoint');
                          } finally {
                            setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: false }));
                          }
                        }}
                      >
                        Reset
                      </Button>
                      <span className="text-[10px] text-gray-500 min-w-[42px]">
                        {paperAlgoEntrypointSaving[item.id]
                          ? 'saving…'
                          : paperAlgoEntrypointSavedAt[item.id]
                            ? 'saved'
                            : ''}
                      </span>
                    </>
                  ) : null}
                  {item.item_type === 'document' ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => navigate(`/search?q=${encodeURIComponent(item.title || item.item_key)}`)}
                    >
                      <Search className="w-4 h-4 mr-1" />
                      Search
                    </Button>
                  ) : null}
                  {item.url ? (
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1 justify-center"
                    >
                      <Link2 className="w-4 h-4" />
                      Open
                    </a>
                  ) : null}
                </div>
              </div>
              {item.feedback ? (
                <div className="mt-3 text-xs text-gray-600 bg-gray-50 border border-gray-100 rounded p-2">
                  Feedback: {item.feedback}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ResearchInboxTab;
