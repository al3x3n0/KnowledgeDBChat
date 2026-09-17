/**
 * Research Inbox -- opportunities the monitors surfaced, and what to do with them.
 *
 * This was a tab on the Runs page. It is triage of papers and documents, not a
 * view of an agent run, so it lives in the Library beside Papers, Reading Lists
 * and Research Notes.
 *
 * The filters live in the URL, not in component state that someone else writes.
 * On Runs they were page state that the health tab reached into through eleven
 * setter props -- and the three hand-rolled drilldowns there set that state
 * WITHOUT updating the URL, so "show me this customer's inbox" produced a view
 * you could not link to or reload. Here the URL is the only source of truth:
 * every drilldown is a link, and the back button works because there is nothing
 * else to get out of step with.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from 'react-query';

import ResearchInboxTab from '../components/agent/tabs/ResearchInboxTab';
import InboxMonitorModal from '../components/agent/InboxMonitorModal';
import MonitorProfilesModal from '../components/agent/MonitorProfilesModal';
import {
  useCreateAgentJobMutation,
  useCreateJobFromChainMutation,
  useFollowUpQueueActionMutation,
  useUpsertMonitorProfileMutation,
} from '../components/agent/agentJobMutations';
import {
  normalizeInboxHealthDrilldown,
  normalizeInboxPolicyDrilldown,
} from '../components/agent/drilldowns';
import type { InboxUrlParams } from '../components/agent/drilldowns';
import { apiClient } from '../services/api';
import type {
  ResearchInboxItem,
  ResearchInboxItemStatus,
  ResearchMonitorAnalyticsResponse,
} from '../types';

const ResearchInboxPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const inboxStatusFilter = (params.get('inbox_status') || '') as ResearchInboxItemStatus | '';
  const inboxTypeFilter = params.get('inbox_type') || '';
  const inboxSearch = params.get('inbox_q') || '';
  const inboxJobFilter = params.get('inbox_job') || '';
  const inboxCustomerFilter = params.get('inbox_customer') || '';
  const inboxHealthDrilldown = normalizeInboxHealthDrilldown(params.get('inbox_health_drilldown'));
  const inboxPolicyDrilldown = normalizeInboxPolicyDrilldown(params.get('inbox_policy_drilldown'));

  /** A filter change is a navigation, which is what makes every view linkable. */
  const setFilter = useCallback((key: InboxUrlParams, value: string) => {
    const next = new URLSearchParams(location.search);
    const normalized = String(value ?? '').trim();
    if (normalized) next.set(key, normalized); else next.delete(key);
    navigate({ pathname: '/research/inbox', search: next.toString() }, { replace: true });
  }, [location.search, navigate]);

  const setInboxStatusFilter = useCallback((v: any) => setFilter('inbox_status', typeof v === 'function' ? v(inboxStatusFilter) : v), [setFilter, inboxStatusFilter]);
  const setInboxTypeFilter = useCallback((v: any) => setFilter('inbox_type', typeof v === 'function' ? v(inboxTypeFilter) : v), [setFilter, inboxTypeFilter]);
  const setInboxSearch = useCallback((v: any) => setFilter('inbox_q', typeof v === 'function' ? v(inboxSearch) : v), [setFilter, inboxSearch]);
  const setInboxJobFilter = useCallback((v: any) => setFilter('inbox_job', typeof v === 'function' ? v(inboxJobFilter) : v), [setFilter, inboxJobFilter]);
  const setInboxCustomerFilter = useCallback((v: any) => setFilter('inbox_customer', typeof v === 'function' ? v(inboxCustomerFilter) : v), [setFilter, inboxCustomerFilter]);
  const setInboxHealthDrilldown = useCallback((v: any) => setFilter('inbox_health_drilldown', typeof v === 'function' ? v(inboxHealthDrilldown) : v), [setFilter, inboxHealthDrilldown]);
  const setInboxPolicyDrilldown = useCallback((v: any) => setFilter('inbox_policy_drilldown', typeof v === 'function' ? v(inboxPolicyDrilldown) : v), [setFilter, inboxPolicyDrilldown]);

  // --- local editing state: drafts and selections, which are not worth a URL
  const [selectedInboxIds, setSelectedInboxIds] = useState<Record<string, boolean>>({});
  const [inboxRejectReasonDrafts, setInboxRejectReasonDrafts] = useState<Record<string, string>>({});
  const [inboxBulkRejectReason, setInboxBulkRejectReason] = useState<string>('');
  const [inboxBulkFollowUpNote, setInboxBulkFollowUpNote] = useState<string>('');
  const [followUpReviewNoteDrafts, setFollowUpReviewNoteDrafts] = useState<Record<string, string>>({});
  const [activeFollowUpReviewKey, setActiveFollowUpReviewKey] = useState<string>('');
  const [paperRepoSelectionDrafts, setPaperRepoSelectionDrafts] = useState<Record<string, string>>({});
  const [showInboxMonitorModal, setShowInboxMonitorModal] = useState(false);
  const [showMonitorProfilesModal, setShowMonitorProfilesModal] = useState(false);

  const { data: chainsData } = useQuery(
    ['agent-job-chains'],
    () => apiClient.listChainDefinitions()
  );

  const { data: monitorAnalyticsData } = useQuery(
    ['research-monitor-analytics'],
    () => apiClient.getResearchMonitorAnalytics(),
    {
      // Health *and* inbox. `healthCustomers` is derived from this response and
      // fills the inbox's customer filter, so gating it on the health tab left
      // that dropdown empty unless you had opened Autonomy Health earlier in
      // the same session -- a filter offering nothing, depending on where you
      // had been. On this page the condition is simply being here.
      staleTime: 30000,
    }
  );

  const { data: inboxStats } = useQuery(
    ['research-inbox-stats'],
    () => apiClient.getResearchInboxStats(),
    {
      refetchInterval: 20000,
    }
  );

  const { data: inboxData, isLoading: inboxLoading, refetch: refetchInbox } = useQuery(
    ['research-inbox', inboxStatusFilter, inboxTypeFilter, inboxCustomerFilter, inboxSearch, inboxJobFilter],
    () =>
      apiClient.listResearchInboxItems({
        status: inboxStatusFilter || undefined,
        item_type: inboxTypeFilter || undefined,
        customer: inboxCustomerFilter || undefined,
        job_id: inboxJobFilter || undefined,
        q: inboxSearch.trim() || undefined,
        limit: 100,
        offset: 0,
      }),
    {
      refetchInterval: 15000,
    }
  );

  const healthCustomers = useMemo(
    () =>
      Array.from(
        new Set(
          ((monitorAnalyticsData as ResearchMonitorAnalyticsResponse | undefined)?.monitors || [])
            .map((monitor) => String(monitor.customer || '').trim())
            .filter(Boolean)
        )
      ).sort((a, b) => a.localeCompare(b)),
    [monitorAnalyticsData]
  );

  const visibleInboxItems = useMemo(
    () => ((inboxData?.items || []) as ResearchInboxItem[]).filter((item) => {
      if (!inboxHealthDrilldown) return true;
      if (String(item.status || '').trim().toLowerCase() !== 'accepted') return false;
      if (String(item.item_type || '').trim() !== 'follow_up_recommendation') return false;
      const outcomeStatus = String(item.follow_up_outcome_status || '').trim().toLowerCase();
      const operatorDecision = String(item.follow_up_operator_decision || '').trim().toLowerCase();
      if (inboxHealthDrilldown === 'completed_follow_up') return outcomeStatus === 'completed';
      if (inboxHealthDrilldown === 'failed_follow_up') return outcomeStatus === 'failed';
      if (inboxHealthDrilldown === 'cancelled_follow_up') return outcomeStatus === 'cancelled';
      if (inboxHealthDrilldown === 'suppressed_relaunch') return operatorDecision === 'rejected';
      return true;
    }),
    [inboxData, inboxHealthDrilldown]
  );

  const selectedInboxItems = useMemo(
    () => visibleInboxItems.filter((item) => selectedInboxIds[item.id]),
    [selectedInboxIds, visibleInboxItems]
  );

  const { data: myPreferences } = useQuery(['me-preferences'], () => apiClient.getMyPreferences(), {
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });

  const { data: unsafeExecAvailability } = useQuery(
    ['unsafe-exec-availability'],
    () => apiClient.getUnsafeExecAvailability(),
    { staleTime: 30_000, refetchOnWindowFocus: false }
  );

  const unsafeExecBadge = useMemo(() => {
    const avail: any = unsafeExecAvailability as any;
    const enabled = !!avail?.enabled;
    const backend = String(avail?.backend || 'subprocess');
    const dockerOk = backend !== 'docker' || (avail?.docker?.available === true && avail?.docker?.image_present === true);
    const status: 'ready' | 'blocked' | 'off' = enabled && dockerOk ? 'ready' : enabled ? 'blocked' : 'off';
    const label =
      status === 'ready'
        ? 'demo-check ready'
        : status === 'blocked'
          ? 'demo-check not ready'
          : 'demo-check off';
    const title =
      status === 'ready'
        ? `Behavioral demo check available (backend: ${backend})`
        : status === 'blocked'
          ? `Behavioral demo check enabled but not ready (backend: ${backend})`
          : 'Behavioral demo check disabled on server';
    const color =
      status === 'ready' ? 'bg-green-500' : status === 'blocked' ? 'bg-amber-500' : 'bg-gray-400';
    return { status, label, title, color };
  }, [unsafeExecAvailability]);

  const { data: monitorProfiles, isLoading: monitorProfilesLoading, refetch: refetchMonitorProfiles } = useQuery(
    ['research-monitor-profiles'],
    () => apiClient.listResearchMonitorProfiles(),
    {
      enabled: showMonitorProfilesModal,
      staleTime: 30000,
    }
  );

  // A selection must not outlive the rows it pointed at.
  useEffect(() => {
    const visibleIds = new Set(visibleInboxItems.map((item) => String(item.id)));
    setSelectedInboxIds((prev) => {
      const nextEntries = Object.entries(prev).filter(([id, enabled]) => enabled && visibleIds.has(id));
      if (nextEntries.length === Object.keys(prev).length) return prev;
      return Object.fromEntries(nextEntries);
    });
  }, [visibleInboxItems]);

  const createInboxMonitorMutation = useCreateAgentJobMutation({
    successMessage: 'Monitor created',
    onCreated: () => setShowInboxMonitorModal(false),
  });

  /**
   * Rows link back to Runs -- "Open Target" opens the domain or fleet
   * opportunity an inbox item came from, which lives there. It takes the same
   * (jobId, extras) shape the Runs page uses, and the extras carry the tab and
   * the opportunity id. Building it from an empty query rather than from this
   * page's own search is deliberate: the inbox filters are not Runs parameters.
   */
  const buildAutonomousAgentsUrl = useCallback(
    (jobId?: string, extras?: Record<string, string | null | undefined>) => {
      const next = new URLSearchParams();
      if (jobId && String(jobId).trim()) next.set('job', String(jobId).trim());
      Object.entries(extras || {}).forEach(([key, value]) => {
        const text = String(value ?? '').trim();
        if (text) next.set(key, text);
      });
      const query = next.toString();
      return query ? `/autonomous-agents?${query}` : '/autonomous-agents';
    },
    []
  );

  const createMutation = useCreateAgentJobMutation({
    onCreated: (job) => navigate(`/autonomous-agents?job=${encodeURIComponent(String(job.id))}`),
  });
  const createFromChainMutation = useCreateJobFromChainMutation({
    onCreated: (job) => navigate(`/autonomous-agents?job=${encodeURIComponent(String(job.id))}`),
  });
  const upsertMonitorProfileMutation = useUpsertMonitorProfileMutation();
  const followUpQueueActionMutation = useFollowUpQueueActionMutation({
    setActiveFollowUpReviewKey,
    setFollowUpReviewNoteDrafts,
    // A launched follow-up is a job, and jobs live on Runs.
    onFollowUpLaunched: (followUpJobId) => {
      navigate(`/autonomous-agents?job=${encodeURIComponent(followUpJobId)}`);
      return true;
    },
  });

  return (
    <>
      <ResearchInboxTab
        activeFollowUpReviewKey={activeFollowUpReviewKey}
        buildAutonomousAgentsUrl={buildAutonomousAgentsUrl}
        chainsData={chainsData}
        createFromChainMutation={createFromChainMutation}
        createMutation={createMutation}
        followUpQueueActionMutation={followUpQueueActionMutation}
        followUpReviewNoteDrafts={followUpReviewNoteDrafts}
        healthCustomers={healthCustomers}
        inboxBulkFollowUpNote={inboxBulkFollowUpNote}
        inboxBulkRejectReason={inboxBulkRejectReason}
        inboxCustomerFilter={inboxCustomerFilter}
        inboxHealthDrilldown={inboxHealthDrilldown}
        inboxJobFilter={inboxJobFilter}
        inboxLoading={inboxLoading}
        inboxPolicyDrilldown={inboxPolicyDrilldown}
        inboxRejectReasonDrafts={inboxRejectReasonDrafts}
        inboxSearch={inboxSearch}
        inboxStats={inboxStats}
        inboxStatusFilter={inboxStatusFilter}
        inboxTypeFilter={inboxTypeFilter}
        location={location}
        myPreferences={myPreferences}
        navigate={navigate}
        paperRepoSelectionDrafts={paperRepoSelectionDrafts}
        queryClient={queryClient}
        refetchInbox={refetchInbox}
        selectedInboxIds={selectedInboxIds}
        selectedInboxItems={selectedInboxItems}
        setActiveTab={() => { /* there are no tabs here; drilldowns are links */ }}
        setFollowUpReviewNoteDrafts={setFollowUpReviewNoteDrafts}
        setInboxBulkFollowUpNote={setInboxBulkFollowUpNote}
        setInboxBulkRejectReason={setInboxBulkRejectReason}
        setInboxCustomerFilter={setInboxCustomerFilter}
        setInboxHealthDrilldown={setInboxHealthDrilldown}
        setInboxJobFilter={setInboxJobFilter}
        setInboxPolicyDrilldown={setInboxPolicyDrilldown}
        setInboxRejectReasonDrafts={setInboxRejectReasonDrafts}
        setInboxSearch={setInboxSearch}
        setInboxStatusFilter={setInboxStatusFilter}
        setInboxTypeFilter={setInboxTypeFilter}
        setPaperRepoSelectionDrafts={setPaperRepoSelectionDrafts}
        setSelectedInboxIds={setSelectedInboxIds}
        setShowInboxMonitorModal={setShowInboxMonitorModal}
        setShowMonitorProfilesModal={setShowMonitorProfilesModal}
        unsafeExecBadge={unsafeExecBadge}
        upsertMonitorProfileMutation={upsertMonitorProfileMutation}
        visibleInboxItems={visibleInboxItems}
      />
      {showInboxMonitorModal && (
        <InboxMonitorModal
          onClose={() => setShowInboxMonitorModal(false)}
          createInboxMonitorMutation={createInboxMonitorMutation}
        />
      )}
      {showMonitorProfilesModal && (
        <MonitorProfilesModal
          onClose={() => setShowMonitorProfilesModal(false)}
          monitorProfiles={monitorProfiles}
          monitorProfilesLoading={monitorProfilesLoading}
          refetchMonitorProfiles={refetchMonitorProfiles}
          upsertMonitorProfileMutation={upsertMonitorProfileMutation}
        />
      )}
    </>
  );
};

export default ResearchInboxPage;
