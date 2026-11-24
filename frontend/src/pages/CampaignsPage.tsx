import {
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDashed,
  Loader2,
  Rocket,
  XCircle,
} from 'lucide-react';
import React, { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Link, useSearchParams } from 'react-router-dom';

import { apiClient } from '../services/api';
import NewCampaignModal from '../components/agent/NewCampaignModal';
import Button from '../components/common/Button';
import ConfirmationModal from '../components/common/ConfirmationModal';
import type { ResearchCampaign, ResearchCampaignAction, ResearchCampaignItem } from '../types';

const STATUS_STYLE: Record<string, string> = {
  active: 'bg-primary-600 text-gray-50',
  completed: 'bg-green-100 text-green-800',
  exhausted: 'bg-amber-100 text-amber-800',
  paused: 'bg-amber-100 text-amber-800',
  failed: 'bg-red-100 text-red-700',
};

const ITEM_ICON: Record<string, React.ReactNode> = {
  done: <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />,
  running: <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-600" />,
  failed: <XCircle className="h-3.5 w-3.5 text-red-600" />,
  dropped: <XCircle className="h-3.5 w-3.5 text-gray-400" />,
};

const ItemRow: React.FC<{ item: ResearchCampaignItem; status: string }> = ({ item, status }) => (
  <div className="flex items-start gap-2 py-1.5">
    <span className="mt-0.5 flex-shrink-0">
      {ITEM_ICON[status] || <CircleDashed className="h-3.5 w-3.5 text-gray-400" />}
    </span>
    <div className="min-w-0 flex-1 break-words">
      <p className="text-sm text-gray-900">{item.title}</p>
      <p className="text-[11px] text-gray-500">
        {status}
        {/* Where a question came from is the whole point of a campaign: one it
            raised for itself, from what an earlier job found, is different
            evidence from one a person typed. */}
        {item.origin === 'discovered'
          ? ` · raised by the campaign (generation ${item.generation})`
          : item.origin === 'goal'
            ? ' · taken from the goal'
            : ' · asked at the start'}
      </p>
      {item.detail && (
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-primary-700">Instructions</summary>
          <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-relaxed text-gray-600">{item.detail}</p>
        </details>
      )}
    </div>
    {item.job_id && (
      <Link
        to={`/autonomous-agents?job=${encodeURIComponent(item.job_id)}`}
        aria-label={`Open run for ${item.title}`}
        className="inline-flex shrink-0 items-center gap-1 text-xs text-primary-700 hover:underline"
      >
        Open run <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
      </Link>
    )}
  </div>
);

const WAITING_ON: Record<string, string> = {
  approval: 'waiting for approval',
  person: 'waiting for an answer from you',
};

const SECTION_LABEL = 'text-xs font-semibold text-gray-600';

function campaignRunUrl(campaignId: string, jobId: string) {
  const target = encodeURIComponent(jobId);
  return `/autonomous-agents?job=${target}&campaign=${encodeURIComponent(campaignId)}&campaign_job=${target}`;
}

const CampaignProgress: React.FC<{ campaign: ResearchCampaign }> = ({ campaign }) => {
  const { id, status, summary } = campaign;
  const current = summary?.current_item;
  const waiting = summary?.waiting_on;
  const remaining = summary?.remaining_jobs ?? Math.max(0, campaign.max_jobs - campaign.jobs_launched);
  const canSchedule = status === 'active' || status === 'paused';
  const next = canSchedule && remaining > 0 ? summary?.next_up?.[0] : undefined;
  const latest = summary?.latest_result;
  const needsDecision = waiting === 'approval' || waiting === 'person';

  return (
    <div className="space-y-3">
      {status === 'paused' && (
        <p className="text-sm text-gray-700">Scheduling is paused. Existing runs can continue in Agent Runs.</p>
      )}
      {status === 'cancelled' && (
        <p className="text-sm text-gray-700">Campaign scheduling ended early. Existing runs can continue in Agent Runs.</p>
      )}
      {(current || waiting || status === 'active') && (
        <div className="rounded-lg border border-gray-200 bg-gray-100 p-3">
          {current && (
            <>
              <p className={SECTION_LABEL}>{status === 'cancelled' ? 'Last scheduled question' : 'Current question'}</p>
              <p className="break-words text-sm font-medium text-gray-900">{current.title}</p>
            </>
          )}
          {needsDecision && (
            <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
              {waiting === 'approval' ? 'Approval needed' : 'Your response is needed'}
            </p>
          )}
          {!needsDecision && (
            <p className="text-sm text-gray-600">
              {waiting === 'job' && 'Still running'}
              {!waiting && !current && 'Waiting for the scheduler.'}
              {!waiting && current && (status === 'active'
                ? 'Run finished. Results will be processed on the next scheduler pass.'
                : 'Run finished.')}
            </p>
          )}
          {waiting === 'approval' && summary?.plan_held && (
            <p className="mt-1 break-words text-xs text-gray-600">Approval limit check found that {summary.plan_held}.</p>
          )}
          {(current || needsDecision) && (
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-sm">
              {current && waiting === 'approval' && (
                <Link
                  to={`/autonomous-agents?tab=queue&queue_job=${encodeURIComponent(current.job_id)}&queue_item_type=approval_checkpoint&campaign=${encodeURIComponent(id)}`}
                  className="font-medium text-primary-700 hover:underline"
                >
                  Review approval
                </Link>
              )}
              {current && (
                <Link to={campaignRunUrl(id, current.job_id)} className="text-primary-700 hover:underline">
                  {waiting === 'person' ? 'Open run to respond' : 'Open run'}
                </Link>
              )}
              {!current && needsDecision && (
                <Link to="/review" className="text-primary-700 hover:underline">Open Review</Link>
              )}
            </div>
          )}
        </div>
      )}
      {canSchedule && (
        <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <div>
            <p className={SECTION_LABEL}>Job budget remaining</p>
            <p className="text-sm text-gray-900">{remaining} of {campaign.max_jobs}</p>
          </div>
          {next && (
            <div>
              <p className={SECTION_LABEL}>Next question candidate</p>
              <p className="break-words text-sm text-gray-900">{next.title}</p>
              {next.why && <p className="break-words text-xs text-gray-600">{next.why}</p>}
            </div>
          )}
        </div>
      )}
      {latest && (
        <div>
          <p className={SECTION_LABEL}>Latest result</p>
          <p className="break-words text-sm font-medium text-gray-900">{latest.title}</p>
          {latest.conclusion && <p className="break-words text-sm text-gray-700">{latest.conclusion}</p>}
          {latest.finding_count > 0 && (
            <p className="text-xs text-gray-600">{latest.finding_count} {latest.finding_count === 1 ? 'finding' : 'findings'}</p>
          )}
          <Link to={campaignRunUrl(id, latest.job_id)} className="text-sm text-primary-700 hover:underline">
            Open result run
          </Link>
        </div>
      )}
    </div>
  );
};

function questionStatus(campaign: ResearchCampaign, item: ResearchCampaignItem) {
  if (campaign.status === 'cancelled' && item.status === 'pending') return 'unscheduled';
  if (item.id === campaign.summary?.current_item?.id) {
    const waiting = campaign.summary.waiting_on;
    if (waiting === 'approval') return 'needs approval';
    if (waiting === 'person') return 'needs a response';
    if (!waiting) return 'run finished';
  }
  return item.status;
}

const Conclusion: React.FC<{ campaign: ResearchCampaign }> = ({ campaign }) => {
  const detail = campaign.conclusion_detail;

  if (!campaign.conclusion && !detail) {
    return (
      <p className="text-sm text-gray-500">
        This campaign finished without recording a conclusion.
      </p>
    );
  }

  return (
    <div>
      <p className="break-words text-sm text-gray-900">{campaign.conclusion || detail?.answer}</p>
      {detail && (
        <div className="mt-2 space-y-2">
          {(detail.evidence || []).length > 0 && (
            <div>
              <p className="text-[11px] font-medium uppercase tracking-wide text-gray-500">
                What it rests on
              </p>
              <ul className="mt-0.5 space-y-0.5">
                {(detail.evidence || []).map((line, idx) => (
                  <li key={idx} className="text-xs text-gray-700">
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {(detail.gaps || []).length > 0 && (
            <div>
              <p className="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wide text-amber-300">
                <AlertTriangle className="h-3 w-3" />
                What it could not establish
              </p>
              <ul className="mt-0.5 space-y-0.5">
                {(detail.gaps || []).map((line, idx) => (
                  <li key={idx} className="text-xs text-amber-200">
                    {line}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {detail.generated_by === 'no_evidence' && (
            <p className="text-xs text-gray-500">
              No finding in this campaign spoke to the goal, so nothing was
              concluded from it.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

const CampaignsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedId = searchParams.get('campaign') || '';
  const queryClient = useQueryClient();
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState<ResearchCampaign | null>(null);

  const list = useQuery(['research-campaigns'], () => apiClient.listResearchCampaigns({ limit: 100 }), {
    refetchInterval: (data) => data?.items?.some((campaign) => campaign.status === 'active') ? 30000 : false,
    retry: false,
  });
  const selected = useQuery(['research-campaign', selectedId], () => apiClient.getResearchCampaign(selectedId), {
    enabled: Boolean(selectedId),
    refetchInterval: (data) => data?.status === 'active' || data?.summary?.waiting_on ? 15000 : false,
    retry: false,
  });
  const scheduling = useMutation(
    ({ id, action }: { id: string; action: ResearchCampaignAction }) => apiClient.actionResearchCampaign(id, action),
    {
      onSuccess: async (campaign) => {
        await queryClient.cancelQueries(['research-campaign', campaign.id]);
        queryClient.setQueryData(['research-campaign', campaign.id], campaign);
        queryClient.setQueryData<{ items: ResearchCampaign[]; total: number } | undefined>(
          ['research-campaigns'],
          (previous) => previous && {
            ...previous,
            items: previous.items.map((item) => item.id === campaign.id
              ? { ...item, status: campaign.status, summary: campaign.summary }
              : item),
          }
        );
        setStopping(null);
        queryClient.invalidateQueries(['research-campaigns']);
      },
      onError: (_error, { id }) => {
        queryClient.invalidateQueries(['research-campaign', id]);
        queryClient.invalidateQueries(['research-campaigns']);
      },
    }
  );

  const listedCampaigns = list.data?.items || [];
  const selectedIsListed = listedCampaigns.some((campaign) => campaign.id === selectedId);
  const campaigns = selected.data && !selectedIsListed
    ? [selected.data, ...listedCampaigns]
    : listedCampaigns;

  const selectCampaign = (id: string | null) => {
    setSearchParams((params) => {
      const next = new URLSearchParams(params);
      if (id) next.set('campaign', id);
      else next.delete('campaign');
      return next;
    });
  };

  return (
    <div className="w-full max-w-5xl p-4 lg:p-6">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Research Campaigns</h1>
          <p className="mt-0.5 text-sm text-gray-600">
            Set a research goal and a job budget. Each finding can guide the next investigation.
          </p>
        </div>
        <Button type="button" onClick={() => setStarting(true)} icon={<Rocket className="h-4 w-4" />}>
          New Campaign
        </Button>
      </div>

      <nav aria-label="Related research pages" className="mb-4 flex flex-wrap gap-x-4 gap-y-1 border-b border-gray-300 pb-3 text-sm">
        <Link to="/autonomous-agents" className="text-primary-700 hover:underline">Agent Runs</Link>
        <Link to="/review" className="text-primary-700 hover:underline">Review</Link>
        <Link to="/research-notes" className="text-primary-700 hover:underline">Research Notes</Link>
        <Link to="/pipelines" className="text-primary-700 hover:underline">Pipeline Studio</Link>
      </nav>

      {starting && (
        <NewCampaignModal
          onClose={() => setStarting(false)}
          onCreated={(campaign) => {
            queryClient.invalidateQueries(['research-campaigns']);
            selectCampaign(campaign.id);
          }}
        />
      )}
      {stopping && (
        <ConfirmationModal
          isOpen
          onClose={() => setStopping(null)}
          onConfirm={() => scheduling.mutate({ id: stopping.id, action: 'stop' })}
          title="Stop campaign?"
          message={`“${stopping.name}” will stop scheduling new questions permanently. Existing runs can continue in Agent Runs.`}
          confirmText="Stop campaign"
          variant="danger"
          isLoading={scheduling.isLoading}
        />
      )}

      {list.isError && (
        <div className="mb-3 flex items-center gap-3 text-sm" role="alert">
          <p>Could not load campaigns.</p>
          <Button type="button" variant="secondary" size="sm" onClick={() => list.refetch()}>Try again</Button>
        </div>
      )}
      {selectedId && !selectedIsListed && selected.isError && (
        <div className="mb-3 flex items-center gap-3 text-sm" role="alert">
          <p>Could not open the selected campaign.</p>
          <Button type="button" variant="secondary" size="sm" onClick={() => selectCampaign(null)}>Back to campaigns</Button>
        </div>
      )}
      {campaigns.length === 0 && list.isLoading && (
        <p className="text-sm text-gray-600" role="status">Loading campaigns…</p>
      )}
      {selectedId && !selectedIsListed && selected.isLoading && !list.isLoading && (
        <p className="mb-3 text-sm text-gray-600" role="status">Opening campaign…</p>
      )}
      {list.isSuccess && campaigns.length === 0 && !selectedId && (
        <div className="rounded-lg border border-dashed border-gray-300 bg-gray-100 px-4 py-10 text-center">
          <Rocket className="mx-auto h-6 w-6 text-gray-400" />
          <p className="mt-2 font-semibold text-gray-900">No campaigns yet.</p>
          <p className="mt-1 text-xs text-gray-500">
            Start with a research question, or <Link to="/chat" className="text-primary-700 hover:underline">ask for one in chat</Link>.
          </p>
        </div>
      )}
      {campaigns.length > 0 && (
        <div className="space-y-2">
          {campaigns.map((campaign) => {
            const expanded = selectedId === campaign.id;
            const detail = expanded ? selected.data : undefined;
            const summary = detail ?? campaign;
            const launched = summary.summary?.jobs_used ?? summary.jobs_launched;
            const budget = summary.max_jobs || 0;
            const waiting = WAITING_ON[summary.summary?.waiting_on ?? ''];
            return (
              <div
                key={campaign.id}
                className="rounded-lg border border-gray-300 bg-white"
              >
                <button
                  type="button"
                  onClick={() => selectCampaign(expanded ? null : campaign.id)}
                  aria-expanded={expanded}
                  className="flex w-full items-start gap-2 rounded-lg px-4 py-3 text-left hover:bg-gray-100"
                >
                  {expanded ? (
                    <ChevronDown className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-500" />
                  ) : (
                    <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-500" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="min-w-0 break-words font-medium text-gray-900">
                        {campaign.name}
                      </span>
                      <span
                        className={`flex-shrink-0 rounded-full px-2 py-0.5 text-xs ${
                          STATUS_STYLE[summary.status] ||
                          'bg-gray-200 text-gray-700'
                        }`}
                      >
                        {summary.status === 'cancelled' ? 'stopped' : summary.status}
                      </span>
                    </div>
                    <p className="mt-0.5 break-words text-sm text-gray-600">
                      {campaign.goal}
                    </p>
                    <p className="mt-0.5 text-xs text-gray-500">
                      {launched} of {budget} job{budget === 1 ? '' : 's'} used
                      {summary.status === 'exhausted' &&
                        ' · stopped early, out of budget'}
                      {!expanded && waiting && ` · ${waiting}`}
                      {!expanded && summary.plan_approval &&
                        ` · approves its own plans up to ${summary.plan_approval.max_stages} ` +
                          `stage${summary.plan_approval.max_stages === 1 ? '' : 's'}, ` +
                          `${summary.plan_approval.max_minutes} min`}
                    </p>
                  </div>
                </button>

                {expanded && (
                  <div className="space-y-3 border-t border-gray-200 px-4 py-3">
                    {selected.isError && (
                      <div className="flex items-center gap-3 text-sm" role="alert">
                        <p>Could not load campaign details.</p>
                        <Button type="button" variant="secondary" size="sm" onClick={() => selected.refetch()}>Try again</Button>
                      </div>
                    )}
                    {selected.isLoading && <p className="text-sm text-gray-600" role="status">Loading campaign details…</p>}
                    {detail && (
                      <>
                        {(detail.conclusion || detail.conclusion_detail ||
                          (detail.status !== 'active' && detail.status !== 'paused' && detail.status !== 'cancelled')) && (
                          <div>
                            <h2 className={SECTION_LABEL}>Conclusion</h2>
                            <div className="mt-1">
                              <Conclusion campaign={detail} />
                            </div>
                          </div>
                        )}
                        <CampaignProgress campaign={detail} />
                        {(detail.status === 'active' || detail.status === 'paused') && (
                          <div>
                            {detail.plan_approval && (
                              <p className="mb-2 text-xs text-gray-600">
                                Automatic plan approvals are limited to {detail.plan_approval.max_stages} {detail.plan_approval.max_stages === 1 ? 'stage' : 'stages'} and {detail.plan_approval.max_minutes} min.
                              </p>
                            )}
                            <div className="flex flex-wrap gap-2">
                              <Button
                                size="sm"
                                variant="secondary"
                                disabled={scheduling.isLoading}
                                onClick={() => scheduling.mutate({
                                  id: detail.id,
                                  action: detail.status === 'active' ? 'pause' : 'resume',
                                })}
                              >
                                {detail.status === 'active' ? 'Pause scheduling' : 'Resume scheduling'}
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={scheduling.isLoading}
                                onClick={() => setStopping(detail)}
                              >
                                Stop campaign
                              </Button>
                            </div>
                          </div>
                        )}
                        <div>
                          <h2 className={SECTION_LABEL}>
                            Questions ({(detail.items || []).length})
                          </h2>
                          <div className="mt-1 divide-y divide-gray-200">
                            {(detail.items || []).map((item) => (
                              <ItemRow key={item.id} item={item} status={questionStatus(detail, item)} />
                            ))}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default CampaignsPage;
