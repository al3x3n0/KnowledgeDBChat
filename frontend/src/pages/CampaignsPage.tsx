/**
 * Campaigns: a line of enquiry that outlives any one job.
 *
 * Campaigns have existed, run jobs and completed for some time with no way to
 * see one. `listResearchCampaigns` and `getResearchCampaign` sat in the API
 * client called by nothing, so the only evidence a campaign had ever run was
 * agent jobs appearing in a list, unattributed.
 *
 * The page is built around the thing a campaign is *for*: the answer. A
 * campaign that ran eight jobs and produced no conclusion is not a success
 * with a missing field, and the list says which of the two it was rather than
 * rendering a blank where an answer would go.
 */

import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleDashed,
  Loader2,
  Rocket,
  XCircle,
} from 'lucide-react';
import React, { useCallback, useEffect, useState } from 'react';

import { apiClient } from '../services/api';
import NewCampaignModal from '../components/agent/NewCampaignModal';
import type { ResearchCampaign, ResearchCampaignItem } from '../types';

const STATUS_STYLE: Record<string, string> = {
  active: 'bg-primary-600 text-gray-50',
  completed: 'bg-green-100 text-green-800',
  exhausted: 'bg-amber-100 text-amber-800',
  failed: 'bg-red-100 text-red-700',
};

const ITEM_ICON: Record<string, React.ReactNode> = {
  done: <CheckCircle2 className="h-3.5 w-3.5 text-green-600" />,
  running: <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-600" />,
  failed: <XCircle className="h-3.5 w-3.5 text-red-600" />,
  dropped: <XCircle className="h-3.5 w-3.5 text-gray-400" />,
};

const ItemRow: React.FC<{ item: ResearchCampaignItem }> = ({ item }) => (
  <div className="flex items-start gap-2 py-1.5">
    <span className="mt-0.5 flex-shrink-0">
      {ITEM_ICON[item.status] || <CircleDashed className="h-3.5 w-3.5 text-gray-400" />}
    </span>
    <div className="min-w-0 flex-1">
      <p className="text-sm text-gray-900">{item.title}</p>
      <p className="text-[11px] text-gray-500">
        {item.status}
        {/* Where a question came from is the whole point of a campaign: one it
            raised for itself, from what an earlier job found, is different
            evidence from one a person typed. */}
        {item.origin === 'discovered'
          ? ` · raised by the campaign (generation ${item.generation})`
          : item.origin === 'goal'
            ? ' · taken from the goal'
            : ' · asked at the start'}
      </p>
    </div>
  </div>
);

const Conclusion: React.FC<{ campaign: ResearchCampaign }> = ({ campaign }) => {
  const detail = campaign.conclusion_detail;
  const finished = campaign.status !== 'active';

  if (!campaign.conclusion && !detail) {
    if (!finished) {
      return (
        <p className="text-sm text-gray-500">
          Still running — it concludes when the work runs out or the budget does.
        </p>
      );
    }
    // Truthful about which of the two this is. A campaign that finished before
    // conclusions were recorded is not one that concluded nothing.
    return (
      <p className="text-sm text-gray-500">
        This campaign finished without recording a conclusion.
      </p>
    );
  }

  return (
    <div>
      <p className="text-sm text-gray-900">{campaign.conclusion}</p>
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
              <p className="flex items-center gap-1 text-[11px] font-medium uppercase tracking-wide text-amber-700">
                <AlertTriangle className="h-3 w-3" />
                What it could not establish
              </p>
              <ul className="mt-0.5 space-y-0.5">
                {(detail.gaps || []).map((line, idx) => (
                  <li key={idx} className="text-xs text-amber-800">
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
  const [campaigns, setCampaigns] = useState<ResearchCampaign[]>([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState<Record<string, ResearchCampaign | 'loading'>>(
    {}
  );
  const [starting, setStarting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiClient.listResearchCampaigns({ limit: 100 });
      setCampaigns(data.items || []);
    } catch {
      // apiClient surfaces the error.
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const toggle = async (campaign: ResearchCampaign) => {
    if (open[campaign.id]) {
      const next = { ...open };
      delete next[campaign.id];
      setOpen(next);
      return;
    }
    // Items come from the detail endpoint, not the list: loading every
    // question of every campaign to render rows that show none of them would
    // be a query per campaign for nothing.
    setOpen({ ...open, [campaign.id]: 'loading' });
    try {
      const full = await apiClient.getResearchCampaign(campaign.id);
      setOpen((prev) => ({ ...prev, [campaign.id]: full }));
    } catch {
      setOpen((prev) => {
        const next = { ...prev };
        delete next[campaign.id];
        return next;
      });
    }
  };

  return (
    <div className="max-w-5xl p-6">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Campaigns</h1>
          <p className="text-gray-500">
            A goal pursued across many agent jobs, each informed by what the last
            one found.
          </p>
        </div>
        <button
          onClick={() => setStarting(true)}
          className="inline-flex flex-shrink-0 items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm text-gray-50 hover:bg-primary-700"
        >
          <Rocket className="h-4 w-4" />
          New Campaign
        </button>
      </div>

      {starting && (
        <NewCampaignModal
          onClose={() => setStarting(false)}
          onCreated={load}
        />
      )}

      {loading ? (
        <p className="text-sm text-gray-500">Loading…</p>
      ) : campaigns.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 px-4 py-8 text-center">
          <Rocket className="mx-auto h-6 w-6 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600">No campaigns yet.</p>
          <p className="mt-1 text-xs text-gray-500">
            Start one from the Runs page, or just ask for one in chat.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {campaigns.map((campaign) => {
            const detail = open[campaign.id];
            const expanded = Boolean(detail);
            const launched = campaign.jobs_launched || 0;
            const budget = campaign.max_jobs || 0;
            return (
              <div
                key={campaign.id}
                className="rounded-lg border border-gray-300 bg-white"
              >
                <button
                  onClick={() => toggle(campaign)}
                  className="flex w-full items-start gap-2 px-4 py-3 text-left"
                >
                  {expanded ? (
                    <ChevronDown className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-500" />
                  ) : (
                    <ChevronRight className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-500" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-medium text-gray-900">
                        {campaign.name}
                      </span>
                      <span
                        className={`flex-shrink-0 rounded-full px-2 py-0.5 text-xs ${
                          STATUS_STYLE[campaign.status] ||
                          'bg-gray-200 text-gray-700'
                        }`}
                      >
                        {campaign.status}
                      </span>
                    </div>
                    <p className="mt-0.5 truncate text-sm text-gray-600">
                      {campaign.goal}
                    </p>
                    <p className="mt-0.5 text-xs text-gray-500">
                      {launched} of {budget} job{budget === 1 ? '' : 's'} used
                      {campaign.status === 'exhausted' &&
                        ' · stopped early, out of budget'}
                    </p>
                  </div>
                </button>

                {expanded && (
                  <div className="space-y-3 border-t border-gray-200 px-4 py-3">
                    {detail === 'loading' ? (
                      <p className="text-sm text-gray-500">Loading…</p>
                    ) : (
                      <>
                        <div>
                          <p className="text-[11px] font-medium uppercase tracking-wide text-gray-500">
                            What it concluded
                          </p>
                          <div className="mt-1">
                            <Conclusion campaign={detail} />
                          </div>
                        </div>
                        <div>
                          <p className="text-[11px] font-medium uppercase tracking-wide text-gray-500">
                            Questions ({(detail.items || []).length})
                          </p>
                          <div className="mt-1 divide-y divide-gray-200">
                            {(detail.items || []).map((item) => (
                              <ItemRow key={item.id} item={item} />
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
