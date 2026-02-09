import React, { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { ExternalLink, FlaskConical, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import type {
  AgentDefinitionUpdate,
  LLMRoutingExperimentListItem,
  LLMRoutingExperimentRecommendationResponse,
} from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';
import Button from '../components/common/Button';
import { useAuth } from '../contexts/AuthContext';

type RangePreset = '24h' | '7d' | '30d';

function presetToDateFrom(preset: RangePreset): string {
  const now = new Date();
  const d = new Date(now);
  if (preset === '24h') d.setHours(d.getHours() - 24);
  if (preset === '7d') d.setDate(d.getDate() - 7);
  if (preset === '30d') d.setDate(d.getDate() - 30);
  return d.toISOString();
}

function getExperiment(rd: any): any | null {
  const exp = rd?.experiment;
  return exp && typeof exp === 'object' ? exp : null;
}

function stableJson(obj: any): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function toLines(s: string): string[] {
  return s.replace(/\r\n/g, '\n').split('\n');
}

// Minimal Myers diff for line arrays, returning a unified diff string.
function unifiedDiff(aLines: string[], bLines: string[], opts?: { aName?: string; bName?: string }): string {
  const aName = opts?.aName || 'before';
  const bName = opts?.bName || 'after';

  const N = aLines.length;
  const M = bLines.length;
  const max = N + M;

  type V = Record<number, number>;
  const trace: V[] = [];
  let v: V = { 1: 0 };

  const getV = (vv: V, k: number) => (vv[k] == null ? -Infinity : vv[k]);

  for (let d = 0; d <= max; d++) {
    const vNext: V = {};
    for (let k = -d; k <= d; k += 2) {
      let x: number;
      if (k === -d || (k !== d && getV(v, k - 1) < getV(v, k + 1))) {
        x = getV(v, k + 1);
      } else {
        x = getV(v, k - 1) + 1;
      }
      let y = x - k;
      while (x < N && y < M && aLines[x] === bLines[y]) {
        x++;
        y++;
      }
      vNext[k] = x;
      if (x >= N && y >= M) {
        const edits: Array<{ type: ' ' | '+' | '-'; line: string }> = [];
        let x2 = N;
        let y2 = M;

        for (let dd = d; dd > 0; dd--) {
          const vv = trace[dd - 1];
          const k2 = x2 - y2;

          let prevK: number;
          if (k2 === -dd || (k2 !== dd && getV(vv, k2 - 1) < getV(vv, k2 + 1))) {
            prevK = k2 + 1;
          } else {
            prevK = k2 - 1;
          }

          const prevX = getV(vv, prevK);
          const prevY = prevX - prevK;

          while (x2 > prevX && y2 > prevY) {
            edits.push({ type: ' ', line: aLines[x2 - 1] });
            x2--;
            y2--;
          }

          if (x2 === prevX) {
            edits.push({ type: '+', line: bLines[y2 - 1] });
            y2--;
          } else {
            edits.push({ type: '-', line: aLines[x2 - 1] });
            x2--;
          }
        }

        while (x2 > 0 && y2 > 0) {
          edits.push({ type: ' ', line: aLines[x2 - 1] });
          x2--;
          y2--;
        }

        edits.reverse();

        const header = `--- ${aName}\n+++ ${bName}\n@@ -1,${N} +1,${M} @@`;
        const body = edits.map((e) => `${e.type}${e.line}`).join('\n');
        return `${header}\n${body}`;
      }
    }
    trace.push(vNext);
    v = vNext;
  }

  const header = `--- ${aName}\n+++ ${bName}\n@@ -1,${N} +1,${M} @@`;
  return header;
}

type PromoteModalState = {
  row: LLMRoutingExperimentListItem;
  recommended_variant_id: string;
  before: any;
  after: any;
  diff: string;
};

const RoutingExperimentsPage: React.FC = () => {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const [preset, setPreset] = useState<RangePreset>('7d');
  const [enabledOnly, setEnabledOnly] = useState(true);
  const [includeSystem, setIncludeSystem] = useState(true);
  const [search, setSearch] = useState('');
  const [promoteModal, setPromoteModal] = useState<PromoteModalState | null>(null);

  const dateFrom = useMemo(() => presetToDateFrom(preset), [preset]);

  const listQuery = useQuery(
    ['llmRoutingExperiments', enabledOnly, includeSystem, search],
    () =>
      apiClient.listLLMRoutingExperiments({
        enabled_only: enabledOnly,
        include_system: includeSystem,
        search: search.trim() || undefined,
      }),
    { enabled: Boolean(user), refetchOnWindowFocus: false, retry: 1 }
  );

  const items: LLMRoutingExperimentListItem[] = listQuery.data?.items || [];

  const updateAgentMutation = useMutation(
    (args: { agentId: string; patch: AgentDefinitionUpdate }) =>
      apiClient.updateAgentDefinition(args.agentId, args.patch),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agents');
        queryClient.invalidateQueries('llmRoutingExperiments');
        toast.success('Agent updated');
      },
      onError: () => {
        toast.error('Failed to update agent');
      },
    }
  );

  const setExperimentEnabled = (row: LLMRoutingExperimentListItem, enabled: boolean) => {
    if (row.agent_is_system) {
      toast.error('System agents cannot be edited');
      return;
    }
    const rd = row.routing_defaults || {};
    const exp = getExperiment(rd);
    if (!exp) return;

    const next = { ...(rd as any) };
    next.experiment = {
      ...exp,
      enabled,
      history: Array.isArray(exp.history)
        ? [...exp.history, { at: new Date().toISOString(), action: enabled ? 'enabled' : 'disabled' }]
        : [{ at: new Date().toISOString(), action: enabled ? 'enabled' : 'disabled' }],
    };

    updateAgentMutation.mutate({ agentId: row.agent_id, patch: { routing_defaults: next } });
  };

  const computePromotion = (row: LLMRoutingExperimentListItem, winner: string): { before: any; after: any } | null => {
    const rd = row.routing_defaults || {};
    const exp = getExperiment(rd);
    if (!exp) return null;

    const variants = Array.isArray(exp.variants) ? exp.variants : [];
    const v = variants.find((x: any) => String(x?.id) === winner);
    if (!v) return null;

    const vr = v.routing && typeof v.routing === 'object' ? v.routing : {};

    const after = { ...(rd as any) };
    if (vr.tier != null) after.tier = String(vr.tier || '').toLowerCase() || undefined;
    if (Array.isArray(vr.fallback_tiers)) after.fallback_tiers = vr.fallback_tiers;
    if (vr.timeout_seconds != null) after.timeout_seconds = Number(vr.timeout_seconds);
    if (vr.max_tokens_cap != null) after.max_tokens_cap = Number(vr.max_tokens_cap);
    if (vr.cooldown_seconds != null) after.cooldown_seconds = Number(vr.cooldown_seconds);

    after.experiment = {
      ...exp,
      enabled: false,
      winner_variant_id: winner,
      promoted_at: new Date().toISOString(),
      history: Array.isArray(exp.history)
        ? [...exp.history, { at: new Date().toISOString(), action: 'promoted', details: { winner_variant_id: winner } }]
        : [{ at: new Date().toISOString(), action: 'promoted', details: { winner_variant_id: winner } }],
    };

    return { before: rd, after };
  };

  const requestPromote = (row: LLMRoutingExperimentListItem, rec?: LLMRoutingExperimentRecommendationResponse) => {
    if (row.agent_is_system) {
      toast.error('System agents cannot be edited');
      return;
    }
    const winner = String((rec as any)?.recommended_variant_id || '').trim();
    if (!winner) {
      toast.error('No recommended variant');
      return;
    }
    const computed = computePromotion(row, winner);
    if (!computed) {
      toast.error('Could not compute promotion patch');
      return;
    }

    const beforeJson = stableJson(computed.before);
    const afterJson = stableJson(computed.after);
    const diff = unifiedDiff(toLines(beforeJson), toLines(afterJson), {
      aName: 'routing_defaults.before.json',
      bName: 'routing_defaults.after.json',
    });

    setPromoteModal({
      row,
      recommended_variant_id: winner,
      before: computed.before,
      after: computed.after,
      diff,
    });
  };

  if (!user) return null;

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-primary-600 p-2 rounded-lg">
            <FlaskConical className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Routing Experiments</h1>
            <p className="text-gray-600">Global view of agent routing A/B tests</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button size="sm" variant={preset === '24h' ? 'primary' : 'secondary'} onClick={() => setPreset('24h')}>24h</Button>
          <Button size="sm" variant={preset === '7d' ? 'primary' : 'secondary'} onClick={() => setPreset('7d')}>7d</Button>
          <Button size="sm" variant={preset === '30d' ? 'primary' : 'secondary'} onClick={() => setPreset('30d')}>30d</Button>
          <Button size="sm" variant="ghost" onClick={() => listQuery.refetch()}>
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="p-4 border-b flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-700 flex items-center gap-2">
              <input type="checkbox" checked={enabledOnly} onChange={(e) => setEnabledOnly(e.target.checked)} />
              Enabled only
            </label>
            <label className="text-sm text-gray-700 flex items-center gap-2">
              <input type="checkbox" checked={includeSystem} onChange={(e) => setIncludeSystem(e.target.checked)} />
              Include system
            </label>
          </div>
          <input
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-80"
            placeholder="Search agents…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {(listQuery.isLoading || listQuery.isFetching) ? (
          <div className="flex justify-center py-10"><LoadingSpinner size="md" /></div>
        ) : listQuery.error ? (
          <div className="p-4 text-sm text-red-600">Failed to load experiments.</div>
        ) : (
          <div className="divide-y divide-gray-200">
            {items.length === 0 ? (
              <div className="p-4 text-sm text-gray-600">No experiments found.</div>
            ) : (
              items.map((row) => (
                <ExperimentRow
                  key={`${row.agent_id}:${String((row.experiment as any)?.id || '')}`}
                  row={row}
                  disabled={updateAgentMutation.isLoading}
                  preset={preset}
                  dateFrom={dateFrom}
                  onToggleEnabled={setExperimentEnabled}
                  onRequestPromote={requestPromote}
                />
              ))
            )}
          </div>
        )}
      </div>

      {promoteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
            <div className="p-4 border-b flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-gray-900">Confirm promotion</div>
                <div className="text-xs text-gray-600">
                  Agent: {promoteModal.row.agent_display_name} · exp: {(promoteModal.row.experiment as any)?.id || '—'} · winner:{' '}
                  <span className="font-mono">{promoteModal.recommended_variant_id}</span>
                </div>
              </div>
              <button
                type="button"
                className="text-xs text-gray-600 hover:text-gray-900"
                onClick={() => setPromoteModal(null)}
              >
                Close
              </button>
            </div>

            <div className="p-4 space-y-3">
              <div className="text-sm text-gray-900 font-medium">Diff (routing_defaults)</div>
              <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-3 overflow-x-auto max-h-96">{promoteModal.diff}</pre>

              <details className="border border-gray-200 rounded p-3">
                <summary className="text-xs text-gray-700 cursor-pointer">Show before/after JSON</summary>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                  <div>
                    <div className="text-xs text-gray-600 mb-1">Before</div>
                    <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto max-h-72">{stableJson(promoteModal.before)}</pre>
                  </div>
                  <div>
                    <div className="text-xs text-gray-600 mb-1">After</div>
                    <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto max-h-72">{stableJson(promoteModal.after)}</pre>
                  </div>
                </div>
              </details>
            </div>

            <div className="p-4 border-t flex items-center justify-end gap-2">
              <Button variant="ghost" onClick={() => setPromoteModal(null)}>Cancel</Button>
              <Button
                variant="ghost"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(promoteModal.diff);
                    toast.success('Copied diff');
                  } catch {
                    toast.error('Failed to copy diff');
                  }
                }}
              >
                Copy diff
              </Button>
              <Button
                variant="primary"
                disabled={updateAgentMutation.isLoading}
                onClick={() => {
                  updateAgentMutation.mutate({
                    agentId: promoteModal.row.agent_id,
                    patch: { routing_defaults: promoteModal.after } as any,
                  });
                  setPromoteModal(null);
                }}
              >
                Confirm promote
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const ExperimentRow: React.FC<{
  row: LLMRoutingExperimentListItem;
  disabled: boolean;
  preset: RangePreset;
  dateFrom: string;
  onToggleEnabled: (row: LLMRoutingExperimentListItem, enabled: boolean) => void;
  onRequestPromote: (row: LLMRoutingExperimentListItem, rec?: LLMRoutingExperimentRecommendationResponse) => void;
}> = ({ row, disabled, preset, dateFrom, onToggleEnabled, onRequestPromote }) => {
  const expId = String((row.experiment as any)?.id || '');
  const enabled = Boolean((row.experiment as any)?.enabled);
  const variants = Array.isArray((row.experiment as any)?.variants) ? (row.experiment as any).variants : [];

  const recQuery = useQuery(
    ['llmRoutingExperimentRecommendation', row.agent_id, expId, dateFrom],
    () =>
      apiClient.getLLMRoutingExperimentRecommendation({
        experiment_id: expId,
        agent_id: row.agent_id,
        date_from: dateFrom,
        limit: 50000,
      }),
    { enabled: Boolean(expId), refetchOnWindowFocus: false, retry: 1 }
  );

  const recommended = (recQuery.data as any)?.recommended_variant_id as string | undefined;
  const rationale = (recQuery.data as any)?.rationale as string | undefined;
  const obsHref = expId
    ? `/usage/routing?experiment_id=${encodeURIComponent(expId)}${recommended ? `&variant_id=${encodeURIComponent(recommended)}` : ''}&preset=${encodeURIComponent(preset)}`
    : '/usage/routing';

  return (
    <div className="p-4 space-y-2">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-sm text-gray-900 font-medium truncate">
            {row.agent_display_name} <span className="text-xs text-gray-500">({row.agent_name})</span>
          </div>
          <div className="text-xs text-gray-600 font-mono">exp: {expId || '—'}</div>
          {row.agent_is_system && <div className="text-xs text-yellow-700">system agent (read-only)</div>}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <a
            className="text-xs text-gray-600 hover:text-gray-900 inline-flex items-center"
            href={`/agent-builder?agentId=${row.agent_id}`}
          >
            <ExternalLink className="w-3 h-3 mr-1" />
            Open
          </a>
          <a
            className="text-xs text-gray-600 hover:text-gray-900 inline-flex items-center"
            href={obsHref}
          >
            <ExternalLink className="w-3 h-3 mr-1" />
            Obs
          </a>
          <Button
            size="sm"
            variant={enabled ? 'secondary' : 'primary'}
            disabled={disabled || row.agent_is_system}
            onClick={() => onToggleEnabled(row, !enabled)}
          >
            {enabled ? 'Stop' : 'Start'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={disabled}
            onClick={() => recQuery.refetch()}
          >
            Refresh rec
          </Button>
          <Button
            size="sm"
            variant="primary"
            disabled={disabled || row.agent_is_system || !recommended}
            onClick={() => onRequestPromote(row, recQuery.data as any)}
          >
            Promote
          </Button>
        </div>
      </div>

      <div className="text-xs text-gray-700">
        Variants:{' '}
        <span className="font-mono">
          {variants.length
            ? variants
                .map((v: any) => `${String(v?.id || '?')}:${Number(v?.weight ?? 0)}`)
                .join('  ')
            : '—'}
        </span>
      </div>

      <div className="text-xs text-gray-700">
        Recommendation:{' '}
        {recQuery.isLoading ? (
          <span className="text-gray-500">loading…</span>
        ) : recQuery.error ? (
          <span className="text-red-600">failed to load</span>
        ) : recommended ? (
          <span>
            <span className="font-mono">{recommended}</span>
            {rationale ? <span className="text-gray-600"> — {rationale}</span> : null}
          </span>
        ) : (
          <span className="text-gray-500">(no data)</span>
        )}
      </div>
    </div>
  );
};

export default RoutingExperimentsPage;
