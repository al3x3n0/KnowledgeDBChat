import React, { useEffect, useMemo, useState } from 'react';
import { useQuery } from 'react-query';
import { BarChart3, Activity } from 'lucide-react';
import { useLocation } from 'react-router-dom';

import { apiClient } from '../services/api';
import type { LLMRoutingSummaryItem, LLMUsageEvent, LLMRoutingExperimentRecommendationResponse } from '../types';
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

const RoutingObservabilityPage: React.FC = () => {
  const { user } = useAuth();
  const location = useLocation();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const initialPreset = useMemo(() => {
    const p = (searchParams.get('preset') || '').toLowerCase();
    if (p === '24h' || p === '7d' || p === '30d') return p as RangePreset;
    return '7d' as RangePreset;
  }, [searchParams]);
  const experimentIdFilter = useMemo(() => (searchParams.get('experiment_id') || '').trim() || null, [searchParams]);
  const experimentVariantIdFilter = useMemo(() => (searchParams.get('variant_id') || '').trim() || null, [searchParams]);

  const [preset, setPreset] = useState<RangePreset>(initialPreset);
  const [includeUnrouted, setIncludeUnrouted] = useState(true);

  const dateFrom = useMemo(() => presetToDateFrom(preset), [preset]);

  const routingQuery = useQuery(
    ['llmRoutingSummary', preset, includeUnrouted, user?.id],
    () =>
      apiClient.getLLMRoutingSummary({
        date_from: dateFrom,
        include_unrouted: includeUnrouted,
        limit: 20000,
      }),
    { enabled: Boolean(user), refetchOnWindowFocus: false }
  );


  const [selectedRow, setSelectedRow] = useState<LLMRoutingSummaryItem | null>(null);

  const eventsQuery = useQuery(
    ['llmUsageEventsForRouting', preset, selectedRow?.provider, selectedRow?.model, selectedRow?.task_type],
    () =>
      apiClient.listLLMUsageEvents({
        date_from: dateFrom,
        provider: selectedRow?.provider || undefined,
        model: selectedRow?.model || undefined,
        task_type: selectedRow?.task_type || undefined,
        page: 1,
        page_size: 200,
      }),
    { enabled: Boolean(user) && Boolean(selectedRow), refetchOnWindowFocus: false }
  );

  const filteredEvents = useMemo(() => {
    const evs: LLMUsageEvent[] = (eventsQuery.data as any)?.items || [];
    if (!selectedRow) return [];
    const tier = selectedRow.routing_tier || null;
    const attempt = selectedRow.routing_attempt ?? null;
    const expId = selectedRow.routing_experiment_id || null;
    const varId = selectedRow.routing_experiment_variant_id || null;

    return evs.filter((e) => {
      const r = (e.extra as any)?.routing;
      if (!r || typeof r !== 'object') return includeUnrouted;
      if (tier && String(r.tier || '').toLowerCase() !== String(tier).toLowerCase()) return false;
      if (attempt != null && Number(r.attempt) !== Number(attempt)) return false;
      if (expId && String(r.experiment_id || '') !== String(expId)) return false;
      if (varId && String(r.experiment_variant_id || '') !== String(varId)) return false;
      return true;
    });
  }, [eventsQuery.data, selectedRow, includeUnrouted]);

  const recommendationQuery = useQuery(
    ['llmRoutingExperimentRecommendation', preset, selectedRow?.routing_experiment_id, selectedRow?.provider, selectedRow?.model],
    () =>
      apiClient.getLLMRoutingExperimentRecommendation({
        experiment_id: String(selectedRow?.routing_experiment_id || ''),
        date_from: dateFrom,
        limit: 50000,
      }),
    { enabled: Boolean(user) && Boolean(selectedRow?.routing_experiment_id), refetchOnWindowFocus: false }
  );
  const items: LLMRoutingSummaryItem[] = routingQuery.data?.items || [];
  const displayedItems: LLMRoutingSummaryItem[] = useMemo(() => {
    let out = items;
    if (experimentIdFilter) {
      out = out.filter((r) => String(r.routing_experiment_id || '') === String(experimentIdFilter));
    }
    if (experimentVariantIdFilter) {
      out = out.filter((r) => String(r.routing_experiment_variant_id || '') === String(experimentVariantIdFilter));
    }
    return out;
  }, [items, experimentIdFilter, experimentVariantIdFilter]);

  useEffect(() => {
    if (!experimentIdFilter) return;
    if (selectedRow) return;
    if (displayedItems.length === 0) return;
    setSelectedRow(displayedItems[0]);
  }, [experimentIdFilter, displayedItems, selectedRow]);

  const totals = useMemo(() => {
    let requests = 0;
    let errors = 0;
    for (const r of displayedItems) {
      requests += r.request_count || 0;
      errors += r.error_count || 0;
    }
    const successRate = requests ? (requests - errors) / requests : 0;
    return { requests, errors, successRate };
  }, [displayedItems]);

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-primary-600 p-2 rounded-lg">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">LLM Routing Observability</h1>
            <p className="text-gray-600">Success rate + latency by routing tier/attempt</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button size="sm" variant={preset === '24h' ? 'primary' : 'secondary'} onClick={() => setPreset('24h')}>24h</Button>
          <Button size="sm" variant={preset === '7d' ? 'primary' : 'secondary'} onClick={() => setPreset('7d')}>7d</Button>
          <Button size="sm" variant={preset === '30d' ? 'primary' : 'secondary'} onClick={() => setPreset('30d')}>30d</Button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="p-4 border-b flex items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Routing Summary
            </h2>
            <p className="text-sm text-gray-600">
              {routingQuery.data?.scanned_events != null ? `Scanned ${routingQuery.data.scanned_events} events` : ''}
              {routingQuery.data?.truncated ? ' (truncated)' : ''}
            </p>
          </div>

          <label className="text-sm text-gray-700 flex items-center gap-2">
            <input
              type="checkbox"
              checked={includeUnrouted}
              onChange={(e) => setIncludeUnrouted(e.target.checked)}
            />
            Include unrouted
          </label>
        </div>

        <div className="p-4 border-b bg-gray-50 text-sm text-gray-700 flex items-center justify-between">
          <div>
            Total requests: <span className="font-medium">{totals.requests}</span> · Errors:{' '}
            <span className="font-medium">{totals.errors}</span>
          </div>
          <div>
            Success rate:{' '}
            <span className="font-medium">{Math.round(totals.successRate * 1000) / 10}%</span>
          </div>
        </div>

        {(routingQuery.isLoading || routingQuery.isFetching) ? (
          <div className="flex justify-center py-10">
            <LoadingSpinner size="md" />
          </div>
        ) : routingQuery.error ? (
          <div className="p-4 text-sm text-red-600">Failed to load routing summary.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Provider / Model</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Task</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Tier</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Experiment</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Attempt</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Req</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Err</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Success %</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">P50 ms</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">P95 ms</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {displayedItems.length === 0 ? (
                  <tr>
                    <td className="px-4 py-4 text-sm text-gray-600" colSpan={10}>No data for this period.</td>
                  </tr>
                ) : (
                  displayedItems.map((row) => {
                    const tier = row.routing_tier || '—';
                    const attemptLabel = row.routing_attempt != null ? String(row.routing_attempt) : '—';
                    const successPct = Math.round((row.success_rate || 0) * 1000) / 10;
                    const expId = row.routing_experiment_id || null;
                    const varId = row.routing_experiment_variant_id || null;

                    return (
                      <tr
                        key={`${row.provider}:${row.model}:${row.task_type}:${tier}:${attemptLabel}:${expId || ''}:${varId || ''}`}
                        className="cursor-pointer hover:bg-gray-50"
                        onClick={() => setSelectedRow(row)}
                      >
                        <td className="px-4 py-2 text-sm text-gray-900">
                          {row.provider} · {row.model || '—'}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-700">{row.task_type || '—'}</td>
                        <td className="px-4 py-2 text-sm text-gray-700">
                          <div className="font-mono">{tier}</div>
                          {row.routing_requested_tier && row.routing_requested_tier !== row.routing_tier && (
                            <div className="text-xs text-gray-500">req: {row.routing_requested_tier}</div>
                          )}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-700">
                          {expId ? (
                            <div>
                              <div className="font-mono">{expId}</div>
                              <div className="text-xs text-gray-500">var: {varId || '—'}</div>
                            </div>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{attemptLabel}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{row.request_count}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{row.error_count}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{successPct}%</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{row.p50_latency_ms ?? '—'}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{row.p95_latency_ms ?? '—'}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        )}


      {selectedRow && (
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-900 font-medium">Selected route</div>
              <div className="text-xs text-gray-600">
                {selectedRow.provider} · {selectedRow.model || '—'} · {selectedRow.task_type || '—'} · tier: {selectedRow.routing_tier || '—'} · attempt: {selectedRow.routing_attempt ?? '—'}
              </div>
            </div>
            <button
              type="button"
              className="text-xs text-gray-600 hover:text-gray-900"
              onClick={() => setSelectedRow(null)}
            >
              Clear
            </button>
          </div>

          {selectedRow.routing_experiment_id && (
            <div className="p-4 border-b bg-gray-50">
              <div className="text-sm text-gray-900 font-medium">Experiment recommendation</div>
              {(recommendationQuery.isLoading || recommendationQuery.isFetching) ? (
                <div className="text-xs text-gray-600 mt-1">Loading…</div>
              ) : recommendationQuery.error ? (
                <div className="text-xs text-red-600 mt-1">Failed to load recommendation</div>
              ) : (
                <div className="text-xs text-gray-700 mt-1">
                  {(recommendationQuery.data as LLMRoutingExperimentRecommendationResponse | undefined)?.recommended_variant_id
                    ? `Recommended: ${(recommendationQuery.data as any).recommended_variant_id} — ${(recommendationQuery.data as any).rationale}`
                    : 'No recommendation (no data).'}
                </div>
              )}
            </div>
          )}

          <div className="p-4">
            <div className="text-sm text-gray-900 font-medium">Recent matching events</div>
            {(eventsQuery.isLoading || eventsQuery.isFetching) ? (
              <div className="flex justify-center py-6"><LoadingSpinner size="sm" /></div>
            ) : (
              <div className="mt-2 space-y-2">
                {filteredEvents.length === 0 ? (
                  <div className="text-xs text-gray-600">No events match this row in the current window.</div>
                ) : (
                  filteredEvents.slice(0, 25).map((e) => {
                    const r = (e.extra as any)?.routing;
                    const decision = r?.decision;
                    return (
                      <details key={e.id} className="border border-gray-200 rounded p-2">
                        <summary className="text-xs text-gray-800 cursor-pointer flex items-center justify-between gap-2">
                          <span className="font-mono">{e.created_at}</span>
                          <span className={e.error ? 'text-red-600' : 'text-green-700'}>{e.error ? 'error' : 'ok'}</span>
                          <span className="text-gray-600">ms: {e.latency_ms ?? '—'} · tokens: {e.total_tokens ?? '—'}</span>
                        </summary>
                        <pre className="mt-2 text-xs bg-gray-50 p-2 rounded overflow-x-auto">{JSON.stringify({ routing: r, decision }, null, 2)}</pre>
                      </details>
                    );
                  })
                )}
              </div>
            )}
          </div>
        </div>
      )}

      </div>
    </div>
  );
};

export default RoutingObservabilityPage;
