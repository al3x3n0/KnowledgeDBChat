import React, { useMemo, useState } from 'react';
import { useQuery } from 'react-query';
import { format } from 'date-fns';
import { BarChart3 } from 'lucide-react';

import { apiClient } from '../services/api';
import type { LLMUsageEvent, LLMUsageSummaryItem } from '../types';
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

const UsagePage: React.FC = () => {
  const { user } = useAuth();
  const [preset, setPreset] = useState<RangePreset>('7d');

  const dateFrom = useMemo(() => presetToDateFrom(preset), [preset]);

  const summaryQuery = useQuery(
    ['llmUsageSummary', preset, user?.id],
    () =>
      apiClient.getLLMUsageSummary({
        date_from: dateFrom,
      }),
    { enabled: Boolean(user), refetchOnWindowFocus: false }
  );

  const eventsQuery = useQuery(
    ['llmUsageEvents', preset, user?.id],
    () =>
      apiClient.listLLMUsageEvents({
        date_from: dateFrom,
        page: 1,
        page_size: 50,
      }),
    { enabled: Boolean(user), refetchOnWindowFocus: false }
  );

  const summaryItems: LLMUsageSummaryItem[] = summaryQuery.data?.items || [];
  const events: LLMUsageEvent[] = eventsQuery.data?.items || [];

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-primary-600 p-2 rounded-lg">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">LLM Usage</h1>
            <p className="text-gray-600">Tokens and latency by provider/model</p>
          </div>
        </div>

        <div className="flex space-x-2">
          <Button size="sm" variant={preset === '24h' ? 'primary' : 'secondary'} onClick={() => setPreset('24h')}>24h</Button>
          <Button size="sm" variant={preset === '7d' ? 'primary' : 'secondary'} onClick={() => setPreset('7d')}>7d</Button>
          <Button size="sm" variant={preset === '30d' ? 'primary' : 'secondary'} onClick={() => setPreset('30d')}>30d</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold text-gray-900">Summary</h2>
            <p className="text-sm text-gray-600">Grouped by provider/model/task</p>
          </div>

          {(summaryQuery.isLoading || summaryQuery.isFetching) ? (
            <div className="flex justify-center py-10">
              <LoadingSpinner size="md" />
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Provider</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Task</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Req</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Tokens</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase">Avg ms</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {summaryItems.length === 0 ? (
                    <tr>
                      <td className="px-4 py-4 text-sm text-gray-600" colSpan={6}>No usage recorded for this period.</td>
                    </tr>
                  ) : (
                    summaryItems.map((row) => (
                      <tr key={`${row.provider}:${row.model}:${row.task_type}`}>
                        <td className="px-4 py-2 text-sm text-gray-900">{row.provider}</td>
                        <td className="px-4 py-2 text-sm text-gray-700">{row.model || '—'}</td>
                        <td className="px-4 py-2 text-sm text-gray-700">{row.task_type || '—'}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{row.request_count}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">{row.total_tokens}</td>
                        <td className="px-4 py-2 text-sm text-gray-700 text-right">
                          {row.avg_latency_ms != null ? Math.round(row.avg_latency_ms) : '—'}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold text-gray-900">Recent Requests</h2>
            <p className="text-sm text-gray-600">Last 50 requests</p>
          </div>

          {(eventsQuery.isLoading || eventsQuery.isFetching) ? (
            <div className="flex justify-center py-10">
              <LoadingSpinner size="md" />
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {events.length === 0 ? (
                <div className="p-4 text-sm text-gray-600">No requests recorded for this period.</div>
              ) : (
                events.map((e) => (
                  <div key={e.id} className="p-4 flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="text-sm text-gray-900 truncate">
                        {e.provider} · {e.model || '—'} · {e.task_type || '—'}
                      </div>
                      <div className="text-xs text-gray-500">
                        {format(new Date(e.created_at), 'yyyy-MM-dd HH:mm:ss')}
                      </div>
                      {e.error && <div className="text-xs text-red-600 mt-1 truncate">{e.error}</div>}
                    </div>
                    <div className="text-right text-sm text-gray-700 shrink-0">
                      <div>tokens: {e.total_tokens ?? '—'}</div>
                      <div className="text-xs text-gray-500">ms: {e.latency_ms ?? '—'}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UsagePage;

