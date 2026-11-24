import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { RefreshCw } from 'lucide-react';

import { apiClient } from '../../services/api';
import Button from '../common/Button';

const operationLabels: Record<string, string> = {
  initialize: 'Initialize connection',
  'notifications/initialized': 'Confirm connection',
  ping: 'Ping',
  'tools/list': 'List tools',
  'tools/call': 'Call tool',
  'resources/list': 'List resources',
  'resources/read': 'Read resource',
  'resources/templates/list': 'List resource templates',
  'prompts/list': 'List prompts',
  'prompts/get': 'Get prompt',
  invalid: 'Invalid request',
  unknown: 'Unknown operation',
};

function describeOperation(endpoint: string): string {
  const tool = endpoint.match(/\/mcp\/(?:rpc\/)?tools\/call\/([^/]+)$/)?.[1];
  if (tool) return `Call ${tool}`;
  if (endpoint.endsWith('/mcp')) return 'MCP request';
  const operation = endpoint.split('/mcp/rpc/')[1];
  return operationLabels[operation] ?? endpoint;
}

function formatLastUsed(value?: string): string {
  if (!value) return 'Never';
  const date = new Date(/(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`);
  return Number.isNaN(date.getTime()) ? 'Unavailable' : date.toLocaleString();
}

export default function MCPProfileUsage({ keyId }: { keyId: string }) {
  const [days, setDays] = useState(30);
  const usage = useQuery(['apiKeyUsage', keyId, days], () => apiClient.getAPIKeyUsage(keyId, days));

  return (
    <section aria-label="MCP profile usage" className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <label className="text-sm font-medium text-gray-700">
          Time range
          <select
            className="mt-1 block rounded-md border bg-white px-3 py-2 text-sm"
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
          >
            {[7, 30, 90].map((window) => <option key={window} value={window}>Last {window} days</option>)}
          </select>
        </label>
        <Button
          variant="secondary"
          size="sm"
          loading={usage.isFetching}
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={() => void usage.refetch()}
        >
          Refresh usage
        </Button>
      </div>
      {usage.isLoading && <p role="status" className="text-sm text-gray-600">Loading usage…</p>}
      {usage.isError && <p role="alert" className="text-sm text-red-700">Could not load usage. Try refreshing.</p>}
      {usage.data && (
        <>
          <dl className="grid gap-3 sm:grid-cols-3">
            {[
              ['Logged requests', usage.data.total_requests.toLocaleString()],
              ['Lifetime authentications', usage.data.lifetime_requests.toLocaleString()],
              ['Last authenticated', formatLastUsed(usage.data.last_used_at)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-md border bg-gray-50 p-3">
                <dt className="text-sm text-gray-600">{label}</dt>
                <dd className="mt-1 font-semibold text-gray-900">{value}</dd>
              </div>
            ))}
          </dl>
          {usage.data.top_endpoints.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <caption className="pb-2 text-left font-medium text-gray-900">Most frequent operations</caption>
                <thead className="border-b text-gray-600">
                  <tr><th scope="col" className="py-2">Operation</th><th scope="col" className="py-2 text-right">Requests</th></tr>
                </thead>
                <tbody className="divide-y">
                  {usage.data.top_endpoints.map(({ endpoint, count }) => (
                    <tr key={endpoint}>
                      <td className="break-all py-2 pr-4 text-gray-900">{describeOperation(endpoint)}</td>
                      <td className="py-2 text-right tabular-nums text-gray-700">{count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="text-sm text-gray-600">No requests in this time range.</p>}
        </>
      )}
    </section>
  );
}
