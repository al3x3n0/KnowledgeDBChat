/**
 * Rendering a view a plugin described.
 *
 * Everything here is first-party React reading plugin *data*. No plugin code
 * runs in this origin, so a plugin can never read the viewer's token or touch
 * anything it was not handed. That is the whole point of the declarative
 * approach, and it holds only as long as nothing below renders plugin content
 * as markup — so every string goes through JSX text, markdown is rendered
 * without raw-HTML support, and there is no `dangerouslySetInnerHTML`.
 *
 * The empty state is the part worth care. A plugin author who writes
 * `result.items` against a tool returning `{items: [...]}` gets no rows, and
 * "no rows" is indistinguishable from a tool that legitimately had nothing to
 * say. So the two are told apart and the first one says where the path broke —
 * turning a blank table into something diagnosable without reading logs.
 */

import { AlertTriangle, Loader2 } from 'lucide-react';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import { useQuery } from 'react-query';

import { apiClient } from '../services/api';
import { asCell, asRows, resolvePath } from './paths';
import type { PluginViewSpec } from './types';

export interface PluginViewProps {
  slug: string;
  viewId: string;
  view: PluginViewSpec;
  /** Rendered small, for a panel in someone else's page. */
  compact?: boolean;
}

const Empty: React.FC<{ message: string; hint?: string }> = ({ message, hint }) => (
  <div className="rounded border border-dashed border-gray-300 px-3 py-4 text-center">
    <p className="text-sm text-gray-500">{message}</p>
    {hint && <p className="mt-1 text-xs text-gray-500">{hint}</p>}
  </div>
);

export const PluginView: React.FC<PluginViewProps> = ({
  slug,
  viewId,
  view,
  compact = false,
}) => {
  const hasSource = Boolean(view.source);

  const { data, isLoading, error } = useQuery(
    ['plugin-view-data', slug, viewId],
    () => apiClient.readPluginViewData(slug, viewId),
    { enabled: hasSource, retry: false, staleTime: 30 * 1000 }
  );

  if (hasSource && isLoading) {
    return (
      <div className="flex items-center gap-2 px-3 py-4 text-sm text-gray-500">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading…
      </div>
    );
  }

  if (hasSource && error) {
    // A policy denial reaches here as a 403 with the reason attached, which is
    // more useful than "failed to load" — a person who cannot see their own
    // plugin's data should be told it was refused, not that it broke.
    const detail =
      (error as any)?.response?.data?.detail || 'Could not load this view.';
    return (
      <div className="flex items-start gap-2 rounded border border-amber-300 bg-amber-50 px-3 py-2">
        <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-700" />
        <p className="text-xs text-amber-800">{String(detail)}</p>
      </div>
    );
  }

  if (view.kind === 'markdown') {
    const resolution = hasSource
      ? resolvePath(data?.data, view.path)
      : { found: true as const, value: view.text };
    const text = resolution.found ? asCell(resolution.value) : view.text || '';
    if (!text) {
      return <Empty message={view.empty || 'Nothing to show.'} />;
    }
    return (
      <div className="prose prose-sm max-w-none text-gray-900">
        {/* No rehype-raw: a plugin's markdown may not become HTML. */}
        <ReactMarkdown>{text}</ReactMarkdown>
      </div>
    );
  }

  const resolution = resolvePath(data?.data, view.path);
  if (!resolution.found) {
    return (
      <Empty
        message={view.empty || 'Nothing to show.'}
        hint={`This view reads "${view.path}" from its tool's output, and that path is not there (stopped at "${resolution.missingAt}").`}
      />
    );
  }

  const rows = asRows(resolution.value);
  const columns = view.columns || [];

  if (rows.length === 0) {
    return <Empty message={view.empty || 'Nothing to show.'} />;
  }

  if (view.kind === 'table') {
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-300 text-left">
              {columns.map((column) => (
                <th
                  key={column.key}
                  className="px-2 py-1.5 text-xs font-medium uppercase tracking-wide text-gray-500"
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, compact ? 5 : 200).map((row, index) => (
              <tr key={index} className="border-b border-gray-200 last:border-0">
                {columns.map((column) => (
                  <td key={column.key} className="px-2 py-1.5 text-gray-900">
                    {asCell(row[column.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length > (compact ? 5 : 200) && (
          <p className="mt-1 px-2 text-xs text-gray-500">
            Showing {compact ? 5 : 200} of {rows.length}.
          </p>
        )}
      </div>
    );
  }

  if (view.kind === 'stats') {
    const row = rows[0];
    return (
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {columns.map((column) => (
          <div
            key={column.key}
            className="rounded border border-gray-200 bg-white px-3 py-2"
          >
            <p className="text-xs uppercase tracking-wide text-gray-500">
              {column.label}
            </p>
            <p className="mt-0.5 truncate text-lg font-semibold text-gray-900">
              {asCell(row[column.key])}
            </p>
          </div>
        ))}
      </div>
    );
  }

  // detail
  const row = rows[0];
  return (
    <dl className="divide-y divide-gray-200">
      {columns.map((column) => (
        <div key={column.key} className="flex gap-3 py-1.5">
          <dt className="w-40 flex-shrink-0 text-xs uppercase tracking-wide text-gray-500">
            {column.label}
          </dt>
          <dd className="min-w-0 flex-1 break-words text-sm text-gray-900">
            {asCell(row[column.key])}
          </dd>
        </div>
      ))}
    </dl>
  );
};

export default PluginView;
