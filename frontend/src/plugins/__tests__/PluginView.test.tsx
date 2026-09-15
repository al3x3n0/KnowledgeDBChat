import React from 'react';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import PluginView from '../PluginView';
import type { PluginViewSpec } from '../types';

jest.mock('../../services/api', () => ({
  apiClient: { readPluginViewData: jest.fn() },
}));

const apiClient = require('../../services/api').apiClient;

const table = (over: Partial<PluginViewSpec> = {}): PluginViewSpec => ({
  kind: 'table',
  title: 'Runs',
  description: '',
  source: { tool: 'recent', params: {} },
  path: 'items',
  empty: 'No runs recorded yet.',
  columns: [
    { key: 'name', label: 'Kernel' },
    { key: 'ms', label: 'ms' },
  ],
  ...over,
});

function renderView(view: PluginViewSpec) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <PluginView slug="runs" viewId="board" view={view} />
    </QueryClientProvider>
  );
}

describe('PluginView', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders rows a tool returned', async () => {
    apiClient.readPluginViewData.mockResolvedValue({
      data: { items: [{ name: 'dotprod', ms: 31 }] },
      static: false,
    });
    renderView(table());

    expect(await screen.findByText('dotprod')).toBeInTheDocument();
    expect(screen.getByText('31')).toBeInTheDocument();
    expect(screen.getByText('Kernel')).toBeInTheDocument();
  });

  it('says where a wrong path broke instead of showing a blank table', async () => {
    // A plugin author writing `result.items` against a tool returning
    // `{items: [...]}` would otherwise see an empty table and have no way to
    // tell it from a tool that returned nothing.
    apiClient.readPluginViewData.mockResolvedValue({
      data: { items: [{ name: 'dotprod' }] },
      static: false,
    });
    renderView(table({ path: 'result.items' }));

    expect(await screen.findByText(/is not there/)).toBeInTheDocument();
    expect(screen.getByText(/stopped at "result"/)).toBeInTheDocument();
  });

  it('shows the authored empty state when a tool legitimately returns nothing', async () => {
    apiClient.readPluginViewData.mockResolvedValue({
      data: { items: [] },
      static: false,
    });
    renderView(table());

    expect(await screen.findByText('No runs recorded yet.')).toBeInTheDocument();
    expect(screen.queryByText(/is not there/)).not.toBeInTheDocument();
  });

  it('surfaces a policy refusal as the reason, not as a generic failure', async () => {
    apiClient.readPluginViewData.mockRejectedValue({
      response: { data: { detail: "Tool 'user_tool:runs:recent' denied by policy" } },
    });
    renderView(table());

    expect(await screen.findByText(/denied by policy/)).toBeInTheDocument();
  });

  it('renders static markdown without calling any tool', async () => {
    renderView({
      kind: 'markdown',
      title: 'About',
      description: '',
      source: null,
      path: '',
      empty: '',
      text: '# Heading',
    });

    expect(await screen.findByTestId('react-markdown')).toBeInTheDocument();
    expect(apiClient.readPluginViewData).not.toHaveBeenCalled();
  });

  it('renders a value as text, never as markup', async () => {
    // The declarative contract holds only while plugin content stays data.
    apiClient.readPluginViewData.mockResolvedValue({
      data: { items: [{ name: '<img src=x onerror=alert(1)>', ms: 1 }] },
      static: false,
    });
    const { container } = renderView(table());

    expect(
      await screen.findByText('<img src=x onerror=alert(1)>')
    ).toBeInTheDocument();
    expect(container.querySelector('img')).toBeNull();
  });

  it('renders an object cell as JSON rather than [object Object]', async () => {
    apiClient.readPluginViewData.mockResolvedValue({
      data: { items: [{ name: { nested: true }, ms: 1 }] },
      static: false,
    });
    renderView(table());

    expect(await screen.findByText('{"nested":true}')).toBeInTheDocument();
  });

  it('renders stats from the first row', async () => {
    apiClient.readPluginViewData.mockResolvedValue({
      data: { items: [{ total: 12, failed: 0 }] },
      static: false,
    });
    renderView(
      table({
        kind: 'stats',
        columns: [
          { key: 'total', label: 'Total' },
          { key: 'failed', label: 'Failed' },
        ],
      })
    );

    expect(await screen.findByText('Total')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
  });
});
