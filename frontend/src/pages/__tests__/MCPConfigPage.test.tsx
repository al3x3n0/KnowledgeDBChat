import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter, useLocation } from 'react-router-dom';

import { apiClient } from '../../services/api';
import MCPConfigPage from '../MCPConfigPage';

jest.mock('../../services/api', () => ({
  apiClient: {
    getMCPEndpoint: () => 'https://lab.example/api/v1/mcp',
    createMCPProfile: jest.fn(),
    getMCPProfileOptions: jest.fn(),
    getMCPKeyConfig: jest.fn(),
    getAPIKeyUsage: jest.fn(),
    listMCPProfileNotes: jest.fn(),
    listAPIKeys: jest.fn(),
    listMCPTools: jest.fn(),
    getDocumentSources: jest.fn(),
  },
}));

function Location() {
  return <span data-testid="route-search">{useLocation().search}</span>;
}

function renderPage(location: string) {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[location]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <MCPConfigPage />
        <Location />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const labSource = {
  id: 'source-1',
  name: 'Lab corpus',
  is_active: true,
  source_type: 'file' as const,
  config: {},
  created_at: '',
  updated_at: '',
};
const extraSource = { ...labSource, id: 'source-2', name: 'Other corpus' };

beforeEach(() => {
  jest.mocked(apiClient.listAPIKeys).mockResolvedValue({ api_keys: [], total: 0 });
  jest.mocked(apiClient.getDocumentSources).mockResolvedValue([]);
  jest.mocked(apiClient.listMCPTools).mockResolvedValue([]);
  jest.mocked(apiClient.getMCPProfileOptions).mockResolvedValue({
    programs: [{ name: 'RTL / Lab', version: '2.1', description: 'Architecture', tools: [] }],
    tools: [
      { name: 'search', display_name: 'Semantic Search' },
      { name: 'list_documents', display_name: 'List Documents' },
      { name: 'get_document', display_name: 'Get Document' },
      { name: 'chat', display_name: 'Chat' },
    ].map((tool) => ({
      ...tool,
      description: '',
      category: 'read',
      required_scope: 'read',
      config_schema: {},
    })),
  });
  jest.mocked(apiClient.listMCPProfileNotes).mockImplementation(async (_query, id) =>
    id === 'note-1' ? [{ id: 'note-1', title: 'Lab findings' }] : []
  );
});

test('prefills linked research context and clears the link on cancel', async () => {
  renderPage('/mcp-config?program=RTL%20%2F%20Lab&note=note-1&view=setup');
  expect(await screen.findByLabelText('Name')).toHaveValue('Lab findings');
  expect(await screen.findByRole('checkbox', { name: /RTL \/ Lab/ })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Lab findings' })).toBeChecked();
  expect(apiClient.listMCPProfileNotes).toHaveBeenCalledWith('', 'note-1');
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeEnabled();
  expect(apiClient.createMCPProfile).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(screen.getByTestId('route-search')).toHaveTextContent('?view=setup');
  expect(screen.queryByLabelText('Name')).toBeNull();
});

test('opens the program-only builder and keeps the profile name editable', async () => {
  renderPage('/mcp-config?program=RTL%20%2F%20Lab');
  expect(await screen.findByRole('checkbox', { name: /RTL \/ Lab/ })).toBeChecked();
  const name = screen.getByLabelText('Name');
  expect(name).toHaveValue('RTL / Lab');
  fireEvent.change(name, { target: { value: 'Shared architecture' } });
  expect(name).toHaveValue('Shared architecture');
});

test('allows a new profile when the linked note is unavailable', async () => {
  renderPage('/mcp-config?note=missing');
  expect(await screen.findByRole('alert')).toHaveTextContent('The linked research note is unavailable.');
  expect(screen.getByRole('button', { name: 'New MCP profile' })).toBeEnabled();
});

test('starts publishing without an API key', async () => {
  renderPage('/mcp-config');
  expect(screen.getByRole('heading', { name: 'MCP Profiles' })).toBeInTheDocument();
  expect(await screen.findByText('No MCP profiles yet.')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'New MCP profile' }));
  expect(await screen.findByLabelText('Name')).toBeInTheDocument();
});

test('lets the user remove an unavailable program while keeping the linked note', async () => {
  renderPage('/mcp-config?program=uninstalled&note=note-1');
  const program = await screen.findByRole('checkbox', { name: 'uninstalled (unavailable)' });
  expect(program).toBeChecked();
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeDisabled();
  fireEvent.click(program);
  expect(screen.getByRole('checkbox', { name: 'Lab findings' })).toBeChecked();
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeEnabled();
});

test('creates a profile scoped to the linked document source', async () => {
  jest.mocked(apiClient.getDocumentSources).mockResolvedValue([labSource, extraSource]);
  jest.mocked(apiClient.createMCPProfile).mockResolvedValue({ id: 'profile-1', api_key: 'one-time-key' });
  jest.mocked(apiClient.getMCPKeyConfig).mockResolvedValue({
    api_key_id: 'profile-1',
    api_key_name: 'Lab corpus',
    mcp_enabled: true,
    source_access_mode: 'restricted',
    tool_configs: [],
    source_access: [],
  });
  renderPage('/mcp-config?source=source-1');
  expect(await screen.findByLabelText('Name')).toHaveValue('Lab corpus');
  expect(await screen.findByRole('checkbox', { name: 'Semantic Search' })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'List Documents' })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Get Document' })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Chat' })).not.toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Limit to selected sources' })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Lab corpus' })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Other corpus' })).not.toBeChecked();
  expect(apiClient.createMCPProfile).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Create MCP profile' }));
  expect(await screen.findByDisplayValue('one-time-key')).toBeInTheDocument();
  expect(apiClient.createMCPProfile).toHaveBeenCalledWith({
    name: 'Lab corpus',
    programs: [],
    tools: ['search', 'list_documents', 'get_document'],
    source_ids: ['source-1'],
  });
  fireEvent.click(screen.getByRole('button', { name: 'Done' }));
  expect(screen.getByTestId('route-search')).toBeEmptyDOMElement();
});

test('requires an active source for document tools when the linked source is unavailable', async () => {
  jest.mocked(apiClient.getDocumentSources).mockResolvedValue([
    { ...labSource, is_active: false },
    extraSource,
  ]);
  renderPage('/mcp-config?source=source-1');
  expect(await screen.findByRole('alert')).toHaveTextContent('The linked document source is unavailable.');
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Alternative corpus' } });
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Semantic Search' }));
  expect(screen.getByRole('checkbox', { name: 'Limit to selected sources' })).toBeChecked();
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeDisabled();
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Other corpus' }));
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeEnabled();
});

test('shows usage for the selected profile and supports changing the time range', async () => {
  const key = {
    id: 'profile-usage',
    name: 'Active profile',
    key_prefix: 'taic_sample',
    is_mcp_profile: true,
    is_active: true,
    rate_limit_per_minute: 60,
    rate_limit_per_day: 10000,
    usage_count: 19,
    created_at: '',
  };
  const legacyKey = { ...key, id: 'legacy', name: 'General key', is_mcp_profile: false };
  jest.mocked(apiClient.listAPIKeys).mockResolvedValue({ api_keys: [legacyKey, key], total: 2 });
  jest.mocked(apiClient.getMCPKeyConfig).mockResolvedValue({
    api_key_id: key.id,
    api_key_name: key.name,
    mcp_enabled: true,
    mcp_programs: [],
    source_access_mode: 'all',
    tool_configs: [],
    source_access: [],
  });
  jest.mocked(apiClient.getAPIKeyUsage).mockImplementation(async (_id, days = 30) => ({
    key_id: key.id,
    key_name: key.name,
    period_days: days,
    total_requests: days === 30 ? 6 : 0,
    lifetime_requests: 19,
    top_endpoints: days === 30 ? [
      { endpoint: '/api/v1/mcp/rpc/tools/call/search', count: 3 },
      { endpoint: '/api/v1/mcp/rpc/resources/read', count: 2 },
      { endpoint: '/api/v1/mcp', count: 1 },
    ] : [],
  }));
  renderPage('/mcp-config');
  const connectionTab = await screen.findByRole('button', { name: 'Connection' });
  expect(screen.queryByRole('button', { name: 'General key' })).not.toBeInTheDocument();
  expect(screen.getByRole('link', { name: 'Manage MCP access for API keys' }))
    .toHaveAttribute('href', '/mcp-config?view=api-keys');
  expect(screen.queryByLabelText('MCP endpoint')).not.toBeInTheDocument();
  fireEvent.click(connectionTab);
  expect(screen.getByLabelText('MCP endpoint')).toHaveValue('https://lab.example/api/v1/mcp');
  const usageTab = await screen.findByRole('button', { name: 'Usage' });
  expect(apiClient.getAPIKeyUsage).not.toHaveBeenCalled();
  fireEvent.click(usageTab);
  const table = await screen.findByRole('table', { name: 'Most frequent operations' });
  expect(within(table).getByRole('row', { name: 'Call search 3' })).toBeInTheDocument();
  expect(within(table).getByRole('row', { name: 'Read resource 2' })).toBeInTheDocument();
  expect(within(table).getByRole('row', { name: 'MCP request 1' })).toBeInTheDocument();
  expect(apiClient.getAPIKeyUsage).toHaveBeenCalledWith(key.id, 30);

  fireEvent.change(screen.getByLabelText('Time range'), { target: { value: '7' } });
  expect(await screen.findByText('No requests in this time range.')).toBeInTheDocument();
  expect(apiClient.getAPIKeyUsage).toHaveBeenCalledWith(key.id, 7);
});
