import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import { apiClient } from '../../../services/api';
import MCPProfileBuilder from '../MCPProfileBuilder';

jest.mock('../../../services/api', () => ({
  apiClient: {
    getMCPEndpoint: () => 'https://lab.example/api/v1/mcp',
    getMCPProfileOptions: jest.fn(),
    listMCPProfileNotes: jest.fn(),
    getDocumentSources: jest.fn(),
    createMCPProfile: jest.fn(),
    updateMCPProfile: jest.fn(),
  },
}));

beforeEach(() => {
  jest.mocked(apiClient.listMCPProfileNotes).mockResolvedValue([]);
  jest.mocked(apiClient.getMCPProfileOptions).mockResolvedValue({
    programs: [{ name: 'Architecture', version: '1', description: 'CPU research', tools: ['measure'] }],
    tools: [
      {
        name: 'search',
        display_name: 'Semantic Search',
        description: 'Search documents',
        category: 'read',
        required_scope: 'read',
        config_schema: {},
      },
      {
        name: 'chat',
        display_name: 'Chat',
        description: 'Ask a question',
        category: 'chat',
        required_scope: 'chat',
        config_schema: {},
      },
    ],
  });
  jest.mocked(apiClient.getDocumentSources).mockResolvedValue([{
    id: 'source-1',
    name: 'Lab reports',
    source_type: 'file',
    is_active: true,
    config: {},
    created_at: '',
    updated_at: '',
  }]);
});

test('creates an MCP from selected program context and tools, then clears the secret on Done', async () => {
  jest.mocked(apiClient.createMCPProfile).mockResolvedValue({ id: 'key-1', api_key: 'one-time-key' });
  const onCreated = jest.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MCPProfileBuilder onCreated={onCreated} />
    </QueryClientProvider>
  );

  fireEvent.click(screen.getByRole('button', { name: 'New MCP profile' }));
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Research client' } });
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeDisabled();
  fireEvent.click(await screen.findByRole('checkbox', { name: /Architecture/ }));
  fireEvent.click(screen.getByRole('checkbox', { name: /Semantic Search/ }));
  expect(screen.getByText('All document sources')).toBeInTheDocument();
  fireEvent.click(screen.getByText('Access preview'));
  expect(screen.getByText('Entire shared knowledge base')).toBeVisible();
  expect(screen.getByRole('list', { name: 'Previewed tools' })).toHaveTextContent('get_research_programs');
  expect(screen.getByRole('list', { name: 'Previewed resources' })).toHaveTextContent('Architecture · Program');
  fireEvent.click(screen.getByRole('checkbox', { name: 'Limit to selected sources' }));
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeDisabled();
  expect(screen.getByText('Access requires a selected source.')).toBeVisible();
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Lab reports' }));
  expect(screen.getByRole('list', { name: 'Previewed document sources' })).toHaveTextContent('Lab reports · Search');
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Create MCP profile' }));
  });

  expect(await screen.findByDisplayValue('one-time-key')).toBeInTheDocument();
  expect(screen.getByLabelText('MCP endpoint')).toHaveValue('https://lab.example/api/v1/mcp');
  expect(apiClient.createMCPProfile).toHaveBeenCalledWith({
    name: 'Research client',
    programs: ['Architecture'],
    tools: ['search'],
    source_ids: ['source-1'],
  });
  expect(onCreated).toHaveBeenCalledWith('key-1');
  fireEvent.click(screen.getByText('Client setup'));
  const configuration = screen.getByRole<HTMLTextAreaElement>('textbox', {
    name: 'GitHub Copilot MCP configuration',
  });
  expect(configuration.value).toContain('Bearer ${KDBC_MCP_KEY_KEY1}');
  expect(configuration.value).not.toContain('one-time-key');
  fireEvent.click(screen.getByRole('button', { name: 'Done' }));
  expect(screen.queryByDisplayValue('one-time-key')).toBeNull();
});

test('edits an existing profile with its current selections', async () => {
  const profile = {
    api_key_id: 'key-1',
    api_key_name: 'Current profile',
    mcp_enabled: true,
    mcp_programs: ['Architecture'],
    research_notes: [{ id: 'note-1', title: 'Lab findings' }],
    prompts: [{
      name: 'lab_summary',
      title: 'Lab summary',
      instructions: 'Summarize the lab measurements.',
      include_resources: true,
      use_inputs: false,
    }],
    allowed_tools: ['search', 'chat', 'get_research_programs', 'get_research_notes'],
    source_access_mode: 'restricted',
    source_access: [{
      source_id: 'source-1',
      source_name: 'Lab reports',
      source_type: 'file',
      can_read: true,
      can_search: true,
      can_chat: false,
    }],
    tool_configs: [{
      tool_name: 'search',
      display_name: 'Semantic Search',
      description: 'Search documents',
      category: 'read',
      is_enabled: false,
    }],
  };
  const updated = {
    ...profile,
    api_key_name: 'Updated profile',
    research_notes: [],
    prompts: [],
    allowed_tools: ['search', 'chat', 'get_research_programs'],
  };
  jest.mocked(apiClient.updateMCPProfile).mockResolvedValue(updated);
  const onSaved = jest.fn();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MCPProfileBuilder profile={profile} onSaved={onSaved} />
    </QueryClientProvider>
  );

  fireEvent.click(screen.getByRole('button', { name: 'Edit profile' }));
  expect(screen.getByLabelText('Name')).toHaveValue('Current profile');
  expect(await screen.findByRole('checkbox', { name: /Architecture/ })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: /Semantic Search/ })).toBeChecked();
  expect(screen.getByRole('checkbox', { name: 'Lab findings' })).toBeChecked();
  await screen.findByRole('checkbox', { name: 'Lab reports' });
  fireEvent.click(screen.getByText('Access preview'));
  const preview = screen.getByRole('list', { name: 'Previewed tools' });
  expect(within(preview).getByText(/Semantic Search/)).toHaveTextContent('Disabled');
  expect(screen.getByRole('list', { name: 'Previewed resources' })).toHaveTextContent('Lab findings · Note');
  expect(screen.getByRole('list', { name: 'Previewed prompts' })).toHaveTextContent('Lab summary');
  expect(screen.getByLabelText('Instructions')).toHaveValue('Summarize the lab measurements.');
  fireEvent.click(screen.getByRole('button', { name: 'Remove prompt' }));
  expect(screen.queryByRole('list', { name: 'Previewed prompts' })).toBeNull();
  expect(screen.getByRole('list', { name: 'Previewed document sources' })).toHaveTextContent(
    'Lab reports · No enabled operations'
  );
  fireEvent.click(screen.getByRole('checkbox', { name: 'Lab findings' }));
  expect(screen.getByRole('list', { name: 'Previewed resources' })).not.toHaveTextContent('Lab findings');
  expect(screen.getByRole('checkbox', { name: 'Lab reports' })).toBeChecked();
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Updated profile' } });
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  });

  expect(apiClient.updateMCPProfile).toHaveBeenCalledWith('key-1', {
    name: 'Updated profile',
    programs: ['Architecture'],
    tools: ['search', 'chat'],
    note_ids: [],
    prompts: [],
    source_ids: ['source-1'],
  });
  expect(onSaved).toHaveBeenCalledWith(updated);
  expect(screen.queryByLabelText('API key')).toBeNull();
});

test('finds a research note by name and creates a notes-only profile', async () => {
  jest.mocked(apiClient.listMCPProfileNotes).mockImplementation(async (query) =>
    query === 'Lab findings' ? [{ id: 'note-1', title: 'Lab findings' }] : []
  );
  jest.mocked(apiClient.createMCPProfile).mockResolvedValue({ id: 'notes-profile', api_key: 'notes-key' });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MCPProfileBuilder onCreated={jest.fn()} />
    </QueryClientProvider>
  );
  fireEvent.click(screen.getByRole('button', { name: 'New MCP profile' }));
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Notes MCP' } });
  fireEvent.change(screen.getByLabelText('Search by note title'), { target: { value: 'Lab findings' } });
  fireEvent.click(screen.getByRole('button', { name: 'Search notes' }));
  fireEvent.click(await screen.findByRole('checkbox', { name: 'Lab findings' }));
  fireEvent.click(screen.getByRole('button', { name: 'Create MCP profile' }));

  expect(await screen.findByDisplayValue('notes-key')).toBeInTheDocument();
  expect(apiClient.createMCPProfile).toHaveBeenCalledWith({
    name: 'Notes MCP',
    programs: [],
    tools: [],
    note_ids: ['note-1'],
  });
});

test('validates custom prompt names and creates a prompt-only profile', async () => {
  jest.mocked(apiClient.createMCPProfile).mockResolvedValue({ id: 'prompt-profile', api_key: 'prompt-key' });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MCPProfileBuilder onCreated={jest.fn()} />
    </QueryClientProvider>
  );
  fireEvent.click(screen.getByRole('button', { name: 'New MCP profile' }));
  await screen.findByRole('checkbox', { name: /Semantic Search/ });
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Lab helper' } });
  fireEvent.click(screen.getByRole('button', { name: 'Add prompt' }));
  fireEvent.change(screen.getByLabelText('Prompt title'), { target: { value: 'Write findings' } });
  fireEvent.change(screen.getByLabelText('Prompt name'), { target: { value: 'research_brief' } });
  fireEvent.change(screen.getByLabelText('Instructions'), { target: { value: 'Summarize the evidence.' } });
  expect(screen.getByRole('alert')).toHaveTextContent('distinct from the built-in prompts');
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeDisabled();

  fireEvent.change(screen.getByLabelText('Prompt name'), { target: { value: 'write_findings' } });
  fireEvent.click(screen.getByRole('checkbox', { name: 'Enable prompt inputs' }));
  expect(screen.getByRole('alert')).toHaveTextContent('Use between one and ten named inputs.');
  expect(screen.getByRole('button', { name: 'Create MCP profile' })).toBeDisabled();
  const instructions = 'Summarize {{topic}} for {{ audience }}. Restate {{topic}}. Preserve {{{literal}}}.';
  fireEvent.change(screen.getByLabelText('Instructions'), { target: { value: instructions } });
  expect(screen.getByText('topic, audience')).toBeVisible();
  fireEvent.click(screen.getByRole('checkbox', { name: 'Reference available profile resources' }));
  fireEvent.click(screen.getByText('Access preview'));
  expect(screen.getByRole('list', { name: 'Previewed prompts' })).toHaveTextContent('Write findings');
  fireEvent.click(screen.getByRole('button', { name: 'Create MCP profile' }));

  expect(await screen.findByDisplayValue('prompt-key')).toBeInTheDocument();
  expect(apiClient.createMCPProfile).toHaveBeenCalledWith({
    name: 'Lab helper',
    programs: [],
    tools: [],
    prompts: [{
      name: 'write_findings',
      title: 'Write findings',
      instructions,
      include_resources: false,
      use_inputs: true,
    }],
  });
});
