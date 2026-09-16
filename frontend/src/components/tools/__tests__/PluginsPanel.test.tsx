import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import toast from 'react-hot-toast';
import { PluginsPanel } from '../PluginsPanel';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    listPlugins: jest.fn(),
    createPlugin: jest.fn(),
    installPlugin: jest.fn(),
    setPluginEnabled: jest.fn(),
    uninstallPlugin: jest.fn(),
    deletePlugin: jest.fn(),
    draftPlugin: jest.fn(),
    getPluginDraft: jest.fn(),
  },
}));

const apiClient = require('../../../services/api').apiClient;

const plugin = {
  id: 'plugin-1',
  slug: 'bench',
  name: 'Benchmarks',
  description: 'Benchmark bookkeeping',
  version: '0.1.0',
  source: 'user',
  owner_id: 'user-1',
  created_at: '2026-09-15T10:00:00Z',
  updated_at: '2026-09-15T10:00:00Z',
  installed: false,
  enabled: false,
  tools: [
    {
      name: 'p_bench_fetch',
      declared_name: 'fetch',
      description: 'Read some metrics',
      tool_type: 'webhook',
      effects: 'write',
      network: 'external',
      cost_tier: 'low',
      job_types: ['research'],
    },
  ],
  workflows: [],
  unavailable: [],
};

describe('PluginsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listPlugins.mockResolvedValue({ items: [plugin], total: 1 });
    apiClient.installPlugin.mockResolvedValue({ ...plugin, installed: true, enabled: true });
    apiClient.setPluginEnabled.mockResolvedValue({ ...plugin, installed: true, enabled: false });
  });

  it('shows what a tool may actually do, not what its description claims', async () => {
    // The tool describes itself as "Read some metrics" and is a webhook, which
    // posts to an arbitrary URL. The row has to say so before anyone installs.
    render(<PluginsPanel />);

    expect(await screen.findByText('Benchmarks')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Benchmarks'));

    expect(await screen.findByText('p_bench_fetch')).toBeInTheDocument();
    expect(screen.getByText('write')).toBeInTheDocument();
    expect(screen.getByText('external')).toBeInTheDocument();
  });

  it('installs a plugin that is not yet installed', async () => {
    render(<PluginsPanel />);

    fireEvent.click(await screen.findByRole('button', { name: 'Install' }));

    await waitFor(() =>
      expect(apiClient.installPlugin).toHaveBeenCalledWith('plugin-1', true)
    );
    expect(toast.success).toHaveBeenCalledWith('Benchmarks installed');
  });

  it('offers disable rather than only uninstall, once installed', async () => {
    apiClient.listPlugins.mockResolvedValue({
      items: [{ ...plugin, installed: true, enabled: true }],
      total: 1,
    });
    render(<PluginsPanel />);

    fireEvent.click(await screen.findByRole('button', { name: /Enabled/ }));

    await waitFor(() =>
      expect(apiClient.setPluginEnabled).toHaveBeenCalledWith('plugin-1', false)
    );
  });

  it('says where the manifest is wrong rather than only that it is', async () => {
    render(<PluginsPanel />);

    await screen.findByText('Benchmarks');
    fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));
    fireEvent.change(screen.getByLabelText('Plugin manifest'), {
      target: { value: '{ not json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect((toast.error as jest.Mock).mock.calls[0][0]).toMatch(
      /Manifest is not valid JSON/
    );
    expect(apiClient.createPlugin).not.toHaveBeenCalled();
  });

  it('surfaces a tool the plugin declares but cannot offer', async () => {
    apiClient.listPlugins.mockResolvedValue({
      items: [
        {
          ...plugin,
          unavailable: [
            { tool: 'shadow', reason: 'already a built-in tool' },
          ],
        },
      ],
      total: 1,
    });
    render(<PluginsPanel />);

    fireEvent.click(await screen.findByText('Benchmarks'));

    expect(await screen.findByText(/already a built-in tool/)).toBeInTheDocument();
  });

  describe('drafting from a description', () => {
    it('fills the manifest box with what was drafted', async () => {
      apiClient.draftPlugin.mockResolvedValue({ task_id: 't1', poll_url: '/x' });
      apiClient.getPluginDraft.mockResolvedValue({
        state: 'SUCCESS',
        stage: 'done',
        attempt: 1,
        notes: [],
        manifest: { id: 'drafted', name: 'Drafted', version: '0.1.0' },
        attempts: 1,
        pending: false,
      });
      render(<PluginsPanel />);
      await screen.findByText('Benchmarks');
      fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));

      fireEvent.change(
        screen.getByLabelText('Describe the plugin'),
        { target: { value: 'a run board' } }
      );
      fireEvent.click(screen.getByRole('button', { name: /^Draft$/ }));

      await waitFor(() =>
        expect(apiClient.draftPlugin).toHaveBeenCalledWith('a run board')
      );
      const box = screen.getByLabelText('Plugin manifest') as HTMLTextAreaElement;
      await waitFor(() => expect(box.value).toContain('"id": "drafted"'));
    });

    it('shows what the draft had to fix, so a repaired draft gets read harder', async () => {
      apiClient.draftPlugin.mockResolvedValue({ task_id: 't1', poll_url: '/x' });
      apiClient.getPluginDraft.mockResolvedValue({
        state: 'SUCCESS',
        stage: 'done',
        attempt: 2,
        notes: ["Attempt 1: id 'Draft-ed' must be lowercase"],
        manifest: { id: 'drafted', name: 'Drafted' },
        attempts: 2,
        pending: false,
      });
      render(<PluginsPanel />);
      await screen.findByText('Benchmarks');
      fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));
      fireEvent.change(
        screen.getByLabelText('Describe the plugin'),
        { target: { value: 'a run board' } }
      );
      fireEvent.click(screen.getByRole('button', { name: /^Draft$/ }));

      expect(await screen.findByText(/must be lowercase/)).toBeInTheDocument();
      expect(toast.success).toHaveBeenCalledWith(
        expect.stringContaining('2 attempts')
      );
    });

    it('does not install anything by drafting', async () => {
      apiClient.draftPlugin.mockResolvedValue({ task_id: 't1', poll_url: '/x' });
      apiClient.getPluginDraft.mockResolvedValue({
        state: 'SUCCESS',
        stage: 'done',
        attempt: 1,
        notes: [],
        manifest: { id: 'drafted' },
        attempts: 1,
        pending: false,
      });
      render(<PluginsPanel />);
      await screen.findByText('Benchmarks');
      fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));
      fireEvent.change(
        screen.getByLabelText('Describe the plugin'),
        { target: { value: 'x' } }
      );
      fireEvent.click(screen.getByRole('button', { name: /^Draft$/ }));

      await waitFor(() => expect(apiClient.draftPlugin).toHaveBeenCalled());
      expect(apiClient.createPlugin).not.toHaveBeenCalled();
      expect(apiClient.installPlugin).not.toHaveBeenCalled();
    });

    it('refuses to draft from nothing', async () => {
      render(<PluginsPanel />);
      await screen.findByText('Benchmarks');
      fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));
      fireEvent.click(screen.getByRole('button', { name: /^Draft$/ }));

      await waitFor(() => expect(toast.error).toHaveBeenCalled());
      expect(apiClient.draftPlugin).not.toHaveBeenCalled();
    });
  });

  it('says what workflows a plugin will create, before it is installed', async () => {
    // Unlike a tool or a view, a shipped workflow leaves editable objects in
    // your account that outlive uninstalling the plugin. That should be said
    // first, not discovered afterwards.
    apiClient.listPlugins.mockResolvedValue({
      items: [
        {
          ...plugin,
          workflows: [
            {
              id: 'stamp_flow',
              name: 'Stamp a note',
              description: '',
              node_count: 3,
              edge_count: 2,
              tools_used: ['stamp'],
            },
          ],
        },
      ],
      total: 1,
    });
    render(<PluginsPanel />);

    fireEvent.click(await screen.findByText('Benchmarks'));

    expect(await screen.findByText('Stamp a note')).toBeInTheDocument();
    expect(screen.getByText(/3 nodes, 2 edges/)).toBeInTheDocument();
    expect(screen.getByText(/stay if you uninstall/)).toBeInTheDocument();
  });

  it('shows why a draft is on its second attempt, not just that it is busy', async () => {
    // The repair loop is what makes the output worth having, so its reasons
    // are the interesting part of the wait. A spinner throws them away.
    apiClient.draftPlugin.mockResolvedValue({ task_id: 't1', poll_url: '/x' });
    apiClient.getPluginDraft
      .mockResolvedValueOnce({
        state: 'PROGRESS',
        stage: 'drafting',
        attempt: 2,
        notes: ["Attempt 1: id 'my-plugin' must be lowercase"],
        manifest: null,
        attempts: 0,
        pending: true,
      })
      .mockResolvedValue({
        state: 'SUCCESS',
        stage: 'done',
        attempt: 2,
        notes: ["Attempt 1: id 'my-plugin' must be lowercase"],
        manifest: { id: 'fixed' },
        attempts: 2,
        pending: false,
      });

    render(<PluginsPanel />);
    await screen.findByText('Benchmarks');
    fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));
    fireEvent.change(screen.getByLabelText('Describe the plugin'), {
      target: { value: 'a thing' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^Draft$/ }));

    expect(await screen.findByText(/must be lowercase/)).toBeInTheDocument();
    await waitFor(() =>
      expect(
        (screen.getByLabelText('Plugin manifest') as HTMLTextAreaElement).value
      ).toContain('"id": "fixed"')
    );
  });

  it('polls rather than waiting on the request', async () => {
    apiClient.draftPlugin.mockResolvedValue({ task_id: 't1', poll_url: '/x' });
    apiClient.getPluginDraft.mockResolvedValue({
      state: 'SUCCESS',
      stage: 'done',
      attempt: 1,
      notes: [],
      manifest: { id: 'd' },
      attempts: 1,
      pending: false,
    });

    render(<PluginsPanel />);
    await screen.findByText('Benchmarks');
    fireEvent.click(screen.getByRole('button', { name: /New Plugin/ }));
    fireEvent.change(screen.getByLabelText('Describe the plugin'), {
      target: { value: 'a thing' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^Draft$/ }));

    await waitFor(() => expect(apiClient.getPluginDraft).toHaveBeenCalledWith('t1'));
    // The POST hands back a task and returns; it does not carry the manifest.
    expect(apiClient.draftPlugin).toHaveBeenCalledWith('a thing');
  });
});
