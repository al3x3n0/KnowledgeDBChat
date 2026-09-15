import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';

import NavigationSettings from '../NavigationSettings';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    getMyPreferences: jest.fn(),
    updateMyPreferences: jest.fn(),
    getLatexStatus: jest.fn(),
  },
}));

jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ user: { id: 'u1', role: 'user' } }),
}));

const apiClient = require('../../services/api').apiClient;

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/settings']}>
        <NavigationSettings />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

/** The `ui.nav` document sent by the nth save. */
const savedNav = (call = 0) =>
  apiClient.updateMyPreferences.mock.calls[call][0].ui.nav;

describe('NavigationSettings', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.getMyPreferences.mockResolvedValue({ ui: null });
    apiClient.getLatexStatus.mockResolvedValue({ enabled: false });
    apiClient.updateMyPreferences.mockImplementation(async (body: any) => body);
  });

  it('lists destinations that are hidden, so they can be brought back', async () => {
    // A settings page showing only what you can currently see would give you
    // no way to un-hide anything.
    apiClient.getMyPreferences.mockResolvedValue({
      ui: { nav: { hidden: ['/papers'] } },
    });
    renderPanel();

    expect(await screen.findByText('Papers')).toBeInTheDocument();
  });

  it('hides a destination', async () => {
    renderPanel();
    await screen.findByText('Papers');

    fireEvent.click(screen.getByTitle('Hide Papers'));
    await waitFor(() => expect(apiClient.updateMyPreferences).toHaveBeenCalled());

    expect(savedNav().hidden).toHaveLength(1);
  });

  it('records a rename against the route, not the label', async () => {
    // Keys must stay stable across a rename or every other preference on that
    // entry -- the pin, the hide -- would be orphaned by it.
    renderPanel();
    const runs = await screen.findByText('Runs');

    fireEvent.click(runs);
    const input = screen.getByLabelText('Rename Runs');
    fireEvent.change(input, { target: { value: 'My Runs' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(apiClient.updateMyPreferences).toHaveBeenCalled());
    expect(savedNav().renamed).toEqual({ '/autonomous-agents': 'My Runs' });
  });

  it('treats clearing the box as restoring the original name', async () => {
    apiClient.getMyPreferences.mockResolvedValue({
      ui: { nav: { renamed: { '/autonomous-agents': 'My Runs' } } },
    });
    renderPanel();
    const runs = await screen.findByText('My Runs');

    fireEvent.click(runs);
    const input = screen.getByLabelText('Rename Runs');
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(apiClient.updateMyPreferences).toHaveBeenCalled());
    expect(savedNav().renamed).toEqual({});
  });

  it('will not let a hidden page become the landing page', async () => {
    apiClient.getMyPreferences.mockResolvedValue({
      ui: { nav: { hidden: ['/papers'] } },
    });
    renderPanel();
    await screen.findByText('Papers');

    expect(
      screen.getByTitle('Papers is hidden, so it cannot be your landing page')
    ).toBeDisabled();
  });

  it('offers a reset only once something has been customized', async () => {
    const { unmount } = renderPanel();
    await screen.findByText('Papers');
    expect(screen.queryByText('Reset to defaults')).not.toBeInTheDocument();
    unmount();

    apiClient.getMyPreferences.mockResolvedValue({
      ui: { nav: { hidden: ['/papers'] } },
    });
    renderPanel();

    expect(await screen.findByText('Reset to defaults')).toBeInTheDocument();
  });

  it('does not offer to reorder the utility door', async () => {
    // Settings stays at the bottom whatever the stored order says, so an
    // arrow that appeared to move it would be a lie.
    renderPanel();
    await screen.findByText('Papers');

    expect(screen.queryByLabelText('Move Settings up')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Move R&D up')).toBeInTheDocument();
  });
});
