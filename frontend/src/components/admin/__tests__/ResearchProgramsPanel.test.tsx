import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import ResearchProgramsPanel from '../ResearchProgramsPanel';
import { apiClient } from '../../../services/api';

jest.mock('../../../services/api', () => ({
  apiClient: { getResearchPrograms: jest.fn() },
}));

const status = apiClient.getResearchPrograms as jest.Mock;
const EMPTY = { loaded: [], failed: [], disabled: [] };

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ResearchProgramsPanel />
    </QueryClientProvider>
  );
}

beforeEach(() => status.mockReset());

it('shows loading until status is available', () => {
  status.mockReturnValue(new Promise(() => {}));
  renderPanel();
  expect(screen.getByText('Loading research programs')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
});

it('explains an empty installation', async () => {
  status.mockResolvedValue(EMPTY);
  renderPanel();
  expect(await screen.findByText(/No research programs are loaded/)).toBeInTheDocument();
  expect(screen.queryByRole('region', { name: 'Failed programs' })).not.toBeInTheDocument();
});

it('shows loaded contributions alongside failures and disabled programs', async () => {
  status.mockResolvedValue({
    loaded: [{
      name: 'uarch_lab', version: '0.1', description: 'Microarchitecture experiments',
      tools: ['run_simulator'], sandbox_images: ['ghcr.io/example/simulator:1'],
    }],
    failed: [{ name: 'broken_lab', reason: 'could not be loaded: missing_driver' }],
    disabled: ['other_lab'],
  });
  renderPanel();
  const loaded = await screen.findByRole('region', { name: 'Loaded programs' });
  expect(within(loaded).getByText('uarch_lab · 0.1')).toBeInTheDocument();
  expect(within(loaded).getByText('Microarchitecture experiments')).toBeInTheDocument();
  fireEvent.click(within(loaded).getByText('Tools (1)'));
  expect(within(loaded).getByText('run_simulator')).toBeVisible();
  fireEvent.click(within(loaded).getByText('Sandbox images (1)'));
  expect(within(loaded).getByText('ghcr.io/example/simulator:1')).toBeVisible();
  expect(within(screen.getByRole('region', { name: 'Failed programs' })).getByText(/missing_driver/)).toBeInTheDocument();
  expect(within(screen.getByRole('region', { name: 'Disabled programs' })).getByText('other_lab')).toBeInTheDocument();
});

it('can retry a failed request', async () => {
  const logError = jest.spyOn(console, 'error').mockImplementation(() => {});
  status.mockRejectedValueOnce(new Error('offline')).mockResolvedValue(EMPTY);
  renderPanel();
  expect(await screen.findByRole('alert')).toHaveTextContent('could not be loaded');
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  expect(await screen.findByText(/No research programs are loaded/)).toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  logError.mockRestore();
});

it('refreshes status and preserves the last result if refreshing fails', async () => {
  const logError = jest.spyOn(console, 'error').mockImplementation(() => {});
  status.mockResolvedValueOnce(EMPTY).mockRejectedValueOnce(new Error('offline'));
  renderPanel();
  await screen.findByText(/No research programs are loaded/);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('last successful request');
  expect(screen.getByText(/No research programs are loaded/)).toBeInTheDocument();
  expect(status).toHaveBeenCalledTimes(2);
  logError.mockRestore();
});
