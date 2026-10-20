import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter, useLocation } from 'react-router-dom';

import AdminPage from '../AdminPage';
import { apiClient } from '../../services/api';

jest.mock('../../services/api', () => ({
  apiClient: {
    getSystemHealth: jest.fn(),
    getSystemStats: jest.fn(),
    getResearchPrograms: jest.fn(),
    getLdapStatus: jest.fn(),
  },
}));

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Current location">{location.pathname}{location.search}</output>;
}

function renderAdmin(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <AdminPage />
        <CurrentLocation />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  jest.clearAllMocks();
  (apiClient.getSystemHealth as jest.Mock).mockResolvedValue({});
  (apiClient.getSystemStats as jest.Mock).mockResolvedValue({});
  (apiClient.getLdapStatus as jest.Mock).mockResolvedValue({ enabled: false });
  (apiClient.getResearchPrograms as jest.Mock).mockResolvedValue({
    loaded: [{
      name: 'uarch_lab', version: '0.1.0', description: 'Compiler research',
      tools: ['measure_kernel'], sandbox_images: [],
    }],
    failed: [], disabled: [],
  });
});

it('opens the programs panel through the admin deep link', async () => {
  renderAdmin('/admin?tab=research-programs');

  expect(await screen.findByRole('heading', { name: 'uarch_lab · 0.1.0' })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Research Programs' })).toBeInTheDocument();
  expect(apiClient.getResearchPrograms).toHaveBeenCalledTimes(1);
});

it('opens programs from the sidebar and preserves navigation back to LDAP', async () => {
  renderAdmin('/admin?tab=directory');
  expect(await screen.findByText('Disabled')).toBeInTheDocument();
  expect(apiClient.getResearchPrograms).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Research Programs' }));
  expect(await screen.findByRole('heading', { name: 'uarch_lab · 0.1.0' })).toBeInTheDocument();
  expect(screen.getByLabelText('Current location')).toHaveTextContent('/admin?tab=research-programs');

  fireEvent.click(screen.getByRole('button', { name: 'Directory (LDAP)' }));
  await waitFor(() => expect(screen.queryByRole('heading', { name: 'Research Programs' })).not.toBeInTheDocument());
  expect(screen.getByLabelText('Current location')).toHaveTextContent('/admin?tab=directory');
});
