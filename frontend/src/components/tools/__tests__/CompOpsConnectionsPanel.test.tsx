import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import toast from 'react-hot-toast';
import { CompOpsConnectionsPanel } from '../CompOpsConnectionsPanel';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    listExternalAgentConnections: jest.fn(),
    createExternalAgentConnection: jest.fn(),
    invokeExternalAgentConnection: jest.fn(),
    listSecrets: jest.fn(),
    storeSecret: jest.fn(),
  },
}));

const apiClient = require('../../../services/api').apiClient;

const connection = {
  id: 'compops-1',
  name: 'CompOps Compiler Research',
  description: null,
  provider_type: 'compops',
  endpoint_url: 'https://compops.example.test',
  capabilities: ['compops.health', 'compops.operators.list'],
  auth_type: 'bearer',
  secret_id: 'secret-1',
  timeout_seconds: 60,
  is_enabled: true,
  version: 1,
  created_at: '2026-07-28T10:00:00Z',
  updated_at: '2026-07-28T10:00:00Z',
};

describe('CompOpsConnectionsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listExternalAgentConnections.mockResolvedValue({
      agents: [connection],
      total: 1,
    });
    apiClient.listSecrets.mockResolvedValue([
      {
        id: 'secret-1',
        name: 'compops-token',
        created_at: '2026-07-28T10:00:00Z',
        updated_at: '2026-07-28T10:00:00Z',
      },
    ]);
    apiClient.invokeExternalAgentConnection.mockResolvedValue({
      status: 'completed',
      audit_id: 'audit-1',
      output: { status: 'ok' },
    });
    apiClient.storeSecret.mockResolvedValue({
      id: 'secret-2',
      name: 'compops-researcher-token',
      created_at: '2026-07-28T10:00:00Z',
      updated_at: '2026-07-28T10:00:00Z',
    });
    apiClient.createExternalAgentConnection.mockResolvedValue(connection);
  });

  it('runs an audited health check against a registered CompOps system', async () => {
    render(<CompOpsConnectionsPanel />);

    expect(await screen.findByText('CompOps Compiler Research')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Health check' }));

    await waitFor(() =>
      expect(apiClient.invokeExternalAgentConnection).toHaveBeenCalledWith(
        'compops-1',
        expect.objectContaining({
          capability: 'compops.health',
          payload: {},
        })
      )
    );
    expect(await screen.findByText('CompOps API is healthy')).toBeInTheDocument();
    expect(screen.getByText('audit audit-1')).toBeInTheDocument();
  });

  it('stores a token separately and registers only its vault reference', async () => {
    const changed = jest.fn();
    render(<CompOpsConnectionsPanel onConnectionsChanged={changed} />);

    await screen.findByText('CompOps Compiler Research');
    fireEvent.click(screen.getByRole('button', { name: 'Add CompOps' }));
    fireEvent.change(screen.getByLabelText('CompOps HTTPS base URL'), {
      target: { value: 'https://new-compops.example.test' },
    });
    fireEvent.change(screen.getByLabelText('New CompOps token'), {
      target: { value: 'plaintext-token' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Store token' }));

    await waitFor(() =>
      expect(apiClient.storeSecret).toHaveBeenCalledWith(
        'compops-researcher-token',
        'plaintext-token'
      )
    );
    await waitFor(() =>
      expect(screen.getByLabelText('New CompOps token')).toHaveValue('')
    );
    await waitFor(() =>
      expect(screen.getByLabelText('Vault credential')).toHaveValue('secret-2')
    );

    fireEvent.click(screen.getByRole('button', { name: 'Register connection' }));
    await waitFor(() =>
      expect(apiClient.createExternalAgentConnection).toHaveBeenCalledWith(
        expect.objectContaining({
          provider_type: 'compops',
          endpoint_url: 'https://new-compops.example.test',
          auth_type: 'bearer',
          secret_id: 'secret-2',
          capabilities: expect.arrayContaining([
            'compops.health',
            'compops.studies.report',
          ]),
        })
      )
    );
    expect(
      apiClient.createExternalAgentConnection.mock.calls[0][0]
    ).not.toHaveProperty('token');
    expect(changed).toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith('CompOps connection registered');
  });
});
