import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import toast from 'react-hot-toast';
import { MLflowConnectionsPanel } from '../MLflowConnectionsPanel';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
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
  id: 'mlflow-1',
  name: 'MLflow Research Tracking',
  provider_type: 'mlflow',
  endpoint_url: 'https://mlflow.example.test',
  capabilities: ['mlflow.experiments.search', 'mlflow.runs.get'],
  auth_type: 'bearer',
  secret_id: 'secret-1',
  timeout_seconds: 60,
  is_enabled: true,
  version: 1,
  created_at: '2026-07-28T10:00:00Z',
  updated_at: '2026-07-28T10:00:00Z',
};

describe('MLflowConnectionsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listExternalAgentConnections.mockResolvedValue({
      agents: [connection],
      total: 1,
    });
    apiClient.listSecrets.mockResolvedValue([]);
    apiClient.invokeExternalAgentConnection.mockResolvedValue({
      status: 'completed',
      audit_id: 'audit-1',
      output: { output: { experiments: [] } },
    });
  });

  it('runs an audited bounded tracking API check', async () => {
    render(<MLflowConnectionsPanel />);

    expect(await screen.findByText('MLflow Research Tracking')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Test tracking API' }));

    await waitFor(() =>
      expect(apiClient.invokeExternalAgentConnection).toHaveBeenCalledWith(
        'mlflow-1',
        expect.objectContaining({
          capability: 'mlflow.experiments.search',
          payload: { max_results: 1 },
        })
      )
    );
    expect(
      await screen.findByText('Tracking API reached; 0 experiment sampled')
    ).toBeInTheDocument();
    expect(toast.success).toHaveBeenCalledWith('MLflow connection check passed');
  });
});
