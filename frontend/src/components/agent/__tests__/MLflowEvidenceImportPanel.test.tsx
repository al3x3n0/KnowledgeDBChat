import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import toast from 'react-hot-toast';
import { MLflowEvidenceImportPanel } from '../MLflowEvidenceImportPanel';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    listExternalAgentConnections: jest.fn(),
    invokeExternalAgentConnection: jest.fn(),
  },
}));

const apiClient = require('../../../services/api').apiClient;

const renderPanel = (onImported = jest.fn()) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MLflowEvidenceImportPanel jobId="rnd-job-1" onImported={onImported} />
    </QueryClientProvider>
  );
  return onImported;
};

describe('MLflowEvidenceImportPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listExternalAgentConnections.mockResolvedValue({
      agents: [
        {
          id: 'mlflow-1',
          name: 'MLflow Tracking',
          provider_type: 'mlflow',
          endpoint_url: 'https://mlflow.example.test',
          capabilities: ['mlflow.runs.get', 'mlflow.artifacts.list'],
          auth_type: 'bearer',
          timeout_seconds: 60,
          is_enabled: true,
          version: 1,
          created_at: '2026-07-28T10:00:00Z',
          updated_at: '2026-07-28T10:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.invokeExternalAgentConnection.mockResolvedValue({
      status: 'completed',
      audit_id: 'audit-mlflow-1',
      evidence_linked: true,
      output: {
        run: { data: { metrics: [{ key: 'cycles', value: 900 }] } },
      },
    });
  });

  it('imports run provenance without rendering the raw MLflow response', async () => {
    const onImported = renderPanel();

    expect(
      await screen.findByRole('heading', { name: 'Import MLflow evidence' })
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Run ID'), {
      target: { value: 'run-42' },
    });
    fireEvent.click(
      screen.getByRole('button', { name: 'Import audited provenance' })
    );

    await waitFor(() =>
      expect(apiClient.invokeExternalAgentConnection).toHaveBeenCalledWith(
        'mlflow-1',
        {
          capability: 'mlflow.runs.get',
          payload: { run_id: 'run-42' },
          request_id: expect.stringMatching(
            /^knowledgeops-mlflow-evidence-\d+$/
          ),
          agent_job_id: 'rnd-job-1',
        }
      )
    );
    expect(onImported).toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith(
      'MLflow provenance added as unverified evidence'
    );
    expect(screen.queryByText('cycles')).not.toBeInTheDocument();
  });
});
