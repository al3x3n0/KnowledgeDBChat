import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import toast from 'react-hot-toast';
import { CompOpsEvidenceImportPanel } from '../CompOpsEvidenceImportPanel';

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
    invokeExternalAgentConnection: jest.fn(),
    listCompOpsEvidenceSubscriptions: jest.fn(),
    createCompOpsEvidenceSubscription: jest.fn(),
    updateCompOpsEvidenceSubscription: jest.fn(),
    syncCompOpsEvidenceSubscription: jest.fn(),
    enableCompOpsSubscriptionWebhook: jest.fn(),
    disableCompOpsSubscriptionWebhook: jest.fn(),
  },
}));

const apiClient = require('../../../services/api').apiClient;
const connection = {
  id: 'compops-1',
  name: 'CompOps Compiler Research',
  provider_type: 'compops',
  endpoint_url: 'https://compops.example.test',
  capabilities: [
    'compops.studies.report',
    'compops.studies.gates.evaluate',
    'compops.runs.get',
    'compops.artifacts.get',
    'compops.artifacts.lineage',
  ],
  auth_type: 'bearer',
  timeout_seconds: 60,
  is_enabled: true,
  version: 1,
  created_at: '2026-07-28T10:00:00Z',
  updated_at: '2026-07-28T10:00:00Z',
};

const renderPanel = (onImported = jest.fn()) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <CompOpsEvidenceImportPanel
        jobId="rnd-job-1"
        onImported={onImported}
      />
    </QueryClientProvider>
  );
  return onImported;
};

describe('CompOpsEvidenceImportPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listExternalAgentConnections.mockResolvedValue({
      agents: [connection],
      total: 1,
    });
    apiClient.invokeExternalAgentConnection.mockResolvedValue({
      status: 'completed',
      audit_id: 'audit-1',
      evidence_linked: true,
      output: {
        raw_report: 'sensitive compiler output that must not be rendered',
      },
    });
    apiClient.listCompOpsEvidenceSubscriptions.mockResolvedValue({
      subscriptions: [],
      total: 0,
    });
    apiClient.createCompOpsEvidenceSubscription.mockResolvedValue({
      evidence_changed: true,
      subscription: {
        id: 'subscription-1',
        job_id: 'rnd-job-1',
        tool_id: 'compops-1',
        capability: 'compops.studies.report',
        remote_id: 'study-42',
        interval_minutes: 15,
        is_enabled: true,
        status: 'active',
      },
    });
    apiClient.enableCompOpsSubscriptionWebhook.mockResolvedValue({
      subscription: {
        id: 'subscription-1',
        webhook_enabled: true,
      },
      callback_path:
        '/api/v1/external-agents/compops-webhooks/subscription-1',
      signing_secret: 'one-time-signing-secret',
      signing_format:
        'v1=hex(hmac_sha256(secret, timestamp.event_id.raw_body))',
    });
  });

  it('imports a study report into the current R&D job without rendering raw output', async () => {
    const onImported = renderPanel();

    expect(
      await screen.findByRole('heading', { name: 'Import CompOps evidence' })
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Study ID'), {
      target: { value: 'study-42' },
    });
    fireEvent.change(screen.getByLabelText('Metric (optional)'), {
      target: { value: 'cycles' },
    });
    fireEvent.click(
      screen.getByRole('button', { name: 'Import as unverified evidence' })
    );

    await waitFor(() =>
      expect(apiClient.invokeExternalAgentConnection).toHaveBeenCalledWith(
        'compops-1',
        {
          capability: 'compops.studies.report',
          payload: { study_id: 'study-42', metric: 'cycles' },
          request_id: expect.stringMatching(/^knowledgeops-evidence-\d+$/),
          agent_job_id: 'rnd-job-1',
        }
      )
    );
    expect(onImported).toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith(
      'CompOps provenance added as unverified evidence'
    );
    expect(
      screen.queryByText('sensitive compiler output that must not be rendered')
    ).not.toBeInTheDocument();
  });

  it('shows the audit reference when policy approval is required', async () => {
    apiClient.invokeExternalAgentConnection.mockResolvedValue({
      status: 'requires_approval',
      audit_id: 'audit-pending-1',
      evidence_linked: false,
    });
    renderPanel();

    await screen.findByRole('heading', { name: 'Import CompOps evidence' });
    fireEvent.change(screen.getByLabelText('Study ID'), {
      target: { value: 'study-approval' },
    });
    fireEvent.click(
      screen.getByRole('button', { name: 'Import as unverified evidence' })
    );

    expect(
      await screen.findByText(/Policy approval pending · audit/)
    ).toHaveTextContent('audit-pending-1');
    expect(toast.success).toHaveBeenCalledWith(
      'CompOps evidence import is waiting for policy approval'
    );
  });

  it('creates a bounded recurring evidence subscription', async () => {
    const onImported = renderPanel();

    await screen.findByRole('heading', { name: 'Import CompOps evidence' });
    fireEvent.change(screen.getByLabelText('Study ID'), {
      target: { value: 'study-recurring' },
    });
    fireEvent.click(screen.getByLabelText('Keep this evidence synchronized'));
    fireEvent.change(screen.getByLabelText('CompOps synchronization interval'), {
      target: { value: '30' },
    });
    fireEvent.click(
      screen.getByRole('button', { name: 'Import and keep synchronized' })
    );

    await waitFor(() =>
      expect(apiClient.createCompOpsEvidenceSubscription).toHaveBeenCalledWith(
        'rnd-job-1',
        {
          tool_id: 'compops-1',
          capability: 'compops.studies.report',
          payload: { study_id: 'study-recurring' },
          interval_minutes: 30,
          sync_immediately: true,
        }
      )
    );
    expect(onImported).toHaveBeenCalled();
    expect(toast.success).toHaveBeenCalledWith(
      'CompOps evidence synchronization started'
    );
  });

  it('enables signed push refreshes and shows the rotated secret once', async () => {
    apiClient.listCompOpsEvidenceSubscriptions.mockResolvedValue({
      subscriptions: [
        {
          id: 'subscription-1',
          job_id: 'rnd-job-1',
          tool_id: 'compops-1',
          capability: 'compops.runs.get',
          remote_id: 'run-push',
          payload: { run_id: 'run-push' },
          interval_minutes: 15,
          is_enabled: true,
          status: 'active',
          webhook_enabled: false,
          created_at: '2026-07-28T10:00:00Z',
          updated_at: '2026-07-28T10:00:00Z',
        },
      ],
      total: 1,
    });
    renderPanel();

    fireEvent.click(
      await screen.findByRole('button', { name: 'Enable push for run-push' })
    );

    await waitFor(() =>
      expect(apiClient.enableCompOpsSubscriptionWebhook).toHaveBeenCalledWith(
        'rnd-job-1',
        'subscription-1'
      )
    );
    expect(
      await screen.findByLabelText('CompOps webhook signing secret')
    ).toHaveValue('one-time-signing-secret');
    expect(screen.getByLabelText('CompOps webhook callback URL')).toHaveValue(
      'http://localhost/api/v1/external-agents/compops-webhooks/subscription-1'
    );
    expect(toast.success).toHaveBeenCalledWith(
      'Signed CompOps push events enabled'
    );
  });
});
