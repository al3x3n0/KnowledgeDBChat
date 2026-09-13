import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import toast from 'react-hot-toast';
import { AutonomousRndVerificationPanel } from '../AutonomousRndVerificationPanel';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    getAutonomousRndJobOutcome: jest.fn(),
    launchAutonomousRndVerificationTask: jest.fn(),
    listResearchNotes: jest.fn(),
    getDocumentSources: jest.fn(),
    createAgentJobProgressWebSocket: jest.fn(),
    createAutonomousRndVerificationAuditSnapshot: jest.fn(),
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
const mockSocket = () => ({
  onmessage: null as ((event: { data: string }) => void) | null,
  close: jest.fn(),
});

const outcome = {
  job_id: 'parent-job',
  job_status: 'completed',
  outcome: {
    verification_plan: {
      tasks: [
        {
          id: 'verify-evidence-1',
          objective: 'Independently verify the external response.',
        },
      ],
    },
  },
  verification_lifecycle: {
    task_count: 1,
    launch_status_counts: { not_launched: 1 },
    evidence_status_counts: { unverified: 1 },
    timeline: [
      {
        event_id: 'verify-evidence-1:proposal_created',
        task_id: 'verify-evidence-1',
        event_type: 'proposal_created',
        at: '2026-07-28T10:00:00Z',
        actor: 'planner',
        label: 'Verification proposed',
        status: 'approval_required',
        entity_type: 'agent_job',
        entity_id: 'parent-job',
      },
      {
        event_id: 'verify-evidence-1:approval_recorded',
        task_id: 'verify-evidence-1',
        event_type: 'approval_recorded',
        at: '2026-07-28T10:01:00Z',
        actor: 'operator',
        label: 'Verification approved',
        status: 'approved',
        entity_type: 'tool_audit',
        entity_id: 'audit-1',
      },
    ],
    tasks: [
      {
        task_id: 'verify-evidence-1',
        evidence_id: 'external-agent:request-1',
        evidence_status: 'unverified',
        priority: 'critical',
        priority_score: 90,
        required_checks: ['run_repeated_controlled_experiment'],
        launch_status: 'not_launched',
        job_status: null,
        approval_status: null,
        reconciliation_status: null,
        experiment_plan_id: null,
        experiment_run_id: null,
        agent_job_id: null,
        audit_id: null,
        budget: {},
      },
    ],
  },
};

const renderPanel = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <AutonomousRndVerificationPanel
        jobId="parent-job"
        defaultResearchNoteId="00000000-0000-0000-0000-000000000002"
        defaultSourceId="00000000-0000-0000-0000-000000000003"
      />
    </QueryClientProvider>
  );
};

describe('AutonomousRndVerificationPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.getAutonomousRndJobOutcome.mockResolvedValue(outcome);
    apiClient.launchAutonomousRndVerificationTask.mockResolvedValue({
      created: true,
      queued: false,
      experiment_plan_id: 'plan-1',
      experiment_run_id: 'run-1',
      agent_job_id: 'verifier-1',
      audit_id: 'audit-1',
      status: 'planned',
      budget: { repeat_count: 2 },
    });
    apiClient.listResearchNotes.mockResolvedValue({
      items: [
        {
          id: '00000000-0000-0000-0000-000000000002',
          title: 'Compiler verification notes',
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    });
    apiClient.getDocumentSources.mockResolvedValue([
      {
        id: '00000000-0000-0000-0000-000000000003',
        name: 'Compiler repository',
        source_type: 'github',
      },
    ]);
    apiClient.createAgentJobProgressWebSocket.mockReturnValue(mockSocket());
    apiClient.createAutonomousRndVerificationAuditSnapshot.mockResolvedValue({
      snapshot: { job_id: 'parent-job' },
      integrity: {
        canonicalization: 'json-sort-keys-compact-v1',
        sha256: 'a'.repeat(64),
        signature_algorithm: 'ed25519',
        signature_encoding: 'hex',
        signature: 'b'.repeat(128),
        key_id: 'knowledgeops-ed25519-v1',
        public_key: 'c'.repeat(64),
      },
    });
    apiClient.listExternalAgentConnections.mockResolvedValue({
      agents: [],
      total: 0,
    });
    apiClient.listCompOpsEvidenceSubscriptions.mockResolvedValue({
      subscriptions: [],
      total: 0,
    });
  });

  it('shows lifecycle state and submits an explicitly approved bounded recipe', async () => {
    renderPanel();

    expect(await screen.findByRole('heading', { name: 'Evidence verification' })).toBeInTheDocument();
    expect(screen.getByText('Evidence: unverified')).toBeInTheDocument();
    expect(screen.getByText('Run: not launched')).toBeInTheDocument();
    expect(screen.getByText('run repeated controlled experiment')).toBeInTheDocument();
    expect(screen.getByText('Audit timeline (2)')).toBeInTheDocument();
    expect(screen.getByText('Verification proposed')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Timeline status'), {
      target: { value: 'approved' },
    });
    expect(screen.getByText('Showing 1 of 2 events')).toBeInTheDocument();
    expect(screen.getByText('Verification approved')).toBeInTheDocument();
    expect(screen.queryByText('Verification proposed')).not.toBeInTheDocument();
    const createObjectUrl = jest.fn(() => 'blob:verification-audit');
    const revokeObjectUrl = jest.fn();
    Object.defineProperty(window.URL, 'createObjectURL', {
      configurable: true,
      value: createObjectUrl,
    });
    Object.defineProperty(window.URL, 'revokeObjectURL', {
      configurable: true,
      value: revokeObjectUrl,
    });
    const digest = jest.fn(async () => new Uint8Array(32).fill(7).buffer);
    Object.defineProperty(window, 'crypto', {
      configurable: true,
      value: { subtle: { digest } },
    });
    Object.defineProperty(global, 'TextEncoder', {
      configurable: true,
      value: class {
        encode(value: string) {
          return new Uint8Array(Array.from(value).map((character) => character.charCodeAt(0)));
        }
      },
    });
    const anchorClick = jest
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined);
    fireEvent.click(screen.getByRole('button', { name: 'Export hashed JSON' }));
    await waitFor(() => expect(createObjectUrl).toHaveBeenCalled());
    expect(digest).toHaveBeenCalledWith('SHA-256', expect.any(Uint8Array));
    expect(anchorClick).toHaveBeenCalled();
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:verification-audit');
    expect(toast.success).toHaveBeenCalledWith('Hashed verification audit exported');

    fireEvent.click(screen.getByRole('button', { name: 'Export signed JSON' }));
    await waitFor(() =>
      expect(apiClient.createAutonomousRndVerificationAuditSnapshot).toHaveBeenCalledWith(
        'parent-job',
        { task_id: undefined, status: 'approved' }
      )
    );
    expect(toast.success).toHaveBeenCalledWith(
      'Immutable signed audit snapshot exported'
    );
    anchorClick.mockRestore();

    fireEvent.click(screen.getByRole('button', { name: 'Configure' }));
    expect(await screen.findByRole('option', { name: 'Compiler verification notes' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Compiler repository · github' })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Approval note'), {
      target: { value: 'Approved bounded local verification' },
    });
    fireEvent.change(screen.getByLabelText('Commands (one per line, maximum four)'), {
      target: { value: 'pytest -q\npython -m compileall app' },
    });
    fireEvent.click(
      screen.getByLabelText('I approve this exact local verification recipe and its resource limits.')
    );
    fireEvent.click(screen.getByRole('button', { name: 'Approve and launch' }));

    await waitFor(() =>
      expect(apiClient.launchAutonomousRndVerificationTask).toHaveBeenCalledWith(
        'parent-job',
        'verify-evidence-1',
        expect.objectContaining({
          approval_confirmed: true,
          approval_note: 'Approved bounded local verification',
          commands: ['pytest -q', 'python -m compileall app'],
          repeat_count: 2,
          timeout_seconds: 30,
          max_runtime_minutes: 2,
          budget_limit: 1,
          start_immediately: false,
        })
      )
    );
    expect(toast.success).toHaveBeenCalledWith('Verification plan created');
  });

  it('refreshes lifecycle state when the verifier job emits progress', async () => {
    const socket = mockSocket();
    apiClient.createAgentJobProgressWebSocket.mockReturnValue(socket);
    apiClient.getAutonomousRndJobOutcome.mockResolvedValue({
      ...outcome,
      verification_lifecycle: {
        ...outcome.verification_lifecycle,
        launch_status_counts: { running: 1 },
        tasks: [
          {
            ...outcome.verification_lifecycle.tasks[0],
            launch_status: 'running',
            job_status: 'running',
            agent_job_id: 'verifier-job',
          },
        ],
      },
    });
    renderPanel();

    await waitFor(() =>
      expect(apiClient.createAgentJobProgressWebSocket).toHaveBeenCalledWith('verifier-job')
    );
    expect(socket.onmessage).not.toBeNull();
    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({ type: 'progress', status: 'completed' }),
      });
    });

    await waitFor(() => expect(apiClient.getAutonomousRndJobOutcome.mock.calls.length).toBeGreaterThan(1));
    expect(socket.close).toHaveBeenCalled();
  });

  it('still offers CompOps evidence import when the job has no verification tasks', async () => {
    apiClient.getAutonomousRndJobOutcome.mockResolvedValue({
      ...outcome,
      verification_lifecycle: {
        task_count: 0,
        launch_status_counts: {},
        evidence_status_counts: {},
        tasks: [],
        timeline: [],
      },
    });
    renderPanel();

    await waitFor(() => expect(apiClient.getAutonomousRndJobOutcome).toHaveBeenCalled());
    expect(screen.queryByRole('heading', { name: 'Evidence verification' })).not.toBeInTheDocument();
    expect(
      await screen.findByRole('heading', { name: 'Import CompOps evidence' })
    ).toBeInTheDocument();
    expect(screen.getByText('Configure CompOps in Tools')).toHaveAttribute(
      'href',
      '/tools'
    );
  });
});
