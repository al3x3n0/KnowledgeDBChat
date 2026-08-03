import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import AgentControlPlanePage from '../AgentControlPlanePage';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    getAgentControlRuns: jest.fn(),
    getAgentControlRun: jest.fn(),
    getAgentControlReviews: jest.fn(),
    listAgentControlRunViews: jest.fn(),
    createAgentControlRunView: jest.fn(),
    updateAgentControlRunView: jest.fn(),
    deleteAgentControlRunView: jest.fn(),
    actOnAgentControlReview: jest.fn(),
    bulkActOnAgentControlReview: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;

const makeRun = (overrides: Record<string, any> = {}) => ({
  id: 'job:run-1',
  source_type: 'job',
  title: 'Root control run',
  subtitle: 'Track compiler regressions',
  status: 'running',
  outcome: 'running',
  created_at: '2026-04-21T12:00:00Z',
  started_at: '2026-04-21T12:05:00Z',
  completed_at: null,
  root_job_id: 'run-1',
  workflow_execution_id: null,
  child_job_count: 1,
  child_execution_count: 0,
  linked_note_count: 1,
  linked_experiment_count: 1,
  decision_count: 2,
  replayability_status: 'full_lineage',
  automation_profile: 'balanced',
  queued_operator_review_count: 1,
  queued_operator_reviews_by_type: { follow_up_recommendation: 1 },
  routing: {
    provider: 'openai',
    model: 'gpt-5.4',
    routing_tier: 'balanced',
    requested_tier: 'balanced',
    request_count: 2,
    summary: 'balanced / openai / gpt-5.4',
  },
  ...overrides,
});

const makeDetail = (overrides: Record<string, any> = {}) => ({
  run: makeRun(),
  nodes: [
    {
      id: 'job:run-1',
      kind: 'agent_job',
      label: 'Root control run',
      status: 'running',
      stage: 'planner',
      timestamp: '2026-04-21T12:00:00Z',
      metadata: { goal: 'Track compiler regressions' },
    },
    {
      id: 'event:event-1',
      kind: 'decision_event',
      label: 'Validation launched',
      status: 'completed',
      stage: 'router',
      timestamp: '2026-04-21T12:10:00Z',
      metadata: {
        decision_type: 'materialize_experiment',
        routing_experiment_id: 'exp-1',
        routing_experiment_variant_id: 'var-1',
        synthesis_job_id: 'syn-1',
      },
    },
    {
      id: 'workflow:wf-exec-1',
      kind: 'workflow_execution',
      label: 'Validation workflow execution',
      status: 'completed',
      stage: 'executor',
      timestamp: '2026-04-21T12:12:00Z',
      metadata: { workflow_execution_id: 'wf-exec-1', workflow_id: 'workflow-1', trigger_type: 'manual' },
    },
  ],
  edges: [
    { source: 'job:run-1', target: 'event:event-1', relation: 'emits_decision', metadata: {} },
    { source: 'event:event-1', target: 'workflow:wf-exec-1', relation: 'executes_workflow', metadata: {} },
  ],
  decision_trace: [
    {
      event_id: 'event-1',
      event_type: 'run_progress',
      event_time: '2026-04-21T12:10:00Z',
      source_kind: 'agent_job',
      source_id: 'run-1',
      source_label: 'Root control run',
      customer: null,
      decision_type: 'materialize_experiment',
      reason_code: null,
      reason_label: null,
      scheduler_state: null,
      status: 'completed',
      severity: null,
      actor_mode: 'autonomous',
      summary: 'Validation launched',
      operator_note: null,
      before_state: null,
      after_state: null,
      deep_link: { target_tab: 'domain', job_id: null, params: { profileId: 'profile-1', opportunityId: 'opp-1' }, label: 'Open domain' },
      metadata: null,
      is_derived: false,
      record_origin: 'persisted',
    },
  ],
  memory_graph: {
    nodes: [{ id: 'mem-1', type: 'lesson', content: 'retry', importance_score: 0.9, tags: [] }],
    edges: [],
    stats: { memory_count: 1, edge_count: 0, job_count: 1 },
    job_id: 'run-1',
  },
  routing: {
    provider: 'openai',
    model: 'gpt-5.4',
    routing_tier: 'balanced',
    requested_tier: 'balanced',
    request_count: 2,
    summary: 'balanced / openai / gpt-5.4',
  },
  replay: {
    replayability_status: 'full_lineage',
    planner_summary: 'Planner summary',
    router_summary: 'Router summary',
    executor_summary: 'Executor summary',
    ended_at: null,
  },
  related_links: [{ label: 'Autonomous Agents', path: '/autonomous-agents?job=run-1' }],
  queued_operator_review_count: 1,
  queued_operator_reviews: [
    {
      review_type: 'follow_up_recommendation',
      review_status: 'queued',
      reason_code: 'follow_up_launch_approval',
      reason_label: 'Follow-up launch approval',
      source_kind: 'profile',
      source_id: 'profile-1',
      opportunity_id: 'opp-1',
      title: 'Compiler follow-up launch',
      action_path: '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-1',
      queue_path: '/autonomous-agents?tab=queue&queue_item_type=follow_up_recommendation&queue_health_drilldown=pending_follow_up_approvals&profileId=profile-1&opportunityId=opp-1',
      note_path: '/research-notes?note=note-1',
      synthesis_path: null,
      item_type: 'follow_up_recommendation',
      queue_item_key: 'profile::profile-1::opp-1::follow_up_recommendation',
      follow_up_launch_status: 'pending_approval',
      follow_up_review_status: 'pending_approval',
      follow_up_recommendation_key: 'compiler-follow-up-1',
      recommendation_score: 0.91,
      available_actions: ['approve_follow_up', 'reject_follow_up'],
      can_acknowledge: false,
      can_approve: true,
      can_reject: true,
      can_defer: false,
      metadata: {},
    },
  ],
  policy_summary: { effective_policy: { autonomy: 'balanced' } },
  metadata: {},
  ...overrides,
});

const renderPage = (initialEntry = '/agent-control-plane') => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });

  return render(
    <MemoryRouter
      initialEntries={[initialEntry]}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <QueryClientProvider client={queryClient}>
        <AgentControlPlanePage />
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe('AgentControlPlanePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listAgentControlRunViews.mockResolvedValue({ items: [], total: 0 });
    apiClient.createAgentControlRunView.mockImplementation(async (payload: Record<string, any>) => ({
      id: 'view-1',
      user_id: 'user-1',
      name: payload.name,
      filters: payload.filters,
      is_default: Boolean(payload.is_default),
      created_at: '2026-04-21T12:00:00Z',
      updated_at: '2026-04-21T12:00:00Z',
    }));
    apiClient.updateAgentControlRunView.mockImplementation(async (_viewId: string, payload: Record<string, any>) => ({
      id: 'view-1',
      user_id: 'user-1',
      name: payload.name || 'Saved View',
      filters: payload.filters || {},
      is_default: Boolean(payload.is_default),
      created_at: '2026-04-21T12:00:00Z',
      updated_at: '2026-04-21T12:00:00Z',
    }));
    apiClient.deleteAgentControlRunView.mockResolvedValue(undefined);
    apiClient.actOnAgentControlReview.mockResolvedValue({
      ok: true,
      action: 'approve_follow_up',
      review_type: 'follow_up_recommendation',
      source_kind: 'profile',
      source_id: 'profile-1',
      opportunity_id: 'opp-1',
      detail: 'Follow-up launched from queue approval',
      follow_up_launch_status: 'launched',
      follow_up_operator_decision: 'approved_launch',
      follow_up_job_id: 'job-follow-up-1',
    });
    apiClient.bulkActOnAgentControlReview.mockResolvedValue({
      ok: true,
      item_type: 'approval_checkpoint',
      action: 'approve',
      requested_count: 1,
      applied: 1,
      failed: 0,
      results: [{ item_key: 'approval:job-approval-1', job_id: 'job-approval-1', ok: true }],
    });
  });

  test('renders run list and auto-selects the first visible run', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun()],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());
    apiClient.getAgentControlReviews.mockResolvedValue({ items: [], total: 0 });

    renderPage();

    expect(await screen.findByText('Root control run')).toBeInTheDocument();
    await waitFor(() =>
      expect(apiClient.getAgentControlRun.mock.calls.some(([runId]: [string]) => runId === 'job:run-1')).toBe(true)
    );
    expect(await screen.findByText('Planner summary')).toBeInTheDocument();
    expect(screen.getByText('Autonomous Agents')).toBeInTheDocument();
    expect(screen.getByText('1 queued reviews')).toBeInTheDocument();
  });

  test('reselects the first filtered run when the current selection is no longer visible', async () => {
    apiClient.getAgentControlRuns.mockImplementation(async (params?: { source_type?: string }) => {
      if (params?.source_type === 'job') {
        return { items: [makeRun()], total: 1 };
      }
      return {
        items: [
          makeRun(),
          makeRun({
            id: 'workflow:wf-1',
            source_type: 'workflow',
            title: 'Workflow control run',
            root_job_id: null,
            workflow_execution_id: 'wf-1',
            child_job_count: 0,
            child_execution_count: 1,
            routing: null,
          }),
        ],
        total: 2,
      };
    });
    apiClient.getAgentControlRun.mockImplementation(async (runId: string) => {
      if (runId === 'job:run-1') return makeDetail();
      return makeDetail({
        run: makeRun({
          id: 'workflow:wf-1',
          source_type: 'workflow',
          title: 'Workflow control run',
          root_job_id: null,
          workflow_execution_id: 'wf-1',
        }),
      });
    });

    renderPage('/agent-control-plane?type=job&run=workflow:wf-1');

    await waitFor(() =>
      expect(apiClient.getAgentControlRun.mock.calls.some(([runId]: [string]) => runId === 'job:run-1')).toBe(true)
    );
    expect(await screen.findByText('Track compiler regressions')).toBeInTheDocument();
    expect(screen.getAllByText('Root control run').length).toBeGreaterThan(0);
  });

  test('updates the node inspector when a graph node is selected', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun()],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());

    renderPage();

    expect(await screen.findByText('Planner summary')).toBeInTheDocument();
    const nodeButton = await screen.findByRole('button', { name: /Validation launched/i });
    fireEvent.click(nodeButton);

    expect(await screen.findByText('decision event')).toBeInTheDocument();
    expect(screen.getAllByText('Validation launched').length).toBeGreaterThan(1);
    expect(screen.getAllByText(/materialize_experiment/i).length).toBeGreaterThan(0);
    expect(screen.getByText('Open routing slice')).toBeInTheDocument();
    expect(screen.getByText('Open in Synthesis')).toBeInTheDocument();
    expect(screen.getAllByText('Validation workflow execution').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Router').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Executor').length).toBeGreaterThan(0);
    expect(screen.getByText('Operator review queue')).toBeInTheDocument();
    expect(screen.getByText('Approve')).toBeInTheDocument();
    expect(screen.getByText('Open in queue')).toBeInTheDocument();
    expect(screen.getByText('Open context')).toBeInTheDocument();
  });

  test('approves a queued follow-up review inline', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun()],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());

    renderPage();

    expect(await screen.findByText('Operator review queue')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('Optional note for the approval decision'), {
      target: { value: 'Looks good to launch.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() =>
      expect(apiClient.actOnAgentControlReview).toHaveBeenCalledWith({
        review_type: 'follow_up_recommendation',
        source_kind: 'profile',
        source_id: 'profile-1',
        opportunity_id: 'opp-1',
        action: 'approve_follow_up',
        operator_note: 'Looks good to launch.',
      })
    );
  });

  test('launches a manual follow-up recommendation inline', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { manual_follow_up_recommendation: 1 } })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_reviews: [
          {
            review_type: 'manual_follow_up_recommendation',
            review_status: 'queued',
            reason_code: 'manual_follow_up_recommendation',
            reason_label: 'Manual follow-up recommendation',
            source_kind: 'profile',
            source_id: 'profile-1',
            opportunity_id: 'opp-manual-1',
            title: 'Manual compiler follow-up',
            action_path: '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-manual-1',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=follow_up_recommendation&queue_health_drilldown=manual_follow_up_recommendations&profileId=profile-1&opportunityId=opp-manual-1',
            note_path: '/research-notes?note=note-1',
            synthesis_path: null,
            item_type: 'follow_up_recommendation',
            queue_item_key: 'profile::profile-1::opp-manual-1::manual_follow_up_recommendation',
            follow_up_launch_status: null,
            follow_up_review_status: 'manual_recommendation',
            follow_up_recommendation_key: 'manual-follow-up-1',
            recommendation_score: 0.77,
            available_actions: ['launch_follow_up'],
            can_acknowledge: false,
            can_approve: false,
            can_reject: false,
            can_defer: false,
            can_launch_follow_up: true,
            can_relaunch_follow_up: false,
            metadata: {},
          },
        ],
      })
    );
    apiClient.actOnAgentControlReview.mockResolvedValueOnce({
      ok: true,
      action: 'launch_follow_up',
      review_type: 'manual_follow_up_recommendation',
      source_kind: 'profile',
      source_id: 'profile-1',
      opportunity_id: 'opp-manual-1',
      detail: 'Follow-up launched from manual recommendation',
      follow_up_launch_status: 'launched',
      follow_up_operator_decision: 'approved_launch',
      follow_up_job_id: 'job-follow-up-2',
    });

    renderPage();

    expect(await screen.findByText('Operator review queue')).toBeInTheDocument();
    expect(screen.getByText('Open in queue')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('Optional note for the approval decision'), {
      target: { value: 'Operator requested manual launch.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Launch follow-up' }));

    await waitFor(() =>
      expect(apiClient.actOnAgentControlReview).toHaveBeenCalledWith({
        review_type: 'manual_follow_up_recommendation',
        source_kind: 'profile',
        source_id: 'profile-1',
        opportunity_id: 'opp-manual-1',
        action: 'launch_follow_up',
        operator_note: 'Operator requested manual launch.',
      })
    );
  });

  test('renders approval checkpoint metadata and approves it inline', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { approval_checkpoint: 1 }, queued_operator_review_count: 1 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 1,
        queued_operator_reviews: [
          {
            review_type: 'approval_checkpoint',
            review_status: 'queued',
            reason_code: 'approval_required',
            reason_label: 'Approval required',
            source_kind: 'job',
            source_id: 'job-approval-1',
            opportunity_id: 'job-approval-1',
            title: 'Approval-gated validation',
            summary: 'Approve the next compiler validation step.',
            evidence_summary: 'Diff and benchmark summary ready.',
            status: 'paused',
            customer: 'compiler',
            job_id: 'job-approval-1',
            job_name: 'Approval-gated validation',
            job_type: 'analysis',
            age_minutes: 125,
            priority_score: 118,
            sla_bucket: 'at_risk',
            escalation_level: 'medium',
            action_path: '/autonomous-agents?job=job-approval-1',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=approval_checkpoint&queue_job=job-approval-1',
            note_path: null,
            synthesis_path: null,
            item_type: 'approval_checkpoint',
            queue_item_key: 'approval:job-approval-1',
            checkpoint: {
              action: {
                tool: 'web.search',
                purpose: 'Find compiler regressions',
                params: { q: 'compiler regression' },
              },
            },
            checkpoint_action_draft: {
              tool: 'web.search',
              purpose: 'Find compiler regressions',
              params: { q: 'compiler regression' },
            },
            scheduler_state: { queue_reason: 'approval_required' },
            available_actions: ['approve', 'edit', 'reject', 'skip'],
            can_acknowledge: false,
            can_approve: true,
            can_reject: true,
            can_defer: false,
            can_launch_follow_up: false,
            can_relaunch_follow_up: false,
            can_skip: true,
            can_restart: false,
            can_resume: false,
            can_cancel: false,
            metadata: {},
          },
        ],
      })
    );
    apiClient.actOnAgentControlReview.mockResolvedValueOnce({
      ok: true,
      action: 'approve',
      review_type: 'approval_checkpoint',
      source_kind: 'job',
      source_id: 'job-approval-1',
      opportunity_id: 'job-approval-1',
      detail: 'approval checkpoint action applied',
      follow_up_job_id: 'job-approval-1',
    });

    renderPage('/agent-control-plane?reviewType=approval_checkpoint');

    expect(await screen.findByText('Approval-gated validation')).toBeInTheDocument();
    expect(screen.getByText(/Pending tool:/i)).toBeInTheDocument();
    expect(screen.getByText('Open in queue')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('Optional note for the approval decision'), {
      target: { value: 'Approve this step.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() =>
      expect(apiClient.actOnAgentControlReview).toHaveBeenCalledWith({
        review_type: 'approval_checkpoint',
        source_kind: 'job',
        source_id: 'job-approval-1',
        opportunity_id: 'job-approval-1',
        action: 'approve',
        operator_note: 'Approve this step.',
      })
    );
    expect(screen.getByRole('button', { name: 'Skip' })).toBeInTheDocument();
  });

  test('edits an approval checkpoint inline before approving', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { approval_checkpoint: 1 }, queued_operator_review_count: 1 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 1,
        queued_operator_reviews: [
          {
            review_type: 'approval_checkpoint',
            review_status: 'queued',
            reason_code: 'approval_required',
            reason_label: 'Approval required',
            source_kind: 'job',
            source_id: 'job-approval-2',
            opportunity_id: 'job-approval-2',
            title: 'Editable approval checkpoint',
            status: 'paused',
            job_id: 'job-approval-2',
            action_path: '/autonomous-agents?job=job-approval-2',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=approval_checkpoint&queue_job=job-approval-2',
            item_type: 'approval_checkpoint',
            queue_item_key: 'approval:job-approval-2',
            checkpoint: {
              action: {
                tool: 'web.search',
                purpose: 'Find regressions',
                params: { q: 'compiler regression' },
              },
            },
            checkpoint_action_draft: {
              tool: 'web.search',
              purpose: 'Find regressions',
              params: { q: 'compiler regression' },
            },
            available_actions: ['approve', 'edit', 'reject', 'skip'],
            can_approve: true,
            can_reject: true,
            can_skip: true,
            metadata: {},
          },
        ],
      })
    );
    apiClient.actOnAgentControlReview.mockResolvedValueOnce({
      ok: true,
      action: 'edit',
      review_type: 'approval_checkpoint',
      source_kind: 'job',
      source_id: 'job-approval-2',
      opportunity_id: 'job-approval-2',
      detail: 'approval checkpoint action applied',
      follow_up_job_id: 'job-approval-2',
    });

    renderPage('/agent-control-plane?reviewType=approval_checkpoint');

    expect(await screen.findByText('Editable approval checkpoint')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Checkpoint tool Editable approval checkpoint'), { target: { value: 'bash.exec' } });
    fireEvent.change(screen.getByLabelText('Checkpoint purpose Editable approval checkpoint'), { target: { value: 'Run a narrower command' } });
    fireEvent.change(screen.getByLabelText('Checkpoint params Editable approval checkpoint'), {
      target: { value: '{\n  "cmd": "make test"\n}' },
    });
    fireEvent.change(screen.getByPlaceholderText('Optional note for the approval decision'), {
      target: { value: 'Edit this before approval.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Edit + approve' }));

    await waitFor(() =>
      expect(apiClient.actOnAgentControlReview).toHaveBeenCalledWith({
        review_type: 'approval_checkpoint',
        source_kind: 'job',
        source_id: 'job-approval-2',
        opportunity_id: 'job-approval-2',
        action: 'edit',
        operator_note: 'Edit this before approval.',
        checkpoint_action_patch: {
          tool: 'bash.exec',
          purpose: 'Run a narrower command',
          params: { cmd: 'make test' },
        },
      })
    );
  });

  test('rejects invalid approval checkpoint edit JSON locally', async () => {
    const toast = require('react-hot-toast').default;
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { approval_checkpoint: 1 }, queued_operator_review_count: 1 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 1,
        queued_operator_reviews: [
          {
            review_type: 'approval_checkpoint',
            review_status: 'queued',
            reason_code: 'approval_required',
            reason_label: 'Approval required',
            source_kind: 'job',
            source_id: 'job-approval-3',
            opportunity_id: 'job-approval-3',
            title: 'Broken approval checkpoint edit',
            item_type: 'approval_checkpoint',
            queue_item_key: 'approval:job-approval-3',
            checkpoint: {
              action: {
                tool: 'web.search',
                purpose: 'Find regressions',
                params: { q: 'compiler regression' },
              },
            },
            checkpoint_action_draft: {
              tool: 'web.search',
              purpose: 'Find regressions',
              params: { q: 'compiler regression' },
            },
            available_actions: ['approve', 'edit', 'reject', 'skip'],
            can_approve: true,
            can_reject: true,
            can_skip: true,
            metadata: {},
          },
        ],
      })
    );

    renderPage('/agent-control-plane?reviewType=approval_checkpoint');

    expect(await screen.findByText('Broken approval checkpoint edit')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Checkpoint params Broken approval checkpoint edit'), {
      target: { value: '{ invalid json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Edit + approve' }));

    expect(apiClient.actOnAgentControlReview).not.toHaveBeenCalled();
    expect(toast.error).toHaveBeenCalled();
  });

  test('renders job recovery metadata and restarts it inline', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { job_recovery: 1 }, queued_operator_review_count: 1 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 1,
        queued_operator_reviews: [
          {
            review_type: 'job_recovery',
            review_status: 'queued',
            reason_code: 'execution_failure',
            reason_label: 'Execution failure',
            source_kind: 'job',
            source_id: 'job-recovery-1',
            opportunity_id: 'job-recovery-1',
            title: 'Recover validation monitor',
            summary: 'Recurring validation needs recovery.',
            status: 'failed',
            customer: 'compiler',
            job_id: 'job-recovery-1',
            job_name: 'Recover validation monitor',
            job_type: 'monitor',
            age_minutes: 180,
            priority_score: 132,
            sla_bucket: 'overdue',
            escalation_level: 'high',
            action_path: '/autonomous-agents?job=job-recovery-1',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=job_recovery&queue_job=job-recovery-1',
            note_path: null,
            synthesis_path: null,
            item_type: 'job_recovery',
            queue_item_key: 'recovery:job-recovery-1',
            checkpoint: null,
            scheduler_state: {
              queue_reason: 'execution_failure',
              retry_count: 3,
              last_error: 'Tool timeout',
            },
            available_actions: ['restart', 'resume', 'cancel'],
            can_acknowledge: false,
            can_approve: false,
            can_reject: false,
            can_defer: false,
            can_launch_follow_up: false,
            can_relaunch_follow_up: false,
            can_skip: false,
            can_restart: true,
            can_resume: true,
            can_cancel: true,
            metadata: {},
          },
        ],
      })
    );
    apiClient.actOnAgentControlReview.mockResolvedValueOnce({
      ok: true,
      action: 'restart',
      review_type: 'job_recovery',
      source_kind: 'job',
      source_id: 'job-recovery-1',
      opportunity_id: 'job-recovery-1',
      detail: 'job recovery action applied',
      follow_up_job_id: 'job-recovery-1',
    });

    renderPage('/agent-control-plane?reviewType=job_recovery');

    expect(await screen.findByText('Recover validation monitor')).toBeInTheDocument();
    expect(screen.getByText(/Reason: execution failure/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Restart' })).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('Optional note for the approval decision'), {
      target: { value: 'Retry with the latest config.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Restart' }));

    await waitFor(() =>
      expect(apiClient.actOnAgentControlReview).toHaveBeenCalledWith({
        review_type: 'job_recovery',
        source_kind: 'job',
        source_id: 'job-recovery-1',
        opportunity_id: 'job-recovery-1',
        action: 'restart',
        operator_note: 'Retry with the latest config.',
      })
    );
    expect(screen.getByRole('button', { name: 'Resume' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
  });

  test('bulk-approves selected approval checkpoints', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { approval_checkpoint: 2 }, queued_operator_review_count: 2 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 2,
        queued_operator_reviews: [
          {
            review_type: 'approval_checkpoint',
            review_status: 'queued',
            reason_code: 'approval_required',
            reason_label: 'Approval required',
            source_kind: 'job',
            source_id: 'job-approval-1',
            opportunity_id: 'job-approval-1',
            title: 'Approval one',
            item_type: 'approval_checkpoint',
            queue_item_key: 'approval:job-approval-1',
            job_id: 'job-approval-1',
            available_actions: ['approve', 'reject', 'skip'],
            can_approve: true,
            can_reject: true,
            can_skip: true,
            metadata: {},
          },
          {
            review_type: 'approval_checkpoint',
            review_status: 'queued',
            reason_code: 'approval_required',
            reason_label: 'Approval required',
            source_kind: 'job',
            source_id: 'job-approval-2',
            opportunity_id: 'job-approval-2',
            title: 'Approval two',
            item_type: 'approval_checkpoint',
            queue_item_key: 'approval:job-approval-2',
            job_id: 'job-approval-2',
            available_actions: ['approve', 'reject', 'skip'],
            can_approve: true,
            can_reject: true,
            can_skip: true,
            metadata: {},
          },
        ],
      })
    );
    apiClient.bulkActOnAgentControlReview.mockResolvedValueOnce({
      ok: true,
      item_type: 'approval_checkpoint',
      action: 'approve',
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { item_key: 'approval:job-approval-1', job_id: 'job-approval-1', ok: true },
        { item_key: 'approval:job-approval-2', job_id: 'job-approval-2', ok: true },
      ],
    });

    renderPage();

    expect(await screen.findByText('Operator review queue')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Select control-plane review Approval one'));
    fireEvent.click(screen.getByLabelText('Select control-plane review Approval two'));
    fireEvent.change(screen.getByPlaceholderText('Shared note for selected approvals'), {
      target: { value: 'Approve these together.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'approve' }));

    await waitFor(() =>
      expect(apiClient.bulkActOnAgentControlReview).toHaveBeenCalledWith({
        item_type: 'approval_checkpoint',
        action: 'approve',
        job_ids: ['job-approval-1', 'job-approval-2'],
        domain_research_profile_id: null,
        profile_opportunity_ids: [],
        portfolio_id: null,
        portfolio_opportunity_ids: [],
        operator_note: 'Approve these together.',
      })
    );
  });

  test('bulk-approves selected pending follow-up reviews within one owner', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { follow_up_recommendation: 2 }, queued_operator_review_count: 2 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 2,
        queued_operator_reviews: [
          {
            review_type: 'follow_up_recommendation',
            review_status: 'queued',
            reason_code: 'follow_up_launch_approval',
            reason_label: 'Follow-up launch approval',
            source_kind: 'profile',
            source_id: 'profile-1',
            opportunity_id: 'opp-1',
            title: 'Follow-up one',
            item_type: 'follow_up_recommendation',
            queue_item_key: 'profile::profile-1::opp-1::follow_up_recommendation',
            follow_up_launch_status: 'pending_approval',
            available_actions: ['approve_follow_up', 'reject_follow_up'],
            can_approve: true,
            can_reject: true,
            metadata: {},
          },
          {
            review_type: 'follow_up_recommendation',
            review_status: 'queued',
            reason_code: 'follow_up_launch_approval',
            reason_label: 'Follow-up launch approval',
            source_kind: 'profile',
            source_id: 'profile-1',
            opportunity_id: 'opp-2',
            title: 'Follow-up two',
            item_type: 'follow_up_recommendation',
            queue_item_key: 'profile::profile-1::opp-2::follow_up_recommendation',
            follow_up_launch_status: 'pending_approval',
            available_actions: ['approve_follow_up', 'reject_follow_up'],
            can_approve: true,
            can_reject: true,
            metadata: {},
          },
        ],
      })
    );
    apiClient.bulkActOnAgentControlReview.mockResolvedValueOnce({
      ok: true,
      item_type: 'follow_up_recommendation',
      action: 'approve_launch',
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { opportunity_id: 'opp-1', ok: true, follow_up_launch_status: 'launched' },
        { opportunity_id: 'opp-2', ok: true, follow_up_launch_status: 'launched' },
      ],
    });

    renderPage();

    expect(await screen.findByText('Operator review queue')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Select control-plane review Follow-up one'));
    fireEvent.click(screen.getByLabelText('Select control-plane review Follow-up two'));
    fireEvent.change(screen.getByPlaceholderText('Shared note for selected follow-up approvals'), {
      target: { value: 'Approve both launches.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

    await waitFor(() =>
      expect(apiClient.bulkActOnAgentControlReview).toHaveBeenCalledWith({
        item_type: 'follow_up_recommendation',
        action: 'approve_launch',
        job_ids: [],
        domain_research_profile_id: 'profile-1',
        profile_opportunity_ids: ['opp-1', 'opp-2'],
        portfolio_id: null,
        portfolio_opportunity_ids: [],
        operator_note: 'Approve both launches.',
      })
    );
  });

  test('renders useful fallbacks when routing and memory are missing and detail fails', async () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ id: 'workflow:wf-1', source_type: 'workflow', title: 'Sparse workflow', routing: null })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockRejectedValue(new Error('not found'));

    try {
      renderPage('/agent-control-plane?run=workflow:wf-1');

      expect(await screen.findByText('This control run could not be loaded.')).toBeInTheDocument();
      expect(consoleErrorSpy).toHaveBeenCalled();
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });

  test('applies the default saved view when no explicit URL filters are present', async () => {
    apiClient.listAgentControlRunViews.mockResolvedValue({
      items: [
        {
          id: 'view-default',
          user_id: 'user-1',
          name: 'Default running jobs',
          filters: {
            source_type: 'job',
            outcome: 'running',
            selected_run_id: 'job:run-1',
            has_operator_review: 'true',
            queue_health_drilldown: 'pending_follow_up_approvals',
          },
          is_default: true,
          created_at: '2026-04-21T12:00:00Z',
          updated_at: '2026-04-21T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.getAgentControlRuns.mockResolvedValue({ items: [makeRun()], total: 1 });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());

    renderPage();

    await waitFor(() =>
      expect(apiClient.getAgentControlRun.mock.calls.some(([runId]: [string]) => runId === 'job:run-1')).toBe(true)
    );
    expect(await screen.findByDisplayValue('Default running jobs')).toBeInTheDocument();
    expect(apiClient.getAgentControlRuns).toHaveBeenCalledWith(expect.objectContaining({ has_operator_review: true }));
    expect(await screen.findByText(/Showing pending approvals slice/i)).toBeInTheDocument();
  });

  test('saves and deletes a control-plane view from current filters', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({ items: [makeRun()], total: 1 });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());

    renderPage('/agent-control-plane?type=job&outcome=running&run=job:run-1&queueStatus=pending_approval&queueHealthDrilldown=pending_follow_up_approvals');

    const nameInput = await screen.findByLabelText('Control-plane view name');
    fireEvent.change(nameInput, {
      target: { value: 'My active jobs' },
    });
    await waitFor(() => expect((nameInput as HTMLInputElement).value).toBe('My active jobs'));
    fireEvent.click(screen.getByRole('button', { name: 'Save Current View' }));

    await waitFor(() =>
      expect(apiClient.createAgentControlRunView).toHaveBeenCalledWith({
        name: 'My active jobs',
        filters: {
          source_type: 'job',
          outcome: 'running',
          selected_run_id: 'job:run-1',
          queue_status: 'pending_approval',
          queue_health_drilldown: 'pending_follow_up_approvals',
        },
        is_default: false,
      })
    );

    fireEvent.click(screen.getByRole('button', { name: 'Delete View' }));

    await waitFor(() => expect(apiClient.deleteAgentControlRunView).toHaveBeenCalledWith('view-1'));
  });

  test('filters the review queue with queue drilldowns and renders blocked context', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun({ queued_operator_reviews_by_type: { follow_up_recommendation: 1, budget_review: 1 }, queued_operator_review_count: 2 })],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 2,
        queued_operator_reviews: [
          {
            ...makeDetail().queued_operator_reviews[0],
            title: 'Pending follow-up approval',
            queue_item_key: 'profile::profile-1::opp-1::follow_up_recommendation',
          },
          {
            review_type: 'budget_review',
            review_status: 'queued',
            reason_code: 'budget_gate',
            reason_label: 'Budget review',
            source_kind: 'profile',
            source_id: 'profile-1',
            opportunity_id: 'opp-budget-1',
            title: 'Blocked by budget',
            item_type: 'budget_review',
            queue_item_key: 'profile::profile-1::opp-budget-1::budget_review',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=budget_review&profileId=profile-1&opportunityId=opp-budget-1',
            action_path: '/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-budget-1',
            follow_up_block_reason: 'Autonomy budgets currently clamp follow-ups to manual mode.',
            follow_up_budget_decision: 'clamped_to_manual',
            follow_up_budget_reason: 'Daily budget exhausted',
            follow_up_customer_budget_decision: 'downgraded_to_queue',
            follow_up_customer_budget_reason: 'Customer monthly budget exhausted',
            available_actions: [],
            metadata: {},
          },
        ],
      })
    );

    renderPage('/agent-control-plane?queueHealthDrilldown=blocked_follow_up');

    expect(await screen.findByText(/Showing blocked follow-ups slice/i)).toBeInTheDocument();
    expect(screen.queryByText('Pending follow-up approval')).not.toBeInTheDocument();
    expect(screen.getByText('Blocked by budget')).toBeInTheDocument();
    expect(screen.getByText(/Blocked: Autonomy budgets currently clamp follow-ups to manual mode./i)).toBeInTheDocument();
    expect(screen.getByText(/Budget: clamped_to_manual/i)).toBeInTheDocument();
    expect(screen.getByText(/Customer budget: downgraded_to_queue/i)).toBeInTheDocument();
  });

  test('renders policy and budget review parity actions and applies a guardrail inline', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [
        makeRun({
          queued_operator_reviews_by_type: { policy_review: 1, budget_review: 1 },
          queued_operator_review_count: 2,
        }),
      ],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(
      makeDetail({
        queued_operator_review_count: 2,
        queued_operator_reviews: [
          {
            review_type: 'policy_review',
            review_status: 'queued',
            reason_code: 'policy_guardrail',
            reason_label: 'Policy review',
            source_kind: 'portfolio',
            source_id: 'fleet-1',
            opportunity_id: 'opp-policy-1',
            title: 'Compiler guardrail review',
            item_type: 'policy_review',
            queue_item_key: 'portfolio::fleet-1::opp-policy-1::policy_review',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=policy_review&fleetId=fleet-1&opportunityId=opp-policy-1&queue_customer=compiler&queue_job=monitor-1',
            action_path: '/autonomous-agents?tab=fleet&fleetId=fleet-1&opportunityId=opp-policy-1',
            customer: 'compiler',
            job_id: 'monitor-1',
            recommended_action: 'apply_guardrail',
            policy_guardrail_action: 'rollback',
            policy_guardrail_target_history_entry_id: 'history-1',
            policy_guardrail_reasons: ['sandbox restriction'],
            policy_rollback_payload: { history_entry_id: 'history-1' },
            available_actions: ['apply_guardrail'],
            metadata: {},
          },
          {
            review_type: 'budget_review',
            review_status: 'queued',
            reason_code: 'budget_throttle',
            reason_label: 'Budget review',
            source_kind: 'portfolio',
            source_id: 'fleet-1',
            opportunity_id: 'opp-budget-1',
            title: 'Compiler budget review',
            item_type: 'budget_review',
            queue_item_key: 'portfolio::fleet-1::opp-budget-1::budget_review',
            queue_path: '/autonomous-agents?tab=queue&queue_item_type=budget_review&fleetId=fleet-1&opportunityId=opp-budget-1&queue_customer=compiler&queue_job=monitor-1',
            action_path: '/autonomous-agents?tab=fleet&fleetId=fleet-1&opportunityId=opp-budget-1',
            customer: 'compiler',
            job_id: 'monitor-1',
            budget_throttle_state: 'paused',
            budget_reason: 'Daily budget exhausted',
            available_actions: [],
            metadata: {},
          },
        ],
      })
    );
    apiClient.actOnAgentControlReview.mockResolvedValue({
      ok: true,
      action: 'apply_guardrail',
      review_type: 'policy_review',
      source_kind: 'portfolio',
      source_id: 'fleet-1',
      opportunity_id: 'opp-policy-1',
      detail: 'Policy safeguard rollback applied',
      monitor_job_id: 'monitor-1',
    });

    renderPage();

    expect(await screen.findByText('Compiler guardrail review')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('Optional note for the approval decision'), {
      target: { value: 'Apply the safeguard.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Apply guardrail' }));

    await waitFor(() =>
      expect(apiClient.actOnAgentControlReview).toHaveBeenCalledWith(
        expect.objectContaining({
          review_type: 'policy_review',
          source_kind: 'portfolio',
          source_id: 'fleet-1',
          opportunity_id: 'opp-policy-1',
          action: 'apply_guardrail',
          operator_note: 'Apply the safeguard.',
        })
      )
    );

    expect(screen.getByRole('link', { name: 'Compare before/after' })).toHaveAttribute(
      'href',
      '/autonomous-agents?tab=health&health_customer=compiler&health_monitor=monitor-1&health_policy_history=history-1'
    );
    expect(screen.getAllByRole('link', { name: 'Open monitor' })[0]).toHaveAttribute(
      'href',
      '/autonomous-agents?tab=health&health_customer=compiler&health_monitor=monitor-1'
    );
    expect(screen.getAllByRole('link', { name: 'Open fleet' }).length).toBe(2);
  });

  test('renders a global queue and can switch back into the owning run', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [
        makeRun(),
        makeRun({
          id: 'workflow:wf-1',
          source_type: 'workflow',
          title: 'Workflow control run',
          root_job_id: null,
          workflow_execution_id: 'wf-1',
          queued_operator_review_count: 1,
          queued_operator_reviews_by_type: { policy_review: 1 },
        }),
      ],
      total: 2,
    });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());
    apiClient.getAgentControlReviews.mockResolvedValue({
      items: [
        {
          ...makeDetail().queued_operator_reviews[0],
          queue_item_key: 'profile::profile-1::opp-1::follow_up_recommendation',
          run_id: 'job:run-1',
          run_title: 'Root control run',
          run_source_type: 'job',
          run_status: 'running',
        },
        {
          ...makeDetail().queued_operator_reviews[0],
          review_type: 'policy_review',
          queue_item_key: 'portfolio::fleet-1::opp-2::policy_review',
          title: 'Workflow guardrail review',
          source_kind: 'portfolio',
          source_id: 'fleet-1',
          opportunity_id: 'opp-2',
          item_type: 'policy_review',
          available_actions: ['apply_guardrail'],
          run_id: 'workflow:wf-1',
          run_title: 'Workflow control run',
          run_source_type: 'workflow',
          run_status: 'running',
        },
      ],
      total: 2,
      summary: {
        total: 2,
        by_type: { follow_up_recommendation: 1, policy_review: 1 },
        by_sla_bucket: { at_risk: 1 },
        by_status: { queued: 2 },
        by_customer: { compiler: 2 },
        by_escalation: { medium: 1 },
      },
      offset: 0,
      limit: 50,
      has_more: false,
    });

    renderPage('/agent-control-plane?queueScope=global');

    expect(await screen.findByText('Global queue')).toBeInTheDocument();
    expect(await screen.findByText('Workflow guardrail review')).toBeInTheDocument();
    expect(screen.getByText(/Run: Workflow control run/i)).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('button', { name: 'Open run' })[1]);

    await waitFor(() =>
      expect(apiClient.getAgentControlRun.mock.calls.some(([runId]: [string]) => runId === 'workflow:wf-1')).toBe(true)
    );
  });

  test('loads more global queue items and preserves sort selection', async () => {
    apiClient.getAgentControlRuns.mockResolvedValue({
      items: [makeRun()],
      total: 1,
    });
    apiClient.getAgentControlRun.mockResolvedValue(makeDetail());
    apiClient.getAgentControlReviews.mockImplementation(async (params?: { offset?: number; sort?: string }) => {
      if ((params?.offset || 0) === 50) {
        return {
          items: [
            {
              ...makeDetail().queued_operator_reviews[0],
              queue_item_key: 'profile::profile-1::opp-2::follow_up_recommendation',
              title: 'Older global follow-up',
              opportunity_id: 'opp-2',
              run_id: 'job:run-1',
              run_title: 'Root control run',
              run_source_type: 'job',
              run_status: 'running',
            },
          ],
          total: 2,
          summary: {
            total: 2,
            by_type: { follow_up_recommendation: 2 },
            by_sla_bucket: { at_risk: 1 },
            by_status: { queued: 2 },
            by_customer: { compiler: 2 },
            by_escalation: { medium: 1 },
          },
          offset: 50,
          limit: 50,
          has_more: false,
        };
      }
      return {
        items: [
          {
            ...makeDetail().queued_operator_reviews[0],
            queue_item_key: 'profile::profile-1::opp-1::follow_up_recommendation',
            title: 'Newest global follow-up',
            run_id: 'job:run-1',
            run_title: 'Root control run',
            run_source_type: 'job',
            run_status: 'running',
          },
        ],
        total: 2,
        summary: {
          total: 2,
          by_type: { follow_up_recommendation: 2 },
          by_sla_bucket: { at_risk: 1 },
          by_status: { queued: 2 },
          by_customer: { compiler: 2 },
          by_escalation: { medium: 1 },
        },
        offset: 0,
        limit: 50,
        has_more: true,
      };
    });

    renderPage('/agent-control-plane?queueScope=global&queueSort=age_desc');

    expect(await screen.findByText('Newest global follow-up')).toBeInTheDocument();
    expect(screen.getByLabelText('Global queue sort')).toHaveValue('age_desc');
    fireEvent.click(screen.getByRole('button', { name: 'Load more' }));

    expect(await screen.findByText('Older global follow-up')).toBeInTheDocument();
    await waitFor(() =>
      expect(apiClient.getAgentControlReviews).toHaveBeenCalledWith(
        expect.objectContaining({
          sort: 'age_desc',
          offset: 50,
          limit: 50,
        })
      )
    );
  });
});
