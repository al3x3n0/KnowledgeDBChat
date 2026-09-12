/**
 * What a run has to say about itself while it is still happening.
 *
 * These are the three collapses the component exists to prevent. Each one is
 * a distinction the API makes and a UI would naturally throw away, and each
 * produces a specific wrong belief in whoever is reading the page:
 *
 *   - a stage that completed WITHOUT meeting its contract shown as a finished
 *     stage: the one state nothing downstream may be built on, painted like
 *     the state it may.
 *   - a run waiting on a PERSON shown as a run that stopped: someone goes
 *     looking for a failure that never happened.
 *   - the stages not yet REACHED left out: two of six looks like the end.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';

import PipelineRunProgress from '../PipelineRunProgress';
import type { PipelineRun, PipelineRunStage } from '../../../types';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    getPipelineRun: jest.fn(),
    restartPipelineStage: jest.fn(),
    insertPipelineStage: jest.fn(),
  },
}));

const { apiClient } = jest.requireMock('../../../services/api');

const stage = (over: Partial<PipelineRunStage>): PipelineRunStage => ({
  stage: 'profile',
  job_id: 'job-1',
  status: 'completed',
  iteration: 3,
  contract_satisfied: true,
  restartable: true,
  goal: 'Find where the kernel spends its time',
  checkpoint: false,
  waiting_on_person: false,
  attempts: 1,
  progress: 100,
  disputed: false,
  ...over,
});

const run = (over: Partial<PipelineRun>): PipelineRun => ({
  root_job_id: 'root-1',
  stages: [stage({})],
  pipeline: 'int8-attention-study',
  status: 'running',
  total_stages: 1,
  completed_stages: 1,
  current_stage: null,
  ...over,
});

const renderRun = () =>
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <PipelineRunProgress rootJobId="root-1" />
    </QueryClientProvider>
  );

afterEach(() => jest.clearAllMocks());

it('counts the stages the run has not reached yet', async () => {
  // The regression this was built for: a view assembled from the stages that
  // have jobs reports a two-stage run at stage two of six.
  apiClient.getPipelineRun.mockResolvedValue(
    run({
      status: 'running',
      total_stages: 4,
      completed_stages: 1,
      current_stage: 'measure',
      stages: [
        stage({ stage: 'profile' }),
        stage({ stage: 'measure', status: 'running', job_id: 'job-2', contract_satisfied: false }),
        stage({ stage: 'attribute', status: 'pending', job_id: '', iteration: 0, attempts: 0 }),
        stage({ stage: 'report', status: 'pending', job_id: '', iteration: 0, attempts: 0 }),
      ],
    })
  );

  renderRun();

  expect(await screen.findByText(/1 of 4 stages/)).toBeInTheDocument();
  expect(screen.getByText('attribute')).toBeInTheDocument();
  expect(screen.getByText('report')).toBeInTheDocument();
});

it('says when a completed stage did not produce what it promised', async () => {
  // A stage can run out of iterations and still be `completed`. Shown as a
  // plain finished stage, the benchmark built on it looks legitimate.
  apiClient.getPipelineRun.mockResolvedValue(
    run({
      status: 'completed_unmet',
      stages: [stage({ stage: 'implement', contract_satisfied: false })],
    })
  );

  renderRun();

  expect(await screen.findByText('contract unmet')).toBeInTheDocument();
  expect(screen.getByText(/contracts unmet/i)).toBeInTheDocument();
});

it('tells a run waiting on a person from a run that stopped', async () => {
  apiClient.getPipelineRun.mockResolvedValue(
    run({
      status: 'waiting',
      stages: [stage({ stage: 'attribute', checkpoint: true, waiting_on_person: true })],
    })
  );

  renderRun();

  expect(await screen.findByText('Waiting for you')).toBeInTheDocument();
  expect(screen.getByText('checkpoint')).toBeInTheDocument();
});

it('does not offer to restart a stage the server would refuse', async () => {
  // Two runners on one stage is what the execution lease exists to stop, so
  // the button that would produce it is disabled rather than answered with a
  // 400 after the click.
  apiClient.getPipelineRun.mockResolvedValue(
    run({ stages: [stage({ stage: 'measure', status: 'running', restartable: false })] })
  );

  renderRun();

  const button = await screen.findByLabelText('Restart measure');
  expect(button).toBeDisabled();
});

it('renders nothing for a job that is not part of a pipeline', async () => {
  // It is mounted for every job in the detail panel, and most jobs are not
  // stages of anything; the endpoint 404s for those.
  apiClient.getPipelineRun.mockRejectedValue({ response: { status: 404 } });

  const { container } = renderRun();

  await waitFor(() => expect(apiClient.getPipelineRun).toHaveBeenCalled());
  await waitFor(() =>
    expect(container.querySelector('[data-testid="pipeline-run-progress"]')).toBeNull()
  );
});

it('sends the correction with the restart, because a stage without one repeats itself', async () => {
  apiClient.getPipelineRun.mockResolvedValue(
    run({ stages: [stage({ stage: 'profile', status: 'completed' })] })
  );
  apiClient.restartPipelineStage.mockResolvedValue({
    root_job_id: 'root-1',
    stage: 'profile',
    job_id: 'job-9',
    note_attached: true,
  });

  renderRun();

  fireEvent.click(await screen.findByLabelText('Restart profile'));
  fireEvent.change(screen.getByLabelText(/What should it do differently/i), {
    target: { value: 'Sample per function' },
  });
  fireEvent.click(screen.getByRole('button', { name: /Run profile again/i }));

  await waitFor(() =>
    expect(apiClient.restartPipelineStage).toHaveBeenCalledWith(
      'root-1',
      'profile',
      'Sample per function'
    )
  );
});

it('renders a real run captured from the API', async () => {
  // Copied verbatim from GET /agent-pipelines/runs/{id}/stages against a live
  // server, so the shape here is the server's rather than this file's opinion
  // of it. That run failed in its first stage, which is the case the panel
  // most needs to render: two stages that will now never start, and the reason
  // the first one stopped.
  apiClient.getPipelineRun.mockResolvedValue({
    root_job_id: '722d5f64-1f15-4b68-a2dd-7cd6c6e02951',
    stages: [
      {
        stage: 'gather',
        job_id: '722d5f64-1f15-4b68-a2dd-7cd6c6e02951',
        status: 'failed',
        iteration: 5,
        contract_satisfied: true,
        restartable: true,
        goal: 'ingest',
        checkpoint: false,
        waiting_on_person: false,
        attempts: 1,
        progress: 48,
        started_at: '2026-09-09T11:07:03.663944+00:00',
        completed_at: '2026-09-09T11:21:01.532649+00:00',
        error:
          "Task error: This Session's transaction has been rolled back due to a previous exception during flush.",
      },
      {
        stage: 'read',
        job_id: '',
        status: 'pending',
        iteration: 0,
        contract_satisfied: false,
        restartable: false,
        goal: 'read',
        checkpoint: false,
        waiting_on_person: false,
        attempts: 0,
        progress: 0,
        started_at: null,
        completed_at: null,
        error: null,
      },
      {
        stage: 'writeup',
        job_id: '',
        status: 'pending',
        iteration: 0,
        contract_satisfied: false,
        restartable: false,
        goal: 'write',
        checkpoint: false,
        waiting_on_person: false,
        attempts: 0,
        progress: 0,
        started_at: null,
        completed_at: null,
        error: null,
      },
    ],
    pipeline: 'attention-survey',
    saved_pipeline_id: 'b78584b6-014c-4404-a46c-375ee6810c28',
    status: 'failed',
    total_stages: 3,
    completed_stages: 0,
    current_stage: 'gather',
  });

  renderRun();

  expect(await screen.findByText('Failed')).toBeInTheDocument();
  expect(screen.getByText(/0 of 3 stages/)).toBeInTheDocument();
  expect(screen.getByText(/rolled back/)).toBeInTheDocument();
  // The failed stage is offered for a restart; the two that never began are
  // not, because there is nothing yet to run again.
  expect(screen.getByLabelText('Restart gather')).toBeEnabled();
  expect(screen.queryByLabelText('Restart read')).toBeNull();
  // `contract_satisfied` is true on that failed stage only because the job
  // never recorded a contract result at all -- the chain gate reads a missing
  // one as "nothing outstanding". On a stage that did not complete the field
  // means nothing, and the panel must not dress a failure up as a met
  // contract, nor as an unmet one.
  expect(screen.queryByText('contract unmet')).toBeNull();
});
