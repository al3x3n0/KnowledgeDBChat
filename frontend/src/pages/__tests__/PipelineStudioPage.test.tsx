/**
 * The pipeline editor, and the distinctions it has to keep separate.
 *
 * Its job is to be the place a bad pipeline dies, so the assertions here are
 * mostly about failure being visible and specific: unparseable text is not the
 * same as an invalid pipeline, and an invalid pipeline is not the same as an
 * unaffordable one.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';

import PipelineStudioPage from '../PipelineStudioPage';

jest.mock('../../services/api', () => ({
  apiClient: {
    checkPipeline: jest.fn(),
    bindPipeline: jest.fn(),
    launchPipeline: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

const apiClient = require('../../services/api').apiClient;

const okCheck = {
  valid: true,
  problems: [],
  expressible: true,
  binding_problems: [],
  description: ['study:', '  2 stages'],
  plan: {
    order: ['profile', 'measure'],
    stages: [
      {
        stage_id: 'profile',
        tools: ['profile_c_workload'],
        iterations: 1,
        seconds: 60,
        checkpoint: false,
        unpriced: [],
      },
      {
        stage_id: 'measure',
        tools: ['benchmark_c_snippet'],
        iterations: 1,
        seconds: 120,
        checkpoint: true,
        unpriced: [],
      },
    ],
    total_seconds: 180,
    critical_path_seconds: 180,
    checkpoints: ['measure'],
  },
  budget: null,
};

beforeEach(() => {
  jest.clearAllMocks();
  window.localStorage.clear();
  apiClient.checkPipeline.mockResolvedValue(okCheck);
});

const typeSpec = (text: string) =>
  fireEvent.change(screen.getByLabelText('Pipeline specification'), {
    target: { value: text },
  });

it('shows the tools it derived, which the author never writes', async () => {
  render(<PipelineStudioPage />);

  // The point of contracts: you say what must be true, and the tools that get
  // there are deduced. An author who names tools can name an impossible one.
  expect(await screen.findByText('profile_c_workload')).toBeInTheDocument();
  expect(screen.getByText('benchmark_c_snippet')).toBeInTheDocument();
});

it('reports a checkpoint before the run stops at it, not when it does', async () => {
  render(<PipelineStudioPage />);

  expect(await screen.findByText(/stops at measure/i)).toBeInTheDocument();
  expect(screen.getByText('checkpoint')).toBeInTheDocument();
});

it('separates unparseable text from an invalid pipeline', async () => {
  render(<PipelineStudioPage />);
  await screen.findByText('profile_c_workload');

  typeSpec('{ not json');

  expect(await screen.findByText('Not valid JSON')).toBeInTheDocument();
  // And it does not ask the server about text that is not a document — that
  // would only return a less specific version of the same complaint.
  const callsBefore = apiClient.checkPipeline.mock.calls.length;
  await new Promise((resolve) => setTimeout(resolve, 700));
  expect(apiClient.checkPipeline.mock.calls.length).toBe(callsBefore);
});

it('lists every problem, not just the first', async () => {
  apiClient.checkPipeline.mockResolvedValue({
    ...okCheck,
    valid: false,
    plan: null,
    problems: [
      "attribute: assumes 'counter_trace', which no stage before it produces",
      'b: no tool produces telepathy -- the contract cannot be satisfied',
    ],
  });
  render(<PipelineStudioPage />);

  expect(await screen.findByText('2 problems')).toBeInTheDocument();
  expect(screen.getByText(/counter_trace/)).toBeInTheDocument();
  expect(screen.getByText(/telepathy/)).toBeInTheDocument();
});

it('says a valid pipeline is unaffordable without calling it invalid', async () => {
  // Three separate answers. A well-formed pipeline that costs too much is a
  // different problem from a broken one, and collapsing them would send the
  // author to edit the wrong thing.
  apiClient.checkPipeline.mockResolvedValue({
    ...okCheck,
    budget: {
      affordable: false,
      budget_seconds: 60,
      estimated_seconds: 1200,
      critical_path_seconds: 1200,
      unpriced_tools: [],
    },
  });
  render(<PipelineStudioPage />);

  expect(await screen.findByText('Spec valid')).toBeInTheDocument();
  expect(screen.getByText(/Needs 20 min/)).toBeInTheDocument();
});

it('surfaces the server saying it is not a pipeline at all', async () => {
  apiClient.checkPipeline.mockRejectedValue({
    response: { data: { detail: "Not a pipeline spec: 'stages' must be a list, got str" } },
  });
  render(<PipelineStudioPage />);

  expect(await screen.findByText('Not a pipeline')).toBeInTheDocument();
  expect(screen.getByText(/'stages' must be a list/)).toBeInTheDocument();
});

it('marks unpriced tools rather than letting a stage look free', async () => {
  apiClient.checkPipeline.mockResolvedValue({
    ...okCheck,
    plan: {
      ...okCheck.plan,
      stages: [
        {
          stage_id: 'profile',
          tools: ['mystery_tool'],
          iterations: 1,
          seconds: 0,
          checkpoint: false,
          unpriced: ['mystery_tool'],
        },
      ],
      order: ['profile'],
      total_seconds: 0,
    },
  });
  render(<PipelineStudioPage />);

  // Zero seconds because nothing knows the cost is not the same as free.
  expect(await screen.findByText('1 unpriced')).toBeInTheDocument();
});


describe('launching', () => {
  it('does not offer to launch a pipeline the server would refuse', async () => {
    apiClient.checkPipeline.mockResolvedValue({
      ...okCheck,
      valid: false,
      plan: null,
      problems: ['a: no tool produces telepathy'],
    });
    render(<PipelineStudioPage />);
    await screen.findByText('1 problem');

    // Offering a button whose only outcome is a refusal teaches the user that
    // buttons here do not mean anything.
    expect(screen.getByRole('button', { name: /Launch/i })).toBeDisabled();
  });

  it('does not offer to launch a pipeline that cannot afford itself', async () => {
    apiClient.checkPipeline.mockResolvedValue({
      ...okCheck,
      budget: {
        affordable: false,
        budget_seconds: 60,
        estimated_seconds: 1200,
        critical_path_seconds: 1200,
        unpriced_tools: [],
      },
    });
    render(<PipelineStudioPage />);
    await screen.findByText(/Needs 20 min/);

    expect(screen.getByRole('button', { name: /Launch/i })).toBeDisabled();
  });

  it('names the price and the stops before spending anything', async () => {
    const confirm = jest.spyOn(window, 'confirm').mockReturnValue(false);
    render(<PipelineStudioPage />);
    await screen.findByText('profile_c_workload');

    fireEvent.click(screen.getByRole('button', { name: /Launch/i }));

    const asked = confirm.mock.calls[0][0] as string;
    expect(asked).toContain('3 min');
    expect(asked).toContain('measure'); // the checkpoint it will stop at
    // Declining means nothing was started.
    expect(apiClient.launchPipeline).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it('sends the estimate it showed, so an edited spec is refused', async () => {
    jest.spyOn(window, 'confirm').mockReturnValue(true);
    apiClient.launchPipeline.mockResolvedValue({
      job_id: 'job-9',
      name: 'study',
      stages: ['profile', 'measure'],
      estimated_seconds: 180,
      checkpoints: [],
    });
    render(<PipelineStudioPage />);
    await screen.findByText('profile_c_workload');

    fireEvent.click(screen.getByRole('button', { name: /Launch/i }));

    await waitFor(() =>
      expect(apiClient.launchPipeline).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({ acknowledgedSeconds: 180 })
      )
    );
    // And it goes to the run it just started, rather than leaving the user on
    // an editor with no sign anything happened.
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith('/autonomous-agents?job=job-9'));
  });
});
