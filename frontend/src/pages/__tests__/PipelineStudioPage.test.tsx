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
    // The library loads on mount. Unmocked it throws into a deliberately
    // silent catch — the tests pass while the page errors on every render,
    // which is worse than a failure because nothing says so.
    listSavedPipelines: jest.fn(),
    saveSavedPipeline: jest.fn(),
    updateSavedPipeline: jest.fn(),
    deleteSavedPipeline: jest.fn(),
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
  apiClient.listSavedPipelines.mockResolvedValue([]);
});

const typeSpec = (text: string) =>
  fireEvent.change(screen.getByLabelText('Pipeline specification'), {
    target: { value: text },
  });

it('shows the tools it derived, which the author never writes', async () => {
  render(<PipelineStudioPage />);

  // The point of contracts: you say what must be true, and the tools that get
  // there are deduced. An author who names tools can name an impossible one.
  //
  // getAllBy, because the default split view shows each tool twice — once in
  // the plan and once on its node in the graph. Two views of one document is
  // the design, so a query that demanded uniqueness would be asserting the
  // opposite of what this page is for.
  expect((await screen.findAllByText('profile_c_workload')).length).toBeGreaterThan(0);
  expect(screen.getAllByText('benchmark_c_snippet').length).toBeGreaterThan(0);
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
    await screen.findAllByText('profile_c_workload');

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
    await screen.findAllByText('profile_c_workload');

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


describe('the library', () => {
  const stored = {
    id: 'p-1',
    name: 'Attention survey',
    spec: { name: 'attention-survey', stages: [{ id: 'gather', contract: {} }] },
    last_check_valid: 'valid' as const,
    last_estimated_seconds: 540,
    launch_count: 3,
    last_job_id: 'job-7',
  };

  it('lists what has been saved, with how often it has run', async () => {
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    render(<PipelineStudioPage />);

    expect(await screen.findByText('Attention survey')).toBeInTheDocument();
    expect(screen.getByText('×3')).toBeInTheDocument();
  });

  it('re-checks a pipeline it opens rather than trusting the stored verdict', async () => {
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));

    // The saved verdict was 'valid' when it was saved. Tools and their costs
    // move underneath a stored spec, so it is asked again.
    await waitFor(() =>
      expect(apiClient.checkPipeline).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'attention-survey' }),
        undefined
      )
    );
  });

  it('records which saved pipeline a launch came from', async () => {
    jest.spyOn(window, 'confirm').mockReturnValue(true);
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    apiClient.launchPipeline.mockResolvedValue({
      job_id: 'job-9',
      name: 'x',
      stages: [],
      estimated_seconds: 180,
      checkpoints: [],
    });
    render(<PipelineStudioPage />);
    fireEvent.click(await screen.findByText('Attention survey'));
    await screen.findAllByText('profile_c_workload');

    fireEvent.click(screen.getByRole('button', { name: /Launch/i }));

    await waitFor(() =>
      expect(apiClient.launchPipeline).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({ pipelineId: 'p-1' })
      )
    );
  });

  it('warns that deleting a pipeline keeps its runs', async () => {
    const confirm = jest.spyOn(window, 'confirm').mockReturnValue(false);
    apiClient.listSavedPipelines.mockResolvedValue([stored]);
    render(<PipelineStudioPage />);
    await screen.findByText('Attention survey');

    fireEvent.click(screen.getByRole('button', { name: /Delete Attention survey/i }));

    expect(confirm.mock.calls[0][0]).toContain('runs it started stay');
    expect(apiClient.deleteSavedPipeline).not.toHaveBeenCalled();
    confirm.mockRestore();
  });
});

describe('the starters', () => {
  beforeEach(() => {
    apiClient.listSavedPipelines.mockResolvedValue([]);
    apiClient.checkPipeline.mockResolvedValue(okCheck);
    window.localStorage.clear();
  });

  it('offers a reproduction pipeline as well as the default', () => {
    render(<PipelineStudioPage />);
    const picker = screen.getByLabelText('Start from a worked example');
    expect(picker).toHaveTextContent('Reproduce a paper');
  });

  it('loads a starter that is itself a valid pipeline', async () => {
    // The studio's copy of this spec is a duplicate of the one the backend
    // suite validates, so a typo here would ship a starter that fails its own
    // check — which teaches the wrong thing about the format on first contact.
    render(<PipelineStudioPage />);
    fireEvent.change(screen.getByLabelText('Start from a worked example'), {
      target: { value: 'Reproduce a paper' },
    });

    await waitFor(() => {
      const sent = apiClient.checkPipeline.mock.calls.at(-1)?.[0];
      expect(sent).toBeDefined();
      expect(sent.name).toBe('reproduce-paper-algorithm');
      expect(sent.stages.map((s: { id: string }) => s.id)).toEqual([
        'find',
        'specify',
        'implement',
        'measure',
        'compare',
      ]);
    });
  });

  it('asks for a verified implementation, not merely one that compiles', async () => {
    // The distinction the whole chain rests on: timing unchecked code produces
    // an accurate number for work nobody looked at.
    render(<PipelineStudioPage />);
    fireEvent.change(screen.getByLabelText('Start from a worked example'), {
      target: { value: 'Reproduce a paper' },
    });

    await waitFor(() => {
      const sent = apiClient.checkPipeline.mock.calls.at(-1)?.[0];
      const implement = sent.stages.find((s: { id: string }) => s.id === 'implement');
      expect(implement.contract.required_finding_types).toContain(
        'implementation_verified',
      );
      // And it loops: writing an algorithm from prose does not work first try.
      expect(implement.loop.max_iterations).toBeGreaterThan(1);
    });
  });
});
