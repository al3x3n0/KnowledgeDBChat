import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter, useLocation } from 'react-router-dom';

import CampaignsPage from '../CampaignsPage';

jest.mock('../../services/api', () => ({
  apiClient: {
    listResearchCampaigns: jest.fn(),
    getResearchCampaign: jest.fn(),
    createResearchCampaign: jest.fn(),
    actionResearchCampaign: jest.fn(),
    createSynthesisJob: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

const apiClient = require('../../services/api').apiClient;

const campaign = (over = {}) => ({
  id: 'c1',
  name: 'Prefetcher study',
  goal: 'Establish whether a stride prefetcher helps dotprod',
  status: 'completed',
  max_jobs: 4,
  jobs_launched: 2,
  conclusion: null,
  conclusion_detail: null,
  created_at: '2026-09-15T10:00:00Z',
  items: [],
  ...over,
});

function CurrentRoute() {
  const { pathname, search } = useLocation();
  return <output aria-label="Current route">{pathname}{search}</output>;
}

function renderCampaigns(path = '/campaigns') {
  return render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { cacheTime: 0 } } })}>
      <MemoryRouter initialEntries={[path]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <CampaignsPage />
        <CurrentRoute />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('CampaignsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [campaign()],
      total: 1,
    });
    apiClient.getResearchCampaign.mockResolvedValue(campaign());
  });

  it('shows the goal, not just the name', async () => {
    // A campaign is identified by what it is trying to settle.
    renderCampaigns();

    expect(await screen.findByText('Prefetcher study')).toBeInTheDocument();
    expect(
      screen.getByText(/whether a stride prefetcher helps dotprod/)
    ).toBeInTheDocument();
  });

  it('says when it has stopped for a person, and where to go', async () => {
    // Its job proposed a plan; the campaign asks nothing more until someone
    // approves it, and the counters alone would read as "still running".
    const waiting = campaign({
      status: 'active',
      summary: { waiting_on: 'approval' },
    });
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [waiting],
      total: 1,
    });
    apiClient.getResearchCampaign.mockResolvedValue(waiting);
    renderCampaigns();

    expect(
      await screen.findByText(/waiting for approval/)
    ).toBeInTheDocument();
    fireEvent.click(screen.getByText('Prefetcher study'));
    const link = await screen.findByText('Open Review');
    expect(link).toHaveAttribute('href', '/review');
    expect(screen.queryByText(/Still running/)).not.toBeInTheDocument();
  });

  it('does not call a job at work a wait on anyone', async () => {
    const working = campaign({ status: 'active', summary: { waiting_on: 'job' } });
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [working],
      total: 1,
    });
    apiClient.getResearchCampaign.mockResolvedValue(working);
    renderCampaigns();

    fireEvent.click(await screen.findByText('Prefetcher study'));
    expect(await screen.findByText('Still running')).toBeInTheDocument();
    expect(screen.queryByText(/waiting for/)).not.toBeInTheDocument();
  });

  it('leads with the answer once one exists', async () => {
    apiClient.getResearchCampaign.mockResolvedValue(
      campaign({
        conclusion: 'It helps by 18% (confidence: medium)',
        conclusion_detail: {
          answer: 'It helps by 18%',
          confidence: 'medium',
          evidence: ['dotprod: 31ms with, 38ms without'],
          gaps: ['Only one kernel measured.'],
        },
      })
    );
    renderCampaigns();

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(await screen.findByText(/It helps by 18%/)).toBeInTheDocument();
    expect(screen.getByText(/31ms with, 38ms without/)).toBeInTheDocument();
    expect(screen.getByText(/Only one kernel measured/)).toBeInTheDocument();
  });

  it.each([
    ['queued', 'Queued for a worker', 'job'],
    ['sandbox', 'Waiting for a sandbox experiment', 'job'],
    ['external', 'Waiting for an external service', 'job'],
    ['finalizing', 'Finalizing results', 'job'],
    ['paused', 'Run paused', 'person'],
    ['review', 'Run needs review', 'person'],
  ])('renders the %s activity without a response action', async (waitState, message, waitingOn) => {
    const working = campaign({ status: 'active', summary: {
      waiting_on: waitingOn, wait_state: waitState,
      current_item: { id: 'q1', title: 'Measure the kernel', job_id: 'current-attempt' },
    } });
    apiClient.listResearchCampaigns.mockResolvedValue({ items: [working], total: 1 });
    apiClient.getResearchCampaign.mockResolvedValue(working);
    renderCampaigns('/campaigns?campaign=c1');

    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /respond|approval/i })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open run' })).toHaveAttribute('href', '/autonomous-agents?job=current-attempt&campaign=c1&campaign_job=current-attempt');
    expect(screen.queryByRole('button', { name: 'Write up campaign' })).not.toBeInTheDocument();
  });

  it('shows missing evidence and starts a campaign-backed Synthesis report', async () => {
    apiClient.getResearchCampaign.mockResolvedValue(campaign({
      status: 'exhausted',
      items: [{ id: 'q1', title: 'Repeat the benchmark', origin: 'seed', status: 'failed', blockers: ['The benchmark did not produce all required evidence.'] }],
    }));
    apiClient.createSynthesisJob.mockResolvedValue({ id: 'report-1' });
    renderCampaigns('/campaigns?campaign=c1');

    expect(await screen.findByText(/incomplete · asked at the start/)).toBeInTheDocument();
    expect(screen.getByText('The benchmark did not produce all required evidence.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Write up campaign' }));

    await waitFor(() => expect(screen.getByLabelText('Current route')).toHaveTextContent('/synthesis?job=report-1'));
    expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
      job_type: 'research_report', title: 'Write-up: Prefetcher study', document_ids: [], campaign_id: 'c1',
    });
  });

  it('says a finished campaign recorded no conclusion, rather than showing a blank', async () => {
    // Four campaigns finished before conclusions were recorded. "No answer"
    // and "an answer we failed to render" must not look the same.
    renderCampaigns();

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(
      await screen.findByText(/finished without recording a conclusion/)
    ).toBeInTheDocument();
  });

  it('says a running campaign has not concluded yet, which is different', async () => {
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [campaign({ status: 'active' })],
      total: 1,
    });
    apiClient.getResearchCampaign.mockResolvedValue(campaign({ status: 'active' }));
    renderCampaigns();

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(await screen.findByText('Waiting for the scheduler.')).toBeInTheDocument();
  });

  it('says when a campaign stopped early rather than finished', async () => {
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [campaign({ status: 'exhausted' })],
      total: 1,
    });
    renderCampaigns();

    expect(
      await screen.findByText(/stopped early, out of budget/)
    ).toBeInTheDocument();
  });

  it('distinguishes a question the campaign raised itself', async () => {
    // The difference between a question a person asked and one the campaign
    // derived from a finding is the whole reason it is a campaign.
    apiClient.getResearchCampaign.mockResolvedValue(
      campaign({
        items: [
          {
            id: 'i1',
            title: 'Profile the kernel',
            status: 'done',
            origin: 'seed',
            generation: 0,
          },
          {
            id: 'i2',
            title: 'Cost the top candidate',
            status: 'running',
            origin: 'discovered',
            generation: 1,
          },
        ],
      })
    );
    renderCampaigns();

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(await screen.findByText(/asked at the start/)).toBeInTheDocument();
    expect(
      screen.getByText(/raised by the campaign \(generation 1\)/)
    ).toBeInTheDocument();
  });

  it('fetches items only when a campaign is opened', async () => {
    // Loading every question of every campaign to render a list that shows
    // none of them would be a query per campaign for nothing.
    renderCampaigns();
    await screen.findByText('Prefetcher study');

    expect(apiClient.getResearchCampaign).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText('Prefetcher study'));

    await waitFor(() =>
      expect(apiClient.getResearchCampaign).toHaveBeenCalledWith('c1')
    );
  });

  it('opens a linked campaign outside the current list and links its question to the run', async () => {
    apiClient.getResearchCampaign.mockResolvedValue(campaign({
      id: 'c2',
      name: 'Cache policy study',
      items: [{ id: 'i1', title: 'Compare replacement policies', status: 'running', origin: 'seed', job_id: 'job-2' }],
    }));

    renderCampaigns('/campaigns?campaign=c2');

    expect(await screen.findByRole('link', { name: 'Open run for Compare replacement policies' }))
      .toHaveAttribute('href', '/autonomous-agents?job=job-2');
    expect(apiClient.getResearchCampaign).toHaveBeenCalledWith('c2');
    expect(screen.getByRole('button', { name: /Cache policy study/ })).toHaveAttribute('aria-expanded', 'true');
  });

  it('shows the budget and latest result beside the exact approval checkpoint', async () => {
    apiClient.getResearchCampaign.mockResolvedValue(campaign({
      status: 'active',
      summary: {
        waiting_on: 'approval',
        current_item: { id: 'i2', title: 'Compare the cache policies', job_id: 'paused-stage' },
        jobs_used: 3,
        remaining_jobs: 1,
        next_up: [{ title: 'Check a second workload', why: 'Tests whether the result generalizes' }],
        latest_result: {
          title: 'Profile the baseline',
          job_id: 'baseline',
          conclusion: 'Memory stalls dominate.',
          finding_count: 2,
        },
      },
    }));
    renderCampaigns('/campaigns?campaign=c1');

    expect(await screen.findByRole('link', { name: 'Review approval' })).toHaveAttribute(
      'href', '/autonomous-agents?tab=queue&queue_job=paused-stage&queue_item_type=approval_checkpoint&campaign=c1'
    );
    expect(screen.getByText('1 of 4')).toBeInTheDocument();
    expect(screen.getByText('3 of 4 jobs used')).toBeInTheDocument();
    expect(screen.getByText('Check a second workload')).toBeInTheDocument();
    expect(screen.getByText('Memory stalls dominate.')).toBeInTheDocument();
    expect(screen.getByText('2 findings')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open result run' })).toHaveAttribute(
      'href', '/autonomous-agents?job=baseline&campaign=c1&campaign_job=baseline'
    );
  });

  it('pauses, resumes and confirms stopping while preserving existing runs', async () => {
    const active = campaign({
      status: 'active',
      items: [
        { id: 'i1', title: 'Measure the baseline', status: 'running', origin: 'seed', job_id: 'baseline' },
        { id: 'i2', title: 'Optimize the kernel', status: 'pending', origin: 'seed' },
      ],
    });
    apiClient.getResearchCampaign.mockResolvedValue(active);
    apiClient.actionResearchCampaign
      .mockResolvedValueOnce({ ...active, status: 'paused' })
      .mockResolvedValueOnce(active)
      .mockResolvedValueOnce({ ...active, status: 'cancelled' });
    renderCampaigns('/campaigns?campaign=c1');

    fireEvent.click(await screen.findByRole('button', { name: 'Pause scheduling' }));
    const resume = await screen.findByRole('button', { name: 'Resume scheduling' });
    await waitFor(() => expect(resume).toBeEnabled());
    fireEvent.click(resume);
    await waitFor(() => expect(apiClient.actionResearchCampaign.mock.calls).toEqual([['c1', 'pause'], ['c1', 'resume']]));
    const pause = await screen.findByRole('button', { name: 'Pause scheduling' });
    await waitFor(() => expect(pause).toBeEnabled());

    fireEvent.click(screen.getByRole('button', { name: 'Stop campaign' }));
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Stop campaign?' })).getByRole('button', { name: 'Cancel' }));
    expect(apiClient.actionResearchCampaign).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole('button', { name: 'Stop campaign' }));
    const confirmation = within(screen.getByRole('dialog', { name: 'Stop campaign?' }));
    expect(confirmation.getByText(/Existing runs can continue/)).toBeInTheDocument();
    fireEvent.click(confirmation.getByRole('button', { name: 'Stop campaign' }));

    expect(await screen.findByText(/Campaign scheduling ended early/)).toBeInTheDocument();
    expect(apiClient.actionResearchCampaign).toHaveBeenNthCalledWith(3, 'c1', 'stop');
    expect(screen.queryByRole('button', { name: 'Resume scheduling' })).not.toBeInTheDocument();
    expect(screen.getByText(/unscheduled/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open run for Measure the baseline' })).toBeInTheDocument();
  });

  it('points somewhere useful when there are none', async () => {
    apiClient.listResearchCampaigns.mockResolvedValue({ items: [], total: 0 });
    renderCampaigns();

    expect(await screen.findByText('No campaigns yet.')).toBeInTheDocument();
    expect(screen.getByText(/ask for one in chat/)).toBeInTheDocument();
  });

  it('can start one from here, not only from somewhere else', async () => {
    // A Campaigns page with no way to start a campaign sends you elsewhere to
    // do the thing the page is about.
    renderCampaigns();
    await screen.findByText('Prefetcher study');

    fireEvent.click(screen.getByRole('button', { name: /New Campaign/ }));

    expect(await screen.findByText('Start campaign')).toBeInTheDocument();
  });

  it('prefills the CPU co-design template and submits edited instructions separately from the question', async () => {
    apiClient.createResearchCampaign.mockResolvedValue(campaign());
    renderCampaigns();
    fireEvent.click(screen.getByRole('button', { name: 'New Campaign' }));

    const form = within(screen.getByRole('dialog', { name: 'New Campaign' }));
    fireEvent.click(form.getByRole('button', { name: 'Use CPU co-design template' }));
    expect((form.getByLabelText(/Research goal/) as HTMLTextAreaElement).value).toContain('INT8 attention');
    expect(form.getByLabelText(/Maximum agent jobs/)).toHaveValue(6);
    expect(apiClient.createResearchCampaign).not.toHaveBeenCalled();

    fireEvent.click(form.getAllByText('Edit instructions')[1]);
    const matrixInstructions = form.getByLabelText('Instructions for question 2') as HTMLTextAreaElement;
    expect(matrixInstructions.value).toContain('D (optimized kernel, enlarged load queue)');
    fireEvent.change(matrixInstructions, { target: { value: `${matrixInstructions.value} Use the 64-token test input.` } });
    fireEvent.click(form.getByRole('button', { name: 'Start campaign' }));
    await waitFor(() => expect(apiClient.createResearchCampaign).toHaveBeenCalledWith(expect.objectContaining({
      max_jobs: 6,
      plan_approval: null,
      items: [
        expect.objectContaining({ title: 'Which baseline and correctness checks should the study use?' }),
        { title: 'How do the four kernel and load queue combinations compare?', detail: expect.stringContaining('Use the 64-token test input.') },
        expect.objectContaining({ title: 'Does the result hold and which change is worth pursuing?' }),
      ],
    })));
  });

  it('shows the minimum starting work and keeps an intentionally short budget available', async () => {
    renderCampaigns();
    fireEvent.click(screen.getByRole('button', { name: 'New Campaign' }));
    const form = within(screen.getByRole('dialog', { name: 'New Campaign' }));
    const budget = form.getByLabelText(/Maximum agent jobs/);
    expect(form.getByText(/research goal is the first question/)).toBeInTheDocument();

    fireEvent.click(form.getByRole('button', { name: 'Use CPU co-design template' }));
    expect(form.getByText(/3 starting questions · at least 3 jobs/)).toBeInTheDocument();
    fireEvent.change(budget, { target: { value: '2' } });
    expect(form.getByText(/2-job budget can leave 1 or more starting questions uninvestigated/)).toBeInTheDocument();
    expect(form.getByRole('button', { name: 'Start campaign' })).toBeEnabled();

    fireEvent.change(budget, { target: { value: '0' } });
    expect(budget).toHaveAttribute('aria-invalid', 'true');
    expect(form.getByRole('button', { name: 'Start campaign' })).toBeDisabled();
    expect(apiClient.createResearchCampaign).not.toHaveBeenCalled();
  });

  it('says how much a campaign may approve itself', async () => {
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [campaign({ plan_approval: { max_stages: 2, max_minutes: 60 } })],
      total: 1,
    });
    renderCampaigns();

    expect(
      await screen.findByText(/approves its own plans up to 2 stages, 60 min/)
    ).toBeInTheDocument();
  });

  it('says why the campaign left a plan for you', async () => {
    const held = campaign({
      status: 'active',
      plan_approval: { max_stages: 1, max_minutes: 60 },
      summary: {
        waiting_on: 'approval',
        plan_held: 'the plan has 2 stages and this campaign approves at most 1',
      },
    });
    apiClient.listResearchCampaigns.mockResolvedValue({ items: [held], total: 1 });
    apiClient.getResearchCampaign.mockResolvedValue(held);
    renderCampaigns();

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(
      await screen.findByText(/the plan has 2 stages/)
    ).toBeInTheDocument();
  });

  it('starts a campaign that approves its own plans only when asked to', async () => {
    apiClient.createResearchCampaign.mockResolvedValue(campaign({ name: 'n' }));
    renderCampaigns();
    await screen.findByText('Prefetcher study');
    fireEvent.click(screen.getByRole('button', { name: /New Campaign/ }));
    const goalField = () => screen.findByPlaceholderText(/stride prefetcher speeds up/);
    fireEvent.change(await goalField(), { target: { value: 'settle it' } });

    fireEvent.click(screen.getByRole('button', { name: /Start campaign/ }));
    await waitFor(() => expect(apiClient.createResearchCampaign).toHaveBeenCalled());
    expect(apiClient.createResearchCampaign.mock.calls[0][0].plan_approval).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /New Campaign/ }));
    fireEvent.change(await goalField(), { target: { value: 'settle it' } });
    fireEvent.click(screen.getByLabelText('Approve its own plans'));
    fireEvent.change(screen.getByLabelText('Up to stages'), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: /Start campaign/ }));
    await waitFor(() => expect(apiClient.createResearchCampaign).toHaveBeenCalledTimes(2));
    expect(apiClient.createResearchCampaign.mock.calls[1][0].plan_approval).toEqual({
      max_stages: 3,
      max_minutes: 60,
    });
  });
});
