import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import CampaignsPage from '../CampaignsPage';

jest.mock('../../services/api', () => ({
  apiClient: {
    listResearchCampaigns: jest.fn(),
    getResearchCampaign: jest.fn(),
    createResearchCampaign: jest.fn(),
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
    render(<CampaignsPage />);

    expect(await screen.findByText('Prefetcher study')).toBeInTheDocument();
    expect(
      screen.getByText(/whether a stride prefetcher helps dotprod/)
    ).toBeInTheDocument();
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
    render(<CampaignsPage />);

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(await screen.findByText(/It helps by 18%/)).toBeInTheDocument();
    expect(screen.getByText(/31ms with, 38ms without/)).toBeInTheDocument();
    expect(screen.getByText(/Only one kernel measured/)).toBeInTheDocument();
  });

  it('says a finished campaign recorded no conclusion, rather than showing a blank', async () => {
    // Four campaigns finished before conclusions were recorded. "No answer"
    // and "an answer we failed to render" must not look the same.
    render(<CampaignsPage />);

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
    render(<CampaignsPage />);

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(await screen.findByText(/Still running/)).toBeInTheDocument();
  });

  it('says when a campaign stopped early rather than finished', async () => {
    apiClient.listResearchCampaigns.mockResolvedValue({
      items: [campaign({ status: 'exhausted' })],
      total: 1,
    });
    render(<CampaignsPage />);

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
    render(<CampaignsPage />);

    fireEvent.click(await screen.findByText('Prefetcher study'));

    expect(await screen.findByText(/asked at the start/)).toBeInTheDocument();
    expect(
      screen.getByText(/raised by the campaign \(generation 1\)/)
    ).toBeInTheDocument();
  });

  it('fetches items only when a campaign is opened', async () => {
    // Loading every question of every campaign to render a list that shows
    // none of them would be a query per campaign for nothing.
    render(<CampaignsPage />);
    await screen.findByText('Prefetcher study');

    expect(apiClient.getResearchCampaign).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText('Prefetcher study'));

    await waitFor(() =>
      expect(apiClient.getResearchCampaign).toHaveBeenCalledWith('c1')
    );
  });

  it('points somewhere useful when there are none', async () => {
    apiClient.listResearchCampaigns.mockResolvedValue({ items: [], total: 0 });
    render(<CampaignsPage />);

    expect(await screen.findByText('No campaigns yet.')).toBeInTheDocument();
    expect(screen.getByText(/ask for one in chat/)).toBeInTheDocument();
  });

  it('can start one from here, not only from somewhere else', async () => {
    // A Campaigns page with no way to start a campaign sends you elsewhere to
    // do the thing the page is about.
    render(<CampaignsPage />);
    await screen.findByText('Prefetcher study');

    fireEvent.click(screen.getByRole('button', { name: /New Campaign/ }));

    expect(await screen.findByText('Start campaign')).toBeInTheDocument();
  });
});
