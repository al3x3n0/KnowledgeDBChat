import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';

import ReviewPage from '../ReviewPage';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    getReviewInbox: jest.fn(),
    approveToolAudit: jest.fn(),
    rejectToolAudit: jest.fn(),
    runToolAudit: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;
const toast = require('react-hot-toast').default;

const section = (key: string, title: string, link: string, extra = {}) => ({
  key,
  title,
  description: `${title} description`,
  link,
  count: 0,
  items: [],
  error: null,
  ...extra,
});

const INBOX = {
  total: 4,
  sections: [
    section('runs', 'Runs', '/agent-control-plane', {
      count: 1,
      items: [
        {
          id: 'job-1',
          title: 'attention-survey: writeup',
          detail: '3 consecutive rounds produced no new findings',
          link: '/autonomous-agents?job=job-1',
          kind: 'blocked_run',
        },
      ],
    }),
    section('tool_approvals', 'Tool approvals', '/review', {
      count: 1,
      items: [{ id: 'audit-1', title: 'delete_document', detail: 'Requested in your session' }],
    }),
    section('patches', 'Proposed patches', '/patch-prs', {
      count: 2,
      items: [{ id: 'p-1', title: 'Fix the off-by-one', link: '/patch-prs' }],
    }),
    section('drafts', 'Drafts in review', '/artifact-drafts'),
    section('inbox', 'Research inbox', '/research/inbox', {
      error: 'This section could not be read.',
    }),
  ],
};

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/review']}>
        <ReviewPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('ReviewPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    apiClient.getReviewInbox.mockResolvedValue(INBOX);
  });

  it('lists what is waiting and links each item to where it is handled', async () => {
    renderPage();

    expect(await screen.findByText('4 things are waiting on you')).toBeInTheDocument();
    const run = screen.getByRole('link', { name: 'attention-survey: writeup' });
    expect(run).toHaveAttribute('href', '/autonomous-agents?job=job-1');
    expect(screen.getByText('3 consecutive rounds produced no new findings')).toBeInTheDocument();
    // The count covers every item, not only the ones named.
    expect(within(screen.getByLabelText('Proposed patches')).getByText('and 1 more')).toBeInTheDocument();
  });

  it('keeps empty sections out of the way but says they were checked', async () => {
    renderPage();

    expect(await screen.findByText('Nothing waiting in: Drafts in review.')).toBeInTheDocument();
    expect(screen.queryByLabelText('Drafts in review')).not.toBeInTheDocument();
  });

  it('says a section that failed is unknown, not empty', async () => {
    renderPage();

    const inbox = await screen.findByLabelText('Research inbox');
    expect(within(inbox).getByText(/Whether anything is waiting here is unknown/)).toBeInTheDocument();
  });

  it('runs an action once the last approval it needed is given', async () => {
    apiClient.approveToolAudit.mockResolvedValue({ approval_status: 'approved' });
    apiClient.runToolAudit.mockResolvedValue({});
    renderPage();

    const approvals = await screen.findByLabelText('Tool approvals');
    fireEvent.click(within(approvals).getByRole('button', { name: /Approve/ }));

    await waitFor(() => expect(apiClient.runToolAudit).toHaveBeenCalledWith('audit-1'));
    expect(toast.success).toHaveBeenCalledWith('Approved and run');
    // And the list is read again, so the item is gone.
    await waitFor(() => expect(apiClient.getReviewInbox).toHaveBeenCalledTimes(2));
  });

  it('does not run an action an admin has still to approve', async () => {
    apiClient.approveToolAudit.mockResolvedValue({ approval_status: 'pending_admin' });
    renderPage();

    const approvals = await screen.findByLabelText('Tool approvals');
    fireEvent.click(within(approvals).getByRole('button', { name: /Approve/ }));

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Approved. Waiting for an admin.'));
    expect(apiClient.runToolAudit).not.toHaveBeenCalled();
  });

  it('rejects without running', async () => {
    apiClient.rejectToolAudit.mockResolvedValue({});
    renderPage();

    const approvals = await screen.findByLabelText('Tool approvals');
    fireEvent.click(within(approvals).getByRole('button', { name: /Reject/ }));

    await waitFor(() => expect(apiClient.rejectToolAudit).toHaveBeenCalledWith('audit-1'));
    expect(apiClient.runToolAudit).not.toHaveBeenCalled();
  });

  it('says so when nothing is waiting', async () => {
    apiClient.getReviewInbox.mockResolvedValue({
      total: 0,
      sections: [section('runs', 'Runs', '/agent-control-plane')],
    });
    renderPage();

    expect(await screen.findByText('Nothing is waiting on you')).toBeInTheDocument();
  });
});
