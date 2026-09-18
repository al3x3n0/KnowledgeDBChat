import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';

import { RejectWithReason } from '../RejectWithReason';
import { apiClient } from '../../../services/api';

jest.mock('../../../services/api', () => ({
  apiClient: { getResearchInboxRejectionReasons: jest.fn() },
}));

const mocked = apiClient.getResearchInboxRejectionReasons as jest.Mock;

function renderControl(onReject = jest.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <RejectWithReason onReject={onReject} />
    </QueryClientProvider>
  );
  return onReject;
}

const REASONS = {
  reasons: [
    {
      key: 'off_topic',
      label: 'Not my subject',
      teaches_topic: true,
      effect: 'Shows fewer items using these words.',
    },
    {
      key: 'low_quality',
      label: 'My subject, done badly',
      teaches_topic: false,
      effect: 'Changes nothing.',
    },
  ],
};

describe('RejectWithReason', () => {
  beforeEach(() => mocked.mockReset());

  it('sends the reason that was chosen', async () => {
    mocked.mockResolvedValue(REASONS);
    const onReject = renderControl();

    fireEvent.click(await screen.findByRole('button', { name: /Reject/ }));
    fireEvent.click(screen.getByText('My subject, done badly'));

    expect(onReject).toHaveBeenCalledWith('low_quality');
  });

  it('says what each choice will teach', async () => {
    // Three of the four choices deliberately teach nothing. A person who is not
    // told that will assume every rejection trains the profile.
    mocked.mockResolvedValue(REASONS);
    renderControl();

    fireEvent.click(await screen.findByRole('button', { name: /Reject/ }));
    expect(screen.getByText('Changes nothing.')).toBeInTheDocument();
  });

  it('still rejects when the vocabulary cannot be fetched', async () => {
    // An older backend must not make triage impossible; no reason is exactly
    // what every rejection meant before reasons existed. The menu gives way to
    // a plain Reject once the request has settled with nothing to offer — it
    // never becomes a dead end, and it never rejects on a click meant to open
    // a menu that was still loading.
    mocked.mockRejectedValue(new Error('404'));
    const onReject = renderControl();

    fireEvent.click(await screen.findByRole('button', { name: /Reject/ }));
    await waitFor(() =>
      expect(screen.queryByText('Reject because…')).not.toBeInTheDocument()
    );
    expect(onReject).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: /Reject/ }));
    expect(onReject).toHaveBeenCalledWith(undefined);
  });

  it('does not reject on opening the menu', async () => {
    mocked.mockResolvedValue(REASONS);
    const onReject = renderControl();

    fireEvent.click(await screen.findByRole('button', { name: /Reject/ }));
    expect(onReject).not.toHaveBeenCalled();
  });
});
