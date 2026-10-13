import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

import AnswerSources, { scorePercent } from '../AnswerSources';

jest.mock('../../../services/api', () => ({
  apiClient: { getRetrievalTrace: jest.fn() },
}));

const apiClient = require('../../../services/api').apiClient;

const OUTPUT = {
  answer: 'A stride prefetcher tracks deltas.',
  confidence: 'high',
  retrieval_trace_id: 'trace-1',
  sources: [
    { document_id: 'd1', title: 'Prefetching survey', relevance_score: 0.82, excerpt: 'Stride tracks deltas.' },
    { document_id: 'd2', title: 'Cache notes', relevance_score: 0.51 },
  ],
};

describe('AnswerSources', () => {
  beforeEach(() => jest.clearAllMocks());

  it('lists the documents the answer was written from', () => {
    render(<AnswerSources output={OUTPUT} />);

    expect(screen.getByText('Prefetching survey')).toBeInTheDocument();
    expect(screen.getByText('Relevance: 82%')).toBeInTheDocument();
    expect(screen.getByText('Stride tracks deltas.')).toBeInTheDocument();
    expect(screen.getByText('Cache notes')).toBeInTheDocument();
    expect(screen.getByText('Confidence: high')).toBeInTheDocument();
  });

  it('loads the retrieval record only when asked, and once', async () => {
    apiClient.getRetrievalTrace.mockResolvedValue({
      id: 'trace-1',
      trace_type: 'tool',
      query: 'How does stride work?',
      processed_query: 'stride prefetcher',
      trace: { selected_results: [{ id: 'c1', title: 'Prefetching survey', score: 0.82 }] },
      created_at: '2026-10-13T00:00:00Z',
    });
    render(<AnswerSources output={OUTPUT} />);
    expect(apiClient.getRetrievalTrace).not.toHaveBeenCalled();

    const button = screen.getByRole('button', { name: /How this was retrieved/ });
    fireEvent.click(button);

    expect(await screen.findByText('stride prefetcher')).toBeInTheDocument();
    expect(screen.getByText('Prefetching survey (82%)')).toBeInTheDocument();
    expect(apiClient.getRetrievalTrace).toHaveBeenCalledWith('trace-1');

    fireEvent.click(button);
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByText('stride prefetcher')).toBeInTheDocument());
    expect(apiClient.getRetrievalTrace).toHaveBeenCalledTimes(1);
  });

  it('says so when the record cannot be loaded', async () => {
    apiClient.getRetrievalTrace.mockRejectedValue(new Error('403'));
    render(<AnswerSources output={OUTPUT} />);

    fireEvent.click(screen.getByRole('button', { name: /How this was retrieved/ }));

    expect(await screen.findByText('The retrieval record could not be loaded.')).toBeInTheDocument();
  });

  it('offers no record when the answer has none', () => {
    render(<AnswerSources output={{ answer: 'x', sources: [] }} />);

    expect(screen.getByText('No sources: nothing relevant was found.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /How this was retrieved/ })).not.toBeInTheDocument();
  });
});

describe('scorePercent', () => {
  it('shows a score as a percentage and nothing for anything else', () => {
    expect(scorePercent(0.826)).toBe('83%');
    expect(scorePercent(0)).toBe('0%');
    expect(scorePercent(1.7)).toBe('100%');
    expect(scorePercent(undefined)).toBeNull();
    expect(scorePercent('0.5')).toBeNull();
    expect(scorePercent(NaN)).toBeNull();
  });
});
