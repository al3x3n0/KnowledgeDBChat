/**
 * The memories section, now that it can be rendered on its own.
 *
 * Inside the 4,000-line panel these behaviours could only be reached by
 * rendering the whole job detail view with a job in the right state. Three of
 * them had never been checked at all.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';

import { JobMemoriesSection } from '../JobMemoriesSection';
import type { AgentJob } from '../../../types';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    getJobMemories: jest.fn(),
    extractJobMemories: jest.fn(),
  },
}));

const apiClient = require('../../../services/api').apiClient;
const toast = require('react-hot-toast').default;

const jobWith = (status: string): AgentJob =>
  ({ id: 'job-1', name: 'run', status } as unknown as AgentJob);

const memory = (over: Record<string, any> = {}) => ({
  id: 'm-1',
  type: 'insight',
  content: 'The arithmetic is not the ceiling',
  importance_score: 0.82,
  tags: ['int8', 'attention'],
  ...over,
});

beforeEach(() => {
  jest.clearAllMocks();
  apiClient.getJobMemories.mockResolvedValue({ memories: [memory()], total: 1 });
});

it('asks for nothing until the memories are shown', async () => {
  render(<JobMemoriesSection job={jobWith('completed')} onExtracted={jest.fn()} />);

  expect(apiClient.getJobMemories).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Show Memories' }));

  expect(await screen.findByText('The arithmetic is not the ceiling')).toBeInTheDocument();
  expect(screen.getByText('82% importance')).toBeInTheDocument();
  expect(apiClient.getJobMemories).toHaveBeenCalledTimes(1);
});

it('does not refetch when reopened', async () => {
  render(<JobMemoriesSection job={jobWith('completed')} onExtracted={jest.fn()} />);

  fireEvent.click(screen.getByRole('button', { name: 'Show Memories' }));
  await screen.findByText('The arithmetic is not the ceiling');
  fireEvent.click(screen.getByRole('button', { name: 'Hide Memories' }));
  fireEvent.click(screen.getByRole('button', { name: 'Show Memories' }));

  await screen.findByText('The arithmetic is not the ceiling');
  expect(apiClient.getJobMemories).toHaveBeenCalledTimes(1);
});

it('offers extraction only once the run has ended', () => {
  const { rerender } = render(
    <JobMemoriesSection job={jobWith('running')} onExtracted={jest.fn()} />
  );
  expect(
    screen.queryByRole('button', { name: 'Extract memories from job results' })
  ).not.toBeInTheDocument();

  rerender(<JobMemoriesSection job={jobWith('failed')} onExtracted={jest.fn()} />);
  // A failed run still produced something worth remembering.
  expect(
    screen.getByRole('button', { name: 'Extract memories from job results' })
  ).toBeInTheDocument();
});

it('hands the extraction numbers back up, and reloads the list', async () => {
  // The persistence panel above shows these, preferring a manual run's
  // numbers over the job's own — so they have to leave this component.
  apiClient.extractJobMemories.mockResolvedValue({
    status: 'completed',
    created_count: 3,
    skipped_duplicates: 2,
  });
  const onExtracted = jest.fn();

  render(<JobMemoriesSection job={jobWith('completed')} onExtracted={onExtracted} />);
  fireEvent.click(screen.getByRole('button', { name: 'Extract memories from job results' }));

  await waitFor(() => expect(onExtracted).toHaveBeenCalledTimes(1));
  expect(onExtracted.mock.calls[0][0]).toMatchObject({
    created_count: 3,
    skipped_duplicates: 2,
  });
  expect(toast.success).toHaveBeenCalledWith('Extracted 3 memories (2 duplicates skipped)');
  await waitFor(() => expect(apiClient.getJobMemories).toHaveBeenCalledTimes(1));
});

it('reports a failed extraction rather than a silent one', async () => {
  apiClient.extractJobMemories.mockRejectedValue(new Error('LLM unavailable'));
  const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  const onExtracted = jest.fn();

  render(<JobMemoriesSection job={jobWith('completed')} onExtracted={onExtracted} />);
  fireEvent.click(screen.getByRole('button', { name: 'Extract memories from job results' }));

  await waitFor(() => expect(toast.error).toHaveBeenCalledWith('LLM unavailable'));
  expect(onExtracted).not.toHaveBeenCalled();
  // And the button comes back, rather than staying disabled for ever.
  await waitFor(() =>
    expect(
      screen.getByRole('button', { name: 'Extract memories from job results' })
    ).not.toBeDisabled()
  );
  expect(consoleErrorSpy).toHaveBeenCalled();
  consoleErrorSpy.mockRestore();
});

it('offers extraction from the empty state too', async () => {
  apiClient.getJobMemories.mockResolvedValue({ memories: [], total: 0 });

  render(<JobMemoriesSection job={jobWith('completed')} onExtracted={jest.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: 'Show Memories' }));

  expect(await screen.findByText('No memories extracted yet')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Extract Memories/ })).toBeInTheDocument();
});
