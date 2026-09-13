/**
 * The environment a run worked in, as a reader meets it.
 *
 * The three states are the substance here, because each means something
 * different and collapsing any two of them misinforms: a run that never made a
 * workspace (nothing to say), one whose files are there (read them), and one
 * whose files were swept (the reason a measurement cannot be re-derived).
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';

import WorkspaceSection from '../WorkspaceSection';
import type { AgentJob, AgentJobWorkspace } from '../../../types';

jest.mock('../../../services/api', () => ({
  apiClient: { getJobWorkspace: jest.fn(), getJobWorkspaceFile: jest.fn() },
}));

const { apiClient } = jest.requireMock('../../../services/api');

const workspace = (over: Partial<AgentJobWorkspace> = {}): AgentJobWorkspace => ({
  job_id: 'job-1',
  workspace_id: 'ws-1',
  status: 'retained',
  source_id: null,
  repo_url: 'https://example.invalid/repo.git',
  branch: 'main',
  path: '.',
  entries: [
    { path: 'kernel.c', is_dir: false, size: 26, changed: true },
    { path: 'README.md', is_dir: false, size: 4, changed: false },
  ],
  truncated: false,
  modified: ['kernel.c'],
  added: [],
  deleted: [],
  ...over,
});

const renderSection = () =>
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <WorkspaceSection job={{ id: 'job-1' } as AgentJob} />
    </QueryClientProvider>
  );

afterEach(() => jest.clearAllMocks());

it('leads with what the run changed', async () => {
  // The one thing looking at the directory cannot tell you. It exists only
  // because the hashes of the starting files were kept.
  apiClient.getJobWorkspace.mockResolvedValue(
    workspace({ modified: ['kernel.c'], added: ['bench.c'], deleted: ['gone.c'] })
  );

  renderSection();

  expect(await screen.findByText(/3 files touched/)).toBeInTheDocument();
  expect(screen.getByText('1 modified')).toBeInTheDocument();
  expect(screen.getByText('1 added')).toBeInTheDocument();
  expect(screen.getByText('1 deleted')).toBeInTheDocument();
});

it('names deleted files, which cannot appear in the tree', async () => {
  apiClient.getJobWorkspace.mockResolvedValue(
    workspace({ deleted: ['gone.c', 'stale.h'] })
  );

  renderSection();

  expect(await screen.findByText(/gone.c, stale.h/)).toBeInTheDocument();
});

it('marks the files the run touched', async () => {
  apiClient.getJobWorkspace.mockResolvedValue(workspace());

  renderSection();

  expect(await screen.findByText('kernel.c')).toBeInTheDocument();
  expect(screen.getByText('README.md')).toBeInTheDocument();
  // One badge, on the file that changed.
  expect(screen.getAllByText('changed')).toHaveLength(1);
});

it('shows the real change when git can answer', async () => {
  // A workspace cloned from a repository keeps its .git, so the patch is
  // already there and only needed asking for.
  apiClient.getJobWorkspace.mockResolvedValue(workspace());
  apiClient.getJobWorkspaceFile.mockResolvedValue({
    job_id: 'job-1',
    workspace_id: 'ws-1',
    path: 'kernel.c',
    content: 'int main(void){return 42;}',
    changed: true,
    diff: '@@ -1 +1 @@\n-int main(void){return 0;}\n+int main(void){return 42;}',
    diff_available: true,
  });

  renderSection();
  fireEvent.click(await screen.findByText('kernel.c'));

  expect(await screen.findByText('-int main(void){return 0;}')).toBeInTheDocument();
  expect(screen.getByText('+int main(void){return 42;}')).toBeInTheDocument();
});

it('says the change cannot be shown when nothing can answer', async () => {
  // A knowledge-base workspace has no repository and no stored originals.
  // "Cannot be shown" is not "nothing changed", and saying the wrong one is a
  // confident wrong answer.
  apiClient.getJobWorkspace.mockResolvedValue(workspace());
  apiClient.getJobWorkspaceFile.mockResolvedValue({
    job_id: 'job-1',
    workspace_id: 'ws-1',
    path: 'kernel.c',
    content: 'int main(void){return 42;}',
    changed: true,
    diff: null,
    diff_available: false,
  });

  renderSection();
  fireEvent.click(await screen.findByText('kernel.c'));

  expect(await screen.findByText(/change itself cannot be shown/)).toBeInTheDocument();
});

it('tells an unchanged tracked file from one it cannot speak for', async () => {
  apiClient.getJobWorkspace.mockResolvedValue(workspace());
  apiClient.getJobWorkspaceFile.mockResolvedValue({
    job_id: 'job-1',
    workspace_id: 'ws-1',
    path: 'README.md',
    content: 'docs',
    changed: false,
    diff: '',
    diff_available: true,
  });

  renderSection();
  fireEvent.click(await screen.findByText('README.md'));

  expect(await screen.findByText(/unchanged since the run started/)).toBeInTheDocument();
});

it('says so when the files were swept', async () => {
  // "The evidence is gone" is exactly what a reader needs when a measurement
  // cannot be re-derived. Silence would read as "this run did no coding".
  apiClient.getJobWorkspace.mockRejectedValue({ response: { status: 410 } });

  renderSection();

  expect(
    await screen.findByText(/removed after the retention window/)
  ).toBeInTheDocument();
});

it('renders nothing for a run that never made a workspace', async () => {
  // Most runs are not coding runs, and this is mounted for all of them.
  apiClient.getJobWorkspace.mockRejectedValue({ response: { status: 404 } });

  const { container } = renderSection();

  await waitFor(() => expect(apiClient.getJobWorkspace).toHaveBeenCalled());
  await waitFor(() =>
    expect(container.querySelector('[data-testid="workspace-section"]')).toBeNull()
  );
});

it('does not offer to open a directory', async () => {
  apiClient.getJobWorkspace.mockResolvedValue(
    workspace({ entries: [{ path: 'src', is_dir: true, size: 0, changed: false }] })
  );

  renderSection();

  const row = (await screen.findByText('src')).closest('button');
  expect(row).toBeDisabled();
});
