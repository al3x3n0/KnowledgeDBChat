import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import apiClient from '../../services/api';
import PatchPRsPage from '../PatchPRsPage';

jest.mock('../../services/api', () => ({
  __esModule: true,
  default: {
    listPatchPRs: jest.fn(),
    getPatchPR: jest.fn(),
    approvePatchPR: jest.fn(),
  },
}));

test('shows the diff and matching checks and sends the reviewed digest for approval', async () => {
  const patch = {
    id: 'patch-1',
    source_id: 'repo-1',
    title: 'Fix parser',
    status: 'open',
    selected_proposal_id: 'proposal-1',
    diff_unified: '--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-before\n+after',
    diff_sha256: 'a'.repeat(64),
    checks: {
      experiment_runs: [
        { proposal_id: 'proposal-1', ok: true, commands: ['pytest'] },
        { proposal_id: 'old-proposal', ok: false, commands: ['stale check'] },
      ],
    },
  };
  (apiClient.listPatchPRs as jest.Mock).mockResolvedValue({ items: [patch], total: 1 });
  (apiClient.getPatchPR as jest.Mock).mockResolvedValue(patch);
  (apiClient.approvePatchPR as jest.Mock).mockResolvedValue({
    ...patch, status: 'approved', approval_valid: true,
  });
  render(
    <MemoryRouter initialEntries={['/patch-prs?pr=patch-1']}>
      <PatchPRsPage />
    </MemoryRouter>
  );

  const diff = await screen.findByRole('region', { name: 'Selected patch diff' });
  expect(within(diff).getByText('+after')).toBeInTheDocument();
  expect(screen.getByText('pytest')).toBeInTheDocument();
  expect(screen.queryByText('stale check')).toBeNull();
  expect(screen.getByRole('button', { name: 'Apply to Library' })).toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Approve diff' }));
  await waitFor(() => expect(apiClient.approvePatchPR).toHaveBeenCalledWith(
    patch.id, { diff_sha256: patch.diff_sha256 },
  ));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Apply to Library' })).toBeEnabled());
});
