/**
 * When a run's output is visible at all.
 *
 * This markup used to live inside the customer-research card, whose condition
 * is a customer profile, domain research, a document artifact, a reading list
 * or an arXiv source. A coding run has none of those — its artifact is a
 * `code_patch_proposal` — so the patch it produced was rendered into a
 * container that never opened. These tests are the condition, stated.
 */

import { render, screen } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter } from 'react-router-dom';

import { RunOutputSection } from '../RunOutputSection';
import type { AgentJob } from '../../../types';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

jest.mock('../../../services/api', () => ({
  apiClient: {
    downloadCodePatchProposal: jest.fn(),
    applyCodePatchProposal: jest.fn(),
  },
}));

const renderSection = (job: Partial<AgentJob>) =>
  render(
    <MemoryRouter>
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <RunOutputSection
          job={{ id: 'job-1', name: 'run', status: 'completed', ...job } as AgentJob}
          operatorInterventionSummary={
            { latestLabel: '', latestOutcome: '', latestOutcomeReason: '', recentItems: [] } as any
          }
          graphHealthReasons={[]}
          openCheckpointQueue={jest.fn()}
          openDocument={jest.fn()}
          createMutation={{ isLoading: false, mutate: jest.fn() }}
          unsafeExecBadge={{ status: 'off', label: '', title: '', color: '' }}
        />
      </QueryClientProvider>
    </MemoryRouter>
  );

it('shows a coding run its patch, with no research artifact in sight', () => {
  // The regression: this job has no customer profile, no document artifact,
  // no reading list and no arXiv source. It used to render nothing.
  renderSection({
    results: { code_patch: { proposal_id: 'p-42', title: 'Fix the off-by-one' } } as any,
    output_artifacts: [{ type: 'code_patch_proposal', id: 'p-42', title: 'Fix the off-by-one' }] as any,
  });

  expect(screen.getByText('Run output')).toBeInTheDocument();
  expect(screen.getByText('Code patch')).toBeInTheDocument();
  expect(screen.getByText('Fix the off-by-one')).toBeInTheDocument();
  expect(screen.getByText('p-42')).toBeInTheDocument();
});

it('renders nothing for a run that produced none of it', () => {
  // A pure research run: its output belongs in the research card, not here.
  const { container } = renderSection({
    results: { customer_profile: { name: 'Acme' }, findings: [] } as any,
  });

  expect(container).toBeEmptyDOMElement();
});

it('appears for a generated project alone', () => {
  renderSection({
    results: {
      generated_project: {
        source_id: 'src-9',
        source_name: 'demo-proj',
        project_name: 'demo-proj',
        file_count: 4,
      },
    } as any,
  });

  expect(screen.getByText('Run output')).toBeInTheDocument();
});

it('appears for a demo check alone', () => {
  renderSection({
    results: { demo_check: { source_id: 'src-9', entrypoint: 'demo.py', ok: true } } as any,
  });

  expect(screen.getByText('Run output')).toBeInTheDocument();
});
