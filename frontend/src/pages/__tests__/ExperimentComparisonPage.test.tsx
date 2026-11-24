import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';

import ExperimentComparisonPage from '../ExperimentComparisonPage';
import { apiClient } from '../../services/api';
import type { ExperimentRun } from '../../types';

jest.mock('../../services/api', () => ({
  apiClient: { getExperimentPlan: jest.fn(), listExperimentRuns: jest.fn() },
}));

const api = jest.mocked(apiClient);
const route = '/research-notes/experiments/plan-1/compare';
const at = '2026-11-23T00:00:00Z';
const baseline: ExperimentRun = {
  id: 'base', user_id: 'owner', experiment_plan_id: 'plan-1', agent_job_id: 'job-base', name: 'Original workload',
  status: 'completed', benchmark_family: 'cpu', benchmark_suite_id: 'cache-study', benchmark_case_ids: ['trace-2', 'trace-1'],
  sandbox_profile_id: 'benchmark-sandbox', progress: 100, created_at: at, updated_at: at,
  measurement_summary: { runtime_ms: 100, cycles: 1500, baseline_measurements: { runtime_ms: 800 } },
  perf_counters: { branch_misses: 0, unverified_counter: 'missing' },
  artifact_inventory: ['compiler_logs'], compiler_artifacts: { log_paths: ['out/baseline.log'] },
};
const candidate: ExperimentRun = {
  ...baseline, id: 'next', agent_job_id: 'job-next', name: 'Modified workload', benchmark_case_ids: ['trace-1', 'trace-2'],
  measurement_summary: { runtime_ms: '80', retired_instructions: 12000 },
  perf_counters: { branch_misses: 4 },
  artifact_inventory: [], compiler_artifacts: null,
};

const LocationProbe = () => <output aria-label="Location">{useLocation().search}</output>;

function renderComparison(search = '') {
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } })}>
      <MemoryRouter initialEntries={[route + search]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Routes><Route path="/research-notes/experiments/:planId/compare" element={<ExperimentComparisonPage />} /></Routes>
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('ExperimentComparisonPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    api.getExperimentPlan.mockResolvedValue({
      id: 'plan-1', user_id: 'owner', research_note_id: 'note-1',
      title: 'Cache study', plan: {}, created_at: at, updated_at: at,
    });
    api.listExperimentRuns.mockResolvedValue({ runs: [candidate, baseline] });
  });

  it('uses matching recorded scopes, compares only observed values and links both original records', async () => {
    renderComparison();
    const table = within(await screen.findByRole('table', { name: 'Experiment measurements' }));
    expect(screen.getByRole('combobox', { name: 'Baseline' })).toHaveValue('base');
    expect(screen.getByRole('combobox', { name: 'Candidate' })).toHaveValue('next');
    await waitFor(() => expect(screen.getByLabelText('Location')).toHaveTextContent('?baseline=base&candidate=next'));
    expect(table.getByRole('row', { name: /Runtime/ })).toHaveTextContent('100 ms80 ms-20 ms-20%');
    expect(table.getByRole('row', { name: /branch misses/ })).toHaveTextContent('0' + '4' + '+4Zero baseline');
    expect(table.getByRole('row', { name: /Cycles/ })).toHaveTextContent('1,500Missing');
    expect(table.queryByText(/800|unverified counter/)).not.toBeInTheDocument();
    const evidence = within(screen.getByRole('article', { name: 'Baseline evidence' }));
    expect(evidence.getByText('out/baseline.log')).toBeInTheDocument();
    expect(evidence.getByRole('link', { name: 'Open baseline run' })).toHaveAttribute('href', '/research-notes?note=note-1&plan=plan-1&run=base');
    expect(evidence.getByRole('link', { name: 'Agent job' })).toHaveAttribute('href', '/autonomous-agents?job=job-base');

    fireEvent.click(screen.getByRole('button', { name: 'Swap runs' }));
    expect(screen.getByRole('combobox', { name: 'Baseline' })).toHaveValue('next');
    expect(table.getByRole('row', { name: /Runtime/ })).toHaveTextContent('80 ms100 ms+20 ms+25%');
    expect(screen.getByLabelText('Location')).toHaveTextContent('?baseline=next&candidate=base');
  });

  it('prefers a compatible baseline and explains why explicitly selected mismatched runs have no delta', async () => {
    const differentCase = { ...baseline, id: 'other', name: 'Different workload', benchmark_case_ids: ['trace-1'] };
    api.listExperimentRuns.mockResolvedValue({ runs: [candidate, differentCase, baseline] });
    renderComparison();
    const select = await screen.findByRole('combobox', { name: 'Baseline' });
    expect(select).toHaveValue('base');

    fireEvent.change(select, { target: { value: 'other' } });
    expect(screen.getByText('The runs cover different benchmark cases.')).toBeInTheDocument();
    const runtime = within(screen.getByRole('row', { name: /Runtime/ }));
    expect(runtime.getByText('80 ms')).toBeInTheDocument();
    expect(runtime.getByLabelText('Change unavailable')).toBeInTheDocument();
    expect(screen.queryByText('-20%')).not.toBeInTheDocument();
  });

  it('honors linked IDs and excludes planned estimates and baseline metadata', async () => {
    const planned: ExperimentRun = {
      ...candidate,
      status: 'planned',
      measurement_summary: { status: 'pending', runtime_ms: 80, baseline_measurements: { runtime_ms: 100 } },
      perf_counters: { branch_misses: 4 },
    };
    api.listExperimentRuns.mockResolvedValue({ runs: [planned, baseline] });
    renderComparison('?baseline=base&candidate=next');

    expect(await screen.findByText('Both runs need a successful completion for deltas.')).toBeInTheDocument();
    const table = within(screen.getByRole('table'));
    expect(table.getByRole('row', { name: /Runtime/ })).toHaveTextContent('100 msMissing');
    expect(table.queryByText('80 ms')).not.toBeInTheDocument();
  });

  it('keeps a missing linked run unselected so another run is never substituted', async () => {
    renderComparison('?baseline=deleted&candidate=next');
    expect(await screen.findByText('Choose two available runs from this plan.')).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Baseline' })).toHaveValue('');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });
});
