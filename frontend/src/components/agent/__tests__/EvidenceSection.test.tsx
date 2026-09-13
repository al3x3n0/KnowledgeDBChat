/**
 * The distinctions a findings count destroys.
 *
 * `Findings: 4` told you four things existed. These tests are the four
 * questions it could not answer, each of which is a reason to disbelieve a
 * number that otherwise looks fine: was the thing I asked for produced at all,
 * does it carry a spread, what machine was it measured on, and is it the kind
 * of evidence a later change invalidates.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';

import EvidenceSection from '../EvidenceSection';
import type { AgentJob, AgentJobEvidence } from '../../../types';

jest.mock('../../../services/api', () => ({
  apiClient: {
    getJobEvidence: jest.fn(),
    disputeJobEvidence: jest.fn(),
    withdrawJobEvidenceDispute: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: { success: jest.fn(), error: jest.fn() },
}));

const { apiClient } = jest.requireMock('../../../services/api');

const benchmark = (over = {}) => ({
  index: 0,
  type: 'benchmark_measurement',
  title: 'kernel @ c -O3',
  values: [
    { label: 'fastest_ms', value: '12.5' },
    { label: 'measurement_environment', value: 'quiet' },
  ],
  has_uncertainty: true,
  measurement_environment: 'quiet',
  warning: '',
  perishable: false,
  disputed: false,
  dispute_reason: '',
  ...over,
});

const evidence = (over: Partial<AgentJobEvidence> = {}): AgentJobEvidence => ({
  job_id: 'job-1',
  evidence: [benchmark()],
  requirements: [
    {
      finding_type: 'benchmark_measurement',
      satisfied: true,
      satisfied_by: [0],
      uncertainty_required: false,
      missing_uncertainty: [],
    },
  ],
  unrequested: [],
  unrequested_total: 0,
  contract_enabled: true,
  contract_satisfied: true,
  disputed_count: 0,
  missing: [],
  unsettled_predictions: [],
  ...over,
});

const renderSection = () =>
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <EvidenceSection job={{ id: 'job-1' } as AgentJob} />
    </QueryClientProvider>
  );

afterEach(() => jest.clearAllMocks());

it('shows a requirement that nothing answered', async () => {
  // The most important row on the screen: the thing that was asked for is not
  // there. A count of the findings that DID arrive hides exactly this.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      requirements: [
        {
          finding_type: 'reproduction_verdict',
          satisfied: false,
          satisfied_by: [],
          uncertainty_required: false,
          missing_uncertainty: [],
        },
      ],
    })
  );

  renderSection();

  expect(await screen.findByText('reproduction_verdict')).toBeInTheDocument();
  expect(screen.getByText(/Nothing of this type was produced/)).toBeInTheDocument();
});

it('names the finding that arrived without its required spread', async () => {
  // It did arrive. It is not a result. Those are different states and the
  // contract knows which findings are at fault.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [benchmark({ has_uncertainty: false })],
      requirements: [
        {
          finding_type: 'benchmark_measurement',
          satisfied: true,
          satisfied_by: [0],
          uncertainty_required: true,
          missing_uncertainty: [0],
        },
      ],
      contract_satisfied: false,
    })
  );

  renderSection();

  expect(await screen.findByText(/1 of 1 arrived without the spread/)).toBeInTheDocument();
  expect(screen.getByText('spread required')).toBeInTheDocument();
  expect(screen.getByText('no spread')).toBeInTheDocument();
});

it('shows the machine a number was measured on', async () => {
  // A wall-clock number taken on a saturated host is not a measurement.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [
        benchmark({
          measurement_environment: 'saturated',
          warning: 'host was saturated; treat as advisory',
        }),
      ],
    })
  );

  renderSection();

  expect(await screen.findByText('saturated')).toBeInTheDocument();
  expect(screen.getByText(/treat as advisory/)).toBeInTheDocument();
});

it('marks perishable evidence', async () => {
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({ evidence: [benchmark({ perishable: true })] })
  );

  renderSection();

  expect(await screen.findByText('perishable')).toBeInTheDocument();
});

it('keeps evidence nobody asked for out of the way but reachable', async () => {
  // 4,247 `document` findings against 97 benchmarks in this database. Mixed
  // in, the required evidence cannot be found; dropped, a reader loses the
  // run's own account of what it learnt.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [
        benchmark(),
        { ...benchmark(), index: 1, type: 'document', title: 'some paper' },
      ],
      unrequested: [1],
    })
  );

  renderSection();

  const toggle = await screen.findByText(/1 finding nothing asked for/);
  expect(screen.queryByText('document')).toBeNull();

  fireEvent.click(toggle);

  expect(await screen.findByText('document')).toBeInTheDocument();
});

it('reports a prediction the run never settled', async () => {
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({ unsettled_predictions: ['p1', 'p2'] })
  );

  renderSection();

  expect(
    await screen.findByText(/2 predictions recorded and never settled/)
  ).toBeInTheDocument();
});

it('renders nothing for a run with no evidence', async () => {
  // Mounted for every run, and most runs in this database are not research
  // runs.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({ evidence: [], requirements: [], contract_enabled: false })
  );

  const { container } = renderSection();

  await waitFor(() => expect(apiClient.getJobEvidence).toHaveBeenCalled());
  await waitFor(() =>
    expect(container.querySelector('[data-testid="evidence-section"]')).toBeNull()
  );
});

it('renders nothing when the run is not visible to this user', async () => {
  apiClient.getJobEvidence.mockRejectedValue({ response: { status: 404 } });

  const { container } = renderSection();

  await waitFor(() => expect(apiClient.getJobEvidence).toHaveBeenCalled());
  await waitFor(() =>
    expect(container.querySelector('[data-testid="evidence-section"]')).toBeNull()
  );
});

it('records a rejection with its reason', async () => {
  // The reason is the whole point: a later run has to tell a result rejected
  // for a harness defect from one rejected because the question changed.
  apiClient.getJobEvidence.mockResolvedValue(evidence());
  apiClient.disputeJobEvidence.mockResolvedValue({
    job_id: 'job-1',
    index: 0,
    reason: 'host was saturated',
    advisory: true,
  });

  renderSection();

  fireEvent.click(await screen.findByLabelText('Reject finding 0'));
  fireEvent.change(screen.getByLabelText(/Why is this result not to be believed/), {
    target: { value: 'host was saturated' },
  });
  fireEvent.click(screen.getByRole('button', { name: /^Reject$/ }));

  await waitFor(() =>
    expect(apiClient.disputeJobEvidence).toHaveBeenCalledWith(
      'job-1',
      0,
      'host was saturated'
    )
  );
});

it('shows a rejected result in full, with the reason, still under its requirement', async () => {
  // It happened. Hiding it would lose the run's own account, and moving it
  // would suggest the contract is no longer satisfied — which it is, because
  // the rejection is advisory.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [benchmark({ disputed: true, dispute_reason: 'host was saturated' })],
      disputed_count: 1,
      contract_satisfied: true,
    })
  );

  renderSection();

  expect(await screen.findByText('rejected')).toBeInTheDocument();
  expect(screen.getByText(/Rejected: host was saturated/)).toBeInTheDocument();
  expect(screen.getByText('Contract met')).toBeInTheDocument();
  expect(screen.getByText('1 rejected')).toBeInTheDocument();
});

it('offers to withdraw a rejection rather than reject twice', async () => {
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [benchmark({ disputed: true, dispute_reason: 'a reason' })],
      disputed_count: 1,
    })
  );
  apiClient.withdrawJobEvidenceDispute.mockResolvedValue({
    job_id: 'job-1',
    index: 0,
    withdrawn: 1,
  });

  renderSection();

  expect(screen.queryByLabelText('Reject finding 0')).toBeNull();
  fireEvent.click(await screen.findByLabelText('Withdraw the rejection of finding 0'));

  await waitFor(() =>
    expect(apiClient.withdrawJobEvidenceDispute).toHaveBeenCalledWith('job-1', 0)
  );
});

it('says when it is showing a sample rather than the whole', async () => {
  // The cap exists because this database holds thousands of `document`
  // findings against a handful of measurements. A sample presented as the
  // whole is worse than a sample.
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [
        benchmark(),
        { ...benchmark(), index: 1, type: 'document', title: 'a paper' },
      ],
      unrequested: [1],
      unrequested_total: 4247,
    })
  );

  renderSection();

  expect(await screen.findByText(/1 finding nothing asked for/)).toBeInTheDocument();
  expect(screen.getByText(/\(of 4247\)/)).toBeInTheDocument();
});

it('does not claim a sample when it has everything', async () => {
  apiClient.getJobEvidence.mockResolvedValue(
    evidence({
      evidence: [
        benchmark(),
        { ...benchmark(), index: 1, type: 'document', title: 'a paper' },
      ],
      unrequested: [1],
      unrequested_total: 1,
    })
  );

  renderSection();

  await screen.findByText(/1 finding nothing asked for/);
  expect(screen.queryByText(/\(of /)).toBeNull();
});
