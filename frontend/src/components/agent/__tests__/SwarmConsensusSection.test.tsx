/**
 * What a swarm actually produced, as a reader meets it.
 *
 * The old merge could only ever say "the roles did not agree", because it
 * compared the prose of finding titles. These tests are about the four
 * verdicts that replaced it, and chiefly about the two that matter: a
 * disagreement, shown with both values, first -- and under it the run that
 * measured the same thing twice through an instrument too coarse to tell.
 */

import { fireEvent, render, screen } from '@testing-library/react';
import React from 'react';

import SwarmConsensusSection from '../SwarmConsensusSection';
import type { AgentJob } from '../../../types';

const group = (over = {}) => ({
  finding_type: 'benchmark_measurement',
  subject: 'dotprod',
  verdict: 'corroborated',
  detail: 'researcher 12, verifier 12.3 (within 10%)',
  roles: ['researcher', 'verifier'],
  ...over,
});

const renderWith = (fanIn: any) =>
  render(
    <SwarmConsensusSection
      job={{ id: 'job-1', results: { swarm_fan_in: fanIn } } as unknown as AgentJob}
    />
  );

it('puts a disagreement first, with both values', async () => {
  // Four agents agreeing is mildly reassuring; two measuring 12ms and 48ms for
  // the same kernel is the reason you ran four agents.
  renderWith({
    contested: [
      group({
        verdict: 'contested',
        subject: 'dotprod',
        detail: 'researcher 12, verifier 48 — 300% apart, beyond 10%',
      }),
    ],
    corroborated: [group({ subject: 'matmul' })],
    uncorroborated: [],
    typed_agreement: 0.5,
  });

  expect(screen.getByText(/1 contested/)).toBeInTheDocument();
  expect(screen.getByText(/researcher 12, verifier 48/)).toBeInTheDocument();
  expect(screen.getByText('50% agreement')).toBeInTheDocument();
});

it('says nothing was cross-checked rather than showing 0%', async () => {
  // Nothing checked is a different statement from everything checked having
  // disagreed, and 0% would assert the second.
  renderWith({
    contested: [],
    corroborated: [],
    uncorroborated: [group({ verdict: 'uncorroborated', roles: ['researcher'] })],
    typed_agreement: null,
  });

  expect(screen.getByText('nothing cross-checked')).toBeInTheDocument();
  expect(screen.queryByText(/0% agreement/)).toBeNull();
});

it('keeps single-role findings out of the way but reachable', async () => {
  renderWith({
    contested: [],
    corroborated: [group()],
    uncorroborated: [
      group({ verdict: 'uncorroborated', subject: 'solo', roles: ['critic'] }),
    ],
    typed_agreement: 1,
  });

  expect(screen.queryByText(/· solo/)).toBeNull();
  fireEvent.click(screen.getByText(/only one role spoke to/));
  expect(screen.getByText(/· solo/)).toBeInTheDocument();
});

it('renders nothing for a run that was not a swarm', async () => {
  const { container } = render(
    <SwarmConsensusSection job={{ id: 'j', results: {} } as unknown as AgentJob} />
  );

  expect(container.querySelector('[data-testid="swarm-consensus-section"]')).toBeNull();
});


it('shows an unresolved measurement, and does not call it agreement', async () => {
  // The measured case: two roles reporting 130% and 142% of their own
  // variation. The old scoring called 60ms vs 45ms "corroborated, 100%
  // agreement" -- true arithmetic, misleading to read, because at that width
  // two numbers 2.4x apart would have passed too.
  renderWith({
    inconclusive: [
      group({
        verdict: 'inconclusive',
        detail:
          'researcher 60, verifier 45 — 33% apart, but these measurements ' +
          'report 142% of their own variation, too noisy to tell agreement ' +
          'from coincidence',
      }),
    ],
    typed_agreement: null,
  });

  expect(await screen.findByText(/measured too noisily/i)).toBeInTheDocument();
  expect(screen.getByText(/142% of their own variation/)).toBeInTheDocument();
  expect(screen.queryByText(/100% agreement/)).not.toBeInTheDocument();
});

it('does not claim nothing was cross-checked when something was', async () => {
  // Both states leave typed_agreement null, and they mean opposite things:
  // nobody checked, versus the check could not resolve. Saying the first when
  // the second happened tells the reader the swarm did less than it did.
  renderWith({
    inconclusive: [group({ verdict: 'inconclusive', detail: 'too noisy' })],
    typed_agreement: null,
  });

  expect(screen.getByText(/cross-checked, unresolved/i)).toBeInTheDocument();
  expect(screen.queryByText(/nothing cross-checked/i)).not.toBeInTheDocument();
});

it('still puts a real disagreement above an unresolved one', async () => {
  renderWith({
    contested: [
      group({ verdict: 'contested', subject: 'alpha', detail: '300% apart' }),
    ],
    inconclusive: [
      group({ verdict: 'inconclusive', subject: 'beta', detail: 'too noisy' }),
    ],
    typed_agreement: 0,
  });

  const body = document.body.textContent || '';
  expect(body.indexOf('contested')).toBeGreaterThanOrEqual(0);
  expect(body.indexOf('contested')).toBeLessThan(body.indexOf('unresolved'));
});
