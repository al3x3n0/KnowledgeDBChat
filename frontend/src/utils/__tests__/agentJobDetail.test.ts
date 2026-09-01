/**
 * The execution-graph derivations, tested without rendering anything.
 *
 * These fourteen values used to be computed inline in a 4,000-line component,
 * which meant the only way to check them was to render the whole panel. As a
 * pure function of a job they can be asked directly — including the cases a
 * rendered test would never set up, like a job whose results are absent.
 */

import { codePatchView, executionGraphView } from '../agentJobDetail';
import type { AgentJob } from '../../types';

const jobWith = (results: any): AgentJob =>
  ({ id: 'j-1', name: 'run', status: 'running', results } as unknown as AgentJob);

describe('executionGraphView', () => {
  it('reads nothing from a job with no results at all', () => {
    const view = executionGraphView(jobWith(undefined));
    expect(view.executionGraph).toBeNull();
    expect(view.scopeObservability).toBeNull();
    expect(view.graphHealthStatus).toBe('');
    expect(view.scopeEvents).toEqual([]);
    expect(view.scopeGuardBlocks).toBe(0);
    // An absent graph must not colour the badge as though it were healthy.
    expect(view.graphHealthBadgeClass).toContain('gray');
  });

  it('colours the health badge by status', () => {
    const badge = (status: string) =>
      executionGraphView(
        jobWith({ execution_strategy: { execution_graph: { graph_health: { status } } } })
      ).graphHealthBadgeClass;

    expect(badge('critical')).toContain('red');
    expect(badge('warning')).toContain('amber');
    expect(badge('ok')).toContain('emerald');
    expect(badge('something else')).toContain('gray');
    // Status is compared lowercased, so casing from the backend cannot change
    // the colour.
    expect(badge('CRITICAL')).toContain('red');
  });

  it('caps recommended actions and drops blank ones', () => {
    const view = executionGraphView(
      jobWith({
        execution_strategy: {
          execution_graph: {
            recommended_actions: ['a', '', '   ', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
          },
        },
      })
    );
    expect(view.graphRecommendedActions).toHaveLength(6);
    expect(view.graphRecommendedActions).not.toContain('');
  });

  it('shows the four most recent scope events, newest first', () => {
    const events = [1, 2, 3, 4, 5, 6].map((n) => ({ type: 'step', n }));
    const view = executionGraphView(
      jobWith({ execution_strategy: { scope_observability: { events } } })
    );

    expect(view.scopeEvents).toHaveLength(6);
    expect(view.recentScopeEvents.map((e: any) => e.n)).toEqual([6, 5, 4, 3]);
  });

  it('counts only the guard blocks among the events', () => {
    const view = executionGraphView(
      jobWith({
        execution_strategy: {
          scope_observability: {
            events: [
              { type: 'scope_guard_blocked' },
              { type: 'step' },
              { type: ' scope_guard_blocked ' },
              { type: 'scope_guard_allowed' },
            ],
          },
        },
      })
    );
    // The type is trimmed before comparison, so a padded value still counts.
    expect(view.scopeGuardBlocks).toBe(2);
  });

  it('treats a non-array events field as no events rather than throwing', () => {
    const view = executionGraphView(
      jobWith({ execution_strategy: { scope_observability: { events: 'not an array' } } })
    );
    expect(view.scopeEvents).toEqual([]);
    expect(view.recentScopeEvents).toEqual([]);
  });
});

describe('codePatchView', () => {
  it('prefers the proposal recorded in results over an output artifact', () => {
    const view = codePatchView(
      jobWith({ code_patch: { proposal_id: 'p-1', title: 'From results' } })
    );
    expect(view.codePatchProposal?.proposal_id).toBe('p-1');
    expect(view.codePatchProposal?.title).toBe('From results');
  });

  it('falls back to a code_patch_proposal artifact when results carry none', () => {
    const job = {
      id: 'j-1',
      name: 'run',
      status: 'running',
      results: {},
      output_artifacts: [
        { type: 'chart', id: 'a-0' },
        { type: 'code_patch_proposal', id: 'a-1', title: 'From artifact' },
      ],
    } as unknown as AgentJob;
    const view = codePatchView(job);
    expect(view.codePatchProposal?.proposal_id).toBe('a-1');
    expect(view.codePatchProposal?.title).toBe('From artifact');
  });

  it('deduplicates the proposal history by id', () => {
    const view = codePatchView(
      jobWith({
        code_patches: [
          { proposal_id: 'p-1', title: 'First' },
          { proposal_id: 'p-1', title: 'Same id again' },
          { proposal_id: '  ', title: 'Blank id' },
          { proposal_id: 'p-2', title: 'Second' },
        ],
      })
    );
    expect(view.codePatchProposals.map((p) => p.proposal_id)).toEqual(['p-1', 'p-2']);
  });

  it('reads recovery state lowercased and trimmed', () => {
    const view = codePatchView(
      jobWith({ code_patch_execution: { recovery: { recovery_state: '  AWAITING_OPERATOR ' } } })
    );
    expect(view.codingRecoveryState).toBe('awaiting_operator');
  });

  it('returns empty collections rather than undefined for a bare job', () => {
    const view = codePatchView(jobWith(undefined));
    expect(view.codePatchProposal).toBeNull();
    expect(view.codePatchExecution).toBeNull();
    expect(view.codePatchProposals).toEqual([]);
    expect(view.codePatchExecutionPlan).toEqual([]);
    expect(view.codePatchDetectedStack).toEqual([]);
    expect(view.codingRecoveryState).toBe('');
  });
});
