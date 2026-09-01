/**
 * The execution-graph derivations, tested without rendering anything.
 *
 * These fourteen values used to be computed inline in a 4,000-line component,
 * which meant the only way to check them was to render the whole panel. As a
 * pure function of a job they can be asked directly — including the cases a
 * rendered test would never set up, like a job whose results are absent.
 */

import { executionGraphView } from '../agentJobDetail';
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
