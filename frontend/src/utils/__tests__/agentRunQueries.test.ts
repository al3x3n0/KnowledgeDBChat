import { readFileSync } from 'fs';
import { join } from 'path';

import {
  AGENT_RUN_QUERY_KEYS,
  invalidateAgentRunQueries,
} from '../agentRunQueries';

const PAGES = join(__dirname, '../../pages');

describe('invalidating every surface that shows a run', () => {
  it('refreshes both the Runs queue and the Control Plane reviews', () => {
    // The bug this exists for: the two are projections of the same backend
    // function, and each invalidated only its own view, so an item actioned in
    // one went on being offered in the other.
    const invalidated: unknown[] = [];
    invalidateAgentRunQueries({
      invalidateQueries: (key: unknown) => invalidated.push(key),
    } as any);

    expect(invalidated).toContainEqual(['agent-checkpoint-queue']);
    expect(invalidated).toContainEqual(['agent-control-reviews']);
    expect(invalidated).toContainEqual(['agent-control-run']);
    expect(invalidated).toContainEqual(['agent-jobs']);
  });

  it('passes through keys only one caller needs', () => {
    const invalidated: unknown[] = [];
    invalidateAgentRunQueries(
      { invalidateQueries: (k: unknown) => invalidated.push(k) } as any,
      ['research-portfolios']
    );

    expect(invalidated).toContainEqual(['research-portfolios']);
    expect(invalidated.length).toBe(AGENT_RUN_QUERY_KEYS.length + 1);
  });

  it('is used rather than bypassed by either page', () => {
    // A helper nothing calls is the same as no helper. This is what makes the
    // fix hold when a twenty-fifth mutation is added.
    for (const page of ['AutonomousAgentsPage.tsx', 'AgentControlPlanePage.tsx']) {
      const src = readFileSync(join(PAGES, page), 'utf8');
      expect(src).toContain('invalidateAgentRunQueries');
      // No page may hand-roll the keys the helper owns.
      expect(src).not.toContain("invalidateQueries(['agent-checkpoint-queue'])");
      expect(src).not.toContain("invalidateQueries(['agent-control-reviews'])");
    }
  });
});
