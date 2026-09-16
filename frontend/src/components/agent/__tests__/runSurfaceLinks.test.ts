import { readFileSync } from 'fs';
import { join } from 'path';

/**
 * The two surfaces that show a run must each be able to reach the other.
 *
 * The Control Plane has offered "Open in Autonomous Agents" on an agent-job
 * node for some time. Nothing went the other way, so the deep inspection of a
 * run -- decision trace, policy snapshot, escalations, memory graph -- was
 * reachable only by finding the run again in a different page's list.
 *
 * Asserted on the source because the link is a *route*, and what breaks is a
 * path string: neither renders the other, so no integration test covers the
 * pair.
 */
const SRC = join(__dirname, '../../..');

describe('the seam between Runs and the Control Plane', () => {
  it('lets you open a job from Runs in the Control Plane', () => {
    const panel = readFileSync(
      join(SRC, 'components/agent/JobDetailPanel.tsx'),
      'utf8'
    );
    expect(panel).toContain('/agent-control-plane?run=job:');
  });

  it('lets you open a Control Plane run back in Runs', () => {
    const controlPlane = readFileSync(
      join(SRC, 'pages/AgentControlPlanePage.tsx'),
      'utf8'
    );
    expect(controlPlane).toContain('/autonomous-agents?job=');
  });

  it('addresses an agent job by the prefix the Control Plane parses', () => {
    // It namespaces run ids -- `job:<id>` and `workflow:<id>` -- which is also
    // why it stays its own destination rather than becoming the Runs page's
    // detail view: a workflow execution is not an agent job.
    const controlPlane = readFileSync(
      join(SRC, 'pages/AgentControlPlanePage.tsx'),
      'utf8'
    );
    expect(controlPlane).toContain("startsWith('job:')");
  });
});
