import { describe, expect, test } from '@jest/globals';
import { mergeProgressUpdateIntoJob, TERMINAL_JOB_STATUSES } from '../agentJobProgress';
import type { AgentJob, AgentJobProgressUpdate } from '../../types';

const makeJob = (): AgentJob => ({
  id: 'job-1',
  name: 'Job',
  goal: 'Goal',
  job_type: 'research',
  user_id: 'user-1',
  status: 'running',
  progress: 10,
  iteration: 1,
  max_iterations: 10,
  max_tool_calls: 10,
  max_llm_calls: 10,
  max_runtime_minutes: 10,
  tool_calls_used: 1,
  llm_calls_used: 1,
  tokens_used: 0,
  error_count: 0,
  chain_depth: 0,
  chain_triggered: false,
  results: {
    execution_strategy: {
      execution_graph: {
        graph_health: { status: 'ok', severity_score: 0 },
      },
      scope_observability: {
        resolved_scope_id: 'old-scope',
      },
    },
  },
  created_at: '2026-03-10T00:00:00Z',
});

describe('agentJobProgress', () => {
  test('mergeProgressUpdateIntoJob overlays runtime execution graph and scope data', () => {
    const job = makeJob();
    const update: AgentJobProgressUpdate = {
      type: 'progress',
      job_id: 'job-1',
      progress: 55,
      phase: 'acting',
      status: 'running',
      iteration: 4,
      phase_details: 'Executed tool',
      execution_graph_runtime: {
        verification_attempts: 2,
        graph_health: { status: 'warning', severity_score: 20 },
      },
      scope_observability_runtime: {
        resolved_scope_id: 'scope-1',
        scope_source: 'config.source_id',
      },
      timestamp: '2026-03-10T00:00:01Z',
    };

    const merged = mergeProgressUpdateIntoJob(job, update);

    expect(merged.progress).toBe(55);
    expect(merged.iteration).toBe(4);
    expect(merged.current_phase).toBe('acting');
    expect(merged.phase_details).toBe('Executed tool');
    expect((merged.results as any)?.execution_strategy?.execution_graph?.verification_attempts).toBe(2);
    expect((merged.results as any)?.execution_strategy?.execution_graph?.graph_health?.status).toBe('warning');
    expect((merged.results as any)?.execution_strategy?.scope_observability?.resolved_scope_id).toBe('scope-1');
    expect((merged.results as any)?.execution_strategy?.scope_observability?.scope_source).toBe('config.source_id');
  });

  test('TERMINAL_JOB_STATUSES contains terminal states used by live updates', () => {
    expect(TERMINAL_JOB_STATUSES.has('completed')).toBe(true);
    expect(TERMINAL_JOB_STATUSES.has('failed')).toBe(true);
    expect(TERMINAL_JOB_STATUSES.has('cancelled')).toBe(true);
    expect(TERMINAL_JOB_STATUSES.has('running')).toBe(false);
  });
});
