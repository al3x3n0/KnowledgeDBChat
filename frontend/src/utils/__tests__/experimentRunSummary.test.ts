import {
  getExperimentRecoveryPriority,
  isExperimentRecoveryOpen,
  summarizeExperimentRecoveryGuidance,
  summarizeExperimentRun,
  summarizeOperatorInterventions,
} from '../experimentRunSummary';

describe('summarizeExperimentRun', () => {
  it('normalizes command lists, phases, and detected stack', () => {
    const summary = summarizeExperimentRun({
      source_id: 'repo-1',
      source_name: 'Knowledge Repo',
      commands: ['npm test'],
      bootstrap_commands: ['npm install'],
      fallback_commands: ['python3 -m pytest -q'],
      phases: ['primary', 'fallback'],
      failed_commands: ['npm test'],
      final_phase: 'fallback',
      inferred_project_profile: {
        detected_stack: ['node', 'python'],
      },
    });

    expect(summary.verificationCommands).toEqual(['npm test']);
    expect(summary.bootstrapCommands).toEqual(['npm install']);
    expect(summary.fallbackCommands).toEqual(['python3 -m pytest -q']);
    expect(summary.phases).toEqual(['primary', 'fallback']);
    expect(summary.failedCommands).toEqual(['npm test']);
    expect(summary.finalPhase).toBe('fallback');
    expect(summary.sourceId).toBe('repo-1');
    expect(summary.sourceName).toBe('Knowledge Repo');
    expect(summary.detectedStack).toEqual(['node', 'python']);
  });

  it('falls back to explicit verification commands when present', () => {
    const summary = summarizeExperimentRun({
      commands: ['npm test'],
      verification_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
    });

    expect(summary.verificationCommands).toEqual([
      'CI=true npm --prefix frontend test -- --watchAll=false',
    ]);
  });

  it('detects unresolved recovery from failed fallback runs', () => {
    const run = {
      failed_commands: ['npm test'],
      fallback_attempted: true,
      fallback_ok: false,
    };
    const summary = summarizeExperimentRun(run);

    expect(isExperimentRecoveryOpen(run, summary)).toBe(true);
    expect(getExperimentRecoveryPriority(run, summary)).toBe(5);
  });

  it('ranks recovered fallback ahead of bootstrap recovery', () => {
    const fallbackRecovered = {
      failed_commands: ['npm test'],
      fallback_attempted: true,
      fallback_ok: true,
    };
    const bootstrapRecovered = {
      failed_commands: ['npm test'],
      bootstrap_attempted: true,
      bootstrap_ok: true,
    };

    expect(getExperimentRecoveryPriority(fallbackRecovered)).toBe(4);
    expect(getExperimentRecoveryPriority(bootstrapRecovered)).toBe(2);
  });

  it('extracts recovery reasons and recommended actions from execution graph data', () => {
    const guidance = summarizeExperimentRecoveryGuidance({
      graph_health: {
        reasons: ['fallback verification still failing', '', null],
      },
      recommended_actions: ['Inspect failing fallback output', 'Retry bootstrap'],
    });

    expect(guidance.reasons).toEqual(['fallback verification still failing']);
    expect(guidance.recommendedActions).toEqual([
      'Inspect failing fallback output',
      'Retry bootstrap',
    ]);
  });

  it('summarizes latest operator intervention and bounded recent timeline', () => {
    const summary = summarizeOperatorInterventions([
      {
        action: 'pause',
        job_status_before: 'running',
        job_status_after: 'paused',
        note: 'Paused for manual inspection',
      },
      {
        action: 'resume',
        job_status_before: 'paused',
        job_status_after: 'running',
        outcome_status: 'superseded',
      },
      {
        action: 'restart',
        job_status_before: 'failed',
        job_status_after: 'pending',
        note: 'Retry after fallback failure',
        outcome_status: 'superseded',
      },
      {
        action: 'relaunch',
        job_status_before: 'failed',
        job_status_after: 'pending',
        note: 'Relaunch clean environment',
        outcome_status: 'resolved',
        outcome_reason: 'Job completed after intervention',
      },
    ]);

    expect(summary.count).toBe(4);
    expect(summary.latestLabel).toBe('relaunch (failed -> pending)');
    expect(summary.latestNote).toBe('Relaunch clean environment');
    expect(summary.latestOutcome).toBe('resolved');
    expect(summary.latestOutcomeReason).toBe('Job completed after intervention');
    expect(summary.recentItems).toEqual([
      'resume (paused -> running) [superseded]',
      'restart (failed -> pending): Retry after fallback failure [superseded]',
      'relaunch (failed -> pending): Relaunch clean environment [resolved]',
    ]);
  });

  it('returns an empty operator intervention summary when no entries exist', () => {
    expect(summarizeOperatorInterventions([])).toEqual({
      count: 0,
      latestLabel: '',
      latestNote: '',
      latestOutcome: '',
      latestOutcomeReason: '',
      recentItems: [],
    });
  });
});
