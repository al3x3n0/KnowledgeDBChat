import { Notification } from '../../types';
import { buildNotificationToastSummary, summarizeExperimentNotification, summarizeFollowUpOutcomeNotification, summarizeHypothesisReevaluationNotification, summarizePolicyGuardrailNotification, summarizeQueueUrgencyNotification } from '../notificationSummary';

function makeNotification(overrides: Partial<Notification> = {}): Notification {
  return {
    id: 'notif-1',
    notification_type: 'experiment_run_update',
    title: 'Experiment run failed',
    message: 'Recovery remains open.',
    priority: 'high',
    is_read: false,
    created_at: '2026-03-11T12:00:00Z',
    ...overrides,
  };
}

describe('summarizeExperimentNotification', () => {
  it('returns badges and guidance for unresolved recovery notifications', () => {
    const summary = summarizeExperimentNotification(
      makeNotification({
        data: {
          final_phase: 'fallback',
          source_name: 'Knowledge Repo',
          fallback_attempted: true,
          fallback_ok: false,
          failed_command_count: 2,
          recovery_open: true,
          recovery_reason: 'fallback verification still failing',
          recommended_action: 'Inspect failing fallback output',
          latest_operator_action: 'restart',
          latest_operator_status_before: 'failed',
          latest_operator_status_after: 'pending',
          latest_operator_note: 'Retry after fallback failure',
          latest_operator_outcome: 'unresolved',
          latest_operator_outcome_reason: 'Job failed after intervention',
        },
      }),
    );

    expect(summary).not.toBeNull();
    expect(summary?.badges).toEqual([
      'Phase fallback',
      'Repo Knowledge Repo',
      'Fallback attempted',
      'Recovery open',
      'Failed cmds 2',
    ]);
    expect(summary?.reason).toBe('fallback verification still failing');
    expect(summary?.nextAction).toBe('Inspect failing fallback output');
    expect(summary?.latestOperator).toBe('restart (failed -> pending)');
    expect(summary?.latestOperatorNote).toBe('Retry after fallback failure');
    expect(summary?.latestOperatorOutcome).toBe('unresolved');
    expect(summary?.latestOperatorOutcomeReason).toBe('Job failed after intervention');
  });

  it('returns null for non-experiment notifications', () => {
    const summary = summarizeExperimentNotification(
      makeNotification({
        notification_type: 'system_maintenance',
      }),
    );

    expect(summary).toBeNull();
  });

  it('builds a toast summary with recovery guidance for experiment notifications', () => {
    const summary = buildNotificationToastSummary(
      makeNotification({
        data: {
          final_phase: 'fallback',
          source_name: 'Knowledge Repo',
          fallback_attempted: true,
          fallback_ok: false,
          failed_command_count: 2,
          recovery_open: true,
          recovery_reason: 'fallback verification still failing',
          recommended_action: 'Inspect failing fallback output',
          latest_operator_action: 'restart',
          latest_operator_status_before: 'failed',
          latest_operator_status_after: 'pending',
          latest_operator_outcome: 'unresolved',
          latest_operator_outcome_reason: 'Job failed after intervention',
        },
      }),
    );

    expect(summary.title).toBe('Experiment run failed');
    expect(summary.description).toContain('Phase fallback');
    expect(summary.description).toContain('Repo Knowledge Repo');
    expect(summary.description).toContain('Fallback attempted');
    expect(summary.description).toContain('Reason: fallback verification still failing');
    expect(summary.description).toContain('Next: Inspect failing fallback output');
    expect(summary.description).toContain('Last operator: restart (failed -> pending)');
    expect(summary.description).toContain('Operator outcome: unresolved');
    expect(summary.description).toContain('Outcome reason: Job failed after intervention');
  });

  it('returns badges and guidance for queue urgency notifications', () => {
    const summary = summarizeQueueUrgencyNotification(
      makeNotification({
        notification_type: 'queue_urgency_alert',
        title: 'Queue alert: Approval Required Job',
        message: 'approval checkpoint · overdue · escalation high',
        data: {
          queue_item_type: 'approval_checkpoint',
          sla_bucket: 'overdue',
          escalation_level: 'high',
          customer: 'Acme',
          age_minutes: 300,
          priority_score: 142,
          recommended_action: 'approve',
          reason_label: 'Approval required',
          evidence_summary: 'Human approval required before next action.',
          scheduler_state: {
            queue_reason: 'execution_failure',
            last_run_status: 'failed',
            failure_streak: 3,
            last_scheduled_at: '2026-03-16T09:00:00Z',
            last_dispatched_at: '2026-03-16T09:05:00Z',
          },
        },
      }),
    );

    expect(summary).not.toBeNull();
    expect(summary?.badges).toEqual([
      'approval checkpoint',
      'overdue',
      'Esc high',
      'Customer Acme',
      'Age 300m',
      'Urgency 142',
    ]);
    expect(summary?.reason).toBe('Approval required');
    expect(summary?.nextAction).toBe('approve');
    expect(summary?.evidenceSummary).toBe('Human approval required before next action.');
    expect(summary?.schedulerState).toEqual({
      queue_reason: 'execution_failure',
      last_run_status: 'failed',
      failure_streak: 3,
      last_scheduled_at: '2026-03-16T09:00:00Z',
      last_dispatched_at: '2026-03-16T09:05:00Z',
    });
  });

  it('omits malformed scheduler state from queue urgency summaries', () => {
    const summary = summarizeQueueUrgencyNotification(
      makeNotification({
        notification_type: 'queue_urgency_alert',
        data: {
          reason_label: 'Approval required',
          scheduler_state: 'bad-payload' as any,
        },
      }),
    );

    expect(summary).not.toBeNull();
    expect(summary?.schedulerState).toBeNull();
  });

  it('returns badges and summary for follow-up outcome notifications', () => {
    const summary = summarizeFollowUpOutcomeNotification(
      makeNotification({
        notification_type: 'follow_up_outcome_alert',
        title: 'Follow-up failed: Accepted note',
        message: 'Compilation failed in verification step.',
        data: {
          follow_up_outcome_status: 'failed',
          follow_up_recommendation_key: 'single_research_job',
          follow_up_outcome_summary: 'Compilation failed in verification step.',
          origin_source_kind: 'profile',
          customer: 'Acme',
          follow_up_policy_mode: 'auto_launch_safe',
        },
      }),
    );

    expect(summary).not.toBeNull();
    expect(summary?.badges).toEqual([
      'failed',
      'single research job',
      'Profile follow-up',
      'Customer Acme',
      'Policy auto launch safe',
    ]);
    expect(summary?.summary).toBe('Compilation failed in verification step.');
  });

  it('returns badges and reasons for policy guardrail notifications', () => {
    const summary = summarizePolicyGuardrailNotification(
      makeNotification({
        notification_type: 'policy_guardrail_alert',
        title: 'Policy safeguard: Beta Watch',
        message: 'degrading policy evaluation · suggested rollback',
        data: {
          policy_guardrail_action: 'rollback',
          policy_guardrail_reasons: ['Completion rate fell from 75.0% to 25.0%'],
          customer: 'Beta',
        },
      }),
    );

    expect(summary).not.toBeNull();
    expect(summary?.badges).toEqual(['policy safeguard', 'rollback', 'Customer Beta']);
    expect(summary?.reasons).toEqual(['Completion rate fell from 75.0% to 25.0%']);
  });

  it('returns badges and summary for hypothesis reevaluation notifications', () => {
    const summary = summarizeHypothesisReevaluationNotification(
      makeNotification({
        notification_type: 'hypothesis_reevaluation_update',
        title: 'Reevaluation ready: Compiler note',
        message: 'Hypotheses were re-scored using the latest experiment evidence.',
        data: {
          reevaluation_status: 'completed',
          source_run_ids: ['run-1', 'run-2'],
          reprioritization_summary: 'Hypothesis A moved ahead after the benchmark result.',
          origin_source_kind: 'profile',
        },
      }),
    );

    expect(summary).not.toBeNull();
    expect(summary?.badges).toEqual([
      'completed',
      'Runs run-1, run-2',
      'Domain opportunity',
    ]);
    expect(summary?.summary).toBe('Hypothesis A moved ahead after the benchmark result.');
  });
});
