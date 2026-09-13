import type { AutonomousRndJobOutcomeResponse } from '../../types';
import {
  buildVerificationAuditReport,
  canonicalizeVerificationAudit,
} from '../verificationAuditReport';

describe('buildVerificationAuditReport', () => {
  it('exports only allowlisted lifecycle fields and honors filters', () => {
    const response = {
      job_id: 'parent-job',
      job_status: 'completed',
      outcome: {
        commands: ['do-not-export'],
        raw_output: 'secret-output',
      },
      verification_lifecycle: {
        task_count: 1,
        launch_status_counts: { succeeded: 1 },
        evidence_status_counts: { verified: 1 },
        tasks: [
          {
            task_id: 'verify-1',
            evidence_id: 'evidence-1',
            evidence_status: 'verified',
            priority: 'critical',
            priority_score: 90,
            required_checks: [],
            launch_status: 'succeeded',
            job_status: 'completed',
            approval_status: 'approved',
            reconciliation_status: 'support_recorded',
            experiment_plan_id: 'plan-1',
            experiment_run_id: 'run-1',
            agent_job_id: 'verifier-1',
            audit_id: 'audit-1',
            budget: { repeat_count: 2 },
            commands: ['also-secret'],
            stdout: 'raw-command-output',
          },
        ],
        timeline: [
          {
            event_id: 'verify-1:approval_recorded',
            task_id: 'verify-1',
            event_type: 'approval_recorded',
            at: '2026-07-28T10:00:00Z',
            actor: 'operator',
            label: 'Verification approved',
            status: 'approved',
            entity_type: 'tool_audit',
            entity_id: 'audit-1',
            approval_note: 'private operator note',
          },
        ],
      },
    } as unknown as AutonomousRndJobOutcomeResponse;

    const report = buildVerificationAuditReport(
      response,
      { task_id: 'verify-1', status: 'approved' },
      '2026-07-28T12:00:00Z'
    );
    const serialized = JSON.stringify(report);

    expect(report.summary).toEqual({ task_count: 1, timeline_event_count: 1 });
    expect(report.filters).toEqual({ task_id: 'verify-1', status: 'approved' });
    expect(report.generated_at).toBe('2026-07-28T12:00:00Z');
    expect(serialized).not.toContain('do-not-export');
    expect(serialized).not.toContain('secret-output');
    expect(serialized).not.toContain('also-secret');
    expect(serialized).not.toContain('raw-command-output');
    expect(serialized).not.toContain('private operator note');
  });

  it('canonicalizes equivalent objects identically regardless of key order', () => {
    expect(canonicalizeVerificationAudit({ z: 1, a: { d: 2, b: 3 } })).toBe(
      canonicalizeVerificationAudit({ a: { b: 3, d: 2 }, z: 1 })
    );
  });
});
