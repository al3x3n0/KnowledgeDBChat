import type {
  AutonomousRndJobOutcomeResponse,
  AutonomousRndVerificationAuditEnvelope,
  AutonomousRndVerificationTask,
  AutonomousRndVerificationTimelineEvent,
} from '../types';

export interface VerificationAuditReportFilters {
  task_id?: string;
  status?: string;
}

const text = (value: unknown): string | null => {
  const normalized = String(value || '').trim();
  return normalized || null;
};

const safeTask = (task: AutonomousRndVerificationTask) => ({
  task_id: String(task.task_id),
  evidence_id: text(task.evidence_id),
  evidence_status: text(task.evidence_status),
  priority: text(task.priority),
  priority_score: Number.isFinite(Number(task.priority_score))
    ? Number(task.priority_score)
    : null,
  required_checks: Array.isArray(task.required_checks)
    ? task.required_checks.map((check) => String(check))
    : [],
  launch_status: String(task.launch_status || ''),
  job_status: text(task.job_status),
  approval_status: text(task.approval_status),
  reconciliation_status: text(task.reconciliation_status),
  reconciliation_recorded_at: text(task.reconciliation_recorded_at),
  experiment_plan_id: text(task.experiment_plan_id),
  experiment_run_id: text(task.experiment_run_id),
  agent_job_id: text(task.agent_job_id),
  audit_id: text(task.audit_id),
  budget: {
    repeat_count: Number(task.budget?.repeat_count || 0) || null,
    timeout_seconds: Number(task.budget?.timeout_seconds || 0) || null,
    max_runtime_minutes: Number(task.budget?.max_runtime_minutes || 0) || null,
    budget_limit: Number(task.budget?.budget_limit || 0) || null,
  },
});

const safeEvent = (event: AutonomousRndVerificationTimelineEvent) => ({
  event_id: String(event.event_id),
  task_id: String(event.task_id),
  event_type: String(event.event_type),
  at: String(event.at),
  actor: String(event.actor),
  label: String(event.label),
  status: text(event.status),
  entity_type: text(event.entity_type),
  entity_id: text(event.entity_id),
});

export const buildVerificationAuditReport = (
  response: AutonomousRndJobOutcomeResponse,
  filters: VerificationAuditReportFilters = {},
  generatedAt: string = new Date().toISOString()
) => {
  const taskId = text(filters.task_id);
  const status = text(filters.status)?.toLowerCase() || null;
  const lifecycle = response.verification_lifecycle;
  const timeline = (lifecycle.timeline || []).filter((event) => {
    if (taskId && event.task_id !== taskId) return false;
    if (status && String(event.status || '').toLowerCase() !== status) return false;
    return true;
  });
  const timelineTaskIds = new Set(timeline.map((event) => event.task_id));
  const tasks = (lifecycle.tasks || []).filter((task) => {
    if (taskId && task.task_id !== taskId) return false;
    if (
      status
      && ![
        task.evidence_status,
        task.launch_status,
        task.job_status,
        task.approval_status,
        task.reconciliation_status,
      ].some((value) => String(value || '').toLowerCase() === status)
      && !timelineTaskIds.has(task.task_id)
    ) {
      return false;
    }
    return true;
  });

  return {
    schema_version: 1,
    report_type: 'autonomous_rnd_verification_audit',
    generated_at: generatedAt,
    job_id: String(response.job_id),
    job_status: String(response.job_status || ''),
    filters: {
      task_id: taskId,
      status,
    },
    summary: {
      task_count: tasks.length,
      timeline_event_count: timeline.length,
    },
    tasks: tasks.map(safeTask),
    timeline: timeline.map(safeEvent),
  };
};

export const canonicalizeVerificationAudit = (value: unknown): string => {
  const normalize = (item: any): any => {
    if (Array.isArray(item)) return item.map(normalize);
    if (item && typeof item === 'object') {
      return Object.keys(item)
        .sort()
        .reduce<Record<string, any>>((result, key) => {
          result[key] = normalize(item[key]);
          return result;
        }, {});
    }
    return item;
  };
  return JSON.stringify(normalize(value));
};

const sha256Hex = async (value: string): Promise<string> => {
  if (!window.crypto?.subtle) {
    throw new Error('Secure hashing is unavailable in this browser');
  }
  const digest = await window.crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(value)
  );
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
};

export const buildHashedVerificationAuditEnvelope = async (
  report: ReturnType<typeof buildVerificationAuditReport>
): Promise<AutonomousRndVerificationAuditEnvelope> => {
  const canonical = canonicalizeVerificationAudit(report);
  return {
    snapshot: report,
    integrity: {
      canonicalization: 'json-sort-keys-compact-v1',
      sha256: await sha256Hex(canonical),
    },
  };
};

export const downloadVerificationAuditEnvelope = (
  envelope: AutonomousRndVerificationAuditEnvelope,
  jobId: string,
  suffix: string
) => {
  const blob = new Blob([JSON.stringify(envelope, null, 2)], {
    type: 'application/json',
  });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `verification-audit-${jobId}-${suffix}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const downloadVerificationAuditReport = async (
  report: ReturnType<typeof buildVerificationAuditReport>
) => {
  const envelope = await buildHashedVerificationAuditEnvelope(report);
  downloadVerificationAuditEnvelope(envelope, report.job_id, 'sha256');
  return envelope;
};
