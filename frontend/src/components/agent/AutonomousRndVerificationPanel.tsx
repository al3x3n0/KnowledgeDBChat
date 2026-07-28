import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, Clock3, Download, ExternalLink, FlaskConical, Loader2, Play, RefreshCw, ShieldCheck } from 'lucide-react';
import toast from 'react-hot-toast';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { apiClient } from '../../services/api';
import type {
  AutonomousRndVerificationLaunchRequest,
  AutonomousRndVerificationTask,
} from '../../types';
import {
  buildVerificationAuditReport,
  downloadVerificationAuditEnvelope,
  downloadVerificationAuditReport,
} from '../../utils/verificationAuditReport';
import Button from '../common/Button';
import CompOpsEvidenceImportPanel from './CompOpsEvidenceImportPanel';

interface Props {
  jobId: string;
  defaultResearchNoteId?: string;
  defaultSourceId?: string;
  focusTaskId?: string;
  onOpenAgentJob?: (jobId: string) => void;
}

interface LaunchDraft {
  approvalNote: string;
  researchNoteId: string;
  sourceId: string;
  sandboxProfileId: string;
  commands: string;
  repeatCount: string;
  timeoutSeconds: string;
  maxRuntimeMinutes: string;
  budgetLimit: string;
  startImmediately: boolean;
  approved: boolean;
}

const makeDraft = (defaultResearchNoteId = '', defaultSourceId = ''): LaunchDraft => ({
  approvalNote: '',
  researchNoteId: defaultResearchNoteId,
  sourceId: defaultSourceId,
  sandboxProfileId: 'scientific-generic-sandbox',
  commands: '',
  repeatCount: '2',
  timeoutSeconds: '30',
  maxRuntimeMinutes: '2',
  budgetLimit: '1',
  startImmediately: false,
  approved: false,
});

const formatLabel = (value?: string | null) =>
  String(value || 'unknown').replace(/_/g, ' ');

const badgeClass = (value?: string | null) => {
  const status = String(value || '').toLowerCase();
  if (['verified', 'succeeded', 'approved', 'support_recorded'].includes(status)) {
    return 'border-emerald-200 bg-emerald-50 text-emerald-700';
  }
  if (['rejected', 'failed', 'blocked'].includes(status)) {
    return 'border-rose-200 bg-rose-50 text-rose-700';
  }
  if (['running', 'queued', 'pending', 'planned', 'corroborated'].includes(status)) {
    return 'border-blue-200 bg-blue-50 text-blue-700';
  }
  return 'border-gray-200 bg-gray-50 text-gray-700';
};

const finiteNumber = (value: string, fallback: number) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const formatTimestamp = (value: string) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

export const AutonomousRndVerificationPanel: React.FC<Props> = ({
  jobId,
  defaultResearchNoteId = '',
  defaultSourceId = '',
  focusTaskId = '',
  onOpenAgentJob,
}) => {
  const queryClient = useQueryClient();
  const taskRefs = useRef<Record<string, HTMLElement | null>>({});
  const [expandedTaskId, setExpandedTaskId] = useState('');
  const [draft, setDraft] = useState<LaunchDraft>(() => makeDraft(defaultResearchNoteId, defaultSourceId));
  const [noteSearch, setNoteSearch] = useState('');
  const [sourceSearch, setSourceSearch] = useState('');
  const [timelineTaskFilter, setTimelineTaskFilter] = useState('');
  const [timelineStatusFilter, setTimelineStatusFilter] = useState('');

  const outcomeQuery = useQuery(
    ['autonomous-rnd-outcome', jobId],
    () => apiClient.getAutonomousRndJobOutcome(jobId),
    {
      enabled: Boolean(jobId),
      refetchInterval: 60000,
      refetchOnWindowFocus: false,
      retry: false,
    }
  );

  const tasks = useMemo(
    () => outcomeQuery.data?.verification_lifecycle?.tasks || [],
    [outcomeQuery.data]
  );
  const activeVerifierJobIds = useMemo(
    () =>
      Array.from(
        new Set(
          tasks
            .filter((task) => {
              const status = String(task.job_status || '').toLowerCase();
              return task.agent_job_id && !['completed', 'failed', 'cancelled'].includes(status);
            })
            .map((task) => String(task.agent_job_id))
        )
      ),
    [tasks]
  );
  const notesQuery = useQuery(
    ['research-notes', 'verification-picker', noteSearch],
    () => apiClient.listResearchNotes({ q: noteSearch.trim() || undefined, limit: 50, offset: 0 }),
    {
      enabled: Boolean(expandedTaskId),
      staleTime: 15000,
      refetchOnWindowFocus: false,
    }
  );
  const sourcesQuery = useQuery(
    ['document-sources', 'verification-picker'],
    () => apiClient.getDocumentSources(),
    {
      enabled: Boolean(expandedTaskId),
      staleTime: 30000,
      refetchOnWindowFocus: false,
    }
  );
  const filteredSources = useMemo(() => {
    const query = sourceSearch.trim().toLowerCase();
    const rows = Array.isArray(sourcesQuery.data) ? sourcesQuery.data : [];
    if (!query) return rows;
    return rows.filter((source) =>
      `${source.name} ${source.source_type} ${source.id}`.toLowerCase().includes(query)
    );
  }, [sourceSearch, sourcesQuery.data]);
  const plannedTasks = useMemo(() => {
    const rows = outcomeQuery.data?.outcome?.verification_plan?.tasks;
    return new Map(
      (Array.isArray(rows) ? rows : []).map((row: Record<string, any>) => [String(row?.id || ''), row])
    );
  }, [outcomeQuery.data]);

  const launchMutation = useMutation(
    ({ taskId, payload }: { taskId: string; payload: AutonomousRndVerificationLaunchRequest }) =>
      apiClient.launchAutonomousRndVerificationTask(jobId, taskId, payload),
    {
      onSuccess: (response) => {
        toast.success(response.created ? 'Verification plan created' : 'Verification plan already exists');
        setExpandedTaskId('');
        setDraft(makeDraft(defaultResearchNoteId, defaultSourceId));
        queryClient.invalidateQueries(['autonomous-rnd-outcome', jobId]);
        queryClient.invalidateQueries(['agent-jobs']);
      },
      onError: (error: any) => {
        toast.error(error?.message || 'Failed to launch verification');
      },
    }
  );

  useEffect(() => {
    if (activeVerifierJobIds.length === 0) return;
    const sockets: WebSocket[] = [];
    for (const verifierJobId of activeVerifierJobIds) {
      try {
        const socket = apiClient.createAgentJobProgressWebSocket(verifierJobId);
        socket.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            if (!payload || typeof payload !== 'object' || String(payload.type || '') !== 'progress') return;
            queryClient.invalidateQueries(['autonomous-rnd-outcome', jobId]);
            const status = String(payload.status || '').toLowerCase();
            if (['completed', 'failed', 'cancelled'].includes(status)) socket.close();
          } catch {
            // Ignore malformed progress events; the fallback refresh remains active.
          }
        };
        sockets.push(socket);
      } catch {
        // A missing/expired socket credential falls back to the periodic refresh.
      }
    }
    return () => sockets.forEach((socket) => socket.close());
  }, [activeVerifierJobIds, jobId, queryClient]);

  useEffect(() => {
    if (!focusTaskId || tasks.length === 0) return;
    taskRefs.current[focusTaskId]?.scrollIntoView?.({
      block: 'nearest',
      behavior: 'smooth',
    });
  }, [focusTaskId, tasks]);

  const lifecycle = outcomeQuery.data?.verification_lifecycle;
  const timeline = useMemo(() => lifecycle?.timeline || [], [lifecycle]);
  const timelineStatuses = useMemo(
    () =>
      Array.from(
        new Set(
          timeline
            .map((event) => String(event.status || '').trim())
            .filter(Boolean)
        )
      ).sort(),
    [timeline]
  );
  const visibleTimeline = useMemo(
    () =>
      timeline.filter((event) => {
        if (timelineTaskFilter && event.task_id !== timelineTaskFilter) return false;
        if (
          timelineStatusFilter
          && String(event.status || '') !== timelineStatusFilter
        ) {
          return false;
        }
        return true;
      }),
    [timeline, timelineStatusFilter, timelineTaskFilter]
  );
  const exportAudit = async () => {
    if (!outcomeQuery.data) return;
    try {
      const report = buildVerificationAuditReport(outcomeQuery.data, {
        task_id: timelineTaskFilter || undefined,
        status: timelineStatusFilter || undefined,
      });
      await downloadVerificationAuditReport(report);
      toast.success('Hashed verification audit exported');
    } catch (error: any) {
      toast.error(error?.message || 'Failed to hash verification audit');
    }
  };
  const signedAuditMutation = useMutation(
    () =>
      apiClient.createAutonomousRndVerificationAuditSnapshot(jobId, {
        task_id: timelineTaskFilter || undefined,
        status: timelineStatusFilter || undefined,
      }),
    {
      onSuccess: (envelope) => {
        downloadVerificationAuditEnvelope(envelope, jobId, 'signed');
        toast.success('Immutable signed audit snapshot exported');
      },
      onError: (error: any) => {
        toast.error(error?.message || 'Failed to create signed verification audit');
      },
    }
  );

  if (outcomeQuery.isLoading) {
    return (
      <div className="mb-4 rounded-lg border border-violet-100 bg-violet-50 p-3 text-xs text-violet-700">
        <Loader2 className="mr-2 inline h-3.5 w-3.5 animate-spin" />
        Loading evidence verification…
      </div>
    );
  }
  if (outcomeQuery.isError) return null;

  const compOpsImport = (
    <CompOpsEvidenceImportPanel
      jobId={jobId}
      onImported={() => outcomeQuery.refetch()}
    />
  );
  if (tasks.length === 0) return compOpsImport;

  const lifecycleView = outcomeQuery.data!.verification_lifecycle;
  const submitLaunch = (task: AutonomousRndVerificationTask) => {
    const commands = draft.commands
      .split('\n')
      .map((command) => command.trim())
      .filter(Boolean);
    if (!draft.approved) {
      toast.error('Explicit approval is required');
      return;
    }
    if (!draft.approvalNote.trim() || !draft.researchNoteId.trim() || !draft.sourceId.trim()) {
      toast.error('Approval note, research note, and source are required');
      return;
    }
    if (commands.length < 1 || commands.length > 4) {
      toast.error('Enter between one and four verification commands');
      return;
    }
    launchMutation.mutate({
      taskId: task.task_id,
      payload: {
        approval_confirmed: true,
        approval_note: draft.approvalNote.trim(),
        research_note_id: draft.researchNoteId.trim(),
        source_id: draft.sourceId.trim(),
        sandbox_profile_id: draft.sandboxProfileId.trim(),
        commands,
        repeat_count: finiteNumber(draft.repeatCount, 2),
        timeout_seconds: finiteNumber(draft.timeoutSeconds, 30),
        max_runtime_minutes: finiteNumber(draft.maxRuntimeMinutes, 2),
        budget_limit: finiteNumber(draft.budgetLimit, 1),
        start_immediately: draft.startImmediately,
      },
    });
  };

  return (
    <>
      {compOpsImport}
      <section className="mb-4 rounded-lg border border-violet-200 bg-white p-3" aria-label="Evidence verification lifecycle">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <h3 className="flex items-center gap-1.5 text-sm font-semibold text-gray-900">
            <ShieldCheck className="h-4 w-4 text-violet-600" />
            Evidence verification
          </h3>
          <p className="mt-0.5 text-xs text-gray-500">
            {lifecycleView.task_count} task{lifecycleView.task_count === 1 ? '' : 's'} · approval-gated local experiments
          </p>
        </div>
        <button
          type="button"
          className="rounded p-1 text-gray-500 hover:bg-gray-100"
          onClick={() => outcomeQuery.refetch()}
          aria-label="Refresh verification lifecycle"
        >
          <RefreshCw className={`h-4 w-4 ${outcomeQuery.isFetching ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {timeline.length > 0 && (
        <details className="mb-3 rounded border border-gray-200 bg-gray-50 p-2" open={Boolean(focusTaskId)}>
          <summary className="cursor-pointer text-xs font-medium text-gray-700">Audit timeline ({timeline.length})</summary>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <label className="text-xs text-gray-600">
              Timeline task
              <select
                className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
                value={timelineTaskFilter}
                onChange={(event) => setTimelineTaskFilter(event.target.value)}
              >
                <option value="">All tasks</option>
                {tasks.map((task) => (
                  <option key={task.task_id} value={task.task_id}>{task.task_id}</option>
                ))}
              </select>
            </label>
            <label className="text-xs text-gray-600">
              Timeline status
              <select
                className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
                value={timelineStatusFilter}
                onChange={(event) => setTimelineStatusFilter(event.target.value)}
              >
                <option value="">All statuses</option>
                {timelineStatuses.map((status) => (
                  <option key={status} value={status}>{formatLabel(status)}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="mt-2 flex items-center justify-between gap-2">
            <span className="text-[11px] text-gray-500">
              Showing {visibleTimeline.length} of {timeline.length} events
            </span>
            <div className="flex gap-1">
              <Button size="sm" variant="ghost" onClick={exportAudit}>
                <Download className="mr-1 h-3.5 w-3.5" />
                Export hashed JSON
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => signedAuditMutation.mutate()}
                disabled={signedAuditMutation.isLoading}
              >
                {signedAuditMutation.isLoading
                  ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
                  : <ShieldCheck className="mr-1 h-3.5 w-3.5" />}
                Export signed JSON
              </Button>
            </div>
          </div>
          <ol className="mt-3 space-y-3 border-l border-violet-200 pl-3">
            {visibleTimeline.map((event) => (
              <li key={event.event_id} className="relative text-xs">
                <Clock3 className="absolute -left-[19px] top-0.5 h-3 w-3 rounded-full bg-gray-50 text-violet-600" />
                <div className="font-medium text-gray-800">{event.label}</div>
                <div className="mt-0.5 text-gray-500">
                  {formatTimestamp(event.at)} · {formatLabel(event.actor)}
                  {event.status ? ` · ${formatLabel(event.status)}` : ''}
                </div>
                <div className="mt-0.5 truncate font-mono text-[10px] text-gray-400">{event.task_id}</div>
              </li>
            ))}
          </ol>
          {visibleTimeline.length === 0 && (
            <div className="mt-3 text-xs text-gray-500">No audit events match these filters.</div>
          )}
        </details>
      )}

      <div className="space-y-3">
        {tasks.map((task) => {
          const proposed = plannedTasks.get(task.task_id) || {};
          const canConfigure = task.launch_status === 'not_launched';
          const expanded = expandedTaskId === task.task_id;
          return (
            <article
              key={task.task_id}
              ref={(element) => {
                taskRefs.current[task.task_id] = element;
              }}
              className={`rounded border p-2.5 ${
                focusTaskId === task.task_id
                  ? 'border-violet-400 ring-2 ring-violet-200'
                  : 'border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate font-mono text-[11px] text-gray-500">{task.task_id}</div>
                  <div className="mt-1 flex flex-wrap gap-1">
                    <span className={`rounded-full border px-2 py-0.5 text-[11px] ${badgeClass(task.evidence_status)}`}>
                      Evidence: {formatLabel(task.evidence_status)}
                    </span>
                    <span className={`rounded-full border px-2 py-0.5 text-[11px] ${badgeClass(task.launch_status)}`}>
                      Run: {formatLabel(task.launch_status)}
                    </span>
                    {task.approval_status && (
                      <span className={`rounded-full border px-2 py-0.5 text-[11px] ${badgeClass(task.approval_status)}`}>
                        Approval: {formatLabel(task.approval_status)}
                      </span>
                    )}
                  </div>
                </div>
                {canConfigure ? (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      setExpandedTaskId(expanded ? '' : task.task_id);
                      setDraft(makeDraft(defaultResearchNoteId, defaultSourceId));
                      setNoteSearch('');
                      setSourceSearch('');
                    }}
                  >
                    <FlaskConical className="mr-1 h-3.5 w-3.5" />
                    {expanded ? 'Close' : 'Configure'}
                  </Button>
                ) : task.agent_job_id && onOpenAgentJob ? (
                  <Button size="sm" variant="ghost" onClick={() => onOpenAgentJob(task.agent_job_id!)}>
                    <ExternalLink className="mr-1 h-3.5 w-3.5" />
                    Verifier
                  </Button>
                ) : null}
              </div>

              {proposed?.objective && <p className="mt-2 text-xs text-gray-600">{String(proposed.objective)}</p>}
              {task.required_checks.length > 0 && (
                <ul className="mt-2 space-y-1 text-xs text-gray-600">
                  {task.required_checks.map((check) => (
                    <li key={check} className="flex gap-1.5">
                      <AlertCircle className="mt-0.5 h-3 w-3 shrink-0 text-amber-500" />
                      {formatLabel(check)}
                    </li>
                  ))}
                </ul>
              )}
              {task.reconciliation_status && (
                <div className="mt-2 flex items-center gap-1 text-xs text-emerald-700">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Reconciliation: {formatLabel(task.reconciliation_status)}
                </div>
              )}

              {expanded && (
                <div className="mt-3 space-y-2 border-t border-gray-100 pt-3">
                  <label className="block text-xs text-gray-700">
                    Approval note
                    <textarea
                      className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5 text-xs"
                      rows={2}
                      value={draft.approvalNote}
                      onChange={(event) => setDraft({ ...draft, approvalNote: event.target.value })}
                    />
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-xs text-gray-700">
                        Find research note
                        <input
                          className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
                          placeholder="Search note titles"
                          value={noteSearch}
                          onChange={(event) => setNoteSearch(event.target.value)}
                        />
                      </label>
                      <label className="mt-1 block text-xs text-gray-700">
                        Research note
                        <select
                          className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
                          value={draft.researchNoteId}
                          onChange={(event) => setDraft({ ...draft, researchNoteId: event.target.value })}
                        >
                          <option value="">Select a research note</option>
                          {draft.researchNoteId
                            && !(notesQuery.data?.items || []).some((note) => note.id === draft.researchNoteId)
                            && <option value={draft.researchNoteId}>Current · {draft.researchNoteId}</option>}
                          {(notesQuery.data?.items || []).map((note) => (
                            <option key={note.id} value={note.id}>{note.title}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div>
                      <label className="text-xs text-gray-700">
                        Find source
                        <input
                          className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
                          placeholder="Search source names"
                          value={sourceSearch}
                          onChange={(event) => setSourceSearch(event.target.value)}
                        />
                      </label>
                      <label className="mt-1 block text-xs text-gray-700">
                        Source
                        <select
                          className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5"
                          value={draft.sourceId}
                          onChange={(event) => setDraft({ ...draft, sourceId: event.target.value })}
                        >
                          <option value="">Select a source</option>
                          {draft.sourceId
                            && !(sourcesQuery.data || []).some((source) => source.id === draft.sourceId)
                            && <option value={draft.sourceId}>Current · {draft.sourceId}</option>}
                          {filteredSources.map((source) => (
                            <option key={source.id} value={source.id}>
                              {source.name} · {formatLabel(source.source_type)}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </div>
                  <label className="block text-xs text-gray-700">
                    Commands (one per line, maximum four)
                    <textarea
                      className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5 font-mono text-xs"
                      rows={3}
                      value={draft.commands}
                      onChange={(event) => setDraft({ ...draft, commands: event.target.value })}
                    />
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="text-xs text-gray-700">
                      Sandbox profile
                      <input className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5" value={draft.sandboxProfileId} onChange={(event) => setDraft({ ...draft, sandboxProfileId: event.target.value })} />
                    </label>
                    <label className="text-xs text-gray-700">
                      Repeat count
                      <input type="number" min={2} max={10} className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5" value={draft.repeatCount} onChange={(event) => setDraft({ ...draft, repeatCount: event.target.value })} />
                    </label>
                    <label className="text-xs text-gray-700">
                      Timeout, seconds
                      <input type="number" min={5} max={600} className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5" value={draft.timeoutSeconds} onChange={(event) => setDraft({ ...draft, timeoutSeconds: event.target.value })} />
                    </label>
                    <label className="text-xs text-gray-700">
                      Max runtime, minutes
                      <input type="number" min={1} max={60} className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5" value={draft.maxRuntimeMinutes} onChange={(event) => setDraft({ ...draft, maxRuntimeMinutes: event.target.value })} />
                    </label>
                    <label className="text-xs text-gray-700">
                      Budget limit
                      <input type="number" min="0.01" max={1000} step="0.01" className="mt-1 w-full rounded border border-gray-300 px-2 py-1.5" value={draft.budgetLimit} onChange={(event) => setDraft({ ...draft, budgetLimit: event.target.value })} />
                    </label>
                  </div>
                  <label className="flex items-start gap-2 text-xs text-gray-700">
                    <input type="checkbox" className="mt-0.5" checked={draft.startImmediately} onChange={(event) => setDraft({ ...draft, startImmediately: event.target.checked })} />
                    Queue the verifier immediately after creating the bounded experiment.
                  </label>
                  <label className="flex items-start gap-2 rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
                    <input type="checkbox" className="mt-0.5" checked={draft.approved} onChange={(event) => setDraft({ ...draft, approved: event.target.checked })} />
                    I approve this exact local verification recipe and its resource limits.
                  </label>
                  <Button size="sm" variant="primary" onClick={() => submitLaunch(task)} disabled={launchMutation.isLoading}>
                    {launchMutation.isLoading ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : <Play className="mr-1 h-3.5 w-3.5" />}
                    Approve and launch
                  </Button>
                </div>
              )}
            </article>
          );
        })}
      </div>
      </section>
    </>
  );
};

export default AutonomousRndVerificationPanel;
