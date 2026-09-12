/**
 * A pipeline run while it is still happening.
 *
 * Everything a pipeline knows about its own progress lived only in the API
 * until now: a run was launched from the studio, the browser was sent to the
 * job that happened to be its head, and the five stages behind it were
 * invisible. So the questions this answers are the ordinary ones nobody could
 * ask — how far through is it, which stage is it on, did the stage that
 * finished actually produce anything — plus the one the endpoint was built
 * for: where do I restart.
 *
 * Three distinctions are load-bearing, and each one exists because collapsing
 * it produces a specific wrong belief:
 *
 *   - a stage that COMPLETED is not a stage that met its CONTRACT. A stage can
 *     run out of iterations or give up and still be `completed`, and that is
 *     exactly the stage whose output nothing downstream should be built on.
 *   - a run WAITING on a person is not a run that DIED. They have the same
 *     status; told apart, one needs an approval and the other needs someone to
 *     find out what killed the worker.
 *   - a stage not yet REACHED is not a stage that does not EXIST. A view built
 *     only from stages that have jobs reports two stages at stage two of six,
 *     which reads as a finished run.
 *
 * The two write actions are the ones that do not restart the world. Restarting
 * a stage re-fires the chain from its predecessor, so the evidence the retry
 * builds on is the evidence that was already established; inserting a stage
 * adds the step the plan was missing without redoing either side of it. Both
 * take a note, because the reason a stage stopped is usually something only a
 * person knows, and a stage restarted without one repeats what it just did.
 */

import clsx from 'clsx';
import {
  AlertTriangle,
  Ban,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDashed,
  Clock,
  Flag,
  Loader2,
  PauseCircle,
  Plus,
  RotateCcw,
  UserCheck,
  X,
  XCircle,
} from 'lucide-react';
import React, { useCallback, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { useQuery } from 'react-query';

import { apiClient } from '../../services/api';
import type { PipelineRun, PipelineRunStage } from '../../types';
import Button from '../common/Button';

export interface PipelineRunProgressProps {
  /** The head of the chain. Every stage of the run hangs off it, and it is
   *  the id every one of these endpoints is addressed by. */
  rootJobId: string;
  /** The job currently being looked at, so its stage can say so. Optional:
   *  the studio shows a run with no job open. */
  currentJobId?: string;
  /** Open one stage's job. Without it the stage rows are read-only, which is
   *  the right default for a caller that has nowhere to send you. */
  onOpenJob?: (jobId: string) => void;
  className?: string;
}

/** A run that is still moving is polled; a finished one is not. Polling a
 *  completed run forever is the cheapest possible bug to ship and the hardest
 *  to notice. */
const LIVE_STATUSES = new Set(['pending', 'running', 'paused']);

const RUN_STATUS_LABEL: Record<string, string> = {
  pending: 'Queued',
  running: 'Running',
  waiting: 'Waiting for you',
  paused: 'Paused',
  failed: 'Failed',
  cancelled: 'Cancelled',
  completed: 'Completed',
  completed_unmet: 'Finished — contracts unmet',
};

const RUN_STATUS_CLASS: Record<string, string> = {
  pending: 'border-gray-300 bg-gray-100 text-gray-600',
  running: 'border-primary-500/60 bg-primary-500/10 text-primary-700',
  waiting: 'border-amber-300 bg-amber-50 text-amber-700',
  paused: 'border-gray-300 bg-gray-100 text-gray-600',
  failed: 'border-red-300 bg-red-50 text-red-700',
  cancelled: 'border-gray-300 bg-gray-100 text-gray-600',
  completed: 'border-primary-500/60 bg-primary-500/10 text-primary-700',
  // Deliberately not the completed colour. A run that finished without the
  // evidence it promised is not a green run, and painting it green is how a
  // pipeline reports success for work nobody produced.
  completed_unmet: 'border-amber-300 bg-amber-50 text-amber-700',
};

function StageIcon({ stage }: { stage: PipelineRunStage }) {
  if (stage.waiting_on_person) {
    return <UserCheck className="w-4 h-4 text-amber-600" aria-hidden />;
  }
  switch (stage.status) {
    case 'running':
      return <Loader2 className="w-4 h-4 text-primary-600 animate-spin" aria-hidden />;
    case 'completed':
      return stage.contract_satisfied ? (
        <CheckCircle2 className="w-4 h-4 text-primary-600" aria-hidden />
      ) : (
        // Completed, and it did not produce what it promised. The icon has to
        // differ from the green tick or the one state you must not build on
        // looks like the one you can.
        <AlertTriangle className="w-4 h-4 text-amber-600" aria-hidden />
      );
    case 'failed':
      return <XCircle className="w-4 h-4 text-red-600" aria-hidden />;
    case 'cancelled':
      return <Ban className="w-4 h-4 text-gray-500" aria-hidden />;
    case 'paused':
      return <PauseCircle className="w-4 h-4 text-gray-500" aria-hidden />;
    default:
      return <CircleDashed className="w-4 h-4 text-gray-400" aria-hidden />;
  }
}

function elapsed(stage: PipelineRunStage): string {
  if (!stage.started_at) return '';
  const started = new Date(stage.started_at).getTime();
  const ended = stage.completed_at ? new Date(stage.completed_at).getTime() : Date.now();
  const seconds = Math.max(0, Math.round((ended - started) / 1000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

/** Why a stage cannot be restarted, in the words the server would use. Said
 *  up front rather than as a 400 after the click. */
function whyNotRestartable(stage: PipelineRunStage): string {
  if (stage.status === 'running') {
    return 'Still running. Cancel it first — two runners on one stage is what the execution lease exists to stop.';
  }
  if (!stage.job_id) return 'This stage has not started yet.';
  return 'This stage cannot be run again.';
}

export const PipelineRunProgress: React.FC<PipelineRunProgressProps> = ({
  rootJobId,
  currentJobId,
  onOpenJob,
  className,
}) => {
  const [openForm, setOpenForm] = useState<
    { kind: 'restart' | 'insert'; stage: string } | null
  >(null);
  const [note, setNote] = useState('');
  const [newStageId, setNewStageId] = useState('');
  const [newStageGoal, setNewStageGoal] = useState('');
  const [newStageEvidence, setNewStageEvidence] = useState('');
  const [newStageJobType, setNewStageJobType] = useState('research');
  const [submitting, setSubmitting] = useState(false);

  const { data, isLoading, error, refetch } = useQuery<PipelineRun>(
    ['pipeline-run', rootJobId],
    () => apiClient.getPipelineRun(rootJobId),
    {
      enabled: Boolean(rootJobId),
      // A job that is not part of a pipeline 404s here, and this component is
      // rendered for every job so that it can find out. Retrying that is
      // noise.
      retry: false,
      refetchInterval: (run) =>
        run && LIVE_STATUSES.has(run.status) ? 6000 : false,
    }
  );

  const closeForm = useCallback(() => {
    setOpenForm(null);
    setNote('');
    setNewStageId('');
    setNewStageGoal('');
    setNewStageEvidence('');
    setNewStageJobType('research');
  }, []);

  const handleRestart = useCallback(
    async (stage: string) => {
      setSubmitting(true);
      try {
        await apiClient.restartPipelineStage(rootJobId, stage, note.trim() || undefined);
        toast.success(`Restarted ${stage}`);
        closeForm();
        refetch();
      } catch (err: any) {
        // The refusals are the substance of this endpoint: it declines to
        // restart a stage whose ground is not real, and says which stage to
        // restart instead. Passing that through verbatim is the whole value.
        toast.error(err?.response?.data?.detail || 'Could not restart the stage');
      } finally {
        setSubmitting(false);
      }
    },
    [rootJobId, note, closeForm, refetch]
  );

  const handleInsert = useCallback(
    async (after: string) => {
      const evidence = newStageEvidence
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      if (!newStageId.trim() || !newStageGoal.trim() || evidence.length === 0) {
        toast.error('A new stage needs an id, a goal, and something it must produce');
        return;
      }
      setSubmitting(true);
      try {
        const result = await apiClient.insertPipelineStage(rootJobId, {
          after,
          stage: {
            id: newStageId.trim(),
            goal: newStageGoal.trim(),
            job_type: newStageJobType,
            contract: { required_finding_types: evidence },
          },
          note: note.trim() || undefined,
        });
        toast.success(
          result.displaced.length
            ? `Inserted ${result.stage}; ${result.displaced.join(', ')} now runs after it`
            : `Inserted ${result.stage}`
        );
        closeForm();
        refetch();
      } catch (err: any) {
        toast.error(err?.response?.data?.detail || 'Could not insert the stage');
      } finally {
        setSubmitting(false);
      }
    },
    [
      rootJobId,
      newStageId,
      newStageGoal,
      newStageEvidence,
      newStageJobType,
      note,
      closeForm,
      refetch,
    ]
  );

  const percent = useMemo(() => {
    if (!data || !data.total_stages) return 0;
    return Math.round((data.completed_stages / data.total_stages) * 100);
  }, [data]);

  // Not a pipeline run, or not one this user can see. Rendering nothing is
  // correct: this sits inside a panel shown for every job, most of which are
  // not stages of anything.
  if (error || (!isLoading && !data)) return null;
  if (isLoading && !data) return null;
  if (!data || data.stages.length === 0) return null;

  const statusLabel = RUN_STATUS_LABEL[data.status] || data.status;
  const statusClass =
    RUN_STATUS_CLASS[data.status] || 'border-gray-300 bg-gray-100 text-gray-600';

  return (
    <div className={clsx('mb-4', className)} data-testid="pipeline-run-progress">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
          <Flag className="w-4 h-4" />
          Pipeline run
        </h3>
        {data.pipeline && (
          <span className="text-xs text-gray-500 truncate max-w-[16rem]">
            {data.pipeline}
          </span>
        )}
        <span
          className={clsx(
            'px-2 py-0.5 rounded-full border text-[11px] font-medium',
            statusClass
          )}
        >
          {statusLabel}
        </span>
      </div>

      <div className="bg-gray-100 border border-gray-300 rounded-lg p-3">
        <div className="flex items-center justify-between text-xs text-gray-600 mb-1.5">
          <span>
            {data.completed_stages} of {data.total_stages} stages
            {data.current_stage ? ` · on ${data.current_stage}` : ''}
          </span>
          <span>{percent}%</span>
        </div>
        <div
          className="h-1.5 rounded-full bg-gray-200 overflow-hidden mb-3"
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Pipeline progress"
        >
          <div
            className={clsx(
              'h-full rounded-full transition-all duration-slow ease-ui',
              data.status === 'failed' ? 'bg-red-500' : 'bg-primary-500'
            )}
            style={{ width: `${percent}%` }}
          />
        </div>

        <ol className="space-y-1">
          {data.stages.map((stage) => {
            const isCurrent = stage.stage === data.current_stage;
            const isOpenJob = Boolean(currentJobId && stage.job_id === currentJobId);
            const formOpen = openForm?.stage === stage.stage;
            return (
              <li
                key={stage.stage}
                className={clsx(
                  'rounded-md border px-2.5 py-2 transition-colors duration-fast',
                  isOpenJob
                    ? 'border-primary-500/60 bg-primary-500/5'
                    : isCurrent
                      ? 'border-gray-400 bg-gray-50'
                      : 'border-transparent bg-gray-50/60'
                )}
              >
                <div className="flex items-start gap-2">
                  <div className="pt-0.5 flex-none">
                    <StageIcon stage={stage} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-xs font-medium text-gray-800">
                        {stage.stage}
                      </span>
                      {isOpenJob && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-primary-500/60 text-primary-700">
                          open
                        </span>
                      )}
                      {stage.checkpoint && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded-full border border-amber-300 bg-amber-50 text-amber-700"
                          title="This stage holds the run until a person approves it, so completed and idle is expected here."
                        >
                          checkpoint
                        </span>
                      )}
                      {stage.status === 'completed' && !stage.contract_satisfied && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded-full border border-amber-300 bg-amber-50 text-amber-700"
                          title="It finished without producing what it promised. Nothing downstream should be built on this."
                        >
                          contract unmet
                        </span>
                      )}
                      {stage.attempts > 1 && (
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded-full border border-gray-300 text-gray-600"
                          title="This stage has been restarted; everything shown is the latest attempt."
                        >
                          attempt {stage.attempts}
                        </span>
                      )}
                    </div>
                    {stage.goal && (
                      <p className="text-[11px] text-gray-500 mt-0.5 line-clamp-2">
                        {stage.goal}
                      </p>
                    )}
                    <div className="flex flex-wrap items-center gap-2 mt-1 text-[10px] text-gray-500">
                      <span className="capitalize">
                        {stage.waiting_on_person ? 'waiting for you' : stage.status}
                      </span>
                      {stage.iteration > 0 && <span>iteration {stage.iteration}</span>}
                      {elapsed(stage) && (
                        <span className="inline-flex items-center gap-0.5">
                          <Clock className="w-3 h-3" />
                          {elapsed(stage)}
                        </span>
                      )}
                    </div>
                    {stage.error && (
                      <p className="text-[11px] text-red-600 mt-1 line-clamp-2">
                        {stage.error}
                      </p>
                    )}
                  </div>

                  <div className="flex-none flex items-center gap-1">
                    {stage.job_id && onOpenJob && (
                      <button
                        type="button"
                        className="text-[11px] text-gray-600 hover:text-primary-700 inline-flex items-center"
                        onClick={() => onOpenJob(stage.job_id)}
                      >
                        Open
                        <ChevronRight className="w-3 h-3" />
                      </button>
                    )}
                    {stage.job_id && (
                      <button
                        type="button"
                        aria-label={`Restart ${stage.stage}`}
                        title={
                          stage.restartable
                            ? 'Run this stage again on the evidence the stages before it established'
                            : whyNotRestartable(stage)
                        }
                        disabled={!stage.restartable}
                        className="p-1 rounded text-gray-500 hover:text-primary-700 hover:bg-gray-200 disabled:opacity-40 disabled:hover:bg-transparent"
                        onClick={() => {
                          setOpenForm(
                            formOpen && openForm?.kind === 'restart'
                              ? null
                              : { kind: 'restart', stage: stage.stage }
                          );
                          setNote('');
                        }}
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {stage.status === 'completed' && stage.contract_satisfied && (
                      <button
                        type="button"
                        aria-label={`Insert a stage after ${stage.stage}`}
                        title="Add a stage the plan did not have, without redoing the stages that worked"
                        className="p-1 rounded text-gray-500 hover:text-primary-700 hover:bg-gray-200"
                        onClick={() => {
                          setOpenForm(
                            formOpen && openForm?.kind === 'insert'
                              ? null
                              : { kind: 'insert', stage: stage.stage }
                          );
                          // Both forms take a note and share the field. A
                          // correction typed for a restart is not the reason
                          // for an insertion.
                          setNote('');
                        }}
                      >
                        <Plus className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {formOpen && openForm?.kind === 'restart' && (
                  <div className="mt-2 pl-6 space-y-2">
                    <label
                      className="block text-[11px] text-gray-600"
                      htmlFor={`restart-note-${stage.stage}`}
                    >
                      What should it do differently? A stage restarted without a
                      correction usually repeats what it just did.
                    </label>
                    <textarea
                      id={`restart-note-${stage.stage}`}
                      rows={2}
                      value={note}
                      onChange={(e) => setNote(e.target.value)}
                      placeholder="The profile was too coarse — sample per function, not per file"
                      className="w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300
                        focus:outline-none focus:border-primary-600"
                    />
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        loading={submitting}
                        disabled={submitting}
                        onClick={() => handleRestart(stage.stage)}
                      >
                        <Check className="w-3.5 h-3.5 mr-1" />
                        Run {stage.stage} again
                      </Button>
                      <Button size="sm" variant="ghost" onClick={closeForm}>
                        <X className="w-3.5 h-3.5 mr-1" />
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}

                {formOpen && openForm?.kind === 'insert' && (
                  <div className="mt-2 pl-6 space-y-2">
                    <p className="text-[11px] text-gray-600">
                      A new stage after <span className="font-medium">{stage.stage}</span>.
                      Whatever ran after it re-derives beneath the new stage.
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      <input
                        aria-label="New stage id"
                        value={newStageId}
                        onChange={(e) => setNewStageId(e.target.value)}
                        placeholder="stage id"
                        className="px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 focus:outline-none focus:border-primary-600"
                      />
                      <select
                        aria-label="New stage job type"
                        value={newStageJobType}
                        onChange={(e) => setNewStageJobType(e.target.value)}
                        className="px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 focus:outline-none focus:border-primary-600"
                      >
                        {/* A stage's tools are restricted by its job type, and
                            a coding stage left at research is planned with
                            tools it then cannot run. */}
                        {['research', 'analysis', 'coding', 'synthesis', 'custom'].map(
                          (t) => (
                            <option key={t} value={t}>
                              {t}
                            </option>
                          )
                        )}
                      </select>
                    </div>
                    <input
                      aria-label="New stage goal"
                      value={newStageGoal}
                      onChange={(e) => setNewStageGoal(e.target.value)}
                      placeholder="What must be true when this stage is done"
                      className="w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 focus:outline-none focus:border-primary-600"
                    />
                    <input
                      aria-label="Evidence the new stage must produce"
                      value={newStageEvidence}
                      onChange={(e) => setNewStageEvidence(e.target.value)}
                      placeholder="finding types it must produce, comma separated"
                      className="w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 focus:outline-none focus:border-primary-600"
                    />
                    <textarea
                      aria-label="Why this stage is being inserted"
                      rows={2}
                      value={note}
                      onChange={(e) => setNote(e.target.value)}
                      placeholder="Why the plan needed this stage"
                      className="w-full px-2 py-1 text-xs rounded-md bg-gray-50 border border-gray-300 focus:outline-none focus:border-primary-600"
                    />
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        loading={submitting}
                        disabled={submitting}
                        onClick={() => handleInsert(stage.stage)}
                      >
                        <Plus className="w-3.5 h-3.5 mr-1" />
                        Insert after {stage.stage}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={closeForm}>
                        <X className="w-3.5 h-3.5 mr-1" />
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
};

export default PipelineRunProgress;
