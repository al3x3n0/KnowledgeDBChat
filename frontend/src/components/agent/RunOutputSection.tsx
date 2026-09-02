/**
 * What a run produced: a patch proposal, the execution behind it, the
 * experiment runs, the project it generated, the demo check on that project.
 *
 * All of this used to live *inside* the customer-research card, which is
 * rendered only when the job carries a customer profile, a domain-research
 * block, a document artifact, a reading list or an arXiv source. A coding run
 * has none of those — its artifact is a `code_patch_proposal` — so a run whose
 * whole output was a patch showed nothing at all. As a sibling section with
 * its own condition, it appears when there is something to appear for.
 *
 * The views it renders are pure functions of the job (`codePatchView`,
 * `experimentRunsView`), so they are derived here rather than threaded in.
 */

import React, { useMemo } from 'react';
import toast from 'react-hot-toast';
import { useQueryClient } from 'react-query';
import { useNavigate } from 'react-router-dom';

import { apiClient } from '../../services/api';
import type { AgentJob } from '../../types';
import {
  codePatchView,
  executionGraphView,
  experimentRunsView,
} from '../../utils/agentJobDetail';
import { copyText } from '../../utils/clipboard';
import {
  isExperimentRecoveryOpen,
  summarizeExperimentRun,
  type OperatorInterventionSummary,
} from '../../utils/experimentRunSummary';
import Button from '../common/Button';
import type { UnsafeExecBadge } from './jobConfig';

interface RunOutputSectionProps {
  job: AgentJob;
  /** The interventions banner on the latest experiment run. */
  operatorInterventionSummary: OperatorInterventionSummary;
  /** Why the execution graph is unhealthy, shown against an open recovery. */
  graphHealthReasons: string[];
  openCheckpointQueue: () => void;
  openDocument: (docId: string) => void;
  /** Used to launch a demo check; its loading state disables the button. */
  createMutation: any;
  unsafeExecBadge: UnsafeExecBadge;
}

/** The generated project, from results or from the artifact that announced it. */
const generatedProjectOf = (job: AgentJob) => {
  const fromResults = (job.results as any)?.generated_project;
  if (fromResults?.source_id) {
    const behavioral = fromResults?.sanity_check?.behavioral;
    return {
      source_id: String(fromResults.source_id),
      source_name: String(fromResults.source_name || 'Generated project'),
      project_name: String(fromResults.project_name || fromResults.source_name || 'Generated project'),
      entrypoint: String(fromResults.entrypoint || 'demo.py'),
      file_count: Number(fromResults.file_count || 0),
      sanity_ok: fromResults?.sanity_check?.ok === true,
      sanity_errors_count: Array.isArray(fromResults?.sanity_check?.syntax_errors)
        ? fromResults.sanity_check.syntax_errors.length
        : 0,
      behavioral,
    };
  }
  const arts = (job.output_artifacts as any[]) || [];
  const art = arts.find((a) => a?.type === 'generated_project' && a?.source_id);
  if (art?.source_id) {
    return {
      source_id: String(art.source_id),
      source_name: String(art.title || 'Generated project'),
      project_name: String(art.title || 'Generated project'),
      entrypoint: 'demo.py',
      file_count: 0,
      sanity_ok: false,
      sanity_errors_count: 0,
      behavioral: null,
    };
  }
  return null;
};

const demoCheckOf = (job: AgentJob) => {
  const fromResults = (job.results as any)?.demo_check;
  if (!fromResults?.source_id) return null;
  return {
    source_id: String(fromResults.source_id),
    source_name: String(fromResults.source_name || ''),
    entrypoint: String(fromResults.entrypoint || 'demo.py'),
    ok: fromResults.ok === true,
    behavioral: fromResults.behavioral,
  };
};

export const RunOutputSection: React.FC<RunOutputSectionProps> = ({
  job,
  operatorInterventionSummary,
  graphHealthReasons,
  openCheckpointQueue,
  openDocument,
  createMutation,
  unsafeExecBadge,
}) => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // The whole view: it is one derivation and the markup below reads most of it.
  const {
    codePatchProposal,
    codePatchProposals,
    codePatchExecution,
    codePatchExecutionPlan,
    codePatchDetectedStack,
    codePatchApply,
    codePatchKbApply,
    codingRecoveryState,
    codePatchWorkspace,
    codePatchVerificationPlan,
    codePatchVerificationCommands,
    codePatchBootstrapCommands,
    codePatchFallbackCommands,
    codePatchFailedCommands,
    codePatchSuggestedActions,
    codePatchRecovery,
  } = useMemo(() => codePatchView(job), [job]);
  const { graphRecommendedActions } = useMemo(() => executionGraphView(job), [job]);
  const { experimentRuns, latestExperimentRunIndex } = useMemo(
    () => experimentRunsView(job),
    [job]
  );
  const generatedProject = useMemo(() => generatedProjectOf(job), [job]);
  const demoCheck = useMemo(() => demoCheckOf(job), [job]);
  const launchMode = String((job as any)?.launch_mode || (job.config as any)?.launch_mode || '')
    .trim()
    .toLowerCase();

  const hasAnything = Boolean(
    codePatchProposal?.proposal_id ||
      codePatchProposals.length > 1 ||
      codePatchExecution ||
      experimentRuns.length > 0 ||
      codePatchApply ||
      codePatchKbApply ||
      generatedProject?.source_id ||
      demoCheck?.source_id
  );
  if (!hasAnything) return null;

  return (
    <div className="mb-4">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Run output</h3>
      <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-3">
        {codePatchProposal?.proposal_id && (
          <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
            <div className="text-xs text-gray-700 min-w-0">
              <div className="font-medium text-gray-800">Code patch</div>
              <div className="text-gray-600 truncate">{codePatchProposal.title}</div>
              <div className="text-gray-600 font-mono truncate">{codePatchProposal.proposal_id}</div>
            </div>
            <div className="flex gap-2 shrink-0">
              <Button
                size="sm"
                variant="secondary"
                onClick={() =>
                  apiClient.downloadCodePatchProposal(codePatchProposal.proposal_id, codePatchProposal.title)
                }
              >
                Download
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={async () => {
                  const ok = window.confirm(
                    'Apply this patch to KnowledgeDB code documents now? This updates the stored file contents.'
                  );
                  if (!ok) return;
                  try {
                    const res = await apiClient.applyCodePatchProposal(codePatchProposal.proposal_id);
                    if ((res.errors || []).length > 0) {
                      toast.error(`Applied with errors: ${(res.errors || []).length}`);
                    } else {
                      toast.success('Patch applied to KB');
                    }
                    queryClient.invalidateQueries(['agent-jobs']);
                  } catch (e: any) {
                    toast.error(e?.response?.data?.detail || e?.message || 'Failed to apply patch');
                  }
                }}
              >
                Apply to KB
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => copyText(codePatchProposal.proposal_id, 'Proposal ID')}
              >
                Copy ID
              </Button>
            </div>
          </div>
        )}
        
        {codePatchProposals.length > 1 ? (
          <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
            <summary className="cursor-pointer text-xs font-medium text-gray-800">Code patch history</summary>
            <div className="mt-2 space-y-2">
              {codePatchProposals.map((p) => (
                <div key={p.proposal_id} className="flex items-center justify-between gap-3 bg-white border border-gray-200 rounded-lg p-2">
                  <div className="text-xs text-gray-700 min-w-0">
                    <div className="text-gray-600 truncate">{p.title}</div>
                    <div className="text-gray-600 font-mono truncate">{p.proposal_id}</div>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <Button size="sm" variant="secondary" onClick={() => apiClient.downloadCodePatchProposal(p.proposal_id, p.title)}>
                      Download
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={async () => {
                        const ok = window.confirm(
                          'Apply this patch to KnowledgeDB code documents now? This updates the stored file contents.'
                        );
                        if (!ok) return;
                        try {
                          const res = await apiClient.applyCodePatchProposal(p.proposal_id);
                          if ((res.errors || []).length > 0) toast.error(`Applied with errors: ${(res.errors || []).length}`);
                          else toast.success('Patch applied to KB');
                          queryClient.invalidateQueries(['agent-jobs']);
                        } catch (e: any) {
                          toast.error(e?.response?.data?.detail || e?.message || 'Failed to apply patch');
                        }
                      }}
                    >
                      Apply to KB
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => copyText(p.proposal_id, 'Proposal ID')}>
                      Copy ID
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </details>
        ) : null}
        
        {codePatchExecution ? (
          <details className="bg-gray-50 border border-gray-200 rounded-lg p-2" open={launchMode === 'quick_start_repo_bug_triage'}>
            <summary className="cursor-pointer text-xs font-medium text-gray-800">Coding execution</summary>
            <div className="mt-2 space-y-3 text-xs text-gray-700">
              <div className="flex flex-wrap gap-1.5">
                {codePatchExecution.mode ? (
                  <span className="px-2 py-0.5 rounded-full bg-slate-50 text-slate-700 border border-slate-100">
                    {String(codePatchExecution.mode)}
                  </span>
                ) : null}
                {codePatchExecution.scope ? (
                  <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-100">
                    Scope {String(codePatchExecution.scope)}
                  </span>
                ) : null}
                {codePatchWorkspace?.created ? (
                  <span className="px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-100">
                    Workspace {Number(codePatchWorkspace.file_count || 0)} files
                  </span>
                ) : null}
                {codePatchVerificationPlan?.auto_inferred ? (
                  <span className="px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-100">
                    Verification auto-inferred
                  </span>
                ) : null}
                {codePatchExecution.proposal_strategy ? (
                  <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
                    Strategy {String(codePatchExecution.proposal_strategy)}
                  </span>
                ) : null}
                {codingRecoveryState ? (
                  <span className="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-100">
                    Recovery {codingRecoveryState.replace(/_/g, ' ')}
                  </span>
                ) : null}
              </div>
        
              {codePatchRecovery ? (
                <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                  <div className="font-medium text-gray-800">Recovery</div>
                  {codePatchRecovery.retry_reason ? (
                    <div className="text-gray-600">{String(codePatchRecovery.retry_reason)}</div>
                  ) : null}
                  {codePatchRecovery.resume_hint ? (
                    <div className="text-gray-600">Resume hint: {String(codePatchRecovery.resume_hint)}</div>
                  ) : null}
                  {codePatchFailedCommands.length > 0 ? (
                    <div>
                      <div className="text-gray-700 font-medium mb-1">Failed commands</div>
                      <div className="space-y-1">
                        {codePatchFailedCommands.map((cmd, idx) => (
                          <div key={`recovery-failed-${idx}`} className="font-mono text-gray-600 break-all">{cmd}</div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {codePatchRecovery.latest_failed_output ? (
                    <div>
                      <div className="text-gray-700 font-medium mb-1">Latest failed output</div>
                      <div className="text-gray-600 font-mono whitespace-pre-wrap break-words">
                        {String(codePatchRecovery.latest_failed_output)}
                      </div>
                    </div>
                  ) : null}
                  {codePatchSuggestedActions.length > 0 ? (
                    <div className="text-gray-500">
                      Suggested actions: {codePatchSuggestedActions.map((item) => String(item).replace(/_/g, ' ')).join(', ')}
                    </div>
                  ) : null}
                </div>
              ) : null}
        
              {codePatchExecution.failure_symptom ? (
                <div>
                  <div className="font-medium text-gray-800 mb-1">Failure symptom</div>
                  <div className="text-gray-600 whitespace-pre-wrap">{String(codePatchExecution.failure_symptom)}</div>
                </div>
              ) : null}
        
              {codePatchExecution.error_output ? (
                <div>
                  <div className="font-medium text-gray-800 mb-1">Error output</div>
                  <div className="text-gray-600 font-mono whitespace-pre-wrap break-words">
                    {String(codePatchExecution.error_output)}
                  </div>
                </div>
              ) : null}
        
              {codePatchWorkspace ? (
                <div className="bg-white border border-gray-200 rounded p-2 space-y-1">
                  <div className="font-medium text-gray-800">Workspace</div>
                  <div className="text-gray-600">
                    {codePatchWorkspace.created ? 'ready' : 'not created'}
                    {codePatchWorkspace.source_type ? ` • source ${String(codePatchWorkspace.source_type)}` : ''}
                    {codePatchWorkspace.file_count !== undefined ? ` • files ${Number(codePatchWorkspace.file_count || 0)}` : ''}
                  </div>
                  {codePatchWorkspace.workspace_id ? (
                    <div className="text-gray-500 font-mono break-all">{String(codePatchWorkspace.workspace_id)}</div>
                  ) : null}
                  {codePatchWorkspace.error ? (
                    <div className="text-rose-700">{String(codePatchWorkspace.error)}</div>
                  ) : null}
                </div>
              ) : null}
        
              {codePatchDetectedStack.length > 0 ? (
                <div className="bg-white border border-gray-200 rounded p-2">
                  <div className="font-medium text-gray-800 mb-1">Inferred project profile</div>
                  <div className="text-gray-600">Detected stack: {codePatchDetectedStack.join(', ')}</div>
                </div>
              ) : null}
        
              {codePatchVerificationCommands.length > 0 ? (
                <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                  <div className="font-medium text-gray-800">Verification plan</div>
                  <div>
                    <div className="text-gray-700 font-medium mb-1">Primary commands</div>
                    <div className="text-gray-600 font-mono whitespace-pre-wrap">{codePatchVerificationCommands.join('\n')}</div>
                  </div>
                  {codePatchBootstrapCommands.length > 0 ? (
                    <div>
                      <div className="text-gray-700 font-medium mb-1">Bootstrap</div>
                      <div className="text-gray-600 font-mono whitespace-pre-wrap">{codePatchBootstrapCommands.join('\n')}</div>
                    </div>
                  ) : null}
                  {codePatchFallbackCommands.length > 0 ? (
                    <div>
                      <div className="text-gray-700 font-medium mb-1">Fallback</div>
                      <div className="text-gray-600 font-mono whitespace-pre-wrap">{codePatchFallbackCommands.join('\n')}</div>
                    </div>
                  ) : null}
                </div>
              ) : null}
        
              {codePatchExecutionPlan.length > 0 ? (
                <div className="bg-white border border-gray-200 rounded p-2 space-y-2">
                  <div className="font-medium text-gray-800">Execution plan</div>
                  <div className="space-y-2">
                    {codePatchExecutionPlan.map((step, idx) => (
                      <div key={String(step?.step_id || idx)} className="border border-gray-100 rounded p-2">
                        <div className="flex items-center justify-between gap-2">
                          <div className="font-medium text-gray-800">{String(step?.title || `Step ${idx + 1}`)}</div>
                          {step?.status ? (
                            <span className="px-2 py-0.5 rounded-full bg-slate-50 text-slate-700 border border-slate-100">
                              {String(step.status)}
                            </span>
                          ) : null}
                        </div>
                        {step?.objective ? (
                          <div className="mt-1 text-gray-600">{String(step.objective)}</div>
                        ) : null}
                        {Array.isArray(step?.commands) && step.commands.length > 0 ? (
                          <div className="mt-1 text-gray-600 font-mono whitespace-pre-wrap">
                            {step.commands.join('\n')}
                          </div>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </details>
        ) : null}
        
        {experimentRuns.length > 0 ? (
          <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
            <summary className="cursor-pointer text-xs font-medium text-gray-800">Experiment runs</summary>
            <div className="mt-2 space-y-2">
              {experimentRuns.map((er, idx) => {
                const okVal = er?.ok;
                const label = okVal === true ? 'PASS' : okVal === false ? 'FAIL' : 'SKIP';
                const labelClass = okVal === true ? 'text-green-700' : okVal === false ? 'text-red-700' : 'text-amber-700';
                const cmds = Array.isArray(er?.commands) ? er.commands : [];
                const isLatestExperimentRun = idx === latestExperimentRunIndex;
                const {
                  verificationCommands: verificationCmds,
                  bootstrapCommands: bootstrapCmds,
                  fallbackCommands: fallbackCmds,
                  phases,
                  failedCommands,
                  finalPhase,
                  sourceId,
                  sourceName,
                  detectedStack,
                } = summarizeExperimentRun(er);
                const recoveryOpen = isExperimentRecoveryOpen(er, {
                  verificationCommands: verificationCmds,
                  bootstrapCommands: bootstrapCmds,
                  fallbackCommands: fallbackCmds,
                  phases,
                  failedCommands,
                  finalPhase,
                  sourceId,
                  sourceName,
                  detectedStack,
                });
                const pid = String(er?.proposal_id || '').trim();
                return (
                  <div key={idx} className="bg-white border border-gray-200 rounded-lg p-2">
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <div className="text-gray-700 min-w-0">
                        <span className={`font-medium ${labelClass}`}>{label}</span>
                        {er?.source_name ? <span className="text-gray-500"> — {String(er.source_name)}</span> : null}
                        {pid ? <span className="text-gray-500"> • </span> : null}
                        {pid ? <span className="text-gray-500 font-mono truncate">{pid}</span> : null}
                      </div>
                      {cmds.length > 0 ? <div className="text-gray-500">{cmds.length} cmd(s)</div> : null}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1 text-[11px]">
                      {finalPhase ? (
                        <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                          Final {finalPhase}
                        </span>
                      ) : null}
                      {Boolean(er?.bootstrap_attempted) ? (
                        <span className={`px-2 py-0.5 rounded-full border ${er?.bootstrap_ok ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                          Bootstrap {er?.bootstrap_ok ? 'ok' : 'attempted'}
                        </span>
                      ) : null}
                      {Boolean(er?.fallback_attempted) ? (
                        <span className={`px-2 py-0.5 rounded-full border ${er?.fallback_ok ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                          Fallback {er?.fallback_ok ? 'ok' : 'attempted'}
                        </span>
                      ) : null}
                      {recoveryOpen ? (
                        <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 border border-rose-200">
                          Recovery open
                        </span>
                      ) : null}
                      {recoveryOpen && graphRecommendedActions.length > 0 ? (
                        <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
                          Next {graphRecommendedActions[0]}
                        </span>
                      ) : null}
                      {phases.length > 0 ? (
                        <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 border border-gray-200">
                          Phases {phases.join(' -> ')}
                        </span>
                      ) : null}
                      {sourceName ? (
                        <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                          Source {sourceName}
                        </span>
                      ) : null}
                      {sourceId ? (
                        <span className="px-2 py-0.5 rounded-full bg-white text-gray-700 border border-gray-200 font-mono">
                          {sourceId}
                        </span>
                      ) : null}
                      {detectedStack.length > 0 ? (
                        <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                          Stack {detectedStack.join(', ')}
                        </span>
                      ) : null}
                      {isLatestExperimentRun && operatorInterventionSummary.latestLabel ? (
                        <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200">
                          Last {operatorInterventionSummary.latestLabel}
                        </span>
                      ) : null}
                      {isLatestExperimentRun && operatorInterventionSummary.latestOutcome ? (
                        <span className="px-2 py-0.5 rounded-full bg-orange-50 text-orange-700 border border-orange-100">
                          Outcome {operatorInterventionSummary.latestOutcome}
                        </span>
                      ) : null}
                    </div>
                    {isLatestExperimentRun && operatorInterventionSummary.recentItems.length > 1 ? (
                      <div className="mt-2 text-[11px] text-amber-800">
                        <div className="font-medium mb-1">Recent intervention timeline</div>
                        <ul className="space-y-1">
                          {operatorInterventionSummary.recentItems.map((item, itemIdx) => (
                            <li key={`${idx}-timeline-${itemIdx}`}>- {item}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {isLatestExperimentRun && operatorInterventionSummary.latestOutcomeReason ? (
                      <div className="mt-2 text-[11px] text-orange-700">
                        <span className="font-medium">Outcome reason:</span> {operatorInterventionSummary.latestOutcomeReason}
                      </div>
                    ) : null}
                    {recoveryOpen && graphHealthReasons.length > 0 ? (
                      <div className="mt-2 text-[11px] text-rose-700">
                        <span className="font-medium">Reason:</span> {graphHealthReasons[0]}
                      </div>
                    ) : null}
                    {recoveryOpen && isLatestExperimentRun ? (
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <Button size="sm" variant="primary" onClick={openCheckpointQueue}>
                          Open in Checkpoint Queue
                        </Button>
                        {failedCommands.length > 0 ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyText(String(failedCommands[0]), 'Failed command')}
                          >
                            Copy failed command
                          </Button>
                        ) : null}
                        {graphRecommendedActions.length > 0 ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyText(String(graphRecommendedActions[0]), 'Recovery next step')}
                          >
                            Copy next step
                          </Button>
                        ) : null}
                      </div>
                    ) : null}
                    {verificationCmds.length > 0 ? (
                      <div className="mt-2 space-y-1">
                        <div className="text-[11px] font-medium text-gray-700">Verification</div>
                        <div className="text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                          {verificationCmds.slice(0, 6).join('\n')}
                        </div>
                      </div>
                    ) : null}
                    {bootstrapCmds.length > 0 ? (
                      <div className="mt-2 space-y-1">
                        <div className="text-[11px] font-medium text-gray-700">Bootstrap</div>
                        <div className="text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                          {bootstrapCmds.slice(0, 4).join('\n')}
                        </div>
                      </div>
                    ) : null}
                    {fallbackCmds.length > 0 ? (
                      <div className="mt-2 space-y-1">
                        <div className="text-[11px] font-medium text-gray-700">Fallback verification</div>
                        <div className="text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                          {fallbackCmds.slice(0, 4).join('\n')}
                        </div>
                      </div>
                    ) : null}
                    {failedCommands.length > 0 ? (
                      <div className="mt-2 space-y-1">
                        <div className="text-[11px] font-medium text-rose-700">Failed commands</div>
                        <div className="text-[11px] text-rose-700 font-mono whitespace-pre-wrap">
                          {failedCommands.slice(0, 4).join('\n')}
                        </div>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </details>
        ) : null}
        
        {codePatchApply ? (
          <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
            <div className="text-xs text-gray-700 min-w-0">
              <div className="font-medium text-gray-800">Patch apply (sandbox)</div>
              <div className="text-gray-600">
                applied: {Array.isArray(codePatchApply.applied) ? codePatchApply.applied.length : 0} • errors:{' '}
                {Array.isArray(codePatchApply.errors) ? codePatchApply.errors.length : 0}
              </div>
              {codePatchApply.proposal_id ? (
                <div className="text-gray-600 font-mono truncate">{String(codePatchApply.proposal_id)}</div>
              ) : null}
            </div>
          </div>
        ) : null}
        
        {codePatchKbApply ? (
          <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
            <summary className="cursor-pointer text-xs font-medium text-gray-800">Patch apply (Knowledge DB)</summary>
            <div className="mt-2 space-y-2 text-xs text-gray-700">
              <div className="text-gray-600">
                {codePatchKbApply.enabled === false
                  ? 'skipped'
                  : codePatchKbApply.dry_run
                    ? `dry-run — ok: ${String(codePatchKbApply.ok)}`
                    : `applied: ${String(codePatchKbApply.did_apply)} — ok: ${String(codePatchKbApply.ok)}`}
                {' • '}
                errors: {Array.isArray(codePatchKbApply.errors) ? codePatchKbApply.errors.length : 0}
                {' • '}
                files: {Array.isArray(codePatchKbApply.applied_files) ? codePatchKbApply.applied_files.length : 0}
              </div>
              {codePatchKbApply.blocked_reason ? (
                <div className="text-yellow-800 bg-yellow-50 border border-yellow-200 rounded px-2 py-1">
                  Blocked: {String(codePatchKbApply.blocked_reason)}
                </div>
              ) : null}
              {codePatchKbApply.proposal_strategy ? (
                <div className="text-gray-500">strategy: {String(codePatchKbApply.proposal_strategy)}</div>
              ) : null}
              {codePatchKbApply.proposal_id ? (
                <div className="text-gray-600 font-mono truncate">{String(codePatchKbApply.proposal_id)}</div>
              ) : null}
        
              {Array.isArray(codePatchKbApply.applied_files) && codePatchKbApply.applied_files.length > 0 ? (
                <div className="space-y-1">
                  <div className="font-medium text-gray-800">Applied files</div>
                  <div className="space-y-1">
                    {codePatchKbApply.applied_files.slice(0, 50).map((f: any, i: number) => (
                      <div key={String(f?.document_id || f?.path || i)} className="flex items-center justify-between gap-2 bg-white border border-gray-200 rounded px-2 py-1">
                        <div className="min-w-0">
                          <div className="text-gray-600 font-mono truncate">{String(f?.path || '(unknown path)')}</div>
                          {f?.document_id ? (
                            <div className="text-gray-500 font-mono truncate">{String(f.document_id)}</div>
                          ) : null}
                        </div>
                        <div className="flex gap-2 shrink-0">
                          {f?.document_id ? (
                            <Button size="sm" variant="secondary" onClick={() => openDocument(String(f.document_id))}>
                              Open
                            </Button>
                          ) : null}
                          {f?.document_id ? (
                            <Button size="sm" variant="ghost" onClick={() => copyText(String(f.document_id), 'Document ID')}>
                              Copy ID
                            </Button>
                          ) : null}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
        
              {Array.isArray(codePatchKbApply.errors) && codePatchKbApply.errors.length > 0 ? (
                <div className="space-y-1">
                  <div className="font-medium text-gray-800">Errors</div>
                  <div className="space-y-1">
                    {codePatchKbApply.errors.slice(0, 50).map((e: any, i: number) => (
                      <div key={String(e?.path || i)} className="bg-white border border-red-200 rounded px-2 py-1">
                        <div className="text-red-800 font-mono">{String(e?.path || '(unknown file)')}</div>
                        <div className="text-red-700">{String(e?.error || e?.message || '')}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          </details>
        ) : null}
        
        {generatedProject?.source_id && (
          <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
            <div className="text-xs text-gray-700 min-w-0">
              <div className="font-medium text-gray-800">Generated project</div>
              <div className="text-gray-600 truncate">{generatedProject.project_name}</div>
              <div className="text-gray-600 font-mono truncate">{generatedProject.source_id}</div>
              {generatedProject.file_count ? (
                <div className="text-gray-500">{generatedProject.file_count} files</div>
              ) : null}
              {generatedProject.sanity_errors_count ? (
                <div className="text-red-600">Syntax errors: {generatedProject.sanity_errors_count}</div>
              ) : generatedProject.sanity_ok ? (
                <div className="text-green-700">Sanity check: OK</div>
              ) : null}
              {generatedProject.sanity_ok && generatedProject.behavioral?.enabled === false ? (
                <div className="text-amber-700">Behavioral check: skipped (server disabled)</div>
              ) : generatedProject.behavioral?.ran ? (
                generatedProject.behavioral?.ok ? (
                  <div className="text-green-700">Behavioral check: OK</div>
                ) : (
                  <div className="text-red-600">Behavioral check: failed</div>
                )
              ) : null}
              {generatedProject.behavioral?.ran ? (
                <details className="mt-2">
                  <summary className="cursor-pointer text-gray-700">Behavior details</summary>
                  <div className="mt-2 space-y-2">
                    <div className="text-gray-700">
                      Backend: <span className="font-mono">{String(generatedProject.behavioral.backend || '')}</span>
                      {typeof generatedProject.behavioral.duration_ms === 'number' ? (
                        <span className="ml-2">({generatedProject.behavioral.duration_ms}ms)</span>
                      ) : null}
                      {generatedProject.behavioral.timed_out ? <span className="ml-2 text-red-600">timeout</span> : null}
                    </div>
                    {generatedProject.behavioral.error ? (
                      <div className="text-red-700">Error: {String(generatedProject.behavioral.error)}</div>
                    ) : null}
                    {typeof generatedProject.behavioral.exit_code === 'number' ? (
                      <div className="text-gray-700">
                        Exit code: <span className="font-mono">{String(generatedProject.behavioral.exit_code)}</span>
                      </div>
                    ) : null}
                    {typeof generatedProject.behavioral.stdout === 'string' && generatedProject.behavioral.stdout.trim() ? (
                      <div>
                        <div className="flex items-center justify-between">
                          <div className="text-gray-700">stdout</div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyText(String(generatedProject.behavioral.stdout || ''), 'stdout')}
                          >
                            Copy
                          </Button>
                        </div>
                        <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                          {String(generatedProject.behavioral.stdout)}
                        </pre>
                      </div>
                    ) : null}
                    {typeof generatedProject.behavioral.stderr === 'string' && generatedProject.behavioral.stderr.trim() ? (
                      <div>
                        <div className="flex items-center justify-between">
                          <div className="text-gray-700">stderr</div>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => copyText(String(generatedProject.behavioral.stderr || ''), 'stderr')}
                          >
                            Copy
                          </Button>
                        </div>
                        <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                          {String(generatedProject.behavioral.stderr)}
                        </pre>
                      </div>
                    ) : null}
                  </div>
                </details>
              ) : null}
            </div>
            <div className="flex gap-2 shrink-0">
              <Button
                size="sm"
                variant="secondary"
                onClick={() =>
                  apiClient.downloadDocumentSourceZip(generatedProject.source_id, generatedProject.project_name)
                }
              >
                Download ZIP
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => navigate('/documents', { state: { selectedSourceId: generatedProject.source_id } })}
              >
                Open
              </Button>
              <Button
                size="sm"
                variant="secondary"
                disabled={unsafeExecBadge.status !== 'ready' || createMutation.isLoading}
                title={
                  unsafeExecBadge.status === 'ready'
                    ? 'Run sandboxed demo check again'
                    : 'Demo check not available (see badge on Implement Algorithm)'
                }
                onClick={() =>
                  createMutation.mutate({
                    name: `Demo check — ${generatedProject.project_name}`.slice(0, 120),
                    job_type: 'monitor' as any,
                    goal: `Run demo check (${generatedProject.project_name})`,
                    config: {
                      deterministic_runner: 'generated_project_demo_check',
                      source_id: generatedProject.source_id,
                      entrypoint: generatedProject.entrypoint || 'demo.py',
                    },
                    max_iterations: 1,
                    max_tool_calls: 0,
                    max_llm_calls: 0,
                    max_runtime_minutes: 5,
                    start_immediately: true,
                  })
                }
              >
                Re-run demo
              </Button>
              <Button size="sm" variant="ghost" onClick={() => copyText(generatedProject.source_id, 'Source ID')}>
                Copy ID
              </Button>
            </div>
          </div>
        )}
        
        {demoCheck?.source_id && (
          <div className="flex items-start justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
            <div className="text-xs text-gray-700 min-w-0">
              <div className="font-medium text-gray-800">Demo check</div>
              <div className="text-gray-600 font-mono truncate">{demoCheck.source_id}</div>
              <div className={demoCheck.ok ? 'text-green-700' : 'text-red-600'}>
                {demoCheck.ok ? 'OK' : 'FAILED'} • {demoCheck.entrypoint}
              </div>
              {demoCheck.behavioral?.ran ? (
                <details className="mt-2">
                  <summary className="cursor-pointer text-gray-700">Details</summary>
                  <div className="mt-2 space-y-2">
                    <div className="text-gray-700">
                      Backend: <span className="font-mono">{String(demoCheck.behavioral.backend || '')}</span>
                      {typeof demoCheck.behavioral.duration_ms === 'number' ? (
                        <span className="ml-2">({demoCheck.behavioral.duration_ms}ms)</span>
                      ) : null}
                      {demoCheck.behavioral.timed_out ? <span className="ml-2 text-red-600">timeout</span> : null}
                    </div>
                    {demoCheck.behavioral.error ? (
                      <div className="text-red-700">Error: {String(demoCheck.behavioral.error)}</div>
                    ) : null}
                    {typeof demoCheck.behavioral.exit_code === 'number' ? (
                      <div className="text-gray-700">
                        Exit code: <span className="font-mono">{String(demoCheck.behavioral.exit_code)}</span>
                      </div>
                    ) : null}
                    {typeof demoCheck.behavioral.stdout === 'string' && demoCheck.behavioral.stdout.trim() ? (
                      <div>
                        <div className="flex items-center justify-between">
                          <div className="text-gray-700">stdout</div>
                          <Button size="sm" variant="ghost" onClick={() => copyText(String(demoCheck.behavioral.stdout || ''), 'stdout')}>
                            Copy
                          </Button>
                        </div>
                        <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                          {String(demoCheck.behavioral.stdout)}
                        </pre>
                      </div>
                    ) : null}
                    {typeof demoCheck.behavioral.stderr === 'string' && demoCheck.behavioral.stderr.trim() ? (
                      <div>
                        <div className="flex items-center justify-between">
                          <div className="text-gray-700">stderr</div>
                          <Button size="sm" variant="ghost" onClick={() => copyText(String(demoCheck.behavioral.stderr || ''), 'stderr')}>
                            Copy
                          </Button>
                        </div>
                        <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                          {String(demoCheck.behavioral.stderr)}
                        </pre>
                      </div>
                    ) : null}
                  </div>
                </details>
              ) : null}
            </div>
            <div className="flex gap-2 shrink-0">
              <Button size="sm" variant="ghost" onClick={() => copyText(demoCheck.source_id, 'Source ID')}>
                Copy source
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RunOutputSection;
