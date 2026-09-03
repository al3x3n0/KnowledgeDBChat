/**
 * One agent run, as a card in the list.
 *
 * Lifted out of AutonomousAgentsPage's render closure, where it was one of
 * eighteen components declared inside the component body — a new function
 * identity on every render, which React treats as a new component type and
 * remounts rather than reconciles. On a page holding 328 hooks that meant
 * every card in the list remounting on every keystroke anywhere on it.
 *
 * The interesting part of the move is the props. In the closure this card
 * reached for thirteen values from the page: the selection, the navigation
 * helper, and eight filter setters. Out here those are named for what the
 * card is asking to happen — narrow the list to swarm runs, show this run's
 * relaunch children — and the page decides how. The coupling did not go away;
 * it became visible and typed.
 */

import { Cpu, GitBranch, Link2, MessageSquare, RefreshCw } from 'lucide-react';
import React from 'react';

import type {
  AgentJob,
  AgentJobCodePatchExecution,
  AgentJobExecutionGraph,
  AgentJobOperatorIntervention,
  AgentJobStatus,
  AgentJobType,
} from '../../types';
import { normalizeJobMemoryPersistenceSummary } from '../../utils/agentMemoryExtraction';
import {
  isExperimentRecoveryOpen,
  summarizeExperimentRecoveryGuidance,
  summarizeExperimentRun,
  summarizeOperatorInterventions,
} from '../../utils/experimentRunSummary';
import Button from '../common/Button';
import { JOB_TYPE_CONFIG, STATUS_CONFIG } from './jobConfig';
import { getLatestExperimentRun } from './jobFields';
import { copyText } from '../../utils/clipboard';

export interface JobCardProps {
  job: AgentJob;
  /** This job is the deep-linked one and sits outside the current filters. */
  isPinnedDeepLink?: boolean;
  isSelected: boolean;
  /** Open this run: select it and put it in the URL. */
  onOpen: (job: AgentJob) => void;
  /** Open some other run by id — the one this was relaunched from. */
  onOpenRunById: (jobId: string) => void;
  onGoToQueue: () => void;
  onClearLaunchModeFilter: () => void;
  onShowRelaunchChildren: (job: AgentJob) => void;
  /** Drop the deep link, returning the list to its own filters. */
  onClearDeepLink: () => void;
  /** Narrow the list to swarm runs, optionally sorted or thresholded. */
  onNarrowToSwarm: (opts?: { sortBy?: string; minConsensus?: number }) => void;
  onViewChainStatus: (jobId: string) => void;
}

export const JobCard: React.FC<JobCardProps> = ({
  job,
  isPinnedDeepLink = false,
  isSelected,
  onOpen,
  onOpenRunById,
  onGoToQueue,
  onClearLaunchModeFilter,
  onShowRelaunchChildren,
  onClearDeepLink,
  onNarrowToSwarm,
  onViewChainStatus,
}) => {
    const typeConfig = JOB_TYPE_CONFIG[job.job_type as AgentJobType] || JOB_TYPE_CONFIG.custom;
    const statusConfig = STATUS_CONFIG[job.status as AgentJobStatus] || STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;
    const rawFanIn = (job.results as any)?.swarm_fan_in;
    const swarmSummary = ((job as any)?.swarm_summary && typeof (job as any)?.swarm_summary === 'object')
      ? (job as any).swarm_summary
      : null;
    const goalContractSummary = ((job as any)?.goal_contract_summary && typeof (job as any)?.goal_contract_summary === 'object')
      ? (job as any).goal_contract_summary
      : (((job.results as any)?.goal_contract && typeof (job.results as any)?.goal_contract === 'object') ? (job.results as any).goal_contract : null);
    const contractEnabled = Boolean(goalContractSummary?.enabled);
    const contractSatisfied = contractEnabled ? Boolean(goalContractSummary?.satisfied) : true;
    const contractMissingCount = Number(
      goalContractSummary?.missing_count ??
      (Array.isArray(goalContractSummary?.missing) ? goalContractSummary.missing.length : 0)
    );
    const approvalCheckpoint = ((job as any)?.approval_checkpoint && typeof (job as any)?.approval_checkpoint === 'object')
      ? (job as any).approval_checkpoint
      : (((job.results as any)?.approval_checkpoint && typeof (job.results as any)?.approval_checkpoint === 'object')
          ? (job.results as any).approval_checkpoint
          : (((job.results as any)?.execution_strategy?.approval_checkpoints?.pending && typeof (job.results as any)?.execution_strategy?.approval_checkpoints?.pending === 'object')
              ? (job.results as any).execution_strategy.approval_checkpoints.pending
              : null));
    const executionGraph = ((((job.results as any)?.execution_strategy?.execution_graph) && typeof ((job.results as any)?.execution_strategy?.execution_graph) === 'object')
      ? ((job.results as any).execution_strategy.execution_graph as AgentJobExecutionGraph)
      : null);
    const scopeObservability = ((((job.results as any)?.execution_strategy?.scope_observability) && typeof ((job.results as any)?.execution_strategy?.scope_observability) === 'object')
      ? ((job.results as any).execution_strategy.scope_observability as Record<string, any>)
      : null);
    const operatorInterventions = (
      Array.isArray(job.operator_interventions)
        ? (job.operator_interventions as AgentJobOperatorIntervention[])
        : Array.isArray((job.results as any)?.execution_strategy?.operator_interventions)
          ? ((job.results as any).execution_strategy.operator_interventions as AgentJobOperatorIntervention[])
        : []
    );
    const operatorInterventionSummary = summarizeOperatorInterventions(operatorInterventions);
    const cardMemoryPersistence = normalizeJobMemoryPersistenceSummary(
      (job.results as any)?.execution_strategy?.memory_persistence
    );
    const cardMemoryExtraction = cardMemoryPersistence?.extraction || null;
    const graphHealthStatus = String((executionGraph as any)?.graph_health?.status || '').toLowerCase();
    const graphHealthSeverity = Number((executionGraph as any)?.graph_health?.severity_score || 0);
    const { reasons: graphHealthReasons, recommendedActions: graphRecommendedActions } =
      summarizeExperimentRecoveryGuidance(executionGraph as Record<string, any> | null);
    const graphVerificationActions = Array.isArray((executionGraph as any)?.verification_actions)
      ? ((executionGraph as any).verification_actions as Array<Record<string, any>>)
      : [];
    const scopeResolvedId = String(scopeObservability?.resolved_scope_id || '').trim();
    const scopeGuardBlocks = Array.isArray(scopeObservability?.events)
      ? (scopeObservability?.events as Array<Record<string, any>>).filter((event) => String(event?.type || '').trim() === 'scope_guard_blocked').length
      : 0;
    const latestExperimentRun = getLatestExperimentRun(job);
    const latestExperimentSummary = summarizeExperimentRun(latestExperimentRun);
    const latestExperimentFailedCount = latestExperimentSummary.failedCommands.length;
    const latestExperimentVerificationCount = latestExperimentSummary.verificationCommands.length;
    const latestExperimentRecoveryOpen = isExperimentRecoveryOpen(latestExperimentRun, latestExperimentSummary);
    const graphHealthBadgeClass =
      graphHealthStatus === 'critical'
        ? 'bg-red-50 text-red-700 border-red-100'
        : graphHealthStatus === 'warning'
          ? 'bg-amber-50 text-amber-700 border-amber-100'
          : graphHealthStatus === 'ok'
            ? 'bg-emerald-50 text-emerald-700 border-emerald-100'
            : 'bg-gray-50 text-gray-700 border-gray-100';
    const launchMode = String((job as any)?.launch_mode || ((job.config as any)?.launch_mode || '')).toLowerCase();
    const relaunchFromJobId = String((job as any)?.relaunch_from_job_id || ((job.config as any)?.relaunch_from_job_id || '')).trim();
    const relaunchChildrenCount = Math.max(0, Number((job as any)?.relaunch_children_count || 0));
    const hasSwarm = Boolean(swarmSummary || (rawFanIn && typeof rawFanIn === 'object'));
    const consensusCount = Number(
      swarmSummary?.consensus_count ??
      (Array.isArray(rawFanIn?.consensus_findings) ? rawFanIn.consensus_findings.length : 0)
    );
    const conflictCount = Number(
      swarmSummary?.conflict_count ??
      (Array.isArray(rawFanIn?.conflicts) ? rawFanIn.conflicts.length : 0)
    );
    const confidenceOverall = Number(
      swarmSummary?.confidence?.overall ??
      rawFanIn?.confidence?.overall ??
      0
    );
    const cardCodePatchExecution = (((job.results as any)?.code_patch_execution) && typeof ((job.results as any)?.code_patch_execution) === 'object')
      ? ((job.results as any).code_patch_execution as AgentJobCodePatchExecution)
      : null;
    const cardWorkspace = cardCodePatchExecution?.workspace || null;
    const cardExecutionPlan = Array.isArray(cardCodePatchExecution?.execution_plan)
      ? (cardCodePatchExecution?.execution_plan || [])
      : [];
    const cardVerificationPlan = cardCodePatchExecution?.verification_plan || null;
    const cardVerificationCommands = Array.isArray(cardVerificationPlan?.commands)
      ? (cardVerificationPlan?.commands || [])
      : [];
    const cardDetectedStack = Array.isArray((cardCodePatchExecution?.inferred_project_profile as any)?.detected_stack)
      ? ((cardCodePatchExecution?.inferred_project_profile as any)?.detected_stack as any[])
          .map((item) => String(item || '').trim())
          .filter(Boolean)
      : [];
    const cardDomainResearch = (((job.results as any)?.domain_research) && typeof ((job.results as any)?.domain_research) === 'object')
      ? ((job.results as any).domain_research as Record<string, any>)
      : null;
    const cardDomainIdeas = Array.isArray(cardDomainResearch?.proposed_ideas) ? (cardDomainResearch?.proposed_ideas as any[]) : [];
    const cardDomainNoteIds = Array.isArray(cardDomainResearch?.research_note_ids) ? (cardDomainResearch?.research_note_ids as any[]) : [];
    const cardDomainIdeaCount = cardDomainIdeas.length;
    const cardDomainNoteCount = cardDomainNoteIds.length;

    return (
      <div
        // The three utilities `border rounded-lg p-4` stay exactly as they
        // are: eight places in the test suite reach this element with
        // `closest('.border.rounded-lg.p-4')`, so they are structure here, not
        // decoration. Everything below is additive.
        className={`bg-white border rounded-lg p-4 cursor-pointer
          transition-all duration-fast ease-ui
          hover:shadow-level-2 hover:-translate-y-px
          active:translate-y-0 active:shadow-level-1 ${
          isSelected
            // The current card carries an accent bar down its leading edge as
            // well as the ring: at a glance in a grid, the bar is what reads.
            ? 'border-primary-500 ring-2 ring-primary-200 shadow-[inset_3px_0_0_0_theme(colors.primary.600)]'
            : isPinnedDeepLink
              ? 'border-primary-300 ring-1 ring-primary-100'
              : 'border-gray-200 hover:border-gray-400'
        }`}
        onClick={() => {
          onOpen(job);
        }}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-lg ${typeConfig.color}`}>
              <TypeIcon className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 truncate max-w-[200px]">{job.name}</h3>
              <p className="text-xs text-gray-500">{typeConfig.label}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
            <StatusIcon className={`w-3 h-3 ${job.status === 'running' ? 'animate-spin' : ''}`} />
            <span className="capitalize">{job.status}</span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
            <span>Progress</span>
            <span>{job.progress}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                job.status === 'completed' ? 'bg-green-500' :
                job.status === 'failed' ? 'bg-red-500' :
                'bg-primary-500'
              }`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>

        {/* Current phase */}
        {job.current_phase && (
          <p className="text-xs text-gray-600 mb-2 truncate">
            <span className="font-medium">Phase:</span> {job.current_phase}
          </p>
        )}

        {(graphVerificationActions.length > 0 || scopeResolvedId || scopeGuardBlocks > 0) && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            {graphVerificationActions.length > 0 && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-violet-50 text-violet-700 border border-violet-100">
                Verify {graphVerificationActions.length}
              </span>
            )}
            {scopeResolvedId && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-sky-50 text-sky-700 border border-sky-100">
                Scope {scopeResolvedId}
              </span>
            )}
            {scopeGuardBlocks > 0 && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-100">
                Guard blocks {scopeGuardBlocks}
              </span>
            )}
          </div>
        )}

        {(latestExperimentSummary.finalPhase ||
          latestExperimentRun?.bootstrap_attempted ||
          latestExperimentRun?.fallback_attempted ||
          latestExperimentSummary.sourceName ||
          operatorInterventionSummary.latestLabel) && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            {latestExperimentSummary.finalPhase ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-100 text-gray-700 border border-gray-200">
                Final {latestExperimentSummary.finalPhase}
              </span>
            ) : null}
            {latestExperimentRun?.bootstrap_attempted ? (
              <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                latestExperimentRun?.bootstrap_ok ? 'bg-blue-50 text-blue-700 border-blue-100' : 'bg-amber-50 text-amber-700 border-amber-100'
              }`}>
                Bootstrap {latestExperimentRun?.bootstrap_ok ? 'ok' : 'attempted'}
              </span>
            ) : null}
            {latestExperimentRun?.fallback_attempted ? (
              <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                latestExperimentRun?.fallback_ok ? 'bg-indigo-50 text-indigo-700 border-indigo-100' : 'bg-amber-50 text-amber-700 border-amber-100'
              }`}>
                Fallback {latestExperimentRun?.fallback_ok ? 'ok' : 'attempted'}
              </span>
            ) : null}
            {latestExperimentRecoveryOpen ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-rose-100 text-rose-800 border border-rose-200">
                Recovery open
              </span>
            ) : null}
            {latestExperimentFailedCount > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-100">
                Failed cmds {latestExperimentFailedCount}
              </span>
            ) : null}
            {latestExperimentVerificationCount > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-violet-50 text-violet-700 border border-violet-100">
                Verify cmds {latestExperimentVerificationCount}
              </span>
            ) : null}
            {latestExperimentSummary.sourceName ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-100">
                Repo {latestExperimentSummary.sourceName}
              </span>
            ) : null}
            {operatorInterventionSummary.latestLabel ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 text-amber-800 border border-amber-200">
                Last {operatorInterventionSummary.latestLabel}
              </span>
            ) : null}
            {operatorInterventionSummary.latestOutcome ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-orange-50 text-orange-700 border border-orange-100">
                Outcome {operatorInterventionSummary.latestOutcome}
              </span>
            ) : null}
          </div>
        )}

        {latestExperimentRecoveryOpen && graphHealthReasons.length > 0 && (
          <p className="text-[11px] text-rose-700 mb-2 truncate">
            <span className="font-medium">Reason:</span> {graphHealthReasons[0]}
          </p>
        )}
        {latestExperimentRecoveryOpen && graphRecommendedActions.length > 0 && (
          <p className="text-[11px] text-amber-700 mb-2 truncate">
            <span className="font-medium">Next:</span> {graphRecommendedActions[0]}
          </p>
        )}
        {latestExperimentRecoveryOpen && (
          <div className="flex flex-wrap items-center gap-2 mt-1 mb-2">
            <Button
              size="sm"
              variant="primary"
              onClick={(e) => {
                e.stopPropagation();
                onGoToQueue();
              }}
            >
              Open in Checkpoint Queue
            </Button>
            {latestExperimentFailedCount > 0 && latestExperimentRun?.failed_commands?.[0] ? (
              <Button
                size="sm"
                variant="ghost"
                onClick={(e) => {
                  e.stopPropagation();
                  copyText(String(latestExperimentRun.failed_commands?.[0] || ''), 'Failed command');
                }}
              >
                Copy failed command
              </Button>
            ) : null}
          </div>
        )}

        {launchMode === 'quick_start_claude_backend' && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-100">
              Quick Start Claude Backend
            </span>
          </div>
        )}
        {launchMode === 'quick_start_domain_research' && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-cyan-50 text-cyan-700 border border-cyan-100">
              Quick Start Domain Research
            </span>
            {cardDomainIdeaCount > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-100">
                Ideas {cardDomainIdeaCount}
              </span>
            ) : null}
            {cardDomainNoteCount > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-100 text-gray-700 border border-gray-200">
                Notes {cardDomainNoteCount}
              </span>
            ) : null}
          </div>
        )}
        {launchMode === 'quick_start_repo_bug_triage' && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-100">
              Quick Start Repo Bug Triage
            </span>
            {cardWorkspace?.created ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-sky-50 text-sky-700 border border-sky-100">
                Workspace {Number(cardWorkspace.file_count || 0)}
              </span>
            ) : null}
            {cardVerificationCommands.length > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-violet-50 text-violet-700 border border-violet-100">
                Planned verify {cardVerificationCommands.length}
              </span>
            ) : null}
            {cardExecutionPlan.length > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-100 text-gray-700 border border-gray-200">
                Plan {cardExecutionPlan.length} steps
              </span>
            ) : null}
            {cardDetectedStack.length > 0 ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-100">
                Stack {cardDetectedStack.slice(0, 2).join(', ')}
              </span>
            ) : null}
          </div>
        )}
        {['quick_start_bug_triage_swarm', 'quick_start_build_break_swarm', 'quick_start_frontend_regression_swarm'].includes(launchMode) && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <span
              className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                launchMode === 'quick_start_build_break_swarm'
                  ? 'bg-amber-50 text-amber-700 border-amber-100'
                  : launchMode === 'quick_start_frontend_regression_swarm'
                    ? 'bg-cyan-50 text-cyan-700 border-cyan-100'
                    : 'bg-rose-50 text-rose-700 border-rose-100'
              }`}
            >
              {launchMode === 'quick_start_build_break_swarm'
                ? 'Quick Start Build Break Swarm'
                : launchMode === 'quick_start_frontend_regression_swarm'
                  ? 'Quick Start Frontend Regression Swarm'
                  : 'Quick Start Bug Triage Swarm'}
            </span>
            {swarmSummary?.winning_role ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-orange-50 text-orange-700 border border-orange-100">
                Winner {String(swarmSummary.winning_role)}
              </span>
            ) : null}
            {swarmSummary?.repair_chain_job_id ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-100">
                Repair handoff
              </span>
            ) : null}
            {String(swarmSummary?.review_state || '').trim() === 'tie_break_running' ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-100">
                Tie-break running
              </span>
            ) : null}
            {['needs_review', 'insufficient_swarm_consensus', 'consensus_failed'].includes(String(swarmSummary?.review_state || '').trim()) ? (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-200 text-gray-700 border border-gray-200">
                Needs review
              </span>
            ) : null}
          </div>
        )}
        {launchMode === 'quick_start_role_workflow' && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-teal-50 text-teal-700 border border-teal-100">
              Quick Start Role Workflow
            </span>
          </div>
        )}
        {!launchMode && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-gray-50 text-gray-700 border border-gray-200 hover:bg-gray-100"
              title="Filter jobs with no launch mode (manual/legacy)"
              onClick={(e) => {
                e.stopPropagation();
                onClearLaunchModeFilter();
              }}
            >
              No launch mode
            </button>
          </div>
        )}
        {(relaunchFromJobId || relaunchChildrenCount > 0) && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            {relaunchFromJobId && (
              <button
                type="button"
                className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-100 hover:bg-indigo-100"
                title={`Open relaunch parent job ${relaunchFromJobId}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onOpenRunById(relaunchFromJobId);
                }}
              >
                From {relaunchFromJobId.slice(0, 8)}
              </button>
            )}
            {relaunchChildrenCount > 0 && (
              <button
                type="button"
                className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-cyan-50 text-cyan-700 border border-cyan-100 hover:bg-cyan-100"
                title="Filter jobs relaunched from this job"
                onClick={(e) => {
                  e.stopPropagation();
                  onShowRelaunchChildren(job);
                }}
              >
                Relaunched x{relaunchChildrenCount}
              </button>
            )}
          </div>
        )}

        {isPinnedDeepLink && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-primary-50 text-primary-700 border border-primary-100 hover:bg-primary-100"
              onClick={(e) => {
                e.stopPropagation();
                onClearDeepLink();
              }}
              title="Unpin deep-linked job"
            >
              Deep-linked (unpin)
            </button>
          </div>
        )}

        {(approvalCheckpoint || contractEnabled) && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            {approvalCheckpoint && job.status === 'paused' && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-100">
                Awaiting approval
              </span>
            )}
            {contractEnabled && (
              <span
                className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                  contractSatisfied
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-100'
                    : 'bg-amber-50 text-amber-700 border-amber-100'
                }`}
              >
                {contractSatisfied ? 'Contract satisfied' : `Contract missing ${Math.max(0, contractMissingCount)}`}
              </span>
            )}
          </div>
        )}

        {graphHealthStatus && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${graphHealthBadgeClass}`}>
              Graph {graphHealthStatus}
            </span>
            <span className="text-[11px] text-gray-500">Severity {graphHealthSeverity}</span>
          </div>
        )}

        {/* Swarm quick chips */}
        {hasSwarm && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-100 hover:bg-indigo-100"
              title="Filter to swarm jobs"
              onClick={(e) => {
                e.stopPropagation();
                onNarrowToSwarm();
              }}
            >
              Swarm
            </button>
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] bg-emerald-50 text-emerald-700 border border-emerald-100 hover:bg-emerald-100"
              title="Sort by consensus and set minimum consensus threshold"
              onClick={(e) => {
                e.stopPropagation();
                onNarrowToSwarm({ sortBy: 'swarm_consensus_desc', minConsensus: Math.max(1, consensusCount) });
              }}
            >
              Consensus {consensusCount}
            </button>
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] bg-amber-50 text-amber-700 border border-amber-100 hover:bg-amber-100"
              title="Sort by conflicts in swarm jobs"
              onClick={(e) => {
                e.stopPropagation();
                onNarrowToSwarm({ sortBy: 'swarm_conflicts_desc' });
              }}
            >
              Conflicts {conflictCount}
            </button>
            {confidenceOverall > 0 && (
              <button
                type="button"
                className="px-2 py-0.5 rounded-full text-[11px] bg-sky-50 text-sky-700 border border-sky-100 hover:bg-sky-100"
                title="Sort by swarm confidence"
                onClick={(e) => {
                  e.stopPropagation();
                  onNarrowToSwarm({ sortBy: 'swarm_confidence_desc' });
                }}
              >
                Confidence {(confidenceOverall * 100).toFixed(0)}%
              </button>
            )}
          </div>
        )}

        {/* Chain indicator */}
        {(job.parent_job_id || job.chain_config) && (
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-100">
            <GitBranch className="w-3 h-3 text-purple-500" />
            <span className="text-xs text-purple-600">
              {job.parent_job_id ? `Step ${job.chain_depth + 1} in chain` : 'Chain root'}
              {job.chain_triggered && ' • Children triggered'}
            </span>
            <button
              className="ml-auto text-xs text-purple-600 hover:text-purple-800 flex items-center gap-1"
              onClick={(e) => {
                e.stopPropagation();
                onViewChainStatus(job.id);
              }}
            >
              <Link2 className="w-3 h-3" />
              View Chain
            </button>
          </div>
        )}

        {cardMemoryExtraction?.status && (
          <div className="flex flex-wrap items-center gap-1.5 mt-2 pt-2 border-t border-gray-100">
            <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-violet-50 text-violet-700 border border-violet-100">
              Memory {String(cardMemoryExtraction.status)}
              {cardMemoryExtraction?.created_count !== undefined
                ? ` (${Number(cardMemoryExtraction.created_count || 0)})`
                : ''}
            </span>
            {cardMemoryExtraction?.skipped_duplicates !== undefined && (
              <span className="px-2 py-0.5 rounded-full text-[11px] bg-indigo-50 text-indigo-700 border border-indigo-100">
                Dedup {Number(cardMemoryExtraction.skipped_duplicates || 0)}
              </span>
            )}
            {cardMemoryExtraction?.parsed_count !== undefined && (
              <span className="px-2 py-0.5 rounded-full text-[11px] bg-gray-100 text-gray-700 border border-gray-200">
                Parsed {Number(cardMemoryExtraction.parsed_count || 0)}
              </span>
            )}
            {cardMemoryExtraction?.is_relaunch_chain && (
              <span className="px-2 py-0.5 rounded-full text-[11px] bg-cyan-50 text-cyan-700 border border-cyan-100">
                Relaunch dedup
              </span>
            )}
          </div>
        )}

        {/* Stats row */}
        <div className="flex items-center gap-4 text-xs text-gray-500 mt-2">
          <span className="flex items-center gap-1">
            <RefreshCw className="w-3 h-3" />
            {job.iteration}/{job.max_iterations}
          </span>
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3" />
            {job.tool_calls_used}/{job.max_tool_calls}
          </span>
          <span className="flex items-center gap-1">
            <MessageSquare className="w-3 h-3" />
            {job.llm_calls_used}/{job.max_llm_calls}
          </span>
        </div>
      </div>
    );
};

export default JobCard;
