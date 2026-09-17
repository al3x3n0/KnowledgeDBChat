/**
 * The opportunity surface: everything a domain research profile and a research
 * portfolio share.
 *
 * These are different nouns -- a profile is a standing line of enquiry, a
 * portfolio is a programme -- but the controls under them are the same:
 * opportunities you launch, suppress or relaunch, follow-up reviews, the
 * scientific sandbox profiles a validation runs in, and the explainability panel
 * behind each decision. That machinery was 1,900 lines on the Runs page,
 * reachable from exactly two tabs and nothing else.
 *
 * Both tabs are destinations of their own now, so the machinery lives here
 * rather than being copied into each. What the caller supplies is the handful
 * of things that genuinely differ per page: where a link to Runs points, which
 * repositories exist, and where to go when a research pack is created.
 *
 * See also [autonomyShared] for the presentational half -- the metric grid and
 * review lists both surfaces render.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { useMutation, useQuery } from 'react-query';
import type { QueryClient } from 'react-query';
import type { NavigateFunction } from 'react-router-dom';

import Button from '../common/Button';
import { apiClient } from '../../services/api';
import { humanizeDecisionTraceValue } from '../../utils/agentJobDetail';
import { invalidateAgentRunQueries } from '../../utils/agentRunQueries';
import {
  DEFAULT_VALIDATION_POLICY,
  DOMAIN_TRACK_OPTIONS,
  splitUniqueLines,
} from '../../pages/autonomousAgentQuickStarts';
import { ThumbsDown, ThumbsUp } from 'lucide-react';
import type {
  DomainResearchProfile,
  ResearchOpportunity,
  ScientificSandboxProfile,
  ScientificSandboxProfileCreate,
  ScientificSandboxProfileUpdate,
  ScientificValidationRunSummary,
} from '../../types';
import {
  AUTONOMY_FOCUS_ROW_CLASS,
  canRelaunchOpportunityRow,
} from './autonomyShared';

const scientificResearchPackBlueprint = (repoSourceIds: string[]) => {
  const sourceScope = repoSourceIds.length > 0 ? 'kb_plus_arxiv_plus_repo' : 'kb_plus_arxiv';
  return {
    sourceScope,
    compiler: {
      title: 'Compiler Research Pack',
      domain: 'Compiler optimization and code generation',
      objective: 'Identify evidence-backed compiler opportunities, regressions, and validation experiments for the customer codebase.',
      track_type: 'compiler' as const,
      monitor_queries: [
        'llvm optimization pass regression',
        'mlir codegen scheduling',
        'auto-vectorization blocker benchmark',
      ],
      benchmark_queries: [
        'compile time regression',
        'vectorization benchmark',
        'codegen hotspot',
      ],
    },
    microarchitecture: {
      title: 'Microarchitecture Research Pack',
      domain: 'CPU microarchitecture performance and bottlenecks',
      objective: 'Surface testable microarchitecture opportunities tied to cache behavior, branch behavior, SIMD usage, and benchmark regressions.',
      track_type: 'microarchitecture' as const,
      monitor_queries: [
        'cache miss bottleneck benchmark',
        'branch predictor workload analysis',
        'simd throughput regression',
      ],
      benchmark_queries: [
        'ipc stall benchmark',
        'branch miss benchmark',
        'memory bandwidth benchmark',
      ],
    },
    portfolio: {
      title: 'Scientific Research Fleet',
      objective: 'Continuously rank novel and testable compiler and microarchitecture ideas, auto-create validation plans, and auto-launch bounded deep dives within budget.',
    },
  };
};

const humanizeScientificValidationReason = (value?: string | null) =>
  String(value || '').trim().replaceAll('_', ' ') || 'unknown reason';

const scientificValidationStatusClasses = (status?: string | null) => {
  const normalized = String(status || '').trim().toLowerCase();
  if (normalized === 'blocked' || normalized === 'failed') return 'bg-rose-100 text-rose-700';
  if (normalized === 'running' || normalized === 'paused' || normalized === 'provisioning' || normalized === 'queued') return 'bg-amber-100 text-amber-800';
  if (normalized === 'succeeded' || normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  return 'bg-gray-200 text-gray-700';
};

const synthesisStatusClasses = (status?: string | null) => {
  const normalized = String(status || '').trim().toLowerCase();
  if (normalized === 'completed') return 'bg-emerald-100 text-emerald-700';
  if (normalized === 'failed' || normalized === 'cancelled') return 'bg-rose-100 text-rose-700';
  if (normalized) return 'bg-amber-100 text-amber-800';
  return 'bg-gray-200 text-gray-700';
};

const renderOpportunityReprioritizationMeta = (row: Record<string, any>) => {
  const reprioritizedAt = String(row.reprioritized_at || '').trim();
  if (!reprioritizedAt) return null;
  const confidenceDelta = formatOpportunityDelta(row.confidence, row.prior_confidence);
  const readinessDelta = formatOpportunityDelta(row.readiness, row.prior_readiness);
  const followUpReviewStatus = String(row.follow_up_review_status || '').trim();
  const childJobIds = Array.isArray(row.child_job_ids)
    ? row.child_job_ids.filter((value: unknown) => String(value || '').trim())
    : [];
  const sourceRunIds = Array.isArray(row.reprioritization_source_run_ids)
    ? row.reprioritization_source_run_ids.filter((value: unknown) => String(value || '').trim())
    : [];
  return (
    <div className="mt-2 rounded border border-emerald-200 bg-emerald-50 p-2 text-[11px] text-emerald-800">
      <div className="font-medium">Reprioritized from experiment evidence</div>
      <div className="mt-1">
        {new Date(reprioritizedAt).toLocaleString()}
        {row.reprioritization_reason ? ` · ${String(row.reprioritization_reason)}` : ''}
      </div>
      <div className="mt-1">
        Confidence {confidenceDelta ?? 'n/a'}
        {' '}· Readiness {readinessDelta ?? 'n/a'}
        {' '}· Autonomy {humanizeDecisionTraceValue(String(row.autonomy_state || 'eligible'))}
      </div>
      {followUpReviewStatus ? (
        <div className="mt-1">
          Follow-up {humanizeDecisionTraceValue(followUpReviewStatus)}
          {childJobIds.length > 0 ? ` · Child job ${childJobIds[0]}` : ''}
        </div>
      ) : null}
      {sourceRunIds.length > 0 ? (
        <div className="mt-1">Source runs {sourceRunIds.slice(0, 3).join(', ')}</div>
      ) : null}
    </div>
  );
};

const resolveOpportunityExplanationHeading = (row: Record<string, any>) => {
  const reviewStatus = String(row.follow_up_review_status || '').trim().toLowerCase();
  const autonomyState = String(row.autonomy_state || '').trim().toLowerCase();
  const stage = String(row.stage || '').trim().toLowerCase();
  if (reviewStatus === 'pending_approval') return 'Why queued';
  if (reviewStatus === 'rejected') return 'Why rejected';
  if (reviewStatus === 'manual_recommendation') return 'Why manual';
  if (autonomyState === 'blocked_structural' || stage === 'blocked') return 'Why blocked';
  if (autonomyState === 'cooldown') return 'Why cooling down';
  if (autonomyState === 'completed_waiting_change') return 'Why waiting';
  if (String(row.last_skip_reason_code || '').trim() || String(row.reason_code || '').trim()) return 'Why skipped';
  return 'Why this state';
};

const resolveOpportunityExplanationRows = (row: Record<string, any>) => {
  const supportingEvidence = Array.isArray(row.supporting_evidence)
    ? row.supporting_evidence.map((value: unknown) => String(value || '').trim()).filter(Boolean)
    : [];
  const childJobIds = Array.isArray(row.child_job_ids)
    ? row.child_job_ids.map((value: unknown) => String(value || '').trim()).filter(Boolean)
    : [];
  const sourceRunIds = Array.isArray(row.reprioritization_source_run_ids)
    ? row.reprioritization_source_run_ids.map((value: unknown) => String(value || '').trim()).filter(Boolean)
    : [];
  const reevaluationReviewOutcome = String(row.last_reevaluation_review_outcome || '').trim();
  const reevaluationReviewedAt = String(row.last_reevaluation_reviewed_at || '').trim();
  const reevaluationReviewJobId = String(row.last_reevaluation_review_job_id || '').trim();
  const reevaluationReviewNote = String(row.last_reevaluation_review_note || '').trim();
  const reevaluationReviewSourceNoteId = String(row.last_reevaluation_review_source_note_id || '').trim();
  const reevaluationReviewTargetNoteId = String(row.last_reevaluation_review_target_note_id || '').trim();
  const rows: Array<{ label: string; value: string }> = [];
  const reasonCode = resolveOpportunityReasonCode(row);
  const operatorNote = String(row.follow_up_review_note || row.operator_note || '').trim();
  const evidenceRevision = String(row.evidence_revision || '').trim();
  const nextEligibleAt = String(row.next_eligible_at || '').trim();
  const followUpReviewStatus = String(row.follow_up_review_status || '').trim();
  const autonomyState = String(row.autonomy_state || '').trim();
  const stage = String(row.stage || '').trim();
  const hypothesis = String(row.hypothesis || '').trim();
  const sourceRuns = sourceRunIds.slice(0, 3).join(', ');
  const childJobs = childJobIds.slice(0, 3).join(', ');
  const confidenceDelta = formatOpportunityDelta(row.confidence, row.prior_confidence);
  const readinessDelta = formatOpportunityDelta(row.readiness, row.prior_readiness);
  const followUpOutcomeStatus = String(row.follow_up_outcome_status || '').trim();
  const followUpOutcomeSummary = String(row.follow_up_outcome_summary || '').trim();
  const followUpOutcomeRecordedAt = String(row.follow_up_outcome_recorded_at || '').trim();
  const followUpLastJobId = String(row.follow_up_last_job_id || '').trim();

  if (reasonCode) rows.push({ label: 'Reason', value: humanizeDecisionTraceValue(reasonCode) });
  if (operatorNote) rows.push({ label: 'Note', value: operatorNote });
  if (supportingEvidence.length > 0) rows.push({ label: 'Evidence', value: supportingEvidence.slice(0, 3).join(' · ') });
  else if (hypothesis) rows.push({ label: 'Hypothesis', value: hypothesis });
  if (followUpReviewStatus) rows.push({ label: 'Review', value: humanizeDecisionTraceValue(followUpReviewStatus) });
  if (autonomyState || stage) rows.push({ label: 'State', value: [autonomyState && humanizeDecisionTraceValue(autonomyState), stage && humanizeDecisionTraceValue(stage)].filter(Boolean).join(' · ') });
  if (evidenceRevision) rows.push({ label: 'Evidence rev', value: evidenceRevision });
  if (nextEligibleAt) rows.push({ label: 'Next eligible', value: new Date(nextEligibleAt).toLocaleString() });
  if (sourceRuns) rows.push({ label: 'Source runs', value: sourceRuns });
  if (childJobs) rows.push({ label: 'Child jobs', value: childJobs });
  if (followUpOutcomeStatus) rows.push({ label: 'Outcome', value: humanizeDecisionTraceValue(followUpOutcomeStatus) });
  if (followUpOutcomeRecordedAt) rows.push({ label: 'Outcome at', value: new Date(followUpOutcomeRecordedAt).toLocaleString() });
  if (followUpLastJobId) rows.push({ label: 'Outcome job', value: followUpLastJobId });
  if (followUpOutcomeSummary) rows.push({ label: 'Outcome summary', value: followUpOutcomeSummary });
  if (reevaluationReviewOutcome) rows.push({ label: 'Reevaluation review', value: humanizeDecisionTraceValue(reevaluationReviewOutcome) });
  if (reevaluationReviewedAt) rows.push({ label: 'Reevaluation at', value: new Date(reevaluationReviewedAt).toLocaleString() });
  if (reevaluationReviewJobId) rows.push({ label: 'Reevaluation job', value: reevaluationReviewJobId });
  if (reevaluationReviewSourceNoteId) rows.push({ label: 'Reevaluation source note', value: reevaluationReviewSourceNoteId });
  if (reevaluationReviewTargetNoteId && reevaluationReviewTargetNoteId !== reevaluationReviewSourceNoteId) rows.push({ label: 'Reevaluation saved note', value: reevaluationReviewTargetNoteId });
  if (reevaluationReviewNote) rows.push({ label: 'Reevaluation note', value: reevaluationReviewNote });
  if (confidenceDelta || readinessDelta) rows.push({ label: 'Score delta', value: `Confidence ${confidenceDelta ?? 'n/a'} · Readiness ${readinessDelta ?? 'n/a'}` });
  return rows;
};

const buildScientificSandboxProfileDraft = (profile?: ScientificSandboxProfile | null) => ({
  id: String(profile?.id || ''),
  name: String(profile?.name || ''),
  description: String(profile?.description || ''),
  track_type: String(profile?.track_type || 'generic'),
  backend: String(profile?.backend || 'docker'),
  docker_image: String(profile?.docker_image || 'python:3.11-slim'),
  timeout_seconds: String(profile?.timeout_seconds ?? 900),
  memory_mb: String((profile?.resource_caps as any)?.memory_mb ?? 2048),
  cpus: String((profile?.resource_caps as any)?.cpus ?? 1.5),
  pids_limit: String((profile?.resource_caps as any)?.pids_limit ?? 192),
  allowed_benchmark_families: Array.isArray(profile?.allowed_benchmark_families) ? profile!.allowed_benchmark_families.join('\n') : 'generic_validation',
  allowed_perf_collectors: Array.isArray(profile?.allowed_perf_collectors) ? profile!.allowed_perf_collectors.join('\n') : 'benchmark_output',
  required_capabilities: Array.isArray(profile?.required_capabilities) ? profile!.required_capabilities.join('\n') : 'repo_reconstruction',
  toolchains: Array.isArray(profile?.toolchains) ? profile!.toolchains.join('\n') : 'python\npytest',
  budget_limit_default: String(profile?.budget_limit_default ?? 25),
  enabled: Boolean(profile?.enabled ?? true),
  is_default: Boolean(profile?.is_default ?? false),
});

const formatOpportunityDelta = (nextValue: unknown, previousValue: unknown) => {
  const nextNum = Number(nextValue);
  const prevNum = Number(previousValue);
  if (!Number.isFinite(nextNum) || !Number.isFinite(prevNum)) return null;
  const delta = nextNum - prevNum;
  if (Math.abs(delta) < 0.0001) return '0.00';
  return `${delta > 0 ? '+' : ''}${delta.toFixed(2)}`;
};

const resolveOpportunityReasonCode = (row: Record<string, any>) => (
  String(
    row.last_decision_reason_code
    || row.last_blocked_reason_code
    || row.last_skip_reason_code
    || row.reason_code
    || ''
  ).trim()
);

export interface OpportunitySurfaceDeps {
  navigate: NavigateFunction;
  queryClient: QueryClient;
  isAdmin: boolean;
  codeSources: any[];
  buildAutonomousAgentsUrl: (jobId?: string, extras?: Record<string, string | null | undefined>) => string;
  followUpQueueActionMutation: any;
  /** Creating a research pack lands on the fleet; each page decides how. */
  goToFleet: () => void;
  /**
   * A deep link can name an owner and an opportunity to scroll to and flash.
   * Each page reads that from its own URL -- the hook only needs to know what
   * to focus, not which parameter carried it.
   */
  focusTarget?: { scope: 'domain' | 'fleet'; ownerId: string; opportunityId?: string } | null;
  /** Open the card a deep link points at. */
  expandTarget?: (ownerId: string) => void;
  /**
   * Inline review rows: which row is mid-action, and the note typed into it.
   * These belong to the page rather than the hook because the follow-up
   * mutation needs them, and that mutation is an input here -- owning them in
   * both places would be two states with one name.
   */
  activeFollowUpReviewKey: string;
  setActiveFollowUpReviewKey: React.Dispatch<React.SetStateAction<string>>;
  followUpReviewNoteDrafts: Record<string, string>;
  setFollowUpReviewNoteDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
}

export function useOpportunitySurface({
  navigate,
  queryClient,
  isAdmin,
  codeSources,
  buildAutonomousAgentsUrl,
  followUpQueueActionMutation,
  goToFleet,
  focusTarget,
  expandTarget,
  activeFollowUpReviewKey,
  setActiveFollowUpReviewKey,
  followUpReviewNoteDrafts,
  setFollowUpReviewNoteDrafts,
}: OpportunitySurfaceDeps) {
  const { data: domainProfilesData, isLoading: domainProfilesLoading, refetch: refetchDomainProfiles } = useQuery(
    ['domain-research-profiles'],
    () => apiClient.listDomainResearchProfiles({ limit: 100, offset: 0 }),
    {
      refetchInterval: 15000,
    }
  );

  const { data: scientificSandboxProfilesData } = useQuery(
    ['scientific-sandbox-profiles', isAdmin ? 'include-disabled' : 'enabled-only'],
    () => apiClient.listScientificSandboxProfiles(isAdmin ? { include_disabled: true } : undefined),
    {
      staleTime: 60000,
    }
  );

  const { data: researchPortfoliosData, isLoading: researchPortfoliosLoading, refetch: refetchResearchPortfolios } = useQuery(
    ['research-portfolios'],
    () => apiClient.listResearchPortfolios({ limit: 100, offset: 0 }),
    {
      refetchInterval: 15000,
    }
  );
  const [highlightedAutonomyRowKey, setHighlightedAutonomyRowKey] = useState<string>('');
  const [highlightedAutonomyCardKey, setHighlightedAutonomyCardKey] = useState<string>('');
  const [expandedOpportunityExplanationRows, setExpandedOpportunityExplanationRows] = useState<Record<string, boolean>>({});
  const [bulkFollowUpSelection, setBulkFollowUpSelection] = useState<Record<string, boolean>>({});
  const [bulkFollowUpNotes, setBulkFollowUpNotes] = useState<Record<string, string>>({});
  const [activeBulkFollowUpOwnerKey, setActiveBulkFollowUpOwnerKey] = useState<string>('');
  const [opportunityNoteDraft, setOpportunityNoteDraft] = useState<{
    mode: 'suppress' | 'launch' | 'relaunch';
    surface: 'domain' | 'fleet';
    ownerId: string;
    opportunityId: string;
    value: string;
  } | null>(null);
  const [showDisabledSandboxProfiles, setShowDisabledSandboxProfiles] = useState(false);
  const [editingScientificSandboxProfileId, setEditingScientificSandboxProfileId] = useState('');
  const [sandboxProfileDraft, setSandboxProfileDraft] = useState(() => buildScientificSandboxProfileDraft());
  const autonomyTargetRowRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const autonomyTargetCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const buildResearchNoteExperimentUrl = useCallback(
    (noteId?: string | null, extras?: Record<string, string | null | undefined>) => {
      const params = new URLSearchParams();
      const normalizedNoteId = String(noteId || '').trim();
      if (normalizedNoteId) {
        params.set('note', normalizedNoteId);
      }
      Object.entries(extras || {}).forEach(([key, value]) => {
        const text = String(value || '').trim();
        if (text) {
          params.set(key, text);
        }
      });
      const qs = params.toString();
      return `/research-notes${qs ? `?${qs}` : ''}`;
    },
    []
  );
  const buildAutonomyCardKey = useCallback((scope: 'domain' | 'fleet', ownerId: string) => (
    `${scope}:${String(ownerId || '').trim()}`
  ), []);
  const buildAutonomyOpportunityRowKey = useCallback((scope: 'domain' | 'fleet', ownerId: string, opportunityId: string) => (
    `${scope}:${String(ownerId || '').trim()}:opportunity:${String(opportunityId || '').trim()}`
  ), []);
  const buildAutonomyReviewRowKey = useCallback((
    scope: 'domain' | 'fleet',
    ownerId: string,
    reviewKind: 'pending' | 'manual' | 'suppressed',
    opportunityId: string,
  ) => (
    `${scope}:${String(ownerId || '').trim()}:review:${reviewKind}:${String(opportunityId || '').trim()}`
  ), []);
  const registerAutonomyCardRef = useCallback((key: string) => (node: HTMLDivElement | null) => {
    if (!key) return;
    autonomyTargetCardRefs.current[key] = node;
  }, []);
  const registerAutonomyRowRef = useCallback((key: string) => (node: HTMLDivElement | null) => {
    if (!key) return;
    autonomyTargetRowRefs.current[key] = node;
  }, []);
  const renderAutonomySummaryRow = useCallback((
    scope: 'domain' | 'fleet',
    ownerId: string,
    reviewKind: 'pending' | 'manual' | 'suppressed',
    row: Record<string, any>,
    idx: number,
    content: React.ReactNode,
  ) => {
    const opportunityId = String(row.opportunity_id || row.canonical_key || idx).trim();
    const rowKey = buildAutonomyReviewRowKey(scope, ownerId, reviewKind, opportunityId);
    return (
      <div
        key={rowKey}
        ref={registerAutonomyRowRef(rowKey)}
        className={`rounded border px-2 py-1 text-gray-600 transition-colors ${highlightedAutonomyRowKey === rowKey ? AUTONOMY_FOCUS_ROW_CLASS : 'border-transparent'}`}
      >
        {content}
      </div>
    );
  }, [buildAutonomyReviewRowKey, highlightedAutonomyRowKey, registerAutonomyRowRef]);
  const resolveOpportunityContextRow = useCallback((
    row: Record<string, any>,
    opportunities: Array<Record<string, any>> | undefined | null,
  ) => {
    const opportunityId = String(row.opportunity_id || '').trim();
    if (!opportunityId || !Array.isArray(opportunities)) return row;
    return opportunities.find((candidate) => String(candidate?.opportunity_id || '').trim() === opportunityId) || row;
  }, []);
  const domainProfileById = useMemo(
    () =>
      Object.fromEntries(
        ((((domainProfilesData as any)?.items || []) as DomainResearchProfile[]).map((profile) => [String(profile.id), profile]))
      ) as Record<string, DomainResearchProfile>,
    [domainProfilesData]
  );
  const createCompilerArtifactMutation = useMutation(
    (data: {
      job_type: 'compiler_regression_explanation' | 'compiler_patch_proposal' | 'compiler_patch_draft';
      title: string;
      document_ids: string[];
      research_note_id?: string;
      experiment_run_ids?: string[];
      primary_run_id?: string;
      comparison_run_id?: string;
      source_id?: string;
      output_format?: string;
      output_style?: string;
    }) => apiClient.createSynthesisJob(data),
    {
      onSuccess: (job: any) => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        toast.success(`Started ${String(job?.job_type || 'compiler artifact').replace(/_/g, ' ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to create compiler artifact');
      },
    }
  );
  const saveCompilerArtifactNoteMutation = useMutation(
    ({ jobId }: { jobId: string }) => apiClient.saveSynthesisJobAsResearchNote(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        toast.success('Compiler artifact note saved');
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Failed to save compiler artifact note');
      },
    }
  );
  const scientificSandboxProfiles = useMemo(
    () => (((scientificSandboxProfilesData as any)?.items || []) as ScientificSandboxProfile[]),
    [scientificSandboxProfilesData]
  );
  const scientificSandboxProfileById = useMemo(
    () =>
      Object.fromEntries(
        scientificSandboxProfiles.map((profile) => [String(profile.id), profile] as const)
      ) as Record<string, ScientificSandboxProfile>,
    [scientificSandboxProfiles]
  );
  const resolveSandboxProfileId = useCallback(
    (trackType: string) => {
      const normalizedTrackType = String(trackType || 'generic').trim().toLowerCase() || 'generic';
      const exactDefault = scientificSandboxProfiles.find(
        (profile) => String(profile.track_type || '').trim().toLowerCase() === normalizedTrackType && profile.is_default
      );
      if (exactDefault?.id) return String(exactDefault.id);
      const exactEnabled = scientificSandboxProfiles.find(
        (profile) => String(profile.track_type || '').trim().toLowerCase() === normalizedTrackType && profile.enabled
      );
      if (exactEnabled?.id) return String(exactEnabled.id);
      const genericDefault = scientificSandboxProfiles.find(
        (profile) => String(profile.track_type || '').trim().toLowerCase() === 'generic' && profile.is_default
      );
      if (genericDefault?.id) return String(genericDefault.id);
      if (normalizedTrackType === 'compiler') return 'scientific-compiler-sandbox';
      if (normalizedTrackType === 'microarchitecture') return 'scientific-microarchitecture-sandbox';
      return 'scientific-generic-sandbox';
    },
    [scientificSandboxProfiles]
  );
  const visibleScientificSandboxProfiles = useMemo(
    () =>
      scientificSandboxProfiles.filter(
        (profile) => isAdmin || profile.enabled
      ),
    [isAdmin, scientificSandboxProfiles]
  );
  const filteredScientificSandboxProfiles = useMemo(
    () =>
      visibleScientificSandboxProfiles.filter((profile) =>
        isAdmin ? (showDisabledSandboxProfiles ? true : profile.enabled) : profile.enabled
      ),
    [isAdmin, showDisabledSandboxProfiles, visibleScientificSandboxProfiles]
  );
  const editingScientificSandboxProfile = useMemo(
    () =>
      editingScientificSandboxProfileId
        ? scientificSandboxProfileById[String(editingScientificSandboxProfileId)] || null
        : null,
    [editingScientificSandboxProfileId, scientificSandboxProfileById]
  );
  const editingScientificSandboxSystemManaged = Boolean(editingScientificSandboxProfile?.system_managed);
  const resetScientificSandboxDraft = useCallback(() => {
    setEditingScientificSandboxProfileId('');
    setSandboxProfileDraft(buildScientificSandboxProfileDraft());
  }, []);
  const openResearchNote = useCallback((noteId?: string | null) => {
    const normalized = String(noteId || '').trim();
    if (!normalized) return;
    navigate(`/research-notes?note=${encodeURIComponent(normalized)}`);
  }, [navigate]);
  const runCompilerArtifactAction = useCallback((
    run: ScientificValidationRunSummary,
    action: 'create_regression_explanation' | 'create_patch_proposal' | 'create_patch_draft',
    ownerProfile?: DomainResearchProfile | null,
  ) => {
    const artifactSummary = (run.compiler_artifact_summary && typeof run.compiler_artifact_summary === 'object')
      ? run.compiler_artifact_summary
      : null;
    if (!artifactSummary) {
      toast.error('Compiler artifact context is unavailable for this validation run');
      return;
    }
    if (action === 'create_regression_explanation') {
      const experimentRunIds = Array.isArray(artifactSummary.source_run_ids) ? artifactSummary.source_run_ids.filter(Boolean) : [];
      const primaryRunId = String(artifactSummary.primary_run_id || '').trim();
      const comparisonRunId = String(artifactSummary.comparison_run_id || '').trim();
      if (experimentRunIds.length !== 2 || !primaryRunId || !comparisonRunId) {
        toast.error('Explanation generation requires two compared validation runs');
        return;
      }
      createCompilerArtifactMutation.mutate({
        job_type: 'compiler_regression_explanation',
        title: `${String(run.name || 'Compiler Validation').trim()} Explanation`,
        document_ids: [],
        experiment_run_ids: experimentRunIds,
        primary_run_id: primaryRunId,
        comparison_run_id: comparisonRunId,
        output_format: 'markdown',
        output_style: 'technical',
      });
      return;
    }
    if (action === 'create_patch_proposal') {
      const noteId = String(artifactSummary.explanation_note_id || '').trim();
      if (!noteId) {
        toast.error('Patch proposal generation requires an explanation note');
        return;
      }
      createCompilerArtifactMutation.mutate({
        job_type: 'compiler_patch_proposal',
        title: `${String(run.name || 'Compiler Validation').trim()} Patch Proposal`,
        document_ids: [],
        research_note_id: noteId,
        output_format: 'markdown',
        output_style: 'technical',
      });
      return;
    }
    const noteId = String(artifactSummary.proposal_note_id || '').trim();
    const explicitSourceId = String(artifactSummary.source_id || '').trim();
    const profileRepoSourceIds = Array.isArray(ownerProfile?.repo_source_ids) ? ownerProfile?.repo_source_ids.filter(Boolean) : [];
    const sourceId = explicitSourceId || (profileRepoSourceIds.length === 1 ? String(profileRepoSourceIds[0]) : '');
    if (!noteId) {
      toast.error('Patch draft generation requires a proposal note');
      return;
    }
    if (!sourceId) {
      toast.error('Patch draft generation requires one repo source');
      return;
    }
    createCompilerArtifactMutation.mutate({
      job_type: 'compiler_patch_draft',
      title: `${String(run.name || 'Compiler Validation').trim()} Patch Draft`,
      document_ids: [],
      research_note_id: noteId,
      source_id: sourceId,
      output_format: 'markdown',
      output_style: 'technical',
    });
  }, [createCompilerArtifactMutation]);
  const renderScientificValidationRuns = useCallback(
    (
      runs?: Array<Record<string, any>> | null,
      options?: {
        ownerProfile?: DomainResearchProfile | null;
      }
    ) => {
      if (!Array.isArray(runs) || runs.length === 0) {
        return <div className="text-gray-500">No recent validation runs.</div>;
      }
      return (
        <div className="space-y-2">
          {runs.slice(0, 5).map((run) => {
            const status = String(run.status || 'unknown');
            const blockedReason = String(run.blocked_reason_code || '').trim();
            const latestOperatorAction = String(run.latest_operator_action || '').trim();
            const latestOperatorOutcome = String(run.latest_operator_outcome_status || '').trim();
            const sandboxName =
              String(run.sandbox_profile_name || '').trim() ||
              String(scientificSandboxProfileById[String(run.sandbox_profile_id || '')]?.name || '').trim() ||
              String(run.sandbox_profile_id || '').trim() ||
              'default sandbox';
            const typedRun = run as ScientificValidationRunSummary;
            const resolvedOwnerProfile =
              options?.ownerProfile
              || domainProfileById[String(typedRun.domain_research_profile_id || '')]
              || null;
            const artifactSummary = (typedRun.compiler_artifact_summary && typeof typedRun.compiler_artifact_summary === 'object')
              ? typedRun.compiler_artifact_summary
              : null;
            const availableActions = Array.isArray(artifactSummary?.available_actions) ? artifactSummary?.available_actions : [];
            const showPatchDraftAction = availableActions.includes('create_patch_draft')
              && (
                String(artifactSummary?.source_id || '').trim()
                || (Array.isArray(resolvedOwnerProfile?.repo_source_ids) && resolvedOwnerProfile?.repo_source_ids.length === 1)
              );
            return (
              <div key={String(run.id || run.name)} className="border border-gray-100 rounded p-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium text-gray-900">{String(run.name || run.id)}</div>
                  <span className={`text-[11px] px-2 py-0.5 rounded ${scientificValidationStatusClasses(status)}`}>
                    {status}
                  </span>
                </div>
                <div className="mt-1 text-gray-600">
                  {run.recipe_family ? `Recipe ${String(run.recipe_family)}` : 'Scientific validation'}
                  {run.recipe_id ? ` · ${String(run.recipe_id)}` : ''}
                  {sandboxName ? ` · Sandbox ${sandboxName}` : ''}
                </div>
                <div className="mt-1 text-gray-500">
                  Progress {Number(run.progress || 0)}%
                  {run.completed_at ? ` · Completed ${new Date(String(run.completed_at)).toLocaleString()}` : ''}
                  {!run.completed_at && run.created_at ? ` · Created ${new Date(String(run.created_at)).toLocaleString()}` : ''}
                </div>
                {latestOperatorAction ? (
                  <div className="mt-1 text-gray-600">
                    Latest action: {latestOperatorAction}
                    {latestOperatorOutcome ? ` · ${latestOperatorOutcome}` : ''}
                  </div>
                ) : null}
                {Number(run.retry_count || 0) > 0 || run.parent_run_id || run.latest_child_run_id ? (
                  <div className="mt-1 text-gray-500">
                    Retry lineage
                    {Number(run.retry_count || 0) > 0 ? ` · attempt ${Number(run.retry_count || 0)}` : ''}
                    {run.parent_run_id ? ` · parent ${String(run.parent_run_id)}` : ''}
                    {run.latest_child_run_id ? ` · child ${String(run.latest_child_run_id)}` : ''}
                  </div>
                ) : null}
                {blockedReason ? (
                  <div className="mt-1 text-rose-700">Blocked: {humanizeScientificValidationReason(blockedReason)}</div>
                ) : null}
                {artifactSummary ? (
                  <div className="mt-2 rounded border border-indigo-100 bg-indigo-50 p-2">
                    <div className="text-[11px] font-medium text-indigo-900">Compiler artifact handoff</div>
                    <div className="mt-1 flex flex-wrap gap-2 text-[11px]">
                      {artifactSummary.explanation_note_id ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">Explanation ready</span>
                      ) : artifactSummary.explanation_synthesis_job_id ? (
                        <span className={`px-2 py-0.5 rounded ${synthesisStatusClasses(artifactSummary.explanation_synthesis_status)}`}>
                          Explanation {String(artifactSummary.explanation_synthesis_status || 'pending').replace(/_/g, ' ')}
                        </span>
                      ) : null}
                      {artifactSummary.proposal_note_id ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">Proposal ready</span>
                      ) : artifactSummary.proposal_synthesis_job_id ? (
                        <span className={`px-2 py-0.5 rounded ${synthesisStatusClasses(artifactSummary.proposal_synthesis_status)}`}>
                          Proposal {String(artifactSummary.proposal_synthesis_status || 'pending').replace(/_/g, ' ')}
                        </span>
                      ) : null}
                      {artifactSummary.patch_draft_note_id ? (
                        <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-700">Patch draft ready</span>
                      ) : artifactSummary.patch_draft_synthesis_job_id ? (
                        <span className={`px-2 py-0.5 rounded ${synthesisStatusClasses(artifactSummary.patch_draft_synthesis_status)}`}>
                          Patch draft {String(artifactSummary.patch_draft_synthesis_status || 'pending').replace(/_/g, ' ')}
                        </span>
                      ) : null}
                    </div>
                    {artifactSummary.source_run_ids?.length ? (
                      <div className="mt-1 text-[11px] text-indigo-800">
                        Run pair: {artifactSummary.source_run_ids.join(' vs ')}
                      </div>
                    ) : null}
                    {artifactSummary.source_id || artifactSummary.source_name ? (
                      <div className="mt-1 text-[11px] text-indigo-800">
                        Repo source: {String(artifactSummary.source_name || artifactSummary.source_id)}
                      </div>
                    ) : null}
                    <div className="mt-2 flex flex-wrap gap-2">
                      {availableActions.includes('create_regression_explanation') ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={createCompilerArtifactMutation.isLoading}
                          onClick={() => runCompilerArtifactAction(typedRun, 'create_regression_explanation', resolvedOwnerProfile)}
                        >
                          Create explanation
                        </Button>
                      ) : null}
                      {availableActions.includes('create_patch_proposal') ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={createCompilerArtifactMutation.isLoading}
                          onClick={() => runCompilerArtifactAction(typedRun, 'create_patch_proposal', resolvedOwnerProfile)}
                        >
                          Create proposal
                        </Button>
                      ) : null}
                      {showPatchDraftAction ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={createCompilerArtifactMutation.isLoading}
                          onClick={() => runCompilerArtifactAction(typedRun, 'create_patch_draft', resolvedOwnerProfile)}
                        >
                          Create patch draft
                        </Button>
                      ) : null}
                      {artifactSummary.explanation_note_id ? (
                        <Button size="sm" variant="ghost" onClick={() => openResearchNote(artifactSummary.explanation_note_id)}>
                          Open explanation note
                        </Button>
                      ) : null}
                      {artifactSummary.proposal_note_id ? (
                        <Button size="sm" variant="ghost" onClick={() => openResearchNote(artifactSummary.proposal_note_id)}>
                          Open proposal note
                        </Button>
                      ) : null}
                      {artifactSummary.patch_draft_note_id ? (
                        <Button size="sm" variant="ghost" onClick={() => openResearchNote(artifactSummary.patch_draft_note_id)}>
                          Open patch draft note
                        </Button>
                      ) : null}
                      {!artifactSummary.explanation_note_id && artifactSummary.explanation_synthesis_job_id && String(artifactSummary.explanation_synthesis_status || '').trim().toLowerCase() === 'completed' ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={saveCompilerArtifactNoteMutation.isLoading}
                          onClick={() => saveCompilerArtifactNoteMutation.mutate({ jobId: String(artifactSummary.explanation_synthesis_job_id) })}
                        >
                          Save explanation note
                        </Button>
                      ) : null}
                      {!artifactSummary.proposal_note_id && artifactSummary.proposal_synthesis_job_id && String(artifactSummary.proposal_synthesis_status || '').trim().toLowerCase() === 'completed' ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={saveCompilerArtifactNoteMutation.isLoading}
                          onClick={() => saveCompilerArtifactNoteMutation.mutate({ jobId: String(artifactSummary.proposal_synthesis_job_id) })}
                        >
                          Save proposal note
                        </Button>
                      ) : null}
                      {!artifactSummary.patch_draft_note_id && artifactSummary.patch_draft_synthesis_job_id && String(artifactSummary.patch_draft_synthesis_status || '').trim().toLowerCase() === 'completed' ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={saveCompilerArtifactNoteMutation.isLoading}
                          onClick={() => saveCompilerArtifactNoteMutation.mutate({ jobId: String(artifactSummary.patch_draft_synthesis_job_id) })}
                        >
                          Save patch draft note
                        </Button>
                      ) : null}
                      {run.agent_job_id ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => navigate(buildAutonomousAgentsUrl(String(run.agent_job_id)))}
                        >
                          Open source job
                        </Button>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      );
    },
    [buildAutonomousAgentsUrl, createCompilerArtifactMutation.isLoading, domainProfileById, navigate, openResearchNote, runCompilerArtifactAction, saveCompilerArtifactNoteMutation, scientificSandboxProfileById]
  );
  const createScientificSandboxProfileMutation = useMutation(
    (data: ScientificSandboxProfileCreate) => apiClient.createScientificSandboxProfile(data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['scientific-sandbox-profiles']);
        toast.success('Scientific sandbox profile created');
        setEditingScientificSandboxProfileId('');
        setSandboxProfileDraft(buildScientificSandboxProfileDraft());
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create scientific sandbox profile');
      },
    }
  );
  const updateScientificSandboxProfileMutation = useMutation(
    ({ profileId, data }: { profileId: string; data: ScientificSandboxProfileUpdate }) =>
      apiClient.updateScientificSandboxProfile(profileId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['scientific-sandbox-profiles']);
        toast.success('Scientific sandbox profile updated');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update scientific sandbox profile');
      },
    }
  );
  const deleteScientificSandboxProfileMutation = useMutation(
    (profileId: string) => apiClient.deleteScientificSandboxProfile(profileId),
    {
      onSuccess: (_, profileId) => {
        queryClient.invalidateQueries(['scientific-sandbox-profiles']);
        toast.success('Scientific sandbox profile deleted');
        if (String(editingScientificSandboxProfileId) === String(profileId)) {
          setEditingScientificSandboxProfileId('');
          setSandboxProfileDraft(buildScientificSandboxProfileDraft());
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to delete scientific sandbox profile');
      },
    }
  );
  const submitScientificSandboxDraft = useCallback(() => {
    const id = String(sandboxProfileDraft.id || '').trim();
    const name = String(sandboxProfileDraft.name || '').trim();
    if (!id || !name) {
      toast.error('Sandbox profile id and name are required');
      return;
    }
    const payload: ScientificSandboxProfileCreate = {
      id,
      name,
      description: String(sandboxProfileDraft.description || '').trim() || undefined,
      track_type: String(sandboxProfileDraft.track_type || 'generic').trim() || 'generic',
      backend: String(sandboxProfileDraft.backend || 'docker').trim() || 'docker',
      docker_image: String(sandboxProfileDraft.docker_image || '').trim() || undefined,
      timeout_seconds: Number(sandboxProfileDraft.timeout_seconds) > 0 ? Number(sandboxProfileDraft.timeout_seconds) : 900,
      resource_caps: {
        memory_mb: Number(sandboxProfileDraft.memory_mb) > 0 ? Number(sandboxProfileDraft.memory_mb) : 2048,
        cpus: Number(sandboxProfileDraft.cpus) > 0 ? Number(sandboxProfileDraft.cpus) : 1.5,
        pids_limit: Number(sandboxProfileDraft.pids_limit) > 0 ? Number(sandboxProfileDraft.pids_limit) : 192,
      },
      allowed_benchmark_families: splitUniqueLines(String(sandboxProfileDraft.allowed_benchmark_families || ''), 16),
      allowed_perf_collectors: splitUniqueLines(String(sandboxProfileDraft.allowed_perf_collectors || ''), 16),
      required_capabilities: splitUniqueLines(String(sandboxProfileDraft.required_capabilities || ''), 16),
      toolchains: splitUniqueLines(String(sandboxProfileDraft.toolchains || ''), 24),
      budget_limit_default: Number(sandboxProfileDraft.budget_limit_default) > 0 ? Number(sandboxProfileDraft.budget_limit_default) : 25,
      enabled: Boolean(sandboxProfileDraft.enabled),
      is_default: Boolean(sandboxProfileDraft.is_default),
    };
    if (editingScientificSandboxProfileId) {
      const updatePayload: ScientificSandboxProfileUpdate = editingScientificSandboxSystemManaged
        ? {
            name: payload.name,
            description: payload.description,
            enabled: payload.enabled,
            is_default: payload.is_default,
          }
        : payload;
      updateScientificSandboxProfileMutation.mutate({
        profileId: String(editingScientificSandboxProfileId),
        data: updatePayload,
      });
      return;
    }
    createScientificSandboxProfileMutation.mutate(payload);
  }, [
    createScientificSandboxProfileMutation,
    editingScientificSandboxProfileId,
    editingScientificSandboxSystemManaged,
    sandboxProfileDraft,
    updateScientificSandboxProfileMutation,
  ]);
  const createScientificResearchPackMutation = useMutation(
    async () => {
      const repoSourceIds = codeSources
        .map((source) => String((source as any)?.id || '').trim())
        .filter(Boolean);
      const blueprint = scientificResearchPackBlueprint(repoSourceIds);
      const compilerSandboxProfileId = resolveSandboxProfileId('compiler');
      const microarchitectureSandboxProfileId = resolveSandboxProfileId('microarchitecture');
      const compilerProfile = await apiClient.createDomainResearchProfile({
        ...blueprint.compiler,
        source_scope: blueprint.sourceScope,
        repo_source_ids: repoSourceIds,
        sandbox_profile_id: compilerSandboxProfileId,
        automation_profile: 'max_autonomy',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          auto_execute_validation_runs: true,
        },
        interval_minutes: 1440,
        persist_artifacts: true,
        auto_launch_follow_up: true,
        auto_create_experiment_plans: true,
        start_immediately: true,
      });
      const microarchitectureProfile = await apiClient.createDomainResearchProfile({
        ...blueprint.microarchitecture,
        source_scope: blueprint.sourceScope,
        repo_source_ids: repoSourceIds,
        sandbox_profile_id: microarchitectureSandboxProfileId,
        automation_profile: 'max_autonomy',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          auto_execute_validation_runs: true,
        },
        interval_minutes: 1440,
        persist_artifacts: true,
        auto_launch_follow_up: true,
        auto_create_experiment_plans: true,
        start_immediately: true,
      });
      return apiClient.createResearchPortfolio({
        title: blueprint.portfolio.title,
        objective: blueprint.portfolio.objective,
        linked_profile_ids: [compilerProfile.id, microarchitectureProfile.id],
        automation_profile: 'max_autonomy',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          auto_execute_validation_runs: true,
          auto_launch_experiment_runs: true,
          max_auto_follow_up_launches: 4,
          confidence_threshold: 0.68,
          experiment_readiness_threshold: 0.72,
          max_concurrent_validation_runs: 2,
          max_validation_runtime_minutes: 30,
          max_validation_budget_per_run: 50,
          duplicate_window_items: 120,
        },
        sandbox_profile_id: compilerSandboxProfileId,
        start_immediately: true,
      });
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Scientific research pack created');
        goToFleet();
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to create scientific research pack');
      },
    }
  );
  const invalidateOpportunityExperimentQueries = useCallback(
    (ownerResponse: any, opportunityId?: string) => {
      const normalizedOpportunityId = String(opportunityId || '').trim();
      const opportunities = Array.isArray(ownerResponse?.opportunities) ? ownerResponse.opportunities : [];
      const matchedOpportunity = opportunities.find((row: any) => String(row?.opportunity_id || '').trim() === normalizedOpportunityId);
      const noteId = String((matchedOpportunity?.source_note_ids || [ownerResponse?.latest_note_ids?.[0] || ''])[0] || '').trim();
      const planId = String(
        matchedOpportunity?.latest_experiment_plan_id
        || (Array.isArray(matchedOpportunity?.linked_experiment_plan_ids) ? matchedOpportunity.linked_experiment_plan_ids[0] : '')
        || ''
      ).trim();
      if (noteId) {
        queryClient.invalidateQueries(['research-notes']);
        queryClient.invalidateQueries(['experiment-plans', noteId]);
      }
      if (planId) {
        queryClient.invalidateQueries(['experiment-runs', planId]);
      }
    },
    [queryClient]
  );
  const domainOpportunityActionMutation = useMutation(
    ({
      profileId,
      opportunityId,
      action,
      operatorNote,
      startImmediately,
    }: {
      profileId: string;
      opportunityId: string;
      action: 'accept' | 'suppress' | 'reopen' | 'create_plan' | 'launch_validation' | 'materialize_experiment' | 'launch_follow_up' | 'relaunch_follow_up';
      operatorNote?: string;
      startImmediately?: boolean;
    }) => apiClient.actOnDomainResearchOpportunity(profileId, opportunityId, {
      action,
      operator_note: operatorNote,
      start_immediately: startImmediately,
    }),
    {
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['domain-research-profiles']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (variables?.action === 'materialize_experiment' || variables?.action === 'launch_validation') {
          invalidateOpportunityExperimentQueries(response, variables?.opportunityId);
        }
        if (variables?.action === 'launch_follow_up' || variables?.action === 'relaunch_follow_up') {
          invalidateAgentRunQueries(queryClient, [
            'research-inbox',
            'research-inbox-stats',
            'agent-decision-trace',
            'agent-decision-trace-analytics',
            'notifications',
            'notifications-unread-count',
          ]);
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Opportunity action failed');
      },
    }
  );
  const researchPortfolioOpportunityActionMutation = useMutation(
    ({
      portfolioId,
      opportunityId,
      action,
      operatorNote,
      startImmediately,
    }: {
      portfolioId: string;
      opportunityId: string;
      action: 'accept' | 'suppress' | 'reopen' | 'create_plan' | 'launch_validation' | 'materialize_experiment' | 'launch_follow_up' | 'relaunch_follow_up';
      operatorNote?: string;
      startImmediately?: boolean;
    }) => apiClient.actOnResearchPortfolioOpportunity(portfolioId, opportunityId, {
      action,
      operator_note: operatorNote,
      start_immediately: startImmediately,
    }),
    {
      onSuccess: (response, variables) => {
        queryClient.invalidateQueries(['research-portfolios']);
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        if (variables?.action === 'materialize_experiment' || variables?.action === 'launch_validation') {
          invalidateOpportunityExperimentQueries(response, variables?.opportunityId);
        }
        if (variables?.action === 'launch_follow_up' || variables?.action === 'relaunch_follow_up') {
          invalidateAgentRunQueries(queryClient, [
            'research-inbox',
            'research-inbox-stats',
            'agent-decision-trace',
            'agent-decision-trace-analytics',
            'notifications',
            'notifications-unread-count',
          ]);
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Opportunity action failed');
      },
    }
  );
  const beginOpportunityAction = useCallback(
    (mode: 'suppress' | 'launch' | 'relaunch', surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity | Record<string, any>) => {
      setOpportunityNoteDraft({
        mode,
        surface,
        ownerId,
        opportunityId: String(opportunity.opportunity_id || ''),
        value: mode === 'suppress' ? String(opportunity.operator_note || '') : '',
      });
    },
    []
  );
  const beginOpportunitySuppression = useCallback(
    (surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity) => {
      beginOpportunityAction('suppress', surface, ownerId, opportunity);
    },
    [beginOpportunityAction]
  );
  const beginOpportunityRelaunch = useCallback(
    (surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity | Record<string, any>) => {
      beginOpportunityAction('relaunch', surface, ownerId, opportunity);
    },
    [beginOpportunityAction]
  );
  const beginOpportunityLaunch = useCallback(
    (surface: 'domain' | 'fleet', ownerId: string, opportunity: ResearchOpportunity | Record<string, any>) => {
      beginOpportunityAction('launch', surface, ownerId, opportunity);
    },
    [beginOpportunityAction]
  );
  const cancelOpportunityAction = useCallback(() => {
    setOpportunityNoteDraft(null);
  }, []);
  const submitOpportunityAction = useCallback(() => {
    if (!opportunityNoteDraft) return;
    const note = String(opportunityNoteDraft.value || '').trim();
    if (opportunityNoteDraft.mode === 'suppress' && !note) {
      toast.error('Suppression note is required');
      return;
    }
    const action = opportunityNoteDraft.mode === 'suppress'
      ? 'suppress'
      : opportunityNoteDraft.mode === 'launch'
        ? 'launch_follow_up'
        : 'relaunch_follow_up';
    if (opportunityNoteDraft.surface === 'domain') {
      domainOpportunityActionMutation.mutate({
        profileId: opportunityNoteDraft.ownerId,
        opportunityId: opportunityNoteDraft.opportunityId,
        action,
        operatorNote: note || undefined,
      });
    } else {
      researchPortfolioOpportunityActionMutation.mutate({
        portfolioId: opportunityNoteDraft.ownerId,
        opportunityId: opportunityNoteDraft.opportunityId,
        action,
        operatorNote: note || undefined,
      });
    }
    setOpportunityNoteDraft(null);
  }, [domainOpportunityActionMutation, opportunityNoteDraft, researchPortfolioOpportunityActionMutation]);
  const renderOpportunityExplainabilityPanel = useCallback((
    rowKey: string,
    row: Record<string, any>,
    context?: { surface: 'domain' | 'fleet'; ownerId: string } | null,
  ) => {
    const explanationRows = resolveOpportunityExplanationRows(row);
    const reprioritizationMeta = renderOpportunityReprioritizationMeta(row);
    const canRelaunch = Boolean(context?.ownerId && canRelaunchOpportunityRow(row));
    if (explanationRows.length === 0 && !reprioritizationMeta && !canRelaunch) return null;
    const isExpanded = Boolean(expandedOpportunityExplanationRows[rowKey]) || highlightedAutonomyRowKey === rowKey;
    const isRelaunchDraft = opportunityNoteDraft?.mode === 'relaunch'
      && opportunityNoteDraft.surface === context?.surface
      && String(opportunityNoteDraft.ownerId) === String(context?.ownerId || '')
      && String(opportunityNoteDraft.opportunityId) === String(row.opportunity_id || '');
    return (
      <div className="mt-2 rounded border border-gray-200 bg-gray-100 p-2 text-[11px] text-gray-700">
        <div className="flex items-center justify-between gap-2">
          <div className="font-medium text-gray-800">{resolveOpportunityExplanationHeading(row)}</div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setExpandedOpportunityExplanationRows((prev) => ({ ...prev, [rowKey]: !isExpanded }))}
          >
            {isExpanded ? 'Hide details' : 'Show details'}
          </Button>
        </div>
        {isExpanded ? (
          <div className="mt-2 space-y-1">
            {explanationRows.map((item) => (
              <div key={`${rowKey}-${item.label}`}>
                <span className="font-medium text-gray-800">{item.label}:</span>{' '}
                <span>{item.value}</span>
              </div>
            ))}
            {reprioritizationMeta}
            {canRelaunch ? (
              <div className="pt-1">
                {isRelaunchDraft ? (
                  <div className="rounded border border-emerald-200 bg-emerald-50 p-2">
                    <div className="text-[11px] font-medium text-emerald-700">Relaunch note</div>
                    <textarea
                      aria-label={`${context?.surface === 'fleet' ? 'Fleet' : 'Domain'} relaunch note`}
                      className="mt-2 w-full border border-emerald-200 rounded px-2 py-1 text-xs"
                      rows={3}
                      value={opportunityNoteDraft?.value || ''}
                      onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                    />
                    <div className="mt-2 flex gap-2">
                      <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                        Relaunch follow-up
                      </Button>
                      <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      if (!context?.ownerId || !context.surface) return;
                      beginOpportunityRelaunch(context.surface, context.ownerId, row);
                    }}
                  >
                    Relaunch Follow-up
                  </Button>
                )}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    );
  }, [beginOpportunityRelaunch, cancelOpportunityAction, expandedOpportunityExplanationRows, highlightedAutonomyRowKey, opportunityNoteDraft, submitOpportunityAction]);
  const renderManualRecommendationAction = useCallback((
    surface: 'domain' | 'fleet',
    ownerId: string,
    row: Record<string, any>,
  ) => {
    const opportunityId = String(row.opportunity_id || '').trim();
    if (!ownerId || !opportunityId) return null;
    const isRelaunch = canRelaunchOpportunityRow(row);
    const hasChildJobs = Array.isArray(row.child_job_ids) && row.child_job_ids.length > 0;
    if (!isRelaunch && hasChildJobs) return null;
    const isDraftOpen = opportunityNoteDraft?.surface === surface
      && String(opportunityNoteDraft.ownerId) === String(ownerId)
      && String(opportunityNoteDraft.opportunityId) === opportunityId
      && (opportunityNoteDraft.mode === 'launch' || opportunityNoteDraft.mode === 'relaunch');
    const labelPrefix = surface === 'fleet' ? 'Fleet' : 'Domain';
    return (
      <div className="mt-2">
        {isDraftOpen ? (
          <div className="rounded border border-emerald-200 bg-emerald-50 p-2">
            <div className="text-[11px] font-medium text-emerald-700">
              {isRelaunch ? 'Relaunch note' : 'Follow-up note'}
            </div>
            <textarea
              aria-label={`${labelPrefix} ${isRelaunch ? 'relaunch' : 'follow-up'} note`}
              className="mt-2 w-full border border-emerald-200 rounded px-2 py-1 text-xs"
              rows={3}
              value={opportunityNoteDraft?.value || ''}
              onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
            />
            <div className="mt-2 flex gap-2">
              <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                {isRelaunch ? 'Relaunch follow-up' : 'Launch follow-up'}
              </Button>
              <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              if (isRelaunch) {
                beginOpportunityRelaunch(surface, ownerId, row);
                return;
              }
              beginOpportunityLaunch(surface, ownerId, row);
            }}
          >
            {isRelaunch ? 'Relaunch Follow-up' : 'Follow-up'}
          </Button>
        )}
      </div>
    );
  }, [beginOpportunityLaunch, beginOpportunityRelaunch, cancelOpportunityAction, opportunityNoteDraft, submitOpportunityAction]);
  const bulkFollowUpQueueActionMutation = useMutation(
    ({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
      ownerKey,
      refreshTarget,
    }: {
      domain_research_profile_id?: string;
      profile_opportunity_ids?: string[];
      portfolio_id?: string;
      portfolio_opportunity_ids?: string[];
      action: 'approve_launch' | 'reject_launch';
      operator_note?: string;
      ownerKey: string;
      refreshTarget: 'domain' | 'fleet';
    }) => apiClient.bulkActionAgentCheckpointQueueFollowUp({
      domain_research_profile_id,
      profile_opportunity_ids,
      portfolio_id,
      portfolio_opportunity_ids,
      action,
      operator_note,
    }),
    {
      onMutate: (variables) => {
        setActiveBulkFollowUpOwnerKey(String(variables.ownerKey || ''));
      },
      onSuccess: (response, variables) => {
        invalidateAgentRunQueries(queryClient, [
          'research-inbox',
          'research-inbox-stats',
          'research-portfolios',
          'domain-research-profiles',
        ]);
        if (variables.refreshTarget === 'domain') {
          void refetchDomainProfiles();
        } else {
          void refetchResearchPortfolios();
        }
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setBulkFollowUpSelection((prev) => {
            const next = { ...prev };
            successfulIds.forEach((opportunityId) => {
              delete next[`${variables.ownerKey}:${opportunityId}`];
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setBulkFollowUpNotes((prev) => {
            if (!(variables.ownerKey in prev)) return prev;
            const next = { ...prev };
            delete next[variables.ownerKey];
            return next;
          });
          toast.success(
            variables.action === 'approve_launch'
              ? `Launched ${response.applied} follow-up${response.applied === 1 ? '' : 's'}`
              : `Rejected ${response.applied} follow-up${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.profile_opportunity_id || row.portfolio_opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk follow-up action failed');
      },
      onSettled: () => {
        setActiveBulkFollowUpOwnerKey('');
      },
    }
  );
  const bulkManualFollowUpActionMutation = useMutation(
    async ({
      scope,
      ownerId,
      opportunityIds,
      action,
      operator_note,
    }: {
      scope: 'domain' | 'fleet';
      ownerId: string;
      opportunityIds: string[];
      action: 'launch_follow_up' | 'relaunch_follow_up';
      operator_note?: string;
    }) => {
      const results = await Promise.all(
        opportunityIds.map(async (opportunityId) => {
          try {
            if (scope === 'domain') {
              await apiClient.actOnDomainResearchOpportunity(ownerId, opportunityId, {
                action,
                operator_note,
              });
            } else {
              await apiClient.actOnResearchPortfolioOpportunity(ownerId, opportunityId, {
                action,
                operator_note,
              });
            }
            return {
              opportunity_id: opportunityId,
              ok: true,
              action,
            };
          } catch (error: any) {
            return {
              opportunity_id: opportunityId,
              ok: false,
              action,
              error: error?.response?.data?.detail || error?.message || 'failed',
            };
          }
        })
      );
      return {
        requested_count: opportunityIds.length,
        applied: results.filter((row) => row.ok).length,
        failed: results.filter((row) => !row.ok).length,
        results,
      };
    },
    {
      onMutate: (variables) => {
        setActiveBulkFollowUpOwnerKey(buildBulkFollowUpOwnerKey(variables.scope, variables.ownerId));
      },
      onSuccess: (response, variables) => {
        invalidateAgentRunQueries(queryClient, [
          'research-portfolios',
          'domain-research-profiles',
          'research-inbox',
          'research-inbox-stats',
          'agent-decision-trace',
          'agent-decision-trace-analytics',
          'notifications',
          'notifications-unread-count',
        ]);
        if (variables.scope === 'domain') {
          void refetchDomainProfiles();
        } else {
          void refetchResearchPortfolios();
        }
        const ownerKey = buildBulkFollowUpOwnerKey(variables.scope, variables.ownerId);
        const successfulIds = new Set(
          response.results
            .filter((row) => row.ok)
            .map((row) => String(row.opportunity_id || '').trim())
            .filter(Boolean)
        );
        if (successfulIds.size > 0) {
          setBulkFollowUpSelection((prev) => {
            const next = { ...prev };
            successfulIds.forEach((opportunityId) => {
              delete next[`${ownerKey}:${opportunityId}`];
            });
            return next;
          });
        }
        if (response.failed === 0) {
          setBulkFollowUpNotes((prev) => {
            if (!(ownerKey in prev)) return prev;
            const next = { ...prev };
            delete next[ownerKey];
            return next;
          });
          toast.success(
            `${variables.action === 'relaunch_follow_up' ? 'Relaunched' : 'Launched'} ${response.applied} follow-up${response.applied === 1 ? '' : 's'}`
          );
          return;
        }
        const failedLabels = response.results
          .filter((row) => !row.ok)
          .slice(0, 3)
          .map((row) => `${String(row.opportunity_id || '').slice(0, 20)}: ${row.error || 'failed'}`);
        toast.error(`Applied ${response.applied}/${response.requested_count}. ${failedLabels.join(' | ')}`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error?.message || 'Bulk manual follow-up action failed');
      },
      onSettled: () => {
        setActiveBulkFollowUpOwnerKey('');
      },
    }
  );
  const buildInlineFollowUpReviewKey = useCallback(
    (scope: 'domain' | 'fleet', ownerId: string, row: Record<string, any>, idx: number) =>
      `${scope}:${String(ownerId)}:${String(row.opportunity_id || row.canonical_key || idx)}`,
    []
  );
  const buildBulkFollowUpOwnerKey = useCallback(
    (scope: 'domain' | 'fleet', ownerId: string) => `${scope}:${String(ownerId)}`,
    []
  );
  const buildBulkFollowUpSelectionKey = useCallback(
    (scope: 'domain' | 'fleet', ownerId: string, opportunityId: string) =>
      `${buildBulkFollowUpOwnerKey(scope, ownerId)}:${String(opportunityId)}`,
    [buildBulkFollowUpOwnerKey]
  );
  const renderInlineFollowUpApprovalRow = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      row: Record<string, any>,
      idx: number,
    ) => {
      const opportunityId = String(row.opportunity_id || '').trim();
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectionKey = buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId);
      const reviewKey = buildInlineFollowUpReviewKey(scope, ownerId, row, idx);
      const noteValue = followUpReviewNoteDrafts[reviewKey] || '';
      const isSubmitting = followUpQueueActionMutation.isLoading && activeFollowUpReviewKey === reviewKey;
      const isBulkSubmitting = bulkFollowUpQueueActionMutation.isLoading && activeBulkFollowUpOwnerKey === ownerKey;
      const isSelected = Boolean(bulkFollowUpSelection[selectionKey]);
      const missingIdentifiers = !ownerId || !opportunityId;
      const launchStatus = String(row.follow_up_launch_status || row.follow_up_review_status || '').trim();
      const childJobId = String(row.follow_up_job_id || row.child_job_id || '').trim();
      const submitAction = (action: 'approve_launch' | 'reject_launch') => {
        if (missingIdentifiers) {
          toast.error('Missing follow-up approval identifiers');
          return;
        }
        followUpQueueActionMutation.mutate({
          domain_research_profile_id: scope === 'domain' ? ownerId : undefined,
          profile_opportunity_id: scope === 'domain' ? opportunityId : undefined,
          portfolio_id: scope === 'fleet' ? ownerId : undefined,
          portfolio_opportunity_id: scope === 'fleet' ? opportunityId : undefined,
          action,
          operator_note: noteValue.trim() || undefined,
          navigateOnLaunch: false,
          refreshTarget: scope,
          reviewRowKey: reviewKey,
        });
      };
      return renderAutonomySummaryRow(
        scope,
        ownerId,
        'pending',
        row,
        idx,
        <div className="rounded border border-gray-200 p-2">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex items-start gap-2">
              <input
                type="checkbox"
                className="mt-1 rounded border-gray-300"
                aria-label={`Select ${String(row.title || row.canonical_key || 'opportunity')}`}
                checked={isSelected}
                disabled={missingIdentifiers || isSubmitting || isBulkSubmitting}
                onChange={() => {
                  if (missingIdentifiers) return;
                  setBulkFollowUpSelection((prev) => {
                    const next = { ...prev };
                    if (next[selectionKey]) {
                      delete next[selectionKey];
                    } else {
                      next[selectionKey] = true;
                    }
                    return next;
                  });
                }}
              />
              <div className="min-w-0">
                <div className="text-gray-800">
                  {String(row.title || row.canonical_key || 'Opportunity')}
                  {row.reason_code ? ` · ${String(row.reason_code).replaceAll('_', ' ')}` : ''}
                </div>
                {(launchStatus || childJobId) ? (
                  <div className="mt-1 text-xs text-gray-500">
                    {launchStatus ? `State ${launchStatus.replaceAll('_', ' ')}` : null}
                    {launchStatus && childJobId ? ' · ' : null}
                    {childJobId ? `Job ${childJobId}` : null}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Button
                size="sm"
                variant="primary"
                disabled={isSubmitting || isBulkSubmitting || missingIdentifiers}
                onClick={() => submitAction('approve_launch')}
              >
                <ThumbsUp className="w-4 h-4 mr-1" />
                Approve
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={isSubmitting || isBulkSubmitting || missingIdentifiers}
                onClick={() => submitAction('reject_launch')}
              >
                <ThumbsDown className="w-4 h-4 mr-1" />
                Reject
              </Button>
            </div>
          </div>
          <textarea
            aria-label={`Operator note for ${String(row.title || row.canonical_key || 'opportunity')}`}
            className="mt-2 w-full border border-gray-300 rounded px-2 py-1 text-xs"
            rows={2}
            placeholder="Operator note (optional)"
            value={noteValue}
            disabled={isSubmitting || isBulkSubmitting || missingIdentifiers}
            onChange={(e) => setFollowUpReviewNoteDrafts((prev) => ({ ...prev, [reviewKey]: e.target.value }))}
          />
          {missingIdentifiers ? (
            <div className="mt-1 text-[11px] text-rose-700">Missing identifiers for follow-up approval.</div>
          ) : null}
        </div>
      );
    },
    [
      activeFollowUpReviewKey,
      followUpReviewNoteDrafts,
      setFollowUpReviewNoteDrafts,
      activeBulkFollowUpOwnerKey,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      buildInlineFollowUpReviewKey,
      bulkFollowUpQueueActionMutation,
      bulkFollowUpSelection,
      followUpQueueActionMutation,
        renderAutonomySummaryRow,
    ]
  );
  const resolveManualBulkFollowUpAction = useCallback(
    (
      row: Record<string, any>,
      opportunities: Array<Record<string, any>> | undefined,
    ): { opportunity: Record<string, any>; action: 'launch_follow_up' | 'relaunch_follow_up' } | null => {
      const opportunityId = String(row.opportunity_id || '').trim();
      if (!opportunityId) return null;
      const matchingOpportunity = (opportunities || []).find(
        (opportunity) => String(opportunity.opportunity_id || '').trim() === opportunityId
      ) as Record<string, any> | undefined;
      if (!matchingOpportunity) return null;
      if (canRelaunchOpportunityRow(matchingOpportunity)) {
        return { opportunity: matchingOpportunity, action: 'relaunch_follow_up' };
      }
      const hasChildJobs = Array.isArray(matchingOpportunity.child_job_ids) && matchingOpportunity.child_job_ids.length > 0;
      if (hasChildJobs) return null;
      return { opportunity: matchingOpportunity, action: 'launch_follow_up' };
    },
    []
  );
  const renderInlineManualRecommendationRow = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      row: Record<string, any>,
      idx: number,
      opportunities: Array<Record<string, any>> | undefined,
    ) => {
      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
      const bulkAction = resolveManualBulkFollowUpAction(row, opportunities);
      const opportunity = bulkAction?.opportunity;
      const opportunityId = String(opportunity?.opportunity_id || '').trim();
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectionKey = opportunityId ? buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId) : '';
      const isSelected = selectionKey ? Boolean(bulkFollowUpSelection[selectionKey]) : false;
      const isBulkSubmitting = (
        (bulkFollowUpQueueActionMutation.isLoading || bulkManualFollowUpActionMutation.isLoading)
        && activeBulkFollowUpOwnerKey === ownerKey
      );
      return renderAutonomySummaryRow(
        scope,
        ownerId,
        'manual',
        row,
        idx,
        <>
          <div className="flex items-start gap-2">
            {bulkAction && opportunityId ? (
              <input
                type="checkbox"
                className="mt-1 rounded border-gray-300"
                aria-label={`Select ${String(row.title || row.canonical_key || 'manual recommendation')}`}
                checked={isSelected}
                disabled={isBulkSubmitting}
                onChange={() => {
                  setBulkFollowUpSelection((prev) => {
                    const next = { ...prev };
                    if (next[selectionKey]) {
                      delete next[selectionKey];
                    } else {
                      next[selectionKey] = true;
                    }
                    return next;
                  });
                }}
              />
            ) : null}
            <div className="min-w-0">
              <div>{String(row.title || row.canonical_key || 'Manual recommendation')}{row.reason_code ? ` · ${String(row.reason_code)}` : ''}</div>
              {bulkAction ? (
                <div className="mt-1 text-[11px] text-gray-500">
                  Bulk action {bulkAction.action === 'relaunch_follow_up' ? 'relaunch' : 'launch'} ready
                </div>
              ) : null}
            </div>
          </div>
          {opportunity ? renderManualRecommendationAction(scope, ownerId, opportunity) : null}
          {renderOpportunityExplainabilityPanel(
            buildAutonomyReviewRowKey(scope, ownerId, 'manual', String(row.opportunity_id || row.canonical_key || idx)),
            resolvedRow,
            { surface: scope, ownerId: String(ownerId) }
          )}
        </>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      bulkFollowUpQueueActionMutation.isLoading,
      bulkFollowUpSelection,
      bulkManualFollowUpActionMutation.isLoading,
      renderAutonomySummaryRow,
      renderManualRecommendationAction,
      renderOpportunityExplainabilityPanel,
      buildAutonomyReviewRowKey,
      resolveManualBulkFollowUpAction,
      resolveOpportunityContextRow,
    ]
  );
  const resolveSuppressedBulkRelaunchAction = useCallback(
    (
      row: Record<string, any>,
      opportunities: Array<Record<string, any>> | undefined,
    ): { opportunity: Record<string, any>; action: 'relaunch_follow_up' } | null => {
      const opportunityId = String(row.opportunity_id || '').trim();
      if (!opportunityId) return null;
      const matchingOpportunity = (opportunities || []).find(
        (opportunity) => String(opportunity.opportunity_id || '').trim() === opportunityId
      ) as Record<string, any> | undefined;
      if (!matchingOpportunity || !canRelaunchOpportunityRow(matchingOpportunity)) return null;
      return { opportunity: matchingOpportunity, action: 'relaunch_follow_up' };
    },
    []
  );
  const renderInlineSuppressedRelaunchRow = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      row: Record<string, any>,
      idx: number,
      opportunities: Array<Record<string, any>> | undefined,
    ) => {
      const resolvedRow = resolveOpportunityContextRow(row, opportunities as Array<Record<string, any>>);
      const bulkAction = resolveSuppressedBulkRelaunchAction(row, opportunities);
      const matchingOpportunity = bulkAction?.opportunity;
      const canRelaunch = Boolean(bulkAction && matchingOpportunity);
      const opportunityId = String(matchingOpportunity?.opportunity_id || '').trim();
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectionKey = opportunityId ? buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId) : '';
      const isSelected = selectionKey ? Boolean(bulkFollowUpSelection[selectionKey]) : false;
      const isBulkSubmitting = (
        (bulkFollowUpQueueActionMutation.isLoading || bulkManualFollowUpActionMutation.isLoading)
        && activeBulkFollowUpOwnerKey === ownerKey
      );
      const isDraftOpen = canRelaunch
        && opportunityNoteDraft?.mode === 'relaunch'
        && opportunityNoteDraft.surface === scope
        && String(opportunityNoteDraft.ownerId) === String(ownerId)
        && String(opportunityNoteDraft.opportunityId) === String(matchingOpportunity?.opportunity_id || '');
      const labelPrefix = scope === 'fleet' ? 'Fleet' : 'Domain';
      return renderAutonomySummaryRow(
        scope,
        ownerId,
        'suppressed',
        row,
        idx,
        <>
          <div className="flex items-start gap-2">
            {canRelaunch && opportunityId ? (
              <input
                type="checkbox"
                className="mt-1 rounded border-gray-300"
                aria-label={`Select ${String(row.title || row.canonical_key || 'suppressed relaunch')}`}
                checked={isSelected}
                disabled={isBulkSubmitting}
                onChange={() => {
                  setBulkFollowUpSelection((prev) => {
                    const next = { ...prev };
                    if (next[selectionKey]) {
                      delete next[selectionKey];
                    } else {
                      next[selectionKey] = true;
                    }
                    return next;
                  });
                }}
              />
            ) : null}
            <div className="min-w-0">
              <div>{String(row.title || row.canonical_key || 'Suppressed relaunch')} · {String(row.reason_code || 'suppressed').replaceAll('_', ' ')}</div>
              {canRelaunch ? (
                <div className="mt-1 text-[11px] text-gray-500">
                  Bulk action relaunch ready
                </div>
              ) : null}
            </div>
          </div>
          {canRelaunch ? (
            <div className="mt-2">
              {isDraftOpen ? (
                <div className="rounded border border-emerald-200 bg-emerald-50 p-2">
                  <div className="text-[11px] font-medium text-emerald-700">Relaunch note</div>
                  <textarea
                    aria-label={`${labelPrefix} relaunch note`}
                    className="mt-2 w-full border border-emerald-200 rounded px-2 py-1 text-xs"
                    rows={3}
                    value={opportunityNoteDraft?.value || ''}
                    onChange={(e) => setOpportunityNoteDraft((prev) => prev ? { ...prev, value: e.target.value } : prev)}
                  />
                  <div className="mt-2 flex gap-2">
                    <Button size="sm" variant="secondary" onClick={submitOpportunityAction}>
                      Relaunch follow-up
                    </Button>
                    <Button size="sm" variant="ghost" onClick={cancelOpportunityAction}>
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    if (!matchingOpportunity) return;
                    beginOpportunityRelaunch(scope, ownerId, matchingOpportunity);
                  }}
                >
                  Relaunch Follow-up
                </Button>
              )}
            </div>
          ) : null}
          {renderOpportunityExplainabilityPanel(
            buildAutonomyReviewRowKey(scope, ownerId, 'suppressed', String(row.opportunity_id || row.canonical_key || idx)),
            resolvedRow,
            { surface: scope, ownerId: String(ownerId) }
          )}
        </>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      beginOpportunityRelaunch,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      bulkFollowUpQueueActionMutation.isLoading,
      bulkFollowUpSelection,
      bulkManualFollowUpActionMutation.isLoading,
      cancelOpportunityAction,
      opportunityNoteDraft,
      renderAutonomySummaryRow,
      renderOpportunityExplainabilityPanel,
      buildAutonomyReviewRowKey,
      resolveSuppressedBulkRelaunchAction,
      resolveOpportunityContextRow,
      submitOpportunityAction,
    ]
  );
  const renderBulkFollowUpControls = useCallback(
    (
      scope: 'domain' | 'fleet',
      ownerId: string,
      approvalRows: Array<Record<string, any>> | undefined,
      manualRows?: Array<Record<string, any>> | undefined,
      suppressedRows?: Array<Record<string, any>> | undefined,
      opportunities?: Array<Record<string, any>> | undefined,
    ) => {
      const availableApprovalRows = (approvalRows || []).filter((row) => String(row.opportunity_id || '').trim());
      const availableActionRows = [
        ...(manualRows || []).map((row) => ({ kind: 'manual' as const, row })),
        ...(suppressedRows || []).map((row) => ({ kind: 'suppressed' as const, row })),
      ]
        .map((row) => {
          const resolved = row.kind === 'suppressed'
            ? resolveSuppressedBulkRelaunchAction(row.row, opportunities)
            : resolveManualBulkFollowUpAction(row.row, opportunities);
          if (!resolved) return null;
          return {
            row: row.row,
            opportunityId: String(resolved.opportunity.opportunity_id || '').trim(),
            action: resolved.action,
            kind: row.kind,
          };
        })
        .filter(Boolean) as Array<{
          row: Record<string, any>;
          opportunityId: string;
          action: 'launch_follow_up' | 'relaunch_follow_up';
          kind: 'manual' | 'suppressed';
        }>;
      const totalSelectable = availableApprovalRows.length + availableActionRows.length;
      if (totalSelectable === 0) return null;
      const ownerKey = buildBulkFollowUpOwnerKey(scope, ownerId);
      const selectedApprovalIds = availableApprovalRows
        .map((row) => String(row.opportunity_id || '').trim())
        .filter((opportunityId) => Boolean(bulkFollowUpSelection[buildBulkFollowUpSelectionKey(scope, ownerId, opportunityId)]));
      const selectedActionRows = availableActionRows.filter((row) =>
        Boolean(bulkFollowUpSelection[buildBulkFollowUpSelectionKey(scope, ownerId, row.opportunityId)])
      );
      const selectedActionModes = Array.from(new Set(selectedActionRows.map((row) => row.action)));
      const selectedOpportunityIds = selectedApprovalIds.length > 0
        ? selectedApprovalIds
        : selectedActionRows.map((row) => row.opportunityId);
      const noteValue = bulkFollowUpNotes[ownerKey] || '';
      const isSubmitting = (
        (bulkFollowUpQueueActionMutation.isLoading || bulkManualFollowUpActionMutation.isLoading)
        && activeBulkFollowUpOwnerKey === ownerKey
      );
      const selectedCount = selectedApprovalIds.length + selectedActionRows.length;
      const mixedSelection = selectedApprovalIds.length > 0 && selectedActionRows.length > 0;
      const mixedActionModes = selectedActionModes.length > 1;
      const disabledReason = mixedSelection
        ? 'Bulk follow-up actions cannot mix pending approvals and launch/relaunch selections.'
        : mixedActionModes
          ? 'Bulk manual follow-up actions cannot mix launch and relaunch selections.'
          : '';
      const submitBulkAction = (action: 'approve_launch' | 'reject_launch') => {
        if (selectedApprovalIds.length === 0) {
          toast.error('Select at least one follow-up approval');
          return;
        }
        bulkFollowUpQueueActionMutation.mutate({
          domain_research_profile_id: scope === 'domain' ? ownerId : undefined,
          profile_opportunity_ids: scope === 'domain' ? selectedOpportunityIds : undefined,
          portfolio_id: scope === 'fleet' ? ownerId : undefined,
          portfolio_opportunity_ids: scope === 'fleet' ? selectedOpportunityIds : undefined,
          action,
          operator_note: noteValue.trim() || undefined,
          ownerKey,
          refreshTarget: scope,
        });
      };
      const submitBulkManualAction = (action: 'launch_follow_up' | 'relaunch_follow_up') => {
        if (selectedActionRows.length === 0) {
          toast.error(`Select at least one follow-up to ${action === 'relaunch_follow_up' ? 'relaunch' : 'launch'}`);
          return;
        }
        bulkManualFollowUpActionMutation.mutate({
          scope,
          ownerId,
          opportunityIds: selectedActionRows.map((row) => row.opportunityId),
          action,
          operator_note: noteValue.trim() || undefined,
        });
      };
      return (
        <div className="rounded border border-gray-200 bg-gray-100 p-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
            <Button
              size="sm"
              variant="ghost"
              disabled={isSubmitting}
              onClick={() => {
                setBulkFollowUpSelection((prev) => {
                  const next = { ...prev };
                  availableApprovalRows.forEach((row) => {
                    next[buildBulkFollowUpSelectionKey(scope, ownerId, String(row.opportunity_id || '').trim())] = true;
                  });
                  availableActionRows.forEach((row) => {
                    next[buildBulkFollowUpSelectionKey(scope, ownerId, row.opportunityId)] = true;
                  });
                  return next;
                });
              }}
            >
              Select all
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={isSubmitting}
              onClick={() => {
                setBulkFollowUpSelection((prev) => {
                  const next = { ...prev };
                  availableApprovalRows.forEach((row) => {
                    delete next[buildBulkFollowUpSelectionKey(scope, ownerId, String(row.opportunity_id || '').trim())];
                  });
                  availableActionRows.forEach((row) => {
                    delete next[buildBulkFollowUpSelectionKey(scope, ownerId, row.opportunityId)];
                  });
                  return next;
                });
              }}
            >
              Clear
            </Button>
            <span>Selected {selectedCount} of {totalSelectable}</span>
          </div>
          <textarea
            className="w-full border border-gray-300 rounded px-2 py-1 text-xs"
            rows={2}
            placeholder="Shared operator note (optional)"
            value={noteValue}
            disabled={isSubmitting}
            onChange={(e) => setBulkFollowUpNotes((prev) => ({ ...prev, [ownerKey]: e.target.value }))}
          />
          {disabledReason ? (
            <div className="text-[11px] text-amber-700">{disabledReason}</div>
          ) : null}
          <div className="flex flex-wrap gap-2">
            {selectedApprovalIds.length > 0 || (selectedCount === 0 && availableApprovalRows.length > 0) ? (
              <>
                <Button size="sm" variant="primary" disabled={isSubmitting || selectedApprovalIds.length === 0 || Boolean(disabledReason)} onClick={() => submitBulkAction('approve_launch')}>
                  Approve Selected
                </Button>
                <Button size="sm" variant="ghost" disabled={isSubmitting || selectedApprovalIds.length === 0 || Boolean(disabledReason)} onClick={() => submitBulkAction('reject_launch')}>
                  Reject Selected
                </Button>
              </>
            ) : null}
            {selectedActionRows.length > 0 || (selectedCount === 0 && availableActionRows.length > 0) ? (
              <Button
                size="sm"
                variant="secondary"
                disabled={isSubmitting || selectedActionRows.length === 0 || Boolean(disabledReason)}
                onClick={() => submitBulkManualAction(selectedActionModes[0] || 'launch_follow_up')}
              >
                {(selectedActionModes[0] || 'launch_follow_up') === 'relaunch_follow_up' ? 'Relaunch Selected' : 'Launch Selected'}
              </Button>
            ) : null}
          </div>
        </div>
      );
    },
    [
      activeBulkFollowUpOwnerKey,
      buildBulkFollowUpOwnerKey,
      buildBulkFollowUpSelectionKey,
      bulkFollowUpNotes,
      bulkFollowUpQueueActionMutation,
      bulkFollowUpSelection,
      bulkManualFollowUpActionMutation,
      resolveManualBulkFollowUpAction,
      resolveSuppressedBulkRelaunchAction,
    ]
  );
  const renderScientificSandboxManagementPanel = () => (
    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium text-gray-800">Scientific Sandboxes</div>
          <div className="text-xs text-gray-500">Stored runtime profiles for recipe-backed scientific validation.</div>
        </div>
        {isAdmin ? (
          <label className="text-xs text-gray-600 flex items-center gap-2">
            <input
              type="checkbox"
              checked={showDisabledSandboxProfiles}
              onChange={(e) => setShowDisabledSandboxProfiles(e.target.checked)}
            />
            Show disabled
          </label>
        ) : null}
      </div>
      <div className="space-y-2 max-h-64 overflow-auto">
        {filteredScientificSandboxProfiles.map((profile) => (
          <div key={String(profile.id)} className="border border-gray-200 rounded bg-white p-2">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="font-medium text-gray-900">{String(profile.name)}</div>
                  <span className={`text-[11px] px-2 py-0.5 rounded ${profile.enabled ? 'bg-emerald-100 text-emerald-700' : 'bg-gray-200 text-gray-600'}`}>
                    {profile.enabled ? 'enabled' : 'disabled'}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-indigo-100 text-indigo-700">
                    {String(profile.track_type || 'generic')}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-gray-200 text-gray-700">
                    {profile.system_managed ? 'system' : 'custom'}
                  </span>
                  {profile.is_default ? (
                    <span className="text-[11px] px-2 py-0.5 rounded bg-amber-100 text-amber-800">default</span>
                  ) : null}
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  {String(profile.backend || 'docker')} · {String(profile.docker_image || 'no image')}
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  Timeout {Number(profile.timeout_seconds || 0)}s · Budget {Number(profile.budget_limit_default || 0)}
                </div>
              </div>
              {isAdmin ? (
                <div className="flex gap-1 shrink-0">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setEditingScientificSandboxProfileId(String(profile.id));
                      setSandboxProfileDraft(buildScientificSandboxProfileDraft(profile));
                    }}
                  >
                    Edit
                  </Button>
                  {!profile.system_managed ? (
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={deleteScientificSandboxProfileMutation.isLoading}
                      onClick={() => deleteScientificSandboxProfileMutation.mutate(String(profile.id))}
                    >
                      Delete
                    </Button>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        ))}
        {filteredScientificSandboxProfiles.length === 0 ? (
          <div className="text-xs text-gray-500">No sandbox profiles available.</div>
        ) : null}
      </div>
      {isAdmin ? (
        <details className="border border-gray-200 rounded bg-white p-3" open={Boolean(editingScientificSandboxProfileId)}>
          <summary className="cursor-pointer text-xs font-medium text-gray-800">
            {editingScientificSandboxProfileId ? 'Edit sandbox profile' : 'Create custom sandbox profile'}
          </summary>
          <div className="mt-3 space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Profile id"
                value={sandboxProfileDraft.id}
                disabled={Boolean(editingScientificSandboxProfileId)}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, id: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Display name"
                value={sandboxProfileDraft.name}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, name: e.target.value }))}
              />
            </div>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Description"
              value={sandboxProfileDraft.description}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, description: e.target.value }))}
            />
            <div className="grid grid-cols-2 gap-2">
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                value={sandboxProfileDraft.track_type}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, track_type: e.target.value }))}
              >
                {DOMAIN_TRACK_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Docker image"
                value={sandboxProfileDraft.docker_image}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, docker_image: e.target.value }))}
              />
            </div>
            <div className="grid grid-cols-4 gap-2">
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Timeout"
                value={sandboxProfileDraft.timeout_seconds}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, timeout_seconds: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="Memory MB"
                value={sandboxProfileDraft.memory_mb}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, memory_mb: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="CPUs"
                value={sandboxProfileDraft.cpus}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, cpus: e.target.value }))}
              />
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
                placeholder="PIDs"
                value={sandboxProfileDraft.pids_limit}
                disabled={editingScientificSandboxSystemManaged}
                onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, pids_limit: e.target.value }))}
              />
            </div>
            <input
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              placeholder="Budget limit"
              value={sandboxProfileDraft.budget_limit_default}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, budget_limit_default: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Benchmark families, one per line"
              value={sandboxProfileDraft.allowed_benchmark_families}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, allowed_benchmark_families: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Perf collectors, one per line"
              value={sandboxProfileDraft.allowed_perf_collectors}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, allowed_perf_collectors: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Required capabilities, one per line"
              value={sandboxProfileDraft.required_capabilities}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, required_capabilities: e.target.value }))}
            />
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs"
              rows={2}
              placeholder="Toolchains, one per line"
              value={sandboxProfileDraft.toolchains}
              disabled={editingScientificSandboxSystemManaged}
              onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, toolchains: e.target.value }))}
            />
            <div className="flex items-center gap-4 text-xs text-gray-700">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={Boolean(sandboxProfileDraft.enabled)}
                  onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, enabled: e.target.checked }))}
                />
                Enabled
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={Boolean(sandboxProfileDraft.is_default)}
                  onChange={(e) => setSandboxProfileDraft((prev) => ({ ...prev, is_default: e.target.checked }))}
                />
                Default for track
              </label>
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="primary"
                disabled={createScientificSandboxProfileMutation.isLoading || updateScientificSandboxProfileMutation.isLoading}
                onClick={submitScientificSandboxDraft}
              >
                {editingScientificSandboxProfileId ? 'Save Profile' : 'Create Profile'}
              </Button>
              <Button size="sm" variant="ghost" onClick={resetScientificSandboxDraft}>
                Reset
              </Button>
            </div>
          </div>
        </details>
      ) : null}
    </div>
  );

  useEffect(() => {
    const ownerId = focusTarget?.ownerId;
    const scope = focusTarget?.scope;
    if (!ownerId || !scope) return;
    if (!ownerId) return;
    const cardKey = buildAutonomyCardKey(scope, ownerId);
    const cardNode = autonomyTargetCardRefs.current[cardKey];
    const opportunityId = focusTarget?.opportunityId;
    const candidateRowKeys = opportunityId
      ? [
          buildAutonomyOpportunityRowKey(scope, ownerId, opportunityId),
          buildAutonomyReviewRowKey(scope, ownerId, 'pending', opportunityId),
          buildAutonomyReviewRowKey(scope, ownerId, 'manual', opportunityId),
          buildAutonomyReviewRowKey(scope, ownerId, 'suppressed', opportunityId),
        ]
      : [];
    const rowKey = candidateRowKeys.find((key) => autonomyTargetRowRefs.current[key]);
    const rowNode = rowKey ? autonomyTargetRowRefs.current[rowKey] : null;
    const targetNode = rowNode || cardNode;
    if (!targetNode) return;
    // Which collection of expanded ids that is belongs to the page, since a
    // profile and a portfolio each keep their own.
    expandTarget?.(ownerId);
    targetNode.scrollIntoView?.({ block: 'center', behavior: 'auto' });
    if (rowKey) {
      setHighlightedAutonomyRowKey(rowKey);
      setHighlightedAutonomyCardKey('');
    } else {
      setHighlightedAutonomyRowKey('');
      setHighlightedAutonomyCardKey(cardKey);
    }
    const timer = window.setTimeout(() => {
      setHighlightedAutonomyRowKey((current) => (rowKey && current === rowKey ? '' : current));
      setHighlightedAutonomyCardKey((current) => (!rowKey && current === cardKey ? '' : current));
    }, 2200);
    return () => window.clearTimeout(timer);
  }, [
    focusTarget,
    expandTarget,
    buildAutonomyCardKey,
    buildAutonomyOpportunityRowKey,
    buildAutonomyReviewRowKey,
    domainProfilesData,
    researchPortfoliosData,
  ]);

  return {
    domainProfilesData,
    domainProfilesLoading,
    refetchDomainProfiles,
    researchPortfoliosData,
    researchPortfoliosLoading,
    refetchResearchPortfolios,
    scientificSandboxProfilesData,
    highlightedAutonomyRowKey,
    highlightedAutonomyCardKey,
    expandedOpportunityExplanationRows,
    followUpReviewNoteDrafts,
    activeFollowUpReviewKey,
    bulkFollowUpSelection,
    bulkFollowUpNotes,
    activeBulkFollowUpOwnerKey,
    opportunityNoteDraft,
    showDisabledSandboxProfiles,
    editingScientificSandboxProfileId,
    sandboxProfileDraft,
    autonomyTargetRowRefs,
    autonomyTargetCardRefs,
    buildResearchNoteExperimentUrl,
    buildAutonomyCardKey,
    buildAutonomyOpportunityRowKey,
    buildAutonomyReviewRowKey,
    registerAutonomyCardRef,
    registerAutonomyRowRef,
    renderAutonomySummaryRow,
    resolveOpportunityContextRow,
    domainProfileById,
    createCompilerArtifactMutation,
    saveCompilerArtifactNoteMutation,
    scientificSandboxProfiles,
    scientificSandboxProfileById,
    resolveSandboxProfileId,
    visibleScientificSandboxProfiles,
    filteredScientificSandboxProfiles,
    editingScientificSandboxProfile,
    editingScientificSandboxSystemManaged,
    resetScientificSandboxDraft,
    openResearchNote,
    runCompilerArtifactAction,
    renderScientificValidationRuns,
    createScientificSandboxProfileMutation,
    updateScientificSandboxProfileMutation,
    deleteScientificSandboxProfileMutation,
    submitScientificSandboxDraft,
    createScientificResearchPackMutation,
    invalidateOpportunityExperimentQueries,
    domainOpportunityActionMutation,
    researchPortfolioOpportunityActionMutation,
    beginOpportunityAction,
    beginOpportunitySuppression,
    beginOpportunityRelaunch,
    beginOpportunityLaunch,
    cancelOpportunityAction,
    submitOpportunityAction,
    renderOpportunityExplainabilityPanel,
    renderManualRecommendationAction,
    bulkFollowUpQueueActionMutation,
    bulkManualFollowUpActionMutation,
    buildInlineFollowUpReviewKey,
    buildBulkFollowUpOwnerKey,
    buildBulkFollowUpSelectionKey,
    renderInlineFollowUpApprovalRow,
    resolveManualBulkFollowUpAction,
    renderInlineManualRecommendationRow,
    resolveSuppressedBulkRelaunchAction,
    renderInlineSuppressedRelaunchRow,
    renderBulkFollowUpControls,
    renderScientificSandboxManagementPanel,
    setHighlightedAutonomyRowKey,
    setHighlightedAutonomyCardKey,
    setExpandedOpportunityExplanationRows,
    setBulkFollowUpSelection,
    setBulkFollowUpNotes,
    setActiveBulkFollowUpOwnerKey,
    setOpportunityNoteDraft,
    setShowDisabledSandboxProfiles,
    setEditingScientificSandboxProfileId,
    setSandboxProfileDraft,
  };
}
