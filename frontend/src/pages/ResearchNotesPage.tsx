/**
 * Research Notes Page
 *
 * Research-native notes for labs: hypotheses, experiment plans, insights.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { FileText, Search, Trash2, Copy, Download, Plus, Tag, Eye, Quote } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import type { BenchmarkSuite, ExperimentPlan, ExperimentRun, ResearchNote } from '../types';
import Button from '../components/common/Button';
import RecoveryAuditPanel from '../components/agent/RecoveryAuditPanel';
import LoadingSpinner from '../components/common/LoadingSpinner';
import JsonViewer from '../components/common/JsonViewer';
import {
  isExperimentRecoveryOpen,
  summarizeExperimentRecoveryGuidance,
  summarizeOperatorInterventions,
  summarizeExperimentRun,
} from '../utils/experimentRunSummary';

const ResearchNotesPage: React.FC = () => {
  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();

  const [selectedNote, setSelectedNote] = useState<ResearchNote | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [tagFilter, setTagFilter] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [onlyCitationIssues, setOnlyCitationIssues] = useState(false);
  const [citationPolicy, setCitationPolicy] = useState<'sentence' | 'paragraph'>('sentence');
  const [citationUpdateContent, setCitationUpdateContent] = useState(false);
  const [citationStrict, setCitationStrict] = useState(false);
  const [citationUseVector, setCitationUseVector] = useState(true);
  const [citationChunksPerSource, setCitationChunksPerSource] = useState(3);
  const [citationMaxSources, setCitationMaxSources] = useState(10);
  const [citationChunkQuery, setCitationChunkQuery] = useState('');
  const [citationAppendBibliography, setCitationAppendBibliography] = useState(true);
  const [citationOverrideDocsEnabled, setCitationOverrideDocsEnabled] = useState(false);
  const [citationOverrideDocIdsText, setCitationOverrideDocIdsText] = useState('');
  const [expandedEvidenceKeys, setExpandedEvidenceKeys] = useState<Record<string, boolean>>({});
  const [newExperimentRunName, setNewExperimentRunName] = useState('');
  const [experimentSourceId, setExperimentSourceId] = useState('');
  const [experimentCommandsText, setExperimentCommandsText] = useState('python -m pytest -q');
  const [experimentLoopMaxRuns, setExperimentLoopMaxRuns] = useState(3);
  const [experimentGitSourceSearch, setExperimentGitSourceSearch] = useState('');
  const [experimentStopOnOk, setExperimentStopOnOk] = useState(false);
  const [experimentStopMetricRegex, setExperimentStopMetricRegex] = useState('');
  const [experimentStopMetricDirection, setExperimentStopMetricDirection] = useState<'higher_better' | 'lower_better'>(
    'higher_better'
  );
  const [experimentStopMetricWindow, setExperimentStopMetricWindow] = useState(3);
  const [experimentStopMetricMinImprovement, setExperimentStopMetricMinImprovement] = useState(0);
  const [experimentRunActionNotes, setExperimentRunActionNotes] = useState<Record<string, string>>({});
  const [selectedBenchmarkSuiteId, setSelectedBenchmarkSuiteId] = useState('');
  const [selectedBenchmarkCaseIds, setSelectedBenchmarkCaseIds] = useState<string[]>([]);
  const [recentGitSources, setRecentGitSources] = useState<any[]>([]);
  const EXPERIMENT_LOOP_CHAIN_ID = '9e267663-48d6-4a69-9679-984d1cdf6205';

  const EXPERIMENT_SETTINGS_KEY = useMemo(() => {
    const id = selectedNote?.id ? String(selectedNote.id) : '';
    return id ? `research_note_experiment_settings:${id}` : '';
  }, [selectedNote?.id]);
  const RECENT_GIT_SOURCES_KEY = 'recent_git_sources:v1';

  const invalidateLinkedOpportunityQueries = useCallback((run?: ExperimentRun | null) => {
    if (!run) return;
    if (run.domain_research_profile_id) {
      queryClient.invalidateQueries(['domain-research-profiles']);
    }
    if (run.research_portfolio_id) {
      queryClient.invalidateQueries(['research-portfolios']);
    }
    if (run.agent_job_id) {
      queryClient.invalidateQueries(['agent-jobs']);
      queryClient.invalidateQueries(['agent-jobs-stats']);
    }
  }, [queryClient]);

  const extractAutonomousOpportunityTargets = useCallback((note?: ResearchNote | null) => {
    const hypotheses = Array.isArray(note?.structured_payload?.hypotheses) ? note!.structured_payload!.hypotheses! : [];
    const targets: Array<{ sourceKind: 'profile' | 'portfolio'; sourceId: string; opportunityId: string }> = [];
    for (const hypothesis of hypotheses) {
      const origins = [
        hypothesis?.autonomous_origin,
        ...(Array.isArray(hypothesis?.experiment_evidence)
          ? hypothesis.experiment_evidence.map((row) => row?.autonomous_origin)
          : []),
      ];
      for (const origin of origins) {
        const sourceKind = String(origin?.source_kind || '').trim().toLowerCase();
        const sourceId = String(origin?.source_id || '').trim();
        const opportunityId = String(origin?.opportunity_id || '').trim();
        if ((sourceKind !== 'profile' && sourceKind !== 'portfolio') || !sourceId || !opportunityId) continue;
        if (!targets.some((item) => item.sourceKind === sourceKind && item.sourceId === sourceId && item.opportunityId === opportunityId)) {
          targets.push({
            sourceKind: sourceKind as 'profile' | 'portfolio',
            sourceId,
            opportunityId,
          });
        }
      }
    }
    return targets;
  }, []);

  const invalidateAutonomousOriginsForNote = useCallback((note?: ResearchNote | null) => {
    const targets = extractAutonomousOpportunityTargets(note);
    if (targets.some((item) => item.sourceKind === 'profile')) {
      queryClient.invalidateQueries(['domain-research-profiles']);
    }
    if (targets.some((item) => item.sourceKind === 'portfolio')) {
      queryClient.invalidateQueries(['research-portfolios']);
    }
  }, [extractAutonomousOpportunityTargets, queryClient]);

  const summarizeGitSource = useCallback((src: any): { id: string; name: string; source_type: string; detail?: string } | null => {
    if (!src) return null;
    const id = String(src.id || '').trim();
    const name = String(src.name || '').trim();
    const source_type = String(src.source_type || '').trim();
    if (!id || !name || !source_type) return null;
    const cfg = (src.config && typeof src.config === 'object') ? src.config : {};
    let detail = '';
    const display = (cfg.display && typeof cfg.display === 'object') ? cfg.display : null;
    const displayUrl = display ? String((display as any).url || (display as any).repo_url || '').trim() : '';
    if (displayUrl) {
      detail = displayUrl;
    } else if (source_type === 'github') {
      const repos = Array.isArray(cfg.repos) ? cfg.repos : (Array.isArray(cfg.repositories) ? cfg.repositories : []);
      const cleaned = repos.map((r: any) => String(r || '').trim()).filter(Boolean);
      if (cleaned.length) {
        detail = `repos: ${cleaned.slice(0, 2).join(', ')}${cleaned.length > 2 ? ` (+${cleaned.length - 2})` : ''}`;
      }
    } else if (source_type === 'gitlab') {
      const base = String(cfg.gitlab_url || '').trim();
      const projects = Array.isArray(cfg.projects) ? cfg.projects : [];
      const ids = projects.map((p: any) => String(p?.id || '').trim()).filter(Boolean);
      if (ids.length) {
        const projPart = `${ids.slice(0, 2).join(', ')}${ids.length > 2 ? ` (+${ids.length - 2})` : ''}`;
        detail = base ? `projects: ${projPart} @ ${base}` : `projects: ${projPart}`;
      } else if (base) {
        detail = base;
      }
    }
    return { id, name, source_type, detail: detail || undefined };
  }, []);

  const selectGitSource = useCallback(
    (src: any) => {
      const s = summarizeGitSource(src);
      if (!s) return;
      setExperimentSourceId(s.id);
      setRecentGitSources((prev) => {
        const next = (Array.isArray(prev) ? prev : []).filter((x: any) => String(x?.id) !== s.id);
        next.unshift({ ...s, last_used_at: new Date().toISOString() });
        return next.slice(0, 8);
      });
    },
    [summarizeGitSource]
  );

  const urlSelectedNoteId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('note');
  }, [location.search]);

  const urlPlanId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('plan');
  }, [location.search]);

  const urlRunId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('run');
  }, [location.search]);

  const urlAction = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('action');
  }, [location.search]);

  const { data, isLoading, refetch } = useQuery(
    ['research-notes', searchQuery, tagFilter],
    () =>
      apiClient.listResearchNotes({
        q: searchQuery || undefined,
        tag: tagFilter || undefined,
        limit: 50,
        offset: 0,
      }),
    { refetchInterval: 10000 }
  );

  useEffect(() => {
    if (!urlSelectedNoteId) {
      setSelectedNote(null);
      return;
    }

    const match = (data?.items || []).find((n) => n.id === urlSelectedNoteId);
    if (match) setSelectedNote(match);

    let cancelled = false;
    apiClient
      .getResearchNote(urlSelectedNoteId)
      .then((note) => {
        if (!cancelled) setSelectedNote(note);
      })
      .catch(() => {
        // ignore
      });
    return () => {
      cancelled = true;
    };
  }, [urlSelectedNoteId, data?.items]);

  // Load persisted experiment UI settings per note (localStorage).
  useEffect(() => {
    if (!EXPERIMENT_SETTINGS_KEY) return;
    try {
      const raw = window.localStorage.getItem(EXPERIMENT_SETTINGS_KEY);
      if (!raw) return;
      const parsed: any = JSON.parse(raw);
      if (typeof parsed?.source_id === 'string') setExperimentSourceId(parsed.source_id);
      if (typeof parsed?.commands_text === 'string') setExperimentCommandsText(parsed.commands_text);
      if (typeof parsed?.max_runs === 'number') setExperimentLoopMaxRuns(parsed.max_runs);
      if (typeof parsed?.stop_on_ok === 'boolean') setExperimentStopOnOk(parsed.stop_on_ok);
      if (typeof parsed?.stop_metric_regex === 'string') setExperimentStopMetricRegex(parsed.stop_metric_regex);
      if (parsed?.stop_metric_direction === 'higher_better' || parsed?.stop_metric_direction === 'lower_better') {
        setExperimentStopMetricDirection(parsed.stop_metric_direction);
      }
      if (typeof parsed?.stop_metric_window === 'number') setExperimentStopMetricWindow(parsed.stop_metric_window);
      if (typeof parsed?.stop_metric_min_improvement === 'number') setExperimentStopMetricMinImprovement(parsed.stop_metric_min_improvement);
    } catch {
      // ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [EXPERIMENT_SETTINGS_KEY]);

  // Load recent git sources (shared across notes).
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(RECENT_GIT_SOURCES_KEY);
      if (!raw) return;
      const parsed: any = JSON.parse(raw);
      if (Array.isArray(parsed)) setRecentGitSources(parsed.slice(0, 8));
    } catch {
      // ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist experiment UI settings per note (localStorage).
  useEffect(() => {
    if (!EXPERIMENT_SETTINGS_KEY) return;
    const payload = {
      source_id: experimentSourceId,
      commands_text: experimentCommandsText,
      max_runs: experimentLoopMaxRuns,
      stop_on_ok: experimentStopOnOk,
      stop_metric_regex: experimentStopMetricRegex,
      stop_metric_direction: experimentStopMetricDirection,
      stop_metric_window: experimentStopMetricWindow,
      stop_metric_min_improvement: experimentStopMetricMinImprovement,
      updated_at: new Date().toISOString(),
    };
    const t = window.setTimeout(() => {
      try {
        window.localStorage.setItem(EXPERIMENT_SETTINGS_KEY, JSON.stringify(payload));
      } catch {
        // ignore
      }
    }, 350);
    return () => window.clearTimeout(t);
  }, [
    EXPERIMENT_SETTINGS_KEY,
    experimentSourceId,
    experimentCommandsText,
    experimentLoopMaxRuns,
    experimentStopOnOk,
    experimentStopMetricRegex,
    experimentStopMetricDirection,
    experimentStopMetricWindow,
    experimentStopMetricMinImprovement,
  ]);

  // Persist recent git sources (shared across notes).
  useEffect(() => {
    const t = window.setTimeout(() => {
      try {
        window.localStorage.setItem(RECENT_GIT_SOURCES_KEY, JSON.stringify((recentGitSources || []).slice(0, 8)));
      } catch {
        // ignore
      }
    }, 250);
    return () => window.clearTimeout(t);
  }, [RECENT_GIT_SOURCES_KEY, recentGitSources]);

  const deleteMutation = useMutation((noteId: string) => apiClient.deleteResearchNote(noteId), {
    onSuccess: () => {
      toast.success('Note deleted');
      queryClient.invalidateQueries(['research-notes']);
      if (selectedNote) {
        setSelectedNote(null);
        navigate('/research-notes', { replace: true });
      }
    },
    onError: (e: any) => {
      toast.error(e?.message || 'Delete failed');
    },
  });

  const lintRecentMutation = useMutation(
    () => apiClient.lintRecentResearchNotes({ window_hours: 24, max_notes: 200, max_sources: 10, max_uncited_examples: 10 }),
    {
      onSuccess: (res) => {
        toast.success(`Linted: ${res.updated} updated (${res.skipped} skipped, ${res.missing_sources} missing sources)`);
        queryClient.invalidateQueries(['research-notes']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Lint recent failed');
      },
    }
  );

  const parsedOverrideDocIds = useMemo(() => {
    if (!citationOverrideDocsEnabled) return null;
    const raw = citationOverrideDocIdsText
      .split(/[\s,]+/g)
      .map((s) => s.trim())
      .filter(Boolean);
    const uuidRe = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    const out: string[] = [];
    for (const x of raw) {
      if (uuidRe.test(x) && !out.includes(x)) out.push(x);
    }
    return out.length ? out : [];
  }, [citationOverrideDocsEnabled, citationOverrideDocIdsText]);

  const loadSettingsFromLastRun = () => {
    const a: any = selectedNote?.attribution;
    if (!a) return;
    if (a.policy === 'sentence' || a.policy === 'paragraph') setCitationPolicy(a.policy);
    if (typeof a.update_content === 'boolean') setCitationUpdateContent(a.update_content);
    if (typeof a.append_bibliography === 'boolean') setCitationAppendBibliography(a.append_bibliography);
    if (typeof a.strict === 'boolean') setCitationStrict(a.strict);
    if (typeof a.use_vector_snippets === 'boolean') setCitationUseVector(a.use_vector_snippets);
    if (typeof a.chunks_per_source === 'number') setCitationChunksPerSource(a.chunks_per_source);
    if (typeof a.max_sources === 'number') setCitationMaxSources(a.max_sources);
    if (typeof a.chunk_query_used === 'string') setCitationChunkQuery(a.chunk_query_used);
    if (Array.isArray(a.document_ids_used) && a.document_ids_used.length > 0) {
      setCitationOverrideDocsEnabled(true);
      setCitationOverrideDocIdsText(a.document_ids_used.join('\n'));
    }
  };

  const enforceCitationsMutation = useMutation(
    (noteId: string) => {
      if (citationOverrideDocsEnabled && parsedOverrideDocIds && parsedOverrideDocIds.length === 0) {
        return Promise.reject(new Error('Override sources is enabled but no valid UUIDs were provided.'));
      }
      return apiClient.enforceResearchNoteCitations(noteId, {
        policy: citationPolicy,
        update_content: citationUpdateContent,
        append_bibliography: citationAppendBibliography,
        max_sources: citationMaxSources,
        strict: citationStrict,
        use_vector_snippets: citationUseVector,
        chunks_per_source: citationChunksPerSource,
        chunk_query: citationChunkQuery.trim() || undefined,
        document_ids: parsedOverrideDocIds || undefined,
      });
    },
    {
      onSuccess: (note) => {
        toast.success('Citations enforced');
        setSelectedNote(note);
        queryClient.invalidateQueries(['research-notes']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Citation enforcement failed');
      },
    }
  );

  const lintCitationsMutation = useMutation(
    (noteId: string) => {
      if (citationOverrideDocsEnabled && parsedOverrideDocIds && parsedOverrideDocIds.length === 0) {
        return Promise.reject(new Error('Override sources is enabled but no valid UUIDs were provided.'));
      }
      return apiClient.lintResearchNoteCitations(noteId, {
        max_sources: citationMaxSources,
        max_uncited_examples: 10,
        document_ids: parsedOverrideDocIds || undefined,
      });
    },
    {
      onSuccess: (note) => {
        toast.success('Citation lint complete');
        setSelectedNote(note);
        queryClient.invalidateQueries(['research-notes']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Lint failed');
      },
    }
  );

  const quickFixMutation = useMutation(
    async (noteId: string) => {
      if (citationOverrideDocsEnabled && parsedOverrideDocIds && parsedOverrideDocIds.length === 0) {
        throw new Error('Override sources is enabled but no valid UUIDs were provided.');
      }
      const updated = await apiClient.enforceResearchNoteCitations(noteId, {
        policy: 'sentence',
        strict: true,
        update_content: true,
        append_bibliography: true,
        use_vector_snippets: true,
        max_sources: 10,
        chunks_per_source: 3,
        document_ids: parsedOverrideDocIds || undefined,
      });
      // Lint after applying the rewrite, for immediate feedback.
      return apiClient.lintResearchNoteCitations(updated.id, {
        max_sources: 10,
        max_uncited_examples: 10,
        document_ids: parsedOverrideDocIds || undefined,
      });
    },
    {
      onSuccess: (note) => {
        toast.success('Fixed citations');
        setSelectedNote(note);
        queryClient.invalidateQueries(['research-notes']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Fix failed');
      },
    }
  );

  const { data: experimentPlansData, refetch: refetchExperimentPlans } = useQuery(
    ['experiment-plans', selectedNote?.id],
    () => apiClient.listExperimentPlansForNote(selectedNote!.id, 10),
    { enabled: !!selectedNote?.id, staleTime: 5000 }
  );

  const { data: gitSourcesData } = useQuery(
    ['git-sources-search', experimentGitSourceSearch],
    () =>
      apiClient.searchGitDocumentSources({
        q: experimentGitSourceSearch.trim() || undefined,
        limit: 25,
      }),
    { staleTime: 15000 }
  );

  const { data: activeGitSourcesData } = useQuery(['git-sources-active'], () => apiClient.getActiveGitSources(), {
    staleTime: 10000,
    refetchInterval: 15000,
  });

  const { data: benchmarkSuitesData } = useQuery(
    ['benchmark-suites', 'compiler'],
    () => apiClient.listBenchmarkSuites('compiler'),
    { staleTime: 30000 }
  );

  const activeGitById = useMemo(() => {
    const map: Record<string, { pending: boolean; is_syncing: boolean; task_id?: string }> = {};
    const items = (activeGitSourcesData || []) as any[];
    for (const row of items) {
      const src = row?.source;
      const id = String(src?.id || '').trim();
      if (!id) continue;
      map[id] = {
        pending: !!row?.pending,
        is_syncing: !!src?.is_syncing,
        task_id: row?.task_id ? String(row.task_id) : undefined,
      };
    }
    return map;
  }, [activeGitSourcesData]);

  const gitSources = useMemo(() => {
    const items = (gitSourcesData || []) as any[];
    return items.filter(Boolean);
  }, [gitSourcesData]);

  const gitSourcesWithStatus = useMemo(() => {
    return (gitSources || []).map((s: any) => {
      const id = String(s?.id || '').trim();
      const st = id ? activeGitById[id] : undefined;
      if (!st) return s;
      return { ...s, _active: st };
    });
  }, [gitSources, activeGitById]);

  const benchmarkSuites = useMemo(() => {
    return Array.isArray((benchmarkSuitesData as any)?.items) ? (((benchmarkSuitesData as any).items || []) as BenchmarkSuite[]) : [];
  }, [benchmarkSuitesData]);

  const selectedBenchmarkSuite = useMemo(() => {
    return benchmarkSuites.find((suite) => String(suite.id) === selectedBenchmarkSuiteId) || null;
  }, [benchmarkSuites, selectedBenchmarkSuiteId]);

  useEffect(() => {
    if (!selectedBenchmarkSuiteId) {
      setSelectedBenchmarkCaseIds([]);
      return;
    }
    if (!selectedBenchmarkSuite) {
      setSelectedBenchmarkCaseIds([]);
      return;
    }
    setSelectedBenchmarkCaseIds((current) => current.filter((caseId) => selectedBenchmarkSuite.cases.some((row) => String(row.id) === caseId)));
  }, [selectedBenchmarkSuiteId, selectedBenchmarkSuite]);

  const resolvedSelectedGitSource = useMemo(() => {
    const id = experimentSourceId.trim();
    if (!id) return null;
    const fromSearch = (gitSourcesWithStatus || []).find((s: any) => String(s?.id) === id) || null;
    if (fromSearch) return fromSearch;
    const fromRecent = (recentGitSources || []).find((s: any) => String(s?.id) === id) || null;
    return fromRecent;
  }, [experimentSourceId, gitSourcesWithStatus, recentGitSources]);

  const resolvedSelectedGitSourceSummary = useMemo(() => {
    return summarizeGitSource(resolvedSelectedGitSource);
  }, [resolvedSelectedGitSource, summarizeGitSource]);

  const selectedGitActiveStatus = useMemo(() => {
    const id = experimentSourceId.trim();
    return id ? activeGitById[id] || null : null;
  }, [experimentSourceId, activeGitById]);

  const lastAutoRecentIdRef = useRef<string>('');
  useEffect(() => {
    const id = experimentSourceId.trim();
    if (!id || !resolvedSelectedGitSourceSummary) return;
    if (lastAutoRecentIdRef.current === id) return;
    lastAutoRecentIdRef.current = id;
    setRecentGitSources((prev) => {
      const next = (Array.isArray(prev) ? prev : []).filter((x: any) => String(x?.id) !== id);
      next.unshift({ ...resolvedSelectedGitSourceSummary, last_used_at: new Date().toISOString() });
      return next.slice(0, 8);
    });
  }, [experimentSourceId, resolvedSelectedGitSourceSummary]);

  const latestExperimentPlan: ExperimentPlan | null = useMemo(() => {
    const plans = (experimentPlansData as any)?.plans || [];
    return plans.length ? (plans[0] as ExperimentPlan) : null;
  }, [experimentPlansData]);

  const latestPlanExecutionHandoff = useMemo(() => {
    if (!latestExperimentPlan) return null;
    const details = latestExperimentPlan.generator_details || {};
    const plan = latestExperimentPlan.plan || {};
    const provenance = plan.provenance && typeof plan.provenance === 'object' ? plan.provenance : {};
    return {
      execution_handoff_version: 1,
      plan_scope:
        (typeof details.plan_mode === 'string' && details.plan_mode) ||
        (typeof plan.plan_scope === 'string' && plan.plan_scope) ||
        undefined,
      selected_hypothesis_ids: Array.isArray(details.selected_hypothesis_ids)
        ? details.selected_hypothesis_ids
        : (Array.isArray(plan.selected_hypothesis_ids) ? plan.selected_hypothesis_ids : []),
      supporting_sources: Array.isArray(details.supporting_sources)
        ? details.supporting_sources
        : (Array.isArray(plan.supporting_sources) ? plan.supporting_sources : []),
      source_paper_ids: Array.isArray(details.source_paper_ids)
        ? details.source_paper_ids
        : (Array.isArray((provenance as any).source_paper_ids) ? (provenance as any).source_paper_ids : []),
      source_document_ids: Array.isArray(details.source_document_ids)
        ? details.source_document_ids
        : (Array.isArray((provenance as any).source_document_ids) ? (provenance as any).source_document_ids : []),
      benchmark_family:
        (typeof details.benchmark_family === 'string' && details.benchmark_family) ||
        (typeof plan.benchmark_family === 'string' && plan.benchmark_family) ||
        (typeof (provenance as any).benchmark_family === 'string' ? (provenance as any).benchmark_family : undefined),
      benchmark_suite_id:
        (typeof details.benchmark_suite_id === 'string' && details.benchmark_suite_id) ||
        (typeof plan.benchmark_suite_id === 'string' && plan.benchmark_suite_id) ||
        (typeof (provenance as any).benchmark_suite_id === 'string' ? (provenance as any).benchmark_suite_id : undefined),
      benchmark_case_ids: Array.isArray(details.benchmark_case_ids)
        ? details.benchmark_case_ids
        : (Array.isArray(plan.benchmark_case_ids) ? plan.benchmark_case_ids : (Array.isArray((provenance as any).benchmark_case_ids) ? (provenance as any).benchmark_case_ids : [])),
      benchmark_baseline_id:
        (typeof details.benchmark_baseline_id === 'string' && details.benchmark_baseline_id) ||
        (typeof plan.benchmark_baseline_id === 'string' && plan.benchmark_baseline_id) ||
        (typeof (provenance as any).benchmark_baseline_id === 'string' ? (provenance as any).benchmark_baseline_id : undefined),
    };
  }, [latestExperimentPlan]);

  const latestPlanBenchmarkCommands = useMemo(() => {
    const details = latestExperimentPlan?.generator_details || {};
    return Array.isArray(details.benchmark_default_commands) ? details.benchmark_default_commands.filter(Boolean) : [];
  }, [latestExperimentPlan]);

  useEffect(() => {
    if (!latestExperimentPlan || newExperimentRunName.trim()) return;
    const details = latestExperimentPlan.generator_details || {};
    const selectedIds = Array.isArray(details.selected_hypothesis_ids) ? details.selected_hypothesis_ids : [];
    const suggestedName =
      details.plan_mode === 'single_hypothesis'
        ? `${latestExperimentPlan.title} · ${selectedIds[0] || 'run'}`
        : details.plan_mode === 'aggregate_note'
          ? `${latestExperimentPlan.title} · aggregate run`
          : `${latestExperimentPlan.title} · run`;
    setNewExperimentRunName(suggestedName);
  }, [latestExperimentPlan, newExperimentRunName]);

  useEffect(() => {
    if (!latestPlanBenchmarkCommands.length) return;
    if (experimentCommandsText.trim() && experimentCommandsText.trim() !== 'python -m pytest -q') return;
    setExperimentCommandsText(latestPlanBenchmarkCommands.join('\n'));
  }, [latestPlanBenchmarkCommands, experimentCommandsText]);

  const { data: experimentRunsData, refetch: refetchExperimentRuns } = useQuery(
    ['experiment-runs', latestExperimentPlan?.id],
    () => apiClient.listExperimentRuns(latestExperimentPlan!.id),
    { enabled: !!latestExperimentPlan?.id, staleTime: 5000, refetchInterval: 10000 }
  );

  const latestPlanRef = useRef<HTMLDivElement | null>(null);
  const experimentRunRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const registerExperimentRunRef = useCallback((runId: string) => (node: HTMLDivElement | null) => {
    if (!runId) return;
    experimentRunRefs.current[runId] = node;
  }, []);

  const experimentRuns: ExperimentRun[] = useMemo(() => {
    return ((experimentRunsData as any)?.runs || []) as ExperimentRun[];
  }, [experimentRunsData]);

  const isDeepLinkedLatestPlan = useMemo(() => {
    const targetPlanId = String(urlPlanId || '').trim();
    return Boolean(targetPlanId && latestExperimentPlan && String(latestExperimentPlan.id) === targetPlanId);
  }, [latestExperimentPlan, urlPlanId]);

  const deepLinkedRunId = useMemo(() => {
    const targetRunId = String(urlRunId || '').trim();
    if (!targetRunId) return '';
    return experimentRuns.some((run) => String(run.id) === targetRunId) ? targetRunId : '';
  }, [experimentRuns, urlRunId]);

  useEffect(() => {
    if (!urlSelectedNoteId) return;
    const targetRunId = String(urlRunId || '').trim();
    const targetPlanId = String(urlPlanId || '').trim();
    const targetNode =
      (targetRunId ? experimentRunRefs.current[targetRunId] : null)
      || (targetPlanId && latestExperimentPlan && String(latestExperimentPlan.id) === targetPlanId ? latestPlanRef.current : null);
    if (!targetNode) return;
    const timeoutId = window.setTimeout(() => {
      if (typeof targetNode.scrollIntoView === 'function') {
        targetNode.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }, 75);
    return () => window.clearTimeout(timeoutId);
  }, [latestExperimentPlan, urlPlanId, urlRunId, urlSelectedNoteId, experimentRuns]);

  const areRunsComparable = useCallback((primary: ExperimentRun, comparison: ExperimentRun): boolean => {
    if (!primary?.benchmark_family || !comparison?.benchmark_family) return false;
    if (String(primary.benchmark_family) !== String(comparison.benchmark_family)) return false;
    if (String(primary.benchmark_suite_id || '') !== String(comparison.benchmark_suite_id || '')) return false;
    const primaryCases = Array.isArray(primary.benchmark_case_ids) ? primary.benchmark_case_ids.filter(Boolean) : [];
    const comparisonCases = Array.isArray(comparison.benchmark_case_ids) ? comparison.benchmark_case_ids.filter(Boolean) : [];
    if (!primaryCases.length || !comparisonCases.length) return false;
    return primaryCases.some((id) => comparisonCases.includes(id));
  }, []);

  const findComparisonRun = useCallback((primaryRunId: string): ExperimentRun | null => {
    const index = experimentRuns.findIndex((run) => String(run.id) === String(primaryRunId));
    if (index === -1) return null;
    const primary = experimentRuns[index];
    for (let i = index + 1; i < experimentRuns.length; i += 1) {
      const candidate = experimentRuns[i];
      if (areRunsComparable(primary, candidate)) return candidate;
    }
    return null;
  }, [experimentRuns, areRunsComparable]);

  const summarizeMeasurementSummary = useCallback((summary?: Record<string, any> | null): string => {
    if (!summary || typeof summary !== 'object') return '';
    const preferredKeys = ['compile_time_ms', 'runtime_ms', 'binary_size_bytes', 'artifact_diff_score', 'comparison', 'repeat_count'];
    const parts: string[] = [];
    preferredKeys.forEach((key) => {
      if (summary[key] === undefined || summary[key] === null || summary[key] === '') return;
      parts.push(`${key}=${String(summary[key])}`);
    });
    return parts.slice(0, 6).join(' · ');
  }, []);

  const summarizeArtifactInventory = useCallback((inventory?: string[] | null): string => {
    if (!Array.isArray(inventory) || inventory.length === 0) return '';
    return inventory.slice(0, 4).join(', ');
  }, []);

  const summarizePerfCounters = useCallback((counters?: Record<string, any> | null): string => {
    if (!counters || typeof counters !== 'object') return '';
    return Object.entries(counters)
      .slice(0, 4)
      .map(([key, value]) => `${key}=${String(value)}`)
      .join(' · ');
  }, []);

  const summarizeCompilerArtifacts = useCallback((artifacts?: Record<string, any> | null): string[] => {
    if (!artifacts || typeof artifacts !== 'object') return [];
    const flags: string[] = [];
    if (artifacts.capture_ir) flags.push('IR captured');
    if (artifacts.capture_asm) flags.push('ASM captured');
    if (artifacts.capture_remarks) flags.push('Pass remarks captured');
    if (artifacts.capture_perf_stat) flags.push('Perf counters captured');
    const diffSummary = typeof artifacts.diff_summary === 'string' ? artifacts.diff_summary.trim() : '';
    if (diffSummary) flags.push(`Diff: ${diffSummary}`);
    const passSignals = Array.isArray(artifacts.pass_signals) ? artifacts.pass_signals.filter(Boolean) : [];
    if (passSignals.length > 0) flags.push(`Signals: ${passSignals.slice(0, 3).join(', ')}`);
    return flags.slice(0, 5);
  }, []);

  const structuredHypotheses = useMemo(() => {
    return Array.isArray(selectedNote?.structured_payload?.hypotheses) ? selectedNote!.structured_payload!.hypotheses! : [];
  }, [selectedNote]);

  const previousHypothesesById = useMemo(() => {
    const rows = Array.isArray(selectedNote?.structured_payload?.previous_hypotheses)
      ? selectedNote!.structured_payload!.previous_hypotheses!
      : [];
    return rows.reduce<Record<string, typeof rows[number]>>((acc, row) => {
      if (row?.id) acc[String(row.id)] = row;
      return acc;
    }, {});
  }, [selectedNote]);

  const reevaluationHistory = useMemo(() => {
    return Array.isArray(selectedNote?.structured_payload?.reevaluation_history)
      ? selectedNote!.structured_payload!.reevaluation_history!
      : [];
  }, [selectedNote]);

  const latestReevaluationEntry = useMemo(() => {
    return reevaluationHistory.length ? reevaluationHistory[reevaluationHistory.length - 1] : null;
  }, [reevaluationHistory]);

  const reevaluationStats = useMemo(() => {
    const deltas = Array.isArray(selectedNote?.structured_payload?.priority_deltas)
      ? selectedNote!.structured_payload!.priority_deltas!
      : [];
    const counts = { up: 0, down: 0, unchanged: 0, archived: 0, other: 0 };
    for (const delta of deltas) {
      const status = String((delta as any)?.status || '').trim().toLowerCase();
      if (status === 'up' || status === 'down' || status === 'unchanged' || status === 'archived') counts[status] += 1;
      else if (status) counts.other += 1;
    }
    return counts;
  }, [selectedNote]);

  const reevaluatedAt = useMemo(() => {
    const scoringPolicy = selectedNote?.structured_payload?.scoring_policy;
    const fromPolicy = scoringPolicy && typeof scoringPolicy === 'object' ? (scoringPolicy as any).reevaluated_at : undefined;
    return String(fromPolicy || latestReevaluationEntry?.saved_at || '').trim();
  }, [selectedNote, latestReevaluationEntry]);

  const isReevaluatedNote = useMemo(() => {
    return String(selectedNote?.structured_payload?.artifact_type || '').trim() === 'hypothesis_reevaluation';
  }, [selectedNote]);

  const pendingReevaluationJobId = useMemo(() => {
    return String(selectedNote?.structured_payload?.pending_reevaluation_job_id || '').trim();
  }, [selectedNote]);

  const pendingReevaluationCreatedAt = useMemo(() => {
    return String(selectedNote?.structured_payload?.pending_reevaluation_created_at || '').trim();
  }, [selectedNote]);

  const pendingReevaluationStatus = useMemo(() => {
    return String(selectedNote?.structured_payload?.pending_reevaluation_status || '').trim().toLowerCase();
  }, [selectedNote]);

  const pendingReevaluationCompletedAt = useMemo(() => {
    return String(selectedNote?.structured_payload?.pending_reevaluation_completed_at || '').trim();
  }, [selectedNote]);

  const pendingReevaluationError = useMemo(() => {
    return String(selectedNote?.structured_payload?.pending_reevaluation_error || '').trim();
  }, [selectedNote]);

  const autonomousOpportunityTargets = useMemo(() => {
    return extractAutonomousOpportunityTargets(selectedNote);
  }, [extractAutonomousOpportunityTargets, selectedNote]);

  const primaryAutonomousOpportunityTarget = useMemo(() => {
    return autonomousOpportunityTargets.length ? autonomousOpportunityTargets[0] : null;
  }, [autonomousOpportunityTargets]);
  const sourceReevaluationNoteId = useMemo(() => {
    return String((selectedNote?.attribution as any)?.saved_from_synthesis?.research_note_id || '').trim();
  }, [selectedNote]);

  const recommendedHypothesisId = useMemo(() => {
    if (!structuredHypotheses.length) return '';
    const sorted = [...structuredHypotheses].sort((a, b) => {
      const rankDiff = Number(a.rank || 9999) - Number(b.rank || 9999);
      if (rankDiff !== 0) return rankDiff;
      return Number(b.overall_score || 0) - Number(a.overall_score || 0);
    });
    return String(sorted[0]?.id || '').trim();
  }, [structuredHypotheses]);

  const recommendedHypothesisTitle = useMemo(() => {
    if (!recommendedHypothesisId) return '';
    const match = structuredHypotheses.find((hypothesis) => String(hypothesis.id || '').trim() === recommendedHypothesisId);
    return String(match?.title || '').trim();
  }, [structuredHypotheses, recommendedHypothesisId]);

  const hasFreshExperimentEvidence = useMemo(() => {
    if (!selectedNote?.structured_payload) return false;
    const payload = selectedNote.structured_payload;
    const hasEvidence = Array.isArray(payload.hypotheses)
      && payload.hypotheses.some((hypothesis) => Array.isArray(hypothesis.experiment_evidence) && hypothesis.experiment_evidence.length > 0);
    if (!hasEvidence) return false;
    if (payload.artifact_type !== 'hypothesis_reevaluation') return true;
    const updatedAt = reevaluatedAt ? new Date(reevaluatedAt).getTime() : (selectedNote.updated_at ? new Date(selectedNote.updated_at).getTime() : 0);
    const lastAppendedAt = payload.last_appended_at ? new Date(payload.last_appended_at).getTime() : 0;
    return lastAppendedAt > 0 && updatedAt > 0 && lastAppendedAt >= updatedAt;
  }, [selectedNote, reevaluatedAt]);

  const refreshSelectedResearchNote = useCallback(async () => {
    if (!selectedNote?.id) return;
    try {
      const note = await apiClient.getResearchNote(selectedNote.id);
      setSelectedNote(note);
      invalidateAutonomousOriginsForNote(note);
    } catch {
      // ignore
    }
  }, [invalidateAutonomousOriginsForNote, selectedNote?.id]);

  useEffect(() => {
    if (!selectedNote) return;
    const artifactType = String(selectedNote.structured_payload?.artifact_type || '').trim();
    const hasPendingReevaluation = Boolean(String(selectedNote.structured_payload?.pending_reevaluation_job_id || '').trim());
    if (artifactType === 'hypothesis_reevaluation' || hasPendingReevaluation) {
      invalidateAutonomousOriginsForNote(selectedNote);
    }
  }, [
    invalidateAutonomousOriginsForNote,
    selectedNote,
    selectedNote?.id,
    selectedNote?.structured_payload?.artifact_type,
    selectedNote?.structured_payload?.pending_reevaluation_job_id,
    selectedNote?.structured_payload?.pending_reevaluation_status,
    selectedNote?.structured_payload?.pending_reevaluation_completed_at,
    selectedNote?.structured_payload?.last_appended_at,
  ]);

  useEffect(() => {
    if (!selectedNote?.id || !pendingReevaluationJobId || pendingReevaluationStatus !== 'pending') return undefined;
    const timer = window.setInterval(() => {
      refreshSelectedResearchNote();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [selectedNote?.id, pendingReevaluationJobId, pendingReevaluationStatus, refreshSelectedResearchNote]);

  const reevaluateHypothesesMutation = useMutation(
    () =>
      apiClient.createSynthesisJob({
        job_type: 'hypothesis_reevaluation',
        title: `Hypothesis Re-evaluation · ${selectedNote!.title}`.slice(0, 500),
        document_ids: [],
        research_note_id: selectedNote!.id,
        output_format: 'markdown',
        output_style: 'technical',
      }),
    {
      onSuccess: (job) => {
        toast.success('Hypothesis re-evaluation started');
        navigate(`/synthesis?job=${encodeURIComponent(job.id)}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start hypothesis re-evaluation');
      },
    }
  );

  const explainRegressionMutation = useMutation(
    ({ primaryRunId, comparisonRunId }: { primaryRunId: string; comparisonRunId: string }) =>
      apiClient.createSynthesisJob({
        job_type: 'compiler_regression_explanation',
        title: `Compiler Regression Explanation · ${selectedNote!.title}`.slice(0, 500),
        document_ids: [],
        research_note_id: selectedNote?.id,
        experiment_run_ids: [primaryRunId, comparisonRunId],
        primary_run_id: primaryRunId,
        comparison_run_id: comparisonRunId,
        output_format: 'markdown',
        output_style: 'technical',
      }),
    {
      onSuccess: (job) => {
        toast.success('Compiler regression explanation started');
        navigate(`/synthesis?job=${encodeURIComponent(job.id)}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start compiler regression explanation');
      },
    }
  );

  const generatePatchProposalMutation = useMutation(
    () =>
      apiClient.createSynthesisJob({
        job_type: 'compiler_patch_proposal',
        title: `Compiler Patch Proposal · ${selectedNote!.title}`.slice(0, 500),
        document_ids: [],
        research_note_id: selectedNote!.id,
        output_format: 'markdown',
        output_style: 'technical',
      }),
    {
      onSuccess: (job) => {
        toast.success('Compiler patch proposal started');
        navigate(`/synthesis?job=${encodeURIComponent(job.id)}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start compiler patch proposal');
      },
    }
  );

  const generatePatchDraftMutation = useMutation(
    () => {
      const sourceId = experimentSourceId.trim();
      if (!sourceId) {
        return Promise.reject(new Error('Select a repo source before generating a patch draft'));
      }
      return apiClient.createSynthesisJob({
        job_type: 'compiler_patch_draft',
        title: `Compiler Patch Draft · ${selectedNote!.title}`.slice(0, 500),
        document_ids: [],
        research_note_id: selectedNote!.id,
        source_id: sourceId,
        output_format: 'markdown',
        output_style: 'technical',
      });
    },
    {
      onSuccess: (job) => {
        toast.success('Compiler patch draft started');
        navigate(`/synthesis?job=${encodeURIComponent(job.id)}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start compiler patch draft');
      },
    }
  );

  const generateExperimentPlanMutation = useMutation(
    (payload?: { plan_mode?: 'aggregate_note' | 'single_hypothesis' | 'compiler_regression_followup'; hypothesis_id?: string }) =>
      apiClient.generateExperimentPlan({
        note_id: selectedNote!.id,
        prefer_section: 'hypothesis',
        max_note_chars: 12000,
        plan_mode: payload?.plan_mode,
        hypothesis_id: payload?.hypothesis_id,
        benchmark_suite_id: selectedBenchmarkSuiteId || undefined,
        benchmark_case_ids: selectedBenchmarkCaseIds,
        include_ablations: true,
        include_timeline: true,
        include_risks: true,
        include_repro_checklist: true,
      }),
    {
      onSuccess: () => {
        toast.success('Experiment plan generated');
        refetchExperimentPlans();
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Experiment plan generation failed');
      },
    }
  );

  const buildRecommendedPlan = async () => {
    if (!selectedNote?.id) throw new Error('Select a note first');
    return apiClient.generateExperimentPlan({
      note_id: selectedNote.id,
      prefer_section: 'hypothesis',
      max_note_chars: 12000,
      benchmark_suite_id: selectedBenchmarkSuiteId || undefined,
      benchmark_case_ids: selectedBenchmarkCaseIds,
      include_ablations: true,
      include_timeline: true,
      include_risks: true,
      include_repro_checklist: true,
    });
  };

  const recommendedPostRunActions = selectedNote?.id
    ? {
        auto_append_to_note: true,
        target_note_id: selectedNote.id,
        append_status: 'pending',
      }
    : undefined;

  const createExperimentRunMutation = useMutation(
    (payload: { planId: string; name: string; config?: Record<string, any> | null; summary?: string | null }) =>
      apiClient.createExperimentRun(payload.planId, { name: payload.name, config: payload.config, summary: payload.summary }),
    {
      onSuccess: () => {
        toast.success('Run created');
        setNewExperimentRunName('');
        refetchExperimentRuns();
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to create run');
      },
    }
  );

  const runRecommendedHypothesisMutation = useMutation(
    async () => {
      if (!selectedNote?.id) throw new Error('Select a note first');
      const plan = await buildRecommendedPlan();
      const details = plan?.generator_details || {};
      const planBody = plan?.plan || {};
      const provenance = planBody.provenance && typeof planBody.provenance === 'object' ? planBody.provenance : {};
      const executionHandoff = {
        execution_handoff_version: 1,
        plan_scope:
          (typeof details.plan_mode === 'string' && details.plan_mode) ||
          (typeof planBody.plan_scope === 'string' && planBody.plan_scope) ||
          undefined,
        selected_hypothesis_ids: Array.isArray(details.selected_hypothesis_ids)
          ? details.selected_hypothesis_ids
          : (Array.isArray(planBody.selected_hypothesis_ids) ? planBody.selected_hypothesis_ids : []),
        supporting_sources: Array.isArray(details.supporting_sources)
          ? details.supporting_sources
          : (Array.isArray(planBody.supporting_sources) ? planBody.supporting_sources : []),
        source_paper_ids: Array.isArray(details.source_paper_ids)
          ? details.source_paper_ids
          : (Array.isArray((provenance as any).source_paper_ids) ? (provenance as any).source_paper_ids : []),
        source_document_ids: Array.isArray(details.source_document_ids)
          ? details.source_document_ids
          : (Array.isArray((provenance as any).source_document_ids) ? (provenance as any).source_document_ids : []),
      };
      const selectedIds = Array.isArray(executionHandoff.selected_hypothesis_ids) ? executionHandoff.selected_hypothesis_ids : [];
      const suggestedName =
        executionHandoff.plan_scope === 'single_hypothesis'
          ? `${plan.title} · ${selectedIds[0] || 'run'}`
          : `${plan.title} · run`;
      const cmds = experimentCommandsText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);
      const run = await apiClient.createExperimentRun(plan.id, {
        name: suggestedName,
        config: {
          execution_handoff: executionHandoff,
          source_id: experimentSourceId.trim() || undefined,
          commands: cmds.slice(0, 6),
          timeout_seconds: 60,
          post_run_actions: recommendedPostRunActions,
        },
        summary:
          typeof plan.plan?.objective === 'string' && plan.plan.objective
            ? String(plan.plan.objective)
            : undefined,
      });
      return { plan, run };
    },
    {
      onSuccess: ({ plan, run }) => {
        toast.success('Recommended run created');
        queryClient.invalidateQueries(['experiment-plans', selectedNote?.id]);
        queryClient.invalidateQueries(['experiment-runs', plan.id]);
        setNewExperimentRunName('');
        navigate(`/research-notes?note=${encodeURIComponent(selectedNote!.id)}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to create recommended run');
      },
    }
  );

  const updateExperimentRunMutation = useMutation(
    (payload: { runId: string; status: ExperimentRun['status'] }) => apiClient.updateExperimentRun(payload.runId, { status: payload.status }),
    {
      onSuccess: (run) => {
        refetchExperimentRuns();
        invalidateLinkedOpportunityQueries(run);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to update run');
      },
    }
  );

  const startExperimentRunMutation = useMutation(
    async (payload: { runId: string }) => {
      const sourceId = experimentSourceId.trim();
      if (!sourceId) throw new Error('Missing repo source ID');
      const cmds = experimentCommandsText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);
      if (cmds.length === 0) throw new Error('Provide at least one command');
      return apiClient.startExperimentRun(payload.runId, {
        source_id: sourceId,
        commands: cmds,
        timeout_seconds: 60,
        start_immediately: true,
      });
    },
    {
      onSuccess: (res) => {
        toast.success('Started experiment runner job');
        refetchExperimentRuns();
        invalidateLinkedOpportunityQueries(res?.run);
        if (res?.agent_job_id) {
          navigate(`/autonomous-agents?job=${encodeURIComponent(res.agent_job_id)}`);
        }
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start run');
      },
    }
  );

  const syncExperimentRunMutation = useMutation(
    (payload: { runId: string }) => apiClient.syncExperimentRun(payload.runId),
    {
      onSuccess: async (res) => {
        toast.success('Synced run');
        refetchExperimentRuns();
        queryClient.invalidateQueries(['research-notes']);
        invalidateLinkedOpportunityQueries(res?.run);
        await refreshSelectedResearchNote();
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Sync failed');
      },
    }
  );

  const performExperimentRunActionMutation = useMutation(
    (payload: { runId: string; action: 'start' | 'sync' | 'pause' | 'resume' | 'cancel' | 'retry' | 'requeue'; note?: string; startImmediately?: boolean }) =>
      apiClient.performExperimentRunAction(payload.runId, {
        action: payload.action,
        note: payload.note,
        start_immediately: payload.startImmediately,
      }),
    {
      onSuccess: async (res, variables) => {
        toast.success(`Run ${variables.action} applied`);
        setExperimentRunActionNotes((current) => ({ ...current, [variables.runId]: '' }));
        refetchExperimentRuns();
        queryClient.invalidateQueries(['research-notes']);
        invalidateLinkedOpportunityQueries(res?.run);
        await refreshSelectedResearchNote();
        if (variables.action === 'start' && res?.agent_job_id) {
          navigate(`/autonomous-agents?job=${encodeURIComponent(String(res.agent_job_id))}`);
        }
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to update scientific validation run');
      },
    }
  );

  const agentJobActionMutation = useMutation(
    (payload: { jobId: string; action: 'restart' }) => apiClient.performAgentJobAction(payload.jobId, payload.action, {}),
    {
      onSuccess: () => {
        toast.success('Recovery job restarted');
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to restart recovery job');
      },
    }
  );

  const startExperimentLoopMutation = useMutation(
    async () => {
      if (!selectedNote?.id) throw new Error('Select a note first');
      const sourceId = experimentSourceId.trim();
      if (!sourceId) throw new Error('Missing repo source ID');
      const cmds = experimentCommandsText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);
      if (cmds.length === 0) throw new Error('Provide at least one command');

      return apiClient.createJobFromChain({
        chain_definition_id: EXPERIMENT_LOOP_CHAIN_ID,
        name_prefix: `Experiment Loop - ${selectedNote.title}`.slice(0, 150),
        variables: { research_note_id: selectedNote.id },
        config_overrides: {
          research_note_id: selectedNote.id,
          source_id: sourceId,
          commands: cmds.slice(0, 6),
          max_runs: Math.max(1, Math.min(20, Number(experimentLoopMaxRuns) || 3)),
          timeout_seconds: 60,
          enable_experiments: true,
          append_to_note: true,
          stop_on_ok: !!experimentStopOnOk,
          stop_metric_regex: experimentStopMetricRegex.trim() || undefined,
          stop_metric_direction: experimentStopMetricDirection,
          stop_metric_window: Math.max(2, Math.min(10, Number(experimentStopMetricWindow) || 3)),
          stop_metric_min_improvement: Number(experimentStopMetricMinImprovement) || 0,
        },
        start_immediately: true,
      } as any);
    },
    {
      onSuccess: (job: any) => {
        toast.success('Experiment loop started');
        navigate(`/autonomous-agents?job=${encodeURIComponent(String(job.id))}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start experiment loop');
      },
    }
  );

  const startRecommendedLoopMutation = useMutation(
    async () => {
      if (!selectedNote?.id) throw new Error('Select a note first');
      const sourceId = experimentSourceId.trim();
      if (!sourceId) throw new Error('Missing repo source ID');
      const cmds = experimentCommandsText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);
      if (cmds.length === 0) throw new Error('Provide at least one command');

      const plan = await buildRecommendedPlan();
      const details = plan?.generator_details || {};
      const selectedIds = Array.isArray(details.selected_hypothesis_ids) ? details.selected_hypothesis_ids : [];
      return apiClient.createJobFromChain({
        chain_definition_id: EXPERIMENT_LOOP_CHAIN_ID,
        name_prefix: `Experiment Loop - ${selectedNote.title}`.slice(0, 150),
        variables: {
          research_note_id: selectedNote.id,
          experiment_plan_id: plan.id,
        },
        config_overrides: {
          research_note_id: selectedNote.id,
          experiment_plan_id: plan.id,
          source_id: sourceId,
          commands: cmds.slice(0, 6),
          max_runs: Math.max(1, Math.min(20, Number(experimentLoopMaxRuns) || 3)),
          timeout_seconds: 60,
          enable_experiments: true,
          append_to_note: true,
          stop_on_ok: !!experimentStopOnOk,
          stop_metric_regex: experimentStopMetricRegex.trim() || undefined,
          stop_metric_direction: experimentStopMetricDirection,
          stop_metric_window: Math.max(2, Math.min(10, Number(experimentStopMetricWindow) || 3)),
          stop_metric_min_improvement: Number(experimentStopMetricMinImprovement) || 0,
          selected_hypothesis_ids: selectedIds,
          reevaluation_mode: !!details.reevaluation_mode,
          reevaluation_source_job_id: details.reevaluation_source_job_id,
          post_run_actions: recommendedPostRunActions,
        },
        start_immediately: true,
      } as any);
    },
    {
      onSuccess: (job: any) => {
        toast.success('Recommended experiment loop started');
        navigate(`/autonomous-agents?job=${encodeURIComponent(String(job.id))}`);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to start recommended experiment loop');
      },
    }
  );

  const appendRunToNoteMutation = useMutation(
    (payload: { runId: string }) => apiClient.appendExperimentRunToNote(payload.runId),
    {
      onSuccess: (note) => {
        toast.success('Appended results to note');
        setSelectedNote(note);
        queryClient.invalidateQueries(['research-notes']);
        const appendedRunId = String((note as any)?.structured_payload?.last_appended_run_id || '').trim();
        if (appendedRunId) {
          const matchingRun = experimentRuns.find((row) => String(row.id) === appendedRunId) || null;
          invalidateLinkedOpportunityQueries(matchingRun);
        }
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Append failed');
      },
    }
  );

  const applyGeneratedMarkdownMutation = useMutation(
    (payload: { noteId: string; content: string }) =>
      apiClient.updateResearchNote(payload.noteId, { content_markdown: payload.content }),
    {
      onSuccess: (note) => {
        toast.success('Note updated');
        setSelectedNote(note);
        queryClient.invalidateQueries(['research-notes']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Update failed');
      },
    }
  );

  const CreateModal: React.FC = () => {
    const [title, setTitle] = useState('Research Note');
    const [content, setContent] = useState('');
    const [tags, setTags] = useState('hypotheses');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const handleCreate = async () => {
      if (!title.trim() || !content.trim()) {
        toast.error('Title and content are required');
        return;
      }
      setIsSubmitting(true);
      try {
        const created = await apiClient.createResearchNote({
          title: title.trim(),
          content_markdown: content,
          tags: tags
            .split(',')
            .map((t) => t.trim())
            .filter(Boolean),
        });
        toast.success('Note created');
        queryClient.invalidateQueries(['research-notes']);
        setShowCreateModal(false);
        navigate(`/research-notes?note=${encodeURIComponent(created.id)}`);
      } catch (e: any) {
        toast.error(e?.message || 'Create failed');
      } finally {
        setIsSubmitting(false);
      }
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
          <div className="p-6 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-lg font-semibold">New Research Note</h2>
            <Button variant="ghost" size="sm" onClick={() => setShowCreateModal(false)}>
              ✕
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Tags (comma-separated)</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Content (Markdown)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                rows={14}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="# Hypothesis\n\n..."
              />
            </div>
          </div>

          <div className="p-6 border-t border-gray-200 flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreate} disabled={isSubmitting}>
              {isSubmitting ? 'Creating…' : 'Create Note'}
            </Button>
          </div>
        </div>
      </div>
    );
  };

  const notes = useMemo(() => data?.items || [], [data?.items]);
  const filteredNotes = useMemo(() => {
    if (!onlyCitationIssues) return notes;
    return notes.filter((note: any) => {
      const lint = note?.attribution?.lint || note?.attribution;
      const coverage = typeof lint?.line_citation_coverage === 'number' ? lint.line_citation_coverage : null;
      const unknown = Array.isArray(lint?.unknown_citation_keys) ? lint.unknown_citation_keys : [];
      const bibliographyPresent = typeof lint?.bibliography_present === 'boolean' ? lint.bibliography_present : null;
      const lowCoverage = typeof coverage === 'number' ? coverage < 0.7 : false;
      const missingBiblio = bibliographyPresent === false;
      return unknown.length > 0 || lowCoverage || missingBiblio;
    });
  }, [notes, onlyCitationIssues]);

  const downloadMarkdown = (note: ResearchNote) => {
    const blob = new Blob([note.content_markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const safeTitle = (note.title || 'research_note').replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '_');
    a.download = `${safeTitle || 'research_note'}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const copyMarkdown = async (note: ResearchNote) => {
    try {
      await navigator.clipboard.writeText(note.content_markdown);
      toast.success('Copied to clipboard');
    } catch (e: any) {
      toast.error(e?.message || 'Copy failed');
    }
  };

  const copyText = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(`${label} copied`);
    } catch (e: any) {
      toast.error(e?.message || `Failed to copy ${label.toLowerCase()}`);
    }
  };

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Research Notes</h1>
          <p className="text-gray-500">Capture hypotheses, experiment plans, and insights</p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="w-4 h-4 mr-2" />
          New Note
        </Button>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            className="w-full border border-gray-300 rounded-lg pl-10 pr-4 py-2 text-sm"
            placeholder="Search notes…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="relative w-64">
          <Tag className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            className="w-full border border-gray-300 rounded-lg pl-10 pr-4 py-2 text-sm"
            placeholder="Filter by tag…"
            value={tagFilter}
            onChange={(e) => setTagFilter(e.target.value)}
          />
        </div>
        <Button variant="ghost" size="sm" onClick={() => refetch()}>
          ↻
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={lintRecentMutation.isLoading}
          onClick={() => lintRecentMutation.mutate()}
          title="Lint citations for recently updated notes (no LLM)"
        >
          {lintRecentMutation.isLoading ? 'Linting…' : 'Lint recent'}
        </Button>
        <label className="flex items-center gap-2 text-xs text-gray-600 ml-2 select-none">
          <input
            type="checkbox"
            checked={onlyCitationIssues}
            onChange={(e) => setOnlyCitationIssues(e.target.checked)}
          />
          Only citation issues
        </label>
      </div>

      <div className="flex-1 flex gap-6 min-h-0">
        <div className="w-2/3 overflow-y-auto">
          {isLoading ? (
            <div className="flex justify-center items-center h-full">
              <LoadingSpinner />
            </div>
          ) : filteredNotes.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <FileText className="w-12 h-12 mb-3 text-gray-400" />
              <p className="text-lg font-medium">{onlyCitationIssues ? 'No citation issues found' : 'No research notes yet'}</p>
              <p className="text-sm">
                {onlyCitationIssues ? 'Try running “Lint recent” or disable the filter.' : 'Create a note or save one from Synthesis'}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {filteredNotes.map((note: any) => (
                <div
                  key={note.id}
                  className={`bg-white border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
                    selectedNote?.id === note.id ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
                  }`}
                  onClick={() => {
                    setSelectedNote(note);
                    navigate(`/research-notes?note=${encodeURIComponent(note.id)}`);
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="font-medium text-gray-900 truncate">{note.title}</h3>
                      <p className="text-xs text-gray-500">
                        {note.updated_at ? new Date(note.updated_at).toLocaleString() : '-'}
                      </p>
                      {(() => {
                        const lint = (note as any)?.attribution?.lint || (note as any)?.attribution;
                        const cov = typeof lint?.line_citation_coverage === 'number' ? lint.line_citation_coverage : null;
                        const unknown = Array.isArray(lint?.unknown_citation_keys) ? lint.unknown_citation_keys : [];
                        if (cov === null && (!unknown || unknown.length === 0)) return null;
                        const pct = cov === null ? null : Math.round(cov * 100);
                        const color =
                          unknown.length > 0 ? 'bg-red-50 text-red-700' : pct !== null && pct < 70 ? 'bg-orange-50 text-orange-700' : 'bg-green-50 text-green-700';
                        return (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {pct !== null && (
                              <span className={`text-xs px-2 py-0.5 rounded ${color}`}>
                                Citations {pct}%
                              </span>
                            )}
                            {unknown.length > 0 && (
                              <span className="text-xs bg-red-50 text-red-700 px-2 py-0.5 rounded">
                                Unknown keys: {unknown.slice(0, 3).join(', ')}
                                {unknown.length > 3 ? '…' : ''}
                              </span>
                            )}
                            {(typeof lint?.bibliography_present === 'boolean' && lint.bibliography_present === false) && (
                              <span className="text-xs bg-orange-50 text-orange-700 px-2 py-0.5 rounded">
                                Missing bibliography
                              </span>
                            )}
                          </div>
                        );
                      })()}
                      {note.tags && note.tags.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {note.tags.slice(0, 6).map((t: string) => (
                            <span key={t} className="text-xs bg-gray-100 text-gray-700 px-2 py-0.5 rounded">
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm('Delete this note?')) {
                          deleteMutation.mutate(note.id);
                        }
                      }}
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </Button>
                  </div>
                  {(() => {
                    const lint = note?.attribution?.lint || note?.attribution;
                    const coverage = typeof lint?.line_citation_coverage === 'number' ? lint.line_citation_coverage : null;
                    const unknown = Array.isArray(lint?.unknown_citation_keys) ? lint.unknown_citation_keys : [];
                    const bibliographyPresent = typeof lint?.bibliography_present === 'boolean' ? lint.bibliography_present : null;
                    const lowCoverage = typeof coverage === 'number' ? coverage < 0.7 : false;
                    const missingBiblio = bibliographyPresent === false;
                    const needsAttention = unknown.length > 0 || lowCoverage || missingBiblio;
                    if (!needsAttention) return null;
                    return (
                      <div className="mt-3">
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/research-notes?note=${encodeURIComponent(note.id)}&action=citation-fix`);
                          }}
                        >
                          Fix citations
                        </Button>
                      </div>
                    );
                  })()}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="w-1/3">
          {selectedNote ? (
            <div className="bg-white border border-gray-200 rounded-lg h-full overflow-hidden flex flex-col">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold truncate">{selectedNote.title}</h2>
                <div className="flex items-center gap-2 mt-3">
                  <Button size="sm" variant="secondary" onClick={() => copyMarkdown(selectedNote)}>
                    <Copy className="w-4 h-4 mr-1" />
                    Copy
                  </Button>
                  <Button size="sm" variant="secondary" onClick={() => downloadMarkdown(selectedNote)}>
                    <Download className="w-4 h-4 mr-1" />
                    Download MD
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={enforceCitationsMutation.isLoading}
                    onClick={() => enforceCitationsMutation.mutate(selectedNote.id)}
                    title="Rewrite note to add citations based on its sources"
                  >
                    <Quote className="w-4 h-4 mr-1" />
                    {enforceCitationsMutation.isLoading ? 'Enforcing…' : 'Enforce citations'}
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={lintCitationsMutation.isLoading}
                    onClick={() => lintCitationsMutation.mutate(selectedNote.id)}
                    title="Analyze the current note for missing/invalid citations (no rewrite)"
                  >
                    {lintCitationsMutation.isLoading ? 'Linting…' : 'Lint'}
                  </Button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {urlAction === 'citation-fix' && (
                  <div className="mb-3 border border-orange-200 bg-orange-50 rounded-lg p-3 text-sm">
                    <div className="font-medium text-orange-900">Citation issue detected</div>
                    <div className="text-orange-800 text-xs mt-1">
                      This note was flagged by the citation monitor. Run lint to inspect, or enforce citations to rewrite.
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={lintCitationsMutation.isLoading}
                        onClick={() => lintCitationsMutation.mutate(selectedNote.id)}
                      >
                        {lintCitationsMutation.isLoading ? 'Linting…' : 'Lint now'}
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={enforceCitationsMutation.isLoading}
                        onClick={() => enforceCitationsMutation.mutate(selectedNote.id)}
                      >
                        {enforceCitationsMutation.isLoading ? 'Enforcing…' : 'Enforce citations'}
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={quickFixMutation.isLoading}
                        onClick={() => quickFixMutation.mutate(selectedNote.id)}
                        title="Strict rewrite + append bibliography + apply to note, then lint"
                      >
                        {quickFixMutation.isLoading ? 'Fixing…' : 'Fix now'}
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          const params = new URLSearchParams(location.search);
                          params.delete('action');
                          navigate(`/research-notes?${params.toString()}`, { replace: true });
                        }}
                      >
                        Dismiss
                      </Button>
                    </div>
                  </div>
                )}

                {selectedNote.tags && selectedNote.tags.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-1">
                    {selectedNote.tags.map((t) => (
                      <span key={t} className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded">
                        {t}
                      </span>
                    ))}
                  </div>
                )}

                {selectedNote.source_synthesis_job_id && (
                  <div className="mb-3 text-xs text-gray-600 bg-gray-50 rounded-lg p-3">
                    <div className="font-medium text-gray-700 mb-1">Provenance</div>
                    <div className="truncate">
                      Source synthesis job: {selectedNote.source_synthesis_job_id}
                    </div>
                    {sourceReevaluationNoteId ? (
                      <div className="mt-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => navigate(`/research-notes?note=${encodeURIComponent(sourceReevaluationNoteId)}`)}
                        >
                          Open source note
                        </Button>
                      </div>
                    ) : null}
                  </div>
                )}

                {Array.isArray(selectedNote.structured_payload?.hypotheses) && selectedNote.structured_payload.hypotheses.length > 0 && (
                  <div className="mb-3 text-xs text-gray-700 bg-amber-50 border border-amber-200 rounded-lg p-3">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <div className="font-medium text-amber-900">Ranked hypotheses</div>
                        <div className="text-amber-800 mt-1">
                          {selectedNote.structured_payload?.summary || 'Autonomous research memo with scored hypotheses.'}
                        </div>
                        {selectedNote.structured_payload?.artifact_type === 'hypothesis_reevaluation' && selectedNote.structured_payload?.reprioritization_summary ? (
                          <div className="mt-2 rounded border border-amber-300 bg-white px-3 py-2 text-[11px] text-amber-900">
                            <div className="font-medium">Re-evaluation summary</div>
                            <div className="mt-1">{selectedNote.structured_payload.reprioritization_summary}</div>
                            <div className="mt-1 text-amber-800">
                              {reevaluatedAt ? `Updated ${new Date(reevaluatedAt).toLocaleString()}` : 'Updated recently'}
                              {' · '}
                              {reevaluationStats.up} up
                              {' · '}
                              {reevaluationStats.down} down
                              {' · '}
                              {reevaluationStats.unchanged} unchanged
                              {' · '}
                              {reevaluationStats.archived} archived
                            </div>
                          </div>
                        ) : null}
                        {pendingReevaluationJobId ? (
                          <div className="mt-2 rounded border border-amber-300 bg-white px-3 py-2 text-[11px] text-amber-900">
                            <div className="font-medium">
                              {pendingReevaluationStatus === 'completed'
                                ? 'Reevaluation draft ready for review'
                                : pendingReevaluationStatus === 'failed'
                                  ? 'Reevaluation draft failed'
                                  : pendingReevaluationStatus === 'stale'
                                    ? 'Queued reevaluation draft is stale'
                                    : 'Reevaluation draft queued'}
                            </div>
                            <div className="mt-1">
                              {pendingReevaluationStatus === 'completed'
                                ? 'The queued reevaluation draft has finished and is ready to review before applying it to this note.'
                                : pendingReevaluationStatus === 'failed'
                                  ? (pendingReevaluationError || 'The queued reevaluation draft failed. Start a fresh reevaluation to recover.')
                                  : pendingReevaluationStatus === 'stale'
                                    ? 'A reevaluation draft exists, but newer experiment evidence has arrived since it was queued.'
                                    : 'New experiment evidence has already been queued for reevaluation.'}
                              {pendingReevaluationStatus === 'pending' && pendingReevaluationCreatedAt
                                ? ` Queued ${new Date(pendingReevaluationCreatedAt).toLocaleString()}.`
                                : pendingReevaluationStatus === 'completed' && pendingReevaluationCompletedAt
                                  ? ` Completed ${new Date(pendingReevaluationCompletedAt).toLocaleString()}.`
                                  : ''}
                            </div>
                            {Array.isArray(selectedNote.structured_payload?.pending_reevaluation_source_run_ids)
                              && selectedNote.structured_payload.pending_reevaluation_source_run_ids.length > 0 ? (
                              <div className="mt-1">
                                Source runs {selectedNote.structured_payload.pending_reevaluation_source_run_ids.slice(0, 3).join(', ')}
                              </div>
                            ) : null}
                            <div className="mt-2">
                              <div className="flex flex-wrap gap-2">
                                {pendingReevaluationStatus === 'completed' ? (
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => navigate(`/synthesis?job=${encodeURIComponent(pendingReevaluationJobId)}`)}
                                    title="Review the completed reevaluation draft on the Synthesis page"
                                  >
                                    Review reevaluation draft
                                  </Button>
                                ) : pendingReevaluationStatus === 'failed' || pendingReevaluationStatus === 'stale' ? (
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    disabled={reevaluateHypothesesMutation.isLoading || !selectedNote}
                                    onClick={() => reevaluateHypothesesMutation.mutate()}
                                    title="Start a fresh evidence-aware reevaluation"
                                  >
                                    {reevaluateHypothesesMutation.isLoading ? 'Re-evaluating…' : 'Re-evaluate hypotheses'}
                                  </Button>
                                ) : (
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => navigate(`/synthesis?job=${encodeURIComponent(pendingReevaluationJobId)}`)}
                                    title="Open the queued reevaluation draft on the Synthesis page"
                                  >
                                    Open queued reevaluation
                                  </Button>
                                )}
                                {primaryAutonomousOpportunityTarget ? (
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => navigate(
                                      primaryAutonomousOpportunityTarget.sourceKind === 'profile'
                                        ? `/autonomous-agents?tab=domain&profileId=${encodeURIComponent(primaryAutonomousOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(primaryAutonomousOpportunityTarget.opportunityId)}`
                                        : `/autonomous-agents?tab=fleet&fleetId=${encodeURIComponent(primaryAutonomousOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(primaryAutonomousOpportunityTarget.opportunityId)}`
                                    )}
                                    title="Open the originating autonomous opportunity"
                                  >
                                    Open originating opportunity
                                  </Button>
                                ) : null}
                              </div>
                            </div>
                          </div>
                        ) : hasFreshExperimentEvidence ? (
                          <div className="mt-2 text-[11px] text-amber-900">
                            New experiment evidence is available for re-evaluation.
                          </div>
                        ) : selectedNote.structured_payload?.artifact_type === 'hypothesis_reevaluation' ? (
                          <div className="mt-2 text-[11px] text-amber-900">
                            Hypothesis ranking is up to date with the latest reevaluation snapshot.
                          </div>
                        ) : null}
                        {selectedNote.structured_payload?.artifact_type === 'hypothesis_reevaluation' && reevaluationHistory.length > 0 ? (
                          <div className="mt-3 rounded border border-amber-300 bg-white px-3 py-2 text-[11px] text-amber-900">
                            <div className="font-medium">Reevaluation history</div>
                            <div className="mt-2 space-y-2">
                              {[...reevaluationHistory].slice().reverse().map((entry, index) => {
                                const isLatestEntry = index === 0;
                                const sourceNoteId = String(entry.source_note_id || '').trim();
                                const targetNoteId = String(entry.target_note_id || '').trim();
                                const entryOriginSourceKind = String(entry.origin_source_kind || '').trim().toLowerCase();
                                const entryOriginSourceId = String(entry.origin_source_id || '').trim();
                                const entryOriginOpportunityId = String(entry.origin_opportunity_id || '').trim();
                                const historyOpportunityTarget =
                                  (entryOriginSourceKind === 'profile' || entryOriginSourceKind === 'portfolio')
                                  && entryOriginSourceId
                                  && entryOriginOpportunityId
                                    ? {
                                        sourceKind: entryOriginSourceKind as 'profile' | 'portfolio',
                                        sourceId: entryOriginSourceId,
                                        opportunityId: entryOriginOpportunityId,
                                      }
                                    : primaryAutonomousOpportunityTarget;
                                const showSavedNoteLink = Boolean(
                                  targetNoteId
                                  && targetNoteId !== sourceNoteId
                                  && entry.outcome_status === 'saved_as_new_note'
                                );
                                const outcomeLabel =
                                  entry.outcome_status === 'applied_to_source_note'
                                    ? 'Applied'
                                    : entry.outcome_status === 'saved_as_new_note'
                                      ? 'Saved as new note'
                                      : entry.outcome_status === 'dismissed'
                                        ? 'Dismissed'
                                        : 'Recorded';
                                const statusClass =
                                  entry.outcome_status === 'dismissed'
                                    ? 'bg-gray-100 text-gray-700'
                                    : 'bg-emerald-100 text-emerald-800';
                                return (
                                  <div
                                    key={`${entry.job_id}-${entry.saved_at || index}`}
                                    className={`rounded border px-3 py-2 ${isLatestEntry ? 'border-emerald-300 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}
                                  >
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="font-medium">{isLatestEntry ? 'Latest reevaluation' : 'Historical reevaluation'}</span>
                                      <span className={`rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide ${statusClass}`}>
                                        {outcomeLabel}
                                      </span>
                                      {entry.outcome_recorded_at ? (
                                        <span className="text-amber-800">
                                          {new Date(entry.outcome_recorded_at).toLocaleString()}
                                        </span>
                                      ) : entry.saved_at ? (
                                        <span className="text-amber-800">
                                          {new Date(entry.saved_at).toLocaleString()}
                                        </span>
                                      ) : null}
                                    </div>
                                    {entry.reprioritization_summary ? (
                                      <div className="mt-1">{entry.reprioritization_summary}</div>
                                    ) : null}
                                    {Array.isArray(entry.source_run_ids) && entry.source_run_ids.length > 0 ? (
                                      <div className="mt-1 text-amber-800">
                                        Source runs {entry.source_run_ids.slice(0, 4).join(', ')}
                                      </div>
                                    ) : null}
                                    {entry.outcome_note ? (
                                      <div className="mt-1 text-amber-800">{entry.outcome_note}</div>
                                    ) : null}
                                    <div className="mt-2 flex flex-wrap gap-2">
                                      <Button
                                        size="sm"
                                        variant="ghost"
                                        onClick={() => navigate(`/synthesis?job=${encodeURIComponent(entry.job_id)}`)}
                                      >
                                        Open reevaluation job
                                      </Button>
                                      {sourceNoteId ? (
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          onClick={() => navigate(`/research-notes?note=${encodeURIComponent(sourceNoteId)}`)}
                                        >
                                          Open source note
                                        </Button>
                                      ) : null}
                                      {showSavedNoteLink ? (
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          onClick={() => navigate(`/research-notes?note=${encodeURIComponent(targetNoteId)}`)}
                                        >
                                          Open saved note
                                        </Button>
                                      ) : null}
                                      {historyOpportunityTarget ? (
                                        <Button
                                          size="sm"
                                          variant="ghost"
                                          onClick={() => navigate(
                                            historyOpportunityTarget.sourceKind === 'profile'
                                              ? `/autonomous-agents?tab=domain&profileId=${encodeURIComponent(historyOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(historyOpportunityTarget.opportunityId)}`
                                              : `/autonomous-agents?tab=fleet&fleetId=${encodeURIComponent(historyOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(historyOpportunityTarget.opportunityId)}`
                                          )}
                                        >
                                          Open originating opportunity
                                        </Button>
                                      ) : null}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <div className="text-[11px] uppercase tracking-wide text-amber-700">
                          {String(selectedNote.structured_payload?.research_mode || 'literature_to_hypothesis').replaceAll('_', ' ')}
                        </div>
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={reevaluateHypothesesMutation.isLoading || !selectedNote}
                          onClick={() => reevaluateHypothesesMutation.mutate()}
                          title="Run an evidence-aware synthesis pass to re-score and re-rank these hypotheses"
                        >
                          {reevaluateHypothesesMutation.isLoading ? 'Re-evaluating…' : 'Re-evaluate hypotheses'}
                        </Button>
                        {primaryAutonomousOpportunityTarget ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => navigate(
                              primaryAutonomousOpportunityTarget.sourceKind === 'profile'
                                ? `/autonomous-agents?tab=domain&profileId=${encodeURIComponent(primaryAutonomousOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(primaryAutonomousOpportunityTarget.opportunityId)}`
                                : `/autonomous-agents?tab=fleet&fleetId=${encodeURIComponent(primaryAutonomousOpportunityTarget.sourceId)}&opportunityId=${encodeURIComponent(primaryAutonomousOpportunityTarget.opportunityId)}`
                            )}
                            title="Open the autonomous opportunity most directly tied to this reevaluation"
                          >
                            Open originating opportunity
                          </Button>
                        ) : null}
                      </div>
                    </div>
                    <div className="mt-3 space-y-2">
                      {selectedNote.structured_payload.hypotheses.map((hypothesis) => (
                        <div key={hypothesis.id || `${hypothesis.rank}-${hypothesis.title}`} className="rounded border border-amber-200 bg-white p-3">
                          {(() => {
                            const latestEvidence = Array.isArray(hypothesis.experiment_evidence) && hypothesis.experiment_evidence.length > 0
                              ? hypothesis.experiment_evidence[hypothesis.experiment_evidence.length - 1]
                              : null;
                            const previousHypothesis = hypothesis.id ? previousHypothesesById[String(hypothesis.id)] : undefined;
                            const priorityDelta = Array.isArray(selectedNote.structured_payload?.priority_deltas)
                              ? selectedNote.structured_payload?.priority_deltas?.find((item: any) => String(item?.hypothesis_id || '') === String(hypothesis.id || ''))
                              : null;
                            const isRecommended = Boolean(isReevaluatedNote && hypothesis.id && String(hypothesis.id) === recommendedHypothesisId);
                            const previousRank = previousHypothesis?.rank;
                            const scoreChanged = previousHypothesis
                              && (
                                Number(previousHypothesis.overall_score || 0) !== Number(hypothesis.overall_score || 0)
                                || Number(previousHypothesis.evidence_score || 0) !== Number(hypothesis.evidence_score || 0)
                                || Number(previousHypothesis.testability_score || 0) !== Number(hypothesis.testability_score || 0)
                              );
                            return (
                              <>
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="font-medium text-gray-900">
                                {hypothesis.rank}. {hypothesis.title}
                                {isRecommended ? (
                                  <span className="ml-2 rounded bg-emerald-100 px-2 py-0.5 text-[10px] uppercase tracking-wide text-emerald-800">
                                    Recommended
                                  </span>
                                ) : null}
                              </div>
                              <div className="mt-1 text-gray-700">{hypothesis.claim}</div>
                            </div>
                            <div className="text-right text-[11px] text-gray-600 whitespace-nowrap">
                              <div>Overall {Number(hypothesis.overall_score || 0).toFixed(2)}</div>
                              <div>N {Number(hypothesis.novelty_score || 0).toFixed(2)} · E {Number(hypothesis.evidence_score || 0).toFixed(2)} · T {Number(hypothesis.testability_score || 0).toFixed(2)}</div>
                            </div>
                          </div>
                          {Array.isArray(hypothesis.supporting_sources) && hypothesis.supporting_sources.length > 0 ? (
                            <div className="mt-2 text-gray-600">
                              Evidence: {hypothesis.supporting_sources.slice(0, 3).map((source) => String(source?.title || source?.id || 'source')).join(', ')}
                            </div>
                          ) : null}
                          {hypothesis.recommended_next_step ? (
                            <div className="mt-2 text-gray-700">
                              Next step: <span className="text-gray-600">{hypothesis.recommended_next_step}</span>
                            </div>
                          ) : null}
                          {previousHypothesis ? (
                            <div className="mt-3 rounded border border-sky-200 bg-sky-50 p-2 text-[11px] text-sky-900">
                              <div className="font-medium">Compared with previous snapshot</div>
                              <div className="mt-1">
                                Rank {previousRank} {'->'} {hypothesis.rank}
                                {priorityDelta?.status ? ` · ${String(priorityDelta.status)}` : ''}
                              </div>
                              {(scoreChanged || priorityDelta?.reason) ? (
                                <div className="mt-1 text-sky-800">
                                  Overall {Number(previousHypothesis.overall_score || 0).toFixed(2)} {'->'} {Number(hypothesis.overall_score || 0).toFixed(2)}
                                  {' · '}
                                  Evidence {Number(previousHypothesis.evidence_score || 0).toFixed(2)} {'->'} {Number(hypothesis.evidence_score || 0).toFixed(2)}
                                  {' · '}
                                  Testability {Number(previousHypothesis.testability_score || 0).toFixed(2)} {'->'} {Number(hypothesis.testability_score || 0).toFixed(2)}
                                </div>
                              ) : null}
                              {priorityDelta?.reason ? (
                                <div className="mt-1 text-sky-800">{String(priorityDelta.reason)}</div>
                              ) : null}
                            </div>
                          ) : null}
                          {latestEvidence ? (
                            <div className="mt-3 rounded border border-emerald-200 bg-emerald-50 p-2 text-[11px] text-emerald-900">
                              <div className="font-medium">Latest experiment evidence</div>
                              <div className="mt-1">
                                Run {latestEvidence.run_id}
                                {latestEvidence.status ? ` · ${latestEvidence.status}` : ''}
                                {latestEvidence.plan_scope ? ` · ${latestEvidence.plan_scope}` : ''}
                              </div>
                              {latestEvidence.summary ? (
                                <div className="mt-1 text-emerald-800">{latestEvidence.summary}</div>
                              ) : null}
                              {Array.isArray(latestEvidence.result_highlights) && latestEvidence.result_highlights.length > 0 ? (
                                <div className="mt-1 text-emerald-800">
                                  {latestEvidence.result_highlights.slice(0, 2).join(' · ')}
                                </div>
                              ) : null}
                              {summarizeMeasurementSummary(latestEvidence.measurement_summary || undefined) ? (
                                <div className="mt-1 text-emerald-800">
                                  {summarizeMeasurementSummary(latestEvidence.measurement_summary || undefined)}
                                </div>
                              ) : null}
                              {summarizeArtifactInventory(latestEvidence.artifact_inventory || latestEvidence.compiler_artifacts?.artifact_inventory) ? (
                                <div className="mt-1 text-emerald-800">
                                  Artifacts: {summarizeArtifactInventory(latestEvidence.artifact_inventory || latestEvidence.compiler_artifacts?.artifact_inventory)}
                                </div>
                              ) : null}
                              {summarizeCompilerArtifacts(latestEvidence.compiler_artifacts || undefined).length > 0 ? (
                                <div className="mt-1 text-emerald-800">
                                  {summarizeCompilerArtifacts(latestEvidence.compiler_artifacts || undefined).join(' · ')}
                                </div>
                              ) : null}
                              {summarizePerfCounters(latestEvidence.perf_counters || undefined) ? (
                                <div className="mt-1 text-emerald-800">
                                  Perf: {summarizePerfCounters(latestEvidence.perf_counters || undefined)}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                          <div className="mt-3">
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={generateExperimentPlanMutation.isLoading || !selectedNote}
                              onClick={() =>
                                generateExperimentPlanMutation.mutate({
                                  plan_mode: 'single_hypothesis',
                                  hypothesis_id: hypothesis.id,
                                })
                              }
                              title="Generate a focused experiment plan for this hypothesis"
                            >
                              {generateExperimentPlanMutation.isLoading ? 'Generating…' : 'Generate plan for this hypothesis'}
                            </Button>
                          </div>
                              </>
                            );
                          })()}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {selectedNote.structured_payload?.artifact_type === 'compiler_regression_explanation' && (
                  <div className="mb-3 text-xs text-sky-900 bg-sky-50 border border-sky-200 rounded-lg p-3">
                    <div className="font-medium text-sky-900">Compiler regression explanation</div>
                    {selectedNote.structured_payload?.summary ? (
                      <div className="mt-1 text-sky-800">{selectedNote.structured_payload.summary}</div>
                    ) : null}
                    <div className="mt-2 text-[11px] text-sky-800">
                      {selectedNote.structured_payload?.regression_type ? `Type ${selectedNote.structured_payload.regression_type}` : 'Type mixed'}
                      {selectedNote.structured_payload?.primary_run_id ? ` · Primary ${selectedNote.structured_payload.primary_run_id}` : ''}
                      {selectedNote.structured_payload?.comparison_run_id ? ` · Comparison ${selectedNote.structured_payload.comparison_run_id}` : ''}
                    </div>
                    {selectedNote.structured_payload?.benchmark_suite_id ? (
                      <div className="mt-1 text-[11px] text-sky-800">
                        Benchmark suite {selectedNote.structured_payload.benchmark_suite_id}
                        {Array.isArray(selectedNote.structured_payload?.benchmark_case_ids) && selectedNote.structured_payload.benchmark_case_ids.length > 0
                          ? ` · ${selectedNote.structured_payload.benchmark_case_ids.join(', ')}`
                          : ''}
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.metric_deltas) && selectedNote.structured_payload.metric_deltas.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-sky-900">Metric deltas</div>
                        <ul className="mt-1 space-y-1 text-sky-800">
                          {selectedNote.structured_payload.metric_deltas.slice(0, 5).map((item: any, idx: number) => (
                            <li key={`metric-delta-${idx}`}>
                              {String(item?.metric || 'metric')}: {String(item?.comparison ?? '?')} → {String(item?.primary ?? '?')}
                              {item?.interpretation ? ` · ${String(item.interpretation)}` : ''}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.likely_causes) && selectedNote.structured_payload.likely_causes.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-sky-900">Likely causes</div>
                        <ul className="mt-1 space-y-1 text-sky-800">
                          {selectedNote.structured_payload.likely_causes.slice(0, 4).map((item: any, idx: number) => (
                            <li key={`likely-cause-${idx}`}>
                              {String(item?.title || 'Cause')}
                              {item?.confidence ? ` [${String(item.confidence)}]` : ''}
                              {item?.reason ? ` · ${String(item.reason)}` : ''}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.confounders) && selectedNote.structured_payload.confounders.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-sky-900">Confounders</div>
                        <ul className="mt-1 space-y-1 text-sky-800">
                          {selectedNote.structured_payload.confounders.slice(0, 4).map((item: any, idx: number) => (
                            <li key={`confounder-${idx}`}>{String(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.recommended_next_steps) && selectedNote.structured_payload.recommended_next_steps.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-sky-900">Recommended next steps</div>
                        <ul className="mt-1 space-y-1 text-sky-800">
                          {selectedNote.structured_payload.recommended_next_steps.slice(0, 5).map((item: any, idx: number) => (
                            <li key={`recommended-step-${idx}`}>{String(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                )}

                {selectedNote.structured_payload?.artifact_type === 'compiler_patch_proposal' && (
                  <div className="mb-3 text-xs text-teal-900 bg-teal-50 border border-teal-200 rounded-lg p-3">
                    <div className="font-medium text-teal-900">Compiler patch proposal</div>
                    {selectedNote.structured_payload?.proposal_summary ? (
                      <div className="mt-1 text-teal-800">{selectedNote.structured_payload.proposal_summary}</div>
                    ) : null}
                    {selectedNote.structured_payload?.target_area ? (
                      <div className="mt-2 text-[11px] text-teal-800">
                        Target area {selectedNote.structured_payload.target_area}
                        {selectedNote.structured_payload?.source_explanation_note_id
                          ? ` · Source explanation ${selectedNote.structured_payload.source_explanation_note_id}`
                          : ''}
                      </div>
                    ) : null}
                    {selectedNote.structured_payload?.candidate_change ? (
                      <div className="mt-3">
                        <div className="font-medium text-teal-900">Candidate change</div>
                        <div className="mt-1 text-teal-800 whitespace-pre-wrap">{selectedNote.structured_payload.candidate_change}</div>
                      </div>
                    ) : null}
                    {selectedNote.structured_payload?.expected_effect ? (
                      <div className="mt-3">
                        <div className="font-medium text-teal-900">Expected effect</div>
                        <div className="mt-1 text-teal-800 whitespace-pre-wrap">{selectedNote.structured_payload.expected_effect}</div>
                      </div>
                    ) : null}
                    {selectedNote.structured_payload?.mechanism ? (
                      <div className="mt-3">
                        <div className="font-medium text-teal-900">Mechanism</div>
                        <div className="mt-1 text-teal-800 whitespace-pre-wrap">{selectedNote.structured_payload.mechanism}</div>
                      </div>
                    ) : null}
                    {selectedNote.structured_payload?.validation_plan ? (
                      <div className="mt-3">
                        <div className="font-medium text-teal-900">Validation plan</div>
                        <div className="mt-1 text-teal-800 whitespace-pre-wrap">{selectedNote.structured_payload.validation_plan}</div>
                      </div>
                    ) : null}
                    {selectedNote.structured_payload?.risk_assessment ? (
                      <div className="mt-3">
                        <div className="font-medium text-teal-900">Risk assessment</div>
                        <div className="mt-1 text-teal-800 whitespace-pre-wrap">{selectedNote.structured_payload.risk_assessment}</div>
                      </div>
                    ) : null}
                    {selectedNote.structured_payload?.rollback_or_guardrail ? (
                      <div className="mt-3">
                        <div className="font-medium text-teal-900">Rollback or guardrail</div>
                        <div className="mt-1 text-teal-800 whitespace-pre-wrap">{selectedNote.structured_payload.rollback_or_guardrail}</div>
                      </div>
                    ) : null}
                  </div>
                )}

                {selectedNote.structured_payload?.artifact_type === 'compiler_patch_draft' && (
                  <div className="mb-3 text-xs text-violet-900 bg-violet-50 border border-violet-200 rounded-lg p-3">
                    <div className="font-medium text-violet-900">Compiler patch draft</div>
                    {selectedNote.structured_payload?.draft_summary ? (
                      <div className="mt-1 text-violet-800">{selectedNote.structured_payload.draft_summary}</div>
                    ) : null}
                    <div className="mt-2 text-[11px] text-violet-800">
                      {selectedNote.structured_payload?.source_name ? `Repo source ${selectedNote.structured_payload.source_name}` : ''}
                      {selectedNote.structured_payload?.source_id ? ` · ${selectedNote.structured_payload.source_id}` : ''}
                    </div>
                    {Array.isArray(selectedNote.structured_payload?.target_files) && selectedNote.structured_payload.target_files.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-violet-900">Target files</div>
                        <ul className="mt-1 space-y-1 text-violet-800">
                          {selectedNote.structured_payload.target_files.slice(0, 8).map((item: any, idx: number) => (
                            <li key={`draft-file-${idx}`}>{String(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.target_symbols) && selectedNote.structured_payload.target_symbols.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-violet-900">Target symbols</div>
                        <div className="mt-1 text-violet-800">{selectedNote.structured_payload.target_symbols.slice(0, 8).join(', ')}</div>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.change_plan) && selectedNote.structured_payload.change_plan.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-violet-900">Change plan</div>
                        <ul className="mt-1 space-y-1 text-violet-800">
                          {selectedNote.structured_payload.change_plan.slice(0, 8).map((item: any, idx: number) => (
                            <li key={`draft-step-${idx}`}>{String(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.validation_commands) && selectedNote.structured_payload.validation_commands.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-violet-900">Validation commands</div>
                        <ul className="mt-1 space-y-1 text-violet-800">
                          {selectedNote.structured_payload.validation_commands.slice(0, 6).map((item: any, idx: number) => (
                            <li key={`draft-validation-${idx}`}>{String(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {Array.isArray(selectedNote.structured_payload?.rollback_steps) && selectedNote.structured_payload.rollback_steps.length > 0 ? (
                      <div className="mt-3">
                        <div className="font-medium text-violet-900">Rollback steps</div>
                        <ul className="mt-1 space-y-1 text-violet-800">
                          {selectedNote.structured_payload.rollback_steps.slice(0, 6).map((item: any, idx: number) => (
                            <li key={`draft-rollback-${idx}`}>{String(item)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                )}

                <div className="mb-3 flex items-center gap-3">
                  <div className="text-xs text-gray-600">Citation policy</div>
                  <select
                    className="border border-gray-300 rounded px-2 py-1 text-xs bg-white"
                    value={citationPolicy}
                    onChange={(e) => setCitationPolicy(e.target.value as 'sentence' | 'paragraph')}
                  >
                    <option value="sentence">Sentence</option>
                    <option value="paragraph">Paragraph</option>
                  </select>
                  <label className="flex items-center gap-2 text-xs text-gray-600">
                    <input
                      type="checkbox"
                      checked={citationUpdateContent}
                      onChange={(e) => setCitationUpdateContent(e.target.checked)}
                    />
                    Apply cited markdown to note
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-600">
                    <input
                      type="checkbox"
                      checked={citationAppendBibliography}
                      onChange={(e) => setCitationAppendBibliography(e.target.checked)}
                    />
                    Append bibliography
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-600">
                    <input
                      type="checkbox"
                      checked={citationStrict}
                      onChange={(e) => setCitationStrict(e.target.checked)}
                    />
                    Strict
                  </label>
                </div>

                <div className="mb-3 text-xs text-gray-600 bg-gray-50 rounded-lg p-3">
                  <div className="font-medium text-gray-700 mb-2">Citation evidence</div>
                  {selectedNote.attribution && (
                    <div className="mb-2">
                      <Button size="sm" variant="secondary" onClick={loadSettingsFromLastRun}>
                        Load last run settings
                      </Button>
                    </div>
                  )}
                  <div className="flex flex-wrap items-center gap-3">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={citationUseVector}
                        onChange={(e) => setCitationUseVector(e.target.checked)}
                      />
                      Use vector chunks
                    </label>
                    <label className="flex items-center gap-2">
                      <span>Sources</span>
                      <input
                        className="w-16 border border-gray-300 rounded px-2 py-1"
                        type="number"
                        min={1}
                        max={25}
                        value={citationMaxSources}
                        onChange={(e) => setCitationMaxSources(Math.max(1, Math.min(25, Number(e.target.value) || 1)))}
                      />
                    </label>
                    <label className="flex items-center gap-2">
                      <span>Chunks/source</span>
                      <input
                        className="w-16 border border-gray-300 rounded px-2 py-1"
                        type="number"
                        min={1}
                        max={8}
                        value={citationChunksPerSource}
                        onChange={(e) =>
                          setCitationChunksPerSource(Math.max(1, Math.min(8, Number(e.target.value) || 1)))
                        }
                        disabled={!citationUseVector}
                      />
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={citationOverrideDocsEnabled}
                        onChange={(e) => setCitationOverrideDocsEnabled(e.target.checked)}
                      />
                      Override sources
                    </label>
                  </div>
                  {citationOverrideDocsEnabled && (
                    <div className="mt-2">
                      <div className="text-gray-700 mb-1">
                        Document IDs (UUIDs, comma/newline separated)
                        {selectedNote?.source_document_ids?.length ? (
                          <span className="ml-2 text-gray-500">
                            (note has {selectedNote.source_document_ids.length} source IDs)
                          </span>
                        ) : null}
                      </div>
                      <textarea
                        className="w-full border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                        rows={2}
                        placeholder="e.g. 123e4567-e89b-12d3-a456-426614174000"
                        value={citationOverrideDocIdsText}
                        onChange={(e) => setCitationOverrideDocIdsText(e.target.value)}
                      />
                      {parsedOverrideDocIds && parsedOverrideDocIds.length === 0 && (
                        <div className="mt-1 text-xs text-red-700">No valid UUIDs detected.</div>
                      )}
                      {parsedOverrideDocIds && parsedOverrideDocIds.length > 0 && (
                        <div className="mt-1 text-xs text-gray-600">
                          Using {parsedOverrideDocIds.length} document IDs (will ignore note/synthesis provenance).
                        </div>
                      )}
                    </div>
                  )}
                  <div className="mt-2">
                    <div className="text-gray-700 mb-1">Evidence query (optional)</div>
                    <textarea
                      className="w-full border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                      rows={2}
                      placeholder="Leave empty to auto-derive from title + note"
                      value={citationChunkQuery}
                      onChange={(e) => setCitationChunkQuery(e.target.value)}
                      disabled={!citationUseVector}
                    />
                  </div>
                </div>

                <div className="mb-3 text-xs text-gray-700 bg-gray-50 rounded-lg p-3">
                  <div className="mb-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                    <label className="block">
                      <div className="text-xs font-medium text-gray-700">Compiler benchmark suite</div>
                      <select
                        className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs bg-white"
                        value={selectedBenchmarkSuiteId}
                        onChange={(e) => setSelectedBenchmarkSuiteId(String(e.target.value || '').trim())}
                      >
                        <option value="">No benchmark suite</option>
                        {benchmarkSuites.map((suite) => (
                          <option key={suite.id} value={suite.id}>
                            {suite.name} ({suite.benchmark_family})
                          </option>
                        ))}
                      </select>
                      {selectedBenchmarkSuite ? (
                        <div className="mt-1 text-[11px] text-gray-500">{selectedBenchmarkSuite.description}</div>
                      ) : null}
                    </label>
                    {selectedBenchmarkSuite ? (
                      <label className="block">
                        <div className="text-xs font-medium text-gray-700">Benchmark cases</div>
                        <select
                          className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs bg-white"
                          multiple
                          size={Math.min(5, Math.max(2, selectedBenchmarkSuite.cases.length))}
                          value={selectedBenchmarkCaseIds}
                          onChange={(e) =>
                            setSelectedBenchmarkCaseIds(Array.from(e.target.selectedOptions).map((option) => String(option.value)))
                          }
                        >
                          {selectedBenchmarkSuite.cases.map((benchmarkCase) => (
                            <option key={benchmarkCase.id} value={benchmarkCase.id}>
                              {benchmarkCase.name}
                            </option>
                          ))}
                        </select>
                        <div className="mt-1 text-[11px] text-gray-500">Leave unselected to use the suite default cases.</div>
                      </label>
                    ) : null}
                  </div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <div className="font-medium text-gray-700">Experiment Orchestrator</div>
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-1 text-[11px] text-gray-600">
                        <span>Runs</span>
                        <input
                          className="w-14 border border-gray-300 rounded px-2 py-1 text-[11px]"
                          type="number"
                          min={1}
                          max={20}
                          value={experimentLoopMaxRuns}
                          onChange={(e) =>
                            setExperimentLoopMaxRuns(Math.max(1, Math.min(20, Number(e.target.value) || 1)))
                          }
                          title="How many runs to schedule in the autonomous experiment loop"
                        />
                      </label>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={(isReevaluatedNote ? startRecommendedLoopMutation.isLoading : startExperimentLoopMutation.isLoading) || !selectedNote}
                        onClick={() => startExperimentLoopMutation.mutate()}
                        title="Start the autonomous experiment loop playbook (seeded, configurable runs)"
                      >
                        {startExperimentLoopMutation.isLoading ? 'Starting…' : 'Start loop'}
                      </Button>
                      {isReevaluatedNote ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={startRecommendedLoopMutation.isLoading || !selectedNote}
                          onClick={() => startRecommendedLoopMutation.mutate()}
                          title="Generate the next plan, then start the autonomous loop from the recommended hypothesis"
                        >
                          {startRecommendedLoopMutation.isLoading ? 'Starting…' : 'Start loop from recommended'}
                        </Button>
                      ) : null}
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={generateExperimentPlanMutation.isLoading || !selectedNote}
                        onClick={() =>
                          generateExperimentPlanMutation.mutate(
                            selectedNote?.structured_payload?.artifact_type === 'compiler_regression_explanation'
                              ? undefined
                              : isReevaluatedNote
                              ? undefined
                              : {
                                  plan_mode: 'aggregate_note',
                                }
                          )
                        }
                        title={
                          selectedNote?.structured_payload?.artifact_type === 'compiler_regression_explanation'
                            ? "Generate the next benchmark-backed follow-up plan from this regression explanation"
                            : isReevaluatedNote
                            ? "Generate the next plan from the current top-ranked reevaluated hypothesis"
                            : structuredHypotheses.length > 0
                            ? "Generate a coordinated experiment plan from the note's top hypotheses"
                            : "Generate a runnable experiment template from the note's Hypothesis section"
                        }
                        >
                        {generateExperimentPlanMutation.isLoading
                          ? 'Generating…'
                          : selectedNote?.structured_payload?.artifact_type === 'compiler_regression_explanation'
                            ? 'Generate follow-up plan'
                          : isReevaluatedNote
                            ? 'Generate next plan'
                            : structuredHypotheses.length > 0
                            ? 'Generate aggregate plan'
                            : 'Generate plan'}
                      </Button>
                      {selectedNote?.structured_payload?.artifact_type === 'compiler_regression_explanation' ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={generatePatchProposalMutation.isLoading || !selectedNote}
                          onClick={() => generatePatchProposalMutation.mutate()}
                          title="Turn this explanation into a bounded compiler patch proposal"
                        >
                          {generatePatchProposalMutation.isLoading ? 'Generating…' : 'Generate patch proposal'}
                        </Button>
                      ) : null}
                      {selectedNote?.structured_payload?.artifact_type === 'compiler_patch_proposal' ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={generatePatchDraftMutation.isLoading || !selectedNote || !experimentSourceId.trim()}
                          onClick={() => generatePatchDraftMutation.mutate()}
                          title="Turn this proposal into a repo-aware patch draft using the selected repo source"
                        >
                          {generatePatchDraftMutation.isLoading ? 'Generating…' : 'Generate patch draft'}
                        </Button>
                      ) : null}
                      {isReevaluatedNote ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={runRecommendedHypothesisMutation.isLoading || !selectedNote}
                          onClick={() => runRecommendedHypothesisMutation.mutate()}
                          title="Generate the next plan and create a run for the recommended hypothesis"
                        >
                          {runRecommendedHypothesisMutation.isLoading ? 'Creating…' : 'Run recommended hypothesis'}
                        </Button>
                      ) : null}
                      {isReevaluatedNote ? (
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={generateExperimentPlanMutation.isLoading || !selectedNote}
                          onClick={() =>
                            generateExperimentPlanMutation.mutate({
                              plan_mode: 'aggregate_note',
                            })
                          }
                          title="Generate a coordinated aggregate plan across the reevaluated hypotheses"
                        >
                          {generateExperimentPlanMutation.isLoading ? 'Generating…' : 'Generate aggregate plan'}
                        </Button>
                      ) : null}
                    </div>
                  </div>
                  {isReevaluatedNote && recommendedHypothesisId ? (
                    <div className="mb-3 text-xs text-gray-600">
                      Recommended hypothesis:{' '}
                      <span className="font-medium text-gray-800">
                        {recommendedHypothesisTitle || recommendedHypothesisId}
                      </span>
                      {recommendedHypothesisTitle && recommendedHypothesisTitle !== recommendedHypothesisId ? (
                        <span className="text-gray-500"> · {recommendedHypothesisId}</span>
                      ) : null}
                    </div>
                  ) : null}

                  {latestExperimentPlan ? (
                    <div
                      ref={latestPlanRef}
                      role="region"
                      aria-label={`Experiment plan ${latestExperimentPlan.title}`}
                      className={`space-y-3 rounded border px-3 py-2 transition-colors ${isDeepLinkedLatestPlan ? 'border-primary-400 bg-primary-50/40' : 'border-transparent'}`}
                    >
                      <div className="text-xs text-gray-600">
                        Latest plan: <span className="font-medium text-gray-800">{latestExperimentPlan.title}</span>
                        {latestExperimentPlan.created_at ? (
                          <span className="text-gray-500"> · {new Date(latestExperimentPlan.created_at).toLocaleString()}</span>
                        ) : null}
                        {latestExperimentPlan.generator_details?.plan_mode ? (
                          <span className="text-gray-500">
                            {' '}· {latestExperimentPlan.generator_details.plan_mode === 'single_hypothesis'
                              ? 'Single hypothesis'
                              : latestExperimentPlan.generator_details.plan_mode === 'compiler_regression_followup'
                                ? 'Regression follow-up'
                                : 'Aggregate note'}
                          </span>
                        ) : null}
                        {latestExperimentPlan.generator_details?.reevaluation_mode ? (
                          <span className="text-gray-500"> · Reevaluated ranking</span>
                        ) : null}
                        {latestExperimentPlan.generator_details?.explanation_mode ? (
                          <span className="text-gray-500"> · Explanation-guided</span>
                        ) : null}
                      </div>
                      {latestExperimentPlan.generator_details?.selected_hypothesis_ids?.length ? (
                        <div className="text-xs text-gray-600">
                          Hypotheses: {latestExperimentPlan.generator_details.selected_hypothesis_ids.join(', ')}
                        </div>
                      ) : null}
                      {latestExperimentPlan.generator_details?.reevaluation_source_job_id ? (
                        <div className="text-xs text-gray-600">
                          Reevaluation job: {latestExperimentPlan.generator_details.reevaluation_source_job_id}
                        </div>
                      ) : null}
                      {latestExperimentPlan.generator_details?.source_run_ids?.length ? (
                        <div className="text-xs text-gray-600">
                          Compared runs: {latestExperimentPlan.generator_details.source_run_ids.slice(0, 4).join(', ')}
                          {latestExperimentPlan.generator_details.source_run_ids.length > 4 ? '…' : ''}
                        </div>
                      ) : null}
                      {latestExperimentPlan.generator_details?.regression_type ? (
                        <div className="text-xs text-gray-600">
                          Regression type: {latestExperimentPlan.generator_details.regression_type}
                        </div>
                      ) : null}
                      {latestExperimentPlan.generator_details?.source_paper_ids?.length ? (
                        <div className="text-xs text-gray-600">
                          Source papers: {latestExperimentPlan.generator_details.source_paper_ids.slice(0, 5).join(', ')}
                          {latestExperimentPlan.generator_details.source_paper_ids.length > 5 ? '…' : ''}
                        </div>
                      ) : null}
                      {latestExperimentPlan.generator_details?.benchmark_suite_name || latestExperimentPlan.benchmark_suite_id ? (
                        <div className="text-xs text-gray-600">
                          Benchmark suite: {latestExperimentPlan.generator_details?.benchmark_suite_name || latestExperimentPlan.benchmark_suite_id}
                          {latestExperimentPlan.benchmark_family ? ` · ${latestExperimentPlan.benchmark_family}` : ''}
                        </div>
                      ) : null}
                      {latestExperimentPlan.generator_details?.benchmark_case_names?.length ? (
                        <div className="text-xs text-gray-600">
                          Benchmark cases: {latestExperimentPlan.generator_details.benchmark_case_names.slice(0, 4).join(', ')}
                          {latestExperimentPlan.generator_details.benchmark_case_names.length > 4 ? '…' : ''}
                        </div>
                      ) : null}
                      {latestExperimentPlan.benchmark_baseline_id ? (
                        <div className="text-xs text-gray-600">
                          Baseline: {latestExperimentPlan.benchmark_baseline_id}
                        </div>
                      ) : null}
                      <div className="max-h-64 overflow-y-auto border border-gray-200 rounded bg-white p-2">
                        <JsonViewer json={latestExperimentPlan.plan} />
                      </div>

                      <details className="border border-gray-200 rounded bg-white p-3">
                        <summary className="cursor-pointer text-sm font-medium text-gray-800">Runner settings (repo + commands)</summary>
                        <div className="mt-3 space-y-3">
                          <label className="block">
                            <div className="text-xs font-medium text-gray-700">Git DocumentSource ID (UUID)</div>
                            <input
                              className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                              placeholder="e.g. 123e4567-e89b-12d3-a456-426614174000"
                              value={experimentSourceId}
                              onChange={(e) => setExperimentSourceId(e.target.value)}
                            />
                            <div className="mt-2">
                              <div className="text-[11px] font-medium text-gray-700">Pick a repo source</div>
                              <input
                                className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
                                placeholder="Search your git sources by name…"
                                value={experimentGitSourceSearch}
                                onChange={(e) => setExperimentGitSourceSearch(e.target.value)}
                              />
                              <select
                                className="mt-2 w-full border border-gray-300 rounded px-2 py-1 text-xs"
                                size={Math.min(6, Math.max(3, (gitSourcesWithStatus || []).length || 3))}
                                value=""
                                onChange={(e) => {
                                  const id = String(e.target.value || '').trim();
                                  if (!id) return;
                                  const match = (gitSourcesWithStatus || []).find((s: any) => String(s?.id) === id);
                                  if (match) selectGitSource(match);
                                  else setExperimentSourceId(id);
                                }}
                              >
                                <option value="" disabled>
                                  {gitSourcesWithStatus.length ? 'Select a source…' : 'No matching sources'}
                                </option>
                                {gitSourcesWithStatus.map((s: any) => (
                                  <option key={String(s.id)} value={String(s.id)}>
                                    {String(s.name)} ({String(s.source_type)}
                                    {s?._active?.pending ? ', pending' : s?.is_syncing ? ', syncing' : ''})
                                  </option>
                                ))}
                              </select>
                              {Array.isArray(recentGitSources) && recentGitSources.length ? (
                                <div className="mt-2">
                                  <div className="text-[11px] font-medium text-gray-700">Recent</div>
                                  <div className="mt-1 flex flex-wrap gap-2">
                                    {recentGitSources.slice(0, 6).map((s: any) => (
                                      <button
                                        key={String(s.id)}
                                        type="button"
                                        className="px-2 py-1 rounded border border-gray-200 bg-white text-[11px] text-gray-700 hover:bg-gray-50 flex items-center gap-1"
                                        onClick={() => selectGitSource(s)}
                                        title={String(
                                          [
                                            s.detail || s.name || '',
                                            activeGitById?.[String(s.id || '').trim()]?.pending ? 'pending' : '',
                                            activeGitById?.[String(s.id || '').trim()]?.is_syncing ? 'syncing' : '',
                                            activeGitById?.[String(s.id || '').trim()]?.task_id
                                              ? `task:${activeGitById[String(s.id || '').trim()].task_id}`
                                              : '',
                                          ]
                                            .filter(Boolean)
                                            .join(' · ')
                                        )}
                                      >
                                        {activeGitById?.[String(s.id || '').trim()]?.pending ? (
                                          <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                                        ) : activeGitById?.[String(s.id || '').trim()]?.is_syncing ? (
                                          <span className="inline-block w-2 h-2 rounded-full bg-blue-400" />
                                        ) : (
                                          <span className="inline-block w-2 h-2 rounded-full bg-gray-200" />
                                        )}
                                        {String(s.name || '').slice(0, 32)}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                              {experimentSourceId.trim() ? (
                                <div className="mt-1 text-[11px] text-gray-600">
                                  Selected: <span className="font-mono">{experimentSourceId.trim()}</span>
                                  {resolvedSelectedGitSourceSummary ? (
                                    <span className="text-gray-500">
                                      {' '}
                                      · {resolvedSelectedGitSourceSummary.name} ({resolvedSelectedGitSourceSummary.source_type}
                                      {resolvedSelectedGitSourceSummary.detail ? ` · ${resolvedSelectedGitSourceSummary.detail}` : ''})
                                    </span>
                                  ) : null}
                                  {selectedGitActiveStatus?.pending ? (
                                    <span className="ml-2 text-[11px] text-amber-700">pending</span>
                                  ) : selectedGitActiveStatus?.is_syncing ? (
                                    <span className="ml-2 text-[11px] text-blue-700">syncing</span>
                                  ) : null}
                                  {selectedGitActiveStatus?.pending || selectedGitActiveStatus?.is_syncing || selectedGitActiveStatus?.task_id ? (
                                    <button
                                      type="button"
                                      className="ml-2 text-[11px] text-primary-600 hover:text-primary-700 underline"
                                      onClick={() =>
                                        navigate('/documents', {
                                          state: { selectedSourceId: experimentSourceId.trim(), selectedSourceTab: 'repos' },
                                        } as any)
                                      }
                                      title={
                                        selectedGitActiveStatus?.task_id
                                          ? `Open Documents → Repos (task ${selectedGitActiveStatus.task_id})`
                                          : 'Open Documents → Repos'
                                      }
                                    >
                                      View ingestion
                                    </button>
                                  ) : null}
                                </div>
                              ) : null}
                            </div>
                            <div className="text-[11px] text-gray-500 mt-1">
                              Uses the existing deterministic <span className="font-mono">experiment_runner</span> job (unsafe exec must be enabled on the server).
                            </div>
                          </label>
                          <label className="block">
                            <div className="text-xs font-medium text-gray-700">Commands (one per line)</div>
                            <textarea
                              className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                              rows={3}
                              value={experimentCommandsText}
                              onChange={(e) => setExperimentCommandsText(e.target.value)}
                            />
                          </label>
                          <div className="border border-gray-200 rounded p-2 bg-gray-50">
                            <div className="text-xs font-medium text-gray-700 mb-2">Stop criteria (loop)</div>
                            <label className="flex items-center gap-2 text-xs text-gray-700">
                              <input
                                type="checkbox"
                                checked={experimentStopOnOk}
                                onChange={(e) => setExperimentStopOnOk(e.target.checked)}
                              />
                              Stop when commands succeed (ok=true)
                            </label>
                            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                              <label className="block">
                                <div className="text-[11px] font-medium text-gray-700">Metric regex (optional)</div>
                                <input
                                  className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                                  placeholder="e.g. accuracy\\s*[:=]\\s*(?P<value>\\d+\\.\\d+)"
                                  value={experimentStopMetricRegex}
                                  onChange={(e) => setExperimentStopMetricRegex(e.target.value)}
                                />
                              </label>
                              <div className="grid grid-cols-3 gap-2">
                                <label className="block">
                                  <div className="text-[11px] font-medium text-gray-700">Direction</div>
                                  <select
                                    className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
                                    value={experimentStopMetricDirection}
                                    onChange={(e) =>
                                      setExperimentStopMetricDirection(
                                        (e.target.value as any) === 'lower_better' ? 'lower_better' : 'higher_better'
                                      )
                                    }
                                  >
                                    <option value="higher_better">higher_better</option>
                                    <option value="lower_better">lower_better</option>
                                  </select>
                                </label>
                                <label className="block">
                                  <div className="text-[11px] font-medium text-gray-700">Window</div>
                                  <input
                                    className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
                                    type="number"
                                    min={2}
                                    max={10}
                                    value={experimentStopMetricWindow}
                                    onChange={(e) =>
                                      setExperimentStopMetricWindow(Math.max(2, Math.min(10, Number(e.target.value) || 2)))
                                    }
                                  />
                                </label>
                                <label className="block">
                                  <div className="text-[11px] font-medium text-gray-700">Min Δ</div>
                                  <input
                                    className="mt-1 w-full border border-gray-300 rounded px-2 py-1 text-xs"
                                    type="number"
                                    step="0.0001"
                                    value={experimentStopMetricMinImprovement}
                                    onChange={(e) => setExperimentStopMetricMinImprovement(Number(e.target.value) || 0)}
                                  />
                                </label>
                              </div>
                            </div>
                            <div className="mt-2 text-[11px] text-gray-500">
                              Metric plateau stop triggers when improvement across the window is &lt; min Δ.
                            </div>
                          </div>
                        </div>
                      </details>

                      <div>
                        <div className="font-medium text-gray-700 mb-1">Runs</div>
                        <div className="flex items-center gap-2 mb-2">
                          <input
                            className="flex-1 border border-gray-300 rounded px-2 py-1 text-xs"
                            placeholder="Run name (e.g., Baseline v1)"
                            value={newExperimentRunName}
                            onChange={(e) => setNewExperimentRunName(e.target.value)}
                          />
                          <Button
                            size="sm"
                            variant="secondary"
                            disabled={createExperimentRunMutation.isLoading || !newExperimentRunName.trim()}
                            onClick={() =>
                              createExperimentRunMutation.mutate({
                                planId: latestExperimentPlan.id,
                                name: newExperimentRunName.trim(),
                                config: {
                                  ...(latestPlanExecutionHandoff ? { execution_handoff: latestPlanExecutionHandoff } : {}),
                                  source_id: experimentSourceId.trim() || undefined,
                                  commands: experimentCommandsText
                                    .split('\n')
                                    .map((s) => s.trim())
                                    .filter(Boolean)
                                    .slice(0, 6),
                                  timeout_seconds: 60,
                                },
                                summary:
                                  typeof latestExperimentPlan.plan?.objective === 'string' && latestExperimentPlan.plan.objective
                                    ? String(latestExperimentPlan.plan.objective)
                                    : undefined,
                              })
                            }
                          >
                            {createExperimentRunMutation.isLoading ? 'Creating…' : 'New run'}
                          </Button>
                        </div>

                        {experimentRuns.length === 0 ? (
                          <div className="text-xs text-gray-500">No runs yet.</div>
                        ) : (
                          <div className="space-y-2">
                            {experimentRuns.slice(0, 10).map((r) => {
                              const experimentRun = (r.experiment_run && typeof r.experiment_run === 'object')
                                ? r.experiment_run
                                : null;
                              const isScientificValidation = String(r.validation_kind || '').trim() === 'scientific_validation';
                              const runLaunchMode = String((r.config as any)?.launch_mode || '').trim();
                              const canRelaunchRecovery =
                                Boolean(r.agent_job_id) &&
                                ['quick_start_claude_backend', 'quick_start_repo_bug_triage', 'quick_start_role_workflow'].includes(runLaunchMode);
                              const {
                                phases,
                                verificationCommands,
                                failedCommands,
                                finalPhase,
                                sourceId,
                                sourceName,
                                detectedStack,
                              } = summarizeExperimentRun(experimentRun);
                              const executionGraph =
                                r.results?.execution_strategy?.execution_graph &&
                                typeof r.results.execution_strategy.execution_graph === 'object'
                                  ? r.results.execution_strategy.execution_graph
                                  : null;
                              const {
                                reasons: graphHealthReasons,
                                recommendedActions: graphRecommendedActions,
                              } = summarizeExperimentRecoveryGuidance(executionGraph);
                              const operatorInterventionSummary = summarizeOperatorInterventions(
                                Array.isArray(r.operator_interventions)
                                  ? (r.operator_interventions as any[])
                                  : Array.isArray(r.results?.execution_strategy?.operator_interventions)
                                    ? (r.results?.execution_strategy?.operator_interventions as any[])
                                    : []
                              );
                              const operatorActions = Array.isArray(r.operator_actions) ? r.operator_actions : [];
                              const latestOperatorAction = operatorActions.length > 0 ? operatorActions[operatorActions.length - 1] : null;
                              const scientificActionNote = experimentRunActionNotes[String(r.id)] || '';
                              const executionHandoff =
                                r.config && typeof r.config === 'object' && (r.config as any).execution_handoff && typeof (r.config as any).execution_handoff === 'object'
                                  ? ((r.config as any).execution_handoff as Record<string, any>)
                                  : null;
                              const postRunActions =
                                r.config && typeof r.config === 'object' && (r.config as any).post_run_actions && typeof (r.config as any).post_run_actions === 'object'
                                  ? ((r.config as any).post_run_actions as Record<string, any>)
                                  : null;
                              const appendStatus = String(postRunActions?.append_status || '').trim().toLowerCase();
                              const canPauseScientific = isScientificValidation && r.agent_job_id && r.status === 'running';
                              const canResumeScientific = isScientificValidation && r.agent_job_id && r.status === 'paused';
                              const canCancelScientific =
                                isScientificValidation && r.agent_job_id && ['queued', 'provisioning', 'running', 'paused'].includes(r.status);
                              const canRetryScientific =
                                isScientificValidation && ['succeeded', 'completed', 'failed', 'blocked', 'cancelled'].includes(r.status);
                              const canRequeueScientific =
                                isScientificValidation && ['planned', 'blocked'].includes(r.status);
                              const canStartScientific =
                                isScientificValidation && !r.agent_job_id && r.status === 'planned';
                              const comparisonRun = findComparisonRun(String(r.id));
                              const recoveryOpen = isExperimentRecoveryOpen(experimentRun, {
                                verificationCommands,
                                bootstrapCommands: [],
                                fallbackCommands: [],
                                phases,
                                failedCommands,
                                finalPhase,
                                sourceId,
                                sourceName,
                                detectedStack,
                              });
                              return (
                                <div
                                  key={r.id}
                                  ref={registerExperimentRunRef(String(r.id))}
                                  role="article"
                                  aria-label={`Experiment run ${r.name}`}
                                  className={`border rounded bg-white p-2 transition-colors ${deepLinkedRunId === String(r.id) ? 'border-primary-400 bg-primary-50/40' : 'border-gray-200'}`}
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <div className="min-w-0">
                                      <div className="font-medium text-gray-900 truncate">{r.name}</div>
                                      <div className="text-xs text-gray-600">
                                        Status: <span className="font-medium">{r.status}</span>
                                        {typeof r.progress === 'number' ? <span className="text-gray-500"> · {r.progress}%</span> : null}
                                      </div>
                                      {executionHandoff?.plan_scope ? (
                                        <div className="mt-1 text-[11px] text-gray-600">
                                          Scope {String(executionHandoff.plan_scope).replaceAll('_', ' ')}
                                        </div>
                                      ) : null}
                                      {Array.isArray(executionHandoff?.selected_hypothesis_ids) && executionHandoff.selected_hypothesis_ids.length > 0 ? (
                                        <div className="mt-1 text-[11px] text-gray-600">
                                          Hypotheses {executionHandoff.selected_hypothesis_ids.join(', ')}
                                        </div>
                                      ) : null}
                                      {Array.isArray(executionHandoff?.source_paper_ids) && executionHandoff.source_paper_ids.length > 0 ? (
                                        <div className="mt-1 text-[11px] text-gray-600">
                                          Papers {executionHandoff.source_paper_ids.slice(0, 3).join(', ')}
                                          {executionHandoff.source_paper_ids.length > 3 ? '…' : ''}
                                        </div>
                                      ) : null}
                                      {r.benchmark_suite_id ? (
                                        <div className="mt-1 text-[11px] text-gray-600">
                                          Benchmark suite {r.benchmark_suite_id}
                                          {r.benchmark_baseline_id ? ` · baseline ${r.benchmark_baseline_id}` : ''}
                                        </div>
                                      ) : null}
                                      {postRunActions?.auto_append_to_note ? (
                                        <div className="mt-1 text-[11px] text-gray-600">
                                          {appendStatus === 'completed'
                                            ? 'Auto-appended to note'
                                            : appendStatus === 'failed'
                                              ? 'Auto-append failed'
                                              : 'Auto-append pending'}
                                          {appendStatus === 'failed' && postRunActions?.append_error ? (
                                            <span className="text-red-700"> · {String(postRunActions.append_error).slice(0, 120)}</span>
                                          ) : null}
                                        </div>
                                      ) : null}
                                      {comparisonRun ? (
                                        <div className="mt-1 text-[11px] text-gray-600">
                                          Comparison run {String(comparisonRun.id)}
                                        </div>
                                      ) : null}
                                      {experimentRun ? (
                                        <div className="mt-2 flex flex-wrap gap-1 text-[11px]">
                                          {isScientificValidation ? (
                                            <span className="px-2 py-0.5 rounded-full bg-violet-100 text-violet-800 border border-violet-200">
                                              Scientific validation
                                            </span>
                                          ) : null}
                                          {r.recipe_family ? (
                                            <span className="px-2 py-0.5 rounded-full bg-fuchsia-50 text-fuchsia-700 border border-fuchsia-200">
                                              Recipe {r.recipe_family}
                                            </span>
                                          ) : null}
                                          {r.recipe_id ? (
                                            <span className="px-2 py-0.5 rounded-full bg-white text-gray-700 border border-gray-200 font-mono">
                                              {r.recipe_id}
                                            </span>
                                          ) : null}
                                          {r.benchmark_family ? (
                                            <span className="px-2 py-0.5 rounded-full bg-lime-50 text-lime-700 border border-lime-200">
                                              Benchmark {r.benchmark_family}
                                            </span>
                                          ) : null}
                                          {r.sandbox_profile_id ? (
                                            <span className="px-2 py-0.5 rounded-full bg-cyan-50 text-cyan-700 border border-cyan-200">
                                              Sandbox {r.sandbox_profile_id}
                                            </span>
                                          ) : null}
                                          {r.blocked_reason_code ? (
                                            <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 border border-rose-200">
                                              Blocked {String(r.blocked_reason_code).replace(/_/g, ' ')}
                                            </span>
                                          ) : null}
                                          {finalPhase ? (
                                            <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                                              Final {finalPhase}
                                            </span>
                                          ) : null}
                                          {Boolean(experimentRun.bootstrap_attempted) ? (
                                            <span className={`px-2 py-0.5 rounded-full border ${experimentRun.bootstrap_ok ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                                              Bootstrap {experimentRun.bootstrap_ok ? 'ok' : 'attempted'}
                                            </span>
                                          ) : null}
                                          {Boolean(experimentRun.fallback_attempted) ? (
                                            <span className={`px-2 py-0.5 rounded-full border ${experimentRun.fallback_ok ? 'bg-indigo-50 text-indigo-700 border-indigo-200' : 'bg-amber-50 text-amber-700 border-amber-200'}`}>
                                              Fallback {experimentRun.fallback_ok ? 'ok' : 'attempted'}
                                            </span>
                                          ) : null}
                                          {recoveryOpen ? (
                                            <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-800 border border-rose-200">
                                              Recovery open
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
                                            <span className="px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 border border-teal-200">
                                              Stack {detectedStack.join(', ')}
                                            </span>
                                          ) : null}
                                          {operatorInterventionSummary.latestLabel ? (
                                            <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-800 border border-amber-200">
                                              Last {operatorInterventionSummary.latestLabel}
                                            </span>
                                          ) : null}
                                          {operatorInterventionSummary.latestOutcome ? (
                                            <span className="px-2 py-0.5 rounded-full bg-orange-50 text-orange-700 border border-orange-100">
                                              Outcome {operatorInterventionSummary.latestOutcome}
                                            </span>
                                          ) : null}
                                        </div>
                                      ) : null}
                                      {operatorInterventionSummary.recentItems.length > 1 ? (
                                        <div className="mt-2 text-[11px] text-amber-800">
                                          <div className="font-medium mb-1">Recent intervention timeline</div>
                                          <ul className="space-y-1">
                                            {operatorInterventionSummary.recentItems.map((item, itemIdx) => (
                                              <li key={`${r.id}-timeline-${itemIdx}`}>- {item}</li>
                                            ))}
                                          </ul>
                                        </div>
                                      ) : null}
                                      <RecoveryAuditPanel
                                        className="mt-2"
                                        textClassName="text-[11px]"
                                        latestAction={operatorInterventionSummary.latestLabel}
                                        latestOutcome={operatorInterventionSummary.latestOutcome}
                                        latestOutcomeReason={operatorInterventionSummary.latestOutcomeReason}
                                        recoveryReason={graphHealthReasons[0]}
                                        nextStep={graphRecommendedActions[0]}
                                      />
                                      {operatorInterventionSummary.latestOutcomeReason ? (
                                        <div className="mt-2 text-[11px] text-orange-700">
                                          <span className="font-medium">Outcome reason:</span> {operatorInterventionSummary.latestOutcomeReason}
                                        </div>
                                      ) : null}
                                      {recoveryOpen && graphHealthReasons.length > 0 ? (
                                        <div className="mt-2 text-[11px] text-rose-700">
                                          <span className="font-medium">Reason:</span> {graphHealthReasons[0]}
                                        </div>
                                      ) : null}
                                      {recoveryOpen && graphRecommendedActions.length > 0 ? (
                                        <div className="mt-1 text-[11px] text-amber-700">
                                          <span className="font-medium">Next:</span> {graphRecommendedActions[0]}
                                        </div>
                                      ) : null}
                                      {isScientificValidation ? (
                                        <details className="mt-2 text-[11px] text-gray-600 border border-gray-200 rounded bg-gray-50 p-2">
                                          <summary className="cursor-pointer font-medium text-gray-700">Scientific validation details</summary>
                                          <div className="mt-2 space-y-1">
                                            {r.capability_check ? (
                                              <div>
                                                Capability check: {r.capability_check.ok ? 'ok' : 'blocked'}
                                                {Array.isArray(r.capability_check.missing) && r.capability_check.missing.length > 0
                                                  ? ` · missing ${r.capability_check.missing.join(', ')}`
                                                  : ''}
                                              </div>
                                            ) : null}
                                            {r.profile_snapshot ? (
                                              <div>
                                                Profile snapshot: {String((r.profile_snapshot as any).name || (r.profile_snapshot as any).id || r.sandbox_profile_id || 'unknown')}
                                              </div>
                                            ) : null}
                                            {r.recipe_snapshot ? (
                                              <div>
                                                Recipe snapshot commands: {Array.isArray((r.recipe_snapshot as any).commands) ? (r.recipe_snapshot as any).commands.slice(0, 2).join(' | ') : 'none'}
                                              </div>
                                            ) : null}
                                            {r.benchmark_suite_id ? (
                                              <div>
                                                Benchmark scope: {r.benchmark_suite_id}
                                                {Array.isArray(r.benchmark_case_ids) && r.benchmark_case_ids.length > 0 ? ` · ${r.benchmark_case_ids.join(', ')}` : ''}
                                              </div>
                                            ) : null}
                                            {summarizeMeasurementSummary(r.measurement_summary || undefined) ? (
                                              <div>
                                                Measurement summary: {summarizeMeasurementSummary(r.measurement_summary || undefined)}
                                              </div>
                                            ) : null}
                                            {summarizeArtifactInventory(r.artifact_inventory || r.compiler_artifacts?.artifact_inventory) ? (
                                              <div>
                                                Artifact inventory: {summarizeArtifactInventory(r.artifact_inventory || r.compiler_artifacts?.artifact_inventory)}
                                              </div>
                                            ) : null}
                                            {summarizeCompilerArtifacts(r.compiler_artifacts || undefined).length > 0 ? (
                                              <div>
                                                Compiler observability: {summarizeCompilerArtifacts(r.compiler_artifacts || undefined).join(' · ')}
                                              </div>
                                            ) : null}
                                            {summarizePerfCounters(r.perf_counters || undefined) ? (
                                              <div>
                                                Perf counters: {summarizePerfCounters(r.perf_counters || undefined)}
                                              </div>
                                            ) : null}
                                            {latestOperatorAction ? (
                                              <div>
                                                Latest action: {String(latestOperatorAction.action || 'unknown')}
                                                {latestOperatorAction.outcome_status ? ` · ${String(latestOperatorAction.outcome_status)}` : ''}
                                              </div>
                                            ) : null}
                                            {typeof r.retry_count === 'number' && r.retry_count > 0 ? (
                                              <div>
                                                Retry lineage: attempt {r.retry_count}
                                                {r.parent_run_id ? ` · parent ${r.parent_run_id}` : ''}
                                              </div>
                                            ) : null}
                                            {r.latest_child_run_id ? (
                                              <div>Latest child run: {String(r.latest_child_run_id)}</div>
                                            ) : null}
                                          </div>
                                        </details>
                                      ) : null}
                                      {isScientificValidation && operatorActions.length > 0 ? (
                                        <div className="mt-2 text-[11px] text-slate-700 border border-slate-200 rounded bg-slate-50 p-2">
                                          <div className="font-medium mb-1">Run control history</div>
                                          <ul className="space-y-1">
                                            {operatorActions.slice(-3).reverse().map((actionRow, idx) => (
                                              <li key={`${r.id}-operator-action-${idx}`}>
                                                {String(actionRow.action || 'unknown')}
                                                {actionRow.new_status ? ` -> ${String(actionRow.new_status)}` : ''}
                                                {actionRow.outcome_status ? ` [${String(actionRow.outcome_status)}]` : ''}
                                                {actionRow.note ? `: ${String(actionRow.note)}` : ''}
                                              </li>
                                            ))}
                                          </ul>
                                        </div>
                                      ) : null}
                                    </div>
                                    <div className="flex items-center gap-1">
                                      {r.agent_job_id ? (
                                        <>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            onClick={() => navigate(`/autonomous-agents?job=${encodeURIComponent(r.agent_job_id as string)}`)}
                                          >
                                            Open job
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            disabled={syncExperimentRunMutation.isLoading || performExperimentRunActionMutation.isLoading}
                                            onClick={() =>
                                              isScientificValidation
                                                ? performExperimentRunActionMutation.mutate({
                                                    runId: r.id,
                                                    action: 'sync',
                                                    note: scientificActionNote,
                                                  })
                                                : syncExperimentRunMutation.mutate({ runId: r.id })
                                            }
                                          >
                                            Sync
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            disabled={appendRunToNoteMutation.isLoading || !r.results || Boolean(selectedNote?.content_markdown?.includes(`<!-- experiment_run:${r.id} -->`))}
                                            onClick={() => appendRunToNoteMutation.mutate({ runId: r.id })}
                                            title="Append a summary of this run into the research note"
                                          >
                                            {selectedNote?.content_markdown?.includes(`<!-- experiment_run:${r.id} -->`) ? "Appended" : "Append"}
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            disabled={explainRegressionMutation.isLoading || !comparisonRun}
                                            onClick={() =>
                                              comparisonRun
                                                ? explainRegressionMutation.mutate({
                                                    primaryRunId: String(r.id),
                                                    comparisonRunId: String(comparisonRun.id),
                                                  })
                                                : undefined
                                            }
                                            title={
                                              comparisonRun
                                                ? `Compare against ${String(comparisonRun.id)} and generate a compiler regression explanation`
                                                : 'Requires an older compatible benchmark-backed run'
                                            }
                                          >
                                            {explainRegressionMutation.isLoading ? 'Explaining…' : 'Explain regression'}
                                          </Button>
                                          {recoveryOpen && !isScientificValidation ? (
                                            <>
                                              <Button
                                                size="sm"
                                                variant="secondary"
                                                disabled={agentJobActionMutation.isLoading}
                                                onClick={() => agentJobActionMutation.mutate({ jobId: String(r.agent_job_id), action: 'restart' })}
                                              >
                                                Restart job
                                              </Button>
                                              {canRelaunchRecovery ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={agentJobActionMutation.isLoading}
                                                  onClick={() => apiClient.performAgentJobAction(String(r.agent_job_id), 'relaunch', {})}
                                                >
                                                  Relaunch clean run
                                                </Button>
                                              ) : null}
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
                                            </>
                                          ) : null}
                                          {isScientificValidation ? (
                                            <>
                                              {canPauseScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'pause',
                                                      note: scientificActionNote,
                                                    })
                                                  }
                                                >
                                                  Pause
                                                </Button>
                                              ) : null}
                                              {canResumeScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'resume',
                                                      note: scientificActionNote,
                                                    })
                                                  }
                                                >
                                                  Resume
                                                </Button>
                                              ) : null}
                                              {canCancelScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="ghost"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'cancel',
                                                      note: scientificActionNote,
                                                    })
                                                  }
                                                >
                                                  Cancel
                                                </Button>
                                              ) : null}
                                              {canRetryScientific && !canCancelScientific && !canPauseScientific && !canResumeScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'retry',
                                                      note: scientificActionNote,
                                                      startImmediately: false,
                                                    })
                                                  }
                                                >
                                                  Retry
                                                </Button>
                                              ) : null}
                                              {canRequeueScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'requeue',
                                                      note: scientificActionNote,
                                                      startImmediately: false,
                                                    })
                                                  }
                                                >
                                                  Requeue
                                                </Button>
                                              ) : null}
                                            </>
                                          ) : null}
                                        </>
                                      ) : (
                                        <>
                                          {isScientificValidation ? (
                                            <>
                                              {canStartScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'start',
                                                      note: scientificActionNote,
                                                    })
                                                  }
                                                  title="Start a recipe-backed scientific validation run"
                                                >
                                                  {performExperimentRunActionMutation.isLoading ? 'Starting…' : 'Start'}
                                                </Button>
                                              ) : null}
                                              {canRetryScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'retry',
                                                      note: scientificActionNote,
                                                      startImmediately: false,
                                                    })
                                                  }
                                                >
                                                  Retry
                                                </Button>
                                              ) : null}
                                              {canRequeueScientific ? (
                                                <Button
                                                  size="sm"
                                                  variant="secondary"
                                                  disabled={performExperimentRunActionMutation.isLoading}
                                                  onClick={() =>
                                                    performExperimentRunActionMutation.mutate({
                                                      runId: r.id,
                                                      action: 'requeue',
                                                      note: scientificActionNote,
                                                      startImmediately: false,
                                                    })
                                                  }
                                                >
                                                  Requeue
                                                </Button>
                                              ) : null}
                                            </>
                                          ) : (
                                            <Button
                                              size="sm"
                                              variant="secondary"
                                              disabled={startExperimentRunMutation.isLoading}
                                              onClick={() => startExperimentRunMutation.mutate({ runId: r.id })}
                                              title="Start a sandboxed runner job for this run"
                                            >
                                              {startExperimentRunMutation.isLoading ? 'Starting…' : 'Run (agent)'}
                                            </Button>
                                          )}
                                        </>
                                      )}
                                      {!isScientificValidation ? (
                                        <>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            disabled={updateExperimentRunMutation.isLoading || r.status === 'running'}
                                            onClick={() => updateExperimentRunMutation.mutate({ runId: r.id, status: 'running' })}
                                          >
                                            Run
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="secondary"
                                            disabled={updateExperimentRunMutation.isLoading || r.status === 'completed'}
                                            onClick={() => updateExperimentRunMutation.mutate({ runId: r.id, status: 'completed' })}
                                          >
                                            Done
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="ghost"
                                            disabled={updateExperimentRunMutation.isLoading || r.status === 'failed'}
                                            onClick={() => updateExperimentRunMutation.mutate({ runId: r.id, status: 'failed' })}
                                          >
                                            Fail
                                          </Button>
                                        </>
                                      ) : null}
                                    </div>
                                  </div>
                                  {isScientificValidation ? (
                                    <div className="mt-2">
                                      <input
                                        className="w-full border border-slate-200 rounded px-2 py-1 text-[11px]"
                                        placeholder="Operator note for pause/cancel/retry/requeue"
                                        value={scientificActionNote}
                                        onChange={(e) =>
                                          setExperimentRunActionNotes((current) => ({
                                            ...current,
                                            [String(r.id)]: e.target.value,
                                          }))
                                        }
                                      />
                                    </div>
                                  ) : null}

                                  {experimentRun && verificationCommands.length > 0 ? (
                                    <div className="mt-2 text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                                      {verificationCommands.slice(0, 3).join('\n')}
                                    </div>
                                  ) : null}

                                  {experimentRun && failedCommands.length > 0 ? (
                                    <div className="mt-2 text-[11px] text-rose-700 font-mono whitespace-pre-wrap">
                                      Failed: {failedCommands.slice(0, 2).join(' | ')}
                                    </div>
                                  ) : null}

                                  {r.results && (
                                    <details className="mt-2">
                                      <summary className="cursor-pointer text-xs text-gray-600">Show details</summary>
                                      <div className="mt-2 border border-gray-200 rounded bg-gray-50 p-2 max-h-56 overflow-y-auto">
                                        <JsonViewer json={r.results} />
                                      </div>
                                    </details>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-gray-500">
                      No experiment plan yet. Generate one to get a datasets/metrics/ablations template and start tracking runs.
                    </div>
                  )}
                </div>

                {selectedNote.attribution && (
                  <div className="mb-3 text-xs text-gray-700 bg-gray-50 rounded-lg p-3">
                    <div className="font-medium text-gray-700 mb-2">Citation report</div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1">
                      {selectedNote.attribution.generated_at && (
                        <div>
                          Generated: {new Date(selectedNote.attribution.generated_at).toLocaleString()}
                        </div>
                      )}
                      {typeof selectedNote.attribution.coverage === 'number' && (
                        <div>
                          Coverage:{' '}
                          {Math.round(
                            (selectedNote.attribution.coverage > 1
                              ? selectedNote.attribution.coverage
                              : selectedNote.attribution.coverage * 100) as number
                          )}
                          %
                        </div>
                      )}
                      {typeof selectedNote.attribution.cited_citable_lines === 'number' &&
                        typeof selectedNote.attribution.total_citable_lines === 'number' && (
                          <div>
                            Cited lines: {selectedNote.attribution.cited_citable_lines}/
                            {selectedNote.attribution.total_citable_lines}
                          </div>
                        )}
                      {Array.isArray(selectedNote.attribution.unsupported_claims) && (
                        <div>
                          Unsupported claims: {selectedNote.attribution.unsupported_claims.length}
                        </div>
                      )}
                      {selectedNote.attribution.strict && <div>Strict: on</div>}
                      {Array.isArray(selectedNote.attribution.unknown_citation_keys) &&
                        selectedNote.attribution.unknown_citation_keys.length > 0 && (
                          <div className="text-red-700">
                            Unknown keys: {selectedNote.attribution.unknown_citation_keys.join(', ')}
                          </div>
                        )}
                    </div>

                    {Array.isArray(selectedNote.attribution.unsupported_claims) &&
                      selectedNote.attribution.unsupported_claims.length > 0 && (
                        <div className="mt-2">
                          <div className="font-medium text-red-700 mb-1">Unsupported</div>
                          <ul className="list-disc pl-5 space-y-1">
                            {selectedNote.attribution.unsupported_claims.slice(0, 10).map((c: any, idx: number) => (
                              <li key={idx} className="text-red-700">
                                {c?.claim || 'Unsupported claim'}
                              </li>
                            ))}
                          </ul>
                          {selectedNote.attribution.unsupported_claims.length > 10 && (
                            <div className="mt-1 text-red-700">…and more</div>
                          )}
                        </div>
                      )}

                    {Array.isArray(selectedNote.attribution.uncited_examples) &&
                      selectedNote.attribution.uncited_examples.length > 0 && (
                        <div className="mt-2">
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <div className="font-medium text-orange-700">Missing citations (examples)</div>
                            {!citationStrict && (
                              <Button
                                size="sm"
                                variant="secondary"
                                disabled={enforceCitationsMutation.isLoading}
                                onClick={() => {
                                  setCitationStrict(true);
                                  enforceCitationsMutation.mutate(selectedNote.id);
                                }}
                              >
                                Run strict
                              </Button>
                            )}
                          </div>
                          <ul className="list-disc pl-5 space-y-1">
                            {selectedNote.attribution.uncited_examples.slice(0, 10).map((c: any) => (
                              <li key={`${c?.line_no}-${c?.line}`} className="text-orange-700">
                                {c?.line_no ? `L${c.line_no}: ` : ''}
                                {c?.line || 'Uncited line'}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                    {Array.isArray(selectedNote.attribution.sources) && selectedNote.attribution.sources.length > 0 && (
                      <div className="mt-2">
                        <div className="font-medium text-gray-700 mb-1">Sources</div>
                        <ul className="list-disc pl-5 space-y-1">
                          {selectedNote.attribution.sources.slice(0, 10).map((s: any) => {
                            const key = s?.key || s?.doc_id;
                            const evidence = Array.isArray(selectedNote.attribution?.evidence)
                              ? (selectedNote.attribution?.evidence as any[])?.find((e: any) => e?.key === s?.key)
                                  ?.evidence
                              : null;
                            const evidenceCount = Array.isArray(evidence) ? evidence.length : 0;
                            const expanded = !!expandedEvidenceKeys[key];
                            return (
                              <li key={key}>
                                <div className="flex items-center justify-between gap-2">
                                  <div className="min-w-0">
                                    {s?.key ? `${s.key}: ` : ''}
                                    {s?.title || s?.doc_id}
                                    {evidenceCount > 0 && (
                                      <span className="ml-2 text-gray-500">({evidenceCount} chunks)</span>
                                    )}
                                  </div>
                                  {evidenceCount > 0 && (
                                    <Button
                                      size="sm"
                                      variant="secondary"
                                      onClick={() =>
                                        setExpandedEvidenceKeys((prev) => ({ ...prev, [key]: !prev[key] }))
                                      }
                                    >
                                      {expanded ? 'Hide' : 'Show'}
                                    </Button>
                                  )}
                                </div>

                                {expanded && Array.isArray(evidence) && evidence.length > 0 && (
                                  <div className="mt-2 space-y-2">
                                    {evidence.slice(0, 6).map((ev: any, idx: number) => (
                                      <div key={`${ev?.chunk_id || idx}`} className="bg-white border border-gray-200 rounded p-2">
                                        <div className="flex items-center justify-between gap-2">
                                          <div className="text-[11px] text-gray-600 flex flex-wrap gap-x-3 gap-y-1 min-w-0">
                                            {typeof ev?.chunk_index === 'number' && <span>chunk #{ev.chunk_index}</span>}
                                            {ev?.chunk_id && <span className="truncate">id: {ev.chunk_id}</span>}
                                            {typeof ev?.score === 'number' && <span>score: {ev.score.toFixed(3)}</span>}
                                          </div>
                                          {s?.doc_id && (
                                            <Button
                                              size="sm"
                                              variant="secondary"
                                              onClick={() =>
                                                navigate('/documents', {
                                                  state: {
                                                    openDocId: s.doc_id,
                                                    selectedDocumentId: s.doc_id,
                                                    highlightChunkId: ev?.chunk_id,
                                                  },
                                                })
                                              }
                                            >
                                              Open
                                            </Button>
                                          )}
                                        </div>
                                        {ev?.excerpt && (
                                          <div className="mt-1 text-[12px] text-gray-700 whitespace-pre-wrap">
                                            {ev.excerpt}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {(selectedNote.attribution as any)?.lint && (
                  <div className="mb-3 text-xs text-gray-700 bg-gray-50 rounded-lg p-3">
                    <div className="font-medium text-gray-700 mb-2">Citation lint</div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1">
                      {(selectedNote.attribution as any).lint.generated_at && (
                        <div>
                          Generated: {new Date((selectedNote.attribution as any).lint.generated_at).toLocaleString()}
                        </div>
                      )}
                      {typeof (selectedNote.attribution as any).lint.cited_citable_lines === 'number' &&
                        typeof (selectedNote.attribution as any).lint.total_citable_lines === 'number' && (
                          <div>
                            Cited lines: {(selectedNote.attribution as any).lint.cited_citable_lines}/
                            {(selectedNote.attribution as any).lint.total_citable_lines}
                          </div>
                        )}
                      {(selectedNote.attribution as any).lint.bibliography_present && <div>Bibliography: yes</div>}
                      {(selectedNote.attribution as any).lint.bibliography_present === false && <div>Bibliography: no</div>}
                      {Array.isArray((selectedNote.attribution as any).lint.unknown_citation_keys) &&
                        (selectedNote.attribution as any).lint.unknown_citation_keys.length > 0 && (
                          <div className="text-red-700">
                            Unknown keys: {(selectedNote.attribution as any).lint.unknown_citation_keys.join(', ')}
                          </div>
                        )}
                    </div>

                    {Array.isArray((selectedNote.attribution as any).lint.uncited_examples) &&
                      (selectedNote.attribution as any).lint.uncited_examples.length > 0 && (
                        <div className="mt-2">
                          <div className="font-medium text-orange-700 mb-1">Missing citations (examples)</div>
                          <ul className="list-disc pl-5 space-y-1">
                            {(selectedNote.attribution as any).lint.uncited_examples.slice(0, 10).map((c: any) => (
                              <li key={`${c?.line_no}-${c?.line}`} className="text-orange-700">
                                {c?.line_no ? `L${c.line_no}: ` : ''}
                                {c?.line || 'Uncited line'}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                  </div>
                )}

                <div className="bg-gray-50 rounded-lg p-3 text-sm">
                  <pre className="whitespace-pre-wrap font-sans text-gray-700">
                    {selectedNote.content_markdown}
                  </pre>
                </div>

                {selectedNote.attribution?.generated_markdown &&
                  !citationUpdateContent &&
                  selectedNote.attribution.generated_markdown !== selectedNote.content_markdown && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <div className="text-xs font-medium text-gray-700">Cited markdown (preview)</div>
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => {
                              navigator.clipboard
                                .writeText(selectedNote.attribution!.generated_markdown as string)
                                .then(() => toast.success('Cited markdown copied'))
                                .catch((e: any) => toast.error(e?.message || 'Copy failed'));
                            }}
                          >
                            <Copy className="w-4 h-4 mr-1" />
                            Copy cited
                          </Button>
                          <Button
                            size="sm"
                            variant="secondary"
                            disabled={applyGeneratedMarkdownMutation.isLoading}
                            onClick={() =>
                              applyGeneratedMarkdownMutation.mutate({
                                noteId: selectedNote.id,
                                content: selectedNote.attribution!.generated_markdown as string,
                              })
                            }
                          >
                            {applyGeneratedMarkdownMutation.isLoading ? 'Applying…' : 'Apply to note'}
                          </Button>
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3 text-sm border border-gray-200">
                        <pre className="whitespace-pre-wrap font-sans text-gray-700">
                          {selectedNote.attribution.generated_markdown}
                        </pre>
                      </div>
                    </div>
                  )}
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 border border-gray-200 rounded-lg h-full flex flex-col items-center justify-center text-gray-500">
              <Eye className="w-10 h-10 mb-3 text-gray-400" />
              <p className="font-medium">Select a note</p>
              <p className="text-sm">Click a note to view details</p>
            </div>
          )}
        </div>
      </div>

      {showCreateModal && <CreateModal />}
    </div>
  );
};

export default ResearchNotesPage;
