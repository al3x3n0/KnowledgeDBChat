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
import type { ExperimentPlan, ExperimentRun, ResearchNote } from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import JsonViewer from '../components/common/JsonViewer';

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
  const [recentGitSources, setRecentGitSources] = useState<any[]>([]);
  const EXPERIMENT_LOOP_CHAIN_ID = '9e267663-48d6-4a69-9679-984d1cdf6205';

  const EXPERIMENT_SETTINGS_KEY = useMemo(() => {
    const id = selectedNote?.id ? String(selectedNote.id) : '';
    return id ? `research_note_experiment_settings:${id}` : '';
  }, [selectedNote?.id]);
  const RECENT_GIT_SOURCES_KEY = 'recent_git_sources:v1';

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

  const { data: experimentRunsData, refetch: refetchExperimentRuns } = useQuery(
    ['experiment-runs', latestExperimentPlan?.id],
    () => apiClient.listExperimentRuns(latestExperimentPlan!.id),
    { enabled: !!latestExperimentPlan?.id, staleTime: 5000, refetchInterval: 10000 }
  );

  const experimentRuns: ExperimentRun[] = useMemo(() => {
    return ((experimentRunsData as any)?.runs || []) as ExperimentRun[];
  }, [experimentRunsData]);

  const generateExperimentPlanMutation = useMutation(
    () =>
      apiClient.generateExperimentPlan({
        note_id: selectedNote!.id,
        prefer_section: 'hypothesis',
        max_note_chars: 12000,
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

  const createExperimentRunMutation = useMutation(
    (payload: { planId: string; name: string }) => apiClient.createExperimentRun(payload.planId, { name: payload.name }),
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

  const updateExperimentRunMutation = useMutation(
    (payload: { runId: string; status: ExperimentRun['status'] }) => apiClient.updateExperimentRun(payload.runId, { status: payload.status }),
    {
      onSuccess: () => {
        refetchExperimentRuns();
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
      onSuccess: () => {
        toast.success('Synced run');
        refetchExperimentRuns();
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Sync failed');
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

  const appendRunToNoteMutation = useMutation(
    (payload: { runId: string }) => apiClient.appendExperimentRunToNote(payload.runId),
    {
      onSuccess: (note) => {
        toast.success('Appended results to note');
        setSelectedNote(note);
        queryClient.invalidateQueries(['research-notes']);
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
                        disabled={startExperimentLoopMutation.isLoading || !selectedNote}
                        onClick={() => startExperimentLoopMutation.mutate()}
                        title="Start the autonomous experiment loop playbook (seeded, configurable runs)"
                      >
                        {startExperimentLoopMutation.isLoading ? 'Starting…' : 'Start loop'}
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={generateExperimentPlanMutation.isLoading || !selectedNote}
                        onClick={() => generateExperimentPlanMutation.mutate()}
                        title="Generate a runnable experiment template from the note's Hypothesis section"
                      >
                        {generateExperimentPlanMutation.isLoading ? 'Generating…' : 'Generate plan'}
                      </Button>
                    </div>
                  </div>

                  {latestExperimentPlan ? (
                    <div className="space-y-3">
                      <div className="text-xs text-gray-600">
                        Latest plan: <span className="font-medium text-gray-800">{latestExperimentPlan.title}</span>
                        {latestExperimentPlan.created_at ? (
                          <span className="text-gray-500"> · {new Date(latestExperimentPlan.created_at).toLocaleString()}</span>
                        ) : null}
                      </div>
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
                            {experimentRuns.slice(0, 10).map((r) => (
                              <div key={r.id} className="border border-gray-200 rounded bg-white p-2">
                                <div className="flex items-center justify-between gap-2">
                                  <div className="min-w-0">
                                    <div className="font-medium text-gray-900 truncate">{r.name}</div>
                                    <div className="text-xs text-gray-600">
                                      Status: <span className="font-medium">{r.status}</span>
                                      {typeof r.progress === 'number' ? <span className="text-gray-500"> · {r.progress}%</span> : null}
                                    </div>
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
                                          disabled={syncExperimentRunMutation.isLoading}
                                          onClick={() => syncExperimentRunMutation.mutate({ runId: r.id })}
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
                                  </div>
                                </div>

                                {r.results && (
                                  <details className="mt-2">
                                    <summary className="cursor-pointer text-xs text-gray-600">Show details</summary>
                                    <div className="mt-2 border border-gray-200 rounded bg-gray-50 p-2 max-h-56 overflow-y-auto">
                                      <JsonViewer json={r.results} />
                                    </div>
                                  </details>
                                )}
                              </div>
                            ))}
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
