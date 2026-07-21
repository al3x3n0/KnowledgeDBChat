/**
 * Scientific papers search (arXiv).
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from 'react-query';
import { BrainCircuit, ExternalLink, FileText, FlaskConical, Loader2, Search, StickyNote, Wand2 } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import { ArxivPaper, PaperExtractionJob, ResearchPaper } from '../types';
import ProgressBar from '../components/common/ProgressBar';

const PAGE_SIZE = 10;

interface ArxivImportItem {
  id: string;
  name: string;
  is_syncing: boolean;
  last_error?: string | null;
  last_sync?: string | null;
  created_at?: string | null;
  display?: {
    queries?: string[];
    paper_ids?: string[];
    categories?: string[];
    max_results?: number;
  } | null;
  document_count?: number;
  review_document_id?: string | null;
  review_document_title?: string | null;
}

type IngestionWsEvent =
  | { type: 'progress'; document_id: string; progress: { stage?: string; progress?: number; message?: string } }
  | { type: 'status'; document_id: string; status: { stage?: string; progress?: number; message?: string } }
  | { type: 'complete'; document_id: string; result: any }
  | { type: 'error'; document_id: string; error: string };

const PapersPage: React.FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const highlightSourceId = (searchParams.get('source_id') || '').trim();
  const [query, setQuery] = useState(searchParams.get('q') || 'all:');
  const [submittedQuery, setSubmittedQuery] = useState(searchParams.get('q') || 'all:');
  const [page, setPage] = useState(parseInt(searchParams.get('page') || '1', 10));
  const [sortBy, setSortBy] = useState<'relevance' | 'submittedDate' | 'lastUpdatedDate'>(
    (searchParams.get('sort_by') as any) || 'relevance'
  );
  const [sortOrder, setSortOrder] = useState<'descending' | 'ascending'>(
    (searchParams.get('sort_order') as any) || 'descending'
  );
  const [generateReview, setGenerateReview] = useState(searchParams.get('review') === '1');
  const [reviewTopic, setReviewTopic] = useState(searchParams.get('topic') || '');
  const [reviewTopicDraft, setReviewTopicDraft] = useState(searchParams.get('topic') || '');
  const [reviewTopicDrafts, setReviewTopicDrafts] = useState<Record<string, string>>({});
  const [isTranslating, setIsTranslating] = useState(false);
  const [selectedExtractedPaperId, setSelectedExtractedPaperId] = useState<string | null>(null);
  const [actioningPaperId, setActioningPaperId] = useState<string | null>(null);
  const [selectedHypothesisPaperIds, setSelectedHypothesisPaperIds] = useState<string[]>([]);

  const translateIfNeeded = useCallback(async (rawText: string) => {
    const text = rawText.trim();
    // Heuristic: if user already used arXiv syntax (field:) keep as-is.
    if (/[a-z]{2,5}:/i.test(text)) return text;
    setIsTranslating(true);
    try {
      const res = await apiClient.translateArxivQuery({ text });
      const translated = (res?.query || '').trim();
      if (!translated) throw new Error('Empty translation');
      return translated;
    } finally {
      setIsTranslating(false);
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    const next = query.trim();
    if (/:$/.test(next)) {
      toast.error("Invalid arXiv query: add a term after ':' (e.g. all:transformers)");
      return;
    }
    try {
      const maybeTranslated = await translateIfNeeded(next);
      setQuery(maybeTranslated);
      setSubmittedQuery(maybeTranslated);
      setPage(1);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to translate query');
    }
  }, [query, translateIfNeeded]);

  useEffect(() => {
    const params: Record<string, string> = {};
    if (submittedQuery) params.q = submittedQuery;
    if (page > 1) params.page = String(page);
    if (sortBy !== 'relevance') params.sort_by = sortBy;
    if (sortOrder !== 'descending') params.sort_order = sortOrder;
    if (generateReview) params.review = '1';
    if (reviewTopic.trim()) params.topic = reviewTopic.trim();
    if (highlightSourceId) params.source_id = highlightSourceId;
    setSearchParams(params, { replace: true });
  }, [submittedQuery, page, sortBy, sortOrder, generateReview, reviewTopic, highlightSourceId, setSearchParams]);

  const start = (page - 1) * PAGE_SIZE;

  const { data, isLoading, isFetching, error } = useQuery(
    ['arxivSearch', submittedQuery, start, sortBy, sortOrder],
    () =>
      apiClient.searchArxiv({
        q: submittedQuery,
        start,
        max_results: PAGE_SIZE,
        sort_by: sortBy,
        sort_order: sortOrder,
      }),
    { enabled: submittedQuery.trim().length >= 2 && !/:$/.test(submittedQuery.trim()), keepPreviousData: true, staleTime: 30000 }
  );

  const totalPages = data ? Math.ceil((data.total_results || 0) / PAGE_SIZE) : 0;
  const [ingestingId, setIngestingId] = useState<string | null>(null);
  const [actioningSourceId, setActioningSourceId] = useState<string | null>(null);

  const { data: importsData, isFetching: isImportsFetching } = useQuery(
    ['arxivImports'],
    () => apiClient.listArxivImports({ limit: 10, offset: 0 }),
    { refetchInterval: 10000, staleTime: 5000 }
  );

  const imports: ArxivImportItem[] = useMemo(
    () => (importsData?.items || []) as ArxivImportItem[],
    [importsData?.items]
  );

  const { data: extractedPapersData, isFetching: isExtractedPapersFetching } = useQuery(
    ['research-papers'],
    () => apiClient.listResearchPapers({ limit: 500, offset: 0 }),
    { refetchInterval: 5000, staleTime: 3000 }
  );

  const { data: extractionJobsData } = useQuery(
    ['paper-extraction-jobs'],
    () => apiClient.listPaperExtractionJobs(),
    { refetchInterval: 5000, staleTime: 3000 }
  );

  const extractedPapers = useMemo(() => extractedPapersData?.items || [], [extractedPapersData?.items]);
  const extractionJobs = useMemo(() => extractionJobsData || [], [extractionJobsData]);

  const paperByArxivId = useMemo(() => {
    const map = new Map<string, ResearchPaper>();
    for (const paper of extractedPapers) map.set(paper.arxiv_id, paper);
    return map;
  }, [extractedPapers]);

  const papersBySourceId = useMemo(() => {
    const map = new Map<string, ResearchPaper[]>();
    for (const paper of extractedPapers) {
      const sourceId = String(paper.source_id || '').trim();
      if (!sourceId) continue;
      const current = map.get(sourceId) || [];
      current.push(paper);
      map.set(sourceId, current);
    }
    return map;
  }, [extractedPapers]);

  const latestJobBySourceId = useMemo(() => {
    const map = new Map<string, PaperExtractionJob>();
    for (const job of extractionJobs) {
      const sourceId = String(job.source_id || '').trim();
      if (!sourceId || map.has(sourceId)) continue;
      map.set(sourceId, job);
    }
    return map;
  }, [extractionJobs]);

  const importSourceIdByPaperId = useMemo(() => {
    const map = new Map<string, string>();
    for (const imp of imports) {
      for (const paperId of imp.display?.paper_ids || []) {
        if (!map.has(paperId)) map.set(paperId, imp.id);
      }
    }
    return map;
  }, [imports]);

  const { data: selectedExtractedPaper, isFetching: isSelectedExtractedPaperFetching } = useQuery(
    ['research-paper', selectedExtractedPaperId],
    () => apiClient.getResearchPaper(String(selectedExtractedPaperId)),
    { enabled: Boolean(selectedExtractedPaperId), staleTime: 3000 }
  );

  useEffect(() => {
    if (!highlightSourceId) return;
    if (!imports || imports.length === 0) return;
    const exists = imports.some((x) => String(x.id) === String(highlightSourceId));
    if (!exists) return;

    const el = document.getElementById(`arxiv-import-${highlightSourceId}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightSourceId, imports]);

  const [progressBySourceId, setProgressBySourceId] = useState<Record<string, { stage?: string; progress?: number; message?: string; lastEvent?: string }>>({});

  useEffect(() => {
    const active = imports.filter(i => i.is_syncing).slice(0, 3);
    if (active.length === 0) return;

    const sockets: WebSocket[] = [];
    const cleanupFns: Array<() => void> = [];

    for (const item of active) {
      try {
        const ws = apiClient.createIngestionProgressWebSocket(item.id, { admin: false });
        sockets.push(ws);

        const onMessage = (ev: MessageEvent) => {
          try {
            const msg = JSON.parse(ev.data) as IngestionWsEvent;
            const payload = (msg.type === 'progress' ? msg.progress : msg.type === 'status' ? msg.status : null) as any;
            if (payload && (payload.progress !== undefined || payload.stage || payload.message)) {
              setProgressBySourceId(prev => ({
                ...prev,
                [item.id]: {
                  stage: payload.stage,
                  progress: typeof payload.progress === 'number' ? payload.progress : prev[item.id]?.progress,
                  message: payload.message,
                  lastEvent: msg.type,
                }
              }));
            }
            if (msg.type === 'complete' || msg.type === 'error') {
              queryClient.invalidateQueries('arxivImports');
            }
          } catch {
            // ignore parse errors
          }
        };

        const onError = () => {
          // keep UI quiet; polling will still update
        };

        ws.addEventListener('message', onMessage);
        ws.addEventListener('error', onError);
        cleanupFns.push(() => {
          ws.removeEventListener('message', onMessage);
          ws.removeEventListener('error', onError);
          try { ws.close(); } catch {}
        });
      } catch {
        // ignore websocket init failures (e.g. missing token)
      }
    }

    return () => {
      cleanupFns.forEach(fn => fn());
      sockets.forEach(s => {
        try { if (s.readyState === WebSocket.OPEN) s.close(); } catch {}
      });
    };
  }, [imports, queryClient]);

  const refreshExtractionData = useCallback(() => {
    queryClient.invalidateQueries('research-papers');
    queryClient.invalidateQueries('paper-extraction-jobs');
    queryClient.invalidateQueries('arxivImports');
  }, [queryClient]);

  const queueSourceExtraction = useCallback(async (sourceId: string, force = false) => {
    setActioningSourceId(sourceId);
    try {
      const jobs = await apiClient.extractResearchPapers({ source_id: sourceId, force, limit: 200 });
      toast.success(`Queued ${jobs.length} extraction job${jobs.length === 1 ? '' : 's'}`);
      refreshExtractionData();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to queue extraction');
    } finally {
      setActioningSourceId(null);
    }
  }, [refreshExtractionData]);

  const saveSelectedPaperAsNote = useCallback(async () => {
    if (!selectedExtractedPaper) return;
    setActioningPaperId(selectedExtractedPaper.id);
    try {
      const note = await apiClient.saveResearchPaperAsNote(selectedExtractedPaper.id, {
        title: `Paper Extraction: ${selectedExtractedPaper.title}`,
        tags: ['paper-extraction', 'arxiv'],
      });
      toast.success('Research note created');
      navigate(`/research-notes?note=${encodeURIComponent(note.id)}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to save note');
    } finally {
      setActioningPaperId(null);
    }
  }, [navigate, selectedExtractedPaper]);

  const toggleHypothesisPaper = useCallback((paperId: string) => {
    setSelectedHypothesisPaperIds((current) =>
      current.includes(paperId) ? current.filter((id) => id !== paperId) : [...current, paperId]
    );
  }, []);

  const generateHypothesesFromPapers = useCallback(async () => {
    if (selectedHypothesisPaperIds.length === 0) {
      toast.error('Select at least one extracted paper');
      return;
    }

    const selectedPapers = extractedPapers.filter((paper) => selectedHypothesisPaperIds.includes(paper.id));
    const paperTitles = selectedPapers.map((paper) => paper.title).slice(0, 3);
    const title =
      selectedPapers.length === 1
        ? `Hypotheses: ${selectedPapers[0].title}`
        : `Hypotheses: ${paperTitles.join(', ')}${selectedPapers.length > 3 ? '…' : ''}`;

    setActioningPaperId('generate-hypotheses');
    try {
      const job = await apiClient.createSynthesisJob({
        job_type: 'gap_analysis_hypotheses',
        title,
        document_ids: [],
        paper_ids: selectedHypothesisPaperIds,
        topic: reviewTopic.trim() || submittedQuery.trim() || undefined,
        output_format: 'markdown',
        output_style: 'technical',
        options: {
          domain: 'compilers and microarchitecture',
          desired_outcomes: 'Cross-paper hypotheses and next-step experiments grounded in extracted claims.',
          include_bibliography: true,
        },
      });
      toast.success('Hypothesis job created');
      navigate(`/synthesis?job=${encodeURIComponent(job.id)}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to create hypothesis job');
    } finally {
      setActioningPaperId(null);
    }
  }, [extractedPapers, navigate, reviewTopic, selectedHypothesisPaperIds, submittedQuery]);

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Scientific Papers</h1>
        <p className="text-gray-600">
          Search arXiv using query syntax like <span className="font-mono">all:transformers AND cat:cs.CL</span>
        </p>
      </div>

      {/* Import queue */}
      <div className="mb-6 bg-white rounded-lg border shadow-sm">
        <div className="px-4 py-3 border-b flex items-center justify-between">
          <div>
            <div className="font-semibold text-gray-900">Import Queue</div>
            <div className="text-sm text-gray-600">Recently added arXiv imports into your Knowledge DB</div>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={selectedHypothesisPaperIds.length === 0 || actioningPaperId === 'generate-hypotheses'}
              onClick={() => void generateHypothesesFromPapers()}
              className="text-sm px-3 py-2 rounded-lg bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-50"
              title="Create a gap-analysis hypothesis synthesis from selected extracted papers"
            >
              {actioningPaperId === 'generate-hypotheses' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                `Generate Hypotheses${selectedHypothesisPaperIds.length ? ` (${selectedHypothesisPaperIds.length})` : ''}`
              )}
            </button>
            {highlightSourceId && (
              <div className="flex items-center gap-2">
                <div className="text-xs text-gray-600">
                  Highlighting <span className="font-mono">{highlightSourceId}</span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const next = new URLSearchParams(searchParams);
                    next.delete('source_id');
                    setSearchParams(next, { replace: true });
                  }}
                  className="text-xs px-2 py-1 rounded-lg border border-gray-300 hover:bg-gray-50"
                  title="Clear import highlight"
                >
                  Clear
                </button>
              </div>
            )}
            <div className="text-xs text-gray-500">
              {isImportsFetching ? 'Refreshing imports…' : isExtractedPapersFetching ? 'Refreshing extractions…' : ''}
            </div>
          </div>
        </div>
        <div className="p-4 space-y-3">
          {imports.length === 0 ? (
            <div className="text-sm text-gray-600">No imports yet. Use “Add to DB” on a paper result.</div>
          ) : (
            imports.map((imp) => {
              const status =
                imp.is_syncing ? 'Syncing' :
                imp.last_error ? 'Failed' :
                imp.last_sync ? 'Completed' :
                'Queued';
              const progress = progressBySourceId[imp.id];

              const extractedForSource = papersBySourceId.get(imp.id) || [];
              const isHighlighted = highlightSourceId && String(imp.id) === String(highlightSourceId);
	              return (
	                <div
                    id={`arxiv-import-${imp.id}`}
                    key={imp.id}
                    className={`border rounded-lg p-3 ${isHighlighted ? 'ring-2 ring-primary-400 bg-primary-50' : ''}`}
                  >
	                  <div className="flex items-start justify-between gap-3">
	                    <div className="min-w-0">
	                      <div className="font-medium text-gray-900 truncate">{imp.name}</div>
	                      <div className="text-xs text-gray-500 mt-0.5">
	                        {status}
	                        {imp.created_at ? ` • Created ${imp.created_at.slice(0, 19).replace('T', ' ')}` : ''}
	                        {imp.last_sync ? ` • Last sync ${imp.last_sync.slice(0, 19).replace('T', ' ')}` : ''}
	                      </div>
	                      {imp.display?.paper_ids?.length ? (
	                        <div className="text-xs text-gray-600 mt-1">Paper IDs: {imp.display.paper_ids.slice(0, 4).join(', ')}{imp.display.paper_ids.length > 4 ? '…' : ''}</div>
	                      ) : imp.display?.queries?.length ? (
	                        <div className="text-xs text-gray-600 mt-1">Query: {imp.display.queries[0]}</div>
	                      ) : null}
	                      {typeof imp.document_count === 'number' && (
	                        <div className="text-xs text-gray-500 mt-1">
	                          Documents: {imp.document_count}
	                        </div>
	                      )}
                        {(latestJobBySourceId.get(imp.id) || papersBySourceId.get(imp.id)?.length) ? (
                          <div className="text-xs text-gray-600 mt-1">
                            Extraction: {latestJobBySourceId.get(imp.id)?.status || 'completed'}
                            {papersBySourceId.get(imp.id)?.length ? ` • ${papersBySourceId.get(imp.id)?.length} extracted` : ''}
                          </div>
                        ) : null}
                        {extractedForSource.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {extractedForSource.map((paper) => (
                              <label key={paper.id} className="inline-flex items-center gap-2 text-xs text-gray-700 bg-gray-50 border border-gray-200 rounded-full px-2 py-1">
                                <input
                                  type="checkbox"
                                  checked={selectedHypothesisPaperIds.includes(paper.id)}
                                  onChange={() => toggleHypothesisPaper(paper.id)}
                                  aria-label={`Select paper ${paper.title} for hypothesis generation`}
                                  className="rounded border-gray-300"
                                />
                                <span className="truncate max-w-[220px]">{paper.title}</span>
                              </label>
                            ))}
                          </div>
                        )}
	                    </div>
	                    <div className="flex items-center gap-2">
	                      {imp.review_document_id && (
	                        <button
	                          type="button"
	                          onClick={() => navigate('/documents', { state: { openDocId: imp.review_document_id } })}
	                          className="text-sm px-3 py-2 rounded-lg bg-primary-600 text-white hover:bg-primary-700"
	                          title={imp.review_document_title || 'Open literature review'}
	                        >
	                          Open Review
	                        </button>
	                      )}
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={() => void queueSourceExtraction(imp.id, false)}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Extract structured paper metadata and claims for this import"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Extract Structure'}
                        </button>
                        {(papersBySourceId.get(imp.id) || []).length > 0 && (
                          <button
                            type="button"
                            onClick={() => setSelectedExtractedPaperId((papersBySourceId.get(imp.id) || [])[0].id)}
                            className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50"
                            title="Open extracted paper structure"
                          >
                            Open Extracted
                          </button>
                        )}
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={async () => {
                            setActioningSourceId(imp.id);
                            try {
                              const res = await apiClient.summarizeArxivImport(imp.id, { only_missing: true, force: false, limit: 500 });
                              toast.success(`Queued ${res.queued} summaries`);
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to queue summaries');
                            } finally {
                              setActioningSourceId(null);
                            }
                          }}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Queue summaries for papers in this import"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Summarize All'}
                        </button>
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={async () => {
                            setActioningSourceId(imp.id);
                            try {
                              const topic = String(reviewTopicDrafts[imp.id] || imp.name || '').trim();
                              await apiClient.generateReviewForArxivImport(imp.id, { topic: topic || null });
                              toast.success('Queued literature review');
                              queryClient.invalidateQueries('arxivImports');
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to queue review');
                            } finally {
                              setActioningSourceId(null);
                            }
                          }}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Generate a literature review document for this import"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Generate Review'}
                        </button>
                        <input
                          type="text"
                          value={reviewTopicDrafts[imp.id] ?? imp.name}
                          onChange={(e) => setReviewTopicDrafts((current) => ({ ...current, [imp.id]: e.target.value }))}
                          placeholder="Review topic"
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50"
                        />
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={async () => {
                            setActioningSourceId(imp.id);
                            try {
                              const res = await apiClient.generateSlidesForArxivImport(imp.id, {
                                title: `Slides: ${imp.name}`,
                                topic: imp.name,
                                slide_count: 10,
                                style: 'professional',
                                include_diagrams: true,
                                prefer_review_document: true,
                              });
                              toast.success('Queued slides');
                              navigate('/presentations', { state: { focusJobId: res.presentation_job_id } as any });
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to queue slides');
                            } finally {
                              setActioningSourceId(null);
                            }
                          }}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Generate slides (uses literature review if available)"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Generate Slides'}
                        </button>
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={async () => {
                            setActioningSourceId(imp.id);
                            try {
                              await apiClient.enrichMetadataForArxivImport(imp.id, { force: false, limit: 500 });
                              toast.success('Queued metadata enrichment');
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to queue metadata enrichment');
                            } finally {
                              setActioningSourceId(null);
                            }
                          }}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Fetch BibTeX/DOI metadata (venue, keywords) for this import"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Enrich Metadata'}
                        </button>
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={async () => {
                            setActioningSourceId(imp.id);
                            try {
                              const rl = await apiClient.createReadingList({
                                name: `Reading List: ${imp.name}`,
                                source_id: imp.id,
                                auto_populate_from_source: true,
                              });
                              toast.success('Reading list created');
                              if (rl?.id) navigate(`/reading-lists/${rl.id}`);
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to create reading list');
                            } finally {
                              setActioningSourceId(null);
                            }
                          }}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Create a reading list from this import"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Create Reading List'}
                        </button>
                        <button
                          type="button"
                          disabled={actioningSourceId === imp.id}
                          onClick={async () => {
                            setActioningSourceId(imp.id);
                            try {
                              const draft = await apiClient.synthesizeWorkflow({
                                name: `Import workflow: ${imp.name}`,
                                is_active: false,
                                trigger_config: { type: 'manual' },
                                description:
                                  `Workflow for arXiv import source ${imp.id}.\n` +
                                  `Steps:\n` +
                                  `1) Use tool summarize_documents_in_source with source_id=${imp.id}\n` +
                                  `2) Use tool enrich_arxiv_metadata_for_source with source_id=${imp.id}\n` +
                                  `3) Use tool generate_literature_review_for_source with source_id=${imp.id}\n` +
                                  `4) Use tool generate_slides_for_source with source_id=${imp.id} (prefer_review_document=true)\n` +
                                  `Notes: summarization will also extract paper insights and populate the Knowledge Graph automatically.\n`,
                              });
                              const wf = await apiClient.createWorkflow(draft.workflow);
                              toast.success('Workflow created');
                              if (wf?.id) navigate(`/workflows/${wf.id}/edit`);
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to create workflow');
                            } finally {
                              setActioningSourceId(null);
                            }
                          }}
                          className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                          title="Create a starter workflow for this import"
                        >
                          {actioningSourceId === imp.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Create Workflow'}
                        </button>
	                      <button
	                        type="button"
	                        onClick={() => queryClient.invalidateQueries('arxivImports')}
	                        className="text-sm px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50"
	                      >
	                        Refresh
	                      </button>
	                    </div>
	                  </div>

                  {imp.last_error && (
                    <div className="mt-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
                      {imp.last_error}
                    </div>
                  )}

                  {imp.is_syncing && (
                    <div className="mt-2 space-y-1">
                      <div className="flex items-center justify-between text-xs text-gray-600">
                        <span>{progress?.stage || 'Ingesting…'}</span>
                        <span>{typeof progress?.progress === 'number' ? `${Math.round(progress.progress)}%` : ''}</span>
                      </div>
                      <ProgressBar
                        value={typeof progress?.progress === 'number' ? progress.progress : 30}
                        indeterminate={typeof progress?.progress !== 'number'}
                        size="sm"
                        variant="primary"
                      />
                      {progress?.message && <div className="text-xs text-gray-500">{progress.message}</div>}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                void handleSubmit();
              }
            }}
            placeholder="all:diffusion AND cat:cs.CV"
            className="w-full pl-12 pr-4 py-3 text-lg border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
          {isFetching && (
            <Loader2 className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 animate-spin" />
          )}
        </div>
        <div className="mt-3 flex items-center justify-between gap-3 flex-wrap">
          <button
            onClick={() => void handleSubmit()}
            className="px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={query.trim().length < 2 || isFetching || isTranslating || /:$/.test(query.trim())}
          >
            {isTranslating ? 'Translating…' : 'Search'}
          </button>
          <button
            type="button"
            onClick={async () => {
              const next = query.trim();
              if (!next || next.length < 2) return;
              if (/:$/.test(next)) {
                toast.error("Invalid arXiv query: add a term after ':' (e.g. all:transformers)");
                return;
              }
              try {
                const translated = await translateIfNeeded(next);
                setQuery(translated);
                toast.success('Translated to arXiv syntax');
              } catch (e: any) {
                toast.error(e?.response?.data?.detail || e?.message || 'Failed to translate query');
              }
            }}
            disabled={isTranslating || query.trim().length < 2}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
            title="Translate natural language into arXiv query syntax"
          >
            {isTranslating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
            Translate
          </button>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                checked={generateReview}
                onChange={(e) => setGenerateReview(e.target.checked)}
              />
              <span>Generate review</span>
            </label>
            {generateReview && (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={reviewTopicDraft}
                  onChange={(e) => setReviewTopicDraft(e.target.value)}
                  placeholder="Topic label (optional)"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 min-w-[260px]"
                />
                <button
                  type="button"
                  onClick={() => setReviewTopic(reviewTopicDraft)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50"
                >
                  Apply topic
                </button>
              </div>
            )}
            <span className="text-sm text-gray-600">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500"
            >
              <option value="relevance">Relevance</option>
              <option value="submittedDate">Submitted</option>
              <option value="lastUpdatedDate">Updated</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'descending' ? 'ascending' : 'descending')}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-50"
              title={sortOrder === 'descending' ? 'Descending' : 'Ascending'}
            >
              {sortOrder === 'descending' ? '↓' : '↑'}
            </button>
          </div>
        </div>
      </div>

      {submittedQuery.trim().length < 2 ? (
        <div className="text-center py-16">
          <FileText className="w-16 h-16 mx-auto text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Start searching</h3>
          <p className="text-gray-600">Enter at least 2 characters and click Search</p>
        </div>
      ) : isLoading ? (
        <div className="text-center py-16">
          <Loader2 className="w-8 h-8 mx-auto text-primary-600 animate-spin mb-4" />
          <p className="text-gray-600">Searching arXiv...</p>
        </div>
      ) : error ? (
        <div className="text-center py-16">
          <p className="text-red-600">Failed to search arXiv. Please try again.</p>
        </div>
      ) : data?.items?.length ? (
        <div className="space-y-4">
          <div className="text-sm text-gray-600">
            Showing {start + 1}-{Math.min(start + PAGE_SIZE, data.total_results)} of {data.total_results}
          </div>
          {data.items.map((paper: ArxivPaper) => (
            <div key={paper.entry_url} className="bg-white rounded-lg border shadow-sm p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="font-semibold text-gray-900 truncate">{paper.title}</div>
                  <div className="text-sm text-gray-600 mt-1 truncate">
                    {(paper.authors || []).slice(0, 5).join(', ')}
                    {(paper.authors || []).length > 5 ? ' et al.' : ''}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {paper.primary_category ? `${paper.primary_category} • ` : ''}
                    {paper.published ? `Published ${paper.published.slice(0, 10)} • ` : ''}
                    {paper.updated ? `Updated ${paper.updated.slice(0, 10)}` : ''}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {paperByArxivId.get(paper.id) && (
                    <div className="inline-flex items-center gap-2">
                      <label className="inline-flex items-center gap-1 px-2 py-2 text-sm rounded-lg border border-gray-300 bg-white">
                        <input
                          type="checkbox"
                          checked={selectedHypothesisPaperIds.includes(String(paperByArxivId.get(paper.id)?.id))}
                          onChange={() => toggleHypothesisPaper(String(paperByArxivId.get(paper.id)?.id))}
                          aria-label={`Select paper ${paper.title} for hypothesis generation`}
                          className="rounded border-gray-300"
                        />
                        <span>Select</span>
                      </label>
                      <button
                        type="button"
                        onClick={() => setSelectedExtractedPaperId(String(paperByArxivId.get(paper.id)?.id))}
                        className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50"
                        title="Open extracted paper structure"
                      >
                        <BrainCircuit className="w-4 h-4" />
                        Extracted
                      </button>
                    </div>
                  )}
                  {!paperByArxivId.get(paper.id) && importSourceIdByPaperId.get(paper.id) && (
                    <button
                      type="button"
                      disabled={actioningSourceId === importSourceIdByPaperId.get(paper.id)}
                      onClick={() => void queueSourceExtraction(String(importSourceIdByPaperId.get(paper.id)), false)}
                      className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                      title="Extract structured paper data from the imported document"
                    >
                      <FlaskConical className="w-4 h-4" />
                      Extract
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={async () => {
                      setIngestingId(paper.id);
	                      try {
	                        await apiClient.ingestArxivPapers({
	                          name: `arXiv ${paper.id}`,
	                          paper_ids: [paper.id],
	                          max_results: 1,
	                          start: 0,
	                          sort_by: 'submittedDate',
	                          sort_order: 'descending',
	                          auto_sync: true,
	                          auto_summarize: true,
	                          auto_literature_review: generateReview,
	                          topic: reviewTopic.trim() || submittedQuery.trim() || paper.title,
	                        });
                        toast.success('Queued for ingestion');
                        queryClient.invalidateQueries('arxivImports');
                      } catch (e: any) {
                        toast.error(e?.response?.data?.detail || 'Failed to ingest paper');
                      } finally {
                        setIngestingId(null);
                      }
                    }}
                    disabled={ingestingId === paper.id}
                    className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                    title="Add this paper to the Knowledge DB"
                  >
                    {ingestingId === paper.id ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Add to DB'}
                  </button>
                  {paper.pdf_url && (
                    <a
                      href={paper.pdf_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg bg-primary-600 text-white hover:bg-primary-700"
                    >
                      PDF <ExternalLink className="w-4 h-4" />
                    </a>
                  )}
                  <a
                    href={paper.entry_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50"
                  >
                    arXiv <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>
              {paper.summary && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm text-primary-700 hover:text-primary-800">
                    Abstract
                  </summary>
                  <p className="mt-2 text-sm text-gray-700 whitespace-pre-wrap">{paper.summary}</p>
                </details>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-16">
          <FileText className="w-16 h-16 mx-auto text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No results</h3>
          <p className="text-gray-600">Try a broader query, e.g. <span className="font-mono">all:diffusion</span></p>
        </div>
      )}

      {data && totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-8 pt-6 border-t border-gray-200">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <div className="text-sm text-gray-600">
            Page {page} / {totalPages}
          </div>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}

      {selectedExtractedPaperId && (
        <div className="fixed inset-0 z-50 bg-black/30 flex justify-end">
          <div className="h-full w-full max-w-2xl bg-white shadow-2xl overflow-y-auto">
            <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-start justify-between gap-4">
              <div>
                <div className="text-sm text-gray-500">Extracted paper</div>
                <div className="text-xl font-semibold text-gray-900">
                  {selectedExtractedPaper?.title || 'Loading…'}
                </div>
                {selectedExtractedPaper?.arxiv_id && (
                  <div className="text-sm text-gray-600 mt-1">arXiv ID: {selectedExtractedPaper.arxiv_id}</div>
                )}
              </div>
              <button
                type="button"
                onClick={() => setSelectedExtractedPaperId(null)}
                className="px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50"
              >
                Close
              </button>
            </div>

            <div className="p-6 space-y-6">
              {isSelectedExtractedPaperFetching && !selectedExtractedPaper ? (
                <div className="text-sm text-gray-600 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" /> Loading extracted paper…
                </div>
              ) : selectedExtractedPaper ? (
                <>
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="px-2 py-1 rounded bg-gray-100 text-gray-700 text-xs">
                      {selectedExtractedPaper.extraction_status}
                    </span>
                    {selectedExtractedPaper.latest_job?.status && (
                      <span className="px-2 py-1 rounded bg-blue-50 text-blue-700 text-xs">
                        Job {selectedExtractedPaper.latest_job.status}
                      </span>
                    )}
                    {selectedExtractedPaper.paper_url && (
                      <a
                        href={selectedExtractedPaper.paper_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm text-primary-700 hover:text-primary-800"
                      >
                        Open paper <ExternalLink className="w-4 h-4 inline" />
                      </a>
                    )}
                  </div>

                  {selectedExtractedPaper.summary && (
                    <section>
                      <h3 className="font-semibold text-gray-900 mb-2">Summary</h3>
                      <p className="text-sm text-gray-700 whitespace-pre-wrap">{selectedExtractedPaper.summary}</p>
                    </section>
                  )}

                  {([
                    ['Mechanisms', selectedExtractedPaper.mechanisms],
                    ['Assumptions', selectedExtractedPaper.assumptions],
                    ['Benchmarks', selectedExtractedPaper.benchmarks],
                    ['Metrics', selectedExtractedPaper.metrics],
                    ['Limitations', selectedExtractedPaper.limitations],
                  ] as Array<[string, string[] | null | undefined]>).map(([label, items]) => (
                    items && items.length > 0 ? (
                      <section key={label}>
                        <h3 className="font-semibold text-gray-900 mb-2">{label}</h3>
                        <div className="flex flex-wrap gap-2">
                          {items.map((item) => (
                            <span key={`${label}-${item}`} className="px-2 py-1 rounded-full bg-amber-50 text-amber-900 border border-amber-200 text-xs">
                              {item}
                            </span>
                          ))}
                        </div>
                      </section>
                    ) : null
                  ))}

                  <section>
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <h3 className="font-semibold text-gray-900">Claims</h3>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          disabled={actioningPaperId === selectedExtractedPaper.id}
                          onClick={async () => {
                            setActioningPaperId(selectedExtractedPaper.id);
                            try {
                              await apiClient.reextractResearchPaper(selectedExtractedPaper.id);
                              toast.success('Re-extraction queued');
                              refreshExtractionData();
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || 'Failed to queue re-extraction');
                            } finally {
                              setActioningPaperId(null);
                            }
                          }}
                          className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                        >
                          <FlaskConical className="w-4 h-4" />
                          Re-extract
                        </button>
                        <button
                          type="button"
                          disabled={actioningPaperId === selectedExtractedPaper.id}
                          onClick={() => void saveSelectedPaperAsNote()}
                          className="inline-flex items-center gap-1 px-3 py-2 text-sm rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
                        >
                          <StickyNote className="w-4 h-4" />
                          Save as Note
                        </button>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {selectedExtractedPaper.claims.map((claim) => (
                        <div key={claim.id} className="border rounded-lg p-3">
                          <div className="flex items-start justify-between gap-3">
                            <div className="text-sm text-gray-900">{claim.statement}</div>
                            <div className="text-[11px] text-gray-500 whitespace-nowrap">
                              {claim.kind} • {claim.target_layer}
                              {typeof claim.confidence === 'number' ? ` • ${(claim.confidence * 100).toFixed(0)}%` : ''}
                            </div>
                          </div>
                          {claim.evidence_summary && (
                            <div className="text-xs text-gray-600 mt-2">Evidence: {claim.evidence_summary}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </section>
                </>
              ) : (
                <div className="text-sm text-red-600">Failed to load extracted paper.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PapersPage;
