/**
 * Scientific papers search (arXiv).
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from 'react-query';
import { ExternalLink, FileText, Loader2, Search, Wand2 } from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import { ArxivPaper } from '../types';
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
  const [isTranslating, setIsTranslating] = useState(false);

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
              {isImportsFetching ? 'Refreshing…' : ''}
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
                            const topic = window.prompt('Literature review topic (optional)', imp.name) || '';
                            setActioningSourceId(imp.id);
                            try {
                              await apiClient.generateReviewForArxivImport(imp.id, { topic: topic.trim() || null });
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
              <input
                type="text"
                value={reviewTopic}
                onChange={(e) => setReviewTopic(e.target.value)}
                placeholder="Topic label (optional)"
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500"
              />
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
    </div>
  );
};

export default PapersPage;
