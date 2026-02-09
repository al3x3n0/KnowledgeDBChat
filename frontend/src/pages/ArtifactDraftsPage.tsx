import React, { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import toast from 'react-hot-toast';
import { CheckCircle2, Download, FileText, RefreshCw, Send, UploadCloud } from 'lucide-react';

import apiClient from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import type { ArtifactDraft, ArtifactDraftListItem, RetrievalTrace } from '../types';

type JsonDiffItem = { path: string; type: 'added' | 'removed' | 'changed' };

const diffJson = (a: any, b: any, opts?: { maxItems?: number; maxDepth?: number }): JsonDiffItem[] => {
  const maxItems = opts?.maxItems ?? 200;
  const maxDepth = opts?.maxDepth ?? 4;

  const diffs: JsonDiffItem[] = [];

  const isObject = (v: any) => typeof v === 'object' && v !== null && !Array.isArray(v);
  const same = (x: any, y: any) => {
    if (x === y) return true;
    try {
      return JSON.stringify(x) === JSON.stringify(y);
    } catch {
      return false;
    }
  };

  const walk = (x: any, y: any, path: string, depth: number) => {
    if (diffs.length >= maxItems) return;
    if (depth > maxDepth) {
      if (!same(x, y)) diffs.push({ path, type: 'changed' });
      return;
    }

    if (x === undefined && y !== undefined) {
      diffs.push({ path, type: 'added' });
      return;
    }
    if (x !== undefined && y === undefined) {
      diffs.push({ path, type: 'removed' });
      return;
    }

    if (isObject(x) && isObject(y)) {
      const keys = new Set([...Object.keys(x), ...Object.keys(y)]);
      for (const k of Array.from(keys).sort()) {
        walk(x[k], y[k], path ? `${path}.${k}` : k, depth + 1);
        if (diffs.length >= maxItems) return;
      }
      return;
    }

    if (Array.isArray(x) && Array.isArray(y)) {
      if (!same(x, y)) diffs.push({ path, type: 'changed' });
      return;
    }

    if (!same(x, y)) diffs.push({ path, type: 'changed' });
  };

  walk(a, b, '', 0);
  return diffs;
};

const ArtifactDraftsPage: React.FC = () => {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';
  const location = useLocation();

  const [items, setItems] = useState<ArtifactDraftListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<ArtifactDraft | null>(null);
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [retrievalTrace, setRetrievalTrace] = useState<RetrievalTrace | null>(null);
  const [retrievalTraceLoading, setRetrievalTraceLoading] = useState(false);

  const [artifactType, setArtifactType] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');

  const canSubmit = useMemo(() => Boolean(selected && selected.status === 'draft'), [selected]);
  const canApprove = useMemo(() => {
    if (!selected) return false;
    const st = String(selected.status || '').toLowerCase();
    if (st === 'published') return false;
    if (isAdmin) return true;
    return selected.user_id === user?.id;
  }, [isAdmin, selected, user?.id]);
  const canPublish = useMemo(() => Boolean(selected && String(selected.status).toLowerCase() === 'approved'), [selected]);
  const canDownload = useMemo(() => Boolean(selected && ['approved', 'published'].includes(String(selected.status).toLowerCase())), [selected]);
  const payloadDiff = useMemo(() => {
    if (!selected) return [];
    const draft = selected.draft_payload || {};
    const published = selected.published_payload || {};
    if (!selected.published_payload) return [];
    return diffJson(published, draft, { maxItems: 200, maxDepth: 4 });
  }, [selected]);

  const refresh = async () => {
    setLoading(true);
    try {
      const res = await apiClient.listArtifactDrafts({
        artifact_type: artifactType.trim() || undefined,
        status_filter: statusFilter.trim() || undefined,
        limit: 100,
        offset: 0,
      });
      setItems(res.items || []);
      setTotal(res.total || 0);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load drafts');
    } finally {
      setLoading(false);
    }
  };

  const loadDetail = async (id: string) => {
    setSelectedLoading(true);
    try {
      const d = await apiClient.getArtifactDraft(id);
      setSelected(d);
      setRetrievalTrace(null);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load draft');
    } finally {
      setSelectedLoading(false);
    }
  };

  const submitSelected = async () => {
    if (!selected) return;
    try {
      const d = await apiClient.submitArtifactDraft(selected.id, {});
      setSelected(d);
      toast.success('Submitted for review');
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Submit failed');
    }
  };

  const approveSelected = async () => {
    if (!selected) return;
    try {
      const d = await apiClient.approveArtifactDraft(selected.id, {});
      setSelected(d);
      toast.success('Approved');
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Approve failed');
    }
  };

  const publishSelected = async () => {
    if (!selected) return;
    try {
      const d = await apiClient.publishArtifactDraft(selected.id);
      setSelected(d);
      toast.success('Published');
      await refresh();
    } catch (e: any) {
      toast.error(e?.message || 'Publish failed');
    }
  };

  const downloadSelected = async () => {
    if (!selected) return;
    try {
      await apiClient.downloadArtifactDraft(selected.id);
    } catch (e: any) {
      toast.error(e?.message || 'Download failed');
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const state = (location.state as any) || {};
    const selectedDraftId = String(state.selectedDraftId || '').trim();
    if (selectedDraftId) {
      loadDetail(selectedDraftId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  useEffect(() => {
    const traceId =
      String((selected as any)?.draft_payload?.sources_used?.retrieval_trace_id || (selected as any)?.draft_payload?.retrieval_trace_id || '').trim();
    if (!traceId) {
      setRetrievalTrace(null);
      return;
    }
    setRetrievalTraceLoading(true);
    apiClient.getRetrievalTrace(traceId)
      .then((t) => setRetrievalTrace(t))
      .catch(() => setRetrievalTrace(null))
      .finally(() => setRetrievalTraceLoading(false));
  }, [selected]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Draft Reviews</h1>
          <p className="text-sm text-gray-700">Human-in-the-loop approvals for presentations and repo reports</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={refresh} disabled={loading}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-700">type</span>
          <select
            value={artifactType}
            onChange={(e) => setArtifactType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm bg-gray-50"
          >
            <option value="">all</option>
            <option value="presentation">presentation</option>
            <option value="repo_report">repo_report</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-700">status</span>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm bg-gray-50"
          >
            <option value="">all</option>
            <option value="draft">draft</option>
            <option value="in_review">in_review</option>
            <option value="approved">approved</option>
            <option value="published">published</option>
            <option value="rejected">rejected</option>
          </select>
        </div>
        <Button variant="secondary" onClick={refresh} disabled={loading}>
          Apply
        </Button>
        {isAdmin && (
          <span className="text-xs text-primary-700 border border-primary-500 rounded px-2 py-1">
            admin view (shows all)
          </span>
        )}
      </div>

      <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-3">
          <div className="bg-gray-100/60 border border-gray-300 rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-300 flex items-center justify-between">
              <div className="text-sm text-gray-800">Drafts ({total})</div>
              {loading && <LoadingSpinner size="sm" />}
            </div>
            <div className="divide-y divide-gray-300">
              {(items || []).map((d) => (
                <button
                  key={d.id}
                  className="w-full text-left px-4 py-3 hover:bg-gray-200/40"
                  onClick={() => loadDetail(d.id)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm text-gray-900 truncate">{d.title}</div>
                    <div className="text-xs text-gray-700">{d.status}</div>
                  </div>
                  <div className="mt-1 text-xs text-gray-600 truncate flex items-center gap-2">
                    <span className="inline-flex items-center gap-1">
                      <FileText className="w-3 h-3" />
                      {d.artifact_type}
                    </span>
                    {d.source_id ? <span className="truncate">{d.source_id}</span> : null}
                  </div>
                </button>
              ))}
              {!loading && (items || []).length === 0 && (
                <div className="px-4 py-6 text-sm text-gray-700">No drafts found.</div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-gray-100/60 border border-gray-300 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-800">Details</div>
            {selectedLoading && <LoadingSpinner size="sm" />}
          </div>

          {!selected && <div className="mt-3 text-sm text-gray-700">Select a draft to view.</div>}

          {selected && (
            <div className="mt-3 space-y-3">
              <div>
                <div className="text-sm text-gray-900 font-semibold">{selected.title}</div>
                <div className="text-xs text-gray-700">{selected.artifact_type} • {selected.status}</div>
                <div className="mt-1 text-xs text-gray-600 break-all">{selected.id}</div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" onClick={submitSelected} disabled={!canSubmit}>
                  <Send className="w-4 h-4 mr-2" />
                  Submit
                </Button>
                <Button variant="secondary" onClick={approveSelected} disabled={!canApprove}>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  Approve
                </Button>
                <Button variant="secondary" onClick={publishSelected} disabled={!canPublish}>
                  <UploadCloud className="w-4 h-4 mr-2" />
                  Publish
                </Button>
                <Button onClick={downloadSelected} disabled={!canDownload}>
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>

              <details className="mt-2">
                <summary className="text-xs text-gray-800 cursor-pointer">payload</summary>
                <pre className="mt-2 text-xs text-gray-900 whitespace-pre-wrap break-words">
                  {JSON.stringify(selected.draft_payload || {}, null, 2)}
                </pre>
              </details>

              <details className="mt-2">
                <summary className="text-xs text-gray-800 cursor-pointer">sources</summary>
                <div className="mt-2 text-xs text-gray-900 space-y-2">
                  {selected.artifact_type === 'presentation' && (
                    <div>
                      <div className="text-xs text-gray-700">source_document_ids</div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {((selected.draft_payload?.sources_used?.source_document_ids || selected.draft_payload?.source_document_ids || []) as any[]).map((x, idx) => (
                          <span key={`${x}-${idx}`} className="px-2 py-0.5 rounded border border-gray-300 bg-gray-50">
                            {String(x)}
                          </span>
                        ))}
                        {!((selected.draft_payload?.sources_used?.source_document_ids || selected.draft_payload?.source_document_ids || []) as any[]).length && (
                          <span className="text-gray-700">—</span>
                        )}
                      </div>
                    </div>
                  )}

                  {retrievalTraceLoading && <div className="text-gray-700">Loading retrieval trace…</div>}
                  {!retrievalTraceLoading && retrievalTrace && (
                    <div>
                      <div className="text-xs text-gray-700">retrieval_trace</div>
                      <div className="mt-1">
                        <div className="text-xs text-gray-600 break-all">{retrievalTrace.id}</div>
                        <div className="mt-1 text-xs text-gray-700">
                          provider: {retrievalTrace.provider || '—'} • mode: {String(
                            (retrievalTrace.trace as any)?.vector_store_trace?.mode ||
                            (retrievalTrace.trace as any)?.single_query?.trace?.mode ||
                            (retrievalTrace.trace as any)?.mode ||
                            '—'
                          )}
                        </div>
                        <div className="mt-2 space-y-1">
                          {(((retrievalTrace.trace as any)?.vector_store_trace?.results_final ||
                            (retrievalTrace.trace as any)?.single_query?.trace?.results_final ||
                            (retrievalTrace.trace as any)?.single_query?.trace?.hybrid_filtered ||
                            (retrievalTrace.trace as any)?.single_query?.trace?.semantic_raw ||
                            (retrievalTrace.trace as any)?.results_final ||
                            []) as any[]).slice(0, 8).map((r, idx) => (
                            <div key={`${r?.id || idx}`} className="border border-gray-200 rounded p-2 bg-gray-50">
                              <div className="flex items-center justify-between gap-2">
                                <div className="truncate">{r?.title || r?.document_id || r?.id}</div>
                                <div className="text-gray-600">score: {r?.score != null ? Number(r.score).toFixed(3) : '—'}</div>
                              </div>
                              <div className="mt-1 text-gray-600 break-all">
                                {r?.source ? `source: ${r.source} • ` : ''}{r?.chunk_id ? `chunk: ${r.chunk_id}` : ''}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {selected.artifact_type === 'repo_report' && (
                    <div className="text-xs text-gray-700">
                      repo: {String(selected.draft_payload?.sources_used?.repo_name || selected.draft_payload?.repo_name || '—')} • {String(selected.draft_payload?.sources_used?.repo_url || selected.draft_payload?.repo_url || '—')}
                    </div>
                  )}
                </div>
              </details>

              <details className="mt-2">
                <summary className="text-xs text-gray-800 cursor-pointer">diff</summary>
                {!selected.published_payload ? (
                  <div className="mt-2 text-xs text-gray-700">No published snapshot yet.</div>
                ) : payloadDiff.length === 0 ? (
                  <div className="mt-2 text-xs text-gray-700">No differences (draft matches published).</div>
                ) : (
                  <div className="mt-2 space-y-1 text-xs">
                    <div className="text-gray-700">Changed paths ({payloadDiff.length}):</div>
                    <div className="max-h-64 overflow-auto border border-gray-200 rounded bg-gray-50 p-2">
                      {payloadDiff.slice(0, 200).map((d) => (
                        <div key={`${d.type}:${d.path}`} className="flex items-center justify-between gap-2">
                          <span className="truncate">{d.path || '(root)'}</span>
                          <span className="text-gray-600">{d.type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </details>

              <details className="mt-2">
                <summary className="text-xs text-gray-800 cursor-pointer">approvals</summary>
                <pre className="mt-2 text-xs text-gray-900 whitespace-pre-wrap break-words">
                  {JSON.stringify(selected.approvals || [], null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ArtifactDraftsPage;
