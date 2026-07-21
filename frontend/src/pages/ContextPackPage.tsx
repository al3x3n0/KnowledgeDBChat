import React from 'react';
import { useMutation, useQuery } from 'react-query';
import { useNavigate, useParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { ArrowLeft, Copy, Network, Eye, MessageCircle } from 'lucide-react';

import { apiClient } from '../services/api';
import { RetrievalTrace } from '../types';

const extractArxivId = (raw: string): string | null => {
  const v = String(raw || '').trim();
  if (!v) return null;
  let m = v.match(/(?:^|arxiv:)(\d{4}\.\d{4,5}(?:v\d+)?)(?:$)/i);
  if (m?.[1]) return m[1];
  m = v.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/i);
  if (m?.[1]) return m[1];
  m = v.match(/arxiv\.org\/(?:abs|pdf)\/([\w.\-]+\/\d+(?:v\d+)?)(?:\.pdf)?/i);
  if (m?.[1]) return m[1];
  m = v.match(/^([\w.\-]+\/\d+(?:v\d+)?)$/i);
  if (m?.[1]) return m[1];
  return null;
};

const ContextPackPage: React.FC = () => {
  const { traceId } = useParams<{ traceId: string }>();
  const navigate = useNavigate();

  const { data, isLoading, isError, refetch } = useQuery(
    ['retrieval-trace', traceId],
    () => apiClient.getRetrievalTrace(traceId as string),
    { enabled: !!traceId, staleTime: 15000 }
  );

  const trace = data as RetrievalTrace | undefined;
  const pack: any = (trace?.trace as any)?.kg_context_pack || null;
  const kgContext: string | null = (pack?.kg_context as string) || null;

  const entities: any[] = Array.isArray(pack?.entities) ? pack.entities : [];
  const relationships: any[] = Array.isArray(pack?.relationships) ? pack.relationships : [];
  const stats: any = pack?.stats || null;

  const createSessionMutation = useMutation({
    mutationFn: async () => {
      const title = trace?.query ? `Context pack: ${String(trace.query).slice(0, 64)}` : 'Context pack';
      return await apiClient.createChatSession(title, { context_pack_trace_id: traceId });
    },
    onError: (e: any) => {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to create chat session');
    },
  });

  const openEntityInGlobalKG = (name: string, id?: string) => {
    const params = new URLSearchParams();
    if (name) params.set('search', name);
    if (id) params.set('sel', id);
    navigate(`/kg/global?${params.toString()}`);
  };

  const openDocAtEvidence = (documentId: string, chunkId?: string | null) => {
    navigate('/documents', { state: { openDocId: documentId, highlightChunkId: chunkId || undefined } });
  };

  const openDocGraph = (documentId: string) => {
    navigate(`/documents/${encodeURIComponent(documentId)}/graph`);
  };

  const copyText = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(`Copied ${label}`);
    } catch {
      toast.error('Copy failed');
    }
  };

  const maybeArxivId = React.useMemo(() => {
    for (const e of entities) {
      const t = String(e?.type || '').toLowerCase();
      if (t !== 'url') continue;
      const id = extractArxivId(String(e?.name || ''));
      if (id) return id;
    }
    return null;
  }, [entities]);

  return (
    <div className="p-6 h-full min-h-0 flex flex-col gap-4 flex-1">
      <div className="flex items-center justify-between flex-none">
        <button
          className="inline-flex items-center text-sm text-gray-700 hover:text-gray-900"
          onClick={() => navigate(-1)}
        >
          <ArrowLeft className="w-4 h-4 mr-1" /> Back
        </button>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => refetch()}
          >
            Refresh
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
            disabled={!kgContext}
            onClick={() => kgContext && copyText(kgContext, 'KG context')}
            title="Copy the formatted KG context string"
          >
            <Copy className="w-4 h-4" /> Copy Context
          </button>
          <button
            type="button"
            className="inline-flex items-center gap-1 px-3 py-1.5 text-sm rounded bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
            disabled={!traceId || createSessionMutation.isLoading}
            onClick={async () => {
              const session = await createSessionMutation.mutateAsync();
              const prefill = kgContext ? `Context pack:\n${kgContext}\n\nQuestion: ` : 'Question: ';
              navigate(`/chat/${encodeURIComponent(String(session.id))}`, { state: { prefillMessage: prefill } });
            }}
            title="Create a new chat session and prefill the input with this context pack"
          >
            <MessageCircle className="w-4 h-4" />
            {createSessionMutation.isLoading ? 'Starting…' : 'Start Chat'}
          </button>
        </div>
      </div>

      <div className="flex-none">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Retrieval Trace</div>
          <div className="text-lg font-semibold text-gray-900 break-words">{trace?.query || traceId}</div>
          {stats && (
            <div className="mt-2 text-xs text-gray-600">
              Stats: {JSON.stringify(stats)}
            </div>
          )}
          {!pack && (
            <div className="mt-2 text-sm text-gray-600">
              No `kg_context_pack` found in this trace.
            </div>
          )}
          {maybeArxivId && (
            <div className="mt-3 p-3 rounded border border-gray-200 bg-gray-50">
              <div className="text-sm font-medium text-gray-800">arXiv detected</div>
              <div className="text-xs text-gray-600 mt-0.5 break-all">{maybeArxivId}</div>
              <div className="mt-2 flex items-center gap-2 flex-wrap">
                <a
                  href={`https://arxiv.org/abs/${encodeURIComponent(maybeArxivId)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50"
                >
                  Open abstract
                </a>
                <a
                  href={`https://arxiv.org/pdf/${encodeURIComponent(maybeArxivId)}.pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50"
                >
                  Open PDF
                </a>
                <button
                  type="button"
                  className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50"
                  onClick={() => navigate(`/papers?q=${encodeURIComponent(`id:${maybeArxivId}`)}`)}
                >
                  Open in Papers
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {isLoading ? (
        <div className="p-6 text-gray-600">Loading context pack…</div>
      ) : isError ? (
        <div className="p-6 text-red-600">Failed to load retrieval trace.</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden flex flex-col min-h-0">
            <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
              <div className="font-semibold text-gray-900">Entities ({entities.length})</div>
              <button
                type="button"
                className="text-xs text-primary-700 hover:underline disabled:opacity-50"
                disabled={!entities.length}
                onClick={() => copyText(JSON.stringify(entities, null, 2), 'entities JSON')}
              >
                Copy JSON
              </button>
            </div>
            <div className="p-4 overflow-auto min-h-0 space-y-2">
              {entities.length === 0 ? (
                <div className="text-sm text-gray-600">No entities in pack.</div>
              ) : (
                entities.map((e: any) => (
                  <div key={String(e.id)} className="border border-gray-200 rounded p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <button
                          type="button"
                          className="text-sm font-semibold text-primary-700 hover:text-primary-900 hover:underline break-words text-left"
                          onClick={() => openEntityInGlobalKG(String(e.name || ''), String(e.id || ''))}
                          title="Open in Global KG"
                        >
                          {String(e.name || '').trim() || String(e.id || '').slice(0, 8)}
                        </button>
                        <div className="text-xs text-gray-600 mt-0.5">
                          Type: <span className="capitalize">{String(e.type || 'other')}</span>
                          {typeof e.mention_count === 'number' ? ` · ${e.mention_count} mentions` : ''}
                          {typeof e.document_count === 'number' ? ` · ${e.document_count} docs` : ''}
                        </div>
                        {e.description && (
                          <div className="text-xs text-gray-600 mt-1 line-clamp-3">{String(e.description)}</div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          className="p-1 rounded hover:bg-gray-100"
                          title="Copy name"
                          onClick={() => copyText(String(e.name || ''), 'name')}
                        >
                          <Copy className="w-4 h-4 text-gray-600" />
                        </button>
                      </div>
                    </div>
                    {Array.isArray(e.evidence) && e.evidence.length > 0 && (
                      <div className="mt-2 space-y-2">
                        {e.evidence.slice(0, 3).map((ev: any, idx: number) => (
                          <div key={idx} className="text-xs bg-gray-50 border border-gray-200 rounded p-2">
                            <div className="text-gray-700 line-clamp-3">{String(ev?.text || '')}</div>
                            {ev?.document_id && (
                              <div className="mt-2 flex items-center gap-2">
                                <button
                                  type="button"
                                  className="inline-flex items-center gap-1 px-2 py-1 rounded bg-white border border-gray-300 hover:bg-gray-50"
                                  onClick={() => openDocAtEvidence(String(ev.document_id), ev?.chunk_id ? String(ev.chunk_id) : null)}
                                  title="Open document at evidence"
                                >
                                  <Eye className="w-3 h-3" /> Open doc
                                </button>
                                <button
                                  type="button"
                                  className="inline-flex items-center gap-1 px-2 py-1 rounded bg-white border border-gray-300 hover:bg-gray-50"
                                  onClick={() => openDocGraph(String(ev.document_id))}
                                  title="Open document KG"
                                >
                                  <Network className="w-3 h-3" /> Doc graph
                                </button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden flex flex-col min-h-0">
            <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
              <div className="font-semibold text-gray-900">Relationships ({relationships.length})</div>
              <button
                type="button"
                className="text-xs text-primary-700 hover:underline disabled:opacity-50"
                disabled={!relationships.length}
                onClick={() => copyText(JSON.stringify(relationships, null, 2), 'relationships JSON')}
              >
                Copy JSON
              </button>
            </div>
            <div className="p-4 overflow-auto min-h-0 space-y-2">
              {relationships.length === 0 ? (
                <div className="text-sm text-gray-600">No relationships in pack.</div>
              ) : (
                relationships.map((r: any) => {
                  const src = entities.find((e: any) => String(e.id) === String(r.source));
                  const tgt = entities.find((e: any) => String(e.id) === String(r.target));
                  const srcName = String(src?.name || r.source || '').slice(0, 64);
                  const tgtName = String(tgt?.name || r.target || '').slice(0, 64);
                  return (
                    <div key={String(r.id)} className="border border-gray-200 rounded p-3">
                      <div className="text-sm text-gray-900">
                        <button
                          type="button"
                          className="font-mono text-primary-700 hover:text-primary-900 hover:underline"
                          onClick={() => openEntityInGlobalKG(String(src?.name || ''), String(r.source || ''))}
                          title="Open source in Global KG"
                        >
                          {srcName}
                        </button>{' '}
                        <span className="text-gray-500">--[{String(r.type || '')}]--&gt;</span>{' '}
                        <button
                          type="button"
                          className="font-mono text-primary-700 hover:text-primary-900 hover:underline"
                          onClick={() => openEntityInGlobalKG(String(tgt?.name || ''), String(r.target || ''))}
                          title="Open target in Global KG"
                        >
                          {tgtName}
                        </button>
                        {typeof r.confidence === 'number' && (
                          <span className="text-xs text-gray-500"> ({Math.round(r.confidence * 100)}%)</span>
                        )}
                      </div>
                      {r.evidence && (
                        <div className="mt-2 text-xs bg-gray-50 border border-gray-200 rounded p-2 text-gray-700 line-clamp-3">
                          {String(r.evidence)}
                        </div>
                      )}
                      {(r.document_id || r.chunk_id) && (
                        <div className="mt-2 flex items-center gap-2 flex-wrap">
                          {r.document_id && (
                            <button
                              type="button"
                              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-white border border-gray-300 hover:bg-gray-50"
                              onClick={() => openDocAtEvidence(String(r.document_id), r.chunk_id ? String(r.chunk_id) : null)}
                              title="Open document at evidence"
                            >
                              <Eye className="w-3 h-3" /> Open doc
                            </button>
                          )}
                          {r.document_id && (
                            <button
                              type="button"
                              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-white border border-gray-300 hover:bg-gray-50"
                              onClick={() => openDocGraph(String(r.document_id))}
                              title="Open document KG"
                            >
                              <Network className="w-3 h-3" /> Doc graph
                            </button>
                          )}
                          {r.evidence && (
                            <button
                              type="button"
                              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-white border border-gray-300 hover:bg-gray-50"
                              onClick={() => copyText(String(r.evidence), 'evidence')}
                            >
                              <Copy className="w-3 h-3" /> Copy evidence
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ContextPackPage;
