import React from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';
import { ArrowLeft, RefreshCw, Edit2, Trash2 } from 'lucide-react';
import apiClient from '../services/api';
import ForceGraph, { FGNode, FGEdge, ForceGraphHandle } from '../components/kg/ForceGraph';
import RelationshipEditModal from '../components/kg/RelationshipEditModal';
import RelationshipCreateModal from '../components/kg/RelationshipCreateModal';

const DocumentGraphPage: React.FC = () => {
  const { documentId } = useParams<{ documentId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();

  const { data: graph, isLoading, isError, refetch } = useQuery(
    ['kg-document-graph', documentId],
    () => apiClient.getKGDocumentGraph(documentId as string),
    { enabled: !!documentId }
  );

  const rebuildMutation = useMutation(
    () => apiClient.rebuildKGForDocument(documentId as string),
    {
      onSuccess: (res) => {
        toast.success(`Rebuilt KG: ${res.mentions} mentions, ${res.relationships} relations`);
        queryClient.invalidateQueries(['kg-document-graph', documentId]);
        refetch();
      },
      onError: (e: any) => {
        toast.error(e?.response?.data?.detail || e?.message || 'Failed to rebuild graph');
      }
    }
  );

  const width = 960;
  const height = 600;

  const nodes = (graph?.nodes || []) as FGNode[];
  const edges = (graph?.edges || []) as FGEdge[];

  const [selected, setSelected] = React.useState<string | null>(null);
  const selectedNode = React.useMemo(() => nodes.find(n => n.id === selected) || null, [nodes, selected]);
  const neighborEdges = React.useMemo(() => edges.filter(e => e.source === selected || e.target === selected), [edges, selected]);
  const [selectedEdge, setSelectedEdge] = React.useState<FGEdge | null>(null);

  // Relationship edit/delete/create modal state
  const [editModalOpen, setEditModalOpen] = React.useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = React.useState(false);
  const [createModalOpen, setCreateModalOpen] = React.useState(false);

  // Delete relationship mutation
  const deleteRelationshipMutation = useMutation(
    (relId: string) => apiClient.deleteKGRelationship(relId, true),
    {
      onSuccess: () => {
        toast.success('Relationship deleted');
        setSelectedEdge(null);
        setDeleteConfirmOpen(false);
        queryClient.invalidateQueries(['kg-document-graph', documentId]);
        refetch();
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || 'Failed to delete relationship';
        toast.error(message);
      },
    }
  );
  const { data: chunkPreview } = useQuery(
    ['kg-chunk', selectedEdge?.chunk_id, selectedEdge?.evidence],
    () => apiClient.getKGChunk(selectedEdge?.chunk_id as string, selectedEdge?.evidence || undefined),
    { enabled: !!selectedEdge?.chunk_id }
  );
  const [mentionsLimit, setMentionsLimit] = React.useState(10);
  const [mentionsOffset, setMentionsOffset] = React.useState(0);
  const { data: mentionsPage } = useQuery(
    ['kg-mentions', selected, mentionsLimit, mentionsOffset],
    () => apiClient.getKGMentions(selected as string, mentionsLimit, mentionsOffset),
    { enabled: !!selected }
  );
  const mentions = mentionsPage?.items || [];
  const mentionsTotal = mentionsPage?.total || 0;

  // Chunk preview drawer state
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [drawerLoading, setDrawerLoading] = React.useState(false);
  const [drawerPreview, setDrawerPreview] = React.useState<{ content: string; match_start?: number | null; match_end?: number | null; title?: string; docId?: string; chunkIndex?: number } | null>(null);
  const openPreview = async (chunkId?: string | null, evidence?: string | null) => {
    if (!chunkId) return;
    try {
      setDrawerLoading(true);
      const res = await apiClient.getKGChunk(chunkId, evidence || undefined);
      setDrawerPreview({ content: res.content, match_start: res.match_start, match_end: res.match_end, title: res.document_title, docId: res.document_id, chunkIndex: res.chunk_index });
      setDrawerOpen(true);
    } catch (e) {
      toast.error('Failed to load preview');
    } finally {
      setDrawerLoading(false);
    }
  };

  // Filters and search
  const initialTypes = (searchParams.get('types') || 'person,org,email,url,other').split(',').reduce<Record<string, boolean>>((acc, t) => { if (t) acc[t] = true; return acc; }, { person: false, org: false, email: false, url: false, other: false });
  const [entityTypes, setEntityTypes] = React.useState<Record<string, boolean>>({ person: !!initialTypes.person, org: !!initialTypes.org, email: !!initialTypes.email, url: !!initialTypes.url, other: !!initialTypes.other });
  const allRelTypes = React.useMemo(() => Array.from(new Set(edges.map(e => e.type))).sort(), [edges]);
  const [relTypes, setRelTypes] = React.useState<Record<string, boolean>>({});
  React.useEffect(() => {
    // initialize relation types to enabled when edges change
    const init: Record<string, boolean> = {};
    const paramRels = (searchParams.get('rels') || '').split(',');
    allRelTypes.forEach(t => (init[t] = paramRels.length ? paramRels.includes(t) : true));
    setRelTypes(init);
  }, [allRelTypes.length]);

  const [search, setSearch] = React.useState(searchParams.get('q') || '');

  const filteredNodes = React.useMemo(() => {
    const allowedTypes = new Set(Object.entries(entityTypes).filter(([, v]) => v).map(([k]) => k));
    const nameMatch = (n: FGNode) => (search ? n.name.toLowerCase().includes(search.toLowerCase()) : true);
    return nodes.filter(n => allowedTypes.has((n.type || 'other').toLowerCase()) && nameMatch(n));
  }, [nodes, entityTypes, search]);

  const filteredNodeIds = React.useMemo(() => new Set(filteredNodes.map(n => n.id)), [filteredNodes]);
  const filteredEdges = React.useMemo(() => {
    const allowedRel = new Set(Object.entries(relTypes).filter(([, v]) => v).map(([k]) => k));
    return edges.filter(e => allowedRel.has(e.type) && filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target));
  }, [edges, relTypes, filteredNodeIds]);

  const graphRef = React.useRef<ForceGraphHandle>(null);
  React.useEffect(() => {
    // Auto-center when selection changes
    if (selected) {
      graphRef.current?.centerOnNode(selected, 1.1);
    }
  }, [selected]);

  // Sync UI state to URL params
  React.useEffect(() => {
    const types = Object.entries(entityTypes).filter(([, v]) => v).map(([k]) => k).join(',');
    const rels = Object.entries(relTypes).filter(([, v]) => v).map(([k]) => k).join(',');
    const params: any = {};
    if (types) params.types = types;
    if (rels) params.rels = rels;
    if (search) params.q = search;
    if (selected) params.sel = selected;
    setSearchParams(params, { replace: true });
  }, [entityTypes, relTypes, search, selected]);

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <button className="inline-flex items-center text-sm text-gray-700 hover:text-gray-900" onClick={() => navigate('/documents')}>
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to Documents
          </button>
          <h1 className="text-xl font-semibold text-gray-900">Document Knowledge Graph</h1>
        </div>
        <div className="flex items-center space-x-2">
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => refetch()}
          >
            Refresh
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-primary-600 text-white hover:bg-primary-700"
            onClick={() => rebuildMutation.mutate()}
            disabled={rebuildMutation.isLoading}
          >
            <RefreshCw className="w-4 h-4 mr-1" /> Rebuild
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-green-600 text-white hover:bg-green-700"
            onClick={() => setCreateModalOpen(true)}
          >
            + Add Relationship
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => {
              const payload = JSON.stringify({ nodes: filteredNodes, edges: filteredEdges }, null, 2);
              const blob = new Blob([payload], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `document_${documentId}_graph.json`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            }}
          >
            Export JSON
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => graphRef.current?.fitView(60)}
          >
            Fit
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
            onClick={() => selected && graphRef.current?.centerOnNode(selected, 1.2)}
            disabled={!selected}
          >
            Center
          </button>
        </div>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden grid grid-cols-1 lg:grid-cols-3 gap-0">
        <div className="lg:col-span-2">
          {/* Filters */}
          <div className="border-b border-gray-200 p-3 flex flex-wrap items-center gap-3 text-sm">
            <div className="flex items-center gap-2">
              {(['person','org','email','url','other'] as const).map(t => (
                <label key={t} className="inline-flex items-center gap-1">
                  <input type="checkbox" checked={entityTypes[t]} onChange={e => setEntityTypes(s => ({ ...s, [t]: e.target.checked }))} />
                  <span className="capitalize">{t}</span>
                </label>
              ))}
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {allRelTypes.map(t => (
                <label key={t} className="inline-flex items-center gap-1">
                  <input type="checkbox" checked={!!relTypes[t]} onChange={e => setRelTypes(s => ({ ...s, [t]: e.target.checked }))} />
                  <span>{t}</span>
                </label>
              ))}
            </div>
            <div className="ml-auto flex items-center gap-2">
              <input
                className="border border-gray-300 rounded px-2 py-1 text-sm"
                placeholder="Search nodes"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
              <button
                className="inline-flex items-center px-2 py-1 text-sm rounded bg-gray-100 hover:bg-gray-200"
                onClick={() => {
                  const match = filteredNodes.find(n => n.name.toLowerCase().includes(search.toLowerCase()));
                  if (match) setSelected(match.id); else toast.error('No matching node');
                }}
              >
                Go
              </button>
            </div>
          </div>
          {isLoading ? (
            <div className="p-6 text-gray-600">Loading graph…</div>
          ) : isError ? (
            <div className="p-6 text-red-600">Failed to load graph.</div>
          ) : filteredNodes.length === 0 ? (
            <div className="p-6 text-gray-600">No entities found for this document.</div>
          ) : (
            <div className="w-full overflow-auto">
              <ForceGraph
                ref={graphRef}
                width={width}
                height={height}
                nodes={filteredNodes}
                edges={filteredEdges}
                selectedNodeId={selected}
                selectedEdgeId={selectedEdge?.id || null}
                onNodeClick={(n) => { setSelected(n.id); setSelectedEdge(null); }}
                onEdgeClick={(e) => { setSelectedEdge(e); setSelected(null); }}
              />
            </div>
          )}
        </div>
        <div className="border-l border-gray-200 p-4">
          <h2 className="text-base font-semibold text-gray-900 mb-2">Details</h2>
          {selectedNode ? (
            <div>
              <div className="mb-2">
                <div className="text-sm text-gray-500">Selected entity</div>
                <div className="text-lg font-medium text-gray-900">{selectedNode.name}</div>
                <div className="text-xs text-gray-600">Type: {selectedNode.type}</div>
              </div>
              <div className="mt-4">
                <div className="text-sm font-medium text-gray-800 mb-1">Relationships</div>
                {neighborEdges.length === 0 ? (
                  <div className="text-sm text-gray-500">No relationships</div>
                ) : (
                  <ul className="space-y-1 text-sm text-gray-700 max-h-[240px] overflow-auto">
                    {neighborEdges.map(e => {
                      const otherId = e.source === selected ? e.target : e.source;
                      const other = nodes.find(n => n.id === otherId);
                      return (
                        <li key={e.id} className="flex items-center justify-between">
                          <span>
                            <span className="text-gray-500">{e.type}</span>
                            <span className="mx-1">→</span>
                            <button className="text-primary-700 hover:underline" onClick={() => setSelected(otherId)}>
                              {other?.name || otherId}
                            </button>
                          </span>
                          {typeof e.confidence === 'number' && (
                            <span className="text-xs text-gray-500">{(e.confidence * 100).toFixed(0)}%</span>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
              <div className="mt-6">
                <div className="text-sm font-medium text-gray-800 mb-1">Mentions</div>
                {!mentions || mentions.length === 0 ? (
                  <div className="text-sm text-gray-500">No mentions</div>
                ) : (
                  <ul className="space-y-2 text-sm text-gray-700 max-h-[260px] overflow-auto">
                    {mentions.map((m) => (
                      <li key={m.id} className="border rounded p-2 bg-white">
                        <div className="text-xs text-gray-500 mb-1">{m.document_title || m.document_id}</div>
                        <div className="text-gray-800">
                          {m.sentence || m.text}
                        </div>
                        <div className="mt-2 flex items-center gap-2">
                          <button
                            className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200"
                            onClick={() => navigate('/documents', { state: { openDocId: m.document_id, highlightChunkId: m.chunk_id } })}
                          >
                            Open
                          </button>
                          <button
                            className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                            onClick={() => openPreview(m.chunk_id || null, m.sentence || m.text)}
                            disabled={!m.chunk_id}
                          >
                            Preview
                          </button>
                          <button
                            className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200"
                            onClick={async () => {
                              try {
                                await navigator.clipboard.writeText(m.sentence || m.text || '');
                                toast.success('Copied');
                              } catch {
                                toast.error('Copy failed');
                              }
                            }}
                          >
                            Copy
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                <div className="mt-2 flex items-center gap-2">
                  <button
                    className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                    onClick={() => setMentionsOffset(Math.max(0, mentionsOffset - mentionsLimit))}
                    disabled={mentionsOffset === 0}
                  >
                    Prev
                  </button>
                  <button
                    className="px-2 py-1 text-xs rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                    onClick={() => setMentionsOffset(mentionsOffset + mentionsLimit)}
                    disabled={mentionsOffset + mentionsLimit >= mentionsTotal}
                  >
                    Next
                  </button>
                  <select
                    className="ml-auto border border-gray-300 rounded px-2 py-1 text-xs"
                    value={mentionsLimit}
                    onChange={(e) => { setMentionsLimit(parseInt(e.target.value, 10)); setMentionsOffset(0); }}
                  >
                    {[10, 20, 50].map(n => <option key={n} value={n}>{n}/page</option>)}
                  </select>
                  <span className="text-xs text-gray-500">{mentionsOffset + 1}-{Math.min(mentionsOffset + mentionsLimit, mentionsTotal)} of {mentionsTotal}</span>
                </div>
              </div>
            </div>
          ) : selectedEdge ? (
            <div>
              <div className="mb-2">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500">Selected relationship</div>
                  <div className="flex items-center gap-1">
                    <button
                      className="p-1 text-gray-500 hover:text-primary-600 hover:bg-gray-100 rounded"
                      onClick={() => setEditModalOpen(true)}
                      title="Edit relationship"
                    >
                      <Edit2 className="w-4 h-4" />
                    </button>
                    <button
                      className="p-1 text-gray-500 hover:text-red-600 hover:bg-gray-100 rounded"
                      onClick={() => setDeleteConfirmOpen(true)}
                      title="Delete relationship"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                <div className="text-lg font-medium text-gray-900">{selectedEdge.type}</div>
                {typeof selectedEdge.confidence === 'number' && (
                  <div className="text-xs text-gray-600">Confidence: {(selectedEdge.confidence * 100).toFixed(0)}%</div>
                )}
              </div>
              {chunkPreview ? (
                <div className="mt-2 p-2 bg-gray-50 border border-gray-200 rounded text-sm text-gray-800 max-h-56 overflow-auto">
                  {(() => {
                    const c = chunkPreview.content || '';
                    const s = chunkPreview.match_start ?? -1;
                    const e = chunkPreview.match_end ?? -1;
                    if (s >= 0 && e >= 0) {
                      const before = c.slice(0, s);
                      const mid = c.slice(s, e);
                      const after = c.slice(e);
                      return (
                        <span>
                          <span>{before}</span>
                          <mark className="bg-yellow-200 text-black px-0.5 rounded">{mid}</mark>
                          <span>{after}</span>
                        </span>
                      );
                    }
                    return c;
                  })()}
                </div>
              ) : selectedEdge.evidence ? (
                <div className="mt-2 p-2 bg-gray-50 border border-gray-200 rounded text-sm text-gray-800">
                  {selectedEdge.evidence}
                </div>
              ) : null}
              <div className="mt-2 flex items-center gap-2">
                {selectedEdge.evidence && (
                  <button
                    className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
                    onClick={async () => {
                      try {
                        await navigator.clipboard.writeText(selectedEdge.evidence || '');
                        toast.success('Evidence copied');
                      } catch {
                        toast.error('Copy failed');
                      }
                    }}
                  >
                    Copy Evidence
                  </button>
                )}
              </div>
              <div className="mt-3 flex items-center gap-2">
                <button
                  className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
                  onClick={() => navigate('/documents', { state: { openDocId: documentId, highlightChunkId: selectedEdge.chunk_id } })}
                >
                  Open Document
                </button>
                <button
                  className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                  onClick={() => openPreview(selectedEdge.chunk_id || null, selectedEdge.evidence || null)}
                  disabled={!selectedEdge.chunk_id}
                >
                  Preview Chunk
                </button>
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">Select a node or edge to see details.</div>
          )}
        </div>
      </div>

      {/* Relationship Edit Modal */}
      {editModalOpen && selectedEdge && (
        <RelationshipEditModal
          relationshipId={selectedEdge.id}
          onClose={() => setEditModalOpen(false)}
          onSaved={() => {
            queryClient.invalidateQueries(['kg-document-graph', documentId]);
            refetch();
          }}
        />
      )}

      {/* Relationship Create Modal */}
      {createModalOpen && (
        <RelationshipCreateModal
          documentId={documentId}
          onClose={() => setCreateModalOpen(false)}
          onCreated={() => {
            queryClient.invalidateQueries(['kg-document-graph', documentId]);
            refetch();
          }}
        />
      )}

      {/* Delete Confirmation Modal */}
      {deleteConfirmOpen && selectedEdge && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Relationship</h3>
            <p className="text-sm text-gray-600 mb-4">
              Are you sure you want to delete the "{selectedEdge.type}" relationship? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                onClick={() => setDeleteConfirmOpen(false)}
              >
                Cancel
              </button>
              <button
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50"
                onClick={() => deleteRelationshipMutation.mutate(selectedEdge.id)}
                disabled={deleteRelationshipMutation.isLoading}
              >
                {deleteRelationshipMutation.isLoading ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Preview Drawer / Modal */}
      {drawerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setDrawerOpen(false)}>
          <div className="bg-white border border-gray-200 rounded-lg shadow-lg w-full max-w-3xl max-h-[80vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="p-3 border-b flex items-center justify-between">
              <div className="text-sm font-medium text-gray-900">Chunk Preview</div>
              <button className="text-sm text-gray-600 hover:text-gray-900" onClick={() => setDrawerOpen(false)}>Close</button>
            </div>
            <div className="p-4 overflow-auto" style={{ maxHeight: '70vh' }}>
              {drawerLoading ? (
                <div className="text-gray-600">Loading…</div>
              ) : drawerPreview ? (
                <div className="text-sm text-gray-800 whitespace-pre-wrap">
                  {(() => {
                    const c = drawerPreview.content || '';
                    const s = drawerPreview.match_start ?? -1;
                    const e = drawerPreview.match_end ?? -1;
                    if (s >= 0 && e >= 0) {
                      const before = c.slice(0, s);
                      const mid = c.slice(s, e);
                      const after = c.slice(e);
                      return (
                        <span>
                          <span>{before}</span>
                          <mark className="bg-yellow-200 text-black px-0.5 rounded">{mid}</mark>
                          <span>{after}</span>
                        </span>
                      );
                    }
                    return c;
                  })()}
                </div>
              ) : (
                <div className="text-gray-600">No preview</div>
              )}
            </div>
            <div className="p-3 border-t flex items-center gap-2">
              <button
                className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                disabled={!drawerPreview}
                onClick={() => {
                  if (!drawerPreview) return;
                  const c = drawerPreview.content || '';
                  const s = drawerPreview.match_start ?? -1;
                  const e = drawerPreview.match_end ?? -1;
                  // Build snippet with context when possible
                  let snippet = c;
                  if (s >= 0 && e >= 0) {
                    const ctx = 200;
                    const start = Math.max(0, s - ctx);
                    const end = Math.min(c.length, e + ctx);
                    snippet = c.slice(start, end);
                  }
                  const titlePart = (drawerPreview.title || 'document').replace(/[^a-z0-9-_]+/gi, '_');
                  const name = `${titlePart}_chunk_${drawerPreview.chunkIndex ?? 'X'}_snippet.txt`;
                  const blob = new Blob([snippet], { type: 'text/plain;charset=utf-8' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = name;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                }}
              >
                Download Snippet
              </button>
              <div className="ml-auto text-xs text-gray-500">
                {drawerPreview?.title} {typeof drawerPreview?.chunkIndex === 'number' ? `· Chunk ${drawerPreview?.chunkIndex + 1}` : ''}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentGraphPage;
