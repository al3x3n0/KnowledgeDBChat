import React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useMutation, useQuery } from 'react-query';
import toast from 'react-hot-toast';
import { ArrowLeft, Network, Filter, Download, ZoomIn, Sparkles } from 'lucide-react';
import apiClient from '../services/api';
import ForceGraph, { FGNode, FGEdge, ForceGraphHandle } from '../components/kg/ForceGraph';
import { useElementSize } from '../hooks/useElementSize';
import { useAuth } from '../contexts/AuthContext';

interface GlobalGraphNode extends FGNode {
  mention_count?: number;
  description?: string;
}

const extractArxivId = (raw: string): string | null => {
  const v = String(raw || '').trim();
  if (!v) return null;
  // new-style: 2401.12345 or 2401.12345v2
  let m = v.match(/(?:^|arxiv:)(\d{4}\.\d{4,5}(?:v\d+)?)(?:$)/i);
  if (m?.[1]) return m[1];
  m = v.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/i);
  if (m?.[1]) return m[1];
  // old-style: cs.CL/0001234
  m = v.match(/arxiv\.org\/(?:abs|pdf)\/([\w.-]+\/\d+(?:v\d+)?)(?:\.pdf)?/i);
  if (m?.[1]) return m[1];
  m = v.match(/^([\w.-]+\/\d+(?:v\d+)?)$/i);
  if (m?.[1]) return m[1];
  return null;
};

const parseCsvParam = (v: string | null): string[] | null => {
  // null => "all" (no filter). '__none__' or '' => "none".
  if (v == null) return null;
  const trimmed = v.trim();
  if (!trimmed || trimmed === '__none__') return [];
  return trimmed
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
};

/**
 * How many of the grid's twelve columns the canvas gets.
 *
 * Stated as arithmetic in one place because the previous version hard-coded
 * two span classes that had to agree with the panels beside them, and did not:
 * the details panel was rendered unconditionally at a quarter of the width, so
 * a canvas with nothing selected still gave that quarter away to a
 * placeholder.
 *
 * Tailwind scans source for literal class names, so these are spelled out
 * rather than interpolated — a computed `lg:col-span-${n}` is not in the
 * stylesheet at runtime.
 */
export function graphColumnSpanClass(showFilters: boolean, detailsOpen: boolean): string {
  const used = (showFilters ? 3 : 0) + (detailsOpen ? 3 : 0);
  switch (12 - used) {
    case 12:
      return 'lg:col-span-12';
    case 9:
      return 'lg:col-span-9';
    case 6:
      return 'lg:col-span-6';
    default:
      return 'lg:col-span-9';
  }
}

const GlobalGraphPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { user } = useAuth();
  const isAdmin = String(user?.role || '').toLowerCase() === 'admin';

  // Filter state
  const [selectedEntityTypes, setSelectedEntityTypes] = React.useState<string[] | null>(() => parseCsvParam(searchParams.get('entity_types')));
  const [selectedRelationTypes, setSelectedRelationTypes] = React.useState<string[] | null>(() => parseCsvParam(searchParams.get('relation_types')));

  const [minConfidence, setMinConfidence] = React.useState(
    parseFloat(searchParams.get('min_confidence') || '0')
  );
  const [minMentions, setMinMentions] = React.useState(
    parseInt(searchParams.get('min_mentions') || '1', 10)
  );
  const [search, setSearch] = React.useState(searchParams.get('search') || '');
  const [limitNodes, setLimitNodes] = React.useState(
    parseInt(searchParams.get('limit_nodes') || '300', 10)
  );
  const [limitEdges, setLimitEdges] = React.useState(
    parseInt(searchParams.get('limit_edges') || '1000', 10)
  );
  const [aiFilter, setAiFilter] = React.useState('');

  // Build query params (do not depend on metadata; metadata is derived from this request).
  const queryParams = React.useMemo(() => {
    return {
      entity_types:
        selectedEntityTypes === null ? undefined : (selectedEntityTypes.length ? selectedEntityTypes.join(',') : '__none__'),
      relation_types:
        selectedRelationTypes === null ? undefined : (selectedRelationTypes.length ? selectedRelationTypes.join(',') : '__none__'),
      min_confidence: minConfidence > 0 ? minConfidence : undefined,
      min_mentions: minMentions > 1 ? minMentions : undefined,
      limit_nodes: limitNodes !== 300 ? limitNodes : undefined,
      limit_edges: limitEdges !== 1000 ? limitEdges : undefined,
      search: search || undefined,
    };
  }, [selectedEntityTypes, selectedRelationTypes, minConfidence, minMentions, limitNodes, limitEdges, search]);

  const { data: graphData, isLoading, isError, refetch } = useQuery(
    ['kg-global-graph', queryParams],
    () => apiClient.getGlobalKGGraph(queryParams),
    {
      keepPreviousData: true,
      staleTime: 30000,
    }
  );

  const { data: kgTypes } = useQuery(['kg-types'], () => apiClient.getKGTypes(), { staleTime: 60000 });

  const nodes = React.useMemo(
    () => (graphData?.nodes || []) as GlobalGraphNode[],
    [graphData?.nodes]
  );
  const edges = React.useMemo(() => (graphData?.edges || []) as FGEdge[], [graphData?.edges]);
  const metadata = graphData?.metadata;

  const availableEntityTypes = React.useMemo(() => {
    const fromServer = kgTypes?.entity_types || [];
    const fromMeta = metadata?.entity_types || [];
    const fromNodes = nodes.map((n) => (n.type || 'other'));
    return Array.from(new Set([...fromServer, ...fromMeta, ...fromNodes].filter(Boolean))).sort();
  }, [kgTypes?.entity_types, metadata?.entity_types, nodes]);

  const availableRelationTypes = React.useMemo(() => {
    const fromServer = kgTypes?.relation_types || [];
    const fromMeta = metadata?.relation_types || [];
    const fromEdges = edges.map((e) => e.type);
    return Array.from(new Set([...fromServer, ...fromMeta, ...fromEdges].filter(Boolean))).sort();
  }, [kgTypes?.relation_types, metadata?.relation_types, edges]);

  const enabledEntityTypes = React.useMemo(() => {
    return selectedEntityTypes === null ? availableEntityTypes : selectedEntityTypes;
  }, [selectedEntityTypes, availableEntityTypes]);

  const enabledRelationTypes = React.useMemo(() => {
    return selectedRelationTypes === null ? availableRelationTypes : selectedRelationTypes;
  }, [selectedRelationTypes, availableRelationTypes]);

  const resolveTypesMutation = useMutation(
    (q: string) =>
      apiClient.resolveKGTypes({
        query: q,
        entity_types: availableEntityTypes,
        relation_types: availableRelationTypes,
      }),
    {
      onSuccess: (res) => {
        setSelectedEntityTypes(res.entity_types ?? null);
        setSelectedRelationTypes(res.relation_types ?? null);
        toast.success('AI filters applied');
      },
      onError: (e: any) => {
        toast.error(e?.response?.data?.detail || e?.message || 'AI filter failed');
      },
    }
  );

  React.useEffect(() => {
    if (selectedEntityTypes === null) return;
    const allowed = new Set(availableEntityTypes);
    setSelectedEntityTypes((prev) => (prev === null ? null : prev.filter((t) => allowed.has(t))));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableEntityTypes.join(',')]);

  React.useEffect(() => {
    if (selectedRelationTypes === null) return;
    const allowed = new Set(availableRelationTypes);
    setSelectedRelationTypes((prev) => (prev === null ? null : prev.filter((t) => allowed.has(t))));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableRelationTypes.join(',')]);

  const [selected, setSelected] = React.useState<string | null>(null);
  // Allow deep-linking to a selected node via `?sel=<entity_id>`.
  React.useEffect(() => {
    const sel = (searchParams.get('sel') || '').trim();
    if (sel && sel !== selected) setSelected(sel);
    if (!sel && selected) setSelected(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);
  const selectedNode = React.useMemo(
    () => nodes.find(n => n.id === selected) || null,
    [nodes, selected]
  );
  const neighborEdges = React.useMemo(
    () => edges.filter(e => e.source === selected || e.target === selected),
    [edges, selected]
  );
  const [selectedEdge, setSelectedEdge] = React.useState<FGEdge | null>(null);
  const [isIngestingArxiv, setIsIngestingArxiv] = React.useState(false);
  const [isQueueingArxiv, setIsQueueingArxiv] = React.useState(false);
  const [aiSuggestedType, setAiSuggestedType] = React.useState<{ entity_type: string; confidence?: number | null } | null>(null);
  const [aiSuggestBusy, setAiSuggestBusy] = React.useState(false);
  const [aiApplyBusy, setAiApplyBusy] = React.useState(false);

  React.useEffect(() => {
    setAiSuggestedType(null);
  }, [selected]);

  const graphRef = React.useRef<ForceGraphHandle>(null);


  const graphBox = useElementSize<HTMLDivElement>();
  const width = Math.max(320, graphBox.width || 0);
  const height = Math.max(420, graphBox.height || 0);

  // Sync URL params
  React.useEffect(() => {
    const params: Record<string, string> = {};
    if (selectedEntityTypes !== null) {
      params.entity_types = selectedEntityTypes.length ? selectedEntityTypes.join(',') : '__none__';
    }
    if (selectedRelationTypes !== null) {
      params.relation_types = selectedRelationTypes.length ? selectedRelationTypes.join(',') : '__none__';
    }
    if (minConfidence > 0) params.min_confidence = String(minConfidence);
    if (minMentions > 1) params.min_mentions = String(minMentions);
    if (limitNodes !== 300) params.limit_nodes = String(limitNodes);
    if (limitEdges !== 1000) params.limit_edges = String(limitEdges);
    if (search) params.search = search;
    if (selected) params.sel = selected;

    setSearchParams(params, { replace: true });
  }, [selectedEntityTypes, selectedRelationTypes, minConfidence, minMentions, limitNodes, limitEdges, search, selected, setSearchParams]);

  const [showFilters, setShowFilters] = React.useState(true);

  // The canvas takes whatever the two side panels are not using.
  const detailsOpen = Boolean(selectedNode || selectedEdge);
  const graphColSpanClass = graphColumnSpanClass(showFilters, detailsOpen);

  return (
    <div className="p-6 h-full min-h-0 flex flex-col gap-4 flex-1">
      <div className="flex items-center justify-between flex-none">
        <div className="flex items-center space-x-2">
          <button
            className="inline-flex items-center text-sm text-gray-700 hover:text-gray-900"
            onClick={() => navigate('/documents')}
          >
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </button>
          <Network className="w-5 h-5 text-primary-600" />
          <h1 className="text-xl font-semibold text-gray-900">Global Knowledge Graph</h1>
        </div>
        <div className="flex items-center space-x-2">
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => setShowFilters(!showFilters)}
          >
            <Filter className="w-4 h-4 mr-1" />
            {showFilters ? 'Hide Filters' : 'Show Filters'}
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => refetch()}
          >
            Refresh
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => {
              const payload = JSON.stringify({ nodes, edges, metadata }, null, 2);
              const blob = new Blob([payload], { type: 'application/json' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `global_knowledge_graph.json`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            }}
          >
            <Download className="w-4 h-4 mr-1" /> Export
          </button>
          <button
            className="inline-flex items-center px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
            onClick={() => graphRef.current?.fitView(60)}
          >
            <ZoomIn className="w-4 h-4 mr-1" /> Fit
          </button>
        </div>
      </div>

      {/* Metadata bar */}
      <div className="flex-none">
        {metadata && (
          <div className="flex items-center gap-4 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
            <span>Total Entities: <strong>{metadata.total_entities}</strong></span>
            <span>Total Relationships: <strong>{metadata.total_relationships}</strong></span>
            <span>Showing: <strong>{metadata.filtered_nodes}</strong> nodes, <strong>{metadata.filtered_edges}</strong> edges</span>
          </div>
        )}
      </div>

      {/* Twelve columns rather than four, so the canvas can take whatever the
          panels are not using instead of being pinned to a quarter of the
          width. `lg:grid-rows-1` gives the single row the container's full
          height; below that breakpoint the children stack, which is why the
          canvas carries its own min-height — without one it collapsed to zero
          and the graph disappeared on a narrow window. */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden grid grid-cols-1 lg:grid-cols-12 lg:grid-rows-1 gap-0 flex-1 min-h-0">
        {/* Filters Panel */}
        {showFilters && (
          <div className="border-r border-gray-200 p-4 lg:col-span-3 space-y-4 overflow-auto h-full min-h-0">
            <div>
              <div className="flex items-center justify-between gap-2 mb-2">
                <h3 className="text-sm font-medium text-gray-800 inline-flex items-center gap-1">
                  <Sparkles className="w-4 h-4 text-primary-600" />
                  AI Filter
                </h3>
                <button
                  type="button"
                  className="text-xs text-primary-700 hover:underline disabled:opacity-50"
                  disabled={!aiFilter.trim()}
                  onClick={() => setAiFilter('')}
                  title="Clear AI filter"
                >
                  Clear
                </button>
              </div>

              <textarea
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-200 focus:border-primary-500 placeholder:text-gray-400 resize-none"
                placeholder="Describe what you want to see. Example: show companies and acquisitions"
                rows={2}
                value={aiFilter}
                onChange={e => setAiFilter(e.target.value)}
                onKeyDown={(e) => {
                  // Enter applies; Shift+Enter inserts a newline.
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const q = aiFilter.trim();
                    if (q && !resolveTypesMutation.isLoading) resolveTypesMutation.mutate(q);
                  }
                }}
              />

              <div className="mt-2 flex items-center gap-2">
                <button
                  className="flex-1 px-3 py-2 text-sm rounded-md bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={!aiFilter.trim() || resolveTypesMutation.isLoading}
                  onClick={() => resolveTypesMutation.mutate(aiFilter.trim())}
                >
                  {resolveTypesMutation.isLoading ? 'Applying…' : 'Apply'}
                </button>
              </div>

              <div className="mt-1 text-xs text-gray-500">
                Picks relevant entity and relationship types from the KG. Press Enter to apply.
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">Search</h3>
              <input
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                placeholder="Search entities..."
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">Entity Types</h3>
              <div className="space-y-1">
                {availableEntityTypes.map(t => (
                  <label key={t} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={enabledEntityTypes.includes(t)}
                      onChange={e => {
                        const checked = e.target.checked;
                        setSelectedEntityTypes((prev) => {
                          const base = prev === null ? availableEntityTypes.slice() : prev.slice();
                          const set = new Set(base);
                          if (checked) set.add(t);
                          else set.delete(t);
                          const next = Array.from(set);
                          next.sort();
                          if (next.length === availableEntityTypes.length) return null;
                          return next;
                        });
                      }}
                    />
                    <span className="capitalize">{t}</span>
                  </label>
                ))}
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    setSelectedEntityTypes(null);
                  }}
                >
                  Select All
                </button>
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    setSelectedEntityTypes([]);
                  }}
                >
                  Clear
                </button>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">Relation Types</h3>
              <div className="space-y-1 max-h-40 overflow-auto">
                {availableRelationTypes.map(t => (
                  <label key={t} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={enabledRelationTypes.includes(t)}
                      onChange={e => {
                        const checked = e.target.checked;
                        setSelectedRelationTypes((prev) => {
                          const base = prev === null ? availableRelationTypes.slice() : prev.slice();
                          const set = new Set(base);
                          if (checked) set.add(t);
                          else set.delete(t);
                          const next = Array.from(set);
                          next.sort();
                          if (next.length === availableRelationTypes.length) return null;
                          return next;
                        });
                      }}
                    />
                    <span>{t.replace(/_/g, ' ')}</span>
                  </label>
                ))}
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    setSelectedRelationTypes(null);
                  }}
                >
                  Select All
                </button>
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    setSelectedRelationTypes([]);
                  }}
                >
                  Clear
                </button>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">
                Min Confidence: {(minConfidence * 100).toFixed(0)}%
              </h3>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={minConfidence}
                onChange={e => setMinConfidence(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">
                Min Mentions: {minMentions}
              </h3>
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={minMentions}
                onChange={e => setMinMentions(parseInt(e.target.value, 10))}
                className="w-full"
              />
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">Limits</h3>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm">
                  <span className="w-16">Nodes:</span>
                  <select
                    className="border border-gray-300 rounded px-2 py-1 text-sm"
                    value={limitNodes}
                    onChange={e => setLimitNodes(parseInt(e.target.value, 10))}
                  >
                    {[100, 200, 300, 500, 1000].map(n => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <span className="w-16">Edges:</span>
                  <select
                    className="border border-gray-300 rounded px-2 py-1 text-sm"
                    value={limitEdges}
                    onChange={e => setLimitEdges(parseInt(e.target.value, 10))}
                  >
                    {[500, 1000, 2000, 5000].map(n => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </label>
              </div>
            </div>

            <button
              className="w-full px-3 py-2 text-sm rounded bg-primary-600 text-white hover:bg-primary-700"
              onClick={() => refetch()}
            >
              Apply Filters
            </button>
          </div>
        )}

        {/* Graph Area */}
        <div className={`${graphColSpanClass} h-full min-h-[420px] lg:min-h-0 flex flex-col`}>
          {isLoading ? (
            <div className="p-6 text-gray-600 flex-1">Loading graph...</div>
          ) : isError ? (
            <div className="p-6 text-red-600 flex-1">Failed to load graph.</div>
          ) : nodes.length === 0 ? (
            <div className="p-6 text-gray-600 flex-1">No entities found matching the filters.</div>
          ) : (
            <div ref={graphBox.ref} className="w-full flex-1 min-h-0 overflow-hidden">
              {width > 0 && height > 0 && (
                <ForceGraph
                  ref={graphRef}
                  width={width}
                  height={height}
                  nodes={nodes}
                  edges={edges}
                  selectedNodeId={selected}
                  selectedEdgeId={selectedEdge?.id || null}
                  onBackgroundClick={() => { setSelected(null); setSelectedEdge(null); }}
                  onNodeClick={n => { setSelected(n.id); setSelectedEdge(null); }}
                  onEdgeClick={e => { setSelectedEdge(e); setSelected(null); }}
                />
              )}
            </div>
          )}
        </div>

        {/* Details Panel. Rendered only when there is something to show: an
            empty panel used to hold a quarter of the width open for a
            placeholder, which is the reason the canvas never filled the page. */}
        {detailsOpen && (
        <div className="border-l border-gray-200 p-4 lg:col-span-3 overflow-auto h-full min-h-0">
          <h2 className="text-base font-semibold text-gray-900 mb-2">Details</h2>
          {selectedNode ? (
            <div>
              <div className="mb-3">
                <div className="text-sm text-gray-500">Selected Entity</div>
                {(() => {
                  const t = String(selectedNode.type || '').toLowerCase();
                  const name = String(selectedNode.name || '');
                  const isUrl = t === 'url' || /^https?:\/\//i.test(name);
                  if (!isUrl) return <div className="text-lg font-medium text-gray-900">{name}</div>;
                  return (
                    <a
                      href={name}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-lg font-medium text-primary-700 hover:text-primary-900 hover:underline break-all"
                      title="Open URL"
                    >
                      {name}
                    </a>
                  );
                })()}
                <div className="text-xs text-gray-600 capitalize flex items-center gap-2">
                  <span>Type: {selectedNode.type}</span>
                  {isAdmin && (
                    <>
                      <button
                        type="button"
                        className="px-2 py-0.5 text-[11px] rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                        disabled={!selectedNode?.id || aiSuggestBusy}
                        onClick={async () => {
                          try {
                            setAiSuggestBusy(true);
                            const res = await apiClient.inferKGEntityType(selectedNode.id);
                            setAiSuggestedType(res);
                            toast.success(
                              `AI suggested: ${res.entity_type}` +
                                (typeof res.confidence === 'number' ? ` (${Math.round(res.confidence * 100)}%)` : '')
                            );
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to infer type');
                          } finally {
                            setAiSuggestBusy(false);
                          }
                        }}
                        title="Infer entity type using LLM against the open-list of known types"
                      >
                        {aiSuggestBusy ? 'AI…' : 'AI Suggest'}
                      </button>
                      {aiSuggestedType?.entity_type && aiSuggestedType.entity_type !== selectedNode.type && (
                        <button
                          type="button"
                          className="px-2 py-0.5 text-[11px] rounded bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
                          disabled={aiApplyBusy}
                          onClick={async () => {
                            try {
                              setAiApplyBusy(true);
                              await apiClient.updateKGEntity(selectedNode.id, { entity_type: aiSuggestedType.entity_type });
                              toast.success(`Updated type to ${aiSuggestedType.entity_type}`);
                              refetch();
                            } catch (e: any) {
                              toast.error(e?.response?.data?.detail || e?.message || 'Failed to update type');
                            } finally {
                              setAiApplyBusy(false);
                            }
                          }}
                          title="Apply the AI-suggested type to the entity"
                        >
                          {aiApplyBusy ? 'Saving…' : 'Apply'}
                        </button>
                      )}
                    </>
                  )}
                </div>
                {(selectedNode as GlobalGraphNode).mention_count !== undefined && (
                  <div className="text-xs text-gray-600">
                    Mentions: {(selectedNode as GlobalGraphNode).mention_count}
                  </div>
                )}
                {(selectedNode as GlobalGraphNode).description && (
                  <div className="text-xs text-gray-600 mt-1">
                    {(selectedNode as GlobalGraphNode).description}
                  </div>
                )}
              </div>

              {(() => {
                const name = String(selectedNode.name || '');
                const arxivId = extractArxivId(name);
                if (!arxivId) return null;
                const absUrl = `https://arxiv.org/abs/${encodeURIComponent(arxivId)}`;
                const pdfUrl = `https://arxiv.org/pdf/${encodeURIComponent(arxivId)}.pdf`;
                return (
                  <div className="mb-4 p-3 rounded border border-gray-200 bg-gray-50">
                    <div className="text-sm font-medium text-gray-800">arXiv</div>
                    <div className="text-xs text-gray-600 mt-0.5 break-all">{arxivId}</div>
                    <div className="mt-2 flex items-center gap-2 flex-wrap">
                      <a
                        href={absUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50"
                      >
                        Open abstract
                      </a>
                      <a
                        href={pdfUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50"
                      >
                        Open PDF
                      </a>
                      <button
                        type="button"
                        className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50"
                        onClick={() => navigate(`/papers?q=${encodeURIComponent(`id:${arxivId}`)}`)}
                        title="Open in Papers page"
                      >
                        Open in Papers
                      </button>
                      <button
                        type="button"
                        disabled={isQueueingArxiv}
                        className="px-2.5 py-1.5 text-xs rounded bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-50"
                        onClick={async () => {
                          try {
                            setIsQueueingArxiv(true);
                            const src = await apiClient.ingestArxivPapers({
                              name: `arXiv ${arxivId}`,
                              paper_ids: [arxivId],
                              max_results: 1,
                              start: 0,
                              sort_by: 'submittedDate',
                              sort_order: 'descending',
                              auto_sync: true,
                              auto_summarize: true,
                              auto_literature_review: false,
                              topic: arxivId,
                            });
                            toast.success('Queued arXiv import');
                            navigate(`/papers?source_id=${encodeURIComponent(String(src.id))}&q=${encodeURIComponent(`id:${arxivId}`)}`);
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to queue arXiv import');
                          } finally {
                            setIsQueueingArxiv(false);
                          }
                        }}
                        title="Create an arXiv import source (async ingestion pipeline)"
                      >
                        {isQueueingArxiv ? 'Queueing…' : 'Queue import'}
                      </button>
                      <button
                        type="button"
                        disabled={isIngestingArxiv}
                        className="px-2.5 py-1.5 text-xs rounded bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50"
                        onClick={async () => {
                          try {
                            setIsIngestingArxiv(true);
                            const res = await apiClient.ingestArxivInstant({ arxiv_input: arxivId, auto_summarize: true, auto_enrich: true });
                            toast.success('Ingested arXiv paper');
                            navigate('/documents', { state: { openDocId: res.document_id } });
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to ingest arXiv paper');
                          } finally {
                            setIsIngestingArxiv(false);
                          }
                        }}
                        title="Instantly ingest into the Knowledge DB"
                      >
                        {isIngestingArxiv ? 'Ingesting…' : 'Ingest to DB'}
                      </button>
                    </div>
                    <div className="mt-1 text-[11px] text-gray-500">
                      Instant ingest makes it available immediately. Queue import runs the async source pipeline.
                    </div>
                  </div>
                );
              })()}

              <div className="mt-4">
                <div className="text-sm font-medium text-gray-800 mb-1">
                  Relationships ({neighborEdges.length})
                </div>
                {neighborEdges.length === 0 ? (
                  <div className="text-sm text-gray-500">No relationships</div>
                ) : (
                  <ul className="space-y-1 text-sm text-gray-700 max-h-[300px] overflow-auto">
                    {neighborEdges.map(e => {
                      const otherId = e.source === selected ? e.target : e.source;
                      const other = nodes.find(n => n.id === otherId);
                      const direction = e.source === selected ? '->' : '<-';
                      return (
                        <li key={e.id} className="flex items-center justify-between py-1 border-b border-gray-100">
                          <span>
                            <span className="text-gray-500">{direction} {e.type.replace(/_/g, ' ')}</span>
                            <button
                              className="ml-1 text-primary-700 hover:underline"
                              onClick={() => setSelected(otherId)}
                            >
                              {other?.name || otherId.slice(0, 8)}
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

              <div className="mt-4 flex gap-2">
                <button
                  className="px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
                  onClick={() => navigate(`/kg/admin?q=${encodeURIComponent(selectedNode.name)}`)}
                >
                  View in KG Admin
                </button>
              </div>
            </div>
          ) : selectedEdge ? (
            <div>
              <div className="mb-3">
                <div className="text-sm text-gray-500">Selected Relationship</div>
                <div className="text-lg font-medium text-gray-900">{selectedEdge.type.replace(/_/g, ' ')}</div>
                {typeof selectedEdge.confidence === 'number' && (
                  <div className="text-xs text-gray-600">
                    Confidence: {(selectedEdge.confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>

              <div className="mt-3">
                <div className="text-sm text-gray-600">
                  <span className="font-medium">From:</span>{' '}
                  <button
                    className="text-primary-700 hover:underline"
                    onClick={() => setSelected(selectedEdge.source)}
                  >
                    {nodes.find(n => n.id === selectedEdge.source)?.name || selectedEdge.source.slice(0, 8)}
                  </button>
                </div>
                <div className="text-sm text-gray-600 mt-1">
                  <span className="font-medium">To:</span>{' '}
                  <button
                    className="text-primary-700 hover:underline"
                    onClick={() => setSelected(selectedEdge.target)}
                  >
                    {nodes.find(n => n.id === selectedEdge.target)?.name || selectedEdge.target.slice(0, 8)}
                  </button>
                </div>
              </div>

              {selectedEdge.evidence && (
                <div className="mt-3">
                  <div className="text-sm font-medium text-gray-800 mb-1">Evidence</div>
                  <div className="p-2 bg-gray-50 border border-gray-200 rounded text-sm text-gray-800">
                    {selectedEdge.evidence}
                  </div>
                  <button
                    className="mt-2 px-3 py-1.5 text-sm rounded bg-gray-100 hover:bg-gray-200"
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
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-gray-500">
              Select a node or edge to see details.
              <div className="mt-4 text-xs text-gray-400">
                <p className="mb-2">Use filters on the left to narrow down the graph.</p>
                <p>Click and drag to pan. Scroll to zoom.</p>
              </div>
            </div>
          )}
        </div>
        )}
      </div>
    </div>
  );
};

export default GlobalGraphPage;
