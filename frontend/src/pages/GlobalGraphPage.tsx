import React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from 'react-query';
import toast from 'react-hot-toast';
import { ArrowLeft, Network, Filter, Download, ZoomIn } from 'lucide-react';
import apiClient from '../services/api';
import ForceGraph, { FGNode, FGEdge, ForceGraphHandle } from '../components/kg/ForceGraph';

interface GlobalGraphNode extends FGNode {
  mention_count?: number;
  description?: string;
}

const ENTITY_TYPES = ['person', 'org', 'location', 'product', 'concept', 'technology', 'event', 'email', 'url', 'other'];
const RELATION_TYPES = ['works_for', 'manages', 'reports_to', 'collaborates_with', 'owns', 'uses', 'implements', 'part_of', 'located_in', 'related_to', 'mentions', 'references', 'created_by'];

const GlobalGraphPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  // Filter state
  const [entityTypes, setEntityTypes] = React.useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    const paramTypes = searchParams.get('entity_types')?.split(',') || [];
    ENTITY_TYPES.forEach(t => {
      initial[t] = paramTypes.length === 0 || paramTypes.includes(t);
    });
    return initial;
  });

  const [relationTypes, setRelationTypes] = React.useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    const paramTypes = searchParams.get('relation_types')?.split(',') || [];
    RELATION_TYPES.forEach(t => {
      initial[t] = paramTypes.length === 0 || paramTypes.includes(t);
    });
    return initial;
  });

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

  // Build query params
  const queryParams = React.useMemo(() => {
    const enabledEntityTypes = Object.entries(entityTypes)
      .filter(([, v]) => v)
      .map(([k]) => k);
    const enabledRelTypes = Object.entries(relationTypes)
      .filter(([, v]) => v)
      .map(([k]) => k);

    return {
      entity_types: enabledEntityTypes.length < ENTITY_TYPES.length ? enabledEntityTypes.join(',') : undefined,
      relation_types: enabledRelTypes.length < RELATION_TYPES.length ? enabledRelTypes.join(',') : undefined,
      min_confidence: minConfidence > 0 ? minConfidence : undefined,
      min_mentions: minMentions > 1 ? minMentions : undefined,
      limit_nodes: limitNodes !== 300 ? limitNodes : undefined,
      limit_edges: limitEdges !== 1000 ? limitEdges : undefined,
      search: search || undefined,
    };
  }, [entityTypes, relationTypes, minConfidence, minMentions, limitNodes, limitEdges, search]);

  const { data: graphData, isLoading, isError, refetch } = useQuery(
    ['kg-global-graph', queryParams],
    () => apiClient.getGlobalKGGraph(queryParams),
    {
      keepPreviousData: true,
      staleTime: 30000,
    }
  );

  const nodes = (graphData?.nodes || []) as GlobalGraphNode[];
  const edges = (graphData?.edges || []) as FGEdge[];
  const metadata = graphData?.metadata;

  const [selected, setSelected] = React.useState<string | null>(null);
  const selectedNode = React.useMemo(
    () => nodes.find(n => n.id === selected) || null,
    [nodes, selected]
  );
  const neighborEdges = React.useMemo(
    () => edges.filter(e => e.source === selected || e.target === selected),
    [edges, selected]
  );
  const [selectedEdge, setSelectedEdge] = React.useState<FGEdge | null>(null);

  const graphRef = React.useRef<ForceGraphHandle>(null);
  const width = 960;
  const height = 600;

  // Sync URL params
  React.useEffect(() => {
    const params: Record<string, string> = {};
    const enabledEntityTypes = Object.entries(entityTypes)
      .filter(([, v]) => v)
      .map(([k]) => k);
    const enabledRelTypes = Object.entries(relationTypes)
      .filter(([, v]) => v)
      .map(([k]) => k);

    if (enabledEntityTypes.length < ENTITY_TYPES.length) {
      params.entity_types = enabledEntityTypes.join(',');
    }
    if (enabledRelTypes.length < RELATION_TYPES.length) {
      params.relation_types = enabledRelTypes.join(',');
    }
    if (minConfidence > 0) params.min_confidence = String(minConfidence);
    if (minMentions > 1) params.min_mentions = String(minMentions);
    if (limitNodes !== 300) params.limit_nodes = String(limitNodes);
    if (limitEdges !== 1000) params.limit_edges = String(limitEdges);
    if (search) params.search = search;
    if (selected) params.sel = selected;

    setSearchParams(params, { replace: true });
  }, [entityTypes, relationTypes, minConfidence, minMentions, limitNodes, limitEdges, search, selected, setSearchParams]);

  // Auto-center on selection
  React.useEffect(() => {
    if (selected) {
      graphRef.current?.centerOnNode(selected, 1.1);
    }
  }, [selected]);

  const [showFilters, setShowFilters] = React.useState(true);

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
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
      {metadata && (
        <div className="flex items-center gap-4 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
          <span>Total Entities: <strong>{metadata.total_entities}</strong></span>
          <span>Total Relationships: <strong>{metadata.total_relationships}</strong></span>
          <span>Showing: <strong>{metadata.filtered_nodes}</strong> nodes, <strong>{metadata.filtered_edges}</strong> edges</span>
        </div>
      )}

      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden grid grid-cols-1 lg:grid-cols-4 gap-0">
        {/* Filters Panel */}
        {showFilters && (
          <div className="border-r border-gray-200 p-4 lg:col-span-1 space-y-4 max-h-[700px] overflow-auto">
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
                {ENTITY_TYPES.map(t => (
                  <label key={t} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={entityTypes[t]}
                      onChange={e => setEntityTypes(s => ({ ...s, [t]: e.target.checked }))}
                    />
                    <span className="capitalize">{t}</span>
                  </label>
                ))}
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    const all: Record<string, boolean> = {};
                    ENTITY_TYPES.forEach(t => (all[t] = true));
                    setEntityTypes(all);
                  }}
                >
                  Select All
                </button>
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    const none: Record<string, boolean> = {};
                    ENTITY_TYPES.forEach(t => (none[t] = false));
                    setEntityTypes(none);
                  }}
                >
                  Clear
                </button>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium text-gray-800 mb-2">Relation Types</h3>
              <div className="space-y-1 max-h-40 overflow-auto">
                {RELATION_TYPES.map(t => (
                  <label key={t} className="flex items-center gap-2 text-sm">
                    <input
                      type="checkbox"
                      checked={relationTypes[t]}
                      onChange={e => setRelationTypes(s => ({ ...s, [t]: e.target.checked }))}
                    />
                    <span>{t.replace(/_/g, ' ')}</span>
                  </label>
                ))}
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    const all: Record<string, boolean> = {};
                    RELATION_TYPES.forEach(t => (all[t] = true));
                    setRelationTypes(all);
                  }}
                >
                  Select All
                </button>
                <button
                  className="text-xs text-primary-600 hover:underline"
                  onClick={() => {
                    const none: Record<string, boolean> = {};
                    RELATION_TYPES.forEach(t => (none[t] = false));
                    setRelationTypes(none);
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
        <div className={showFilters ? 'lg:col-span-2' : 'lg:col-span-3'}>
          {isLoading ? (
            <div className="p-6 text-gray-600">Loading graph...</div>
          ) : isError ? (
            <div className="p-6 text-red-600">Failed to load graph.</div>
          ) : nodes.length === 0 ? (
            <div className="p-6 text-gray-600">No entities found matching the filters.</div>
          ) : (
            <div className="w-full overflow-auto">
              <ForceGraph
                ref={graphRef}
                width={width}
                height={height}
                nodes={nodes}
                edges={edges}
                selectedNodeId={selected}
                selectedEdgeId={selectedEdge?.id || null}
                onNodeClick={n => { setSelected(n.id); setSelectedEdge(null); }}
                onEdgeClick={e => { setSelectedEdge(e); setSelected(null); }}
              />
            </div>
          )}
        </div>

        {/* Details Panel */}
        <div className="border-l border-gray-200 p-4 lg:col-span-1 max-h-[700px] overflow-auto">
          <h2 className="text-base font-semibold text-gray-900 mb-2">Details</h2>
          {selectedNode ? (
            <div>
              <div className="mb-3">
                <div className="text-sm text-gray-500">Selected Entity</div>
                <div className="text-lg font-medium text-gray-900">{selectedNode.name}</div>
                <div className="text-xs text-gray-600 capitalize">Type: {selectedNode.type}</div>
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
      </div>
    </div>
  );
};

export default GlobalGraphPage;
