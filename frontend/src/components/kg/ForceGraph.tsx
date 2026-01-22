import React from 'react';

export type FGNode = { id: string; name: string; type: string };
export type FGEdge = { id: string; type: string; source: string; target: string; confidence?: number; evidence?: string | null; chunk_id?: string | null };

type SimNode = FGNode & { x: number; y: number; vx: number; vy: number; fx?: number | null; fy?: number | null };

export interface ForceGraphProps {
  width: number;
  height: number;
  nodes: FGNode[];
  edges: FGEdge[];
  nodeRadius?: number;
  onNodeClick?: (node: FGNode) => void;
  onEdgeClick?: (edge: FGEdge) => void;
  selectedNodeId?: string | null;
  selectedEdgeId?: string | null;
}

export interface ForceGraphHandle {
  fitView: (padding?: number) => void;
  centerOnNode: (nodeId: string, scale?: number) => void;
}

const typeColor = (t: string): string => {
  switch ((t || '').toLowerCase()) {
    case 'person':
      return '#2563eb';
    case 'org':
    case 'organization':
      return '#059669';
    case 'email':
      return '#7c3aed';
    case 'url':
      return '#f59e0b';
    default:
      return '#6b7280';
  }
};

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

const ForceGraph = React.forwardRef<ForceGraphHandle, ForceGraphProps>(({ 
  width,
  height,
  nodes,
  edges,
  nodeRadius = 18,
  onNodeClick,
  onEdgeClick,
  selectedNodeId,
  selectedEdgeId,
}, ref) => {
  const [simNodes, setSimNodes] = React.useState<SimNode[]>([]);
  const nodesRef = React.useRef<SimNode[]>([]);
  const edgesRef = React.useRef<FGEdge[]>(edges);
  const rafRef = React.useRef<number | null>(null);

  const [scale, setScale] = React.useState(1);
  const [tx, setTx] = React.useState(0);
  const [ty, setTy] = React.useState(0);
  const panningRef = React.useRef(false);
  const lastPanRef = React.useRef<{ x: number; y: number } | null>(null);

  const draggingRef = React.useRef<string | null>(null);

  // Initialize nodes when data changes
  React.useEffect(() => {
    const init: SimNode[] = nodes.map((n, i) => ({
      ...n,
      x: (Math.random() * 0.6 + 0.2) * width,
      y: (Math.random() * 0.6 + 0.2) * height,
      vx: 0,
      vy: 0,
      fx: null,
      fy: null,
    }));
    nodesRef.current = init;
    edgesRef.current = edges.slice();
    setSimNodes(init);
  }, [nodes, edges, width, height]);

  // Simple force simulation
  React.useEffect(() => {
    const stiffness = 0.05; // spring strength
    const linkDistance = 150;
    const charge = -1500; // repulsion strength
    const damping = 0.85; // velocity decay

    const step = () => {
      const ns = nodesRef.current;
      const es = edgesRef.current;
      if (!ns.length) return;

      // Spring forces
      for (const e of es) {
        const a = ns.find(n => n.id === e.source);
        const b = ns.find(n => n.id === e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const diff = dist - linkDistance;
        const nx = dx / dist;
        const ny = dy / dist;
        const fx = stiffness * diff * nx;
        const fy = stiffness * diff * ny;
        if (a.fx == null) {
          a.vx += fx;
          a.vy += fy;
        }
        if (b.fx == null) {
          b.vx -= fx;
          b.vy -= fy;
        }
      }

      // Repulsion (O(n^2) naive)
      for (let i = 0; i < ns.length; i++) {
        for (let j = i + 1; j < ns.length; j++) {
          const a = ns[i];
          const b = ns[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist2 = Math.max(25, dx * dx + dy * dy);
          const force = charge / dist2; // inverse square
          const dist = Math.sqrt(dist2);
          const nx = dx / dist;
          const ny = dy / dist;
          if (a.fx == null) {
            a.vx -= force * nx;
            a.vy -= force * ny;
          }
          if (b.fx == null) {
            b.vx += force * nx;
            b.vy += force * ny;
          }
        }
      }

      // Integrate
      for (const n of ns) {
        if (n.fx != null && n.fy != null) {
          n.x = n.fx;
          n.y = n.fy;
          n.vx = 0;
          n.vy = 0;
        } else {
          n.vx *= damping;
          n.vy *= damping;
          n.x += n.vx * 0.02; // small timestep to stabilize
          n.y += n.vy * 0.02;
          n.x = clamp(n.x, nodeRadius, width - nodeRadius);
          n.y = clamp(n.y, nodeRadius, height - nodeRadius);
        }
      }

      setSimNodes([...ns]);
      rafRef.current = requestAnimationFrame(step);
    };

    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [width, height, nodeRadius]);

  // Pan/zoom handlers
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = -e.deltaY;
    const factor = Math.exp(delta * 0.001);
    const newScale = clamp(scale * factor, 0.5, 3);
    setScale(newScale);
  };

  const onBgMouseDown = (e: React.MouseEvent) => {
    // only if not clicking a node
    panningRef.current = true;
    lastPanRef.current = { x: e.clientX, y: e.clientY };
  };
  const onBgMouseMove = (e: React.MouseEvent) => {
    if (!panningRef.current || !lastPanRef.current) return;
    const dx = e.clientX - lastPanRef.current.x;
    const dy = e.clientY - lastPanRef.current.y;
    setTx(t => t + dx);
    setTy(t => t + dy);
    lastPanRef.current = { x: e.clientX, y: e.clientY };
  };
  const onBgMouseUp = () => {
    panningRef.current = false;
    lastPanRef.current = null;
  };

  // Node dragging
  const startDrag = (id: string) => (e: React.MouseEvent) => {
    e.stopPropagation();
    draggingRef.current = id;
    const n = nodesRef.current.find(n => n.id === id);
    if (n) {
      n.fx = n.x;
      n.fy = n.y;
    }
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!draggingRef.current) return;
    const id = draggingRef.current;
    const rect = (e.target as SVGElement).closest('svg')?.getBoundingClientRect();
    if (!rect) return;
    const x = (e.clientX - rect.left - tx) / scale;
    const y = (e.clientY - rect.top - ty) / scale;
    const n = nodesRef.current.find(n => n.id === id);
    if (n) {
      n.fx = clamp(x, nodeRadius, width - nodeRadius);
      n.fy = clamp(y, nodeRadius, height - nodeRadius);
    }
  };
  const endDrag = () => {
    if (!draggingRef.current) return;
    const id = draggingRef.current;
    const n = nodesRef.current.find(n => n.id === id);
    if (n) {
      n.fx = null;
      n.fy = null;
    }
    draggingRef.current = null;
  };

  const neighborSet = React.useMemo(() => {
    const set = new Set<string>();
    if (!selectedNodeId) return set;
    for (const e of edgesRef.current) {
      if (e.source === selectedNodeId) set.add(e.target);
      if (e.target === selectedNodeId) set.add(e.source);
    }
    return set;
  }, [selectedNodeId]);

  const isDimmed = (nodeId: string) => {
    if (!selectedNodeId) return false;
    return nodeId !== selectedNodeId && !neighborSet.has(nodeId);
  };

  // Expose imperative actions
  React.useImperativeHandle(ref, () => ({
    fitView: (padding: number = 60) => {
      const ns = nodesRef.current;
      if (!ns.length) return;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of ns) {
        if (n.x < minX) minX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.x > maxX) maxX = n.x;
        if (n.y > maxY) maxY = n.y;
      }
      const boundsW = Math.max(1, maxX - minX + nodeRadius * 2);
      const boundsH = Math.max(1, maxY - minY + nodeRadius * 2);
      const sx = (width - padding * 2) / boundsW;
      const sy = (height - padding * 2) / boundsH;
      const s = clamp(Math.min(sx, sy), 0.2, 3);
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      setScale(s);
      setTx(width / 2 - s * cx);
      setTy(height / 2 - s * cy);
    },
    centerOnNode: (nodeId: string, newScale?: number) => {
      const n = nodesRef.current.find(n => n.id === nodeId);
      if (!n) return;
      const s = typeof newScale === 'number' ? newScale : scale;
      const clamped = clamp(s, 0.5, 3);
      setScale(clamped);
      setTx(width / 2 - clamped * n.x);
      setTy(height / 2 - clamped * n.y);
    },
  }), [width, height, nodeRadius, scale]);

  return (
    <svg
      width={width}
      height={height}
      onWheel={onWheel}
      onMouseMove={onMouseMove}
      onMouseUp={() => { onBgMouseUp(); endDrag(); }}
      style={{ cursor: panningRef.current ? 'grabbing' : 'default' }}
    >
      <rect
        x={0}
        y={0}
        width={width}
        height={height}
        fill="#ffffff"
        onMouseDown={onBgMouseDown}
        onMouseMove={onBgMouseMove}
        onMouseUp={onBgMouseUp}
        style={{ opacity: 0 }}
      />
      <g transform={`translate(${tx},${ty}) scale(${scale})`}>
        {/* Edges */}
        {edges.map(e => {
          const s = simNodes.find(n => n.id === e.source);
          const t = simNodes.find(n => n.id === e.target);
          if (!s || !t) return null;
          const dim = selectedNodeId && !(e.source === selectedNodeId || e.target === selectedNodeId);
          const selected = e.id === (selectedEdgeId || '');
          return (
            <g key={e.id} onClick={(ev) => { ev.stopPropagation(); onEdgeClick && onEdgeClick(e); }} style={{ cursor: 'pointer' }}>
              <line x1={s.x} y1={s.y} x2={t.x} y2={t.y} stroke={selected ? '#111827' : (dim ? '#e5e7eb' : '#cbd5e1')} strokeWidth={selected ? 2.5 : 1.5} />
              <text x={(s.x + t.x) / 2} y={(s.y + t.y) / 2} fill={selected ? '#111827' : (dim ? '#cbd5e1' : '#64748b')} fontSize={10} textAnchor="middle" dy={-4}>
                {e.type}
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {simNodes.map(n => {
          const dim = isDimmed(n.id);
          const color = typeColor(n.type);
          return (
            <g key={n.id} onMouseDown={startDrag(n.id)} onClick={(e) => { e.stopPropagation(); onNodeClick && onNodeClick(n); }}>
              <circle cx={n.x} cy={n.y} r={nodeRadius} fill={color} opacity={dim ? 0.3 : 0.9} stroke={n.id === selectedNodeId ? '#111827' : '#ffffff'} strokeWidth={n.id === selectedNodeId ? 2 : 1} />
              <text x={n.x} y={n.y + nodeRadius + 14} fill="#111827" fontSize={12} textAnchor="middle" style={{ opacity: dim ? 0.4 : 1 }}>
                {n.name}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
});

export default ForceGraph;
