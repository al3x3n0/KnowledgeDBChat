import React from 'react';

export type FGNode = { id: string; name: string; type: string };
export type FGEdge = {
  id: string;
  type: string;
  source: string;
  target: string;
  confidence?: number;
  evidence?: string | null;
  chunk_id?: string | null;
};

type SimNode = FGNode & { x: number; y: number };

export interface ForceGraphProps {
  width: number;
  height: number;
  nodes: FGNode[];
  edges: FGEdge[];
  nodeRadius?: number;
  minScale?: number;
  maxScale?: number;
  focusMode?: 'none' | 'neighbors' | 'node';
  onNodeClick?: (node: FGNode) => void;
  onEdgeClick?: (edge: FGEdge) => void;
  onBackgroundClick?: () => void;
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
    case 'location':
    case 'place':
      return '#0f766e';
    case 'product':
      return '#b45309';
    case 'concept':
    case 'topic':
      return '#7c3aed';
    case 'technology':
    case 'tool':
    case 'framework':
      return '#4f46e5';
    case 'event':
      return '#be123c';
    case 'email':
      return '#0ea5e9';
    case 'url':
      return '#ea580c';
    default:
      return '#6b7280';
  }
};

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));

const hashU32 = (s: string): number => {
  // FNV-1a 32-bit
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
};

const goldenAngle = Math.PI * (3 - Math.sqrt(5));

const ForceGraph = React.forwardRef<ForceGraphHandle, ForceGraphProps>(
  (
    {
      width,
      height,
      nodes,
      edges,
      nodeRadius = 18,
      minScale: minScaleProp,
      maxScale: maxScaleProp,
      focusMode: focusModeProp = 'neighbors',
      onNodeClick,
      onEdgeClick,
      onBackgroundClick,
      selectedNodeId,
      selectedEdgeId,
    },
    ref
  ) => {
    const minScale = typeof minScaleProp === 'number' ? minScaleProp : 0.15;
    const maxScale = typeof maxScaleProp === 'number' ? maxScaleProp : 12;

    const [simNodes, setSimNodes] = React.useState<SimNode[]>([]);
    const nodesRef = React.useRef<SimNode[]>([]);
    const posRef = React.useRef<Map<string, { x: number; y: number }>>(new Map());
    const svgRef = React.useRef<SVGSVGElement | null>(null);
    const keyboardTrapRef = React.useRef<HTMLDivElement | null>(null);

    const [scale, setScale] = React.useState(1);
    const [tx, setTx] = React.useState(0);
    const [ty, setTy] = React.useState(0);
    const viewportRef = React.useRef({ scale: 1, tx: 0, ty: 0 });
    React.useEffect(() => {
      viewportRef.current = { scale, tx, ty };
    }, [scale, tx, ty]);

    const [editLayout, setEditLayout] = React.useState(false);
    const spaceDownRef = React.useRef(false);

    const [focusMode, setFocusMode] = React.useState<NonNullable<ForceGraphProps['focusMode']>>(focusModeProp);
    React.useEffect(() => {
      setFocusMode(focusModeProp);
    }, [focusModeProp]);

    const [hoveredNodeId, setHoveredNodeId] = React.useState<string | null>(null);
    const [hoveredEdgeId, setHoveredEdgeId] = React.useState<string | null>(null);
    const [tooltip, setTooltip] = React.useState<
      | null
      | {
          kind: 'node';
          node: FGNode;
          x: number;
          y: number;
        }
      | {
          kind: 'edge';
          edge: FGEdge;
          x: number;
          y: number;
        }
    >(null);

    const panningRef = React.useRef(false);
    const lastPanRef = React.useRef<{ x: number; y: number } | null>(null);
    const draggingRef = React.useRef<string | null>(null);

    const pointersRef = React.useRef<Map<number, { x: number; y: number }>>(new Map());
    const pinchRef = React.useRef<{
      startDist: number;
      startScale: number;
      gx0: number;
      gy0: number;
    } | null>(null);

    const startPan = (clientX: number, clientY: number) => {
      panningRef.current = true;
      lastPanRef.current = { x: clientX, y: clientY };
    };
    const panMove = (clientX: number, clientY: number) => {
      if (!panningRef.current || !lastPanRef.current) return;
      const dx = clientX - lastPanRef.current.x;
      const dy = clientY - lastPanRef.current.y;
      setTx((t) => t + dx);
      setTy((t) => t + dy);
      lastPanRef.current = { x: clientX, y: clientY };
    };
    const endPan = () => {
      panningRef.current = false;
      lastPanRef.current = null;
    };

    const dragMove = (clientX: number, clientY: number) => {
      if (!draggingRef.current) return;
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const { scale: s, tx: ttx, ty: tty } = viewportRef.current;
      const x = (clientX - rect.left - ttx) / s;
      const y = (clientY - rect.top - tty) / s;
      const n = nodesRef.current.find((nn) => nn.id === draggingRef.current);
      if (!n) return;
      // Allow dragging beyond the visible viewport. Keep it bounded to avoid runaway coords.
      const cx = width / 2;
      const cy = height / 2;
      const bound = Math.max(width, height) * 6;
      const nx = clamp(x, cx - bound, cx + bound);
      const ny = clamp(y, cy - bound, cy + bound);
      n.x = nx;
      n.y = ny;
      posRef.current.set(n.id, { x: nx, y: ny });
      setSimNodes([...nodesRef.current]);
    };
    const endDrag = () => {
      draggingRef.current = null;
    };

    const endAllGestures = React.useCallback(() => {
      pointersRef.current.clear();
      pinchRef.current = null;
      endPan();
      endDrag();
    }, []);

    const setTooltipAtClient = React.useCallback((clientX: number, clientY: number, next: typeof tooltip) => {
      const host = keyboardTrapRef.current;
      if (!host) return;
      const r = host.getBoundingClientRect();
      const x = clientX - r.left;
      const y = clientY - r.top;
      if (!next) {
        setTooltip(null);
        return;
      }
      if (next.kind === 'node') setTooltip({ ...next, x, y });
      else setTooltip({ ...next, x, y });
    }, []);

    const zoomAt = React.useCallback(
      (newScale: number, px: number, py: number) => {
        const { scale: curScale, tx: curTx, ty: curTy } = viewportRef.current;
        const s = clamp(newScale, minScale, maxScale);
        const gx = (px - curTx) / curScale;
        const gy = (py - curTy) / curScale;
        setScale(s);
        setTx(px - gx * s);
        setTy(py - gy * s);
      },
      [minScale, maxScale]
    );

    const doFitView = React.useCallback(
      (padding: number = 60) => {
        const ns = nodesRef.current;
        if (!ns.length) return;
        let minX = Infinity,
          minY = Infinity,
          maxX = -Infinity,
          maxY = -Infinity;
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
        const s = clamp(Math.min(sx, sy), minScale, maxScale);
        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        setScale(s);
        setTx(width / 2 - s * cx);
        setTy(height / 2 - s * cy);
      },
      [width, height, nodeRadius, minScale, maxScale]
    );

    const doCenterOnNode = React.useCallback(
      (nodeId: string, newScale?: number) => {
        const n = nodesRef.current.find((nn) => nn.id === nodeId);
        if (!n) return;
        const s0 = typeof newScale === 'number' ? newScale : viewportRef.current.scale;
        const s = clamp(s0, minScale, maxScale);
        setScale(s);
        setTx(width / 2 - s * n.x);
        setTy(height / 2 - s * n.y);
      },
      [width, height, minScale, maxScale]
    );

    // Initialize nodes when data changes.
    React.useEffect(() => {
      const cx = width / 2;
      const cy = height / 2;
      // Keep positions within a large, off-screen-allowed layout box so graphs can sprawl.
      const layoutBound = Math.max(width, height) * 6;
      const minX = cx - layoutBound;
      const maxX = cx + layoutBound;
      const minY = cy - layoutBound;
      const maxY = cy + layoutBound;

      const viewMin = Math.min(width, height);
      // Spread nodes even if the graph does not fit the viewport (user can pan/zoom/fit).
      const desiredRadius = viewMin * 0.9;
      const maxRelayoutScale = 10;

      // Prune stale positions to keep the map bounded.
      const ids = new Set(nodes.map((n) => n.id));
      Array.from(posRef.current.keys()).forEach((k) => {
        if (!ids.has(k)) posRef.current.delete(k);
      });

      const edgesLocal = edges.slice();
      const hasPos = (id: string) => posRef.current.has(id);
      const getPos = (id: string) => posRef.current.get(id);
      const neighborsOf = (id: string): string[] => {
        const out: string[] = [];
        for (const e of edgesLocal) {
          if (e.source === id) out.push(e.target);
          else if (e.target === id) out.push(e.source);
        }
        return out;
      };

      const missing = nodes.filter((n) => !hasPos(n.id)).map((n) => n.id);
      missing.sort((a, b) => hashU32(a) - hashU32(b));
      const isInitialLayout = posRef.current.size === 0;
      const hasNewNodes = missing.length > 0;

      let spiralIndex = posRef.current.size;
      for (const id of missing) {
        const neigh = neighborsOf(id).filter(hasPos);
        let x: number;
        let y: number;
        const h = hashU32(id);
        const jx = ((h & 0xffff) / 0xffff - 0.5) * 320;
        const jy = (((h >>> 16) & 0xffff) / 0xffff - 0.5) * 320;

        if (neigh.length) {
          let sx = 0;
          let sy = 0;
          for (const nid of neigh) {
            const p = getPos(nid)!;
            sx += p.x;
            sy += p.y;
          }
          // Push slightly away from neighbors to avoid "sticking" to clusters.
          const a = (h / 0xffffffff) * Math.PI * 2;
          const r = Math.max(nodeRadius * 7, 130);
          x = sx / neigh.length + jx + Math.cos(a) * r;
          y = sy / neigh.length + jy + Math.sin(a) * r;
        } else {
          const i = spiralIndex++;
          const r = 120 + 110 * Math.sqrt(i);
          const a = i * goldenAngle + (h / 0xffffffff) * 0.25;
          x = cx + r * Math.cos(a);
          y = cy + r * Math.sin(a);
        }

        x = clamp(x, minX, maxX);
        y = clamp(y, minY, maxY);
        posRef.current.set(id, { x, y });
      }

      const init: SimNode[] = nodes.map((n) => {
        const p = posRef.current.get(n.id) || { x: cx, y: cy };
        return { ...n, x: p.x, y: p.y };
      });

      // Only re-layout when nodes are newly introduced (or first load). Otherwise keep positions fixed.
      if (isInitialLayout || hasNewNodes) {
        const relaxIterations = Math.max(4, Math.min(14, Math.round(6 + init.length / 90)));
        const minSep = Math.max(nodeRadius * 5.0, 110);
        const repelStrength = 0.55;

        for (let iter = 0; iter < relaxIterations; iter++) {
          for (let i = 0; i < init.length; i++) {
            for (let j = i + 1; j < init.length; j++) {
              const a = init[i];
              const b = init[j];
              const dx = b.x - a.x;
              const dy = b.y - a.y;
              const dist = Math.hypot(dx, dy) || 1;
              if (dist >= minSep) continue;
              const overlap = (minSep - dist) / dist;
              const push = overlap * repelStrength * 0.5;
              a.x -= dx * push;
              a.y -= dy * push;
              b.x += dx * push;
              b.y += dy * push;
            }
          }
          for (const n of init) {
            n.x = clamp(n.x, minX, maxX);
            n.y = clamp(n.y, minY, maxY);
          }
        }

        let mx = 0;
        let my = 0;
        for (const n of init) {
          mx += n.x;
          my += n.y;
        }
        mx /= Math.max(1, init.length);
        my /= Math.max(1, init.length);
        let maxR = 0;
        for (const n of init) maxR = Math.max(maxR, Math.hypot(n.x - mx, n.y - my));
        if (maxR > 1) {
          // Only scale outward (never shrink to "fit"); this keeps distance even if it overflows the viewport.
          const scaleUp = clamp(Math.max(1, desiredRadius / maxR), 1, maxRelayoutScale);
          for (const n of init) {
            n.x = clamp(mx + (n.x - mx) * scaleUp, minX, maxX);
            n.y = clamp(my + (n.y - my) * scaleUp, minY, maxY);
          }
        }
      }

      for (const n of init) posRef.current.set(n.id, { x: n.x, y: n.y });
      nodesRef.current = init;
      setSimNodes(init);
    }, [nodes, edges, width, height, nodeRadius]);

    // Wheel zoom (mouse/trackpad). Zooms around cursor.
    const onWheel = (e: React.WheelEvent) => {
      e.preventDefault();
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const delta = -e.deltaY;
      const factor = Math.exp(delta * 0.001);
      zoomAt(viewportRef.current.scale * factor, px, py);
    };

    const recomputePinchStart = () => {
      if (pointersRef.current.size !== 2) {
        pinchRef.current = null;
        return;
      }
      const arr = Array.from(pointersRef.current.values());
      const p1 = arr[0];
      const p2 = arr[1];
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const px = (p1.x + p2.x) / 2 - rect.left;
      const py = (p1.y + p2.y) / 2 - rect.top;
      const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y) || 1;
      const { scale: s, tx: ttx, ty: tty } = viewportRef.current;
      pinchRef.current = {
        startDist: dist,
        startScale: s,
        gx0: (px - ttx) / s,
        gy0: (py - tty) / s,
      };
    };

    const onPointerMove = (e: React.PointerEvent) => {
      if (!pointersRef.current.has(e.pointerId)) return;
      pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (pointersRef.current.size === 2) {
        if (!pinchRef.current) recomputePinchStart();
        const pinch = pinchRef.current;
        if (!pinch) return;
        const arr = Array.from(pointersRef.current.values());
        const p1 = arr[0];
        const p2 = arr[1];
        const rect = svgRef.current?.getBoundingClientRect();
        if (!rect) return;
        const px = (p1.x + p2.x) / 2 - rect.left;
        const py = (p1.y + p2.y) / 2 - rect.top;
        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y) || 1;
        const ratio = dist / pinch.startDist;
        const s = clamp(pinch.startScale * ratio, minScale, maxScale);
        setScale(s);
        setTx(px - pinch.gx0 * s);
        setTy(py - pinch.gy0 * s);
        return;
      }

      if (draggingRef.current) {
        dragMove(e.clientX, e.clientY);
        return;
      }

      if (panningRef.current) panMove(e.clientX, e.clientY);
    };

    const onPointerUpOrCancel = (e: React.PointerEvent) => {
      pointersRef.current.delete(e.pointerId);
      if (pointersRef.current.size < 2) pinchRef.current = null;
      if (pointersRef.current.size === 0) {
        endPan();
        endDrag();
      }
    };

    const onPointerDownBg = (e: React.PointerEvent) => {
      if (e.button !== 0) return;
      (e.currentTarget as SVGRectElement).setPointerCapture(e.pointerId);
      pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
      startPan(e.clientX, e.clientY);
    };

    const onPointerDownNode = (nodeId: string) => (e: React.PointerEvent) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      (e.currentTarget as SVGGElement).setPointerCapture(e.pointerId);
      pointersRef.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (spaceDownRef.current) {
        startPan(e.clientX, e.clientY);
        return;
      }

      if (!editLayout) return;
      draggingRef.current = nodeId;
    };

    const neighborSet = React.useMemo(() => {
      const set = new Set<string>();
      if (!selectedNodeId) return set;
      if (focusMode !== 'neighbors') return set;
      for (const e of edges) {
        if (e.source === selectedNodeId) set.add(e.target);
        if (e.target === selectedNodeId) set.add(e.source);
      }
      return set;
    }, [selectedNodeId, edges, focusMode]);

    const isDimmed = (nodeId: string) => {
      if (!selectedNodeId) return false;
      if (focusMode === 'none') return false;
      if (focusMode === 'node') return nodeId !== selectedNodeId;
      return nodeId !== selectedNodeId && !neighborSet.has(nodeId);
    };

    React.useImperativeHandle(
      ref,
      () => ({
        fitView: (padding?: number) => doFitView(padding ?? 60),
        centerOnNode: (nodeId: string, s?: number) => doCenterOnNode(nodeId, s),
      }),
      [doFitView, doCenterOnNode]
    );

    const posById = React.useMemo(() => {
      const m = new Map<string, { x: number; y: number }>();
      for (const n of simNodes) m.set(n.id, { x: n.x, y: n.y });
      return m;
    }, [simNodes]);

    const onKeyDown = (e: React.KeyboardEvent) => {
      if (e.key === ' ') {
        spaceDownRef.current = true;
        e.preventDefault();
        return;
      }
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        zoomAt(viewportRef.current.scale * 1.2, width / 2, height / 2);
        return;
      }
      if (e.key === '-') {
        e.preventDefault();
        zoomAt(viewportRef.current.scale / 1.2, width / 2, height / 2);
        return;
      }
      if (e.key === '0') {
        e.preventDefault();
        setScale(1);
        setTx(0);
        setTy(0);
        return;
      }
      if (e.key.toLowerCase() === 'f') {
        e.preventDefault();
        doFitView(60);
        return;
      }
      if (e.key === 'Escape') {
        endAllGestures();
        return;
      }

      const panStep = e.shiftKey ? 90 : 45;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setTx((t) => t + panStep);
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        setTx((t) => t - panStep);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setTy((t) => t + panStep);
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setTy((t) => t - panStep);
        return;
      }
    };

    const onKeyUp = (e: React.KeyboardEvent) => {
      if (e.key === ' ') {
        spaceDownRef.current = false;
        e.preventDefault();
      }
    };

    return (
      <div
        ref={keyboardTrapRef}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onKeyUp={onKeyUp}
        onMouseDown={() => keyboardTrapRef.current?.focus()}
        style={{ position: 'relative', width, height, outline: 'none' }}
      >

        <svg
          ref={svgRef}
          width={width}
          height={height}
          onWheel={onWheel}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUpOrCancel}
          onPointerCancel={onPointerUpOrCancel}
          style={{
            cursor: panningRef.current ? 'grabbing' : spaceDownRef.current ? 'grab' : 'default',
            touchAction: 'none',
          }}
        >
          <defs>
            <pattern id="kg-grid" width="32" height="32" patternUnits="userSpaceOnUse">
              <path d="M 32 0 L 0 0 0 32" fill="none" stroke="rgba(15, 23, 42, 0.06)" strokeWidth="1" />
            </pattern>
            <pattern id="kg-grid-bold" width="160" height="160" patternUnits="userSpaceOnUse">
              <path d="M 160 0 L 0 0 0 160" fill="none" stroke="rgba(15, 23, 42, 0.08)" strokeWidth="1" />
            </pattern>
          </defs>

          <rect
            x={0}
            y={0}
            width={width}
            height={height}
            fill="#f8fafc"
            onPointerDown={onPointerDownBg}
            onClick={() => onBackgroundClick && onBackgroundClick()}
          />
          {/* Visual-only grid overlays; must not intercept pointer events (panning/clicks). */}
          <rect x={0} y={0} width={width} height={height} fill="url(#kg-grid)" pointerEvents="none" />
          <rect x={0} y={0} width={width} height={height} fill="url(#kg-grid-bold)" pointerEvents="none" />

          <g transform={`translate(${tx},${ty}) scale(${scale})`}>
            {/* Edges */}
            {edges.map((e) => {
              const s = posById.get(e.source);
              const t = posById.get(e.target);
              if (!s || !t) return null;
              // In strict node-focus mode, only show incident edges (otherwise it looks like edges vanished).
              const focusNodeEdgesOnly = Boolean(selectedNodeId) && focusMode === 'node';
              if (focusNodeEdgesOnly && !(e.source === selectedNodeId || e.target === selectedNodeId)) return null;
              const dim =
                selectedNodeId && focusMode === 'neighbors'
                  ? !(e.source === selectedNodeId || e.target === selectedNodeId)
                  : false;
              const selected = e.id === (selectedEdgeId || '');
              const hovered = e.id === hoveredEdgeId;
              const stroke = selected ? '#0f172a' : hovered ? '#334155' : dim ? 'rgba(148, 163, 184, 0.35)' : 'rgba(100, 116, 139, 0.55)';
              return (
                <g
                  key={e.id}
                  onClick={(ev) => {
                    ev.stopPropagation();
                    onEdgeClick && onEdgeClick(e);
                  }}
                  onPointerEnter={(ev) => {
                    setHoveredEdgeId(e.id);
                    setHoveredNodeId(null);
                    setTooltipAtClient(ev.clientX, ev.clientY, { kind: 'edge', edge: e, x: 0, y: 0 });
                  }}
                  onPointerMove={(ev) => {
                    setTooltipAtClient(ev.clientX, ev.clientY, tooltip && tooltip.kind === 'edge' ? tooltip : { kind: 'edge', edge: e, x: 0, y: 0 });
                  }}
                  onPointerLeave={() => {
                    setHoveredEdgeId(null);
                    setTooltip(null);
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  <line
                    x1={s.x}
                    y1={s.y}
                    x2={t.x}
                    y2={t.y}
                    stroke={stroke}
                    strokeWidth={selected ? 2.75 : hovered ? 2.25 : 1.5}
                  />
                  <text
                    x={(s.x + t.x) / 2}
                    y={(s.y + t.y) / 2}
                    fill={selected ? '#0f172a' : dim ? 'rgba(148, 163, 184, 0.55)' : 'rgba(100, 116, 139, 0.85)'}
                    fontSize={10}
                    textAnchor="middle"
                    dy={-4}
                  >
                    {e.type}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {simNodes.map((n) => {
              const dim = isDimmed(n.id);
              const color = typeColor(n.type);
              const hovered = n.id === hoveredNodeId;
              const selected = n.id === selectedNodeId;
              return (
                <g
                  key={n.id}
                  onPointerDown={onPointerDownNode(n.id)}
                  onClick={(e) => {
                    e.stopPropagation();
                    onNodeClick && onNodeClick(n);
                  }}
                  onPointerEnter={(ev) => {
                    setHoveredNodeId(n.id);
                    setHoveredEdgeId(null);
                    setTooltipAtClient(ev.clientX, ev.clientY, { kind: 'node', node: n, x: 0, y: 0 });
                  }}
                  onPointerMove={(ev) => {
                    setTooltipAtClient(ev.clientX, ev.clientY, tooltip && tooltip.kind === 'node' ? tooltip : { kind: 'node', node: n, x: 0, y: 0 });
                  }}
                  onPointerLeave={() => {
                    setHoveredNodeId(null);
                    setTooltip(null);
                  }}
                  style={{
                    cursor: editLayout ? 'grab' : 'pointer',
                    // In 'node' focus mode, keep other nodes faint but still visible.
                    opacity: dim ? (focusMode === 'node' ? 0.18 : 0.3) : 1,
                    // Keep pan usable even when focused; disallow selecting other nodes in strict node-focus.
                    pointerEvents: dim && focusMode === 'node' ? 'none' : 'auto',
                  }}
                >
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={nodeRadius}
                    fill={color}
                    opacity={dim ? (focusMode === 'node' ? 0.25 : 0.9) : 0.9}
                    stroke={selected ? '#0f172a' : hovered ? 'rgba(15, 23, 42, 0.55)' : '#ffffff'}
                    strokeWidth={selected ? 2.5 : hovered ? 2 : 1}
                  />
                  <text
                    x={n.x}
                    y={n.y + nodeRadius + 14}
                    fill="#111827"
                    fontSize={12}
                    textAnchor="middle"
                    style={{ opacity: dim ? 0.35 : 1 }}
                  >
                    {n.name}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {tooltip && (
          <div
            style={{
              position: 'absolute',
              left: Math.min(tooltip.x + 12, width - 320),
              top: Math.min(tooltip.y + 12, height - 160),
              maxWidth: 320,
              pointerEvents: 'none',
              zIndex: 30,
            }}
            className="rounded-md border border-slate-200 bg-slate-50/98 shadow-sm px-3 py-2 text-xs text-slate-900"
          >
            {tooltip.kind === 'node' ? (
              <div>
                <div className="font-semibold">{tooltip.node.name}</div>
                <div className="text-gray-600 mt-0.5">Type: {tooltip.node.type || 'other'}</div>
                <div className="text-gray-500 mt-1">Click to select. Scroll to zoom. Drag background to pan.</div>
              </div>
            ) : (
              <div>
                <div className="font-semibold">{tooltip.edge.type}</div>
                {typeof tooltip.edge.confidence === 'number' && (
                  <div className="text-gray-600 mt-0.5">Confidence: {(tooltip.edge.confidence * 100).toFixed(0)}%</div>
                )}
                {tooltip.edge.evidence && (
                  <div className="text-gray-700 mt-1 line-clamp-3">{tooltip.edge.evidence}</div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Controls */}
        <div style={{ position: 'absolute', right: 8, top: 8 }}>
          <div className="flex flex-col gap-1 bg-slate-50/98 border border-slate-200 rounded-md shadow-sm p-1 text-slate-900" style={{ zIndex: 20 }}>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-slate-100"
              onClick={() => {
                setFocusMode((m) => (m === 'neighbors' ? 'node' : m === 'node' ? 'none' : 'neighbors'));
              }}
              title="Cycle focus mode"
              type="button"
            >
              {focusMode === 'neighbors' ? 'Focus: Neigh' : focusMode === 'node' ? 'Focus: Node' : 'Focus: All'}
            </button>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-slate-100"
              onClick={() => setEditLayout((v) => !v)}
              title={editLayout ? 'Lock layout' : 'Edit layout (drag nodes)'}
              type="button"
            >
              {editLayout ? 'Lock' : 'Edit'}
            </button>
            <button
              className="w-8 h-8 text-sm rounded hover:bg-slate-100"
              onClick={() => zoomAt(viewportRef.current.scale * 1.2, width / 2, height / 2)}
              title="Zoom in"
              type="button"
            >
              +
            </button>
            <button
              className="w-8 h-8 text-sm rounded hover:bg-slate-100"
              onClick={() => zoomAt(viewportRef.current.scale / 1.2, width / 2, height / 2)}
              title="Zoom out"
              type="button"
            >
              -
            </button>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-slate-100"
              onClick={() => doFitView(60)}
              title="Fit (f)"
              type="button"
            >
              Fit
            </button>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-slate-100 disabled:opacity-50"
              onClick={() => selectedNodeId && doCenterOnNode(selectedNodeId, Math.min(1.2, maxScale))}
              disabled={!selectedNodeId}
              title="Center on selection"
              type="button"
            >
              Center
            </button>
            <button
              className="w-8 h-8 text-xs rounded hover:bg-slate-100"
              onClick={() => {
                setScale(1);
                setTx(0);
                setTy(0);
              }}
              title="Reset (0)"
              type="button"
            >
              1:1
            </button>
            <div className="text-[11px] text-slate-600 text-center px-1 py-0.5 select-none">
              {Math.round(scale * 100)}%
            </div>
          </div>
        </div>
      </div>
    );
  }
);

export default ForceGraph;
