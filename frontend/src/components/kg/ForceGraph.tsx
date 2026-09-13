import React from 'react';

import { DEFAULT_LAYOUT, GraphLayout } from './kgLayout';

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

/** Entity colours, for a dark canvas.
 *
 *  These were the 600-weight Tailwind colours, chosen for a white background:
 *  #2563eb, #059669, #0f766e and so on. On this app's near-black ground they
 *  are muddy and several of them read as the same dark smudge at node size.
 *  These are the 400-weight equivalents — same hues, so a person is still
 *  blue and an organisation still green, at a lightness that separates them
 *  against #0b0f10. */
const typeColor = (t: string): string => {
  switch ((t || '').toLowerCase()) {
    case 'person':
      return '#60a5fa';
    case 'org':
    case 'organization':
      return '#34d399';
    case 'location':
    case 'place':
      return '#2dd4bf';
    case 'product':
      return '#fbbf24';
    case 'concept':
    case 'topic':
      return '#c084fc';
    case 'technology':
    case 'tool':
    case 'framework':
      return '#818cf8';
    case 'event':
      return '#fb7185';
    case 'email':
      return '#38bdf8';
    case 'url':
      return '#fb923c';
    default:
      return '#94a3b8';
  }
};

/** The canvas palette, matching the app's surfaces so the graph reads as part
 *  of the page rather than a white panel dropped into it. */
const CANVAS = {
  ground: '#0b0f10',
  grid: 'rgba(159, 178, 172, 0.07)',
  gridBold: 'rgba(159, 178, 172, 0.11)',
  label: '#e7f2ec',
  labelHalo: '#0b0f10',
  nodeRing: '#0b0f10',
  accent: '#19c77b',
  edge: 'rgba(159, 178, 172, 0.45)',
  edgeDim: 'rgba(108, 132, 130, 0.22)',
  edgeHover: '#c8d6cf',
} as const;

const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));



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
      // From here on the camera belongs to the user: no automatic fit will
      // move it out from under them.
      userMovedViewRef.current = true;
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
      // Pin it, so the running simulation treats this position as given
      // rather than pulling the node back out from under the pointer.
      layoutRef.current?.pin(n.id, nx, ny);
      setSimNodes([...nodesRef.current]);
    };
    const endDrag = () => {
      const id = draggingRef.current;
      draggingRef.current = null;
      if (!id) return;
      // Let go and let the neighbourhood settle around where it was put. The
      // node stays put itself — a drag is a statement about where it belongs,
      // so undoing it on release would make the gesture pointless.
      const sim = layoutRef.current;
      if (!sim) return;
      sim.reheat(0.3);
      if (frameRef.current === null) {
        const tick = () => {
          let moving = false;
          for (let i = 0; i < 3; i += 1) moving = sim.step() || moving;
          const byId = new Map(nodesRef.current.map((n) => [n.id, n]));
          sim.nodes.forEach((ln, nid) => {
            const meta = byId.get(nid);
            if (meta) {
              meta.x = ln.x;
              meta.y = ln.y;
            }
          });
          setSimNodes([...nodesRef.current]);
          frameRef.current = moving ? requestAnimationFrame(tick) : null;
        };
        frameRef.current = requestAnimationFrame(tick);
      }
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

    React.useEffect(() => {
      fitOnSettleRef.current = () => doFitView(60);
    }, [doFitView]);


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
    // The layout, as an animated simulation rather than a synchronous loop.
    //
    // What was here before ran up to fourteen O(n²) passes of overlap
    // repulsion inside this effect — blocking the main thread for as long as
    // it took — and had no attraction at all, so edge length meant nothing and
    // clusters never formed. `kgLayout` supplies the three forces that make a
    // graph readable; this drives it from requestAnimationFrame and stops when
    // it settles, so a still graph costs nothing.
    const layoutRef = React.useRef<GraphLayout | null>(null);
    const frameRef = React.useRef<number | null>(null);
    /** Set once the user pans, zooms or drags: after that the camera is theirs. */
    const userMovedViewRef = React.useRef(false);
    /** doFitView is defined below this effect; the ref lets the loop reach it
     *  without making the effect depend on its identity. */
    const fitOnSettleRef = React.useRef<(() => void) | null>(null);

    // Kept in a ref so the animation loop reads the current size without
    // being restarted by a resize.
    const sizeRef = React.useRef({ width, height });
    React.useEffect(() => {
      sizeRef.current = { width, height };
      layoutRef.current?.setOptions({ centerX: width / 2, centerY: height / 2 });
    }, [width, height]);

    // Identity of the graph, not of the arrays holding it. A poll that returns
    // the same entities must not disturb a settled picture, and the previous
    // code re-ran its whole effect whenever the parent handed it a new array.
    const graphSignature = React.useMemo(
      () =>
        `${nodes.map((n) => n.id).sort().join(',')}|${edges
          .map((e) => `${e.source}>${e.target}`)
          .sort()
          .join(',')}`,
      [nodes, edges]
    );

    // A different graph is a different picture: fit it once, even if the user
    // had moved the camera around the previous one.
    React.useEffect(() => {
      userMovedViewRef.current = false;
    }, [graphSignature]);

    React.useEffect(() => {
      const { width: w, height: h } = sizeRef.current;
      if (!layoutRef.current) {
        layoutRef.current = new GraphLayout({
          ...DEFAULT_LAYOUT,
          centerX: w / 2,
          centerY: h / 2,
        });
      }
      const sim = layoutRef.current;
      const changed = sim.setGraph(
        nodes.map((n) => n.id),
        edges.map((e) => ({ source: e.source, target: e.target }))
      );
      if (!changed && sim.settled) {
        // Same graph, already laid out: leave the picture exactly as it is.
        return;
      }
      sim.reheat(changed ? 1 : 0.4);

      const publish = () => {
        const byId = new Map(nodes.map((n) => [n.id, n]));
        const next: SimNode[] = [];
        sim.nodes.forEach((ln, id) => {
          const meta = byId.get(id);
          if (meta) next.push({ ...meta, x: ln.x, y: ln.y });
        });
        nodesRef.current = next;
        posRef.current = new Map(next.map((n) => [n.id, { x: n.x, y: n.y }]));
        setSimNodes(next);
      };

      const tick = () => {
        // A few steps per frame: the simulation converges in far fewer frames
        // without ever holding the thread long enough to drop one.
        let moving = false;
        for (let i = 0; i < 3; i += 1) moving = sim.step() || moving;
        publish();
        if (moving) {
          frameRef.current = requestAnimationFrame(tick);
          return;
        }
        frameRef.current = null;
        // Settled: fill the canvas with what we laid out. A force-balanced
        // graph settles at whatever size its forces imply, which for a small
        // graph is a cluster in the middle of a large empty canvas — the
        // layout this replaced compensated by scaling every position outward,
        // which distorts world coordinates to solve a viewport problem.
        //
        // Skipped once the user has moved the view themselves: yanking the
        // camera away from where someone deliberately put it is worse than a
        // graph that does not fill the frame.
        if (!userMovedViewRef.current) {
          fitOnSettleRef.current?.();
        }
      };

      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      frameRef.current = requestAnimationFrame(tick);

      return () => {
        if (frameRef.current !== null) {
          cancelAnimationFrame(frameRef.current);
          frameRef.current = null;
        }
      };
      // Deliberately keyed on the graph's identity rather than the arrays':
      // see graphSignature above.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [graphSignature]);

    // Stop the loop when the component goes away, whatever state it is in.
    React.useEffect(
      () => () => {
        if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
      },
      []
    );

    // Wheel zoom (mouse/trackpad). Zooms around cursor.
    const onWheel = (e: React.WheelEvent) => {
      e.preventDefault();
      userMovedViewRef.current = true;
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
              <path d="M 32 0 L 0 0 0 32" fill="none" stroke={CANVAS.grid} strokeWidth="1" />
            </pattern>
            <pattern id="kg-grid-bold" width="160" height="160" patternUnits="userSpaceOnUse">
              <path d="M 160 0 L 0 0 0 160" fill="none" stroke={CANVAS.gridBold} strokeWidth="1" />
            </pattern>
          </defs>

          <rect
            x={0}
            y={0}
            width={width}
            height={height}
            fill={CANVAS.ground}
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
              const stroke = selected ? CANVAS.accent : hovered ? CANVAS.edgeHover : dim ? CANVAS.edgeDim : CANVAS.edge;
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
                    fill={selected ? CANVAS.accent : dim ? CANVAS.edgeDim : CANVAS.edge}
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
                    stroke={selected ? CANVAS.accent : hovered ? CANVAS.label : CANVAS.nodeRing}
                    strokeWidth={selected ? 2.5 : hovered ? 2 : 1}
                  />
                  <text
                    x={n.x}
                    y={n.y + nodeRadius + 14}
                    fill={CANVAS.label}
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
            className="rounded-md border border-gray-200 bg-gray-100/98 shadow-sm px-3 py-2 text-xs text-gray-900"
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
          <div className="flex flex-col gap-1 bg-gray-100/98 border border-gray-200 rounded-md shadow-sm p-1 text-gray-900" style={{ zIndex: 20 }}>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-gray-200"
              onClick={() => {
                setFocusMode((m) => (m === 'neighbors' ? 'node' : m === 'node' ? 'none' : 'neighbors'));
              }}
              title="Cycle focus mode"
              type="button"
            >
              {focusMode === 'neighbors' ? 'Focus: Neigh' : focusMode === 'node' ? 'Focus: Node' : 'Focus: All'}
            </button>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-gray-200"
              onClick={() => setEditLayout((v) => !v)}
              title={editLayout ? 'Lock layout' : 'Edit layout (drag nodes)'}
              type="button"
            >
              {editLayout ? 'Lock' : 'Edit'}
            </button>
            <button
              className="w-8 h-8 text-sm rounded hover:bg-gray-200"
              onClick={() => zoomAt(viewportRef.current.scale * 1.2, width / 2, height / 2)}
              title="Zoom in"
              type="button"
            >
              +
            </button>
            <button
              className="w-8 h-8 text-sm rounded hover:bg-gray-200"
              onClick={() => zoomAt(viewportRef.current.scale / 1.2, width / 2, height / 2)}
              title="Zoom out"
              type="button"
            >
              -
            </button>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-gray-200"
              onClick={() => doFitView(60)}
              title="Fit (f)"
              type="button"
            >
              Fit
            </button>
            <button
              className="px-2 h-8 text-xs rounded hover:bg-gray-200 disabled:opacity-50"
              onClick={() => selectedNodeId && doCenterOnNode(selectedNodeId, Math.min(1.2, maxScale))}
              disabled={!selectedNodeId}
              title="Center on selection"
              type="button"
            >
              Center
            </button>
            <button
              className="w-8 h-8 text-xs rounded hover:bg-gray-200"
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
            <div className="text-[11px] text-gray-600 text-center px-1 py-0.5 select-none">
              {Math.round(scale * 100)}%
            </div>
          </div>
        </div>
      </div>
    );
  }
);

export default ForceGraph;
