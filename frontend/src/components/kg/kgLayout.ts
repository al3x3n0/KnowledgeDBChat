/**
 * Force-directed layout for the knowledge graph, as a steppable simulation.
 *
 * What this replaces, and why. The previous layout placed each new node near
 * one of its neighbours, then ran a few passes of pure overlap repulsion. It
 * had **no attraction** — nothing ever pulled connected nodes together — so
 * edge length carried no meaning, clusters never formed, and a hub sat
 * wherever it happened to be dropped. A knowledge graph whose picture does not
 * show its structure is decoration.
 *
 * Three forces, which is the minimum that produces a readable graph:
 *
 *   attraction  along each edge, toward a target length. This is what makes
 *               related things sit near each other.
 *   repulsion   between nearby nodes, so dense regions push apart instead of
 *               collapsing into a blob.
 *   gravity     a weak pull toward the centre, which keeps disconnected
 *               components from drifting to infinity.
 *
 * It is a *stepped* simulation rather than a synchronous loop: the caller runs
 * `step()` from requestAnimationFrame and stops when `alpha` decays below the
 * threshold. The old code did all its work inside a `useEffect`, which meant
 * a 1,000-node graph blocked the main thread for as long as it took.
 *
 * Repulsion is bucketed into a grid rather than computed over every pair. The
 * old loop was O(n²) per iteration — 500k pairs at n=1000, times up to 14
 * iterations. Since repulsion falls off with distance, only nearby nodes
 * matter, so a grid of cells the size of the interaction radius gives the same
 * picture for O(n) work.
 */

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  /** Velocity, carried between steps so motion damps smoothly. */
  vx: number;
  vy: number;
  /** A node the user has dragged is pinned: the simulation must not move it. */
  fixed?: boolean;
  /** How many edges touch this node, used to weight it. */
  degree: number;
}

export interface LayoutEdge {
  source: string;
  target: string;
}

export interface LayoutOptions {
  /** Rest length of an edge, in world units. */
  edgeLength: number;
  /** How hard an edge pulls. 0..1; higher converges faster and oscillates more. */
  attraction: number;
  /** How hard nearby nodes push apart. */
  repulsion: number;
  /** Distance beyond which repulsion is ignored — also the grid cell size. */
  repulsionRadius: number;
  /** Pull toward the centre, which keeps islands from drifting away. */
  gravity: number;
  /** Velocity retained each step. Lower settles sooner, higher explores more. */
  damping: number;
  /** Simulation temperature: scales every displacement, and decays each step. */
  alphaDecay: number;
  /** Below this alpha the layout is considered settled. */
  alphaMin: number;
  centerX: number;
  centerY: number;
}

export const DEFAULT_LAYOUT: Omit<LayoutOptions, 'centerX' | 'centerY'> = {
  edgeLength: 150,
  attraction: 0.08,
  repulsion: 9000,
  repulsionRadius: 340,
  gravity: 0.012,
  damping: 0.82,
  alphaDecay: 0.02,
  alphaMin: 0.02,
};

/**
 * A running layout. Construct it with the graph, step it until `settled`.
 *
 * Positions of nodes that already existed are kept, so adding a node to a
 * settled graph nudges the neighbourhood rather than rearranging everything —
 * a graph that reshuffles on every poll is unreadable regardless of how good
 * the layout is.
 */
export class GraphLayout {
  nodes: Map<string, LayoutNode> = new Map();
  private edges: LayoutEdge[] = [];
  private options: LayoutOptions;
  alpha = 1;

  constructor(options: LayoutOptions) {
    this.options = options;
  }

  get settled(): boolean {
    return this.alpha <= this.options.alphaMin;
  }

  /** Wake the simulation up: called when the graph changes or a drag ends. */
  reheat(alpha = 1): void {
    this.alpha = alpha;
  }

  setOptions(options: Partial<LayoutOptions>): void {
    this.options = { ...this.options, ...options };
  }

  /**
   * Replace the graph, keeping positions for nodes that survive.
   *
   * Returns true when the topology actually changed, so the caller can avoid
   * reheating (and therefore avoid moving the picture) when a poll returns the
   * same graph it returned last time.
   */
  setGraph(
    nodeIds: string[],
    edges: LayoutEdge[],
    seed?: (id: string, index: number) => { x: number; y: number }
  ): boolean {
    const incoming = new Set(nodeIds);
    let changed = false;

    for (const id of Array.from(this.nodes.keys())) {
      if (!incoming.has(id)) {
        this.nodes.delete(id);
        changed = true;
      }
    }

    const degree = new Map<string, number>();
    for (const edge of edges) {
      degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
    }

    nodeIds.forEach((id, index) => {
      const existing = this.nodes.get(id);
      if (existing) {
        existing.degree = degree.get(id) || 0;
        return;
      }
      changed = true;
      // A new node starts near its neighbours if it has any that are already
      // placed, so it does not have to travel across the canvas to get there.
      const placed = edges
        .filter((e) => e.source === id || e.target === id)
        .map((e) => (e.source === id ? e.target : e.source))
        .map((other) => this.nodes.get(other))
        .filter((n): n is LayoutNode => Boolean(n));

      let start: { x: number; y: number };
      if (placed.length) {
        const sx = placed.reduce((sum, n) => sum + n.x, 0) / placed.length;
        const sy = placed.reduce((sum, n) => sum + n.y, 0) / placed.length;
        // Offset by a deterministic angle so two nodes joining the same
        // neighbour do not land on the same spot.
        const angle = (hash(id) / 0xffffffff) * Math.PI * 2;
        start = {
          x: sx + Math.cos(angle) * this.options.edgeLength * 0.6,
          y: sy + Math.sin(angle) * this.options.edgeLength * 0.6,
        };
      } else {
        start = seed
          ? seed(id, index)
          : phyllotaxis(index, this.options.centerX, this.options.centerY);
      }

      this.nodes.set(id, {
        id,
        x: start.x,
        y: start.y,
        vx: 0,
        vy: 0,
        degree: degree.get(id) || 0,
      });
    });

    // Only keep edges whose endpoints both exist: an edge to a filtered-out
    // node would otherwise pull toward a position nothing occupies.
    this.edges = edges.filter(
      (e) => this.nodes.has(e.source) && this.nodes.has(e.target)
    );
    return changed;
  }

  pin(id: string, x: number, y: number): void {
    const node = this.nodes.get(id);
    if (!node) return;
    node.x = x;
    node.y = y;
    node.vx = 0;
    node.vy = 0;
    node.fixed = true;
  }

  release(id: string): void {
    const node = this.nodes.get(id);
    if (node) node.fixed = false;
  }

  /** Advance the simulation one frame. Returns true while still moving. */
  step(): boolean {
    if (this.settled) return false;
    const o = this.options;
    const list = Array.from(this.nodes.values());
    if (!list.length) {
      this.alpha = 0;
      return false;
    }

    // --- repulsion, over a grid so this stays linear in the node count
    const cell = Math.max(1, o.repulsionRadius);
    const buckets = new Map<string, LayoutNode[]>();
    const keyOf = (n: LayoutNode) =>
      `${Math.floor(n.x / cell)}:${Math.floor(n.y / cell)}`;
    for (const n of list) {
      const key = keyOf(n);
      const bucket = buckets.get(key);
      if (bucket) bucket.push(n);
      else buckets.set(key, [n]);
    }

    for (const n of list) {
      const gx = Math.floor(n.x / cell);
      const gy = Math.floor(n.y / cell);
      for (let dx = -1; dx <= 1; dx += 1) {
        for (let dy = -1; dy <= 1; dy += 1) {
          const others = buckets.get(`${gx + dx}:${gy + dy}`);
          if (!others) continue;
          for (const m of others) {
            if (m === n) continue;
            let ox = n.x - m.x;
            let oy = n.y - m.y;
            let distSq = ox * ox + oy * oy;
            if (distSq > o.repulsionRadius * o.repulsionRadius) continue;
            if (distSq < 0.01) {
              // Exactly coincident: nudge deterministically rather than
              // dividing by zero and flinging the pair to infinity.
              const angle = (hash(n.id) / 0xffffffff) * Math.PI * 2;
              ox = Math.cos(angle);
              oy = Math.sin(angle);
              distSq = 1;
            }
            const dist = Math.sqrt(distSq);
            const force = o.repulsion / distSq;
            n.vx += (ox / dist) * force * this.alpha;
            n.vy += (oy / dist) * force * this.alpha;
          }
        }
      }
    }

    // --- attraction along edges: the force the old layout was missing
    for (const edge of this.edges) {
      const a = this.nodes.get(edge.source)!;
      const b = this.nodes.get(edge.target)!;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.hypot(dx, dy) || 0.01;
      const displacement = (dist - o.edgeLength) / dist;
      // A hub is pulled less by each of its many edges than a leaf is by its
      // one, otherwise high-degree nodes get dragged around by their crowd.
      const aWeight = 1 / Math.max(1, Math.sqrt(a.degree));
      const bWeight = 1 / Math.max(1, Math.sqrt(b.degree));
      const pull = displacement * o.attraction * this.alpha;
      a.vx += dx * pull * aWeight;
      a.vy += dy * pull * aWeight;
      b.vx -= dx * pull * bWeight;
      b.vy -= dy * pull * bWeight;
    }

    // --- gravity, integration, damping
    let motion = 0;
    for (const n of list) {
      if (n.fixed) {
        n.vx = 0;
        n.vy = 0;
        continue;
      }
      n.vx += (o.centerX - n.x) * o.gravity * this.alpha;
      n.vy += (o.centerY - n.y) * o.gravity * this.alpha;

      n.vx *= o.damping;
      n.vy *= o.damping;

      // Cap the per-frame step. Without this, a node that starts on top of
      // another gets an enormous impulse and shoots off the canvas.
      const speed = Math.hypot(n.vx, n.vy);
      const maxStep = o.edgeLength;
      if (speed > maxStep) {
        n.vx = (n.vx / speed) * maxStep;
        n.vy = (n.vy / speed) * maxStep;
      }

      n.x += n.vx;
      n.y += n.vy;
      motion += Math.abs(n.vx) + Math.abs(n.vy);
    }

    this.alpha *= 1 - o.alphaDecay;
    // Settled early: if nothing is moving, there is no point burning frames
    // waiting for alpha to decay on its own.
    if (motion / list.length < 0.05) this.alpha = 0;
    return !this.settled;
  }

  /** The bounding box of the current layout, for fit-to-view. */
  bounds(): { minX: number; minY: number; maxX: number; maxY: number } {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    this.nodes.forEach((n) => {
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    });
    if (!Number.isFinite(minX)) return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
    return { minX, minY, maxX, maxY };
  }
}

/** A stable 32-bit hash, so a given id always gets the same jitter. */
export function hash(value: string): number {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/** Even spiral placement, for nodes with nothing to attach to. */
export function phyllotaxis(index: number, cx: number, cy: number) {
  const radius = 40 + 26 * Math.sqrt(index);
  const angle = index * GOLDEN_ANGLE;
  return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
}
