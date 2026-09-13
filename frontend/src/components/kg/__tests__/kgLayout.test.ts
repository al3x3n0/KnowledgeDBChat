/**
 * The graph layout, tested as arithmetic.
 *
 * This used to be a loop inside a useEffect, which meant the only way to ask
 * "do connected nodes end up near each other?" was to render a graph and look
 * at it. As a stepped simulation the question has an answer you can assert —
 * and the first thing asserted here is the force the old layout did not have.
 */

import { DEFAULT_LAYOUT, GraphLayout, hash, phyllotaxis } from '../kgLayout';

const layout = (over = {}) =>
  new GraphLayout({ ...DEFAULT_LAYOUT, centerX: 0, centerY: 0, ...over });

/** Run to convergence, with a cap so a bug cannot hang the suite. */
const settle = (sim: GraphLayout, maxSteps = 600) => {
  let steps = 0;
  while (sim.step() && steps < maxSteps) steps += 1;
  return steps;
};

const distance = (sim: GraphLayout, a: string, b: string) => {
  const na = sim.nodes.get(a)!;
  const nb = sim.nodes.get(b)!;
  return Math.hypot(na.x - nb.x, na.y - nb.y);
};

describe('attraction — the force the previous layout lacked', () => {
  it('brings an edge in, measured against the same graph without attraction', () => {
    // The direct statement of what was missing. Same nodes, same starting
    // positions, same everything except the attraction constant — so any
    // difference is that force and nothing else.
    const graph = (attraction: number) => {
      const sim = layout({ attraction, gravity: 0 });
      sim.setGraph(['a', 'b'], [{ source: 'a', target: 'b' }], (_id, i) =>
        // Start them far apart, so attraction has something to do.
        ({ x: i === 0 ? -600 : 600, y: 0 })
      );
      settle(sim);
      return distance(sim, 'a', 'b');
    };

    const withAttraction = graph(DEFAULT_LAYOUT.attraction);
    const withoutAttraction = graph(0);

    expect(withAttraction).toBeLessThan(withoutAttraction);
    // And it actually closes most of the gap rather than nudging it.
    expect(withAttraction).toBeLessThan(1200 * 0.5);
  });

  it('settles an edge near its rest length', () => {
    const sim = layout({ gravity: 0 });
    sim.setGraph(['a', 'b'], [{ source: 'a', target: 'b' }]);
    settle(sim);

    // Repulsion holds them apart and attraction pulls them in; the balance
    // should land within a reasonable band of the configured length.
    const d = distance(sim, 'a', 'b');
    expect(d).toBeGreaterThan(DEFAULT_LAYOUT.edgeLength * 0.4);
    expect(d).toBeLessThan(DEFAULT_LAYOUT.edgeLength * 2.5);
  });

  it('keeps a cluster together and separate from another cluster', () => {
    const sim = layout();
    sim.setGraph(
      ['a1', 'a2', 'a3', 'b1', 'b2', 'b3'],
      [
        { source: 'a1', target: 'a2' },
        { source: 'a2', target: 'a3' },
        { source: 'a3', target: 'a1' },
        { source: 'b1', target: 'b2' },
        { source: 'b2', target: 'b3' },
        { source: 'b3', target: 'b1' },
      ]
    );
    settle(sim);

    const within =
      (distance(sim, 'a1', 'a2') + distance(sim, 'a2', 'a3') + distance(sim, 'a1', 'a3')) / 3;
    const between =
      (distance(sim, 'a1', 'b1') + distance(sim, 'a2', 'b2') + distance(sim, 'a3', 'b3')) / 3;
    expect(within).toBeLessThan(between);
  });
});

describe('stability', () => {
  it('separates nodes that start exactly on top of each other', () => {
    // A divide-by-zero here used to fling the pair to infinity.
    const sim = layout();
    sim.setGraph(['a', 'b'], [], () => ({ x: 0, y: 0 }));
    settle(sim);

    const d = distance(sim, 'a', 'b');
    expect(d).toBeGreaterThan(1);
    expect(Number.isFinite(d)).toBe(true);
  });

  it('never produces a non-finite position', () => {
    const sim = layout();
    const ids = Array.from({ length: 40 }, (_, i) => `n${i}`);
    sim.setGraph(
      ids,
      ids.slice(1).map((id, i) => ({ source: ids[i], target: id }))
    );
    settle(sim);

    sim.nodes.forEach((n) => {
      expect(Number.isFinite(n.x)).toBe(true);
      expect(Number.isFinite(n.y)).toBe(true);
    });
  });

  it('stops rather than running for ever', () => {
    const sim = layout();
    sim.setGraph(['a', 'b', 'c'], [{ source: 'a', target: 'b' }]);

    const steps = settle(sim);
    expect(sim.settled).toBe(true);
    // A simulation that never settles burns a core for as long as the tab is
    // open, which is what requestAnimationFrame makes easy to do by accident.
    expect(steps).toBeLessThan(600);
    expect(sim.step()).toBe(false);
  });

  it('holds disconnected nodes near the centre instead of letting them drift', () => {
    const sim = layout({ centerX: 500, centerY: 300 });
    sim.setGraph(['a', 'b', 'c'], []);
    settle(sim);

    sim.nodes.forEach((n) => {
      expect(Math.hypot(n.x - 500, n.y - 300)).toBeLessThan(2000);
    });
  });
});

describe('keeping the picture still', () => {
  it('keeps the positions of nodes that survive a graph change', () => {
    const sim = layout();
    sim.setGraph(['a', 'b'], [{ source: 'a', target: 'b' }]);
    settle(sim);
    const before = { ...sim.nodes.get('a')! };

    sim.setGraph(['a', 'b', 'c'], [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
    ]);

    // Adding a node must not teleport the ones already on screen; a graph
    // that reshuffles on every poll cannot be read.
    expect(sim.nodes.get('a')!.x).toBe(before.x);
    expect(sim.nodes.get('a')!.y).toBe(before.y);
  });

  it('reports whether the topology actually changed', () => {
    const sim = layout();
    const nodes = ['a', 'b'];
    const edges = [{ source: 'a', target: 'b' }];

    expect(sim.setGraph(nodes, edges)).toBe(true);
    // The same graph again: nothing changed, so the caller can leave the
    // picture alone rather than reheating it on every poll.
    expect(sim.setGraph(nodes, edges)).toBe(false);
    expect(sim.setGraph(['a', 'b', 'c'], edges)).toBe(true);
    expect(sim.setGraph(['a'], [])).toBe(true);
  });

  it('drops a node that left the graph', () => {
    const sim = layout();
    sim.setGraph(['a', 'b'], [{ source: 'a', target: 'b' }]);
    sim.setGraph(['a'], []);

    expect(sim.nodes.has('b')).toBe(false);
    expect(sim.nodes.size).toBe(1);
  });

  it('ignores an edge whose endpoint was filtered out', () => {
    const sim = layout();
    // 'ghost' is not in the node list: pulling toward it would drag 'a'
    // toward a position nothing occupies.
    sim.setGraph(['a', 'b'], [
      { source: 'a', target: 'b' },
      { source: 'a', target: 'ghost' },
    ]);
    settle(sim);

    expect(Number.isFinite(sim.nodes.get('a')!.x)).toBe(true);
    expect(sim.nodes.has('ghost')).toBe(false);
  });
});

describe('pinning', () => {
  it('does not move a node the user is dragging', () => {
    const sim = layout();
    sim.setGraph(['a', 'b', 'c'], [
      { source: 'a', target: 'b' },
      { source: 'b', target: 'c' },
    ]);
    sim.pin('a', 1234, -567);
    settle(sim);

    expect(sim.nodes.get('a')!.x).toBe(1234);
    expect(sim.nodes.get('a')!.y).toBe(-567);
  });

  it('lets go when released', () => {
    const sim = layout();
    sim.setGraph(['a', 'b'], [{ source: 'a', target: 'b' }]);
    sim.pin('a', 5000, 5000);
    settle(sim);
    sim.release('a');
    sim.reheat();
    settle(sim);

    // Freed, it is pulled back toward its neighbour and the centre.
    expect(sim.nodes.get('a')!.x).toBeLessThan(5000);
  });
});

describe('helpers', () => {
  it('hashes an id to the same number every time', () => {
    expect(hash('entity-1')).toBe(hash('entity-1'));
    expect(hash('entity-1')).not.toBe(hash('entity-2'));
    expect(hash('')).toBeGreaterThanOrEqual(0);
  });

  it('spaces spiral placements apart', () => {
    const a = phyllotaxis(0, 0, 0);
    const b = phyllotaxis(1, 0, 0);
    const c = phyllotaxis(2, 0, 0);
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(10);
    expect(Math.hypot(b.x - c.x, b.y - c.y)).toBeGreaterThan(10);
  });

  it('reports bounds, and a sane box when empty', () => {
    const sim = layout();
    expect(sim.bounds()).toEqual({ minX: 0, minY: 0, maxX: 0, maxY: 0 });

    sim.setGraph(['a', 'b'], []);
    sim.pin('a', -10, -20);
    sim.pin('b', 30, 40);
    expect(sim.bounds()).toEqual({ minX: -10, minY: -20, maxX: 30, maxY: 40 });
  });
});
