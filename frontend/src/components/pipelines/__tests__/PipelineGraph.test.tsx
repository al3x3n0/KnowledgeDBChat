/**
 * The graph editor, and the property that makes two editors safe.
 *
 * There is one document. The graph holds no model of its own — every gesture
 * hands back a whole spec — so the assertions here are mostly "did this edit
 * produce the right spec", not "did the graph's internal state change".
 * A node editor with its own graph state plus a JSON view of that state is two
 * representations of one thing, and they diverge the first time one has a bug.
 */

import { render, screen } from '@testing-library/react';
import React from 'react';

import PipelineGraph from '../PipelineGraph';
import type { PipelineCheck } from '../../../types';

// ReactFlow needs layout APIs jsdom does not implement. The component under
// test is the spec arithmetic around it, so the canvas is stubbed and the
// callbacks it would fire are exposed instead.
let captured: any = {};
jest.mock('reactflow', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: ({ nodes, edges, onConnect, onEdgesDelete, onNodesDelete }: any) => {
      captured = { nodes, edges, onConnect, onEdgesDelete, onNodesDelete };
      return (
        <div data-testid="flow">
          {nodes.map((n: any) => (
            <div key={n.id} data-testid={`node-${n.id}`}>
              {n.data.stage.id}
              {n.data.tools.map((t: string) => (
                <span key={t}>{t}</span>
              ))}
              {n.data.problems.length > 0 && <span>{`broken:${n.id}`}</span>}
            </div>
          ))}
          {edges.map((e: any) => (
            <div key={e.id} data-testid={`edge-${e.id}`} />
          ))}
        </div>
      );
    },
    Background: () => null,
    Controls: () => null,
    Handle: () => null,
    Position: { Top: 'top', Bottom: 'bottom' },
    ReactFlowProvider: ({ children }: any) => <>{children}</>,
  };
});

const spec = {
  name: 'study',
  stages: [
    { id: 'gather', goal: 'g', contract: {} },
    { id: 'read', goal: 'r', depends_on: ['gather'], contract: {} },
    { id: 'writeup', goal: 'w', depends_on: ['read'], contract: {} },
  ],
};

const check = {
  valid: true,
  problems: [],
  expressible: true,
  binding_problems: [],
  description: [],
  plan: {
    order: ['gather', 'read', 'writeup'],
    stages: [
      { stage_id: 'gather', tools: ['ingest_paper_by_id'], iterations: 1, seconds: 60, checkpoint: false, unpriced: [] },
      { stage_id: 'read', tools: ['extract_paper_insights'], iterations: 1, seconds: 45, checkpoint: false, unpriced: [] },
      { stage_id: 'writeup', tools: ['create_synthesis_document'], iterations: 1, seconds: 120, checkpoint: false, unpriced: [] },
    ],
    total_seconds: 225,
    critical_path_seconds: 225,
    checkpoints: [],
  },
  budget: null,
} as PipelineCheck;

const mountGraph = (over: Partial<React.ComponentProps<typeof PipelineGraph>> = {}) => {
  const onChange = jest.fn();
  render(
    <PipelineGraph
      spec={spec}
      check={check}
      onChange={onChange}
      selectedStageId={null}
      onSelectStage={jest.fn()}
      {...over}
    />
  );
  return { onChange };
};

beforeEach(() => {
  captured = {};
});

it('draws a node per stage and an edge per dependency', () => {
  mountGraph();

  expect(screen.getByTestId('node-gather')).toBeInTheDocument();
  expect(screen.getByTestId('node-writeup')).toBeInTheDocument();
  expect(screen.getByTestId('edge-gather->read')).toBeInTheDocument();
  expect(screen.getByTestId('edge-read->writeup')).toBeInTheDocument();
});

it('shows the derived tools on the stage, which the author never typed', () => {
  mountGraph();

  expect(screen.getByText('ingest_paper_by_id')).toBeInTheDocument();
  expect(screen.getByText('create_synthesis_document')).toBeInTheDocument();
});

it('puts a problem on the stage it belongs to', () => {
  mountGraph({
    check: {
      ...check,
      valid: false,
      problems: ["read: assumes 'paper_insights', which no stage before it produces"],
    },
  });

  // Not on every node, and not merely in a list above the editor that leaves
  // the author to find the stage themselves.
  expect(screen.getByText('broken:read')).toBeInTheDocument();
  expect(screen.queryByText('broken:gather')).not.toBeInTheDocument();
});

it('does not draw an edge to a stage that does not exist', () => {
  // The checker reports the dangling dependency. Inventing a node for it here
  // would hide the mistake behind a picture that looks fine.
  mountGraph({
    spec: {
      name: 'x',
      stages: [{ id: 'a', depends_on: ['ghost'], contract: {} }],
    },
  });

  expect(screen.getByTestId('node-a')).toBeInTheDocument();
  expect(screen.queryByTestId('node-ghost')).not.toBeInTheDocument();
  expect(screen.queryByTestId('edge-ghost->a')).not.toBeInTheDocument();
});

describe('editing writes the spec, not a private model', () => {
  it('connecting two stages adds a dependency', () => {
    const { onChange } = mountGraph();

    captured.onConnect({ source: 'gather', target: 'writeup' });

    const next = onChange.mock.calls[0][0];
    expect(next.stages.find((s: any) => s.id === 'writeup').depends_on).toEqual([
      'read',
      'gather',
    ]);
    // Everything else is carried through untouched: an edit is an edit, not a
    // rewrite of the document.
    expect(next.name).toBe('study');
    expect(next.stages).toHaveLength(3);
  });

  it('refuses to connect a stage to itself', () => {
    const { onChange } = mountGraph();

    captured.onConnect({ source: 'read', target: 'read' });

    expect(onChange).not.toHaveBeenCalled();
  });

  it('deleting an edge removes just that dependency', () => {
    const { onChange } = mountGraph();

    captured.onEdgesDelete([{ id: 'read->writeup', source: 'read', target: 'writeup' }]);

    const next = onChange.mock.calls[0][0];
    expect(next.stages.find((s: any) => s.id === 'writeup').depends_on).toEqual([]);
    expect(next.stages.find((s: any) => s.id === 'read').depends_on).toEqual(['gather']);
  });

  it('deleting a stage takes the dependencies on it too', () => {
    const { onChange } = mountGraph();

    captured.onNodesDelete([{ id: 'read' }]);

    const next = onChange.mock.calls[0][0];
    expect(next.stages.map((s: any) => s.id)).toEqual(['gather', 'writeup']);
    // Otherwise writeup would still depend on a stage that no longer exists,
    // and the checker would report a problem the user did not create.
    expect(next.stages.find((s: any) => s.id === 'writeup').depends_on).toEqual([]);
  });
});
