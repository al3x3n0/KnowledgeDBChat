/**
 * The pipeline as a graph, editable, over the same spec the text editor edits.
 *
 * The rule that makes two editors safe: **the spec is the only source of
 * truth.** This component never holds a model of its own. It derives nodes and
 * edges from the spec on every render, and every gesture produces a new spec
 * which the page serialises back to the text. So the two views cannot drift —
 * there is nothing for them to drift *between*. A node editor with its own
 * graph state and a JSON view of that state is two representations of one
 * thing, and they diverge the first time one of them has a bug.
 *
 * What the graph shows that the text cannot: the shape. A diamond dependency,
 * a stage nothing depends on, two disconnected halves — those are a sentence
 * of JSON each and a glance in a picture.
 *
 * What it colours: the problems the checker found, on the stage they belong
 * to. A list of problems above an editor makes you find the stage yourself.
 */

import dagre from 'dagre';
import { AlertTriangle, PauseCircle, Repeat } from 'lucide-react';
import React, { useCallback, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  Node,
  NodeProps,
  Position,
  ReactFlowProvider,
  Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';

import type { PipelineCheck } from '../../types';

export interface PipelineStageSpec {
  id: string;
  goal?: string;
  depends_on?: string[];
  assumes?: string[];
  contract?: Record<string, any>;
  checkpoint?: boolean;
  loop?: { max_iterations?: number; until?: string; dry_rounds?: number };
  [key: string]: any;
}

export interface PipelineSpec {
  name?: string;
  stages?: PipelineStageSpec[];
  [key: string]: any;
}

interface PipelineGraphProps {
  spec: PipelineSpec;
  check: PipelineCheck | null;
  onChange: (next: PipelineSpec) => void;
  selectedStageId: string | null;
  onSelectStage: (stageId: string | null) => void;
}

interface StageNodeData {
  stage: PipelineStageSpec;
  tools: string[];
  seconds: number;
  problems: string[];
  selected: boolean;
}

const NODE_WIDTH = 230;
const NODE_HEIGHT = 96;

/** One stage. Wide enough for its derived tools, because the tools are the
 *  part the author never wrote and most needs to see. */
const StageNode: React.FC<NodeProps<StageNodeData>> = ({ data }) => {
  const { stage, tools, seconds, problems, selected } = data;
  const broken = problems.length > 0;
  return (
    <div
      className={[
        'rounded-lg border px-3 py-2 text-left transition-all duration-fast ease-ui',
        'bg-gray-100 shadow-level-1',
        broken
          ? 'border-red-500/70'
          : selected
            ? 'border-primary-500 shadow-accent-glow'
            : 'border-gray-300',
      ].join(' ')}
      style={{ width: NODE_WIDTH }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-500" />
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-gray-900 truncate">{stage.id}</span>
        <span className="flex items-center gap-1 shrink-0">
          {stage.loop?.max_iterations ? (
            <span
              className="flex items-center gap-0.5 text-[10px] font-mono text-gray-500"
              title={`repeats up to ${stage.loop.max_iterations} times`}
            >
              <Repeat className="w-3 h-3" />×{stage.loop.max_iterations}
            </span>
          ) : null}
          {stage.checkpoint && (
            <PauseCircle
              className="w-3.5 h-3.5 text-primary-700"
              aria-label="stops for a person"
            />
          )}
          {broken && <AlertTriangle className="w-3.5 h-3.5 text-red-400" />}
        </span>
      </div>

      {/* The derived tools: what this stage will actually run, deduced from
          its contract rather than typed by the author. */}
      <div className="mt-1.5 flex flex-wrap gap-1">
        {tools.length ? (
          tools.map((tool) => (
            <span
              key={tool}
              className="px-1 py-0.5 rounded bg-gray-200 text-[9px] font-mono text-gray-600"
            >
              {tool}
            </span>
          ))
        ) : (
          <span className="text-[9px] text-gray-500">no tools derived</span>
        )}
      </div>

      {seconds > 0 && (
        <div className="mt-1 text-[10px] font-mono text-gray-500">
          {seconds < 60 ? `${seconds}s` : `${Math.round(seconds / 60)} min`}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-gray-500" />
    </div>
  );
};

const nodeTypes = { stage: StageNode };

/** Top-down, because a pipeline is read as an order and dagre's TB rank
 *  direction is that order made visible. */
const laidOut = (nodes: Node[], edges: Edge[]): Node[] => {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: 'TB', nodesep: 48, ranksep: 72 });
  nodes.forEach((n) => graph.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT }));
  edges.forEach((e) => graph.setEdge(e.source, e.target));
  dagre.layout(graph);
  return nodes.map((n) => {
    const placed = graph.node(n.id);
    return {
      ...n,
      position: {
        x: (placed?.x ?? 0) - NODE_WIDTH / 2,
        y: (placed?.y ?? 0) - NODE_HEIGHT / 2,
      },
    };
  });
};

/** Which problems belong to which stage.
 *
 *  The checker reports "stage_id: what is wrong", so the prefix is the stage.
 *  Matching on the prefix rather than searching the whole string for any stage
 *  id avoids blaming `read` for a problem that merely mentions it. */
const problemsByStage = (problems: string[]): Record<string, string[]> => {
  const map: Record<string, string[]> = {};
  problems.forEach((problem) => {
    const [head, ...rest] = problem.split(':');
    if (rest.length && head && !head.includes(' ')) {
      (map[head.trim()] = map[head.trim()] || []).push(problem);
    }
  });
  return map;
};

const PipelineGraphInner: React.FC<PipelineGraphProps> = ({
  spec,
  check,
  onChange,
  selectedStageId,
  onSelectStage,
}) => {
  const stages = useMemo(
    () => (Array.isArray(spec.stages) ? spec.stages : []),
    [spec.stages]
  );

  const planByStage = useMemo(() => {
    const map: Record<string, { tools: string[]; seconds: number }> = {};
    (check?.plan?.stages || []).forEach((s) => {
      map[s.stage_id] = { tools: s.tools, seconds: s.seconds };
    });
    return map;
  }, [check]);

  const faults = useMemo(() => problemsByStage(check?.problems || []), [check]);

  const { nodes, edges } = useMemo(() => {
    const builtNodes: Node[] = stages.map((stage) => ({
      id: stage.id,
      type: 'stage',
      position: { x: 0, y: 0 },
      data: {
        stage,
        tools: planByStage[stage.id]?.tools || [],
        seconds: planByStage[stage.id]?.seconds || 0,
        problems: faults[stage.id] || [],
        selected: selectedStageId === stage.id,
      } as StageNodeData,
    }));

    const known = new Set(stages.map((s) => s.id));
    const builtEdges: Edge[] = [];
    stages.forEach((stage) => {
      (stage.depends_on || []).forEach((parent) => {
        // An edge to a stage that does not exist is drawn nowhere; the checker
        // reports it, and inventing a node for it would hide the mistake.
        if (!known.has(parent)) return;
        builtEdges.push({
          id: `${parent}->${stage.id}`,
          source: parent,
          target: stage.id,
          animated: false,
          style: { stroke: 'rgba(159, 178, 172, 0.55)' },
        });
      });
    });

    return { nodes: laidOut(builtNodes, builtEdges), edges: builtEdges };
  }, [stages, planByStage, faults, selectedStageId]);

  /** Every gesture rewrites the spec. Nothing is stored here. */
  const writeStages = useCallback(
    (next: PipelineStageSpec[]) => onChange({ ...spec, stages: next }),
    [onChange, spec]
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const { source, target } = connection;
      if (!source || !target || source === target) return;
      writeStages(
        stages.map((stage) =>
          stage.id === target
            ? {
                ...stage,
                depends_on: Array.from(new Set([...(stage.depends_on || []), source])),
              }
            : stage
        )
      );
    },
    [stages, writeStages]
  );

  const onEdgesDelete = useCallback(
    (removed: Edge[]) => {
      const drop = new Set(removed.map((e) => `${e.source}->${e.target}`));
      writeStages(
        stages.map((stage) => ({
          ...stage,
          depends_on: (stage.depends_on || []).filter(
            (parent) => !drop.has(`${parent}->${stage.id}`)
          ),
        }))
      );
    },
    [stages, writeStages]
  );

  const onNodesDelete = useCallback(
    (removed: Node[]) => {
      const gone = new Set(removed.map((n) => n.id));
      writeStages(
        stages
          .filter((stage) => !gone.has(stage.id))
          // A dependency on a deleted stage would be left dangling, which the
          // checker would then report as a problem the user did not create.
          .map((stage) => ({
            ...stage,
            depends_on: (stage.depends_on || []).filter((p) => !gone.has(p)),
          }))
      );
    },
    [stages, writeStages]
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onConnect={onConnect}
      onEdgesDelete={onEdgesDelete}
      onNodesDelete={onNodesDelete}
      onNodeClick={(_, node) => onSelectStage(node.id)}
      onPaneClick={() => onSelectStage(null)}
      deleteKeyCode={['Backspace', 'Delete']}
      fitView
      proOptions={{ hideAttribution: true }}
      className="bg-gray-50"
    >
      <Background color="rgba(159,178,172,0.10)" gap={20} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
};

const PipelineGraph: React.FC<PipelineGraphProps> = (props) => (
  <ReactFlowProvider>
    <PipelineGraphInner {...props} />
  </ReactFlowProvider>
);

export default PipelineGraph;
