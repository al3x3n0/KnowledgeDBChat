/**
 * Zustand store for workflow editor state.
 *
 * Manages:
 * - Workflow metadata
 * - Nodes and edges
 * - Selection state
 * - Undo/redo history
 */

import { create } from 'zustand';
import {
  Node,
  Edge,
  Connection,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
  MarkerType,
} from 'reactflow';
import dagre from 'dagre';

// Node types
export type WorkflowNodeType = 'start' | 'end' | 'tool' | 'condition' | 'parallel' | 'loop' | 'wait' | 'switch' | 'subworkflow';

// Tool schema types
export interface ToolParameter {
  name: string;
  type: string;
  description?: string;
  required: boolean;
  default?: any;
  enum?: string[];
}

export interface ToolSchema {
  name: string;
  description: string;
  parameters: Record<string, any>;
  parameter_list: ToolParameter[];
  tool_type: 'builtin' | 'custom';
}

export interface ContextVariable {
  path: string;
  type: string;
  from_node: string;
  description?: string;
}

// Switch case definition
export interface SwitchCase {
  value: string;
  label: string;
}

// Node data structure
export interface WorkflowNodeData {
  label: string;
  nodeType: WorkflowNodeType;
  toolId?: string;
  builtinTool?: string;
  config: {
    inputMapping?: Record<string, string>;
    outputKey?: string;
    condition?: {
      type: string;
      left?: string;
      right?: string;
      expression?: string;
    };
    loopSource?: string;
    maxIterations?: number;
    waitSeconds?: number;
    // Switch node config
    switch_expression?: string;
    cases?: SwitchCase[];
    default_label?: string;
    // Sub-workflow node config
    workflow_id?: string;
    timeout_seconds?: number;
    on_error?: 'fail' | 'continue';
  };
}

// Workflow metadata
export interface WorkflowMetadata {
  id?: string;
  name: string;
  description: string;
  isActive: boolean;
  triggerConfig: {
    type: 'manual' | 'schedule' | 'event' | 'webhook';
    schedule?: string;
    event?: string;
    webhookSecret?: string;
  };
}

// Store state
interface WorkflowState {
  // Metadata
  metadata: WorkflowMetadata;

  // React Flow state
  nodes: Node<WorkflowNodeData>[];
  edges: Edge[];

  // Selection
  selectedNodeId: string | null;
  selectedEdgeId: string | null;

  // UI state
  isDirty: boolean;
  isSaving: boolean;

  // History for undo/redo
  history: { nodes: Node<WorkflowNodeData>[]; edges: Edge[] }[];
  historyIndex: number;

  // Tool schemas
  toolSchemas: Record<string, ToolSchema>;
  toolSchemasLoading: boolean;

  // Context flow (available variables at each node)
  contextFlow: Record<string, ContextVariable[]>;

  // Actions
  setMetadata: (metadata: Partial<WorkflowMetadata>) => void;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setEdges: (edges: Edge[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
  addNode: (node: Node<WorkflowNodeData>) => void;
  updateNode: (nodeId: string, data: Partial<WorkflowNodeData>) => void;
  deleteNode: (nodeId: string) => void;
  deleteEdge: (edgeId: string) => void;
  selectNode: (nodeId: string | null) => void;
  selectEdge: (edgeId: string | null) => void;

  // Workflow operations
  loadWorkflow: (workflow: any) => void;
  applyWorkflowDraft: (workflow: {
    name?: string;
    description?: string;
    is_active?: boolean;
    trigger_config?: {
      type: 'manual' | 'schedule' | 'event' | 'webhook';
      schedule?: string;
      event?: string;
      webhookSecret?: string;
    };
    nodes: any[];
    edges: any[];
  }) => void;
  resetWorkflow: () => void;
  getWorkflowData: () => { nodes: any[]; edges: any[] };

  // History
  saveToHistory: () => void;
  undo: () => void;
  redo: () => void;

  // State
  setIsDirty: (isDirty: boolean) => void;
  setIsSaving: (isSaving: boolean) => void;

  // Tool schemas
  loadToolSchemas: () => Promise<void>;
  getToolSchema: (toolName: string) => ToolSchema | undefined;

  // Context flow
  computeContextFlow: () => void;
  getAvailableContext: (nodeId: string) => ContextVariable[];
}

// Default metadata
const defaultMetadata: WorkflowMetadata = {
  name: 'New Workflow',
  description: '',
  isActive: true,
  triggerConfig: {
    type: 'manual',
  },
};

// Initial nodes for a new workflow
const initialNodes: Node<WorkflowNodeData>[] = [
  {
    id: 'start',
    type: 'startNode',
    position: { x: 250, y: 50 },
    data: {
      label: 'Start',
      nodeType: 'start',
      config: {},
    },
  },
  {
    id: 'end',
    type: 'endNode',
    position: { x: 250, y: 350 },
    data: {
      label: 'End',
      nodeType: 'end',
      config: {},
    },
  },
];

// Initial edges
const initialEdges: Edge[] = [];

const LAYOUT_NODE_DIMENSIONS: Record<WorkflowNodeType, { width: number; height: number }> = {
  start: { width: 160, height: 64 },
  end: { width: 160, height: 64 },
  tool: { width: 220, height: 84 },
  condition: { width: 200, height: 90 },
  parallel: { width: 200, height: 90 },
  loop: { width: 200, height: 90 },
  wait: { width: 180, height: 80 },
  switch: { width: 200, height: 100 },
  subworkflow: { width: 220, height: 100 },
};

const layoutWorkflowNodes = (
  nodes: Node<WorkflowNodeData>[],
  edges: Edge[]
): Node<WorkflowNodeData>[] => {
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({
    rankdir: 'TB',
    nodesep: 40,
    ranksep: 80,
  });

  nodes.forEach((node) => {
    const nodeType = node.data.nodeType;
    const size = LAYOUT_NODE_DIMENSIONS[nodeType] || { width: 200, height: 80 };
    graph.setNode(node.id, size);
  });

  edges.forEach((edge) => {
    graph.setEdge(edge.source, edge.target);
  });

  dagre.layout(graph);

  return nodes.map((node) => {
    const layout = graph.node(node.id);
    if (!layout) {
      return node;
    }
    const size = LAYOUT_NODE_DIMENSIONS[node.data.nodeType] || { width: 200, height: 80 };
    return {
      ...node,
      position: {
        x: layout.x - size.width / 2,
        y: layout.y - size.height / 2,
      },
    };
  });
};

// Create the store
export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  // Initial state
  metadata: { ...defaultMetadata },
  nodes: [...initialNodes],
  edges: [...initialEdges],
  selectedNodeId: null,
  selectedEdgeId: null,
  isDirty: false,
  isSaving: false,
  history: [],
  historyIndex: -1,
  toolSchemas: {},
  toolSchemasLoading: false,
  contextFlow: {},

  // Metadata actions
  setMetadata: (metadata) => {
    set((state) => ({
      metadata: { ...state.metadata, ...metadata },
      isDirty: true,
    }));
  },

  // Node/edge setters
  setNodes: (nodes) => {
    set({ nodes, isDirty: true });
  },

  setEdges: (edges) => {
    set({ edges, isDirty: true });
  },

  // React Flow handlers
  onNodesChange: (changes) => {
    set((state) => ({
      nodes: applyNodeChanges(changes, state.nodes) as Node<WorkflowNodeData>[],
      isDirty: true,
    }));
  },

  onEdgesChange: (changes) => {
    set((state) => ({
      edges: applyEdgeChanges(changes, state.edges),
      isDirty: true,
    }));
  },

  onConnect: (connection) => {
    set((state) => ({
      edges: addEdge(
        {
          ...connection,
          type: 'smoothstep',
          animated: false,
          markerEnd: { type: MarkerType.ArrowClosed },
        },
        state.edges
      ),
      isDirty: true,
    }));
  },

  // Node operations
  addNode: (node) => {
    get().saveToHistory();
    set((state) => ({
      nodes: [...state.nodes, node],
      isDirty: true,
    }));
  },

  updateNode: (nodeId, data) => {
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, ...data } }
          : node
      ),
      isDirty: true,
    }));
  },

  deleteNode: (nodeId) => {
    if (nodeId === 'start' || nodeId === 'end') return; // Can't delete start/end

    get().saveToHistory();
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== nodeId),
      edges: state.edges.filter(
        (edge) => edge.source !== nodeId && edge.target !== nodeId
      ),
      selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
      isDirty: true,
    }));
  },

  deleteEdge: (edgeId) => {
    get().saveToHistory();
    set((state) => ({
      edges: state.edges.filter((edge) => edge.id !== edgeId),
      selectedEdgeId: state.selectedEdgeId === edgeId ? null : state.selectedEdgeId,
      isDirty: true,
    }));
  },

  // Selection
  selectNode: (nodeId) => {
    set({ selectedNodeId: nodeId, selectedEdgeId: null });
  },

  selectEdge: (edgeId) => {
    set({ selectedEdgeId: edgeId, selectedNodeId: null });
  },

  // Workflow operations
  loadWorkflow: (workflow) => {
    const nodes: Node<WorkflowNodeData>[] = workflow.nodes.map((n: any) => ({
      id: n.node_id,
      type: getNodeType(n.node_type),
      position: { x: n.position_x, y: n.position_y },
      data: {
        label: getNodeLabel(n.node_type, n.builtin_tool),
        nodeType: n.node_type,
        toolId: n.tool_id,
        builtinTool: n.builtin_tool,
        config: n.config || {},
      },
    }));

    const edges: Edge[] = workflow.edges.map((e: any) => ({
      id: e.id,
      source: e.source_node_id,
      target: e.target_node_id,
      sourceHandle: e.source_handle,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
      data: { condition: e.condition },
    }));

    set({
      metadata: {
        id: workflow.id,
        name: workflow.name,
        description: workflow.description || '',
        isActive: workflow.is_active,
        triggerConfig: workflow.trigger_config || { type: 'manual' },
      },
      nodes,
      edges,
      isDirty: false,
      history: [],
      historyIndex: -1,
    });
  },

  applyWorkflowDraft: (workflow) => {
    const nodes: Node<WorkflowNodeData>[] = workflow.nodes.map((n: any) => ({
      id: n.node_id,
      type: getNodeType(n.node_type),
      position: { x: n.position_x, y: n.position_y },
      data: {
        label: getNodeLabel(n.node_type, n.builtin_tool),
        nodeType: n.node_type,
        toolId: n.tool_id,
        builtinTool: n.builtin_tool,
        config: n.config || {},
      },
    }));

    const edges: Edge[] = workflow.edges.map((e: any, index: number) => ({
      id: e.id || `edge_${index}_${e.source_node_id}_${e.target_node_id}`,
      source: e.source_node_id,
      target: e.target_node_id,
      sourceHandle: e.source_handle,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
      data: { condition: e.condition },
    }));

    const layoutNodes = layoutWorkflowNodes(nodes, edges);

    set({
      metadata: {
        id: undefined,
        name: workflow.name || defaultMetadata.name,
        description: workflow.description || '',
        isActive: workflow.is_active ?? true,
        triggerConfig: workflow.trigger_config || { type: 'manual' },
      },
      nodes: layoutNodes,
      edges,
      selectedNodeId: null,
      selectedEdgeId: null,
      isDirty: true,
      history: [],
      historyIndex: -1,
    });
  },

  resetWorkflow: () => {
    set({
      metadata: { ...defaultMetadata },
      nodes: [...initialNodes],
      edges: [...initialEdges],
      selectedNodeId: null,
      selectedEdgeId: null,
      isDirty: false,
      history: [],
      historyIndex: -1,
    });
  },

  getWorkflowData: () => {
    const state = get();
    return {
      nodes: state.nodes.map((n) => ({
        node_id: n.id,
        node_type: n.data.nodeType,
        tool_id: n.data.toolId || null,
        builtin_tool: n.data.builtinTool || null,
        config: n.data.config,
        position_x: Math.round(n.position.x),
        position_y: Math.round(n.position.y),
      })),
      edges: state.edges.map((e) => ({
        source_node_id: e.source,
        target_node_id: e.target,
        source_handle: e.sourceHandle || null,
        condition: (e.data as any)?.condition || null,
      })),
    };
  },

  // History
  saveToHistory: () => {
    const { nodes, edges, history, historyIndex } = get();
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
    });

    // Limit history size
    if (newHistory.length > 50) {
      newHistory.shift();
    }

    set({
      history: newHistory,
      historyIndex: newHistory.length - 1,
    });
  },

  undo: () => {
    const { history, historyIndex } = get();
    if (historyIndex > 0) {
      const prevState = history[historyIndex - 1];
      set({
        nodes: prevState.nodes,
        edges: prevState.edges,
        historyIndex: historyIndex - 1,
        isDirty: true,
      });
    }
  },

  redo: () => {
    const { history, historyIndex } = get();
    if (historyIndex < history.length - 1) {
      const nextState = history[historyIndex + 1];
      set({
        nodes: nextState.nodes,
        edges: nextState.edges,
        historyIndex: historyIndex + 1,
        isDirty: true,
      });
    }
  },

  // State setters
  setIsDirty: (isDirty) => set({ isDirty }),
  setIsSaving: (isSaving) => set({ isSaving }),

  // Tool schemas
  loadToolSchemas: async () => {
    const { toolSchemasLoading, toolSchemas } = get();
    if (toolSchemasLoading || Object.keys(toolSchemas).length > 0) return;

    set({ toolSchemasLoading: true });
    try {
      // Import API dynamically to avoid circular dependencies
      const { default: api } = await import('../../services/api');
      const response = await api.getBuiltinToolSchemas();
      const schemas: Record<string, ToolSchema> = {};
      for (const tool of response.tools) {
        schemas[tool.name] = tool;
      }
      set({ toolSchemas: schemas, toolSchemasLoading: false });
    } catch (error) {
      console.error('Failed to load tool schemas:', error);
      set({ toolSchemasLoading: false });
    }
  },

  getToolSchema: (toolName: string) => {
    return get().toolSchemas[toolName];
  },

  // Context flow computation
  computeContextFlow: () => {
    const { nodes, edges, toolSchemas } = get();
    const contextFlow: Record<string, ContextVariable[]> = {};

    // Build adjacency list (reverse - find what nodes lead to each node)
    const incomingEdges: Record<string, string[]> = {};
    for (const node of nodes) {
      incomingEdges[node.id] = [];
    }
    for (const edge of edges) {
      if (incomingEdges[edge.target]) {
        incomingEdges[edge.target].push(edge.source);
      }
    }

    // Get all ancestors of a node
    const getAncestors = (nodeId: string, visited: Set<string> = new Set()): Set<string> => {
      if (visited.has(nodeId)) return new Set();
      const ancestors = new Set<string>();
      for (const source of (incomingEdges[nodeId] || [])) {
        ancestors.add(source);
        const sourceAncestors = getAncestors(source, new Set([...Array.from(visited), nodeId]));
        sourceAncestors.forEach(a => ancestors.add(a));
      }
      return ancestors;
    };

    // Get outputs from a node
    const getNodeOutputs = (node: Node<WorkflowNodeData>): ContextVariable[] => {
      const outputs: ContextVariable[] = [];
      const outputKey = node.data.config.outputKey || node.id;

      if (node.data.nodeType === 'start' || node.data.nodeType === 'end') {
        return outputs;
      }

      if (node.data.nodeType === 'tool') {
        // Add basic output
        outputs.push({
          path: `context.${outputKey}`,
          type: 'object',
          from_node: node.id,
          description: `Output from ${node.data.builtinTool || 'custom tool'}`,
        });

        // Add known output fields based on tool type
        if (node.data.builtinTool === 'search_documents') {
          outputs.push({
            path: `context.${outputKey}.results`,
            type: 'array',
            from_node: node.id,
            description: 'Search results',
          });
        } else if (node.data.builtinTool === 'web_scrape') {
          outputs.push({
            path: `context.${outputKey}.root_url`,
            type: 'string',
            from_node: node.id,
            description: 'Start URL',
          });
          outputs.push({
            path: `context.${outputKey}.pages`,
            type: 'array',
            from_node: node.id,
            description: 'Scraped pages',
          });
        } else if (node.data.builtinTool === 'ingest_url') {
          outputs.push({
            path: `context.${outputKey}.created`,
            type: 'array',
            from_node: node.id,
            description: 'Created documents',
          });
          outputs.push({
            path: `context.${outputKey}.updated`,
            type: 'array',
            from_node: node.id,
            description: 'Updated documents',
          });
        } else if (node.data.builtinTool === 'get_document_details') {
          outputs.push({
            path: `context.${outputKey}.title`,
            type: 'string',
            from_node: node.id,
            description: 'Document title',
          });
          outputs.push({
            path: `context.${outputKey}.content`,
            type: 'string',
            from_node: node.id,
            description: 'Document content',
          });
        }
      } else if (node.data.nodeType === 'condition') {
        outputs.push({
          path: `context.${outputKey}.condition_result`,
          type: 'boolean',
          from_node: node.id,
          description: 'Condition evaluation result',
        });
      } else if (node.data.nodeType === 'loop') {
        outputs.push({
          path: 'loop.item',
          type: 'any',
          from_node: node.id,
          description: 'Current loop item',
        });
        outputs.push({
          path: 'loop.index',
          type: 'integer',
          from_node: node.id,
          description: 'Current loop index',
        });
        outputs.push({
          path: `context.${outputKey}.results`,
          type: 'array',
          from_node: node.id,
          description: 'Loop results',
        });
      } else if (node.data.nodeType === 'parallel') {
        outputs.push({
          path: `context.${outputKey}.parallel_results`,
          type: 'array',
          from_node: node.id,
          description: 'Parallel branch results',
        });
      } else if (node.data.nodeType === 'switch') {
        outputs.push({
          path: `context.${outputKey}.switch_value`,
          type: 'string',
          from_node: node.id,
          description: 'Value that was evaluated by switch',
        });
        outputs.push({
          path: `context.${outputKey}.matched_case`,
          type: 'string',
          from_node: node.id,
          description: 'The case that was matched (case_0, case_1, etc. or default)',
        });
      } else if (node.data.nodeType === 'subworkflow') {
        outputs.push({
          path: `context.${outputKey}`,
          type: 'object',
          from_node: node.id,
          description: 'Sub-workflow execution result',
        });
        outputs.push({
          path: `context.${outputKey}.status`,
          type: 'string',
          from_node: node.id,
          description: 'Sub-workflow execution status',
        });
        outputs.push({
          path: `context.${outputKey}.output`,
          type: 'object',
          from_node: node.id,
          description: 'Sub-workflow final context/output',
        });
      }

      return outputs;
    };

    // Build node map
    const nodeMap: Record<string, Node<WorkflowNodeData>> = {};
    for (const node of nodes) {
      nodeMap[node.id] = node;
    }

    // Compute available context for each node
    for (const node of nodes) {
      const ancestors = getAncestors(node.id);
      const available: ContextVariable[] = [
        // Trigger data is always available
        {
          path: 'context.trigger_data',
          type: 'object',
          from_node: '_trigger',
          description: 'Data from workflow trigger',
        },
      ];

      // Add outputs from all ancestors
      Array.from(ancestors).forEach((ancestorId) => {
        const ancestorNode = nodeMap[ancestorId];
        if (ancestorNode) {
          available.push(...getNodeOutputs(ancestorNode));
        }
      });

      contextFlow[node.id] = available;
    }

    set({ contextFlow });
  },

  getAvailableContext: (nodeId: string) => {
    const { contextFlow } = get();
    return contextFlow[nodeId] || [];
  },
}));

// Helper functions
function getNodeType(nodeType: string): string {
  switch (nodeType) {
    case 'start': return 'startNode';
    case 'end': return 'endNode';
    case 'tool': return 'toolNode';
    case 'condition': return 'conditionNode';
    case 'parallel': return 'parallelNode';
    case 'loop': return 'loopNode';
    case 'wait': return 'waitNode';
    case 'switch': return 'switchNode';
    case 'subworkflow': return 'subworkflowNode';
    default: return 'toolNode';
  }
}

function getNodeLabel(nodeType: string, builtinTool?: string): string {
  if (builtinTool) {
    return builtinTool.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  }
  switch (nodeType) {
    case 'start': return 'Start';
    case 'end': return 'End';
    case 'condition': return 'Condition';
    case 'parallel': return 'Parallel';
    case 'loop': return 'Loop';
    case 'wait': return 'Wait';
    case 'switch': return 'Switch';
    case 'subworkflow': return 'Sub-Workflow';
    default: return 'Tool';
  }
}
