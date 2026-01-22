/**
 * Main Workflow Editor Component with React Flow canvas.
 *
 * Provides:
 * - Drag and drop node placement
 * - Connection handling
 * - Keyboard shortcuts
 * - Zoom and pan controls
 */

import React, { useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Undo2, Redo2, Save, Trash2, ZoomIn, ZoomOut } from 'lucide-react';
import toast from 'react-hot-toast';

import { useWorkflowStore, WorkflowNodeData } from './useWorkflowStore';
import { nodeTypes } from './nodes';
import WorkflowSidebar from './WorkflowSidebar';

// Edge options
const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: false,
};

interface WorkflowEditorProps {
  onSave: () => Promise<void>;
}

const WorkflowEditorInner: React.FC<WorkflowEditorProps> = ({ onSave }) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { project, fitView, zoomIn, zoomOut } = useReactFlow();

  // Store state and actions
  const nodes = useWorkflowStore((state) => state.nodes);
  const edges = useWorkflowStore((state) => state.edges);
  const selectedNodeId = useWorkflowStore((state) => state.selectedNodeId);
  const isDirty = useWorkflowStore((state) => state.isSaving);
  const isSaving = useWorkflowStore((state) => state.isSaving);

  const onNodesChange = useWorkflowStore((state) => state.onNodesChange);
  const onEdgesChange = useWorkflowStore((state) => state.onEdgesChange);
  const onConnect = useWorkflowStore((state) => state.onConnect);
  const addNode = useWorkflowStore((state) => state.addNode);
  const deleteNode = useWorkflowStore((state) => state.deleteNode);
  const selectNode = useWorkflowStore((state) => state.selectNode);
  const undo = useWorkflowStore((state) => state.undo);
  const redo = useWorkflowStore((state) => state.redo);
  const saveToHistory = useWorkflowStore((state) => state.saveToHistory);

  // Handle drop from sidebar
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (!reactFlowWrapper.current) return;

      const type = event.dataTransfer.getData('application/reactflow/type');
      const label = event.dataTransfer.getData('application/reactflow/label');
      const builtinTool = event.dataTransfer.getData('application/reactflow/builtinTool');

      if (!type) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = project({
        x: event.clientX - bounds.left,
        y: event.clientY - bounds.top,
      });

      const nodeId = `${type}_${Date.now()}`;

      const newNode = {
        id: nodeId,
        type: `${type}Node`,
        position,
        data: {
          label: label || type.charAt(0).toUpperCase() + type.slice(1),
          nodeType: type,
          builtinTool: builtinTool || undefined,
          config: {},
        } as WorkflowNodeData,
      };

      saveToHistory();
      addNode(newNode);
      selectNode(nodeId);
    },
    [project, addNode, selectNode, saveToHistory]
  );

  // Keyboard shortcuts
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      // Delete selected node
      if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNodeId) {
        if (selectedNodeId !== 'start' && selectedNodeId !== 'end') {
          deleteNode(selectedNodeId);
        }
      }

      // Undo: Ctrl+Z
      if (event.ctrlKey && event.key === 'z' && !event.shiftKey) {
        event.preventDefault();
        undo();
      }

      // Redo: Ctrl+Shift+Z or Ctrl+Y
      if (event.ctrlKey && (event.shiftKey && event.key === 'z' || event.key === 'y')) {
        event.preventDefault();
        redo();
      }

      // Save: Ctrl+S
      if (event.ctrlKey && event.key === 's') {
        event.preventDefault();
        onSave();
      }
    },
    [selectedNodeId, deleteNode, undo, redo, onSave]
  );

  // Node click handler
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: any) => {
      selectNode(node.id);
    },
    [selectNode]
  );

  // Pane click handler (deselect)
  const onPaneClick = useCallback(() => {
    selectNode(null);
  }, [selectNode]);

  // Handle save
  const handleSave = async () => {
    try {
      await onSave();
      toast.success('Workflow saved');
    } catch (error: any) {
      toast.error(error.message || 'Failed to save workflow');
    }
  };

  // Memoize node types to prevent recreation
  const memoizedNodeTypes = useMemo(() => nodeTypes, []);

  return (
    <div className="flex h-full" onKeyDown={onKeyDown} tabIndex={0}>
      {/* Sidebar */}
      <WorkflowSidebar />

      {/* Canvas */}
      <div ref={reactFlowWrapper} className="flex-1 h-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodeTypes={memoizedNodeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          fitView
          snapToGrid
          snapGrid={[15, 15]}
          deleteKeyCode={['Delete', 'Backspace']}
          minZoom={0.2}
          maxZoom={2}
        >
          <Background gap={15} size={1} color="#e5e7eb" />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              switch (node.type) {
                case 'startNode': return '#22c55e';
                case 'endNode': return '#ef4444';
                case 'toolNode': return '#3b82f6';
                case 'conditionNode': return '#f59e0b';
                case 'parallelNode': return '#a855f7';
                case 'loopNode': return '#06b6d4';
                case 'waitNode': return '#6b7280';
                default: return '#6b7280';
              }
            }}
            maskColor="#f3f4f6aa"
          />

          {/* Toolbar */}
          <Panel position="top-right" className="flex space-x-2">
            <button
              onClick={undo}
              className="p-2 bg-white rounded-lg shadow border hover:bg-gray-50"
              title="Undo (Ctrl+Z)"
            >
              <Undo2 className="w-4 h-4" />
            </button>
            <button
              onClick={redo}
              className="p-2 bg-white rounded-lg shadow border hover:bg-gray-50"
              title="Redo (Ctrl+Y)"
            >
              <Redo2 className="w-4 h-4" />
            </button>
            <button
              onClick={() => zoomIn()}
              className="p-2 bg-white rounded-lg shadow border hover:bg-gray-50"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => zoomOut()}
              className="p-2 bg-white rounded-lg shadow border hover:bg-gray-50"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center space-x-1 px-3 py-2 bg-primary-600 text-white rounded-lg shadow hover:bg-primary-700 disabled:opacity-50"
              title="Save (Ctrl+S)"
            >
              <Save className="w-4 h-4" />
              <span>{isSaving ? 'Saving...' : 'Save'}</span>
            </button>
          </Panel>
        </ReactFlow>
      </div>
    </div>
  );
};

// Wrap with ReactFlowProvider
const WorkflowEditor: React.FC<WorkflowEditorProps> = (props) => {
  return (
    <ReactFlowProvider>
      <WorkflowEditorInner {...props} />
    </ReactFlowProvider>
  );
};

export default WorkflowEditor;
