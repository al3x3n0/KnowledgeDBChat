/**
 * End node - exit point for workflow.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Square } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

const EndNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  return (
    <div
      className={`px-4 py-3 rounded-full border-2 shadow-md cursor-pointer transition-all ${
        selected
          ? 'border-red-500 ring-2 ring-red-300 bg-red-50'
          : 'border-red-400 hover:border-red-500 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Only target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-red-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <Square className="w-5 h-5 text-red-600" />
        <span className="font-medium text-sm text-gray-800">{data.label}</span>
      </div>
    </div>
  );
};

export default EndNode;
