/**
 * Start node - entry point for workflow.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Play } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

const StartNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  return (
    <div
      className={`px-4 py-3 rounded-full border-2 shadow-md cursor-pointer transition-all ${
        selected
          ? 'border-green-500 ring-2 ring-green-300 bg-green-50'
          : 'border-green-400 hover:border-green-500 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      <div className="flex items-center space-x-2">
        <Play className="w-5 h-5 text-green-600" />
        <span className="font-medium text-sm text-gray-800">{data.label}</span>
      </div>

      {/* Only source handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-green-500 !border-2 !border-white"
      />
    </div>
  );
};

export default StartNode;
