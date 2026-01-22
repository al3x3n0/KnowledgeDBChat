/**
 * Parallel node - forks workflow into parallel branches.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { GitFork } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

const ParallelNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[140px] cursor-pointer transition-all ${
        selected
          ? 'border-purple-500 ring-2 ring-purple-300 bg-purple-50'
          : 'border-purple-300 hover:border-purple-400 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-purple-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <GitFork className="w-5 h-5 text-purple-600" />
        <div>
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          <div className="text-xs text-gray-500">Parallel branches</div>
        </div>
      </div>

      {/* Multiple source handles for parallel branches */}
      <div className="flex justify-center space-x-4 mt-2">
        <Handle
          type="source"
          position={Position.Bottom}
          id="branch1"
          className="!w-3 !h-3 !bg-purple-500 !border-2 !border-white !relative !left-auto !transform-none"
        />
        <Handle
          type="source"
          position={Position.Bottom}
          id="branch2"
          className="!w-3 !h-3 !bg-purple-500 !border-2 !border-white !relative !left-auto !transform-none"
        />
      </div>
    </div>
  );
};

export default ParallelNode;
