/**
 * Wait node - pauses workflow execution.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Clock } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

const WaitNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  const waitSeconds = data.config.waitSeconds || 0;

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[120px] cursor-pointer transition-all ${
        selected
          ? 'border-gray-500 ring-2 ring-gray-300 bg-gray-50'
          : 'border-gray-300 hover:border-gray-400 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-gray-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <Clock className="w-5 h-5 text-gray-600" />
        <div>
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          <div className="text-xs text-gray-500">{formatDuration(waitSeconds)}</div>
        </div>
      </div>

      {/* Source handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-gray-500 !border-2 !border-white"
      />
    </div>
  );
};

export default WaitNode;
