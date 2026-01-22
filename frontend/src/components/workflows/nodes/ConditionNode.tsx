/**
 * Condition node - branches workflow based on condition.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { GitBranch } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

const ConditionNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  const conditionType = data.config.condition?.type || 'truthy';

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[140px] cursor-pointer transition-all ${
        selected
          ? 'border-amber-500 ring-2 ring-amber-300 bg-amber-50'
          : 'border-amber-300 hover:border-amber-400 bg-white'
      }`}
      style={{ transform: 'rotate(0deg)' }}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-amber-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <GitBranch className="w-5 h-5 text-amber-600" />
        <div>
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          <div className="text-xs text-gray-500">{conditionType}</div>
        </div>
      </div>

      {/* True/False source handles */}
      <div className="flex justify-between mt-2 text-xs">
        <span className="text-green-600">True</span>
        <span className="text-red-600">False</span>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        className="!w-3 !h-3 !bg-green-500 !border-2 !border-white"
        style={{ left: '25%' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        className="!w-3 !h-3 !bg-red-500 !border-2 !border-white"
        style={{ left: '75%' }}
      />
    </div>
  );
};

export default ConditionNode;
