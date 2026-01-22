/**
 * Loop node - iterates over a collection.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Repeat } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

const LoopNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  const loopSource = data.config.loopSource || '{{items}}';
  const maxIterations = data.config.maxIterations || 100;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[150px] cursor-pointer transition-all ${
        selected
          ? 'border-cyan-500 ring-2 ring-cyan-300 bg-cyan-50'
          : 'border-cyan-300 hover:border-cyan-400 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-cyan-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <Repeat className="w-5 h-5 text-cyan-600" />
        <div>
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          <div className="text-xs text-gray-500 truncate max-w-[100px]" title={loopSource}>
            {loopSource}
          </div>
          <div className="text-xs text-gray-400">Max: {maxIterations}</div>
        </div>
      </div>

      {/* Loop body handle */}
      <div className="text-xs text-center mt-2 text-cyan-600">Loop Body</div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="body"
        className="!w-3 !h-3 !bg-cyan-500 !border-2 !border-white"
        style={{ left: '30%' }}
      />

      {/* Continue after loop handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="continue"
        className="!w-3 !h-3 !bg-cyan-700 !border-2 !border-white"
        style={{ left: '70%' }}
      />
    </div>
  );
};

export default LoopNode;
