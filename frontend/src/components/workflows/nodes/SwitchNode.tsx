/**
 * Switch node - multi-way branching based on value matching.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { GitMerge } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

interface SwitchCase {
  value: string;
  label: string;
}

const SwitchNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  const cases: SwitchCase[] = data.config.cases || [];
  const totalHandles = cases.length + 1; // cases + default

  // Calculate handle positions
  const getHandlePosition = (index: number) => {
    return `${((index + 1) / (totalHandles + 1)) * 100}%`;
  };

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[160px] cursor-pointer transition-all ${
        selected
          ? 'border-orange-500 ring-2 ring-orange-300 bg-orange-50'
          : 'border-orange-300 hover:border-orange-400 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-orange-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <GitMerge className="w-5 h-5 text-orange-600" />
        <div>
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          <div className="text-xs text-gray-500">
            {cases.length} case{cases.length !== 1 ? 's' : ''} + default
          </div>
        </div>
      </div>

      {/* Case labels */}
      <div className="flex justify-between mt-2 text-xs px-1 gap-1 flex-wrap">
        {cases.map((c, idx) => (
          <span key={idx} className="text-orange-600 truncate max-w-[60px]" title={c.label || c.value}>
            {c.label || c.value}
          </span>
        ))}
        <span className="text-gray-500">default</span>
      </div>

      {/* Dynamic source handles for each case */}
      {cases.map((_, idx) => (
        <Handle
          key={`case_${idx}`}
          type="source"
          position={Position.Bottom}
          id={`case_${idx}`}
          className="!w-3 !h-3 !bg-orange-500 !border-2 !border-white"
          style={{ left: getHandlePosition(idx) }}
        />
      ))}

      {/* Default handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="default"
        className="!w-3 !h-3 !bg-gray-400 !border-2 !border-white"
        style={{ left: getHandlePosition(cases.length) }}
      />
    </div>
  );
};

export default SwitchNode;
