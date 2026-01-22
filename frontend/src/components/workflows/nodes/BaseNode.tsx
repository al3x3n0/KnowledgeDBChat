/**
 * Base node component with common styling and functionality.
 */

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';

interface BaseNodeProps extends NodeProps<WorkflowNodeData> {
  icon: React.ReactNode;
  color: string;
  showSourceHandle?: boolean;
  showTargetHandle?: boolean;
  sourceHandles?: { id: string; label: string; position?: number }[];
}

const BaseNode: React.FC<BaseNodeProps> = ({
  id,
  data,
  selected,
  icon,
  color,
  showSourceHandle = true,
  showTargetHandle = true,
  sourceHandles,
}) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[150px] cursor-pointer transition-all ${
        selected
          ? `border-${color}-500 ring-2 ring-${color}-300`
          : `border-${color}-300 hover:border-${color}-400`
      }`}
      style={{
        backgroundColor: 'white',
        borderColor: selected ? undefined : `var(--${color}-300, #d1d5db)`,
      }}
      onClick={() => selectNode(id)}
    >
      {/* Target handle (top) */}
      {showTargetHandle && (
        <Handle
          type="target"
          position={Position.Top}
          className={`!w-3 !h-3 !bg-${color}-500 !border-2 !border-white`}
          style={{ backgroundColor: `var(--${color}-500, #6b7280)` }}
        />
      )}

      {/* Node content */}
      <div className="flex items-center space-x-2">
        <div className={`text-${color}-600`} style={{ color: `var(--${color}-600, #4b5563)` }}>
          {icon}
        </div>
        <div>
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          {data.builtinTool && (
            <div className="text-xs text-gray-500">{data.builtinTool}</div>
          )}
        </div>
      </div>

      {/* Source handles */}
      {showSourceHandle && !sourceHandles && (
        <Handle
          type="source"
          position={Position.Bottom}
          className={`!w-3 !h-3 !bg-${color}-500 !border-2 !border-white`}
          style={{ backgroundColor: `var(--${color}-500, #6b7280)` }}
        />
      )}

      {/* Multiple source handles for branching nodes */}
      {sourceHandles?.map((handle, index) => (
        <Handle
          key={handle.id}
          type="source"
          position={Position.Bottom}
          id={handle.id}
          className={`!w-3 !h-3 !bg-${color}-500 !border-2 !border-white`}
          style={{
            backgroundColor: `var(--${color}-500, #6b7280)`,
            left: `${((index + 1) / (sourceHandles.length + 1)) * 100}%`,
          }}
        />
      ))}
    </div>
  );
};

export default BaseNode;
