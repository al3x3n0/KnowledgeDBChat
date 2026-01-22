/**
 * Sub-Workflow node - executes another workflow as part of this workflow.
 */

import React, { useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Workflow } from 'lucide-react';
import { useWorkflowStore, WorkflowNodeData } from '../useWorkflowStore';
import apiClient from '../../../services/api';

const SubWorkflowNode: React.FC<NodeProps<WorkflowNodeData>> = ({ id, data, selected }) => {
  const selectNode = useWorkflowStore((state) => state.selectNode);
  const [workflowName, setWorkflowName] = useState<string>('');
  const [loading, setLoading] = useState(false);

  // Fetch workflow name when workflow_id changes
  useEffect(() => {
    const workflowId = data.config.workflow_id;
    if (workflowId) {
      setLoading(true);
      apiClient.getWorkflow(workflowId)
        .then((res) => {
          setWorkflowName(res.name || 'Unknown');
        })
        .catch(() => {
          setWorkflowName('Not found');
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setWorkflowName('');
    }
  }, [data.config.workflow_id]);

  const timeoutSeconds = data.config.timeout_seconds || 300;
  const onError = data.config.on_error || 'fail';

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[160px] cursor-pointer transition-all ${
        selected
          ? 'border-indigo-500 ring-2 ring-indigo-300 bg-indigo-50'
          : 'border-indigo-300 hover:border-indigo-400 bg-white'
      }`}
      onClick={() => selectNode(id)}
    >
      {/* Target handle */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-indigo-500 !border-2 !border-white"
      />

      <div className="flex items-center space-x-2">
        <Workflow className="w-5 h-5 text-indigo-600" />
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm text-gray-800">{data.label}</div>
          <div className="text-xs text-gray-500 truncate">
            {loading ? 'Loading...' : (workflowName || 'Select workflow...')}
          </div>
        </div>
      </div>

      {/* Config summary */}
      <div className="mt-1 text-xs text-gray-400 flex gap-2">
        <span>{timeoutSeconds}s timeout</span>
        {onError === 'continue' && <span className="text-amber-500">continue on error</span>}
      </div>

      {/* Source handle */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-indigo-500 !border-2 !border-white"
      />
    </div>
  );
};

export default SubWorkflowNode;
