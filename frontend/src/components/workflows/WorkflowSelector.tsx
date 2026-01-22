/**
 * Workflow Selector Component.
 *
 * Dropdown for selecting a workflow (used in sub-workflow node configuration).
 */

import React, { useState, useEffect } from 'react';
import apiClient from '../../services/api';

interface WorkflowOption {
  id: string;
  name: string;
  description?: string;
  is_active: boolean;
}

interface WorkflowSelectorProps {
  value?: string;
  onChange: (id: string) => void;
  excludeId?: string;
  disabled?: boolean;
}

const WorkflowSelector: React.FC<WorkflowSelectorProps> = ({
  value,
  onChange,
  excludeId,
  disabled = false,
}) => {
  const [workflows, setWorkflows] = useState<WorkflowOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchWorkflows = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await apiClient.getWorkflowsForSelection(excludeId);
        setWorkflows(data);
      } catch (err) {
        setError('Failed to load workflows');
        console.error('Error loading workflows:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchWorkflows();
  }, [excludeId]);

  if (loading) {
    return (
      <select
        disabled
        className="w-full px-2 py-1 text-sm border rounded bg-gray-100 text-gray-500"
      >
        <option>Loading workflows...</option>
      </select>
    );
  }

  if (error) {
    return (
      <div className="text-xs text-red-500">{error}</div>
    );
  }

  return (
    <select
      value={value || ''}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="w-full px-2 py-1 text-sm border rounded focus:ring-1 focus:ring-primary-500 disabled:bg-gray-100 disabled:text-gray-500"
    >
      <option value="">Select a workflow...</option>
      {workflows.map((wf) => (
        <option key={wf.id} value={wf.id}>
          {wf.name}
        </option>
      ))}
    </select>
  );
};

export default WorkflowSelector;
