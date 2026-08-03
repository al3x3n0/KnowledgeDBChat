/**
 * Workflows List Page.
 *
 * Displays all user workflows with CRUD operations.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Plus,
  Play,
  Pause,
  Trash2,
  Edit,
  Clock,
  Workflow,
  MoreVertical,
  Search,
  Calendar,
  Loader2,
} from 'lucide-react';
import toast from 'react-hot-toast';
import api from '../services/api';

interface WorkflowItem {
  id: string;
  name: string;
  description: string | null;
  is_active: boolean;
  trigger_config: {
    type: string;
    schedule?: string;
    event?: string;
  };
  node_count: number;
  execution_count: number;
  created_at: string;
  updated_at: string;
}

interface ExecutionItem {
  id: string;
  workflow_id: string;
  workflow_name: string | null;
  trigger_type: string;
  status: string;
  progress: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  current_node_id?: string | null;
  context?: Record<string, any>;
}

const WorkflowsPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const selectedExecutionId = String(searchParams.get('executionId') || '').trim();
  const [workflows, setWorkflows] = useState<WorkflowItem[]>([]);
  const [selectedExecution, setSelectedExecution] = useState<ExecutionItem | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);

  // Load workflows
  useEffect(() => {
    loadWorkflows();
  }, []);

  useEffect(() => {
    if (!selectedExecutionId) {
      setSelectedExecution(null);
      return;
    }
    loadExecution(selectedExecutionId);
  }, [selectedExecutionId]);

  const loadWorkflows = async () => {
    setIsLoading(true);
    try {
      const response = await api.get('/workflows');
      setWorkflows(response.data.workflows || []);
    } catch (error: any) {
      toast.error('Failed to load workflows');
    } finally {
      setIsLoading(false);
    }
  };

  const loadExecution = async (executionId: string) => {
    try {
      const response = await api.get(`/workflows/executions/${executionId}`);
      const payload = response.data || {};
      setSelectedExecution({
        id: String(payload.id || ''),
        workflow_id: String(payload.workflow_id || ''),
        workflow_name: null,
        trigger_type: String(payload.trigger_type || ''),
        status: String(payload.status || ''),
        progress: Number(payload.progress || 0),
        error: payload.error || null,
        created_at: String(payload.created_at || ''),
        started_at: payload.started_at || null,
        completed_at: payload.completed_at || null,
        current_node_id: payload.current_node_id || null,
        context: payload.context || {},
      });
    } catch (_error: any) {
      setSelectedExecution(null);
      toast.error('Workflow execution could not be loaded');
    }
  };

  const createWorkflow = async () => {
    try {
      const response = await api.post('/workflows', {
        name: 'New Workflow',
        description: '',
        is_active: false,
        trigger_config: { type: 'manual' },
        nodes: [
          {
            node_id: 'start',
            node_type: 'start',
            config: {},
            position_x: 250,
            position_y: 50,
          },
          {
            node_id: 'end',
            node_type: 'end',
            config: {},
            position_x: 250,
            position_y: 350,
          },
        ],
        edges: [],
      });
      navigate(`/workflows/${response.data.id}/edit`);
    } catch (error: any) {
      toast.error('Failed to create workflow');
    }
  };

  const deleteWorkflow = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this workflow?')) return;

    try {
      await api.delete(`/workflows/${id}`);
      toast.success('Workflow deleted');
      loadWorkflows();
    } catch (error: any) {
      toast.error('Failed to delete workflow');
    }
    setActiveMenuId(null);
  };

  const toggleActive = async (id: string, isActive: boolean) => {
    try {
      await api.put(`/workflows/${id}`, { is_active: !isActive });
      loadWorkflows();
    } catch (error: any) {
      toast.error('Failed to update workflow');
    }
    setActiveMenuId(null);
  };

  const executeWorkflow = async (id: string) => {
    try {
      await api.post(`/workflows/${id}/execute/async`, {
        trigger_type: 'manual',
        trigger_data: {},
        inputs: {},
      });
      toast.success('Workflow execution started');
    } catch (error: any) {
      toast.error('Failed to start workflow');
    }
    setActiveMenuId(null);
  };

  const filteredWorkflows = workflows.filter(
    (wf) =>
      wf.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      wf.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getTriggerLabel = (config: WorkflowItem['trigger_config']) => {
    switch (config.type) {
      case 'manual': return 'Manual';
      case 'schedule': return `Scheduled: ${config.schedule}`;
      case 'event': return `Event: ${config.event}`;
      case 'webhook': return 'Webhook';
      default: return config.type;
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const selectedExecutionWorkflow = selectedExecution
    ? workflows.find((workflow) => workflow.id === selectedExecution.workflow_id) || null
    : null;

  return (
    <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Workflows</h1>
            <p className="text-gray-600">Automate document operations with visual workflows</p>
          </div>
          <button
            onClick={createWorkflow}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            <Plus className="w-5 h-5" />
            <span>New Workflow</span>
          </button>
        </div>

        {/* Search */}
        <div className="relative mb-6">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search workflows..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
          />
        </div>

        {selectedExecutionId && (
          <div className="mb-6 rounded-lg border border-sky-200 bg-sky-50 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-sky-700">Execution Drilldown</div>
                {selectedExecution ? (
                  <>
                    <h2 className="mt-2 text-lg font-semibold text-slate-900">
                      {selectedExecutionWorkflow?.name || 'Workflow execution'} · {selectedExecution.id}
                    </h2>
                    <div className="mt-2 text-sm text-slate-700">
                      Status: <span className="font-medium">{selectedExecution.status}</span> · Trigger: {selectedExecution.trigger_type} · Progress: {selectedExecution.progress}%
                    </div>
                    <div className="mt-1 text-sm text-slate-600">
                      Current node: {selectedExecution.current_node_id || '—'} · Started: {selectedExecution.started_at ? formatDate(selectedExecution.started_at) : '—'}
                    </div>
                    {selectedExecution.error ? (
                      <div className="mt-2 text-sm text-rose-700">Error: {selectedExecution.error}</div>
                    ) : null}
                  </>
                ) : (
                  <div className="mt-2 text-sm text-slate-600">Execution not found or no longer available.</div>
                )}
              </div>
              {selectedExecutionWorkflow ? (
                <button
                  type="button"
                  onClick={() => navigate(`/workflows/${selectedExecutionWorkflow.id}/executions`)}
                  className="rounded-lg border border-sky-300 px-3 py-2 text-sm font-medium text-sky-800 hover:border-sky-500"
                >
                  Open executions
                </button>
              ) : null}
            </div>
          </div>
        )}

        {/* Loading state */}
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          </div>
        ) : filteredWorkflows.length === 0 ? (
          <div className="text-center py-16">
            <Workflow className="w-16 h-16 mx-auto text-gray-300 mb-4" />
            <h3 className="text-lg font-medium text-gray-700">No workflows yet</h3>
            <p className="text-gray-500 mb-4">Create your first workflow to automate tasks</p>
            <button
              onClick={createWorkflow}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
            >
              <Plus className="w-5 h-5" />
              <span>Create Workflow</span>
            </button>
          </div>
        ) : (
          <div className="grid gap-4">
            {filteredWorkflows.map((workflow) => (
              <div
                key={workflow.id}
                className={`bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow ${
                  selectedExecution?.workflow_id === workflow.id ? 'border-sky-400 ring-2 ring-sky-100' : ''
                }`}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between">
                    {/* Workflow info */}
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h3 className="font-medium text-gray-900">{workflow.name}</h3>
                        <span
                          className={`px-2 py-0.5 text-xs rounded-full ${
                            workflow.is_active
                              ? 'bg-green-100 text-green-700'
                              : 'bg-gray-100 text-gray-600'
                          }`}
                        >
                          {workflow.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      {workflow.description && (
                        <p className="text-sm text-gray-500 mt-1">{workflow.description}</p>
                      )}

                      <div className="flex items-center space-x-4 mt-2 text-sm text-gray-500">
                        <span className="flex items-center space-x-1">
                          <Clock className="w-4 h-4" />
                          <span>{getTriggerLabel(workflow.trigger_config)}</span>
                        </span>
                        <span>{workflow.node_count} nodes</span>
                        <span>{workflow.execution_count} executions</span>
                        <span>Updated {formatDate(workflow.updated_at)}</span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-2">
                      {workflow.is_active && (
                        <button
                          onClick={() => executeWorkflow(workflow.id)}
                          className="p-2 text-green-600 hover:bg-green-50 rounded"
                          title="Run workflow"
                        >
                          <Play className="w-5 h-5" />
                        </button>
                      )}
                      <button
                        onClick={() => navigate(`/workflows/${workflow.id}/edit`)}
                        className="p-2 text-blue-600 hover:bg-blue-50 rounded"
                        title="Edit workflow"
                      >
                        <Edit className="w-5 h-5" />
                      </button>

                      {/* More menu */}
                      <div className="relative">
                        <button
                          onClick={() => setActiveMenuId(activeMenuId === workflow.id ? null : workflow.id)}
                          className="p-2 text-gray-500 hover:bg-gray-100 rounded"
                        >
                          <MoreVertical className="w-5 h-5" />
                        </button>

                        {activeMenuId === workflow.id && (
                          <div className="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border z-10">
                            <button
                              onClick={() => toggleActive(workflow.id, workflow.is_active)}
                              className="w-full flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                            >
                              {workflow.is_active ? (
                                <>
                                  <Pause className="w-4 h-4" />
                                  <span>Deactivate</span>
                                </>
                              ) : (
                                <>
                                  <Play className="w-4 h-4" />
                                  <span>Activate</span>
                                </>
                              )}
                            </button>
                            <button
                              onClick={() => navigate(`/workflows/${workflow.id}/executions`)}
                              className="w-full flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                            >
                              <Calendar className="w-4 h-4" />
                              <span>View Executions</span>
                            </button>
                            <hr className="my-1" />
                            <button
                              onClick={() => deleteWorkflow(workflow.id)}
                              className="w-full flex items-center space-x-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                            >
                              <Trash2 className="w-4 h-4" />
                              <span>Delete</span>
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

      {/* Click outside to close menu */}
      {activeMenuId && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setActiveMenuId(null)}
        />
      )}
    </div>
  );
};

export default WorkflowsPage;
