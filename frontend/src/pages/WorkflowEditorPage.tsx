/**
 * Workflow Editor Page.
 *
 * Full-page visual workflow editor with React Flow.
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Settings,
  Save,
  Play,
  Loader2,
  Sparkles,
  X,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import toast from 'react-hot-toast';

import api from '../services/api';
import WorkflowEditor from '../components/workflows/WorkflowEditor';
import { useWorkflowStore } from '../components/workflows/useWorkflowStore';

interface ValidationIssue {
  severity: 'error' | 'warning' | 'info';
  node_id?: string;
  field?: string;
  message: string;
}

const WorkflowEditorPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [showValidation, setShowValidation] = useState(false);
  const [showSynthesis, setShowSynthesis] = useState(false);
  const [synthesisDescription, setSynthesisDescription] = useState('');
  const [synthesisName, setSynthesisName] = useState('');
  const [isSynthesizing, setIsSynthesizing] = useState(false);

  // Store state and actions
  const metadata = useWorkflowStore((state) => state.metadata);
  const isDirty = useWorkflowStore((state) => state.isDirty);
  const setMetadata = useWorkflowStore((state) => state.setMetadata);
  const loadWorkflow = useWorkflowStore((state) => state.loadWorkflow);
  const applyWorkflowDraft = useWorkflowStore((state) => state.applyWorkflowDraft);
  const resetWorkflow = useWorkflowStore((state) => state.resetWorkflow);
  const getWorkflowData = useWorkflowStore((state) => state.getWorkflowData);
  const setIsSaving = useWorkflowStore((state) => state.setIsSaving);
  const setIsDirty = useWorkflowStore((state) => state.setIsDirty);

  // Load workflow on mount
  useEffect(() => {
    const loadData = async () => {
      if (!id) {
        resetWorkflow();
        // Load a draft workflow passed from the agent (or other UI) via localStorage.
        // This enables: chat prompt → proposed workflow → user approves → saved in editor.
        try {
          const rawDraft = localStorage.getItem('workflow_draft_pending');
          if (rawDraft) {
            const draft = JSON.parse(rawDraft);
            applyWorkflowDraft(draft);
            localStorage.removeItem('workflow_draft_pending');
            toast.success('Loaded workflow draft');
          }
        } catch {
          // Ignore invalid drafts
          localStorage.removeItem('workflow_draft_pending');
        }
        setIsLoading(false);
        return;
      }

      try {
        const response = await api.get(`/workflows/${id}`);
        loadWorkflow(response.data);
      } catch (error: any) {
        toast.error('Failed to load workflow');
        navigate('/workflows');
      } finally {
        setIsLoading(false);
      }
    };

    loadData();

    // Cleanup on unmount
    return () => {
      resetWorkflow();
    };
  }, [id]);

  // Warn before leaving with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  // Save workflow
  const handleSave = async () => {
    setIsSaving(true);
    setValidationIssues([]);

    try {
      const { nodes, edges } = getWorkflowData();

      const payload = {
        name: metadata.name,
        description: metadata.description,
        is_active: metadata.isActive,
        trigger_config: metadata.triggerConfig,
        nodes,
        edges,
      };

      let workflowId = metadata.id;

      // Save first to get an ID for validation
      if (workflowId) {
        await api.put(`/workflows/${workflowId}`, payload);
      } else {
        const response = await api.post('/workflows', payload);
        workflowId = response.data.id;
        setMetadata({ id: workflowId });
        navigate(`/workflows/${workflowId}/edit`, { replace: true });
      }

      // Validate the workflow after saving
      try {
        const validation = await api.validateWorkflow(workflowId!);
        setValidationIssues(validation.issues);

        if (!validation.valid) {
          setShowValidation(true);
          toast.error('Workflow saved but has validation errors');
        } else if (validation.issues.length > 0) {
          setShowValidation(true);
          toast('Workflow saved with warnings', { icon: '⚠️' });
        } else {
          toast.success('Workflow saved');
        }
      } catch (validationError) {
        // Validation failed but save succeeded
        toast.success('Workflow saved');
        console.warn('Validation check failed:', validationError);
      }

      setIsDirty(false);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save workflow');
      throw error;
    } finally {
      setIsSaving(false);
    }
  };

  // Execute workflow
  const handleExecute = async () => {
    if (!metadata.id) {
      toast.error('Save the workflow first');
      return;
    }

    try {
      await api.post(`/workflows/${metadata.id}/execute/async`, {
        trigger_type: 'manual',
        trigger_data: {},
        inputs: {},
      });
      toast.success('Workflow execution started');
    } catch (error: any) {
      toast.error('Failed to start workflow');
    }
  };

  // Go back with confirmation
  const handleBack = () => {
    if (isDirty) {
      if (window.confirm('You have unsaved changes. Are you sure you want to leave?')) {
        navigate('/workflows');
      }
    } else {
      navigate('/workflows');
    }
  };

  const openSynthesisModal = () => {
    setSynthesisName(metadata.name || '');
    setSynthesisDescription(metadata.description || '');
    setShowSynthesis(true);
  };

  const handleSynthesize = async () => {
    const description = synthesisDescription.trim();
    if (!description) {
      toast.error('Provide a description to generate a workflow');
      return;
    }

    if (isDirty && !window.confirm('Replace the current workflow with a new draft?')) {
      return;
    }

    setIsSynthesizing(true);
    try {
      const response = await api.synthesizeWorkflow({
        description,
        name: synthesisName.trim() || undefined,
        is_active: metadata.isActive,
        trigger_config: metadata.triggerConfig,
      });
      applyWorkflowDraft(response.workflow as any);
      setShowSynthesis(false);
      toast.success('Workflow draft generated');
      if (response.warnings?.length) {
        toast(`${response.warnings.length} warning${response.warnings.length === 1 ? '' : 's'} generated`, {
          icon: '⚠️',
        });
      }
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to generate workflow');
    } finally {
      setIsSynthesizing(false);
    }
  };

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-100">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white border-b px-4 py-2 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={handleBack}
            className="p-2 text-gray-600 hover:bg-gray-100 rounded"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>

          <div>
            <input
              type="text"
              value={metadata.name}
              onChange={(e) => setMetadata({ name: e.target.value })}
              className="text-lg font-medium bg-transparent border-none focus:ring-0 p-0"
              placeholder="Workflow name"
            />
            {isDirty && (
              <span className="text-xs text-amber-600 ml-2">Unsaved changes</span>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={openSynthesisModal}
            className="flex items-center space-x-1 px-3 py-1.5 text-gray-600 hover:bg-gray-100 rounded"
          >
            <Sparkles className="w-4 h-4" />
            <span>Generate</span>
          </button>

          <button
            onClick={() => setShowSettings(true)}
            className="flex items-center space-x-1 px-3 py-1.5 text-gray-600 hover:bg-gray-100 rounded"
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>

          {/* Validation status button */}
          {validationIssues.length > 0 && (
            <button
              onClick={() => setShowValidation(!showValidation)}
              className={`flex items-center space-x-1 px-3 py-1.5 rounded ${
                validationIssues.some(i => i.severity === 'error')
                  ? 'bg-red-100 text-red-700 hover:bg-red-200'
                  : 'bg-amber-100 text-amber-700 hover:bg-amber-200'
              }`}
            >
              {validationIssues.some(i => i.severity === 'error') ? (
                <AlertCircle className="w-4 h-4" />
              ) : (
                <AlertTriangle className="w-4 h-4" />
              )}
              <span>{validationIssues.length} issue{validationIssues.length !== 1 ? 's' : ''}</span>
            </button>
          )}

          {metadata.id && metadata.isActive && (
            <button
              onClick={handleExecute}
              className="flex items-center space-x-1 px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700"
            >
              <Play className="w-4 h-4" />
              <span>Run</span>
            </button>
          )}
        </div>
      </header>

      {/* Editor */}
      <div className="flex-1 overflow-hidden">
        <WorkflowEditor onSave={handleSave} />
      </div>

      {/* Validation Panel */}
      {showValidation && validationIssues.length > 0 && (
        <div className="absolute bottom-4 right-4 w-96 bg-white rounded-lg shadow-xl border z-40 max-h-80 overflow-hidden flex flex-col">
          <div className="flex items-center justify-between px-4 py-2 border-b bg-gray-50">
            <div className="flex items-center space-x-2">
              {validationIssues.some(i => i.severity === 'error') ? (
                <AlertCircle className="w-4 h-4 text-red-500" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-amber-500" />
              )}
              <h3 className="font-medium text-sm">Validation Issues</h3>
            </div>
            <button
              onClick={() => setShowValidation(false)}
              className="p-1 text-gray-400 hover:text-gray-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {validationIssues.map((issue, index) => (
              <div
                key={index}
                className={`p-2 rounded text-sm flex items-start space-x-2 ${
                  issue.severity === 'error'
                    ? 'bg-red-50 text-red-700'
                    : issue.severity === 'warning'
                    ? 'bg-amber-50 text-amber-700'
                    : 'bg-blue-50 text-blue-700'
                }`}
              >
                {issue.severity === 'error' ? (
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                ) : issue.severity === 'warning' ? (
                  <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                )}
                <div>
                  <p>{issue.message}</p>
                  {issue.node_id && (
                    <p className="text-xs opacity-75 mt-0.5">
                      Node: <code className="bg-white/50 px-1 rounded">{issue.node_id}</code>
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
            <div className="flex items-center justify-between px-4 py-3 border-b">
              <h3 className="font-medium">Workflow Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="p-1 text-gray-500 hover:text-gray-700"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  value={metadata.name}
                  onChange={(e) => setMetadata({ name: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={metadata.description}
                  onChange={(e) => setMetadata({ description: e.target.value })}
                  rows={3}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                  placeholder="What does this workflow do?"
                />
              </div>

              {/* Active toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Active
                  </label>
                  <p className="text-xs text-gray-500">
                    Enable workflow for execution
                  </p>
                </div>
                <button
                  onClick={() => setMetadata({ isActive: !metadata.isActive })}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    metadata.isActive ? 'bg-primary-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      metadata.isActive ? 'left-7' : 'left-1'
                    }`}
                  />
                </button>
              </div>

              {/* Trigger type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Trigger
                </label>
                <select
                  value={metadata.triggerConfig.type}
                  onChange={(e) =>
                    setMetadata({
                      triggerConfig: { ...metadata.triggerConfig, type: e.target.value as any },
                    })
                  }
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                >
                  <option value="manual">Manual</option>
                  <option value="schedule">Scheduled</option>
                  <option value="event">Event</option>
                  <option value="webhook">Webhook</option>
                </select>
              </div>

              {/* Schedule input */}
              {metadata.triggerConfig.type === 'schedule' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Cron Schedule
                  </label>
                  <input
                    type="text"
                    value={metadata.triggerConfig.schedule || ''}
                    onChange={(e) =>
                      setMetadata({
                        triggerConfig: { ...metadata.triggerConfig, schedule: e.target.value },
                      })
                    }
                    placeholder="0 9 * * *"
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 font-mono"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    e.g., "0 9 * * *" = every day at 9 AM
                  </p>
                </div>
              )}

              {/* Event input */}
              {metadata.triggerConfig.type === 'event' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Event Name
                  </label>
                  <select
                    value={metadata.triggerConfig.event || ''}
                    onChange={(e) =>
                      setMetadata({
                        triggerConfig: { ...metadata.triggerConfig, event: e.target.value },
                      })
                    }
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="">Select event...</option>
                    <option value="document.uploaded">Document Uploaded</option>
                    <option value="document.processed">Document Processed</option>
                    <option value="document.deleted">Document Deleted</option>
                  </select>
                </div>
              )}
            </div>

            <div className="flex justify-end space-x-2 px-4 py-3 border-t bg-gray-50 rounded-b-lg">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-200 rounded"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Workflow Synthesis Modal */}
      {showSynthesis && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
            <div className="flex items-center justify-between px-4 py-3 border-b">
              <h3 className="font-medium">Generate Workflow</h3>
              <button
                onClick={() => setShowSynthesis(false)}
                className="p-1 text-gray-500 hover:text-gray-700"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name (optional)
                </label>
                <input
                  type="text"
                  value={synthesisName}
                  onChange={(e) => setSynthesisName(e.target.value)}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                  placeholder="Let the AI choose a name"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={synthesisDescription}
                  onChange={(e) => setSynthesisDescription(e.target.value)}
                  rows={5}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                  placeholder="Describe the workflow you want to build..."
                />
                <p className="text-xs text-gray-500 mt-1">
                  This replaces the current workflow draft in the editor.
                </p>
              </div>
            </div>

            <div className="flex justify-end space-x-2 px-4 py-3 border-t bg-gray-50 rounded-b-lg">
              <button
                onClick={() => setShowSynthesis(false)}
                className="px-4 py-2 text-gray-700 hover:bg-gray-200 rounded"
                disabled={isSynthesizing}
              >
                Cancel
              </button>
              <button
                onClick={handleSynthesize}
                className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-60"
                disabled={isSynthesizing}
              >
                {isSynthesizing ? (
                  <span className="flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Generating</span>
                  </span>
                ) : (
                  'Generate'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WorkflowEditorPage;
