/**
 * Custom Tools Management Page.
 *
 * Allows users to create, edit, test, and manage custom tools.
 */

import React, { useState, useEffect } from 'react';
import {
  Plus,
  Trash2,
  Edit,
  Copy,
  Play,
  Search,
  MoreVertical,
  Loader2,
  Wrench,
  Globe,
  Code,
  MessageSquare,
  Shuffle,
  CheckCircle,
  XCircle,
  ChevronDown,
} from 'lucide-react';
import toast from 'react-hot-toast';
import api from '../services/api';

// Types
interface UserTool {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  tool_type: 'webhook' | 'transform' | 'python' | 'llm_prompt';
  parameters_schema: Record<string, any>;
  config: Record<string, any>;
  is_enabled: boolean;
  version: number;
  created_at: string;
  updated_at: string;
}

// Tool type info
const TOOL_TYPES = {
  webhook: {
    label: 'Webhook',
    description: 'Make HTTP requests to external APIs',
    icon: Globe,
    color: 'blue',
  },
  transform: {
    label: 'Transform',
    description: 'Transform data using templates',
    icon: Shuffle,
    color: 'purple',
  },
  python: {
    label: 'Python',
    description: 'Run sandboxed Python code',
    icon: Code,
    color: 'green',
  },
  llm_prompt: {
    label: 'LLM Prompt',
    description: 'Call LLM with templated prompts',
    icon: MessageSquare,
    color: 'amber',
  },
};

const ToolsPage: React.FC = () => {
  const [tools, setTools] = useState<UserTool[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string | null>(null);
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);

  // Modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showTestModal, setShowTestModal] = useState(false);
  const [selectedTool, setSelectedTool] = useState<UserTool | null>(null);

  // Load tools
  useEffect(() => {
    loadTools();
  }, []);

  const loadTools = async () => {
    setIsLoading(true);
    try {
      const response = await api.get('/user-tools');
      setTools(response.data.tools || []);
    } catch (error: any) {
      toast.error('Failed to load tools');
    } finally {
      setIsLoading(false);
    }
  };

  const deleteTool = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this tool?')) return;

    try {
      await api.delete(`/user-tools/${id}`);
      toast.success('Tool deleted');
      loadTools();
    } catch (error: any) {
      toast.error('Failed to delete tool');
    }
    setActiveMenuId(null);
  };

  const duplicateTool = async (id: string, name: string) => {
    const newName = prompt('Enter name for duplicated tool:', `${name} (copy)`);
    if (!newName) return;

    try {
      await api.post(`/user-tools/${id}/duplicate?new_name=${encodeURIComponent(newName)}`);
      toast.success('Tool duplicated');
      loadTools();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to duplicate tool');
    }
    setActiveMenuId(null);
  };

  const toggleEnabled = async (tool: UserTool) => {
    try {
      await api.put(`/user-tools/${tool.id}`, { is_enabled: !tool.is_enabled });
      loadTools();
    } catch (error: any) {
      toast.error('Failed to update tool');
    }
  };

  const filteredTools = tools.filter((tool) => {
    const matchesSearch =
      tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tool.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = !filterType || tool.tool_type === filterType;
    return matchesSearch && matchesType;
  });

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Custom Tools</h1>
          <p className="text-gray-600">
            Create reusable tools for workflows and AI assistant
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <Plus className="w-5 h-5" />
          <span>New Tool</span>
        </button>
      </div>

      {/* Search and Filter */}
      <div className="flex items-center space-x-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search tools..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
          />
        </div>

        {/* Type filter */}
        <div className="relative">
          <select
            value={filterType || ''}
            onChange={(e) => setFilterType(e.target.value || null)}
            className="appearance-none pl-4 pr-10 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 bg-white"
          >
            <option value="">All Types</option>
            {Object.entries(TOOL_TYPES).map(([type, info]) => (
              <option key={type} value={type}>
                {info.label}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* Loading state */}
      {isLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      ) : filteredTools.length === 0 ? (
        <div className="text-center py-16">
          <Wrench className="w-16 h-16 mx-auto text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-700">No tools yet</h3>
          <p className="text-gray-500 mb-4">
            Create your first custom tool to extend AI capabilities
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center space-x-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            <Plus className="w-5 h-5" />
            <span>Create Tool</span>
          </button>
        </div>
      ) : (
        <div className="grid gap-4">
          {filteredTools.map((tool) => {
            const typeInfo = TOOL_TYPES[tool.tool_type];
            const Icon = typeInfo.icon;

            return (
              <div
                key={tool.id}
                className={`bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow ${
                  !tool.is_enabled ? 'opacity-60' : ''
                }`}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between">
                    {/* Tool info */}
                    <div className="flex items-start space-x-3">
                      <div
                        className={`p-2 rounded-lg bg-${typeInfo.color}-100`}
                      >
                        <Icon
                          className={`w-5 h-5 text-${typeInfo.color}-600`}
                        />
                      </div>

                      <div>
                        <div className="flex items-center space-x-2">
                          <h3 className="font-medium text-gray-900">
                            {tool.name}
                          </h3>
                          <span
                            className={`px-2 py-0.5 text-xs rounded-full ${
                              tool.is_enabled
                                ? 'bg-green-100 text-green-700'
                                : 'bg-gray-100 text-gray-600'
                            }`}
                          >
                            {tool.is_enabled ? 'Enabled' : 'Disabled'}
                          </span>
                        </div>
                        {tool.description && (
                          <p className="text-sm text-gray-500 mt-1">
                            {tool.description}
                          </p>
                        )}
                        <div className="flex items-center space-x-3 mt-2 text-sm text-gray-500">
                          <span className="flex items-center space-x-1">
                            <Icon className="w-4 h-4" />
                            <span>{typeInfo.label}</span>
                          </span>
                          <span>v{tool.version}</span>
                          <span>Updated {formatDate(tool.updated_at)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => {
                          setSelectedTool(tool);
                          setShowTestModal(true);
                        }}
                        className="p-2 text-green-600 hover:bg-green-50 rounded"
                        title="Test tool"
                      >
                        <Play className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => {
                          setSelectedTool(tool);
                          setShowEditModal(true);
                        }}
                        className="p-2 text-blue-600 hover:bg-blue-50 rounded"
                        title="Edit tool"
                      >
                        <Edit className="w-5 h-5" />
                      </button>

                      {/* More menu */}
                      <div className="relative">
                        <button
                          onClick={() =>
                            setActiveMenuId(
                              activeMenuId === tool.id ? null : tool.id
                            )
                          }
                          className="p-2 text-gray-500 hover:bg-gray-100 rounded"
                        >
                          <MoreVertical className="w-5 h-5" />
                        </button>

                        {activeMenuId === tool.id && (
                          <div className="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border z-10">
                            <button
                              onClick={() => toggleEnabled(tool)}
                              className="w-full flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                            >
                              {tool.is_enabled ? (
                                <>
                                  <XCircle className="w-4 h-4" />
                                  <span>Disable</span>
                                </>
                              ) : (
                                <>
                                  <CheckCircle className="w-4 h-4" />
                                  <span>Enable</span>
                                </>
                              )}
                            </button>
                            <button
                              onClick={() => duplicateTool(tool.id, tool.name)}
                              className="w-full flex items-center space-x-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                            >
                              <Copy className="w-4 h-4" />
                              <span>Duplicate</span>
                            </button>
                            <hr className="my-1" />
                            <button
                              onClick={() => deleteTool(tool.id)}
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
            );
          })}
        </div>
      )}

      {/* Click outside to close menu */}
      {activeMenuId && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setActiveMenuId(null)}
        />
      )}

      {/* Create Tool Modal */}
      {showCreateModal && (
        <ToolEditorModal
          onClose={() => setShowCreateModal(false)}
          onSave={() => {
            setShowCreateModal(false);
            loadTools();
          }}
        />
      )}

      {/* Edit Tool Modal */}
      {showEditModal && selectedTool && (
        <ToolEditorModal
          tool={selectedTool}
          onClose={() => {
            setShowEditModal(false);
            setSelectedTool(null);
          }}
          onSave={() => {
            setShowEditModal(false);
            setSelectedTool(null);
            loadTools();
          }}
        />
      )}

      {/* Test Tool Modal */}
      {showTestModal && selectedTool && (
        <ToolTesterModal
          tool={selectedTool}
          onClose={() => {
            setShowTestModal(false);
            setSelectedTool(null);
          }}
        />
      )}
    </div>
  );
};

// =============================================================================
// Tool Editor Modal
// =============================================================================

interface ToolEditorModalProps {
  tool?: UserTool;
  onClose: () => void;
  onSave: () => void;
}

const ToolEditorModal: React.FC<ToolEditorModalProps> = ({
  tool,
  onClose,
  onSave,
}) => {
  const isEditing = !!tool;
  const [isSaving, setIsSaving] = useState(false);

  // Form state
  const [name, setName] = useState(tool?.name || '');
  const [description, setDescription] = useState(tool?.description || '');
  const [toolType, setToolType] = useState<UserTool['tool_type']>(
    tool?.tool_type || 'webhook'
  );
  const [config, setConfig] = useState<Record<string, any>>(
    tool?.config || getDefaultConfig('webhook')
  );
  const [parametersSchema, setParametersSchema] = useState<string>(
    JSON.stringify(tool?.parameters_schema || {}, null, 2)
  );

  function getDefaultConfig(type: string): Record<string, any> {
    switch (type) {
      case 'webhook':
        return {
          method: 'POST',
          url: '',
          headers: {},
          body_template: '',
          response_path: '',
          timeout_seconds: 30,
        };
      case 'transform':
        return {
          transform_type: 'jinja2',
          template: '',
        };
      case 'python':
        return {
          code: '# Access inputs via `input` dict\n# Set results in `output` dict\noutput = {"result": input.get("value", "")}',
          timeout_seconds: 10,
          allowed_imports: ['json', 're', 'datetime', 'math'],
        };
      case 'llm_prompt':
        return {
          system_prompt: '',
          user_prompt: '',
          output_format: 'text',
          model_override: null,
          temperature: null,
          max_tokens: null,
        };
      default:
        return {};
    }
  }

  const handleTypeChange = (newType: UserTool['tool_type']) => {
    setToolType(newType);
    if (!isEditing) {
      setConfig(getDefaultConfig(newType));
    }
  };

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error('Name is required');
      return;
    }

    let parsedSchema = {};
    try {
      parsedSchema = JSON.parse(parametersSchema || '{}');
    } catch (e) {
      toast.error('Invalid parameters schema JSON');
      return;
    }

    setIsSaving(true);

    try {
      const payload = {
        name,
        description: description || null,
        tool_type: toolType,
        config,
        parameters_schema: parsedSchema,
        is_enabled: true,
      };

      if (isEditing) {
        await api.put(`/user-tools/${tool.id}`, payload);
        toast.success('Tool updated');
      } else {
        await api.post('/user-tools', payload);
        toast.success('Tool created');
      }

      onSave();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save tool');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <h2 className="text-lg font-semibold">
            {isEditing ? 'Edit Tool' : 'Create Tool'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-auto p-6 space-y-6">
          {/* Basic info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
                placeholder="My Tool"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Type
              </label>
              <select
                value={toolType}
                onChange={(e) =>
                  handleTypeChange(e.target.value as UserTool['tool_type'])
                }
                disabled={isEditing}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100"
              >
                {Object.entries(TOOL_TYPES).map(([type, info]) => (
                  <option key={type} value={type}>
                    {info.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500"
              placeholder="What does this tool do?"
            />
          </div>

          {/* Type-specific config */}
          <div className="border-t pt-4">
            <h3 className="text-sm font-medium text-gray-700 mb-3">
              Configuration
            </h3>

            {toolType === 'webhook' && (
              <WebhookConfigEditor config={config} onChange={setConfig} />
            )}
            {toolType === 'transform' && (
              <TransformConfigEditor config={config} onChange={setConfig} />
            )}
            {toolType === 'python' && (
              <PythonConfigEditor config={config} onChange={setConfig} />
            )}
            {toolType === 'llm_prompt' && (
              <LLMPromptConfigEditor config={config} onChange={setConfig} />
            )}
          </div>

          {/* Parameters Schema */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Parameters Schema (JSON Schema)
            </label>
            <textarea
              value={parametersSchema}
              onChange={(e) => setParametersSchema(e.target.value)}
              rows={5}
              className="w-full px-3 py-2 border rounded-lg font-mono text-sm focus:ring-2 focus:ring-primary-500"
              placeholder='{"type": "object", "properties": {...}}'
            />
            <p className="text-xs text-gray-500 mt-1">
              Define input parameters for the tool (optional)
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end space-x-2 px-6 py-4 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 hover:bg-gray-200 rounded"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-2 bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : isEditing ? (
              'Update'
            ) : (
              'Create'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// Config Editors
// =============================================================================

interface ConfigEditorProps {
  config: Record<string, any>;
  onChange: (config: Record<string, any>) => void;
}

const WebhookConfigEditor: React.FC<ConfigEditorProps> = ({
  config,
  onChange,
}) => {
  const updateField = (field: string, value: any) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">Method</label>
          <select
            value={config.method || 'POST'}
            onChange={(e) => updateField('method', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg"
          >
            <option value="GET">GET</option>
            <option value="POST">POST</option>
            <option value="PUT">PUT</option>
            <option value="PATCH">PATCH</option>
            <option value="DELETE">DELETE</option>
          </select>
        </div>
        <div className="col-span-3">
          <label className="block text-sm text-gray-600 mb-1">URL</label>
          <input
            type="text"
            value={config.url || ''}
            onChange={(e) => updateField('url', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg font-mono text-sm"
            placeholder="https://api.example.com/{{input.endpoint}}"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm text-gray-600 mb-1">
          Headers (JSON)
        </label>
        <textarea
          value={JSON.stringify(config.headers || {}, null, 2)}
          onChange={(e) => {
            try {
              updateField('headers', JSON.parse(e.target.value));
            } catch {}
          }}
          rows={3}
          className="w-full px-3 py-2 border rounded-lg font-mono text-sm"
          placeholder='{"Authorization": "Bearer {{input.token}}"}'
        />
      </div>

      <div>
        <label className="block text-sm text-gray-600 mb-1">
          Body Template (Jinja2)
        </label>
        <textarea
          value={config.body_template || ''}
          onChange={(e) => updateField('body_template', e.target.value)}
          rows={4}
          className="w-full px-3 py-2 border rounded-lg font-mono text-sm"
          placeholder="{{input | tojson}}"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Response JSONPath
          </label>
          <input
            type="text"
            value={config.response_path || ''}
            onChange={(e) => updateField('response_path', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg font-mono text-sm"
            placeholder="$.data.result"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Timeout (seconds)
          </label>
          <input
            type="number"
            value={config.timeout_seconds || 30}
            onChange={(e) =>
              updateField('timeout_seconds', parseInt(e.target.value))
            }
            min={1}
            max={300}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
      </div>
    </div>
  );
};

const TransformConfigEditor: React.FC<ConfigEditorProps> = ({
  config,
  onChange,
}) => {
  const updateField = (field: string, value: any) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm text-gray-600 mb-1">
          Transform Type
        </label>
        <select
          value={config.transform_type || 'jinja2'}
          onChange={(e) => updateField('transform_type', e.target.value)}
          className="w-full px-3 py-2 border rounded-lg"
        >
          <option value="jinja2">Jinja2 Template</option>
          <option value="jsonpath">JSONPath</option>
        </select>
      </div>

      <div>
        <label className="block text-sm text-gray-600 mb-1">Template</label>
        <textarea
          value={config.template || ''}
          onChange={(e) => updateField('template', e.target.value)}
          rows={6}
          className="w-full px-3 py-2 border rounded-lg font-mono text-sm"
          placeholder={
            config.transform_type === 'jsonpath'
              ? '$.data[*].name'
              : '{{ input.items | selectattr("active") | list }}'
          }
        />
      </div>
    </div>
  );
};

const PythonConfigEditor: React.FC<ConfigEditorProps> = ({
  config,
  onChange,
}) => {
  const updateField = (field: string, value: any) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm text-gray-600 mb-1">Python Code</label>
        <textarea
          value={config.code || ''}
          onChange={(e) => updateField('code', e.target.value)}
          rows={10}
          className="w-full px-3 py-2 border rounded-lg font-mono text-sm"
          placeholder="# Access inputs via `input` dict&#10;output = {'result': input.get('value')}"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Timeout (seconds)
          </label>
          <input
            type="number"
            value={config.timeout_seconds || 10}
            onChange={(e) =>
              updateField('timeout_seconds', parseInt(e.target.value))
            }
            min={1}
            max={60}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Allowed Imports (comma-separated)
          </label>
          <input
            type="text"
            value={(config.allowed_imports || []).join(', ')}
            onChange={(e) =>
              updateField(
                'allowed_imports',
                e.target.value.split(',').map((s) => s.trim())
              )
            }
            className="w-full px-3 py-2 border rounded-lg"
            placeholder="json, re, datetime, math"
          />
        </div>
      </div>
    </div>
  );
};

const LLMPromptConfigEditor: React.FC<ConfigEditorProps> = ({
  config,
  onChange,
}) => {
  const updateField = (field: string, value: any) => {
    onChange({ ...config, [field]: value });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm text-gray-600 mb-1">
          System Prompt
        </label>
        <textarea
          value={config.system_prompt || ''}
          onChange={(e) => updateField('system_prompt', e.target.value)}
          rows={3}
          className="w-full px-3 py-2 border rounded-lg"
          placeholder="You are a helpful assistant..."
        />
      </div>

      <div>
        <label className="block text-sm text-gray-600 mb-1">
          User Prompt Template
        </label>
        <textarea
          value={config.user_prompt || ''}
          onChange={(e) => updateField('user_prompt', e.target.value)}
          rows={5}
          className="w-full px-3 py-2 border rounded-lg"
          placeholder="Analyze this data: {{input.data}}"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Output Format
          </label>
          <select
            value={config.output_format || 'text'}
            onChange={(e) => updateField('output_format', e.target.value)}
            className="w-full px-3 py-2 border rounded-lg"
          >
            <option value="text">Text</option>
            <option value="json">JSON</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Temperature
          </label>
          <input
            type="number"
            value={config.temperature ?? ''}
            onChange={(e) =>
              updateField(
                'temperature',
                e.target.value ? parseFloat(e.target.value) : null
              )
            }
            min={0}
            max={2}
            step={0.1}
            className="w-full px-3 py-2 border rounded-lg"
            placeholder="Default"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-600 mb-1">Max Tokens</label>
          <input
            type="number"
            value={config.max_tokens ?? ''}
            onChange={(e) =>
              updateField(
                'max_tokens',
                e.target.value ? parseInt(e.target.value) : null
              )
            }
            min={1}
            max={32000}
            className="w-full px-3 py-2 border rounded-lg"
            placeholder="Default"
          />
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// Tool Tester Modal
// =============================================================================

interface ToolTesterModalProps {
  tool: UserTool;
  onClose: () => void;
}

const ToolTesterModal: React.FC<ToolTesterModalProps> = ({ tool, onClose }) => {
  const [inputs, setInputs] = useState<string>('{}');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<{
    success: boolean;
    output: any;
    error: string | null;
    execution_time_ms: number;
  } | null>(null);

  const runTest = async () => {
    let parsedInputs: Record<string, any>;
    try {
      parsedInputs = JSON.parse(inputs);
    } catch (e) {
      toast.error('Invalid JSON inputs');
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const response = await api.post(`/user-tools/${tool.id}/test`, {
        inputs: parsedInputs,
      });
      setResult(response.data);
    } catch (error: any) {
      setResult({
        success: false,
        output: null,
        error: error.response?.data?.detail || 'Test failed',
        execution_time_ms: 0,
      });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div>
            <h2 className="text-lg font-semibold">Test Tool: {tool.name}</h2>
            <p className="text-sm text-gray-500">
              {TOOL_TYPES[tool.tool_type].label}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-auto p-6 space-y-4">
          {/* Inputs */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Test Inputs (JSON)
            </label>
            <textarea
              value={inputs}
              onChange={(e) => setInputs(e.target.value)}
              rows={6}
              className="w-full px-3 py-2 border rounded-lg font-mono text-sm focus:ring-2 focus:ring-primary-500"
              placeholder='{"key": "value"}'
            />
          </div>

          {/* Run button */}
          <button
            onClick={runTest}
            disabled={isRunning}
            className="flex items-center justify-center space-x-2 w-full py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
          >
            {isRunning ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Run Test</span>
              </>
            )}
          </button>

          {/* Result */}
          {result && (
            <div
              className={`p-4 rounded-lg border ${
                result.success
                  ? 'bg-green-50 border-green-200'
                  : 'bg-red-50 border-red-200'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className={`flex items-center space-x-1 font-medium ${
                    result.success ? 'text-green-700' : 'text-red-700'
                  }`}
                >
                  {result.success ? (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      <span>Success</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="w-4 h-4" />
                      <span>Failed</span>
                    </>
                  )}
                </span>
                <span className="text-sm text-gray-500">
                  {result.execution_time_ms}ms
                </span>
              </div>

              {result.error && (
                <div className="text-sm text-red-600 mb-2">{result.error}</div>
              )}

              {result.output && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Output
                  </label>
                  <pre className="bg-white p-3 rounded border text-sm overflow-auto max-h-48">
                    {JSON.stringify(result.output, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end px-6 py-4 border-t bg-gray-50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 hover:bg-gray-200 rounded"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ToolsPage;
