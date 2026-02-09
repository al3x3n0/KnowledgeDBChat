/**
 * MCP Configuration page for managing MCP tools and access control per API key.
 */

import React, { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import {
  Server,
  Key,
  Wrench,
  Database,
  Shield,
  Check,
  X,
  ChevronRight,
  RefreshCw,
  AlertTriangle,
  Info,
  ExternalLink,
  Copy,
} from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import {
  APIKey,
  MCPKeyConfigResponse,
  MCPToolInfo,
  MCPToolConfigResponse,
  MCPSourceAccessResponse,
  DocumentSource,
} from '../types';
import Button from '../components/common/Button';

const MCPConfigPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedKeyId, setSelectedKeyId] = useState<string | null>(null);

  // Fetch API keys
  const { data: keysData, isLoading: keysLoading } = useQuery(
    'apiKeys',
    () => apiClient.listAPIKeys(false),
    { refetchOnWindowFocus: false }
  );

  // Fetch available MCP tools
  const { data: mcpTools } = useQuery(
    'mcpTools',
    () => apiClient.listMCPTools(),
    { refetchOnWindowFocus: false }
  );

  // Fetch MCP config for selected key
  const {
    data: mcpConfig,
    isLoading: configLoading,
    refetch: refetchConfig,
  } = useQuery(
    ['mcpConfig', selectedKeyId],
    () => (selectedKeyId ? apiClient.getMCPKeyConfig(selectedKeyId) : null),
    {
      enabled: !!selectedKeyId,
      refetchOnWindowFocus: false,
    }
  );

  // Fetch document sources for source access configuration
  const { data: sources } = useQuery(
    'documentSources',
    async () => {
      const response = await apiClient.get<{ sources: DocumentSource[] }>('/admin/sources');
      return response.data.sources;
    },
    { refetchOnWindowFocus: false }
  );

  const apiKeys = keysData?.api_keys || [];
  const activeKeys = apiKeys.filter((k) => k.is_active && !k.revoked_at);

  // Auto-select first key if none selected
  useEffect(() => {
    if (!selectedKeyId && activeKeys.length > 0) {
      setSelectedKeyId(activeKeys[0].id);
    }
  }, [selectedKeyId, activeKeys]);

  const handleCopyEndpoint = () => {
    const endpoint = `${window.location.origin}/api/v1/mcp`;
    navigator.clipboard.writeText(endpoint);
    toast.success('MCP endpoint copied to clipboard');
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Server className="w-7 h-7" />
          MCP Configuration
        </h1>
        <p className="text-gray-600 mt-1">
          Configure Model Context Protocol (MCP) access for external AI agents
        </p>
      </div>

      {/* MCP Info Banner */}
      <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex gap-3">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-800">
            <p className="font-medium">About MCP Integration</p>
            <p className="mt-1">
              MCP (Model Context Protocol) allows external AI agents like Claude Desktop or custom
              integrations to interact with your knowledge base. Configure which tools and sources
              each API key can access.
            </p>
            <div className="mt-3 flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-gray-600">MCP Endpoint:</span>
                <code className="bg-blue-100 px-2 py-0.5 rounded font-mono text-xs">
                  {window.location.origin}/api/v1/mcp
                </code>
                <button
                  onClick={handleCopyEndpoint}
                  className="p-1 hover:bg-blue-100 rounded"
                  title="Copy endpoint"
                >
                  <Copy className="w-4 h-4 text-blue-600" />
                </button>
              </div>
              <a
                href="https://modelcontextprotocol.io/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-blue-600 hover:underline"
              >
                MCP Docs
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </div>

      {keysLoading ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <RefreshCw className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading API keys...</p>
        </div>
      ) : activeKeys.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <Key className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No API Keys</h3>
          <p className="text-gray-600 mb-4">
            Create an API key first to configure MCP access.
          </p>
          <Button onClick={() => (window.location.href = '/api-keys')}>
            Go to API Keys
          </Button>
        </div>
      ) : (
        <div className="flex gap-6">
          {/* API Key Selector */}
          <div className="w-64 flex-shrink-0">
            <div className="bg-white rounded-lg shadow">
              <div className="px-4 py-3 border-b">
                <h3 className="font-medium text-gray-900">API Keys</h3>
              </div>
              <div className="divide-y">
                {activeKeys.map((key) => (
                  <button
                    key={key.id}
                    onClick={() => setSelectedKeyId(key.id)}
                    className={`w-full px-4 py-3 text-left flex items-center justify-between hover:bg-gray-50 transition-colors ${
                      selectedKeyId === key.id ? 'bg-primary-50' : ''
                    }`}
                  >
                    <div>
                      <div className="font-medium text-gray-900 text-sm">{key.name}</div>
                      <div className="text-xs text-gray-500 font-mono">{key.key_prefix}...</div>
                    </div>
                    <ChevronRight
                      className={`w-4 h-4 ${
                        selectedKeyId === key.id ? 'text-primary-600' : 'text-gray-400'
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Configuration Panel */}
          <div className="flex-1">
            {selectedKeyId && (
              <MCPKeyConfigPanel
                keyId={selectedKeyId}
                config={mcpConfig}
                tools={mcpTools || []}
                sources={sources || []}
                isLoading={configLoading}
                onRefresh={() => {
                  refetchConfig();
                  queryClient.invalidateQueries(['mcpConfig', selectedKeyId]);
                }}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// MCP Key Configuration Panel
const MCPKeyConfigPanel: React.FC<{
  keyId: string;
  config: MCPKeyConfigResponse | null | undefined;
  tools: MCPToolInfo[];
  sources: DocumentSource[];
  isLoading: boolean;
  onRefresh: () => void;
}> = ({ keyId, config, tools, sources, isLoading, onRefresh }) => {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'general' | 'tools' | 'sources'>('general');

  // Update MCP config mutation
  const updateConfigMutation = useMutation(
    (data: { mcp_enabled?: boolean; source_access_mode?: string }) =>
      apiClient.updateMCPKeyConfig(keyId, data),
    {
      onSuccess: () => {
        toast.success('MCP configuration updated');
        queryClient.invalidateQueries(['mcpConfig', keyId]);
      },
      onError: () => {
        toast.error('Failed to update configuration');
      },
    }
  );

  // Update tool config mutation
  const updateToolMutation = useMutation(
    (data: { toolName: string; is_enabled: boolean; config?: Record<string, any> }) =>
      apiClient.updateMCPToolConfig(keyId, data.toolName, {
        tool_name: data.toolName,
        is_enabled: data.is_enabled,
        config: data.config,
      }),
    {
      onSuccess: () => {
        toast.success('Tool configuration updated');
        queryClient.invalidateQueries(['mcpConfig', keyId]);
      },
      onError: () => {
        toast.error('Failed to update tool configuration');
      },
    }
  );

  // Update source access mutation
  const updateSourceMutation = useMutation(
    (data: {
      sourceId: string;
      can_read: boolean;
      can_search: boolean;
      can_chat: boolean;
    }) =>
      apiClient.updateMCPSourceAccess(keyId, data.sourceId, {
        source_id: data.sourceId,
        can_read: data.can_read,
        can_search: data.can_search,
        can_chat: data.can_chat,
      }),
    {
      onSuccess: () => {
        toast.success('Source access updated');
        queryClient.invalidateQueries(['mcpConfig', keyId]);
      },
      onError: () => {
        toast.error('Failed to update source access');
      },
    }
  );

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <RefreshCw className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
        <p className="text-gray-600">Loading configuration...</p>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <AlertTriangle className="w-8 h-8 text-yellow-500 mx-auto mb-4" />
        <p className="text-gray-600">Failed to load configuration</p>
        <Button variant="ghost" onClick={onRefresh} className="mt-4">
          <RefreshCw className="w-4 h-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  const tabs = [
    { id: 'general', name: 'General', icon: Shield },
    { id: 'tools', name: 'Tools', icon: Wrench },
    { id: 'sources', name: 'Sources', icon: Database },
  ];

  // Group tools by category
  const toolsByCategory = config.tool_configs.reduce(
    (acc, tool) => {
      const category = tool.category || 'other';
      if (!acc[category]) acc[category] = [];
      acc[category].push(tool);
      return acc;
    },
    {} as Record<string, MCPToolConfigResponse[]>
  );

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="px-6 py-4 border-b flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">{config.api_key_name}</h2>
          <p className="text-sm text-gray-500">Configure MCP access for this API key</p>
        </div>
        <div className="flex items-center gap-2">
          {config.mcp_enabled ? (
            <span className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded flex items-center gap-1">
              <Check className="w-3 h-3" />
              MCP Enabled
            </span>
          ) : (
            <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded flex items-center gap-1">
              <X className="w-3 h-3" />
              MCP Disabled
            </span>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b px-6">
        <nav className="flex gap-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`flex items-center gap-2 py-3 border-b-2 -mb-px transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-600 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.name}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'general' && (
          <div className="space-y-6">
            {/* MCP Enabled Toggle */}
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <h3 className="font-medium text-gray-900">Enable MCP Access</h3>
                <p className="text-sm text-gray-500">
                  Allow external AI agents to access your knowledge base using this API key
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.mcp_enabled}
                  onChange={(e) =>
                    updateConfigMutation.mutate({ mcp_enabled: e.target.checked })
                  }
                  className="sr-only peer"
                  disabled={updateConfigMutation.isLoading}
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>

            {/* Source Access Mode */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">Source Access Mode</h3>
              <p className="text-sm text-gray-500 mb-4">
                Control which document sources can be accessed via MCP
              </p>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-3 bg-white rounded-lg border cursor-pointer hover:bg-gray-50">
                  <input
                    type="radio"
                    name="source_access_mode"
                    value="all"
                    checked={config.source_access_mode === 'all'}
                    onChange={() =>
                      updateConfigMutation.mutate({ source_access_mode: 'all' })
                    }
                    disabled={updateConfigMutation.isLoading}
                  />
                  <div>
                    <div className="font-medium text-gray-900">All Sources</div>
                    <div className="text-sm text-gray-500">
                      Allow access to all document sources
                    </div>
                  </div>
                </label>
                <label className="flex items-center gap-3 p-3 bg-white rounded-lg border cursor-pointer hover:bg-gray-50">
                  <input
                    type="radio"
                    name="source_access_mode"
                    value="restricted"
                    checked={config.source_access_mode === 'restricted'}
                    onChange={() =>
                      updateConfigMutation.mutate({ source_access_mode: 'restricted' })
                    }
                    disabled={updateConfigMutation.isLoading}
                  />
                  <div>
                    <div className="font-medium text-gray-900">Restricted</div>
                    <div className="text-sm text-gray-500">
                      Only allow access to explicitly configured sources (configure in Sources tab)
                    </div>
                  </div>
                </label>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {config.tool_configs.filter((t) => t.is_enabled).length}
                </div>
                <div className="text-sm text-gray-500">Tools Enabled</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {config.source_access.length}
                </div>
                <div className="text-sm text-gray-500">Source Restrictions</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {config.source_access_mode === 'all' ? 'All' : 'Restricted'}
                </div>
                <div className="text-sm text-gray-500">Access Mode</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'tools' && (
          <div className="space-y-6">
            <p className="text-sm text-gray-500">
              Enable or disable individual MCP tools for this API key. Disabled tools will not be
              available to external agents using this key.
            </p>

            {Object.entries(toolsByCategory).map(([category, categoryTools]) => (
              <div key={category}>
                <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wider mb-3">
                  {category} Tools
                </h3>
                <div className="space-y-2">
                  {categoryTools.map((tool) => (
                    <div
                      key={tool.tool_name}
                      className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium text-gray-900">{tool.display_name}</h4>
                          <span className="px-2 py-0.5 text-xs bg-gray-200 text-gray-600 rounded">
                            {tool.tool_name}
                          </span>
                        </div>
                        <p className="text-sm text-gray-500 mt-1">{tool.description}</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer ml-4">
                        <input
                          type="checkbox"
                          checked={tool.is_enabled}
                          onChange={(e) =>
                            updateToolMutation.mutate({
                              toolName: tool.tool_name,
                              is_enabled: e.target.checked,
                            })
                          }
                          className="sr-only peer"
                          disabled={updateToolMutation.isLoading}
                        />
                        <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
                      </label>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'sources' && (
          <div className="space-y-6">
            <div className="flex items-start gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium">Source Access Configuration</p>
                <p className="mt-1">
                  {config.source_access_mode === 'all'
                    ? 'Source access mode is set to "All Sources". The restrictions below will not apply until you switch to "Restricted" mode in the General tab.'
                    : 'Only the sources listed below will be accessible via MCP. Add sources to grant access.'}
                </p>
              </div>
            </div>

            {/* Configured Sources */}
            {config.source_access.length > 0 && (
              <div>
                <h3 className="font-medium text-gray-900 mb-3">Configured Source Access</h3>
                <div className="space-y-2">
                  {config.source_access.map((access) => (
                    <div
                      key={access.source_id}
                      className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                    >
                      <div>
                        <div className="font-medium text-gray-900">{access.source_name}</div>
                        <div className="text-sm text-gray-500">{access.source_type}</div>
                      </div>
                      <div className="flex items-center gap-4">
                        <label className="flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={access.can_read}
                            onChange={(e) =>
                              updateSourceMutation.mutate({
                                sourceId: access.source_id,
                                can_read: e.target.checked,
                                can_search: access.can_search,
                                can_chat: access.can_chat,
                              })
                            }
                            disabled={updateSourceMutation.isLoading}
                          />
                          Read
                        </label>
                        <label className="flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={access.can_search}
                            onChange={(e) =>
                              updateSourceMutation.mutate({
                                sourceId: access.source_id,
                                can_read: access.can_read,
                                can_search: e.target.checked,
                                can_chat: access.can_chat,
                              })
                            }
                            disabled={updateSourceMutation.isLoading}
                          />
                          Search
                        </label>
                        <label className="flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={access.can_chat}
                            onChange={(e) =>
                              updateSourceMutation.mutate({
                                sourceId: access.source_id,
                                can_read: access.can_read,
                                can_search: access.can_search,
                                can_chat: e.target.checked,
                              })
                            }
                            disabled={updateSourceMutation.isLoading}
                          />
                          Chat
                        </label>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Add Source Access */}
            <div>
              <h3 className="font-medium text-gray-900 mb-3">Add Source Access</h3>
              {sources.length === 0 ? (
                <p className="text-sm text-gray-500">No document sources available</p>
              ) : (
                <div className="space-y-2">
                  {sources
                    .filter(
                      (source) =>
                        !config.source_access.some((a) => a.source_id === source.id)
                    )
                    .map((source) => (
                      <div
                        key={source.id}
                        className="flex items-center justify-between p-4 border border-dashed rounded-lg hover:bg-gray-50"
                      >
                        <div>
                          <div className="font-medium text-gray-900">{source.name}</div>
                          <div className="text-sm text-gray-500">{source.source_type}</div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() =>
                            updateSourceMutation.mutate({
                              sourceId: source.id,
                              can_read: true,
                              can_search: true,
                              can_chat: true,
                            })
                          }
                          disabled={updateSourceMutation.isLoading}
                        >
                          Add Access
                        </Button>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MCPConfigPage;
