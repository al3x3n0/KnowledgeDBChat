/**
 * MCP Configuration page for managing MCP tools and access control per API key.
 */

import React, { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { Link, useSearchParams } from 'react-router-dom';
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
  Copy,
  Activity,
} from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import {
  MCPKeyConfigResponse,
  APIKeyListResponse,
  MCPToolInfo,
  MCPToolConfigResponse,
  DocumentSource,
} from '../types';
import Button from '../components/common/Button';
import MCPProfileBuilder from '../components/mcp/MCPProfileBuilder';
import MCPProfileKeyRotation from '../components/mcp/MCPProfileKeyRotation';
import MCPProfileUsage from '../components/mcp/MCPProfileUsage';
import MCPClientSetup, { copyMCPValue } from '../components/mcp/MCPClientSetup';

const MCPConfigPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [preferredKeyId, setPreferredKeyId] = useState<string | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const managingApiKeys = searchParams.get('view') === 'api-keys';
  const program = searchParams.get('program')?.trim() || undefined;
  const noteId = searchParams.get('note')?.trim() || undefined;
  const sourceId = searchParams.get('source')?.trim() || undefined;
  const linkedNote = useQuery(['mcpProfileNote', noteId], () => apiClient.listMCPProfileNotes('', noteId), {
    enabled: Boolean(noteId) && !managingApiKeys,
    retry: false,
  });

  // Fetch API keys
  const keys = useQuery(
    'apiKeys',
    () => apiClient.listAPIKeys(false),
    { refetchOnWindowFocus: false }
  );
  const activeKeys = (keys.data?.api_keys || []).filter((key) => key.is_active && !key.revoked_at);
  const visibleKeys = activeKeys.filter((key) => Boolean(key.is_mcp_profile) !== managingApiKeys);
  const selectedKey = visibleKeys.find((key) => key.id === preferredKeyId) || visibleKeys[0];
  const selectedKeyId = selectedKey?.id;

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
  const { data: sources, isLoading: sourcesLoading } = useQuery(
    'documentSources',
    () => apiClient.getDocumentSources(),
    { refetchOnWindowFocus: false }
  );
  const linkedSource = sourceId
    ? sources?.find((source) => source.id === sourceId && source.is_active) ?? null
    : undefined;

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Server className="w-7 h-7" />
          {managingApiKeys ? 'MCP Access for API Keys' : 'MCP Profiles'}
        </h1>
        <p className="text-gray-600 mt-1">
          {managingApiKeys
            ? 'Manage tool and source access for existing general-purpose API keys.'
            : 'Share selected research programs, notes, and sources with an AI client.'}
        </p>
        {managingApiKeys ? (
          <Link to="/mcp-config" className="mt-2 inline-block text-sm text-primary-700 underline">Back to MCP Profiles</Link>
        ) : activeKeys.some((key) => !key.is_mcp_profile) && (
          <Link to="/mcp-config?view=api-keys" className="mt-2 inline-block text-sm text-primary-700 underline">Manage MCP access for API keys</Link>
        )}
      </div>

      {!managingApiKeys && ((noteId && linkedNote.isLoading) || (sourceId && sourcesLoading) ? (
        <p className="mb-6 text-sm text-gray-600" role="status">Loading the linked profile context…</p>
      ) : (
        <>
          {noteId && !linkedNote.data?.[0] && (
            <p className="mb-4 text-sm text-amber-800" role="alert">The linked research note is unavailable. You can still create a profile.</p>
          )}
          {linkedSource === null && (
            <p className="mb-4 text-sm text-amber-800" role="alert">
              The linked document source is unavailable. Select an active source to share documents.
            </p>
          )}
          <MCPProfileBuilder
            key={JSON.stringify([program, noteId, sourceId])}
            initialSelection={{ program, note: linkedNote.data?.[0], source: linkedSource }}
            onClose={() => {
              if (!program && !noteId && !sourceId) return;
              const next = new URLSearchParams(searchParams);
              next.delete('program');
              next.delete('note');
              next.delete('source');
              setSearchParams(next, { replace: true });
            }}
            onCreated={(id) => {
              setPreferredKeyId(id);
              void queryClient.invalidateQueries('apiKeys');
            }}
          />
        </>
      ))}

      {keys.isLoading ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <RefreshCw className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">{managingApiKeys ? 'Loading API keys…' : 'Loading profiles…'}</p>
        </div>
      ) : keys.isError ? (
        <div role="alert" className="flex items-center gap-3 rounded-lg border bg-white p-4 text-sm">
          {managingApiKeys ? 'Could not load API keys.' : 'Could not load MCP profiles.'}
          <Button variant="secondary" onClick={() => keys.refetch()}>Retry</Button>
        </div>
      ) : visibleKeys.length === 0 ? (
        <p className="text-sm text-gray-600">{managingApiKeys ? 'No active general-purpose API keys.' : 'No MCP profiles yet.'}</p>
      ) : (
        <div className="flex flex-col gap-6 lg:flex-row">
          {/* API Key Selector */}
          <div className="w-full flex-shrink-0 lg:w-64">
            <div className="bg-white rounded-lg shadow">
              <div className="px-4 py-3 border-b">
                <h3 className="font-medium text-gray-900">{managingApiKeys ? 'API Keys' : 'Your profiles'}</h3>
              </div>
              <div className="divide-y">
                {visibleKeys.map((key) => (
                  <button
                    key={key.id}
                    onClick={() => setPreferredKeyId(key.id)}
                    className={`w-full px-4 py-3 text-left flex items-center justify-between hover:bg-gray-50 transition-colors ${
                      selectedKeyId === key.id ? 'bg-primary-50' : ''
                    }`}
                  >
                    <div>
                      <div className="font-medium text-gray-900 text-sm">{key.name}</div>
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
          <div className="min-w-0 flex-1">
            {selectedKeyId && (
              <MCPKeyConfigPanel
                key={selectedKeyId}
                keyId={selectedKeyId}
                keyPrefix={selectedKey?.key_prefix}
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
  keyPrefix?: string;
  config: MCPKeyConfigResponse | null | undefined;
  tools: MCPToolInfo[];
  sources: DocumentSource[];
  isLoading: boolean;
  onRefresh: () => void;
}> = ({ keyId, keyPrefix, config, tools, sources, isLoading, onRefresh }) => {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<'general' | 'connection' | 'tools' | 'sources' | 'usage'>('general');

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

  const isProfile = config.mcp_programs != null;
  const tabs = [
    { id: 'general', name: 'General', icon: Shield },
    { id: 'connection', name: 'Connection', icon: Key },
    { id: 'tools', name: 'Tools', icon: Wrench },
    { id: 'sources', name: 'Sources', icon: Database },
    { id: 'usage', name: 'Usage', icon: Activity },
  ].filter(({ id }) => isProfile || id !== 'usage');
  const configuredTools = config.tool_configs.filter(
    (tool) => config.allowed_tools == null || config.allowed_tools.includes(tool.tool_name)
  );
  const endpoint = apiClient.getMCPEndpoint();

  // Group tools by category
  const toolsByCategory = configuredTools.reduce(
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
          <p className="text-sm text-gray-500">{isProfile ? 'Profile access and connection details' : 'MCP access for this API key'}</p>
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
        <nav className="flex flex-wrap gap-x-4">
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
        {isProfile && activeTab === 'usage' && <MCPProfileUsage keyId={keyId} />}
        {activeTab === 'connection' && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end gap-2">
              <label className="min-w-0 flex-1 text-sm font-medium text-gray-700">
                MCP endpoint
                <input readOnly value={endpoint} className="mt-1 block w-full rounded-md border bg-gray-50 px-3 py-2 font-mono text-sm" />
              </label>
              <Button variant="secondary" onClick={() => void copyMCPValue(endpoint, 'MCP endpoint')}>
                <Copy className="mr-2 h-4 w-4" /> Copy endpoint
              </Button>
            </div>
            {keyPrefix && <p className="text-sm text-gray-600">Key prefix <code>{keyPrefix}…</code></p>}
            <MCPClientSetup key={keyPrefix} profileId={keyId} endpoint={endpoint} keyPrefix={keyPrefix} />
            {isProfile && keyPrefix && (
              <MCPProfileKeyRotation
                profileId={keyId}
                keyPrefix={keyPrefix}
                onRotated={(prefix) => {
                  queryClient.setQueryData<APIKeyListResponse | undefined>('apiKeys', (current) =>
                    current && {
                      ...current,
                      api_keys: current.api_keys.map((key) =>
                        key.id === keyId ? { ...key, key_prefix: prefix } : key
                      ),
                    }
                  );
                }}
                onRefresh={() => void queryClient.invalidateQueries('apiKeys')}
              />
            )}
          </div>
        )}
        {activeTab === 'general' && (
          <div className="space-y-6">
            {isProfile && (
              <MCPProfileBuilder
                profile={config}
                onSaved={(updated) => {
                  queryClient.setQueryData(['mcpConfig', keyId], updated);
                  void queryClient.invalidateQueries('apiKeys');
                }}
              />
            )}
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

            {isProfile && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <h3 className="font-medium text-gray-900">Research programs</h3>
                <p className="mt-1 text-sm text-gray-600">
                  {config.mcp_programs?.join(', ') || 'No research programs selected'}
                </p>
              </div>
            )}
            {!!config.research_notes?.length && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <h3 className="font-medium text-gray-900">Research notes</h3>
                <ul className="mt-1 list-inside list-disc text-sm text-gray-600">
                  {config.research_notes.map((note) => <li key={note.id}>{note.title}</li>)}
                </ul>
              </div>
            )}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">Source Access Mode</h3>
              <p className="text-sm text-gray-500 mb-4">
                Choose the sources available to document browsing, search, and chat.
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
                      Use the shared knowledge base
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
                      Use the sources and permissions set in the Sources tab
                    </div>
                  </div>
                </label>
              </div>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {configuredTools.filter((t) => t.is_enabled).length}
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
                    : 'Document browsing, search, and chat use the sources and permissions below.'}
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
