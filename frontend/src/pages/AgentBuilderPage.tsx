/**
 * Agent Builder page for creating and managing custom agents.
 */

import React, { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import {
  Bot,
  Plus,
  Search,
  Edit,
  Trash2,
  Copy,
  Play,
  Archive,
  Send,
  ChevronRight,
  Check,
  X,
  RefreshCw,
  Sparkles,
  BarChart3,
  Settings,
  FileText,
  Wrench,
  Filter,
  AlertTriangle,
  Info,
  Zap,
  Target,
} from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import {
  AgentDefinition,
  AgentDefinitionCreate,
  AgentDefinitionUpdate,
  AgentDefinitionSummary,
  CapabilityInfo,
} from '../types';
import Button from '../components/common/Button';

type TabId = 'agents' | 'templates' | 'create';
type AgentFilter = 'all' | 'active' | 'draft' | 'archived' | 'system' | 'mine';

const AgentBuilderPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<TabId>('agents');
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [filter, setFilter] = useState<AgentFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch agents
  const { data: agentsData, isLoading: agentsLoading, refetch: refetchAgents } = useQuery(
    ['agents', filter, searchQuery],
    () =>
      apiClient.listAgentDefinitions({
        search: searchQuery || undefined,
        active_only: filter === 'active',
      }),
    { refetchOnWindowFocus: false }
  );

  // Fetch capabilities
  const { data: capabilitiesData } = useQuery(
    'agentCapabilities',
    () => apiClient.listAgentCapabilities(),
    { refetchOnWindowFocus: false }
  );

  // Fetch tools
  const { data: toolsData } = useQuery(
    'agentTools',
    () => apiClient.listAgentTools(),
    { refetchOnWindowFocus: false }
  );

  // Filter agents based on selected filter
  const allAgents = agentsData?.agents || [];
  const filteredAgents = allAgents.filter((agent) => {
    switch (filter) {
      case 'active':
        return agent.is_active;
      case 'draft':
        return !agent.is_system && (agent as any).lifecycle_status === 'draft';
      case 'archived':
        return (agent as any).lifecycle_status === 'archived';
      case 'system':
        return agent.is_system;
      case 'mine':
        return !agent.is_system;
      default:
        return true;
    }
  });

  const tabs = [
    { id: 'agents', name: 'My Agents', icon: Bot },
    { id: 'templates', name: 'Templates', icon: Sparkles },
    { id: 'create', name: 'Create New', icon: Plus },
  ];

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Bot className="w-7 h-7" />
          Agent Builder
        </h1>
        <p className="text-gray-600 mt-1">
          Create, customize, and manage AI agents for specialized tasks
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b mb-6">
        <nav className="flex gap-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id as TabId);
                  setSelectedAgentId(null);
                }}
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
      {activeTab === 'agents' && (
        <AgentsListTab
          agents={filteredAgents}
          isLoading={agentsLoading}
          filter={filter}
          setFilter={setFilter}
          searchQuery={searchQuery}
          setSearchQuery={setSearchQuery}
          selectedAgentId={selectedAgentId}
          setSelectedAgentId={setSelectedAgentId}
          onRefresh={refetchAgents}
          capabilities={capabilitiesData?.capabilities || []}
          tools={toolsData?.tools || []}
        />
      )}

      {activeTab === 'templates' && (
        <TemplatesTab
          onCreateFromTemplate={(agentId) => {
            setActiveTab('agents');
            setSelectedAgentId(agentId);
            refetchAgents();
          }}
        />
      )}

      {activeTab === 'create' && (
        <CreateAgentTab
          capabilities={capabilitiesData?.capabilities || []}
          tools={toolsData?.tools || []}
          onCreated={(agentId) => {
            setActiveTab('agents');
            setSelectedAgentId(agentId);
            refetchAgents();
          }}
        />
      )}
    </div>
  );
};

// Agents List Tab
const AgentsListTab: React.FC<{
  agents: AgentDefinitionSummary[];
  isLoading: boolean;
  filter: AgentFilter;
  setFilter: (f: AgentFilter) => void;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  selectedAgentId: string | null;
  setSelectedAgentId: (id: string | null) => void;
  onRefresh: () => void;
  capabilities: CapabilityInfo[];
  tools: Array<{ name: string; description: string; parameters: any }>;
}> = ({
  agents,
  isLoading,
  filter,
  setFilter,
  searchQuery,
  setSearchQuery,
  selectedAgentId,
  setSelectedAgentId,
  onRefresh,
  capabilities,
  tools,
}) => {
  const filters: { id: AgentFilter; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'active', label: 'Active' },
    { id: 'draft', label: 'Drafts' },
    { id: 'mine', label: 'My Agents' },
    { id: 'system', label: 'System' },
    { id: 'archived', label: 'Archived' },
  ];

  return (
    <div className="flex gap-6">
      {/* Agent List */}
      <div className="w-80 flex-shrink-0">
        {/* Search and Filter */}
        <div className="mb-4 space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search agents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
          <div className="flex flex-wrap gap-1">
            {filters.map((f) => (
              <button
                key={f.id}
                onClick={() => setFilter(f.id)}
                className={`px-2 py-1 text-xs rounded-full transition-colors ${
                  filter === f.id
                    ? 'bg-primary-100 text-primary-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Agent List */}
        <div className="bg-white rounded-lg shadow divide-y max-h-[calc(100vh-320px)] overflow-y-auto">
          {isLoading ? (
            <div className="p-8 text-center">
              <RefreshCw className="w-6 h-6 text-gray-400 animate-spin mx-auto mb-2" />
              <p className="text-sm text-gray-500">Loading agents...</p>
            </div>
          ) : agents.length === 0 ? (
            <div className="p-8 text-center">
              <Bot className="w-10 h-10 text-gray-300 mx-auto mb-2" />
              <p className="text-sm text-gray-500">No agents found</p>
            </div>
          ) : (
            agents.map((agent) => (
              <button
                key={agent.id}
                onClick={() => setSelectedAgentId(agent.id)}
                className={`w-full p-3 text-left hover:bg-gray-50 transition-colors ${
                  selectedAgentId === agent.id ? 'bg-primary-50' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900 truncate">
                        {agent.display_name}
                      </span>
                      {agent.is_system && (
                        <span className="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
                          System
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5 truncate">
                      {agent.description || 'No description'}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {agent.is_active ? (
                        <span className="flex items-center gap-1 text-xs text-green-600">
                          <Check className="w-3 h-3" />
                          Active
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-xs text-gray-400">
                          <X className="w-3 h-3" />
                          Inactive
                        </span>
                      )}
                      <span className="text-xs text-gray-400">
                        {agent.capabilities.length} capabilities
                      </span>
                    </div>
                  </div>
                  <ChevronRight
                    className={`w-4 h-4 flex-shrink-0 ${
                      selectedAgentId === agent.id ? 'text-primary-600' : 'text-gray-400'
                    }`}
                  />
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* Agent Details / Editor */}
      <div className="flex-1">
        {selectedAgentId ? (
          <AgentDetailPanel
            agentId={selectedAgentId}
            capabilities={capabilities}
            tools={tools}
            onClose={() => setSelectedAgentId(null)}
            onRefresh={onRefresh}
          />
        ) : (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <Bot className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Select an Agent</h3>
            <p className="text-gray-500">
              Choose an agent from the list to view details, edit settings, or run tests.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// Agent Detail Panel
const AgentDetailPanel: React.FC<{
  agentId: string;
  capabilities: CapabilityInfo[];
  tools: Array<{ name: string; description: string; parameters: any }>;
  onClose: () => void;
  onRefresh: () => void;
}> = ({ agentId, capabilities, tools, onClose, onRefresh }) => {
  const queryClient = useQueryClient();
  const [activeSection, setActiveSection] = useState<'overview' | 'edit' | 'test' | 'analytics'>(
    'overview'
  );
  const [isEditing, setIsEditing] = useState(false);

  // Fetch full agent details
  const { data: agent, isLoading, refetch } = useQuery(
    ['agent', agentId],
    () => apiClient.getAgentDefinition(agentId),
    { enabled: !!agentId }
  );

  // Mutations
  const updateMutation = useMutation(
    (data: AgentDefinitionUpdate) => apiClient.updateAgentDefinition(agentId, data),
    {
      onSuccess: () => {
        toast.success('Agent updated');
        refetch();
        onRefresh();
        setIsEditing(false);
      },
      onError: () => {
        toast.error('Failed to update agent');
      },
    }
  );

  const deleteMutation = useMutation(() => apiClient.deleteAgentDefinition(agentId), {
    onSuccess: () => {
      toast.success('Agent deleted');
      onClose();
      onRefresh();
    },
    onError: () => {
      toast.error('Failed to delete agent');
    },
  });

  const duplicateMutation = useMutation(() => apiClient.duplicateAgentDefinition(agentId), {
    onSuccess: (data) => {
      toast.success('Agent duplicated');
      onRefresh();
    },
    onError: () => {
      toast.error('Failed to duplicate agent');
    },
  });

  const publishMutation = useMutation(() => apiClient.publishAgent(agentId), {
    onSuccess: () => {
      toast.success('Agent published');
      refetch();
      onRefresh();
    },
    onError: () => {
      toast.error('Failed to publish agent');
    },
  });

  const archiveMutation = useMutation(() => apiClient.archiveAgent(agentId), {
    onSuccess: () => {
      toast.success('Agent archived');
      refetch();
      onRefresh();
    },
    onError: () => {
      toast.error('Failed to archive agent');
    },
  });

  if (isLoading || !agent) {
    return (
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <RefreshCw className="w-6 h-6 text-gray-400 animate-spin mx-auto mb-2" />
        <p className="text-gray-500">Loading agent...</p>
      </div>
    );
  }

  const sections = [
    { id: 'overview', name: 'Overview', icon: FileText },
    { id: 'edit', name: 'Edit', icon: Edit },
    { id: 'test', name: 'Test', icon: Play },
    { id: 'analytics', name: 'Analytics', icon: BarChart3 },
  ];

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-gray-900">{agent.display_name}</h2>
            {agent.is_system && (
              <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">System</span>
            )}
            {agent.lifecycle_status === 'draft' && (
              <span className="px-2 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded">
                Draft
              </span>
            )}
            {agent.lifecycle_status === 'archived' && (
              <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded">
                Archived
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500">{agent.name}</p>
        </div>
        <div className="flex items-center gap-2">
          {!agent.is_system && (
            <>
              {agent.lifecycle_status === 'draft' && (
                <Button
                  size="sm"
                  onClick={() => publishMutation.mutate()}
                  disabled={publishMutation.isLoading}
                >
                  <Send className="w-4 h-4 mr-1" />
                  Publish
                </Button>
              )}
              {agent.lifecycle_status === 'published' && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => archiveMutation.mutate()}
                  disabled={archiveMutation.isLoading}
                >
                  <Archive className="w-4 h-4 mr-1" />
                  Archive
                </Button>
              )}
              <Button
                size="sm"
                variant="ghost"
                onClick={() => duplicateMutation.mutate()}
                disabled={duplicateMutation.isLoading}
              >
                <Copy className="w-4 h-4" />
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  if (window.confirm('Delete this agent? This cannot be undone.')) {
                    deleteMutation.mutate();
                  }
                }}
                disabled={deleteMutation.isLoading}
              >
                <Trash2 className="w-4 h-4 text-red-500" />
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Section Tabs */}
      <div className="border-b px-4">
        <nav className="flex gap-4">
          {sections.map((section) => {
            const Icon = section.icon;
            return (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id as typeof activeSection)}
                className={`flex items-center gap-1 py-2 border-b-2 -mb-px text-sm transition-colors ${
                  activeSection === section.id
                    ? 'border-primary-600 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                {section.name}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Section Content */}
      <div className="p-4">
        {activeSection === 'overview' && (
          <AgentOverview agent={agent} capabilities={capabilities} />
        )}
        {activeSection === 'edit' && (
          <AgentEditForm
            agent={agent}
            capabilities={capabilities}
            tools={tools}
            onSave={(data) => updateMutation.mutate(data)}
            isSaving={updateMutation.isLoading}
            isSystemAgent={agent.is_system}
          />
        )}
        {activeSection === 'test' && <AgentTestPanel agentId={agentId} agent={agent} />}
        {activeSection === 'analytics' && <AgentAnalyticsPanel agentId={agentId} />}
      </div>
    </div>
  );
};

// Agent Overview
const AgentOverview: React.FC<{
  agent: AgentDefinition;
  capabilities: CapabilityInfo[];
}> = ({ agent, capabilities }) => {
  const capabilityMap = capabilities.reduce(
    (acc, c) => ({ ...acc, [c.name]: c }),
    {} as Record<string, CapabilityInfo>
  );

  const [routingDefaultsText, setRoutingDefaultsText] = useState<string>(() => {
    try {
      return agent.routing_defaults ? JSON.stringify(agent.routing_defaults, null, 2) : '';
    } catch {
      return '';
    }
  });

  return (
    <div className="space-y-6">
      {/* Description */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">Description</h3>
        <p className="text-gray-600">{agent.description || 'No description provided.'}</p>
      </div>

      {/* Status */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500">Status</div>
          <div className="font-medium">
            {agent.is_active ? (
              <span className="text-green-600">Active</span>
            ) : (
              <span className="text-gray-500">Inactive</span>
            )}
          </div>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500">Priority</div>
          <div className="font-medium">{agent.priority}</div>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500">Version</div>
          <div className="font-medium">{agent.version || 1}</div>
        </div>
      </div>



      {/* LLM Routing Defaults */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">LLM Routing Defaults</h3>
        <div className="p-3 bg-gray-50 rounded-lg max-h-48 overflow-y-auto">
          <pre className="text-sm text-gray-600 whitespace-pre-wrap font-mono">
            {agent.routing_defaults ? JSON.stringify(agent.routing_defaults, null, 2) : '—'}
          </pre>
        </div>
      </div>

      {/* Capabilities */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">Capabilities</h3>
        <div className="flex flex-wrap gap-2">
          {agent.capabilities.map((cap) => (
            <div
              key={cap}
              className="px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg text-sm"
              title={capabilityMap[cap]?.description}
            >
              {capabilityMap[cap]?.name || cap}
            </div>
          ))}
        </div>
      </div>

      {/* Tools */}
      {agent.tool_whitelist && agent.tool_whitelist.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            Allowed Tools ({agent.tool_whitelist.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {agent.tool_whitelist.map((tool) => (
              <span key={tool} className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-sm">
                {tool}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* System Prompt Preview */}
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-2">System Prompt</h3>
        <div className="p-3 bg-gray-50 rounded-lg max-h-48 overflow-y-auto">
          <pre className="text-sm text-gray-600 whitespace-pre-wrap font-mono">
            {agent.system_prompt || 'No system prompt defined.'}
          </pre>
        </div>
      </div>

      {/* Timestamps */}
      <div className="text-xs text-gray-500 flex gap-4">
        <span>Created: {new Date(agent.created_at).toLocaleDateString()}</span>
        <span>Updated: {new Date(agent.updated_at).toLocaleDateString()}</span>
      </div>
    </div>
  );
};


// LLM Routing Defaults Builder (UI -> JSON)



// Routing preview uses the same precedence as runtime (agent defaults + overrides + user prefs + feature flags).

type AgentRoutingPreviewAttempt = {
  attempt: number;
  tier: string | null;
  tier_provider?: string | null;
  tier_model?: string | null;
  effective_provider?: string | null;
  effective_model?: string | null;
};

type AgentRoutingPreviewResponse = {
  attempts?: AgentRoutingPreviewAttempt[];
  notes?: string[];
  routing_effective?: any;
  user_llm?: any;
};

const useAgentRoutingPreview = (params: {
  agentId: string | null;
  taskType: string;
  routingJson: string;
}) => {
  const [debouncedJson, setDebouncedJson] = React.useState(params.routingJson);

  React.useEffect(() => {
    const t = window.setTimeout(() => setDebouncedJson(params.routingJson), 400);
    return () => window.clearTimeout(t);
  }, [params.routingJson]);

  const parsed = React.useMemo(() => {
    if (!debouncedJson.trim()) return null;
    try {
      const obj = JSON.parse(debouncedJson);
      return obj && typeof obj === 'object' ? obj : null;
    } catch {
      return '__invalid_json__';
    }
  }, [debouncedJson]);

  const jsonError = parsed === '__invalid_json__' ? 'Invalid JSON' : null;


  const query = useQuery(
    ['agent-routing-preview', params.agentId, params.taskType, debouncedJson],
    async () => {
      if (!params.agentId) return null;
      const res = await apiClient.post<AgentRoutingPreviewResponse>(
        `/agent/agents/${params.agentId}/routing-preview`,
        {
          task_type: params.taskType,
          agent_routing_overrides: parsed && parsed !== '__invalid_json__' ? parsed : null,
        }
      );
      return (res.data || null) as AgentRoutingPreviewResponse | null;
    },
    {
      enabled: Boolean(params.agentId) && !jsonError,
      staleTime: 10000,
      retry: 1,
    }
  );

  return { ...query, jsonError };
};

const RoutingDefaultsResolutionPreview: React.FC<{
  agentId?: string | null;
  routingJson: string;
  taskType?: string;
}> = ({ agentId, routingJson, taskType = 'chat' }) => {
  const { data, isLoading, error, refetch, jsonError } = useAgentRoutingPreview({
    agentId: agentId || null,
    taskType,
    routingJson,
  });

  if (!agentId) {
    return (
      <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="text-xs text-gray-600">Preview is available after the agent is created.</div>
      </div>
    );
  }

  if (jsonError) {
    return (
      <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-lg">
        <div className="text-xs text-red-600">{jsonError}</div>
      </div>
    );
  }

  const attempts: AgentRoutingPreviewAttempt[] = (data as any)?.attempts || [];
  const notes: string[] = (data as any)?.notes || [];
  const attemptOrder = attempts.map((a) => a.tier || 'default').join(' → ');

  return (
    <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-lg">
      {Boolean(isLoading) && <div className="text-xs text-gray-500">Loading routing preview…</div>}
      {Boolean(error) && (
        <div className="text-xs text-red-600 flex items-center justify-between gap-2">
          <span>Failed to load routing preview</span>
          <button
            type="button"
            className="text-xs text-gray-600 hover:text-gray-900"
            onClick={() => refetch()}
          >
            Retry
          </button>
        </div>
      )}

      {!Boolean(isLoading) && !Boolean(error) && (
        <div className="space-y-2">
          <div className="text-xs text-gray-600">Attempt order: {attemptOrder || '(default)'}</div>
          <div className="space-y-1">
            {attempts.map((a) => (
              <div
                key={`${a.attempt}-${a.tier || 'default'}`}
                className="text-xs text-gray-800 flex items-center justify-between gap-2"
              >
                <span className="font-mono">{a.tier || 'default'}</span>
                <span className="text-gray-600">
                  {String(a.effective_provider || '—')} / {String(a.effective_model || '—')}
                </span>
              </div>
            ))}
            {attempts.length === 0 && <div className="text-xs text-gray-500">No attempts resolved.</div>}
          </div>

          {notes.length > 0 && (
            <div className="pt-2 border-t border-gray-200 space-y-1">
              {notes.map((n, i) => (
                <div key={i} className="text-xs text-gray-600">• {n}</div>
              ))}
            </div>
          )}

          <div className="pt-2 border-t border-gray-200 flex justify-end">
            <button
              type="button"
              className="text-xs text-gray-600 hover:text-gray-900"
              onClick={() => refetch()}
            >
              Refresh
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const RoutingDefaultsBuilder: React.FC<{
  value: string;
  onChange: (next: string) => void;
}> = ({ value, onChange }) => {
  const parsed = React.useMemo(() => {
    if (!value.trim()) return {};
    try {
      const obj = JSON.parse(value);
      return obj && typeof obj === 'object' ? obj : {};
    } catch {
      return {};
    }
  }, [value]);

  const tier = String((parsed as any).tier || (parsed as any).llm_tier || '');
  const fallbackTiers = Array.isArray((parsed as any).fallback_tiers)
    ? (parsed as any).fallback_tiers
    : Array.isArray((parsed as any).llm_fallback_tiers)
      ? (parsed as any).llm_fallback_tiers
      : [];
  const timeoutSeconds = (parsed as any).timeout_seconds ?? (parsed as any).llm_timeout_seconds;
  const maxTokensCap = (parsed as any).max_tokens_cap ?? (parsed as any).llm_max_tokens_cap;
  const cooldownSeconds = (parsed as any).cooldown_seconds ?? (parsed as any).llm_unhealthy_cooldown_seconds;

  const update = (patch: any) => {
    const next: any = { ...parsed, ...patch };

    // Canonicalize keys
    if (next.llm_tier !== undefined) delete next.llm_tier;
    if (next.llm_fallback_tiers !== undefined) delete next.llm_fallback_tiers;
    if (next.llm_timeout_seconds !== undefined) delete next.llm_timeout_seconds;
    if (next.llm_max_tokens_cap !== undefined) delete next.llm_max_tokens_cap;
    if (next.llm_unhealthy_cooldown_seconds !== undefined) delete next.llm_unhealthy_cooldown_seconds;

    if (!next.tier) delete next.tier;
    if (!Array.isArray(next.fallback_tiers) || next.fallback_tiers.length === 0) delete next.fallback_tiers;
    if (next.timeout_seconds === '' || next.timeout_seconds == null) delete next.timeout_seconds;
    if (next.max_tokens_cap === '' || next.max_tokens_cap == null) delete next.max_tokens_cap;
    if (next.cooldown_seconds === '' || next.cooldown_seconds == null) delete next.cooldown_seconds;

    // Remove empty object
    if (Object.keys(next).length === 0) {
      onChange('');
      return;
    }
    onChange(JSON.stringify(next, null, 2));
  };

  return (
    <div className="grid grid-cols-2 gap-3">
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">Tier</label>
        <select
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={tier}
          onChange={(e) => update({ tier: e.target.value || undefined })}
        >
          <option value="">(default)</option>
          <option value="fast">fast</option>
          <option value="balanced">balanced</option>
          <option value="deep">deep</option>
        </select>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">Fallback tiers (comma)</label>
        <input
          type="text"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={String((fallbackTiers || []).join(', '))}
          onChange={(e) => {
            const arr = e.target.value
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean);
            update({ fallback_tiers: arr.length ? arr : undefined });
          }}
          placeholder="balanced, fast"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">Timeout (sec)</label>
        <input
          type="number"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={timeoutSeconds === undefined || timeoutSeconds === null ? '' : String(timeoutSeconds)}
          onChange={(e) => update({ timeout_seconds: e.target.value ? parseInt(e.target.value, 10) : undefined })}
          min={2}
          max={600}
          placeholder="120"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1">Max tokens cap</label>
        <input
          type="number"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={maxTokensCap === undefined || maxTokensCap === null ? '' : String(maxTokensCap)}
          onChange={(e) => update({ max_tokens_cap: e.target.value ? parseInt(e.target.value, 10) : undefined })}
          min={64}
          max={20000}
          placeholder="2000"
        />
      </div>

      <div className="col-span-2">
        <label className="block text-xs font-medium text-gray-600 mb-1">Provider cooldown (sec)</label>
        <input
          type="number"
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
          value={cooldownSeconds === undefined || cooldownSeconds === null ? '' : String(cooldownSeconds)}
          onChange={(e) => update({ cooldown_seconds: e.target.value ? parseInt(e.target.value, 10) : undefined })}
          min={5}
          max={3600}
          placeholder="60"
        />
      </div>
    </div>
  );
};



type RoutingExperimentVariant = {
  id: string;
  weight: number;
  routing: any;
};

type RoutingExperimentConfig = {
  id: string;
  enabled?: boolean;
  salt?: string;
  variants: RoutingExperimentVariant[];
  winner_variant_id?: string;
  promoted_at?: string;
  promoted_by?: string;
  history?: Array<{ at: string; by?: string; action: string; details?: any }>;
};


function _routingStableJson(obj: any): string {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

function _routingToLines(s: string): string[] {
  return s.replace(/\r\n/g, '\n').split('\n');
}

// Minimal Myers diff for line arrays, returning a unified diff string.
function _routingUnifiedDiff(
  aLines: string[],
  bLines: string[],
  opts?: { aName?: string; bName?: string }
): string {
  const aName = opts?.aName || 'before';
  const bName = opts?.bName || 'after';

  const N = aLines.length;
  const M = bLines.length;
  const max = N + M;

  type V = Record<number, number>;
  const trace: V[] = [];
  let v: V = { 1: 0 };

  const getV = (vv: V, k: number) => (vv[k] == null ? -Infinity : vv[k]);

  for (let d = 0; d <= max; d++) {
    const vNext: V = {};
    for (let k = -d; k <= d; k += 2) {
      let x: number;
      if (k === -d || (k !== d && getV(v, k - 1) < getV(v, k + 1))) {
        x = getV(v, k + 1);
      } else {
        x = getV(v, k - 1) + 1;
      }
      let y = x - k;
      while (x < N && y < M && aLines[x] === bLines[y]) {
        x++;
        y++;
      }
      vNext[k] = x;
      if (x >= N && y >= M) {
        const edits: Array<{ type: ' ' | '+' | '-'; line: string }> = [];
        let x2 = N;
        let y2 = M;

        for (let dd = d; dd > 0; dd--) {
          const vv = trace[dd - 1];
          const k2 = x2 - y2;

          let prevK: number;
          if (k2 === -dd || (k2 !== dd && getV(vv, k2 - 1) < getV(vv, k2 + 1))) {
            prevK = k2 + 1;
          } else {
            prevK = k2 - 1;
          }

          const prevX = getV(vv, prevK);
          const prevY = prevX - prevK;

          while (x2 > prevX && y2 > prevY) {
            edits.push({ type: ' ', line: aLines[x2 - 1] });
            x2--;
            y2--;
          }

          if (x2 === prevX) {
            edits.push({ type: '+', line: bLines[y2 - 1] });
            y2--;
          } else {
            edits.push({ type: '-', line: aLines[x2 - 1] });
            x2--;
          }
        }

        while (x2 > 0 && y2 > 0) {
          edits.push({ type: ' ', line: aLines[x2 - 1] });
          x2--;
          y2--;
        }

        edits.reverse();

        const header = `--- ${aName}\n+++ ${bName}\n@@ -1,${N} +1,${M} @@`;
        const body = edits.map((e) => `${e.type}${e.line}`).join('\n');
        return `${header}\n${body}`;
      }
    }
    trace.push(vNext);
    v = vNext;
  }

  const header = `--- ${aName}\n+++ ${bName}\n@@ -1,${N} +1,${M} @@`;
  return header;
}

const RoutingExperimentBuilder: React.FC<{
  agentId: string;
  routingJson: string;
  onChange: (next: string) => void;
}> = ({ agentId, routingJson, onChange }) => {
  const parsed = React.useMemo(() => {
    if (!routingJson.trim()) return {};
    try {
      const obj = JSON.parse(routingJson);
      return obj && typeof obj === 'object' ? obj : {};
    } catch {
      return '__invalid_json__';
    }
  }, [routingJson]);

  const jsonError = parsed === '__invalid_json__' ? 'Invalid JSON' : null;

  const [promoteModal, setPromoteModal] = React.useState<null | {
    winner: string;
    before: any;
    after: any;
    diff: string;
  }>(null);

  const exp: RoutingExperimentConfig | null = React.useMemo(() => {
    if (!parsed || parsed === '__invalid_json__') return null;
    const e = (parsed as any).experiment;
    if (!e || typeof e !== 'object') return null;
    const id = String((e as any).id || '').trim();
    const variantsRaw = Array.isArray((e as any).variants) ? (e as any).variants : [];
    const variants: RoutingExperimentVariant[] = variantsRaw
      .map((v: any) => {
        const vid = String(v?.id || '').trim();
        const w = Number(v?.weight ?? 0);
        const routing = v?.routing && typeof v.routing === 'object' ? v.routing : {};
        if (!vid) return null;
        return { id: vid, weight: Number.isFinite(w) ? w : 0, routing };
      })
      .filter(Boolean) as any;

    return {
      id,
      enabled: Boolean((e as any).enabled),
      salt: String((e as any).salt || ''),
      variants,
      winner_variant_id: (e as any).winner_variant_id ? String((e as any).winner_variant_id) : undefined,
      promoted_at: (e as any).promoted_at ? String((e as any).promoted_at) : undefined,
      promoted_by: (e as any).promoted_by ? String((e as any).promoted_by) : undefined,
      history: Array.isArray((e as any).history) ? (e as any).history : undefined,
    };
  }, [parsed]);

  const dateFrom = React.useMemo(() => {
    const d = new Date();
    d.setDate(d.getDate() - 7);
    return d.toISOString();
  }, []);

  const recommendationQuery = useQuery(
    ['llmRoutingExperimentRecommendation', agentId, exp?.id],
    () =>
      apiClient.getLLMRoutingExperimentRecommendation({
        experiment_id: String(exp?.id || ''),
        date_from: dateFrom,
        limit: 50000,
      }),
    { enabled: Boolean(exp?.id), refetchOnWindowFocus: false, retry: 1 }
  );

  const updateRoot = (nextObj: any) => {
    if (!nextObj || typeof nextObj !== 'object') {
      onChange('');
      return;
    }
    onChange(JSON.stringify(nextObj, null, 2));
  };

  const ensureBaseRouting = (obj: any) => {
    const next = { ...(obj || {}) };
    if (next.llm_tier !== undefined) delete next.llm_tier;
    if (next.llm_fallback_tiers !== undefined) delete next.llm_fallback_tiers;
    if (next.llm_timeout_seconds !== undefined) delete next.llm_timeout_seconds;
    if (next.llm_max_tokens_cap !== undefined) delete next.llm_max_tokens_cap;
    if (next.llm_unhealthy_cooldown_seconds !== undefined) delete next.llm_unhealthy_cooldown_seconds;
    return next;
  };

  const startNewAB = () => {
    if (jsonError) return;
    const obj = ensureBaseRouting(parsed);
    const id = `exp_${new Date().toISOString().replace(/[-:.TZ]/g, '')}`;
    obj.experiment = {
      id,
      enabled: true,
      salt: '',
      variants: [
        { id: 'A', weight: 50, routing: { tier: 'deep', fallback_tiers: ['balanced'] } },
        { id: 'B', weight: 50, routing: { tier: 'balanced', fallback_tiers: ['fast'] } },
      ],
      history: [{ at: new Date().toISOString(), action: 'created', details: { id } }],
    };
    updateRoot(obj);
  };

  const setEnabled = (enabled: boolean) => {
    if (jsonError) return;
    const obj = ensureBaseRouting(parsed);
    const e = obj.experiment && typeof obj.experiment === 'object' ? obj.experiment : null;
    if (!e) return;
    e.enabled = enabled;
    e.history = Array.isArray(e.history) ? e.history : [];
    e.history.push({ at: new Date().toISOString(), action: enabled ? 'enabled' : 'disabled' });
    obj.experiment = e;
    updateRoot(obj);
  };

  const updateVariant = (vid: string, patch: Partial<RoutingExperimentVariant>) => {
    if (jsonError) return;
    const obj = ensureBaseRouting(parsed);
    const e = obj.experiment && typeof obj.experiment === 'object' ? obj.experiment : null;
    if (!e) return;
    const variants = Array.isArray(e.variants) ? e.variants.slice() : [];
    const idx = variants.findIndex((v: any) => String(v?.id) === vid);
    if (idx === -1) return;
    const cur = variants[idx] || {};
    variants[idx] = { ...cur, ...patch };
    e.variants = variants;
    obj.experiment = e;
    updateRoot(obj);
  };

  const addVariant = () => {
    if (jsonError) return;
    const obj = ensureBaseRouting(parsed);
    const e = obj.experiment && typeof obj.experiment === 'object' ? obj.experiment : null;
    if (!e) return;
    const variants = Array.isArray(e.variants) ? e.variants.slice() : [];
    const nextId = `V${variants.length + 1}`;
    variants.push({ id: nextId, weight: 10, routing: { tier: 'balanced', fallback_tiers: ['fast'] } });
    e.variants = variants;
    e.history = Array.isArray(e.history) ? e.history : [];
    e.history.push({ at: new Date().toISOString(), action: 'variant_added', details: { id: nextId } });
    obj.experiment = e;
    updateRoot(obj);
  };

  const promoteWinner = () => {
    if (jsonError) return;
    const rec = recommendationQuery.data as any;
    const winner = String(rec?.recommended_variant_id || '').trim();
    if (!winner || !exp) {
      toast.error('No recommended variant');
      return;
    }

    const v = exp.variants.find((x) => x.id === winner);
    if (!v) {
      toast.error('Recommended variant not found in current config');
      return;
    }

    const vr = v.routing && typeof v.routing === 'object' ? v.routing : {};

    const before = ensureBaseRouting(parsed);
    const after = ensureBaseRouting(parsed);

    // Apply variant routing into base routing defaults
    if (vr.tier != null) after.tier = String(vr.tier || '').toLowerCase() || undefined;
    if (Array.isArray(vr.fallback_tiers)) after.fallback_tiers = vr.fallback_tiers;
    if (vr.timeout_seconds != null) after.timeout_seconds = Number(vr.timeout_seconds);
    if (vr.max_tokens_cap != null) after.max_tokens_cap = Number(vr.max_tokens_cap);
    if (vr.cooldown_seconds != null) after.cooldown_seconds = Number(vr.cooldown_seconds);

    // Mark experiment disabled + record winner
    const e = after.experiment && typeof after.experiment === 'object' ? after.experiment : {};
    e.enabled = false;
    e.winner_variant_id = winner;
    e.promoted_at = new Date().toISOString();
    e.history = Array.isArray(e.history) ? e.history : [];
    e.history.push({
      at: new Date().toISOString(),
      action: 'promoted',
      details: { winner_variant_id: winner, applied_routing: vr },
    });
    after.experiment = e;

    const beforeJson = _routingStableJson(before);
    const afterJson = _routingStableJson(after);
    const diff = _routingUnifiedDiff(
      _routingToLines(beforeJson),
      _routingToLines(afterJson),
      { aName: 'routing_defaults.before.json', bName: 'routing_defaults.after.json' }
    );

    setPromoteModal({ winner, before, after, diff });
  };

  if (jsonError) {
    return <div className="text-xs text-red-600">{jsonError}</div>;
  }

  return (
    <div className="space-y-3">
      {!exp && (
        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-600">No experiment configured.</div>
          <Button size="sm" variant="secondary" onClick={startNewAB}>
            Start A/B
          </Button>
        </div>
      )}

      {exp && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-gray-900">Experiment</div>
              <div className="text-xs text-gray-600 font-mono">{exp.id || '—'}</div>
            </div>
            <div className="flex items-center gap-2">
              <Button size="sm" variant={exp.enabled ? 'secondary' : 'primary'} onClick={() => setEnabled(!Boolean(exp.enabled))}>
                {exp.enabled ? 'Stop' : 'Start'}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => recommendationQuery.refetch()}>
                Refresh recommendation
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Salt (optional)</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={exp.salt || ''}
                onChange={(e) => {
                  const obj = ensureBaseRouting(parsed);
                  const ee = obj.experiment && typeof obj.experiment === 'object' ? obj.experiment : {};
                  ee.salt = e.target.value;
                  obj.experiment = ee;
                  updateRoot(obj);
                }}
              />
            </div>
            <div className="flex items-end justify-end">
              <Button size="sm" variant="secondary" onClick={addVariant}>
                Add variant
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            {exp.variants.map((v) => (
              <div key={v.id} className="border border-gray-200 rounded-lg p-3 bg-white">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-sm font-medium text-gray-900">Variant {v.id}</div>
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-gray-600">Weight</label>
                    <input
                      type="number"
                      className="w-24 border border-gray-300 rounded-lg px-2 py-1 text-sm"
                      value={String(v.weight)}
                      min={0}
                      onChange={(e) => updateVariant(v.id, { weight: Number(e.target.value || 0) })}
                    />
                  </div>
                </div>

                <div className="mt-2">
                  <label className="block text-xs font-medium text-gray-600 mb-1">Routing JSON</label>
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs font-mono"
                    rows={4}
                    value={JSON.stringify(v.routing || {}, null, 2)}
                    onChange={(e) => {
                      try {
                        const obj = JSON.parse(e.target.value || '{}');
                        updateVariant(v.id, { routing: obj && typeof obj === 'object' ? obj : {} });
                      } catch {
                        // ignore invalid edits
                      }
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
            <div className="text-sm font-medium text-gray-900">Recommendation (last 7d)</div>
            {recommendationQuery.isLoading ? (
              <div className="text-xs text-gray-600 mt-1">Loading…</div>
            ) : recommendationQuery.error ? (
              <div className="text-xs text-red-600 mt-1">Failed to load recommendation</div>
            ) : (
              <div className="text-xs text-gray-700 mt-1">
                {(recommendationQuery.data as any)?.recommended_variant_id
                  ? `Recommended: ${(recommendationQuery.data as any).recommended_variant_id} — ${(recommendationQuery.data as any).rationale}`
                  : 'No recommendation (no data).'}
              </div>
            )}
            <div className="mt-2 flex justify-end">
              <Button
                size="sm"
                variant="primary"
                disabled={!Boolean((recommendationQuery.data as any)?.recommended_variant_id)}
                onClick={promoteWinner}
              >
                Promote winner
              </Button>
            </div>
          </div>

          {Array.isArray(exp.history) && exp.history.length > 0 && (
            <details className="border border-gray-200 rounded-lg p-3 bg-white">
              <summary className="text-sm text-gray-700 cursor-pointer">History</summary>
              <div className="mt-2 space-y-1">
                {exp.history.slice().reverse().slice(0, 20).map((h, i) => (
                  <div key={i} className="text-xs text-gray-600 font-mono">{h.at} · {h.action}</div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}


      {promoteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
            <div className="p-4 border-b flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-gray-900">Confirm promotion</div>
                <div className="text-xs text-gray-600">
                  Winner: <span className="font-mono">{promoteModal.winner}</span>
                </div>
              </div>
              <button
                type="button"
                className="text-xs text-gray-600 hover:text-gray-900"
                onClick={() => setPromoteModal(null)}
              >
                Close
              </button>
            </div>

            <div className="p-4 space-y-3">
              <div className="text-sm text-gray-900 font-medium">Diff (routing_defaults)</div>
              <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-3 overflow-x-auto max-h-96">{promoteModal.diff}</pre>

              <details className="border border-gray-200 rounded p-3">
                <summary className="text-xs text-gray-700 cursor-pointer">Show before/after JSON</summary>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                  <div>
                    <div className="text-xs text-gray-600 mb-1">Before</div>
                    <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto max-h-72">{_routingStableJson(promoteModal.before)}</pre>
                  </div>
                  <div>
                    <div className="text-xs text-gray-600 mb-1">After</div>
                    <pre className="text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto max-h-72">{_routingStableJson(promoteModal.after)}</pre>
                  </div>
                </div>
              </details>
            </div>

            <div className="p-4 border-t flex items-center justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={() => setPromoteModal(null)}>
                Cancel
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(promoteModal.diff);
                    toast.success('Copied diff');
                  } catch {
                    toast.error('Failed to copy diff');
                  }
                }}
              >
                Copy diff
              </Button>
              <Button
                size="sm"
                variant="primary"
                onClick={() => {
                  updateRoot(promoteModal.after);
                  setPromoteModal(null);
                  toast.success(`Promoted variant ${promoteModal.winner} to base routing`);
                }}
              >
                Confirm promote
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
// Agent Edit Form
const AgentEditForm: React.FC<{
  agent: AgentDefinition;
  capabilities: CapabilityInfo[];
  tools: Array<{ name: string; description: string; parameters: any }>;
  onSave: (data: AgentDefinitionUpdate) => void;
  isSaving: boolean;
  isSystemAgent: boolean;
}> = ({ agent, capabilities, tools, onSave, isSaving, isSystemAgent }) => {
  const [displayName, setDisplayName] = useState(agent.display_name);
  const [description, setDescription] = useState(agent.description || '');
  const [systemPrompt, setSystemPrompt] = useState(agent.system_prompt || '');
  const [routingDefaultsText, setRoutingDefaultsText] = useState(
    agent.routing_defaults ? JSON.stringify(agent.routing_defaults, null, 2) : ''
  );
  const routingDefaultsError = React.useMemo(() => {
    if (!routingDefaultsText.trim()) return null;
    try {
      JSON.parse(routingDefaultsText);
      return null;
    } catch {
      return 'Invalid JSON';
    }
  }, [routingDefaultsText]);
  const [selectedCapabilities, setSelectedCapabilities] = useState<string[]>(agent.capabilities);
  const [selectedTools, setSelectedTools] = useState<string[]>(agent.tool_whitelist || []);
  const [priority, setPriority] = useState(agent.priority);
  const [isActive, setIsActive] = useState(agent.is_active);
  const [useAllTools, setUseAllTools] = useState(!agent.tool_whitelist);

  const handleSubmit = () => {
    if (routingDefaultsError) {
      toast.error('routing defaults must be valid JSON');
      return;
    }
    let routingDefaults: any = null;
    if (routingDefaultsText.trim()) {
      try {
        routingDefaults = JSON.parse(routingDefaultsText);
      } catch {
        toast.error('routing defaults must be valid JSON');
        return;
      }
    }

    onSave({
      display_name: displayName,
      description: description || null,
      system_prompt: systemPrompt,
      capabilities: selectedCapabilities,
      tool_whitelist: useAllTools ? null : selectedTools,
      routing_defaults: routingDefaults,
      priority,
      is_active: isActive,
    });
  };

  const toggleCapability = (cap: string) => {
    setSelectedCapabilities((prev) =>
      prev.includes(cap) ? prev.filter((c) => c !== cap) : [...prev, cap]
    );
  };

  const toggleTool = (tool: string) => {
    setSelectedTools((prev) =>
      prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool]
    );
  };

  if (isSystemAgent) {
    return (
      <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <div className="flex gap-2">
          <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
          <div>
            <p className="font-medium text-yellow-800">System Agent</p>
            <p className="text-sm text-yellow-700">
              System agents cannot be edited. You can duplicate this agent to create your own
              customized version.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Basic Info */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Display Name</label>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
          <input
            type="number"
            value={priority}
            onChange={(e) => setPriority(parseInt(e.target.value) || 50)}
            min={1}
            max={100}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          />
          <p className="text-xs text-gray-500 mt-1">Higher = preferred when multiple agents match</p>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={2}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          placeholder="Describe what this agent does..."
        />
      </div>

      {/* System Prompt */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">System Prompt</label>
        <textarea
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          rows={8}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 font-mono text-sm"
          placeholder="You are an AI assistant specialized in..."
        />
      </div>


      {/* LLM Routing Defaults */}
      <div>
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-700">LLM Routing Defaults</label>
          <button
            type="button"
            className="text-xs text-gray-600 hover:text-gray-900"
            onClick={() => setRoutingDefaultsText('')}
          >
            Clear
          </button>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
          <RoutingDefaultsBuilder value={routingDefaultsText} onChange={setRoutingDefaultsText} />

        <details className="mt-2">
          <summary className="text-xs text-gray-600 cursor-pointer">Preview effective routing</summary>
          <RoutingDefaultsResolutionPreview agentId={agent.id} routingJson={routingDefaultsText} />
        </details>
        <details className="mt-2">
          <summary className="text-xs text-gray-600 cursor-pointer">Experiments</summary>
          <div className="mt-2">
            <RoutingExperimentBuilder agentId={agent.id} routingJson={routingDefaultsText} onChange={setRoutingDefaultsText} />
          </div>
        </details>
        </div>
        <details className="mt-2">
          <summary className="text-xs text-gray-600 cursor-pointer">Advanced JSON</summary>
          <textarea
            value={routingDefaultsText}
            onChange={(e) => setRoutingDefaultsText(e.target.value)}
            rows={5}
            className="mt-2 w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 font-mono text-sm"
            placeholder='{"tier":"balanced","fallback_tiers":["fast"],"timeout_seconds":120,"max_tokens_cap":2000,"cooldown_seconds":60}'
          />
        </details>
        {routingDefaultsError && (
          <div className="mt-2 text-xs text-red-600">{routingDefaultsError}</div>
        )}
      </div>

      {/* Capabilities */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Capabilities</label>
        <div className="grid grid-cols-2 gap-2">
          {capabilities.map((cap) => (
            <label
              key={cap.name}
              className={`flex items-start gap-2 p-2 border rounded-lg cursor-pointer transition-colors ${
                selectedCapabilities.includes(cap.name)
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedCapabilities.includes(cap.name)}
                onChange={() => toggleCapability(cap.name)}
                className="mt-0.5"
              />
              <div>
                <span className="text-sm font-medium text-gray-900">{cap.name}</span>
                <p className="text-xs text-gray-500">{cap.description}</p>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Tools */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-gray-700">Tools</label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useAllTools}
              onChange={(e) => setUseAllTools(e.target.checked)}
            />
            Allow all tools
          </label>
        </div>
        {!useAllTools && (
          <div className="grid grid-cols-3 gap-2 max-h-48 overflow-y-auto p-2 border rounded-lg">
            {tools.map((tool) => (
              <label
                key={tool.name}
                className={`flex items-center gap-2 p-2 rounded cursor-pointer text-sm ${
                  selectedTools.includes(tool.name)
                    ? 'bg-primary-50 text-primary-700'
                    : 'hover:bg-gray-50'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedTools.includes(tool.name)}
                  onChange={() => toggleTool(tool.name)}
                />
                {tool.name}
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Active Status */}
      <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
        <label className="relative inline-flex items-center cursor-pointer">
          <input
            type="checkbox"
            checked={isActive}
            onChange={(e) => setIsActive(e.target.checked)}
            className="sr-only peer"
          />
          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
        </label>
        <div>
          <span className="font-medium text-gray-900">Active</span>
          <p className="text-xs text-gray-500">
            Active agents can be routed to for handling user requests
          </p>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <Button onClick={handleSubmit} disabled={isSaving}>
          {isSaving ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>
    </div>
  );
};

// Agent Test Panel
const AgentTestPanel: React.FC<{
  agentId: string;
  agent: AgentDefinition;
}> = ({ agentId, agent }) => {
  const [testMessage, setTestMessage] = useState('');
  const [testResult, setTestResult] = useState<any>(null);

  const testMutation = useMutation(
    (message: string) => apiClient.testAgentRouting(agentId, message),
    {
      onSuccess: (data) => setTestResult(data),
      onError: () => {
        toast.error('Test failed');
      },
    }
  );

  const handleTest = () => {
    if (!testMessage.trim()) {
      toast.error('Enter a test message');
      return;
    }
    testMutation.mutate(testMessage);
  };

  return (
    <div className="space-y-6">
      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex gap-2">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0" />
          <div className="text-sm text-blue-800">
            <p className="font-medium">Test Agent Routing</p>
            <p>Enter a test message to see how this agent would analyze and handle it.</p>
          </div>
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Test Message</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={testMessage}
            onChange={(e) => setTestMessage(e.target.value)}
            placeholder="e.g., Find all documents about machine learning"
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
            onKeyDown={(e) => e.key === 'Enter' && handleTest()}
          />
          <Button onClick={handleTest} disabled={testMutation.isLoading}>
            <Play className="w-4 h-4 mr-1" />
            Test
          </Button>
        </div>
      </div>

      {testMutation.isLoading && (
        <div className="text-center py-4">
          <RefreshCw className="w-6 h-6 text-gray-400 animate-spin mx-auto mb-2" />
          <p className="text-sm text-gray-500">Running test...</p>
        </div>
      )}

      {testResult && (
        <div className="space-y-4">
          {/* Intent Analysis */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
              <Target className="w-4 h-4" />
              Intent Analysis
            </h4>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-500">Detected Capabilities:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {testResult.intent_analysis.detected_capabilities.map((cap: string) => (
                    <span key={cap} className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
                      {cap}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <span className="text-gray-500">Keywords:</span>
                <span className="ml-2 text-gray-900">
                  {testResult.intent_analysis.intent_keywords.join(', ') || 'None'}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Match Score:</span>
                <span className="ml-2 font-medium text-gray-900">
                  {(testResult.intent_analysis.score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>

          {/* Routing Result */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Routing Result
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                {testResult.routing_result.would_route ? (
                  <Check className="w-4 h-4 text-green-600" />
                ) : (
                  <X className="w-4 h-4 text-red-600" />
                )}
                <span>
                  {testResult.routing_result.would_route
                    ? 'Would route to this agent'
                    : 'Would NOT route to this agent'}
                </span>
              </div>
              {testResult.routing_result.selected_agent && (
                <div>
                  <span className="text-gray-500">Selected Agent:</span>
                  <span className="ml-2 text-gray-900">
                    {testResult.routing_result.selected_agent}
                  </span>
                </div>
              )}
              <div>
                <span className="text-gray-500">Reason:</span>
                <span className="ml-2 text-gray-900">{testResult.routing_result.routing_reason}</span>
              </div>
            </div>
          </div>

          {/* Available Tools */}
          <div>
            <h4 className="font-medium text-gray-700 mb-2">
              Available Tools ({testResult.total_tools_available})
            </h4>
            <div className="flex flex-wrap gap-1">
              {testResult.available_tools.slice(0, 10).map((tool: string) => (
                <span key={tool} className="px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-xs">
                  {tool}
                </span>
              ))}
              {testResult.total_tools_available > 10 && (
                <span className="px-2 py-0.5 text-gray-500 text-xs">
                  +{testResult.total_tools_available - 10} more
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Agent Analytics Panel
const AgentAnalyticsPanel: React.FC<{
  agentId: string;
}> = ({ agentId }) => {
  const [days, setDays] = useState(30);

  const { data: analytics, isLoading } = useQuery(
    ['agentAnalytics', agentId, days],
    () => apiClient.getAgentAnalytics(agentId, days),
    { enabled: !!agentId }
  );

  if (isLoading) {
    return (
      <div className="text-center py-8">
        <RefreshCw className="w-6 h-6 text-gray-400 animate-spin mx-auto mb-2" />
        <p className="text-sm text-gray-500">Loading analytics...</p>
      </div>
    );
  }

  if (!analytics) {
    return (
      <div className="text-center py-8 text-gray-500">No analytics data available</div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Period Selector */}
      <div className="flex justify-end">
        <select
          value={days}
          onChange={(e) => setDays(parseInt(e.target.value))}
          className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm"
        >
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-4 bg-blue-50 rounded-lg text-center">
          <div className="text-2xl font-bold text-blue-700">
            {analytics.summary.total_turns}
          </div>
          <div className="text-sm text-blue-600">Total Turns</div>
        </div>
        <div className="p-4 bg-green-50 rounded-lg text-center">
          <div className="text-2xl font-bold text-green-700">
            {analytics.summary.unique_conversations}
          </div>
          <div className="text-sm text-green-600">Conversations</div>
        </div>
        <div className="p-4 bg-purple-50 rounded-lg text-center">
          <div className="text-2xl font-bold text-purple-700">
            {analytics.summary.total_tool_calls}
          </div>
          <div className="text-sm text-purple-600">Tool Calls</div>
        </div>
        <div className="p-4 bg-orange-50 rounded-lg text-center">
          <div className="text-2xl font-bold text-orange-700">
            {analytics.summary.handoffs_received}
          </div>
          <div className="text-sm text-orange-600">Handoffs</div>
        </div>
      </div>

      {/* Tool Usage */}
      {analytics.tool_usage.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-3">Top Tools</h4>
          <div className="space-y-2">
            {analytics.tool_usage.map((tool) => {
              const maxCount = Math.max(...analytics.tool_usage.map((t) => t.call_count));
              const percentage = (tool.call_count / maxCount) * 100;
              return (
                <div key={tool.tool_name} className="flex items-center gap-3">
                  <div className="w-32 text-sm text-gray-700 truncate">{tool.tool_name}</div>
                  <div className="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-500 h-2 rounded-full"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <div className="text-sm text-gray-600 w-16 text-right">
                    {tool.call_count}
                    {tool.avg_execution_time_ms && (
                      <span className="text-xs text-gray-400 ml-1">
                        ({tool.avg_execution_time_ms.toFixed(0)}ms)
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Daily Trend */}
      {analytics.daily_trend.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-3">Daily Activity</h4>
          <div className="h-32 flex items-end gap-1">
            {analytics.daily_trend.map((day, i) => {
              const maxTurns = Math.max(...analytics.daily_trend.map((d) => d.turns));
              const height = maxTurns > 0 ? (day.turns / maxTurns) * 100 : 0;
              return (
                <div
                  key={i}
                  className="flex-1 bg-primary-200 rounded-t hover:bg-primary-300 transition-colors"
                  style={{ height: `${Math.max(height, 2)}%` }}
                  title={`${day.date}: ${day.turns} turns`}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// Templates Tab
const TemplatesTab: React.FC<{
  onCreateFromTemplate: (agentId: string) => void;
}> = ({ onCreateFromTemplate }) => {
  const [selectedCategory, setSelectedCategory] = useState<string | undefined>();
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);

  const { data: templatesData, isLoading } = useQuery(
    ['agentTemplates', selectedCategory],
    () => apiClient.listAgentTemplates(selectedCategory),
    { refetchOnWindowFocus: false }
  );

  const { data: templateDetails } = useQuery(
    ['agentTemplate', selectedTemplate],
    () => (selectedTemplate ? apiClient.getAgentTemplate(selectedTemplate) : null),
    { enabled: !!selectedTemplate }
  );

  const createMutation = useMutation(
    (templateId: string) => apiClient.createAgentFromTemplate(templateId),
    {
      onSuccess: (data) => {
        toast.success(data.message);
        onCreateFromTemplate(data.agent.id);
      },
      onError: () => {
        toast.error('Failed to create agent from template');
      },
    }
  );

  const templates = templatesData?.templates || [];
  const categories = templatesData?.categories || [];

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <Sparkles className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-800">
          <p className="font-medium">Agent Templates</p>
          <p>
            Start with a pre-configured template and customize it to your needs. Templates provide
            optimized system prompts, capabilities, and tool configurations for common use cases.
          </p>
        </div>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setSelectedCategory(undefined)}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            !selectedCategory
              ? 'bg-primary-100 text-primary-700'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          All
        </button>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors capitalize ${
              selectedCategory === cat
                ? 'bg-primary-100 text-primary-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="text-center py-8">
          <RefreshCw className="w-6 h-6 text-gray-400 animate-spin mx-auto mb-2" />
          <p className="text-gray-500">Loading templates...</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
          {templates.map((template) => (
            <div
              key={template.template_id}
              className={`p-4 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
                selectedTemplate === template.template_id
                  ? 'border-primary-500 ring-2 ring-primary-200'
                  : 'border-gray-200'
              }`}
              onClick={() => setSelectedTemplate(template.template_id)}
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className="font-medium text-gray-900">{template.display_name}</h3>
                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs capitalize">
                  {template.category}
                </span>
              </div>
              <p className="text-sm text-gray-600 mb-3 line-clamp-2">{template.description}</p>
              <div className="flex flex-wrap gap-1 mb-3">
                {template.capabilities.slice(0, 3).map((cap) => (
                  <span key={cap} className="px-2 py-0.5 bg-blue-50 text-blue-700 rounded text-xs">
                    {cap}
                  </span>
                ))}
                {template.capabilities.length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{template.capabilities.length - 3} more
                  </span>
                )}
              </div>
              <Button
                size="sm"
                fullWidth
                onClick={(e) => {
                  e.stopPropagation();
                  createMutation.mutate(template.template_id);
                }}
                disabled={createMutation.isLoading}
              >
                Use Template
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* Template Details Modal */}
      {selectedTemplate && templateDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    {templateDetails.display_name}
                  </h2>
                  <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs capitalize">
                    {templateDetails.category}
                  </span>
                </div>
                <button
                  onClick={() => setSelectedTemplate(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <p className="text-gray-600 mb-4">{templateDetails.description}</p>

              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Capabilities</h3>
                  <div className="flex flex-wrap gap-2">
                    {templateDetails.capabilities.map((cap) => (
                      <span key={cap} className="px-2 py-1 bg-blue-50 text-blue-700 rounded text-sm">
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>

                {templateDetails.use_cases && templateDetails.use_cases.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Use Cases</h3>
                    <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                      {templateDetails.use_cases.map((useCase, i) => (
                        <li key={i}>{useCase}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-2">System Prompt Preview</h3>
                  <div className="p-3 bg-gray-50 rounded-lg max-h-40 overflow-y-auto">
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono">
                      {templateDetails.system_prompt.substring(0, 500)}
                      {templateDetails.system_prompt.length > 500 && '...'}
                    </pre>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end gap-2">
                <Button variant="ghost" onClick={() => setSelectedTemplate(null)}>
                  Cancel
                </Button>
                <Button
                  onClick={() => {
                    createMutation.mutate(templateDetails.template_id);
                    setSelectedTemplate(null);
                  }}
                  disabled={createMutation.isLoading}
                >
                  <Plus className="w-4 h-4 mr-1" />
                  Create from Template
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Create Agent Tab
const CreateAgentTab: React.FC<{
  capabilities: CapabilityInfo[];
  tools: Array<{ name: string; description: string; parameters: any }>;
  onCreated: (agentId: string) => void;
}> = ({ capabilities, tools, onCreated }) => {
  const [name, setName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [description, setDescription] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [routingDefaultsText, setRoutingDefaultsText] = useState('');
  const routingDefaultsErrorCreate = React.useMemo(() => {
    if (!routingDefaultsText.trim()) return null;
    try {
      JSON.parse(routingDefaultsText);
      return null;
    } catch {
      return 'Invalid JSON';
    }
  }, [routingDefaultsText]);
  const [selectedCapabilities, setSelectedCapabilities] = useState<string[]>([]);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [priority, setPriority] = useState(50);
  const [useAllTools, setUseAllTools] = useState(true);

  const createMutation = useMutation(
    (data: AgentDefinitionCreate) => apiClient.createAgentDefinition(data),
    {
      onSuccess: (data) => {
        toast.success('Agent created successfully');
        onCreated(data.id);
      },
      onError: () => {
        toast.error('Failed to create agent');
      },
    }
  );

  const handleCreate = () => {
    if (routingDefaultsErrorCreate) {
      toast.error('routing defaults must be valid JSON');
      return;
    }
    if (!name.trim()) {
      toast.error('Please enter an agent name');
      return;
    }
    if (!displayName.trim()) {
      toast.error('Please enter a display name');
      return;
    }
    if (!systemPrompt.trim()) {
      toast.error('Please enter a system prompt');
      return;
    }
    if (selectedCapabilities.length === 0) {
      toast.error('Please select at least one capability');
      return;
    }

    let routingDefaults: any = null;
    if (routingDefaultsText.trim()) {
      try {
        routingDefaults = JSON.parse(routingDefaultsText);
      } catch {
        toast.error('routing defaults must be valid JSON');
        return;
      }
    }

    createMutation.mutate({
      name: name.trim().toLowerCase().replace(/\s+/g, '_'),
      display_name: displayName.trim(),
      description: description.trim() || null,
      system_prompt: systemPrompt.trim(),
      capabilities: selectedCapabilities,
      tool_whitelist: useAllTools ? null : selectedTools,
      routing_defaults: routingDefaults,
      priority,
      is_active: false, // Start as draft
    });
  };

  const toggleCapability = (cap: string) => {
    setSelectedCapabilities((prev) =>
      prev.includes(cap) ? prev.filter((c) => c !== cap) : [...prev, cap]
    );
  };

  const toggleTool = (tool: string) => {
    setSelectedTools((prev) =>
      prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool]
    );
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-6">Create New Agent</h2>

        <div className="space-y-6">
          {/* Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                placeholder="my_custom_agent"
              />
              <p className="text-xs text-gray-500 mt-1">Unique identifier (lowercase, no spaces)</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Display Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                placeholder="My Custom Agent"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              placeholder="Describe what this agent does..."
            />
          </div>

          {/* System Prompt */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              System Prompt <span className="text-red-500">*</span>
            </label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={8}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 font-mono text-sm"
              placeholder="You are an AI assistant specialized in..."
            />
            <p className="text-xs text-gray-500 mt-1">
              Define the agent's personality, expertise, and behavior
            </p>
          </div>



          {/* LLM Routing Defaults */}
          <div>
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-700">LLM Routing Defaults</label>
              <button
                type="button"
                className="text-xs text-gray-600 hover:text-gray-900"
                onClick={() => setRoutingDefaultsText('')}
              >
                Clear
              </button>
            </div>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <RoutingDefaultsBuilder value={routingDefaultsText} onChange={setRoutingDefaultsText} />

        <details className="mt-2">
          <summary className="text-xs text-gray-600 cursor-pointer">Preview effective routing (after create)</summary>
          <RoutingDefaultsResolutionPreview routingJson={routingDefaultsText} />
        </details>
            </div>
            <details className="mt-2">
              <summary className="text-xs text-gray-600 cursor-pointer">Advanced JSON</summary>
              <textarea
                value={routingDefaultsText}
                onChange={(e) => setRoutingDefaultsText(e.target.value)}
                rows={4}
                className="mt-2 w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 font-mono text-sm"
                placeholder='{"tier":"balanced","fallback_tiers":["fast"],"timeout_seconds":120,"max_tokens_cap":2000,"cooldown_seconds":60}'
              />
            </details>
            {routingDefaultsErrorCreate && (
              <div className="mt-2 text-xs text-red-600">{routingDefaultsErrorCreate}</div>
            )}
          </div>

          {/* Priority */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
            <input
              type="number"
              value={priority}
              onChange={(e) => setPriority(parseInt(e.target.value) || 50)}
              min={1}
              max={100}
              className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Higher priority agents are preferred when multiple agents match (1-100)
            </p>
          </div>

          {/* Capabilities */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Capabilities <span className="text-red-500">*</span>
            </label>
            <div className="grid grid-cols-2 gap-2">
              {capabilities.map((cap) => (
                <label
                  key={cap.name}
                  className={`flex items-start gap-2 p-2 border rounded-lg cursor-pointer transition-colors ${
                    selectedCapabilities.includes(cap.name)
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedCapabilities.includes(cap.name)}
                    onChange={() => toggleCapability(cap.name)}
                    className="mt-0.5"
                  />
                  <div>
                    <span className="text-sm font-medium text-gray-900">{cap.name}</span>
                    <p className="text-xs text-gray-500">{cap.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Tools */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">Tools</label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={useAllTools}
                  onChange={(e) => setUseAllTools(e.target.checked)}
                />
                Allow all tools
              </label>
            </div>
            {!useAllTools && (
              <div className="grid grid-cols-3 gap-2 max-h-48 overflow-y-auto p-2 border rounded-lg">
                {tools.map((tool) => (
                  <label
                    key={tool.name}
                    className={`flex items-center gap-2 p-2 rounded cursor-pointer text-sm ${
                      selectedTools.includes(tool.name)
                        ? 'bg-primary-50 text-primary-700'
                        : 'hover:bg-gray-50'
                    }`}
                    title={tool.description}
                  >
                    <input
                      type="checkbox"
                      checked={selectedTools.includes(tool.name)}
                      onChange={() => toggleTool(tool.name)}
                    />
                    {tool.name}
                  </label>
                ))}
              </div>
            )}
          </div>

          {/* Create Button */}
          <div className="flex justify-end gap-2 pt-4 border-t">
            <Button
              onClick={handleCreate}
              disabled={createMutation.isLoading}
            >
              {createMutation.isLoading ? 'Creating...' : 'Create Agent'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentBuilderPage;
