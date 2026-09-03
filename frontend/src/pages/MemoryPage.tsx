/**
 * Memory management page for conversation context
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { 
  Brain, 
  Search, 
  Trash2, 
  Eye, 
  RefreshCw,
  Tag,
  Clock,
  Star,
  Settings,
  BarChart3,
  FileText,
  Target,
  AlertTriangle,
  X
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import { apiClient } from '../services/api';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ConfirmationModal from '../components/common/ConfirmationModal';
import toast from 'react-hot-toast';

interface Memory {
  id: string;
  memory_type: 'fact' | 'preference' | 'context' | 'summary' | 'goal' | 'constraint';
  content: string;
  importance_score: number;
  context?: any;
  tags?: string[];
  created_at: string;
  last_accessed_at: string;
  access_count: number;
  is_active: boolean;
}

interface MemoryStats {
  total_memories: number;
  memories_by_type: Record<string, number>;
  recent_memories: number;
  most_accessed_memories: Memory[];
  memory_usage_trend: Array<{ date: string; count: number }>;
}

const MemoryPage: React.FC = () => {
  const queryClient = useQueryClient();
  
  const [activeTab, setActiveTab] = useState('memories');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string>('');
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [memoryToDelete, setMemoryToDelete] = useState<string | null>(null);

  // Fetch memories
  const { data: memories, isLoading: memoriesLoading, refetch: refetchMemories } = useQuery(
    ['memories', searchQuery, selectedType],
    () => apiClient.searchMemories({
      query: searchQuery || '',
      memory_types: selectedType ? [selectedType] : undefined,
      limit: 50,
    }),
    {
      refetchOnWindowFocus: false,
    }
  );

  // Fetch memory stats
  const { data: stats, isLoading: statsLoading } = useQuery(
    'memoryStats',
    () => apiClient.getMemoryStats(),
    {
      refetchOnWindowFocus: false,
    }
  );

  // Delete memory mutation
  const deleteMemoryMutation = useMutation(
    (memoryId: string) => apiClient.deleteMemory(memoryId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('memories');
        toast.success('Memory deleted successfully');
      },
      onError: () => {
        toast.error('Failed to delete memory');
      },
    }
  );

  const handleDeleteMemory = (memoryId: string) => {
    setMemoryToDelete(memoryId);
    setDeleteConfirmOpen(true);
  };

  const confirmDeleteMemory = () => {
    if (memoryToDelete) {
      deleteMemoryMutation.mutate(memoryToDelete);
      setDeleteConfirmOpen(false);
      setMemoryToDelete(null);
    }
  };

  const tabs = [
    { id: 'memories', name: 'My Memories', icon: Brain },
    { id: 'stats', name: 'Statistics', icon: BarChart3 },
    { id: 'settings', name: 'Settings', icon: Settings },
  ];

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Conversation Memory</h1>
        <p className="text-gray-600">Manage your AI's memory and conversation context</p>
      </div>

      <div className="flex space-x-8">
        {/* Sidebar */}
        <div className="w-64 flex-shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors duration-200 ${
                    activeTab === tab.id
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  {tab.name}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1">
          {activeTab === 'memories' && (
            <MemoriesTab
              memories={memories || []}
              isLoading={memoriesLoading}
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              selectedType={selectedType}
              setSelectedType={setSelectedType}
              onDeleteMemory={handleDeleteMemory}
              onViewMemory={setSelectedMemory}
              onRefresh={refetchMemories}
            />
          )}
          {activeTab === 'stats' && (
            <StatsTab stats={stats || {
              total_memories: 0,
              memories_by_type: {},
              recent_memories: 0,
              most_accessed_memories: [],
              memory_usage_trend: []
            }} isLoading={statsLoading} />
          )}
          {activeTab === 'settings' && (
            <SettingsTab />
          )}
        </div>
      </div>

      {/* Memory Details Modal */}
      {selectedMemory && (
        <MemoryDetailsModal
          memory={selectedMemory}
          onClose={() => setSelectedMemory(null)}
        />
      )}

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteConfirmOpen}
        onClose={() => {
          setDeleteConfirmOpen(false);
          setMemoryToDelete(null);
        }}
        onConfirm={confirmDeleteMemory}
        title="Delete Memory"
        message="Are you sure you want to delete this memory? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteMemoryMutation.isLoading}
      />
    </div>
  );
};

// Memories Tab Component
interface MemoriesTabProps {
  memories: Memory[];
  isLoading: boolean;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  selectedType: string;
  setSelectedType: (type: string) => void;
  onDeleteMemory: (id: string) => void;
  onViewMemory: (memory: Memory) => void;
  onRefresh: () => void;
}

const MemoriesTab: React.FC<MemoriesTabProps> = ({
  memories,
  isLoading,
  searchQuery,
  setSearchQuery,
  selectedType,
  setSelectedType,
  onDeleteMemory,
  onViewMemory,
  onRefresh
}) => {
  const memoryTypes = [
    { value: '', label: 'All Types' },
    { value: 'fact', label: 'Facts' },
    { value: 'preference', label: 'Preferences' },
    { value: 'context', label: 'Context' },
    { value: 'summary', label: 'Summaries' },
    { value: 'goal', label: 'Goals' },
    { value: 'constraint', label: 'Constraints' },
  ];

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-4 mb-4">
          <div className="flex-1">
            <Input
              placeholder="Search memories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              leftIcon={<Search className="w-4 h-4" />}
            />
          </div>
          <div className="w-48">
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
            >
              {memoryTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>
          <Button
            onClick={onRefresh}
            variant="ghost"
            icon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Memories List */}
      {isLoading ? (
        <LoadingSpinner className="h-32" text="Loading memories..." />
      ) : memories?.length === 0 ? (
        <div className="text-center py-12">
          <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No memories found</h3>
          <p className="text-gray-500">
            {searchQuery || selectedType 
              ? 'Try adjusting your search criteria'
              : 'Start chatting to build up your AI\'s memory!'
            }
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {memories?.map((memory) => (
            <MemoryCard
              key={memory.id}
              memory={memory}
              onView={onViewMemory}
              onDelete={onDeleteMemory}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// Memory Card Component
interface MemoryCardProps {
  memory: Memory;
  onView: (memory: Memory) => void;
  onDelete: (id: string) => void;
}

const MemoryCard: React.FC<MemoryCardProps> = ({ memory, onView, onDelete }) => {
  const getMemoryTypeIcon = (type: string) => {
    switch (type) {
      case 'fact': return <FileText className="w-4 h-4 text-blue-500" />;
      case 'preference': return <Star className="w-4 h-4 text-yellow-500" />;
      case 'context': return <Brain className="w-4 h-4 text-green-500" />;
      case 'summary': return <BarChart3 className="w-4 h-4 text-purple-500" />;
      case 'goal': return <Target className="w-4 h-4 text-red-500" />;
      case 'constraint': return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      default: return <Brain className="w-4 h-4 text-gray-500" />;
    }
  };

  const getMemoryTypeColor = (type: string) => {
    switch (type) {
      case 'fact': return 'bg-blue-100 text-blue-800';
      case 'preference': return 'bg-yellow-100 text-yellow-800';
      case 'context': return 'bg-green-100 text-green-800';
      case 'summary': return 'bg-purple-100 text-purple-800';
      case 'goal': return 'bg-red-100 text-red-800';
      case 'constraint': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 transition-all duration-fast ease-ui hover:shadow-level-2 hover:-translate-y-px hover:border-gray-400 duration-200">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center space-x-2 mb-2">
            {getMemoryTypeIcon(memory.memory_type)}
            <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getMemoryTypeColor(memory.memory_type)}`}>
              {memory.memory_type}
            </span>
            <div className="flex items-center space-x-1 text-sm text-gray-500">
              <Star className="w-3 h-3" />
              <span>{memory.importance_score.toFixed(1)}</span>
            </div>
          </div>

          {/* Content */}
          <p className="text-gray-900 mb-3 line-clamp-3">{memory.content}</p>

          {/* Tags */}
          {memory.tags && memory.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-3">
              {memory.tags.map((tag, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                >
                  <Tag className="w-3 h-3 mr-1" />
                  {tag}
                </span>
              ))}
            </div>
          )}

          {/* Metadata */}
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center space-x-1">
              <Clock className="w-3 h-3" />
              <span>{formatDistanceToNow(new Date(memory.created_at))} ago</span>
            </div>
            <div className="flex items-center space-x-1">
              <Eye className="w-3 h-3" />
              <span>{memory.access_count} views</span>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2 ml-4">
          <Button
            variant="ghost"
            size="sm"
            icon={<Eye className="w-4 h-4" />}
            onClick={() => onView(memory)}
          >
            View
          </Button>
          <Button
            variant="ghost"
            size="sm"
            icon={<Trash2 className="w-4 h-4" />}
            onClick={() => onDelete(memory.id)}
          >
            Delete
          </Button>
        </div>
      </div>
    </div>
  );
};

// Stats Tab Component
interface StatsTabProps {
  stats: MemoryStats;
  isLoading: boolean;
}

const StatsTab: React.FC<StatsTabProps> = ({ stats, isLoading }) => {
  if (isLoading) {
    return <LoadingSpinner className="h-32" text="Loading statistics..." />;
  }

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Memories</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.total_memories || 0}</p>
            </div>
            <Brain className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Recent Memories</p>
              <p className="text-2xl font-bold text-gray-900">{stats?.recent_memories || 0}</p>
              <p className="text-xs text-gray-500">Last 7 days</p>
            </div>
            <Clock className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Most Accessed</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.most_accessed_memories?.[0]?.access_count || 0}
              </p>
            </div>
            <Star className="w-8 h-8 text-yellow-500" />
          </div>
        </div>
      </div>

      {/* Memory Types Distribution */}
      {stats?.memories_by_type && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Memory Types</h3>
          <div className="space-y-3">
            {Object.entries(stats.memories_by_type).map(([type, count]) => (
              <div key={type} className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700 capitalize">{type}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full"
                      style={{ width: `${(count / stats.total_memories) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-600 w-8">{count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Most Accessed Memories */}
      {stats?.most_accessed_memories && stats.most_accessed_memories.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Most Accessed Memories</h3>
          <div className="space-y-3">
            {stats.most_accessed_memories.map((memory) => (
              <div key={memory.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900 truncate">{memory.content}</p>
                  <p className="text-xs text-gray-500 capitalize">{memory.memory_type}</p>
                </div>
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Eye className="w-4 h-4" />
                  <span>{memory.access_count}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Settings Tab Component
const SettingsTab: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedSessionId, setSelectedSessionId] = useState<string>('');
  const [draft, setDraft] = useState<any | null>(null);

  const { data: sessions = [], isLoading: sessionsLoading } = useQuery(
    'memorySettingsChatSessions',
    () => apiClient.getChatSessions(),
    { refetchOnWindowFocus: false }
  );
  const { data: preferences, isLoading: preferencesLoading } = useQuery(
    'memoryPreferences',
    () => apiClient.getMemoryPreferences(),
    { refetchOnWindowFocus: false }
  );

  React.useEffect(() => {
    if (preferences && !draft) {
      setDraft({
        auto_memory_build_enabled: !!preferences.auto_memory_build_enabled,
        auto_memory_build_mode: String(preferences.auto_memory_build_mode || 'per_turn'),
        auto_memory_build_min_messages: Number(preferences.auto_memory_build_min_messages ?? 3),
        auto_memory_build_min_minutes: Number(preferences.auto_memory_build_min_minutes ?? 10),
        max_memories_per_session: Number(preferences.max_memories_per_session ?? 10),
        memory_importance_threshold: Number(preferences.memory_importance_threshold ?? 0.3),
        allow_cross_session_memory: !!preferences.allow_cross_session_memory,
      });
    }
  }, [preferences, draft]);

  React.useEffect(() => {
    if (!selectedSessionId && sessions.length > 0) {
      setSelectedSessionId(String((sessions[0] as any).id));
    }
  }, [selectedSessionId, sessions]);

  const extractMutation = useMutation(
    (sessionId: string) => apiClient.extractMemoriesFromSession(sessionId),
    {
      onSuccess: (res: any) => {
        const added = Array.isArray(res?.memories) ? res.memories.length : 0;
        toast.success(`Memory extraction finished${added ? `: ${added} memories added` : ''}`);
        queryClient.invalidateQueries('memories');
        queryClient.invalidateQueries('memoryStats');
      },
      onError: (e: any) => {
        toast.error(e?.response?.data?.detail || e?.message || 'Memory extraction failed');
      },
    }
  );
  const savePreferencesMutation = useMutation(
    (updates: any) => apiClient.updateMemoryPreferences(updates),
    {
      onSuccess: () => {
        toast.success('Memory settings saved');
        queryClient.invalidateQueries('memoryPreferences');
      },
      onError: (e: any) => {
        toast.error(e?.response?.data?.detail || e?.message || 'Failed to save memory settings');
      },
    }
  );

  const periodicEnabled = !!draft?.auto_memory_build_enabled && draft?.auto_memory_build_mode === 'periodic';

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Memory Settings</h3>
        <p className="text-gray-600">
          Configure how memories are automatically extracted from chat turns.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h4 className="text-base font-semibold text-gray-900 mb-3">Automatic Memory Building</h4>
        {preferencesLoading || !draft ? (
          <div className="text-sm text-gray-600">Loading settings...</div>
        ) : (
          <div className="space-y-4">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={!!draft.auto_memory_build_enabled}
                onChange={(e) => setDraft({ ...draft, auto_memory_build_enabled: e.target.checked })}
              />
              <span className="text-sm text-gray-800">Enable automatic extraction</span>
            </label>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Mode</label>
              <select
                value={String(draft.auto_memory_build_mode || 'per_turn')}
                onChange={(e) => setDraft({ ...draft, auto_memory_build_mode: e.target.value })}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                disabled={!draft.auto_memory_build_enabled}
              >
                <option value="off">Off</option>
                <option value="manual">Manual only</option>
                <option value="per_turn">Every turn</option>
                <option value="periodic">Periodic</option>
              </select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Periodic: min turns</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={Number(draft.auto_memory_build_min_messages ?? 3)}
                  onChange={(e) => setDraft({ ...draft, auto_memory_build_min_messages: Number(e.target.value) || 1 })}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  disabled={!periodicEnabled}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Periodic: min minutes</label>
                <input
                  type="number"
                  min={1}
                  max={1440}
                  value={Number(draft.auto_memory_build_min_minutes ?? 10)}
                  onChange={(e) => setDraft({ ...draft, auto_memory_build_min_minutes: Number(e.target.value) || 1 })}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                  disabled={!periodicEnabled}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Memories loaded per turn</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={Number(draft.max_memories_per_session ?? 10)}
                  onChange={(e) => setDraft({ ...draft, max_memories_per_session: Number(e.target.value) || 1 })}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Importance threshold (0.0-1.0)</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={Number(draft.memory_importance_threshold ?? 0.3)}
                  onChange={(e) => setDraft({ ...draft, memory_importance_threshold: Number(e.target.value) || 0 })}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                />
              </div>
            </div>

            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={!!draft.allow_cross_session_memory}
                onChange={(e) => setDraft({ ...draft, allow_cross_session_memory: e.target.checked })}
              />
              <span className="text-sm text-gray-800">Allow cross-session memory retrieval</span>
            </label>

            <div className="pt-1">
              <Button
                onClick={() => savePreferencesMutation.mutate(draft)}
                disabled={savePreferencesMutation.isLoading}
              >
                {savePreferencesMutation.isLoading ? 'Saving...' : 'Save Settings'}
              </Button>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h4 className="text-base font-semibold text-gray-900 mb-3">Manual Memory Extraction</h4>
        <div className="flex flex-col gap-3 md:flex-row md:items-end">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Chat session
            </label>
            <select
              value={selectedSessionId}
              onChange={(e) => setSelectedSessionId(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
              disabled={sessionsLoading || extractMutation.isLoading}
            >
              <option value="">{sessionsLoading ? 'Loading sessions...' : 'Select session'}</option>
              {sessions.map((s: any) => (
                <option key={String(s.id)} value={String(s.id)}>
                  {String(s.title || `Session ${String(s.id).slice(0, 8)}`)}
                </option>
              ))}
            </select>
          </div>
          <Button
            onClick={() => selectedSessionId && extractMutation.mutate(selectedSessionId)}
            disabled={!selectedSessionId || extractMutation.isLoading}
          >
            {extractMutation.isLoading ? 'Extracting...' : 'Extract Memories'}
          </Button>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          This runs `POST /api/v1/memory/extract/{`{session_id}`}` and updates memory list/statistics.
        </p>
      </div>
    </div>
  );
};

// Memory Details Modal Component
interface MemoryDetailsModalProps {
  memory: Memory;
  onClose: () => void;
}

const MemoryDetailsModal: React.FC<MemoryDetailsModalProps> = ({ memory, onClose }) => {
  const getMemoryTypeColor = (type: string) => {
    switch (type) {
      case 'fact': return 'bg-blue-100 text-blue-800';
      case 'preference': return 'bg-yellow-100 text-yellow-800';
      case 'context': return 'bg-green-100 text-green-800';
      case 'summary': return 'bg-purple-100 text-purple-800';
      case 'goal': return 'bg-red-100 text-red-800';
      case 'constraint': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Memory Details</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-4">
          {/* Type and Importance */}
          <div className="flex items-center space-x-4">
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getMemoryTypeColor(memory.memory_type)}`}>
              {memory.memory_type}
            </span>
            <div className="flex items-center space-x-1 text-sm text-gray-600">
              <Star className="w-4 h-4" />
              <span>Importance: {memory.importance_score.toFixed(1)}</span>
            </div>
          </div>

          {/* Content */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Content</h3>
            <p className="text-gray-900 bg-gray-50 p-3 rounded-lg">{memory.content}</p>
          </div>

          {/* Context */}
          {memory.context && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Context</h3>
              <pre className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg overflow-auto">
                {JSON.stringify(memory.context, null, 2)}
              </pre>
            </div>
          )}

          {/* Tags */}
          {memory.tags && memory.tags.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {memory.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
            <div>
              <span className="font-medium">Created:</span>
              <span className="ml-1">{new Date(memory.created_at).toLocaleString()}</span>
            </div>
            <div>
              <span className="font-medium">Last Accessed:</span>
              <span className="ml-1">{new Date(memory.last_accessed_at).toLocaleString()}</span>
            </div>
            <div>
              <span className="font-medium">Access Count:</span>
              <span className="ml-1">{memory.access_count}</span>
            </div>
            <div>
              <span className="font-medium">Status:</span>
              <span className={`ml-1 ${memory.is_active ? 'text-green-600' : 'text-red-600'}`}>
                {memory.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MemoryPage;
