/**
 * Admin dashboard for system management and monitoring
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { 
  Activity, 
  Database, 
  Settings, 
  Users, 
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Play,
  Trash2,
  Download,
  BarChart3,
  Server,
  HardDrive,
  Cpu,
  Globe
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import { apiClient } from '../services/api';
import { SystemHealth, SystemStats, DocumentSource } from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';
import toast from 'react-hot-toast';

const AdminPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const queryClient = useQueryClient();

  const tabs = [
    { id: 'overview', name: 'Overview', icon: BarChart3 },
    { id: 'health', name: 'System Health', icon: Activity },
    { id: 'sources', name: 'Data Sources', icon: Database },
    { id: 'tasks', name: 'Background Tasks', icon: Settings },
    { id: 'logs', name: 'System Logs', icon: Server },
  ];

  // Fetch system health
  const { data: health, isLoading: healthLoading, refetch: refetchHealth } = useQuery(
    'systemHealth',
    apiClient.getSystemHealth,
    {
      refetchInterval: 30000, // Refresh every 30 seconds
      refetchOnWindowFocus: false,
    }
  );

  // Fetch system stats
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery(
    'systemStats',
    apiClient.getSystemStats,
    {
      refetchInterval: 60000, // Refresh every minute
      refetchOnWindowFocus: false,
    }
  );

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Administration Dashboard</h1>
        <p className="text-gray-600">Monitor and manage your Knowledge Database system</p>
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
          {activeTab === 'overview' && <OverviewTab health={health} stats={stats} />}
          {activeTab === 'health' && <HealthTab health={health} onRefresh={refetchHealth} />}
          {activeTab === 'sources' && <DataSourcesTab />}
          {activeTab === 'tasks' && <TasksTab />}
          {activeTab === 'logs' && <LogsTab />}
        </div>
      </div>
    </div>
  );
};

// Overview Tab
interface OverviewTabProps {
  health?: SystemHealth;
  stats?: SystemStats;
}

const OverviewTab: React.FC<OverviewTabProps> = ({ health, stats }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'degraded': return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'unhealthy': return <XCircle className="w-5 h-5 text-red-500" />;
      default: return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">System Status</p>
              <p className={`text-2xl font-bold ${getStatusColor(health?.overall_status || 'unknown')}`}>
                {health?.overall_status || 'Unknown'}
              </p>
            </div>
            {getStatusIcon(health?.overall_status || 'unknown')}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Documents</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.documents?.total || 0}
              </p>
            </div>
            <Database className="w-8 h-8 text-blue-500" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-500">
              {stats?.documents?.processed || 0} processed
            </span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Chat Sessions</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.chat?.total_sessions || 0}
              </p>
            </div>
            <Users className="w-8 h-8 text-green-500" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-500">
              {stats?.chat?.active_sessions_24h || 0} active today
            </span>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Vector Chunks</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.vector_store?.total_chunks || 0}
              </p>
            </div>
            <HardDrive className="w-8 h-8 text-purple-500" />
          </div>
          <div className="mt-2">
            <span className="text-sm text-gray-500">
              {stats?.vector_store?.embedding_model || 'Unknown model'}
            </span>
          </div>
        </div>
      </div>

      {/* Services Status */}
      {health && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Services Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(health.services).map(([service, serviceHealth]) => (
              <div key={service} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 capitalize">
                    {service.replace('_', ' ')}
                  </h4>
                  {getStatusIcon(serviceHealth.status)}
                </div>
                <p className="text-sm text-gray-600">
                  {serviceHealth.message || serviceHealth.error || 'No additional information'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity */}
      {stats?.processing && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Processing (Last 7 Days)</h3>
          <div className="space-y-2">
            {stats.processing.documents_last_7_days.map((day) => (
              <div key={day.date} className="flex justify-between items-center">
                <span className="text-sm text-gray-600">{day.date}</span>
                <span className="text-sm font-medium text-gray-900">{day.count} documents</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Health Tab
interface HealthTabProps {
  health?: SystemHealth;
  onRefresh: () => void;
}

const HealthTab: React.FC<HealthTabProps> = ({ health, onRefresh }) => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">System Health</h2>
        <Button onClick={onRefresh} icon={<RefreshCw className="w-4 h-4" />}>
          Refresh
        </Button>
      </div>

      {health ? (
        <div className="space-y-4">
          {Object.entries(health.services).map(([service, serviceHealth]) => (
            <div key={service} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900 capitalize">
                  {service.replace('_', ' ')}
                </h3>
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  serviceHealth.status === 'healthy' 
                    ? 'bg-green-100 text-green-800'
                    : serviceHealth.status === 'degraded'
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
                }`}>
                  {serviceHealth.status}
                </span>
              </div>
              
              {serviceHealth.message && (
                <p className="text-gray-600 mb-2">{serviceHealth.message}</p>
              )}
              
              {serviceHealth.error && (
                <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                  <strong>Error:</strong> {serviceHealth.error}
                </div>
              )}

              {/* Additional service-specific info */}
              {service === 'disk_space' && serviceHealth.status !== 'unknown' && (
                <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Total:</span>
                    <span className="ml-1">{(serviceHealth as any).total_gb} GB</span>
                  </div>
                  <div>
                    <span className="font-medium">Used:</span>
                    <span className="ml-1">{(serviceHealth as any).used_gb} GB</span>
                  </div>
                  <div>
                    <span className="font-medium">Usage:</span>
                    <span className="ml-1">{(serviceHealth as any).usage_percent}%</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <LoadingSpinner className="h-32" text="Loading health status..." />
      )}
    </div>
  );
};

// Data Sources Tab
const DataSourcesTab: React.FC = () => {
  const queryClient = useQueryClient();

  const { data: sources, isLoading } = useQuery(
    'documentSources',
    apiClient.getDocumentSources,
    {
      refetchOnWindowFocus: false,
    }
  );

  const syncAllMutation = useMutation(
    apiClient.triggerFullSync,
    {
      onSuccess: () => {
        toast.success('Full synchronization started');
      },
      onError: () => {
        toast.error('Failed to start synchronization');
      },
    }
  );

  const syncSourceMutation = useMutation(
    apiClient.triggerSourceSync,
    {
      onSuccess: () => {
        toast.success('Source synchronization started');
      },
      onError: () => {
        toast.error('Failed to start source synchronization');
      },
    }
  );

  const getSourceTypeIcon = (type: string) => {
    switch (type) {
      case 'gitlab': return <Database className="w-5 h-5 text-orange-500" />;
      case 'confluence': return <Globe className="w-5 h-5 text-blue-500" />;
      case 'web': return <Globe className="w-5 h-5 text-green-500" />;
      case 'file': return <HardDrive className="w-5 h-5 text-gray-500" />;
      default: return <Database className="w-5 h-5 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">Data Sources</h2>
        <Button 
          onClick={() => syncAllMutation.mutate()}
          loading={syncAllMutation.isLoading}
          icon={<Play className="w-4 h-4" />}
        >
          Sync All Sources
        </Button>
      </div>

      {isLoading ? (
        <LoadingSpinner className="h-32" text="Loading data sources..." />
      ) : sources?.length === 0 ? (
        <div className="text-center py-12">
          <Database className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Data Sources</h3>
          <p className="text-gray-500">Configure data sources to start ingesting content</p>
        </div>
      ) : (
        <div className="grid gap-6">
          {sources?.map((source) => (
            <div key={source.id} className="bg-white rounded-lg shadow p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  {getSourceTypeIcon(source.source_type)}
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">{source.name}</h3>
                    <p className="text-sm text-gray-500 capitalize">{source.source_type} Source</p>
                    <div className="mt-2 space-y-1 text-sm text-gray-600">
                      <div>
                        <span className="font-medium">Status:</span>
                        <span className={`ml-1 ${source.is_active ? 'text-green-600' : 'text-red-600'}`}>
                          {source.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Last Sync:</span>
                        <span className="ml-1">
                          {source.last_sync 
                            ? formatDistanceToNow(new Date(source.last_sync)) + ' ago'
                            : 'Never'
                          }
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Created:</span>
                        <span className="ml-1">
                          {formatDistanceToNow(new Date(source.created_at))} ago
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => syncSourceMutation.mutate(source.id)}
                    loading={syncSourceMutation.isLoading}
                    icon={<Play className="w-4 h-4" />}
                  >
                    Sync
                  </Button>
                </div>
              </div>

              {/* Configuration Preview */}
              <div className="mt-4 p-3 bg-gray-50 rounded border">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Configuration</h4>
                <pre className="text-xs text-gray-600 overflow-auto">
                  {JSON.stringify(source.config, null, 2)}
                </pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Tasks Tab
const TasksTab: React.FC = () => {
  const { data: taskStatus, isLoading } = useQuery(
    'taskStatus',
    apiClient.getTaskStatus,
    {
      refetchInterval: 5000, // Refresh every 5 seconds
      refetchOnWindowFocus: false,
    }
  );

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-gray-900">Background Tasks</h2>

      {isLoading ? (
        <LoadingSpinner className="h-32" text="Loading task status..." />
      ) : (
        <div className="space-y-4">
          {/* Active Tasks */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Active Tasks</h3>
            {taskStatus?.active_tasks && Object.keys(taskStatus.active_tasks).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(taskStatus.active_tasks).map(([worker, tasks]) => (
                  <div key={worker}>
                    <h4 className="font-medium text-gray-700">{worker}</h4>
                    {Array.isArray(tasks) && tasks.map((task: any, index: number) => (
                      <div key={index} className="ml-4 text-sm text-gray-600">
                        {task.name} - {task.id}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No active tasks</p>
            )}
          </div>

          {/* Scheduled Tasks */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Scheduled Tasks</h3>
            {taskStatus?.scheduled_tasks && Object.keys(taskStatus.scheduled_tasks).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(taskStatus.scheduled_tasks).map(([worker, tasks]) => (
                  <div key={worker}>
                    <h4 className="font-medium text-gray-700">{worker}</h4>
                    {Array.isArray(tasks) && tasks.map((task: any, index: number) => (
                      <div key={index} className="ml-4 text-sm text-gray-600">
                        {task.name} - {task.id}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No scheduled tasks</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Logs Tab
const LogsTab: React.FC = () => {
  const [lines, setLines] = useState(100);

  const { data: logs, isLoading, refetch } = useQuery(
    ['systemLogs', lines],
    () => apiClient.getSystemLogs(lines),
    {
      refetchOnWindowFocus: false,
    }
  );

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-900">System Logs</h2>
        <div className="flex items-center space-x-4">
          <select
            value={lines}
            onChange={(e) => setLines(Number(e.target.value))}
            className="rounded border-gray-300 text-sm"
          >
            <option value={50}>Last 50 lines</option>
            <option value={100}>Last 100 lines</option>
            <option value={500}>Last 500 lines</option>
            <option value={1000}>Last 1000 lines</option>
          </select>
          <Button onClick={() => refetch()} icon={<RefreshCw className="w-4 h-4" />}>
            Refresh
          </Button>
        </div>
      </div>

      {isLoading ? (
        <LoadingSpinner className="h-32" text="Loading logs..." />
      ) : (
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b border-gray-200">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Showing {logs?.returned_lines || 0} of {logs?.total_lines || 0} lines</span>
            </div>
          </div>
          <div className="p-4">
            <pre className="text-xs text-gray-800 overflow-auto max-h-96 bg-gray-50 p-4 rounded border font-mono">
              {logs?.logs?.join('\n') || 'No logs available'}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminPage;







