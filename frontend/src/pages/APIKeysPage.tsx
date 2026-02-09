/**
 * API Keys management page for creating and managing API keys for external tools.
 */

import React, { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import {
  Key,
  Plus,
  Copy,
  Trash2,
  Eye,
  EyeOff,
  Clock,
  Activity,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
} from 'lucide-react';
import toast from 'react-hot-toast';

import { apiClient } from '../services/api';
import { APIKey, APIKeyCreate, APIKeyCreateResponse, APIKeyUsageStats } from '../types';
import Button from '../components/common/Button';

// Available scopes for API keys
const AVAILABLE_SCOPES = [
  { id: 'read', label: 'Read', description: 'Read-only access to documents and search' },
  { id: 'write', label: 'Write', description: 'Create and modify documents' },
  { id: 'chat', label: 'Chat', description: 'Access chat and agent functionality' },
  { id: 'documents', label: 'Documents', description: 'Full document management' },
  { id: 'workflows', label: 'Workflows', description: 'Execute workflows' },
  { id: 'admin', label: 'Admin', description: 'Full administrative access' },
];

const APIKeysPage: React.FC = () => {
  const queryClient = useQueryClient();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedKey, setSelectedKey] = useState<APIKey | null>(null);
  const [newKeySecret, setNewKeySecret] = useState<string | null>(null);

  // Fetch API keys
  const {
    data: keysData,
    isLoading,
    refetch,
  } = useQuery('apiKeys', () => apiClient.listAPIKeys(false), {
    refetchOnWindowFocus: false,
  });

  const apiKeys = keysData?.api_keys || [];

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Key className="w-7 h-7" />
            API Keys
          </h1>
          <p className="text-gray-600 mt-1">
            Create and manage API keys for external tools and integrations
          </p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="w-4 h-4 mr-2" />
          Create API Key
        </Button>
      </div>

      {/* Info banner */}
      <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex gap-3">
          <Shield className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-800">
            <p className="font-medium">Using API Keys</p>
            <p className="mt-1">
              Include your API key in the <code className="bg-blue-100 px-1 rounded">X-API-Key</code> header
              when making requests to the API. API keys provide the same access as your user account,
              filtered by the scopes you select.
            </p>
            <p className="mt-2 font-mono text-xs bg-blue-100 p-2 rounded">
              curl -H "X-API-Key: taic_your_key_here" {window.location.origin}/api/v1/documents/
            </p>
          </div>
        </div>
      </div>

      {/* API Keys List */}
      {isLoading ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <RefreshCw className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading API keys...</p>
        </div>
      ) : apiKeys.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <Key className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No API Keys</h3>
          <p className="text-gray-600 mb-4">
            Create your first API key to integrate external tools with this platform.
          </p>
          <Button onClick={() => setShowCreateModal(true)}>
            <Plus className="w-4 h-4 mr-2" />
            Create API Key
          </Button>
        </div>
      ) : (
        <div className="space-y-4">
          {apiKeys.map((key) => (
            <APIKeyCard
              key={key.id}
              apiKey={key}
              onSelect={() => setSelectedKey(key)}
              onRefresh={() => refetch()}
            />
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <CreateAPIKeyModal
          onClose={() => {
            setShowCreateModal(false);
            setNewKeySecret(null);
          }}
          onCreated={(key, secret) => {
            setNewKeySecret(secret);
            refetch();
          }}
          newKeySecret={newKeySecret}
        />
      )}

      {/* Details Modal */}
      {selectedKey && (
        <APIKeyDetailsModal
          apiKey={selectedKey}
          onClose={() => setSelectedKey(null)}
          onRevoke={() => {
            setSelectedKey(null);
            refetch();
          }}
        />
      )}
    </div>
  );
};

// API Key Card Component
const APIKeyCard: React.FC<{
  apiKey: APIKey;
  onSelect: () => void;
  onRefresh: () => void;
}> = ({ apiKey, onSelect, onRefresh }) => {
  const isExpired = apiKey.expires_at && new Date(apiKey.expires_at) < new Date();
  const isRevoked = !!apiKey.revoked_at;
  const isInactive = !apiKey.is_active || isExpired || isRevoked;

  const revokeMutation = useMutation(() => apiClient.revokeAPIKey(apiKey.id), {
    onSuccess: () => {
      toast.success('API key revoked');
      onRefresh();
    },
    onError: () => {
      toast.error('Failed to revoke API key');
    },
  });

  return (
    <div
      className={`bg-white rounded-lg shadow p-4 border-l-4 ${
        isInactive ? 'border-gray-300 opacity-60' : 'border-green-500'
      }`}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-gray-900">{apiKey.name}</h3>
            {isRevoked ? (
              <span className="px-2 py-0.5 text-xs bg-red-100 text-red-700 rounded">Revoked</span>
            ) : isExpired ? (
              <span className="px-2 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded">Expired</span>
            ) : !apiKey.is_active ? (
              <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded">Inactive</span>
            ) : (
              <span className="px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded">Active</span>
            )}
          </div>

          {apiKey.description && (
            <p className="text-sm text-gray-600 mt-1">{apiKey.description}</p>
          )}

          <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-500">
            <span className="font-mono bg-gray-100 px-2 py-0.5 rounded">{apiKey.key_prefix}...</span>
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Created {new Date(apiKey.created_at).toLocaleDateString()}
            </span>
            {apiKey.last_used_at && (
              <span className="flex items-center gap-1">
                <Activity className="w-3 h-3" />
                Last used {new Date(apiKey.last_used_at).toLocaleDateString()}
              </span>
            )}
            <span>
              {apiKey.usage_count.toLocaleString()} requests
            </span>
          </div>

          {apiKey.scopes && apiKey.scopes.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {apiKey.scopes.map((scope) => (
                <span
                  key={scope}
                  className="px-2 py-0.5 text-xs bg-blue-50 text-blue-700 rounded"
                >
                  {scope}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onSelect}>
            <Eye className="w-4 h-4" />
          </Button>
          {!isRevoked && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                if (window.confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
                  revokeMutation.mutate();
                }
              }}
              disabled={revokeMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 text-red-500" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

// Create API Key Modal
const CreateAPIKeyModal: React.FC<{
  onClose: () => void;
  onCreated: (key: APIKey, secret: string) => void;
  newKeySecret: string | null;
}> = ({ onClose, onCreated, newKeySecret }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);
  const [expiresInDays, setExpiresInDays] = useState<number | ''>('');
  const [rateLimitPerMinute, setRateLimitPerMinute] = useState(60);
  const [rateLimitPerDay, setRateLimitPerDay] = useState(10000);
  const [showSecret, setShowSecret] = useState(true);
  const [copied, setCopied] = useState(false);

  const createMutation = useMutation(
    (data: APIKeyCreate) => apiClient.createAPIKey(data),
    {
      onSuccess: (response: APIKeyCreateResponse) => {
        toast.success('API key created successfully');
        onCreated(response, response.api_key);
      },
      onError: () => {
        toast.error('Failed to create API key');
      },
    }
  );

  const handleCreate = () => {
    if (!name.trim()) {
      toast.error('Please enter a name for the API key');
      return;
    }

    createMutation.mutate({
      name: name.trim(),
      description: description.trim() || undefined,
      scopes: selectedScopes.length > 0 ? selectedScopes : undefined,
      expires_in_days: expiresInDays || undefined,
      rate_limit_per_minute: rateLimitPerMinute,
      rate_limit_per_day: rateLimitPerDay,
    });
  };

  const handleCopy = () => {
    if (newKeySecret) {
      navigator.clipboard.writeText(newKeySecret);
      setCopied(true);
      toast.success('API key copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const toggleScope = (scope: string) => {
    setSelectedScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            {newKeySecret ? 'API Key Created' : 'Create New API Key'}
          </h2>

          {newKeySecret ? (
            // Show the created key
            <div className="space-y-4">
              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex gap-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
                  <div className="text-sm text-yellow-800">
                    <p className="font-medium">Save this API key now!</p>
                    <p>This is the only time you'll see the full key. Store it securely.</p>
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Your API Key
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 relative">
                    <input
                      type={showSecret ? 'text' : 'password'}
                      value={newKeySecret}
                      readOnly
                      className="w-full px-3 py-2 pr-20 border border-gray-300 rounded-lg font-mono text-sm bg-gray-50"
                    />
                    <button
                      onClick={() => setShowSecret(!showSecret)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-500 hover:text-gray-700"
                    >
                      {showSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <Button onClick={handleCopy} variant={copied ? 'primary' : 'secondary'}>
                    {copied ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  </Button>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={onClose}>Done</Button>
              </div>
            </div>
          ) : (
            // Create form
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g., CI/CD Pipeline, Slack Integration"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What is this API key used for?"
                  rows={2}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Scopes (leave empty for full access)
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {AVAILABLE_SCOPES.map((scope) => (
                    <label
                      key={scope.id}
                      className={`flex items-start gap-2 p-2 border rounded cursor-pointer transition-colors ${
                        selectedScopes.includes(scope.id)
                          ? 'border-primary-500 bg-primary-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedScopes.includes(scope.id)}
                        onChange={() => toggleScope(scope.id)}
                        className="mt-0.5"
                      />
                      <div>
                        <span className="text-sm font-medium text-gray-900">{scope.label}</span>
                        <p className="text-xs text-gray-500">{scope.description}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Expiration (days)
                </label>
                <input
                  type="number"
                  value={expiresInDays}
                  onChange={(e) => setExpiresInDays(e.target.value ? parseInt(e.target.value) : '')}
                  placeholder="Leave empty for no expiration"
                  min={1}
                  max={365}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Leave empty for a key that never expires
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Rate Limit (per minute)
                  </label>
                  <input
                    type="number"
                    value={rateLimitPerMinute}
                    onChange={(e) => setRateLimitPerMinute(parseInt(e.target.value) || 60)}
                    min={1}
                    max={1000}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Rate Limit (per day)
                  </label>
                  <input
                    type="number"
                    value={rateLimitPerDay}
                    onChange={(e) => setRateLimitPerDay(parseInt(e.target.value) || 10000)}
                    min={1}
                    max={1000000}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-4">
                <Button variant="ghost" onClick={onClose}>
                  Cancel
                </Button>
                <Button onClick={handleCreate} loading={createMutation.isLoading}>
                  Create API Key
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// API Key Details Modal
const APIKeyDetailsModal: React.FC<{
  apiKey: APIKey;
  onClose: () => void;
  onRevoke: () => void;
}> = ({ apiKey, onClose, onRevoke }) => {
  const { data: usageStats, isLoading: loadingUsage } = useQuery(
    ['apiKeyUsage', apiKey.id],
    () => apiClient.getAPIKeyUsage(apiKey.id, 30),
    { enabled: !!apiKey.id }
  );

  const revokeMutation = useMutation(() => apiClient.revokeAPIKey(apiKey.id), {
    onSuccess: () => {
      toast.success('API key revoked');
      onRevoke();
    },
    onError: () => {
      toast.error('Failed to revoke API key');
    },
  });

  const isExpired = apiKey.expires_at && new Date(apiKey.expires_at) < new Date();
  const isRevoked = !!apiKey.revoked_at;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-start mb-4">
            <h2 className="text-xl font-semibold text-gray-900">{apiKey.name}</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
              <XCircle className="w-6 h-6" />
            </button>
          </div>

          <div className="space-y-4">
            {/* Status */}
            <div className="flex items-center gap-2">
              {isRevoked ? (
                <>
                  <XCircle className="w-5 h-5 text-red-500" />
                  <span className="text-red-700 font-medium">Revoked</span>
                </>
              ) : isExpired ? (
                <>
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-yellow-700 font-medium">Expired</span>
                </>
              ) : apiKey.is_active ? (
                <>
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-green-700 font-medium">Active</span>
                </>
              ) : (
                <>
                  <XCircle className="w-5 h-5 text-gray-500" />
                  <span className="text-gray-700 font-medium">Inactive</span>
                </>
              )}
            </div>

            {/* Key prefix */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Key Prefix</label>
              <code className="block px-3 py-2 bg-gray-100 rounded font-mono text-sm">
                {apiKey.key_prefix}...
              </code>
            </div>

            {/* Description */}
            {apiKey.description && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <p className="text-gray-600">{apiKey.description}</p>
              </div>
            )}

            {/* Scopes */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Scopes</label>
              {apiKey.scopes && apiKey.scopes.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {apiKey.scopes.map((scope) => (
                    <span
                      key={scope}
                      className="px-2 py-1 text-sm bg-blue-50 text-blue-700 rounded"
                    >
                      {scope}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Full access (no scope restrictions)</p>
              )}
            </div>

            {/* Rate limits */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Rate Limit/min</label>
                <p className="text-gray-900">{apiKey.rate_limit_per_minute}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Rate Limit/day</label>
                <p className="text-gray-900">{apiKey.rate_limit_per_day.toLocaleString()}</p>
              </div>
            </div>

            {/* Dates */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <label className="block font-medium text-gray-700 mb-1">Created</label>
                <p className="text-gray-600">{new Date(apiKey.created_at).toLocaleString()}</p>
              </div>
              {apiKey.expires_at && (
                <div>
                  <label className="block font-medium text-gray-700 mb-1">Expires</label>
                  <p className={isExpired ? 'text-red-600' : 'text-gray-600'}>
                    {new Date(apiKey.expires_at).toLocaleString()}
                  </p>
                </div>
              )}
              {apiKey.last_used_at && (
                <div>
                  <label className="block font-medium text-gray-700 mb-1">Last Used</label>
                  <p className="text-gray-600">{new Date(apiKey.last_used_at).toLocaleString()}</p>
                </div>
              )}
              {apiKey.last_used_ip && (
                <div>
                  <label className="block font-medium text-gray-700 mb-1">Last IP</label>
                  <p className="text-gray-600 font-mono text-xs">{apiKey.last_used_ip}</p>
                </div>
              )}
            </div>

            {/* Usage stats */}
            <div className="border-t pt-4">
              <h3 className="font-medium text-gray-900 mb-2">Usage (Last 30 days)</h3>
              {loadingUsage ? (
                <p className="text-gray-500">Loading usage stats...</p>
              ) : usageStats ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Total Requests:</span>
                      <span className="ml-2 font-medium">{usageStats.total_requests.toLocaleString()}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Lifetime:</span>
                      <span className="ml-2 font-medium">{usageStats.lifetime_requests.toLocaleString()}</span>
                    </div>
                  </div>
                  {usageStats.top_endpoints.length > 0 && (
                    <div>
                      <p className="text-sm text-gray-600 mb-1">Top Endpoints:</p>
                      <div className="space-y-1">
                        {usageStats.top_endpoints.slice(0, 5).map((ep, i) => (
                          <div key={i} className="flex justify-between text-xs">
                            <span className="font-mono text-gray-700 truncate">{ep.endpoint}</span>
                            <span className="text-gray-500 ml-2">{ep.count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-gray-500">No usage data available</p>
              )}
            </div>

            {/* Actions */}
            {!isRevoked && (
              <div className="flex justify-end gap-2 pt-4 border-t">
                <Button variant="ghost" onClick={onClose}>
                  Close
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    if (window.confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
                      revokeMutation.mutate();
                    }
                  }}
                  loading={revokeMutation.isLoading}
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Revoke Key
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default APIKeysPage;
