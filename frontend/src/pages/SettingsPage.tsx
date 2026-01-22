/**
 * Settings page for user preferences and account management
 */

import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { useMutation, useQuery } from 'react-query';
import { User, Lock, Bell, Palette, Shield, Bot } from 'lucide-react';

import { useAuth } from '../contexts/AuthContext';
import { useNotifications } from '../contexts/NotificationContext';
import { apiClient } from '../services/api';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import toast from 'react-hot-toast';

interface PasswordChangeForm {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

const SettingsPage: React.FC = () => {
  const { user, updateUser } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');

  const tabs = [
    { id: 'profile', name: 'Profile', icon: User },
    { id: 'security', name: 'Security', icon: Lock },
    { id: 'llm', name: 'LLM Settings', icon: Bot },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'appearance', name: 'Appearance', icon: Palette },
  ];

  // Add admin tab if user is admin
  if (user?.role === 'admin') {
    tabs.push({ id: 'admin', name: 'Administration', icon: Shield });
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Manage your account preferences and settings</p>
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
          {activeTab === 'profile' && <ProfileTab />}
          {activeTab === 'security' && <SecurityTab />}
          {activeTab === 'llm' && <LLMSettingsTab />}
          {activeTab === 'notifications' && <NotificationsTab />}
          {activeTab === 'appearance' && <AppearanceTab />}
          {activeTab === 'admin' && user?.role === 'admin' && <AdminTab />}
        </div>
      </div>
    </div>
  );
};

// Profile Tab
const ProfileTab: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Profile Information</h2>
      
      <div className="space-y-6">
        {/* Avatar */}
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-primary-600 rounded-full flex items-center justify-center">
            <User className="w-8 h-8 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900">Profile Picture</h3>
            <p className="text-sm text-gray-500">Upload a profile picture to personalize your account</p>
            <Button variant="ghost" size="sm" className="mt-2">
              Change Picture
            </Button>
          </div>
        </div>

        {/* User Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Input
            label="Username"
            value={user?.username || ''}
            disabled
            helpText="Username cannot be changed"
          />
          
          <Input
            label="Email"
            value={user?.email || ''}
            disabled
            helpText="Contact admin to change email"
          />
          
          <Input
            label="Full Name"
            value={user?.full_name || ''}
            placeholder="Enter your full name"
          />
          
          <Input
            label="Role"
            value={user?.role || ''}
            disabled
            helpText="Role is assigned by administrators"
          />
        </div>

        {/* Account Status */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Account Status</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Account Status:</span>
              <span className={`font-medium ${user?.is_active ? 'text-green-600' : 'text-red-600'}`}>
                {user?.is_active ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Verification Status:</span>
              <span className={`font-medium ${user?.is_verified ? 'text-green-600' : 'text-yellow-600'}`}>
                {user?.is_verified ? 'Verified' : 'Pending'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Member Since:</span>
              <span className="text-gray-600">
                {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <Button>Save Changes</Button>
        </div>
      </div>
    </div>
  );
};

// Security Tab
const SecurityTab: React.FC = () => {
  const { user } = useAuth();
  const { register, handleSubmit, formState: { errors }, watch, reset } = useForm<PasswordChangeForm>();

  const changePasswordMutation = useMutation(
    async (data: { current_password: string; new_password: string }) => {
      // This endpoint doesn't exist yet, but the structure is ready
      const response = await fetch('/api/v1/users/me/password', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        throw new Error('Failed to change password');
      }
      
      return response.json();
    },
    {
      onSuccess: () => {
        toast.success('Password changed successfully');
        reset();
      },
      onError: () => {
        toast.error('Failed to change password');
      },
    }
  );

  const onSubmit = (data: PasswordChangeForm) => {
    if (data.newPassword !== data.confirmPassword) {
      toast.error('Passwords do not match');
      return;
    }

    changePasswordMutation.mutate({
      current_password: data.currentPassword,
      new_password: data.newPassword,
    });
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Security Settings</h2>
      
      <div className="space-y-8">
        {/* Password Change */}
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Change Password</h3>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 max-w-md">
            <Input
              label="Current Password"
              type="password"
              error={errors.currentPassword?.message}
              {...register('currentPassword', {
                required: 'Current password is required'
              })}
            />
            
            <Input
              label="New Password"
              type="password"
              error={errors.newPassword?.message}
              {...register('newPassword', {
                required: 'New password is required',
                minLength: {
                  value: 6,
                  message: 'Password must be at least 6 characters'
                }
              })}
            />
            
            <Input
              label="Confirm New Password"
              type="password"
              error={errors.confirmPassword?.message}
              {...register('confirmPassword', {
                required: 'Please confirm your password'
              })}
            />
            
            <Button
              type="submit"
              loading={changePasswordMutation.isLoading}
            >
              Change Password
            </Button>
          </form>
        </div>

        {/* Login History */}
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Account Activity</h3>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Last Login:</span>
                <span className="text-gray-600">
                  {user?.last_login ? new Date(user.last_login).toLocaleString() : 'Never'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Total Logins:</span>
                <span className="text-gray-600">{user?.login_count || 0}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Task types for per-task model configuration
const LLM_TASK_TYPES = [
  { id: 'chat', label: 'Chat', description: 'Main conversation model' },
  { id: 'title_generation', label: 'Title Generation', description: 'Chat session titles' },
  { id: 'summarization', label: 'Summarization', description: 'Document summaries' },
  { id: 'query_expansion', label: 'Query Expansion', description: 'Search query variations' },
  { id: 'memory_extraction', label: 'Memory Extraction', description: 'Conversation memories' },
  { id: 'workflow_synthesis', label: 'Workflow Synthesis', description: 'Generate workflows from descriptions' },
];

// LLM Presets for quick configuration
interface LLMPreset {
  id: string;
  name: string;
  provider: string;
  model: string;
  apiUrl: string;
  description: string;
  requiresApiKey: boolean;
}

const LLM_PRESETS: LLMPreset[] = [
  {
    id: 'deepseek_official',
    name: 'DeepSeek Official',
    provider: 'deepseek',
    model: 'deepseek-chat',
    apiUrl: 'https://api.deepseek.com/v1',
    description: 'Official DeepSeek API - fast and affordable',
    requiresApiKey: true,
  },
  {
    id: 'deepseek_reasoner',
    name: 'DeepSeek Reasoner',
    provider: 'deepseek',
    model: 'deepseek-reasoner',
    apiUrl: 'https://api.deepseek.com/v1',
    description: 'DeepSeek R1 reasoning model for complex tasks',
    requiresApiKey: true,
  },
  {
    id: 'openai_gpt4',
    name: 'OpenAI GPT-4o',
    provider: 'openai',
    model: 'gpt-4o',
    apiUrl: 'https://api.openai.com/v1',
    description: 'OpenAI GPT-4o - powerful multimodal model',
    requiresApiKey: true,
  },
  {
    id: 'openai_gpt4_mini',
    name: 'OpenAI GPT-4o Mini',
    provider: 'openai',
    model: 'gpt-4o-mini',
    apiUrl: 'https://api.openai.com/v1',
    description: 'OpenAI GPT-4o Mini - fast and cost-effective',
    requiresApiKey: true,
  },
  {
    id: 'ollama_llama3',
    name: 'Ollama Llama 3.2',
    provider: 'ollama',
    model: 'llama3.2:3b',
    apiUrl: 'http://ollama:11434',
    description: 'Local Llama 3.2 3B via Ollama - no API key needed',
    requiresApiKey: false,
  },
  {
    id: 'ollama_mistral',
    name: 'Ollama Mistral 7B',
    provider: 'ollama',
    model: 'mistral:7b',
    apiUrl: 'http://ollama:11434',
    description: 'Local Mistral 7B via Ollama - no API key needed',
    requiresApiKey: false,
  },
  {
    id: 'ollama_qwen',
    name: 'Ollama Qwen 2.5',
    provider: 'ollama',
    model: 'qwen2.5:7b',
    apiUrl: 'http://ollama:11434',
    description: 'Local Qwen 2.5 7B via Ollama - multilingual',
    requiresApiKey: false,
  },
];

// LLM Settings Tab
const LLMSettingsTab: React.FC = () => {
  const [provider, setProvider] = useState<string>('');
  const [model, setModel] = useState<string>('');
  const [apiUrl, setApiUrl] = useState<string>('');
  const [apiKey, setApiKey] = useState<string>('');
  const [temperature, setTemperature] = useState<string>('');
  const [maxTokens, setMaxTokens] = useState<string>('');
  const [taskModels, setTaskModels] = useState<Record<string, string>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  // Handle preset selection - auto-populate fields
  const handlePresetSelect = (presetId: string) => {
    setSelectedPreset(presetId);
    if (!presetId) return;

    const preset = LLM_PRESETS.find(p => p.id === presetId);
    if (preset) {
      setProvider(preset.provider);
      setModel(preset.model);
      setApiUrl(preset.apiUrl);
      // Don't clear API key - user may have already entered it
      toast.success(`Applied "${preset.name}" preset. ${preset.requiresApiKey ? 'Remember to enter your API key.' : ''}`);
    }
  };

  // Fetch current settings
  const { data: settings, isLoading, refetch } = useQuery(
    'userLLMSettings',
    () => apiClient.getUserLLMSettings(),
    {
      onSuccess: (data) => {
        setProvider(data.llm_provider || '');
        setModel(data.llm_model || '');
        setApiUrl(data.llm_api_url || '');
        setTemperature(data.llm_temperature !== null ? String(data.llm_temperature) : '');
        setMaxTokens(data.llm_max_tokens !== null ? String(data.llm_max_tokens) : '');
        setTaskModels(data.llm_task_models || {});
        // Don't overwrite apiKey - just show indicator
        // Show advanced section if task models are configured
        if (data.llm_task_models && Object.keys(data.llm_task_models).length > 0) {
          setShowAdvanced(true);
        }
      },
    }
  );

  const updateMutation = useMutation(
    (data: Parameters<typeof apiClient.updateUserLLMSettings>[0]) =>
      apiClient.updateUserLLMSettings(data),
    {
      onSuccess: () => {
        toast.success('LLM settings updated');
        refetch();
      },
      onError: () => {
        toast.error('Failed to update LLM settings');
      },
    }
  );

  const clearMutation = useMutation(() => apiClient.clearUserLLMSettings(), {
    onSuccess: () => {
      setProvider('');
      setModel('');
      setApiUrl('');
      setApiKey('');
      setTemperature('');
      setMaxTokens('');
      setTaskModels({});
      setSelectedPreset('');
      toast.success('LLM settings cleared - using system defaults');
      refetch();
    },
    onError: () => {
      toast.error('Failed to clear LLM settings');
    },
  });

  const handleSave = () => {
    // Filter out empty task models
    const filteredTaskModels = Object.fromEntries(
      Object.entries(taskModels).filter(([_, v]) => v && v.trim())
    );

    updateMutation.mutate({
      llm_provider: provider || null,
      llm_model: model || null,
      llm_api_url: apiUrl || null,
      llm_api_key: apiKey || undefined, // Only send if provided
      llm_temperature: temperature ? parseFloat(temperature) : null,
      llm_max_tokens: maxTokens ? parseInt(maxTokens, 10) : null,
      llm_task_models: Object.keys(filteredTaskModels).length > 0 ? filteredTaskModels : null,
    });
  };

  const handleTaskModelChange = (taskId: string, value: string) => {
    setTaskModels(prev => ({ ...prev, [taskId]: value }));
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/4"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-2">LLM Settings</h2>
      <p className="text-sm text-gray-500 mb-6">
        Configure your personal LLM preferences. Leave fields empty to use system defaults.
      </p>

      <div className="space-y-6">
        {/* Quick Presets */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Quick Presets
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {LLM_PRESETS.map((preset) => (
              <button
                key={preset.id}
                onClick={() => handlePresetSelect(preset.id)}
                className={`p-3 text-left border rounded-lg transition-all hover:shadow-md ${
                  selectedPreset === preset.id
                    ? 'border-primary-500 bg-primary-50 ring-2 ring-primary-500'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm text-gray-900">{preset.name}</span>
                  {preset.requiresApiKey && (
                    <span className="text-xs px-1.5 py-0.5 bg-amber-100 text-amber-700 rounded">
                      API Key
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500">{preset.description}</p>
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Click a preset to auto-fill settings, or configure manually below
          </p>
        </div>

        <hr className="border-gray-200" />

        {/* Provider Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Provider
          </label>
          <select
            value={provider}
            onChange={(e) => {
              setProvider(e.target.value);
              setSelectedPreset(''); // Clear preset selection on manual change
            }}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="">Use system default</option>
            <option value="ollama">Ollama (Local)</option>
            <option value="deepseek">DeepSeek</option>
            <option value="openai">OpenAI</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Select your preferred LLM provider
          </p>
        </div>

        {/* Model Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Model Name
          </label>
          <input
            type="text"
            value={model}
            onChange={(e) => {
              setModel(e.target.value);
              setSelectedPreset('');
            }}
            placeholder="e.g., gpt-4, llama3.2:1b, deepseek-chat"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Specific model to use (leave empty for provider default)
          </p>
        </div>

        {/* API URL */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            API URL Override
          </label>
          <input
            type="url"
            value={apiUrl}
            onChange={(e) => {
              setApiUrl(e.target.value);
              setSelectedPreset('');
            }}
            placeholder="https://api.example.com/v1"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Custom API endpoint (for self-hosted or alternative providers)
          </p>
        </div>

        {/* API Key */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            API Key
            {settings?.llm_api_key_set && (
              <span className="ml-2 text-xs text-green-600">(Key is set)</span>
            )}
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={settings?.llm_api_key_set ? '••••••••' : 'Enter API key'}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Your personal API key for external providers
          </p>
        </div>

        {/* Advanced Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Temperature
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={temperature}
              onChange={(e) => setTemperature(e.target.value)}
              placeholder="0.7"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <p className="text-xs text-gray-500 mt-1">0.0 - 2.0</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Tokens
            </label>
            <input
              type="number"
              min="1"
              max="32000"
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value)}
              placeholder="1000"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
            <p className="text-xs text-gray-500 mt-1">Max response length</p>
          </div>
        </div>

        {/* Advanced Settings - Per-Task Model Configuration */}
        <div className="border rounded-lg">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50 rounded-lg"
          >
            <div>
              <span className="font-medium text-gray-900">Advanced: Per-Task Model Configuration</span>
              <p className="text-xs text-gray-500 mt-0.5">
                Configure different models for specific tasks
              </p>
            </div>
            <svg
              className={`w-5 h-5 text-gray-500 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showAdvanced && (
            <div className="px-4 pb-4 border-t space-y-4">
              <p className="text-xs text-gray-500 pt-3">
                Leave fields empty to use the default model (configured above)
              </p>
              {LLM_TASK_TYPES.map((task) => (
                <div key={task.id}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {task.label}
                  </label>
                  <input
                    type="text"
                    value={taskModels[task.id] || ''}
                    onChange={(e) => handleTaskModelChange(task.id, e.target.value)}
                    placeholder={`e.g., gpt-4-turbo, llama3.2:1b (${task.description})`}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between pt-4 border-t">
          <Button
            variant="ghost"
            onClick={() => clearMutation.mutate()}
            disabled={clearMutation.isLoading}
          >
            Reset to Defaults
          </Button>
          <Button
            onClick={handleSave}
            disabled={updateMutation.isLoading}
          >
            {updateMutation.isLoading ? 'Saving...' : 'Save Settings'}
          </Button>
        </div>
      </div>
    </div>
  );
};

// Notifications Tab
const NotificationsTab: React.FC = () => {
  const { preferences, updatePreferences, isLoading } = useNotifications();
  const [localPrefs, setLocalPrefs] = useState<Record<string, boolean>>({});
  const [isSaving, setIsSaving] = useState(false);

  // Sync local state with preferences from context
  useEffect(() => {
    if (preferences) {
      setLocalPrefs({
        notify_document_processing: preferences.notify_document_processing,
        notify_document_errors: preferences.notify_document_errors,
        notify_sync_complete: preferences.notify_sync_complete,
        notify_ingestion_complete: preferences.notify_ingestion_complete,
        notify_transcription_complete: preferences.notify_transcription_complete,
        notify_summarization_complete: preferences.notify_summarization_complete,
        notify_maintenance: preferences.notify_maintenance,
        notify_quota_warnings: preferences.notify_quota_warnings,
        notify_admin_broadcasts: preferences.notify_admin_broadcasts,
        notify_mentions: preferences.notify_mentions,
        notify_shares: preferences.notify_shares,
        notify_comments: preferences.notify_comments,
        play_sound: preferences.play_sound,
        show_desktop_notification: preferences.show_desktop_notification,
      });
    }
  }, [preferences]);

  const handleToggle = (key: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalPrefs(prev => ({ ...prev, [key]: e.target.checked }));
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await updatePreferences(localPrefs);
    } finally {
      setIsSaving(false);
    }
  };

  const notificationGroups = [
    {
      title: 'Document Notifications',
      items: [
        { key: 'notify_document_processing', label: 'Document Processing', description: 'Notify when document processing is complete' },
        { key: 'notify_document_errors', label: 'Processing Errors', description: 'Notify when document processing fails' },
        { key: 'notify_ingestion_complete', label: 'Ingestion Complete', description: 'Notify when document ingestion is complete' },
        { key: 'notify_transcription_complete', label: 'Transcription Complete', description: 'Notify when video/audio transcription is complete' },
        { key: 'notify_summarization_complete', label: 'Summarization Complete', description: 'Notify when document summarization is complete' },
        { key: 'notify_sync_complete', label: 'Source Sync', description: 'Notify when data source sync is complete' },
      ],
    },
    {
      title: 'System Notifications',
      items: [
        { key: 'notify_maintenance', label: 'System Maintenance', description: 'Receive notifications about system maintenance' },
        { key: 'notify_quota_warnings', label: 'Quota Warnings', description: 'Notify when approaching storage or usage limits' },
        { key: 'notify_admin_broadcasts', label: 'Admin Broadcasts', description: 'Receive important announcements from administrators' },
      ],
    },
    {
      title: 'Collaboration Notifications',
      items: [
        { key: 'notify_mentions', label: 'Mentions', description: 'Notify when someone mentions you' },
        { key: 'notify_shares', label: 'Shares', description: 'Notify when someone shares content with you' },
        { key: 'notify_comments', label: 'Comments', description: 'Notify when someone comments on your content' },
      ],
    },
    {
      title: 'Display Settings',
      items: [
        { key: 'play_sound', label: 'Play Sound', description: 'Play a sound for new notifications' },
        { key: 'show_desktop_notification', label: 'Desktop Notifications', description: 'Show browser desktop notifications' },
      ],
    },
  ];

  if (isLoading && !preferences) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/4"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
          <div className="h-10 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Notification Preferences</h2>

      <div className="space-y-8">
        {notificationGroups.map((group) => (
          <div key={group.title}>
            <h3 className="text-md font-medium text-gray-900 mb-4">{group.title}</h3>
            <div className="space-y-4">
              {group.items.map((item) => (
                <label key={item.key} className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    className="mt-1 h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    checked={localPrefs[item.key] ?? false}
                    onChange={handleToggle(item.key)}
                  />
                  <div>
                    <div className="font-medium text-gray-900">{item.label}</div>
                    <div className="text-sm text-gray-500">{item.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        ))}

        <div className="flex justify-end pt-4 border-t">
          <Button onClick={handleSave} loading={isSaving}>
            Save Preferences
          </Button>
        </div>
      </div>
    </div>
  );
};

// Appearance Tab
const AppearanceTab: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Appearance Settings</h2>
      
      <div className="space-y-6">
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Theme</h3>
          <div className="space-y-2">
            <label className="flex items-center space-x-3">
              <input type="radio" name="theme" value="light" defaultChecked />
              <span>Light Theme</span>
            </label>
            <label className="flex items-center space-x-3">
              <input type="radio" name="theme" value="dark" />
              <span>Dark Theme</span>
            </label>
            <label className="flex items-center space-x-3">
              <input type="radio" name="theme" value="auto" />
              <span>Auto (System Preference)</span>
            </label>
          </div>
        </div>
        
        <div>
          <h3 className="text-md font-medium text-gray-900 mb-4">Language</h3>
          <select className="block w-48 rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500">
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
          </select>
        </div>
        
        <div className="flex justify-end">
          <Button>Save Settings</Button>
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;





const AdminTab: React.FC = () => {
  const [flags, setFlags] = React.useState<{ knowledge_graph_enabled: boolean; summarization_enabled: boolean; auto_summarize_on_process: boolean } | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [embedStats, setEmbedStats] = React.useState<{ embedding_model?: string; available_models?: string[] } | null>(null);
  const [llmModels, setLlmModels] = React.useState<{ models: string[]; default_model?: string } | null>(null);
  const [switchingEmbed, setSwitchingEmbed] = React.useState(false);
  const [switchingLLM, setSwitchingLLM] = React.useState(false);
  React.useEffect(() => {
    let cancel = false;
    apiClient.getFeatureFlags().then(f => { if (!cancel) setFlags(f); });
    apiClient.getVectorStoreStats().then(s => { if (!cancel) setEmbedStats(s); });
    apiClient.listLLMModels().then(m => { if (!cancel) setLlmModels(m); });
    return () => { cancel = true; };
  }, []);

  const handleToggle = (k: keyof NonNullable<typeof flags>) => (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!flags) return;
    setFlags({ ...flags, [k]: e.target.checked } as any);
  };

  const onSave = async () => {
    if (!flags) return;
    setSaving(true);
    try {
      await apiClient.updateFeatureFlags(flags);
      toast.success('Feature flags updated');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to update flags');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-6">Administration</h2>
      {!flags ? (
        <div className="text-gray-600">Loading…</div>
      ) : (
        <div className="space-y-6">
          <div>
            <h3 className="text-md font-medium text-gray-900 mb-2">Feature Flags</h3>
            <div className="space-y-2 text-sm">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={flags.knowledge_graph_enabled} onChange={handleToggle('knowledge_graph_enabled')} />
                <span>Knowledge Graph Enabled</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={flags.summarization_enabled} onChange={handleToggle('summarization_enabled')} />
                <span>Summarization Enabled</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={flags.auto_summarize_on_process} onChange={handleToggle('auto_summarize_on_process')} />
                <span>Auto-summarize on Processing</span>
              </label>
            </div>
            <div className="mt-4">
              <Button onClick={onSave} loading={saving}>Save</Button>
            </div>
          </div>

          <div>
            <h3 className="text-md font-medium text-gray-900 mb-2">Embedding Model</h3>
            {!embedStats ? (
              <div className="text-gray-600">Loading…</div>
            ) : (
              <div className="space-y-2 text-sm">
                <div>Current: <span className="font-medium">{embedStats.embedding_model || 'Unknown'}</span></div>
                <div className="flex items-center gap-2">
                  <select id="embed-model" className="border rounded px-2 py-1 text-sm">
                    {(embedStats.available_models || []).map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <Button
                    onClick={async () => {
                      const sel = (document.getElementById('embed-model') as HTMLSelectElement | null)?.value;
                      if (!sel) return;
                      try {
                        setSwitchingEmbed(true);
                        await apiClient.switchEmbeddingModel(sel);
                        const s = await apiClient.getVectorStoreStats();
                        setEmbedStats(s);
                        toast.success('Embedding model updated');
                      } catch (e: any) {
                        toast.error(e?.response?.data?.detail || e?.message || 'Failed to switch model');
                      } finally {
                        setSwitchingEmbed(false);
                      }
                    }}
                    loading={switchingEmbed}
                  >
                    Switch
                  </Button>
                </div>
                <div className="text-xs text-gray-500">Switching models may require reprocessing documents for best results.</div>
              </div>
            )}
          </div>

          <div>
            <h3 className="text-md font-medium text-gray-900 mb-2">LLM Model</h3>
            {!llmModels ? (
              <div className="text-gray-600">Loading…</div>
            ) : (
              <div className="space-y-2 text-sm">
                <div>Default: <span className="font-medium">{llmModels.default_model || 'Unknown'}</span></div>
                <div className="flex items-center gap-2">
                  <select id="llm-model" className="border rounded px-2 py-1 text-sm">
                    {(llmModels.models || []).map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                  <Button
                    onClick={async () => {
                      const sel = (document.getElementById('llm-model') as HTMLSelectElement | null)?.value;
                      if (!sel) return;
                      try {
                        setSwitchingLLM(true);
                        await apiClient.switchLLMModel(sel);
                        const m = await apiClient.listLLMModels();
                        setLlmModels(m);
                        toast.success('LLM model updated');
                      } catch (e: any) {
                        toast.error(e?.response?.data?.detail || e?.message || 'Failed to switch LLM model');
                      } finally {
                        setSwitchingLLM(false);
                      }
                    }}
                    loading={switchingLLM}
                  >
                    Switch
                  </Button>
                </div>
                <div className="text-xs text-gray-500">Switching LLM model affects future generations immediately.</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
