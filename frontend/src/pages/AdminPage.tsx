/**
 * Admin dashboard for system management and monitoring
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';
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
  Globe,
  UserCircle2,
  Bot,
  Copy
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

import { apiClient } from '../services/api';
import { SystemHealth, SystemStats, DocumentSource, Persona, AgentDefinition, AgentDefinitionCreate, AgentDefinitionUpdate, CapabilityInfo, AgentDefinitionSummary, ToolAudit } from '../types';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import ProgressBar from '../components/common/ProgressBar';
import AlertModal from '../components/common/AlertModal';
import LoadingSpinner from '../components/common/LoadingSpinner';
import toast from 'react-hot-toast';
import ConfirmationModal from '../components/common/ConfirmationModal';

const AdminPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState(() => {
    const params = new URLSearchParams(location.search);
    return params.get('tab') || 'overview';
  });
  const [focusedPersonaId, setFocusedPersonaId] = useState<string | undefined>(() => {
    const params = new URLSearchParams(location.search);
    return params.get('personaId') || undefined;
  });

  const tabs = [
    { id: 'overview', name: 'Overview', icon: BarChart3 },
    { id: 'health', name: 'System Health', icon: Activity },
    { id: 'sources', name: 'Data Sources', icon: Database },
    { id: 'tasks', name: 'Background Tasks', icon: Settings },
    { id: 'logs', name: 'System Logs', icon: Server },
    { id: 'personas', name: 'Personas', icon: Users },
    { id: 'agents', name: 'Agents', icon: Bot },
    { id: 'ai-hub', name: 'AI Hub', icon: Cpu },
    { id: 'tool-approvals', name: 'Tool Approvals', icon: CheckCircle },
  ];

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tabParam = params.get('tab') || 'overview';
    if (tabParam !== activeTab) {
      setActiveTab(tabParam);
    }
    const personaParam = params.get('personaId') || undefined;
    if (personaParam && personaParam !== focusedPersonaId) {
      setFocusedPersonaId(personaParam);
    }
  }, [location.search]);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    const params = new URLSearchParams(location.search);
    if (tabId === 'overview') {
      params.delete('tab');
    } else {
      params.set('tab', tabId);
    }
    params.delete('personaId');
    navigate(
      {
        pathname: location.pathname,
        search: params.toString() ? `?${params.toString()}` : '',
      },
      { replace: true }
    );
  };

  const handlePersonaFocusCleared = () => {
    setFocusedPersonaId(undefined);
    const params = new URLSearchParams(location.search);
    if (params.has('personaId')) {
      params.delete('personaId');
      navigate(
        {
          pathname: location.pathname,
          search: params.toString() ? `?${params.toString()}` : '',
        },
        { replace: true }
      );
    }
  };

  // Fetch system health
  const { data: health, isLoading: healthLoading, refetch: refetchHealth } = useQuery(
    'systemHealth',
    () => apiClient.getSystemHealth(),
    {
      refetchInterval: 30000, // Refresh every 30 seconds
      refetchOnWindowFocus: false,
    }
  );

  // Fetch system stats
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery(
    'systemStats',
    () => apiClient.getSystemStats(),
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
                  onClick={() => handleTabChange(tab.id)}
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
          {activeTab === 'personas' && (
            <PersonasTab
              focusPersonaId={focusedPersonaId}
              onFocusPersonaHandled={handlePersonaFocusCleared}
            />
          )}
          {activeTab === 'agents' && <AgentsTab />}
          {activeTab === 'ai-hub' && <AIHubAdminTab />}
          {activeTab === 'tool-approvals' && <ToolApprovalsTab />}
        </div>
      </div>
    </div>
  );
};

const AIHubAdminTab: React.FC = () => {
  const queryClient = useQueryClient();

  const { data: allTemplates, isLoading: templatesLoading } = useQuery(
    ['admin', 'ai-hub', 'eval-templates', 'all'],
    () => apiClient.listAllTrainingEvalTemplates(),
    { refetchOnWindowFocus: false }
  );

  const { data: enabledConfig, isLoading: enabledLoading } = useQuery(
    ['admin', 'ai-hub', 'eval-templates', 'enabled'],
    () => apiClient.getEnabledAIHubEvalTemplates(),
    { refetchOnWindowFocus: false }
  );

  const { data: allPresets, isLoading: presetsLoading } = useQuery(
    ['admin', 'ai-hub', 'dataset-presets', 'all'],
    () => apiClient.listAllDatasetPresets(),
    { refetchOnWindowFocus: false }
  );

  const { data: enabledPresetsConfig, isLoading: enabledPresetsLoading } = useQuery(
    ['admin', 'ai-hub', 'dataset-presets', 'enabled'],
    () => apiClient.getEnabledAIHubDatasetPresets(),
    { refetchOnWindowFocus: false }
  );

  const { data: customerProfileData, isLoading: profileLoading } = useQuery(
    ['admin', 'ai-hub', 'customer-profile'],
    () => apiClient.getAIHubCustomerProfile(),
    { refetchOnWindowFocus: false }
  );

  const enabledSet = useMemo(() => {
    const enabled = enabledConfig?.enabled || [];
    return new Set(enabled);
  }, [enabledConfig?.enabled]);

  const enabledPresetsSet = useMemo(() => {
    const enabled = enabledPresetsConfig?.enabled || [];
    return new Set(enabled);
  }, [enabledPresetsConfig?.enabled]);

  const [localEnabled, setLocalEnabled] = useState<Record<string, boolean>>({});
  const [dirty, setDirty] = useState(false);

  const [localPresetsEnabled, setLocalPresetsEnabled] = useState<Record<string, boolean>>({});
  const [presetsDirty, setPresetsDirty] = useState(false);

  const [profileName, setProfileName] = useState('');
  const [profileKeywords, setProfileKeywords] = useState(''); // comma-separated
  const [profileWorkflows, setProfileWorkflows] = useState<{ triage: boolean; extraction: boolean; literature: boolean }>({
    triage: true,
    extraction: true,
    literature: true,
  });
  const [profileNotes, setProfileNotes] = useState('');
  const [profileDirty, setProfileDirty] = useState(false);

  useEffect(() => {
    const templates = allTemplates?.templates || [];
    const enabled = enabledConfig?.enabled || [];
    const hasAllowlist = enabled.length > 0;

    const next: Record<string, boolean> = {};
    for (const t of templates) {
      next[t.id] = hasAllowlist ? enabledSet.has(t.id) : true;
    }
    setLocalEnabled(next);
    setDirty(false);
  }, [allTemplates?.templates, enabledConfig?.enabled, enabledSet]);

  useEffect(() => {
    const presets = (allPresets as any)?.presets || [];
    const enabled = enabledPresetsConfig?.enabled || [];
    const hasAllowlist = enabled.length > 0;

    const next: Record<string, boolean> = {};
    for (const p of presets) {
      next[p.id] = hasAllowlist ? enabledPresetsSet.has(p.id) : true;
    }
    setLocalPresetsEnabled(next);
    setPresetsDirty(false);
  }, [allPresets, enabledPresetsConfig?.enabled, enabledPresetsSet]);

  useEffect(() => {
    const p = (customerProfileData as any)?.profile;
    if (!p) {
      setProfileName('');
      setProfileKeywords('');
      setProfileNotes('');
      setProfileWorkflows({ triage: true, extraction: true, literature: true });
      setProfileDirty(false);
      return;
    }
    setProfileName(p.name || '');
    setProfileKeywords(Array.isArray(p.keywords) ? p.keywords.join(', ') : '');
    setProfileNotes(p.notes || '');
    const prefs = Array.isArray(p.preferred_workflows) ? p.preferred_workflows : [];
    setProfileWorkflows({
      triage: prefs.length ? prefs.includes('triage') : true,
      extraction: prefs.length ? prefs.includes('extraction') : true,
      literature: prefs.length ? prefs.includes('literature') : true,
    });
    setProfileDirty(false);
  }, [customerProfileData]);

  const saveProfileMutation = useMutation(
    async () => {
      const workflows: string[] = Object.entries(profileWorkflows)
        .filter(([, v]) => v)
        .map(([k]) => k);
      const keywords = profileKeywords
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean);
      return apiClient.setAIHubCustomerProfile({
        profile: {
          name: profileName.trim() || 'Customer',
          keywords,
          preferred_workflows: workflows as any,
          notes: profileNotes.trim() ? profileNotes.trim() : null,
        },
      });
    },
    {
      onSuccess: () => {
        toast.success('Customer profile saved');
        queryClient.invalidateQueries(['admin', 'ai-hub', 'customer-profile']);
        setProfileDirty(false);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to save customer profile');
      },
    }
  );

  const clearFeedbackMutation = useMutation(
    async (profileId: string) => apiClient.clearAIHubRecommendationFeedback(profileId),
    {
      onSuccess: (res) => {
        toast.success(`Cleared ${res.deleted} feedback entries`);
        queryClient.invalidateQueries(['admin', 'ai-hub', 'feedback-stats']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to clear feedback');
      },
    }
  );

  const profileId = (customerProfileData as any)?.profile?.id as string | undefined;
  const profileNameFromServer = (customerProfileData as any)?.profile?.name as string | undefined;

  const { data: feedbackStats } = useQuery(
    ['admin', 'ai-hub', 'feedback-stats', profileId],
    () => apiClient.getAIHubRecommendationFeedbackStats(String(profileId), 50),
    { enabled: !!profileId, refetchOnWindowFocus: false }
  );

  const backfillMutation = useMutation(
    async () => {
      if (!profileId) throw new Error('Missing profile id');
      const name = profileNameFromServer || profileName || 'Customer';
      return apiClient.backfillAIHubRecommendationFeedbackProfileId({ profile_id: String(profileId), profile_name: name });
    },
    {
      onSuccess: (res) => {
        toast.success(`Backfilled ${res.updated} rows`);
        queryClient.invalidateQueries(['admin', 'ai-hub', 'feedback-stats', profileId]);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Backfill failed');
      },
    }
  );

  const saveMutation = useMutation(
    (ids: string[]) => apiClient.setEnabledAIHubEvalTemplates({ enabled: ids }),
    {
      onSuccess: () => {
        toast.success('Saved');
        queryClient.invalidateQueries(['admin', 'ai-hub', 'eval-templates', 'enabled']);
        queryClient.invalidateQueries(['training-eval-templates']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to save');
      },
    }
  );

  const savePresetsMutation = useMutation(
    (ids: string[]) => apiClient.setEnabledAIHubDatasetPresets({ enabled: ids }),
    {
      onSuccess: () => {
        toast.success('Saved');
        queryClient.invalidateQueries(['admin', 'ai-hub', 'dataset-presets', 'enabled']);
        queryClient.invalidateQueries(['ai-hub', 'dataset-presets', 'enabled']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to save');
      },
    }
  );

  const applyBundleMutation = useMutation(
    async (bundle: { evalTemplateIds: string[]; datasetPresetIds: string[] }) => {
      await Promise.all([
        apiClient.setEnabledAIHubEvalTemplates({ enabled: bundle.evalTemplateIds }),
        apiClient.setEnabledAIHubDatasetPresets({ enabled: bundle.datasetPresetIds }),
      ]);
    },
    {
      onSuccess: () => {
        toast.success('Bundle applied');
        queryClient.invalidateQueries(['admin', 'ai-hub', 'eval-templates', 'enabled']);
        queryClient.invalidateQueries(['admin', 'ai-hub', 'dataset-presets', 'enabled']);
        queryClient.invalidateQueries(['training-eval-templates']);
        queryClient.invalidateQueries(['ai-hub', 'dataset-presets', 'enabled']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to apply bundle');
      },
    }
  );

  const templates = allTemplates?.templates || [];
  const presets: any[] = (allPresets as any)?.presets || [];

  const selectedIds = useMemo(() => {
    return templates.filter((t) => localEnabled[t.id]).map((t) => t.id);
  }, [templates, localEnabled]);

  const allSelected = templates.length > 0 && templates.every((t) => localEnabled[t.id]);

  const selectedPresetIds = useMemo(() => {
    return presets.filter((p) => localPresetsEnabled[p.id]).map((p) => p.id);
  }, [presets, localPresetsEnabled]);

  const allPresetsSelected = presets.length > 0 && presets.every((p) => localPresetsEnabled[p.id]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-900">AI Hub</h2>
        <p className="text-sm text-gray-600">Configure customer-visible AI Hub plugins</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">Customer Profile</h3>
            <p className="text-xs text-gray-600 mt-1">
              Used by AI Scientist to generate customer-specific bundles without re-inferring from docs every run.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                setProfileName('');
                setProfileKeywords('');
                setProfileNotes('');
                setProfileWorkflows({ triage: true, extraction: true, literature: true });
                setProfileDirty(true);
              }}
              disabled={profileLoading}
              title="Clears local fields (does not change saved profile until you click Save)"
            >
              Clear
            </Button>
            <Button
              size="sm"
              onClick={() => saveProfileMutation.mutate()}
              disabled={profileLoading || saveProfileMutation.isLoading || !profileDirty}
              title={!profileDirty ? 'No changes to save' : undefined}
            >
              {saveProfileMutation.isLoading ? 'Saving…' : 'Save'}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                const id = (customerProfileData as any)?.profile?.id;
                if (!id) {
                  toast.error('Save a customer profile first');
                  return;
                }
                if (!window.confirm('Clear AI Scientist learning feedback for this customer profile?')) {
                  return;
                }
                clearFeedbackMutation.mutate(String(id));
              }}
              disabled={profileLoading || clearFeedbackMutation.isLoading}
              title="Resets accept/reject learning for this customer profile"
            >
              {clearFeedbackMutation.isLoading ? 'Clearing…' : 'Clear Learning Feedback'}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                if (!profileId) {
                  toast.error('Save a customer profile first');
                  return;
                }
                if (!window.confirm('Backfill legacy feedback rows using profile name?')) {
                  return;
                }
                backfillMutation.mutate();
              }}
              disabled={profileLoading || backfillMutation.isLoading}
              title="If you have older feedback rows that only stored profile name, this fills in profile id."
            >
              {backfillMutation.isLoading ? 'Backfilling…' : 'Backfill Profile ID'}
            </Button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
            <input
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={profileName}
              onChange={(e) => {
                setProfileName(e.target.value);
                setProfileDirty(true);
              }}
              placeholder="e.g., Robotics Lab (Pilot)"
            />
            {(customerProfileData as any)?.profile?.id && (
              <div className="mt-1 text-xs text-gray-500">
                Profile id: <span className="font-mono">{String((customerProfileData as any).profile.id)}</span>
              </div>
            )}
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Preferred workflows</label>
            <div className="flex flex-wrap gap-3 text-sm text-gray-700">
              {(['triage', 'extraction', 'literature'] as const).map((k) => (
                <label key={k} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={(profileWorkflows as any)[k]}
                    onChange={(e) => {
                      setProfileWorkflows((prev) => ({ ...prev, [k]: e.target.checked }));
                      setProfileDirty(true);
                    }}
                  />
                  {k}
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-4">
          <label className="block text-xs font-medium text-gray-700 mb-1">Keywords (comma-separated)</label>
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={profileKeywords}
            onChange={(e) => {
              setProfileKeywords(e.target.value);
              setProfileDirty(true);
            }}
            placeholder="e.g., robotics, slam, control, benchmarking, reproducibility"
          />
        </div>

        <div className="mt-4">
          <label className="block text-xs font-medium text-gray-700 mb-1">Notes (optional)</label>
          <textarea
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
            rows={3}
            value={profileNotes}
            onChange={(e) => {
              setProfileNotes(e.target.value);
              setProfileDirty(true);
            }}
            placeholder="Short description of the customer, what they do, and what success looks like."
          />
        </div>

        {profileId && (
          <div className="mt-4">
            <div className="text-sm font-semibold text-gray-900">Learning analytics</div>
            <div className="text-xs text-gray-600 mt-1">Top accepted/rejected items for this profile.</div>
            <div className="mt-2 border border-gray-200 rounded-lg overflow-hidden">
              <div className="grid grid-cols-12 bg-gray-50 text-xs font-medium text-gray-700 px-3 py-2">
                <div className="col-span-4">Item</div>
                <div className="col-span-3">Type</div>
                <div className="col-span-2 text-right">Accept</div>
                <div className="col-span-2 text-right">Reject</div>
                <div className="col-span-1 text-right">Net</div>
              </div>
              <div className="divide-y divide-gray-100">
                {((feedbackStats as any)?.rows || []).length === 0 ? (
                  <div className="px-3 py-3 text-sm text-gray-500">No feedback yet.</div>
                ) : (
                  ((feedbackStats as any)?.rows || []).slice(0, 20).map((r: any) => (
                    <div key={`${r.item_type}:${r.item_id}`} className="grid grid-cols-12 px-3 py-2 text-sm">
                      <div className="col-span-4 font-mono text-xs text-gray-800 break-words">{r.item_id}</div>
                      <div className="col-span-3 text-gray-600">{r.item_type}</div>
                      <div className="col-span-2 text-right text-gray-700">{r.accepts}</div>
                      <div className="col-span-2 text-right text-gray-700">{r.rejects}</div>
                      <div className="col-span-1 text-right font-medium">{r.net}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">Customer Bundles</h3>
            <p className="text-xs text-gray-600 mt-1">
              Apply a known-good plugin configuration for a customer profile. This updates both dataset presets and eval templates.
            </p>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={() => {
              if (!window.confirm('Apply Research Lab (Perf Triage) bundle? This will overwrite current allowlists.')) {
                return;
              }
              applyBundleMutation.mutate({
                evalTemplateIds: ['perf_regression_triage_v1'],
                datasetPresetIds: ['perf_regression_triage_v1'],
              });
            }}
            disabled={applyBundleMutation.isLoading}
          >
            {applyBundleMutation.isLoading ? 'Applying…' : 'Research Lab (Perf Triage)'}
          </Button>
          <Button
            size="sm"
            onClick={() => {
              if (
                !window.confirm(
                  'Apply Research Lab (All Workflows) bundle? This will overwrite current allowlists.'
                )
              ) {
                return;
              }
              applyBundleMutation.mutate({
                evalTemplateIds: ['perf_regression_triage_v1', 'extraction_quality_v1', 'literature_triage_v1'],
                datasetPresetIds: ['perf_regression_triage_v1', 'repro_checklist_v1', 'gap_analysis_hypotheses_v1'],
              });
            }}
            disabled={applyBundleMutation.isLoading}
          >
            {applyBundleMutation.isLoading ? 'Applying…' : 'Research Lab (All Workflows)'}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              if (!window.confirm('Enable all presets + eval templates (default mode)? This will clear allowlists.')) {
                return;
              }
              applyBundleMutation.mutate({ evalTemplateIds: [], datasetPresetIds: [] });
            }}
            disabled={applyBundleMutation.isLoading}
            title="Clears allowlists so the system defaults to 'all enabled'"
          >
            Enable All (Default Mode)
          </Button>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">Eval Templates</h3>
            <p className="text-xs text-gray-600 mt-1">
              If none are selected, all templates are enabled (default). Select a subset to enforce a customer allowlist.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                const next: Record<string, boolean> = {};
                for (const t of templates) next[t.id] = true;
                setLocalEnabled(next);
                setDirty(true);
              }}
              disabled={templatesLoading || enabledLoading || templates.length === 0}
            >
              Enable all
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                const next: Record<string, boolean> = {};
                for (const t of templates) next[t.id] = false;
                setLocalEnabled(next);
                setDirty(true);
              }}
              disabled={templatesLoading || enabledLoading || templates.length === 0}
              title="Clearing selection reverts to default behavior (all enabled)"
            >
              Clear selection
            </Button>
            <Button
              size="sm"
              onClick={() => saveMutation.mutate(selectedIds)}
              disabled={!dirty || saveMutation.isLoading || templatesLoading || enabledLoading}
            >
              {saveMutation.isLoading ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>

        {templatesLoading || enabledLoading ? (
          <div className="flex justify-center py-10">
            <LoadingSpinner size="md" />
          </div>
        ) : templates.length === 0 ? (
          <div className="text-sm text-gray-600 py-6">No eval templates found on the server.</div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={allSelected}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          const next: Record<string, boolean> = {};
                          for (const t of templates) next[t.id] = checked;
                          setLocalEnabled(next);
                          setDirty(true);
                        }}
                      />
                      Enabled
                    </label>
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Template</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Cases</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {templates.map((t) => (
                  <tr key={t.id}>
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={!!localEnabled[t.id]}
                        onChange={(e) => {
                          setLocalEnabled((prev) => ({ ...prev, [t.id]: e.target.checked }));
                          setDirty(true);
                        }}
                      />
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium text-gray-900">{t.name}</div>
                      {t.description && <div className="text-xs text-gray-500">{t.description}</div>}
                      <div className="text-xs text-gray-400 mt-0.5">v{t.version}</div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-xs font-mono text-gray-700">{t.id}</div>
                    </td>
                    <td className="px-4 py-3 text-right text-sm text-gray-700">{t.case_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-3 text-xs text-gray-500">
              Current mode: {selectedIds.length === 0 ? 'default (all enabled)' : `allowlist (${selectedIds.length} enabled)`}
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-900">Dataset Presets</h3>
            <p className="text-xs text-gray-600 mt-1">
              If none are selected, all presets are enabled (default). Select a subset to enforce a customer allowlist.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                const next: Record<string, boolean> = {};
                for (const p of presets) next[p.id] = true;
                setLocalPresetsEnabled(next);
                setPresetsDirty(true);
              }}
              disabled={presetsLoading || enabledPresetsLoading || presets.length === 0}
            >
              Enable all
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                const next: Record<string, boolean> = {};
                for (const p of presets) next[p.id] = false;
                setLocalPresetsEnabled(next);
                setPresetsDirty(true);
              }}
              disabled={presetsLoading || enabledPresetsLoading || presets.length === 0}
              title="Clearing selection reverts to default behavior (all enabled)"
            >
              Clear selection
            </Button>
            <Button
              size="sm"
              onClick={() => savePresetsMutation.mutate(selectedPresetIds)}
              disabled={!presetsDirty || savePresetsMutation.isLoading || presetsLoading || enabledPresetsLoading}
            >
              {savePresetsMutation.isLoading ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>

        {presetsLoading || enabledPresetsLoading ? (
          <div className="flex justify-center py-10">
            <LoadingSpinner size="md" />
          </div>
        ) : presets.length === 0 ? (
          <div className="text-sm text-gray-600 py-6">No dataset presets found on the server.</div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={allPresetsSelected}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          const next: Record<string, boolean> = {};
                          for (const p of presets) next[p.id] = checked;
                          setLocalPresetsEnabled(next);
                          setPresetsDirty(true);
                        }}
                      />
                      Enabled
                    </label>
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Preset</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {presets.map((p) => (
                  <tr key={p.id}>
                    <td className="px-4 py-3">
                      <input
                        type="checkbox"
                        checked={!!localPresetsEnabled[p.id]}
                        onChange={(e) => {
                          setLocalPresetsEnabled((prev) => ({ ...prev, [p.id]: e.target.checked }));
                          setPresetsDirty(true);
                        }}
                      />
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-sm font-medium text-gray-900">{p.name}</div>
                      {p.description && <div className="text-xs text-gray-500">{p.description}</div>}
                      <div className="text-xs text-gray-400 mt-0.5">type: {p.dataset_type}</div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-xs font-mono text-gray-700">{p.id}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-3 text-xs text-gray-500">
              Current mode: {selectedPresetIds.length === 0 ? 'default (all enabled)' : `allowlist (${selectedPresetIds.length} enabled)`}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ToolApprovalsTab: React.FC = () => {
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery(
    ['toolAudits', 'requires_approval'],
    () => apiClient.listToolAudits({ status: 'requires_approval', limit: 100 }),
    { refetchInterval: 5000, refetchOnWindowFocus: false }
  );

  const approveMutation = useMutation(
    (auditId: string) => apiClient.approveToolAudit(auditId),
    {
      onSuccess: () => {
        toast.success('Approved');
        queryClient.invalidateQueries(['toolAudits', 'requires_approval']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to approve');
      },
    }
  );

  const rejectMutation = useMutation(
    (auditId: string) => apiClient.rejectToolAudit(auditId),
    {
      onSuccess: () => {
        toast.success('Rejected');
        queryClient.invalidateQueries(['toolAudits', 'requires_approval']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to reject');
      },
    }
  );

  const runMutation = useMutation(
    (auditId: string) => apiClient.runToolAudit(auditId),
    {
      onSuccess: () => {
        toast.success('Tool executed');
        queryClient.invalidateQueries(['toolAudits', 'requires_approval']);
      },
      onError: (e: any) => {
        toast.error(e?.message || 'Failed to run tool');
      },
    }
  );

  const rows: ToolAudit[] = data || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Tool Approvals</h2>
          <p className="text-sm text-gray-600">Approve and run dangerous agent tool calls</p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-10">
          <LoadingSpinner size="md" />
        </div>
      ) : rows.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-6 text-gray-600">No pending approvals.</div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tool</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Requested</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Approval</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td className="px-6 py-4">
                      <div className="text-sm font-medium text-gray-900">{row.tool_name}</div>
                      <div className="text-xs text-gray-500 break-all">ID: {row.id}</div>
                      {row.conversation_id && (
                        <div className="text-xs text-gray-500 break-all">Conversation: {row.conversation_id}</div>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {formatDistanceToNow(new Date(row.created_at), { addSuffix: true })}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      <div className="flex items-center space-x-2">
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-yellow-100 text-yellow-800">
                          {row.approval_status || 'pending'}
                        </span>
                        <span className="text-xs text-gray-500">{row.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right space-x-2">
                      <Button
                        size="sm"
                        onClick={() => approveMutation.mutate(row.id)}
                        disabled={approveMutation.isLoading || rejectMutation.isLoading || runMutation.isLoading}
                      >
                        Approve
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => runMutation.mutate(row.id)}
                        disabled={approveMutation.isLoading || rejectMutation.isLoading || runMutation.isLoading || row.approval_status !== 'approved'}
                      >
                        Run
                      </Button>
                      <Button
                        size="sm"
                        variant="danger"
                        onClick={() => rejectMutation.mutate(row.id)}
                        disabled={approveMutation.isLoading || rejectMutation.isLoading || runMutation.isLoading}
                      >
                        Reject
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
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
            {typeof stats?.documents?.without_summary === 'number' && (
              <div className="text-sm text-gray-500">
                {stats?.documents?.without_summary} without summary
              </div>
            )}
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
type GithubSourceFormState = {
  name: string;
  token: string;
  repos: string;
  includeFiles: boolean;
  includeIssues: boolean;
  includePRs: boolean;
  includeWiki: boolean;
  incrementalFiles: boolean;
  useGitignore: boolean;
  fileExtensions: string;
  ignoreGlobs: string;
  apiBase: string;
  maxPages: number;
  startImmediately: boolean;
};

const INITIAL_GITHUB_FORM: GithubSourceFormState = {
  name: '',
  token: '',
  repos: '',
  includeFiles: true,
  includeIssues: true,
  includePRs: false,
  includeWiki: false,
  incrementalFiles: true,
  useGitignore: true,
  fileExtensions: '.md,.txt,.rst,.py,.js,.ts,.tsx',
  ignoreGlobs: '',
  apiBase: '',
  maxPages: 10,
  startImmediately: true,
};

const DataSourcesTab: React.FC = () => {
  const queryClient = useQueryClient();
  const ingestionWebSockets = React.useRef<Record<string, WebSocket>>({});
  const [ingProgress, setIngProgress] = useState<Record<string, { progress?: number; current?: number; total?: number; status?: string; remaining_seconds?: number; remaining_formatted?: string; elapsed?: number }>>({});
  const [dryRunCache, setDryRunCache] = useState<Record<string, any>>({});
  const [dryRunOpen, setDryRunOpen] = useState(false);
  const [dryRunResult, setDryRunResult] = useState<{
    source_id?: string;
    source_name?: string;
    total?: number;
    estimated_new?: number;
    estimated_existing?: number;
    sample?: Array<{ title?: string; identifier?: string; type?: string }>;
    mode?: string;
    by_type?: Record<string, number>;
  } | null>(null);
  const [dryRunOverrides, setDryRunOverrides] = useState<{ include_files: boolean; include_issues: boolean; include_prs: boolean; include_mrs: boolean; include_wiki: boolean }>({ include_files: true, include_issues: true, include_prs: true, include_mrs: true, include_wiki: true });
  const [historySourceId, setHistorySourceId] = useState<string | null>(null);
  const [historyLimit, setHistoryLimit] = useState<number>(20);
  const [historyOffset, setHistoryOffset] = useState<number>(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState<any[] | null>(null);
  const [githubForm, setGithubForm] = useState<GithubSourceFormState>(() => ({ ...INITIAL_GITHUB_FORM }));

  const { data: sources, isLoading } = useQuery(
    'documentSources',
    () => apiClient.getDocumentSources(),
    {
      refetchOnWindowFocus: false,
    }
  );

  const { data: nextRuns } = useQuery(
    'sourcesNextRun',
    () => apiClient.getSourcesNextRun(),
    {
      refetchInterval: 60000, // refresh every minute
      refetchOnWindowFocus: false,
    }
  );

  const { data: ingestionStatus, isLoading: ingestionStatusLoading } = useQuery(
    ['adminIngestionStatus'],
    () => apiClient.getAdminIngestionStatus(),
    {
      refetchInterval: 30000,
      refetchOnWindowFocus: false,
    }
  );

  // Auto-connect ingestion WS for sources already syncing
  useEffect(() => {
    if (!sources) return;
    sources.forEach((source) => {
      const syncing = (source as any)?.is_syncing === true;
      if (syncing && !ingestionWebSockets.current[source.id]) {
        try {
          const ws = apiClient.createIngestionProgressWebSocket(source.id);
          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (data.type === 'ingestion_progress') {
                const p = data.progress || {};
                setIngProgress(prev => ({
                  ...prev,
                  [source.id]: { progress: p.progress, current: p.current, total: p.total, status: p.status }
                }));
              } else if (data.type === 'ingestion_complete' || data.type === 'ingestion_error') {
                if (data.type === 'ingestion_error') {
                  toast.error(`Ingestion error: ${data.error || 'Unknown error'}`);
                }
                if (ingestionWebSockets.current[source.id]) {
                  ingestionWebSockets.current[source.id].close();
                  delete ingestionWebSockets.current[source.id];
                }
                setIngProgress(prev => { const n = { ...prev } as any; delete n[source.id]; return n; });
                queryClient.invalidateQueries('documentSources');
                queryClient.invalidateQueries('documents');
              }
            } catch (e) {
              console.error('Ingestion WS parse error', e);
            }
          };
          ws.onclose = () => { delete ingestionWebSockets.current[source.id]; };
          ingestionWebSockets.current[source.id] = ws;
        } catch (e) {
          console.error('Ingestion WS auto-connect failed', e);
        }
      }
    });
    return () => {
      // leave open across rerenders; closed when completes or tab unmounts
    };
  }, [sources, queryClient]);

  const syncAllMutation = useMutation(
    () => apiClient.triggerFullSync(),
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
    (args: { id: string; forceFull?: boolean }) => apiClient.triggerSourceSync(args.id, { forceFull: args.forceFull }),
    {
      onSuccess: (_res, args) => {
        toast.success('Source synchronization started');
        try {
          const ws = apiClient.createIngestionProgressWebSocket(args.id);
          ws.onopen = () => console.log('Ingestion WS open for', args.id);
          ws.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              if (data.type === 'ingestion_progress') {
                const p = data.progress || {};
                setIngProgress(prev => ({
                  ...prev,
                  [args.id]: { progress: p.progress, current: p.current, total: p.total, status: p.status, remaining_formatted: p.remaining_formatted }
                }));
              } else if (data.type === 'ingestion_complete') {
                // Clear progress on completion
                setIngProgress(prev => { const n = { ...prev } as any; delete n[args.id]; return n; });
                if (ingestionWebSockets.current[args.id]) {
                  ingestionWebSockets.current[args.id].close();
                  delete ingestionWebSockets.current[args.id];
                }
                queryClient.invalidateQueries('documentSources');
                queryClient.invalidateQueries('documents');
              } else if (data.type === 'ingestion_error') {
                toast.error(`Ingestion error: ${data.error || 'Unknown error'}`);
                setIngProgress(prev => { const n = { ...prev } as any; delete n[args.id]; return n; });
                if (ingestionWebSockets.current[args.id]) {
                  ingestionWebSockets.current[args.id].close();
                  delete ingestionWebSockets.current[args.id];
                }
              } else if (data.type === 'ingestion_status') {
                const status = data.status || {};
                if (status.canceled) {
                  toast('Sync canceled', { icon: '⏹️' });
                }
              }
            } catch (err) {
              console.error('Error parsing ingestion message', err);
            }
          };
          ws.onclose = () => {
            console.log('Ingestion WS closed for', args.id);
            delete ingestionWebSockets.current[args.id];
          };
          ws.onerror = (e) => console.error('Ingestion WS error', e);
          ingestionWebSockets.current[args.id] = ws;
        } catch (e) {
          console.error('Failed to open ingestion WS', e);
        }
      },
      onError: () => {
        toast.error('Failed to start source synchronization');
      },
    }
  );

  const clearErrorMutation = useMutation(
    (sourceId: string) => apiClient.clearSourceError(sourceId),
    {
      onSuccess: (_res, sourceId) => {
        toast.success('Cleared last error');
        queryClient.invalidateQueries('documentSources');
      },
      onError: () => {
        toast.error('Failed to clear error');
      },
    }
  );

  const githubSourceMutation = useMutation(
    async (input: { name: string; config: any; startImmediately: boolean }) => {
      const created = await apiClient.createDocumentSource({
        name: input.name,
        source_type: 'github',
        config: input.config,
      });
      if (input.startImmediately && created?.id) {
        try {
          await apiClient.triggerSourceSync(created.id);
          toast.success('GitHub sync started');
        } catch (err) {
          console.error('Failed to start GitHub sync', err);
          toast.error('Source created, but failed to start sync automatically');
        }
      }
      return created;
    },
    {
      onSuccess: () => {
        toast.success('GitHub source created');
        queryClient.invalidateQueries('documentSources');
        queryClient.invalidateQueries('sourcesNextRun');
        setGithubForm({ ...INITIAL_GITHUB_FORM, token: '' });
      },
      onError: () => {
        toast.error('Failed to create GitHub source');
      },
    }
  );

  const handleCreateGithubSource = () => {
    const token = githubForm.token.trim();
    const repoList = githubForm.repos
      .split(/[\n,]+/)
      .map(r => r.trim())
      .filter(Boolean);

    if (!token) {
      toast.error('GitHub access token is required');
      return;
    }
    if (repoList.length === 0) {
      toast.error('Enter at least one repository (e.g. owner/repo)');
      return;
    }

    const fileExtensions = githubForm.fileExtensions
      .split(',')
      .map(ext => ext.trim())
      .filter(Boolean)
      .map(ext => (ext.startsWith('.') ? ext : `.${ext}`));

    const ignoreGlobs = githubForm.ignoreGlobs
      .split(/\r?\n/)
      .map(line => line.trim())
      .filter(Boolean);

    const config: Record<string, any> = {
      token,
      repos: repoList,
      include_files: githubForm.includeFiles,
      include_issues: githubForm.includeIssues,
      include_pull_requests: githubForm.includePRs,
      include_wiki: githubForm.includeWiki,
      incremental_files: githubForm.incrementalFiles,
      use_gitignore: githubForm.useGitignore,
      max_pages: Number.isFinite(githubForm.maxPages) ? githubForm.maxPages : 10,
    };

    if (fileExtensions.length > 0) {
      config.file_extensions = fileExtensions;
    }
    if (ignoreGlobs.length > 0) {
      config.ignore_globs = ignoreGlobs;
    }
    if (githubForm.apiBase.trim()) {
      config.github_api_base = githubForm.apiBase.trim();
    }

    const name = githubForm.name.trim() || `GitHub: ${repoList[0]}`;
    githubSourceMutation.mutate({
      name,
      config,
      startImmediately: githubForm.startImmediately,
    });
  };

  type DryRunResult = { success: boolean; source_id?: string; source_name?: string; total?: number; estimated_existing?: number; estimated_new?: number; sample?: any[]; error?: string; mode?: string; by_type?: Record<string, number> };
  const dryRunMutation = useMutation<DryRunResult, any, { id: string; overrides?: any }>(
    (args) => apiClient.dryRunSource(args.id, args.overrides),
    {
      onSuccess: (res) => {
        if (res.success) {
          setDryRunResult(res as any);
          setDryRunOpen(true);
          if ((res as any)?.source_id) {
            setDryRunCache(prev => ({ ...prev, [(res as any).source_id]: res }));
          }
        } else {
          toast.error(`Dry-run failed: ${res.error || 'Unknown error'}`);
        }
      },
      onError: () => { toast.error('Dry-run failed'); },
    }
  );

  const getSourceTypeIcon = (type: string) => {
    switch (type) {
      case 'gitlab': return <Database className="w-5 h-5 text-orange-500" />;
      case 'github': return <Database className="w-5 h-5 text-black" />;
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

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Indexing Status</h3>
            <p className="text-sm text-gray-600">
              DB processing status vs vector store points (helps diagnose when Qdrant is empty).
            </p>
          </div>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => queryClient.invalidateQueries(['adminIngestionStatus'])}
            icon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh
          </Button>
        </div>

        {ingestionStatusLoading ? (
          <div className="mt-4">
            <LoadingSpinner text="Loading ingestion status..." />
          </div>
        ) : ingestionStatus ? (
          <div className="mt-4 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 border border-gray-200 rounded-lg">
                <div className="text-xs text-gray-500">Vector store</div>
                <div className="mt-1 text-sm text-gray-900">
                  <span className="font-medium">{ingestionStatus.vector_store.provider}</span>
                  {ingestionStatus.vector_store.collection_name ? (
                    <span className="text-gray-500"> · {ingestionStatus.vector_store.collection_name}</span>
                  ) : null}
                </div>
                <div className="mt-1 text-sm text-gray-700">
                  Points: <span className="font-medium">{ingestionStatus.vector_store.points_total ?? '—'}</span>
                  {typeof ingestionStatus.vector_store.collection_exists === 'boolean' ? (
                    <span className="text-gray-500"> · {ingestionStatus.vector_store.collection_exists ? 'collection ok' : 'collection missing'}</span>
                  ) : null}
                </div>
                {ingestionStatus.vector_store.error ? (
                  <div className="mt-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2">
                    {String(ingestionStatus.vector_store.error)}
                  </div>
                ) : null}
              </div>

              <div className="p-3 border border-gray-200 rounded-lg">
                <div className="text-xs text-gray-500">Documents (DB)</div>
                <div className="mt-1 text-sm text-gray-700">
                  Total: <span className="font-medium">{ingestionStatus.db.documents_total}</span>
                </div>
                <div className="text-sm text-gray-700">
                  Processed: <span className="font-medium">{ingestionStatus.db.documents_processed}</span>
                  <span className="text-gray-500"> · Pending: {ingestionStatus.db.documents_pending} · Failed: {ingestionStatus.db.documents_failed}</span>
                </div>
                <div className="text-sm text-gray-700">
                  Without chunks: <span className="font-medium">{ingestionStatus.db.documents_without_chunks}</span>
                </div>
              </div>

              <div className="p-3 border border-gray-200 rounded-lg">
                <div className="text-xs text-gray-500">Chunks (DB)</div>
                <div className="mt-1 text-sm text-gray-700">
                  Total: <span className="font-medium">{ingestionStatus.db.chunks_total}</span>
                </div>
                <div className="text-sm text-gray-700">
                  Embedded: <span className="font-medium">{ingestionStatus.db.chunks_embedded}</span>
                  <span className="text-gray-500"> · Missing: {ingestionStatus.db.chunks_missing_embedding}</span>
                </div>
              </div>
            </div>

            {(() => {
              const points = ingestionStatus.vector_store.points_total ?? null;
              const embedded = ingestionStatus.db.chunks_embedded ?? 0;
              if (points === 0 && embedded > 0) {
                return (
                  <div className="text-sm text-yellow-900 bg-yellow-50 border border-yellow-200 rounded p-3">
                    Warning: DB shows embedded chunks, but vector store has 0 points. This usually means the app is writing embeddings to DB but not to Qdrant (provider mismatch, Qdrant URL wrong, or collection missing).
                  </div>
                );
              }
              if (points !== null && points > 0 && embedded === 0) {
                return (
                  <div className="text-sm text-yellow-900 bg-yellow-50 border border-yellow-200 rounded p-3">
                    Warning: Vector store has points, but DB chunks have no embedding ids. This usually indicates old data or mismatched indexing logic.
                  </div>
                );
              }
              return null;
            })()}

            <div className="border border-gray-200 rounded-lg overflow-hidden">
              <div className="grid grid-cols-12 bg-gray-50 text-xs font-medium text-gray-700 px-3 py-2">
                <div className="col-span-4">Source</div>
                <div className="col-span-2 text-right">Docs</div>
                <div className="col-span-2 text-right">Processed</div>
                <div className="col-span-2 text-right">Pending</div>
                <div className="col-span-2 text-right">Failed</div>
              </div>
              <div className="divide-y divide-gray-100">
                {(ingestionStatus.sources || []).slice(0, 12).map((s: any) => (
                  <div key={s.source_id} className="grid grid-cols-12 px-3 py-2 text-sm">
                    <div className="col-span-4 min-w-0">
                      <div className="font-medium text-gray-900 truncate">{s.name}</div>
                      <div className="text-xs text-gray-500">{s.source_type}{s.is_syncing ? ' · syncing' : ''}</div>
                    </div>
                    <div className="col-span-2 text-right text-gray-700">{s.docs_total}</div>
                    <div className="col-span-2 text-right text-gray-700">{s.docs_processed}</div>
                    <div className="col-span-2 text-right text-gray-700">{s.docs_pending}</div>
                    <div className="col-span-2 text-right text-gray-700">{s.docs_failed}</div>
                  </div>
                ))}
                {(ingestionStatus.sources || []).length === 0 ? (
                  <div className="px-3 py-3 text-sm text-gray-500">No sources configured.</div>
                ) : null}
              </div>
            </div>

            {(ingestionStatus.recent_document_errors || []).length > 0 ? (
              <details className="border border-gray-200 rounded-lg p-3">
                <summary className="cursor-pointer text-sm font-medium text-gray-900">
                  Recent document processing errors ({ingestionStatus.recent_document_errors.length})
                </summary>
                <div className="mt-3 space-y-2">
                  {ingestionStatus.recent_document_errors.slice(0, 10).map((e: any) => (
                    <div key={e.document_id} className="text-xs bg-red-50 border border-red-200 rounded p-2">
                      <div className="font-mono text-red-900">{String(e.document_id).slice(0, 8)}…</div>
                      <div className="text-red-800">{e.title || 'Untitled'}</div>
                      <div className="text-red-700 whitespace-pre-wrap">{e.error}</div>
                    </div>
                  ))}
                </div>
              </details>
            ) : null}
          </div>
        ) : (
          <div className="mt-4 text-sm text-gray-600">No ingestion status available.</div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Process GitHub Repositories</h3>
            <p className="text-sm text-gray-600">Provide a token and repositories to ingest code, issues, or wiki content.</p>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">Source Name</label>
            <input
              type="text"
              className="w-full border rounded px-3 py-2 text-sm"
              placeholder="e.g. GitHub Engineering"
              value={githubForm.name}
              onChange={(e) => setGithubForm(prev => ({ ...prev, name: e.target.value }))}
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">GitHub Token</label>
            <input
              type="password"
              className="w-full border rounded px-3 py-2 text-sm"
              placeholder="ghp_xxxxxxxxx"
              value={githubForm.token}
              onChange={(e) => setGithubForm(prev => ({ ...prev, token: e.target.value }))}
            />
            <p className="text-xs text-gray-500 mt-1">Token requires repo read permissions.</p>
          </div>
        </div>
        <div className="mt-4">
          <label className="text-sm font-medium text-gray-700 block mb-1">Repositories</label>
          <textarea
            className="w-full border rounded px-3 py-2 text-sm"
            rows={3}
            placeholder="owner/repo-one&#10;owner/repo-two"
            value={githubForm.repos}
            onChange={(e) => setGithubForm(prev => ({ ...prev, repos: e.target.value }))}
          />
          <p className="text-xs text-gray-500 mt-1">Comma or newline separated list of repositories.</p>
        </div>
        <div className="grid gap-4 md:grid-cols-3 mt-4">
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">Allowed File Extensions</label>
            <input
              type="text"
              className="w-full border rounded px-3 py-2 text-sm"
              value={githubForm.fileExtensions}
              onChange={(e) => setGithubForm(prev => ({ ...prev, fileExtensions: e.target.value }))}
            />
            <p className="text-xs text-gray-500 mt-1">Comma separated, include dot (e.g. .md,.py).</p>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">Max Pages / API Calls</label>
            <input
              type="number"
              min={1}
              className="w-full border rounded px-3 py-2 text-sm"
              value={githubForm.maxPages}
              onChange={(e) => {
                const val = parseInt(e.target.value || '0', 10);
                setGithubForm(prev => ({ ...prev, maxPages: Number.isNaN(val) ? 10 : val }));
              }}
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">GitHub API Base (optional)</label>
            <input
              type="text"
              className="w-full border rounded px-3 py-2 text-sm"
              placeholder="https://api.github.com"
              value={githubForm.apiBase}
              onChange={(e) => setGithubForm(prev => ({ ...prev, apiBase: e.target.value }))}
            />
          </div>
        </div>
        <div className="mt-4">
          <label className="text-sm font-medium text-gray-700 block mb-1">Ignore Patterns (one glob per line)</label>
          <textarea
            className="w-full border rounded px-3 py-2 text-sm"
            rows={2}
            placeholder="**/node_modules/**"
            value={githubForm.ignoreGlobs}
            onChange={(e) => setGithubForm(prev => ({ ...prev, ignoreGlobs: e.target.value }))}
          />
        </div>
        <div className="mt-4 grid gap-2 md:grid-cols-2">
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.includeFiles}
              onChange={(e) => setGithubForm(prev => ({ ...prev, includeFiles: e.target.checked }))}
            />
            Include repository files
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.includeIssues}
              onChange={(e) => setGithubForm(prev => ({ ...prev, includeIssues: e.target.checked }))}
            />
            Include issues
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.includePRs}
              onChange={(e) => setGithubForm(prev => ({ ...prev, includePRs: e.target.checked }))}
            />
            Include pull requests
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.includeWiki}
              onChange={(e) => setGithubForm(prev => ({ ...prev, includeWiki: e.target.checked }))}
            />
            Include wiki pages
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.incrementalFiles}
              onChange={(e) => setGithubForm(prev => ({ ...prev, incrementalFiles: e.target.checked }))}
            />
            Incremental file sync
          </label>
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.useGitignore}
              onChange={(e) => setGithubForm(prev => ({ ...prev, useGitignore: e.target.checked }))}
            />
            Merge repository .gitignore
          </label>
        </div>
        <div className="flex items-center justify-between mt-6 flex-wrap gap-4">
          <label className="inline-flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={githubForm.startImmediately}
              onChange={(e) => setGithubForm(prev => ({ ...prev, startImmediately: e.target.checked }))}
            />
            Start sync immediately after creating the source
          </label>
          <Button
            onClick={handleCreateGithubSource}
            loading={githubSourceMutation.isLoading}
            disabled={!githubForm.token.trim() || !githubForm.repos.trim()}
          >
            Create GitHub Source
          </Button>
        </div>
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
                        {source.is_syncing && (
                          <span className="ml-2 inline-flex items-center px-2 py-0.5 text-xs font-medium rounded bg-blue-100 text-blue-800">
                            Syncing
                          </span>
                        )}
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
                        <span className="font-medium">Next Run:</span>
                        <span className="ml-1">
                          {(() => {
                            const m = nextRuns?.items?.find((it: any) => it.source_id === source.id);
                            if (!m || !m.next_run) return '—';
                            try { const d = new Date(m.next_run as string); return (<span title={d.toLocaleString()}>{formatDistanceToNow(d)} from now</span>); } catch { return m.next_run; }
                          })()}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Next Run:</span>
                        <span className="ml-1">
                          {(() => {
                            const m = nextRuns?.items?.find(it => it.source_id === source.id);
                            if (!m || !m.next_run) return '—';
                            try { return formatDistanceToNow(new Date(m.next_run)) + ' from now'; } catch { return m.next_run; }
                          })()}
                        </span>
                      </div>
                      <div>
                        <span className="font-medium">Created:</span>
                        <span className="ml-1">
                          {formatDistanceToNow(new Date(source.created_at))} ago
                        </span>
                      </div>
                      {/* Auto Sync and Incremental toggles */}
                      <div className="flex items-center gap-3 mt-1">
                        <label className="inline-flex items-center gap-1">
                          <input
                            type="checkbox"
                            checked={Boolean((source as any).config?.auto_sync)}
                            onChange={async (e) => {
                              const updated = { ...source, config: { ...(source as any).config, auto_sync: e.target.checked } } as any;
                              try {
                                await apiClient.updateDocumentSource(source.id, { name: source.name, source_type: source.source_type, config: updated.config });
                                toast.success('Updated auto sync');
                                queryClient.invalidateQueries('documentSources');
                              } catch { toast.error('Failed to update'); }
                            }}
                          />
                          <span>Auto Sync</span>
                        </label>
                        <label className="inline-flex items-center gap-1">
                          <input
                            type="checkbox"
                            checked={Boolean((source as any).config?.sync_only_changed ?? true)}
                            onChange={async (e) => {
                              const val = e.target.checked;
                              const cfg = { ...(source as any).config, sync_only_changed: val } as any;
                              // Also update connector-specific incremental flag when present
                              cfg.incremental_files = val;
                              try {
                                await apiClient.updateDocumentSource(source.id, { name: source.name, source_type: source.source_type, config: cfg });
                                toast.success('Updated sync mode');
                                queryClient.invalidateQueries('documentSources');
                              } catch { toast.error('Failed to update'); }
                            }}
                          />
                          <span>Sync Only Changed</span>
                        </label>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <label className="text-sm">Interval (min):</label>
                        <input
                          className="w-20 border rounded px-2 py-1 text-sm"
                          type="number"
                          min={0}
                          defaultValue={(source as any).config?.sync_interval_minutes ?? 0}
                          onBlur={async (e) => {
                            const val = parseInt(e.target.value || '0', 10) || 0;
                            const cfg = { ...(source as any).config, sync_interval_minutes: val };
                            try {
                              await apiClient.updateDocumentSource(source.id, { name: source.name, source_type: source.source_type, config: cfg });
                              toast.success('Updated interval');
                              queryClient.invalidateQueries('documentSources');
                            } catch { toast.error('Failed to update'); }
                          }}
                        />
                        <label className="text-sm ml-4">Cron:</label>
                        <input
                          className="w-48 border rounded px-2 py-1 text-sm"
                          type="text"
                          placeholder="e.g. 0 2 * * *"
                          defaultValue={(source as any).config?.cron || ''}
                          onBlur={async (e) => {
                            const val = e.target.value.trim();
                            if (!val) {
                              const cfg = { ...(source as any).config, cron: undefined };
                              try { await apiClient.updateDocumentSource(source.id, { name: source.name, source_type: source.source_type, config: cfg }); toast.success('Cleared cron'); queryClient.invalidateQueries('documentSources'); queryClient.invalidateQueries('sourcesNextRun'); } catch { toast.error('Failed to update'); }
                              return;
                            }
                            try {
                              const v = await apiClient.validateCron(val);
                              if (!v.valid) { toast.error('Invalid cron: ' + (v.error || '')); return; }
                            } catch { toast.error('Invalid cron'); return; }
                            const cfg = { ...(source as any).config, cron: val };
                            try {
                              await apiClient.updateDocumentSource(source.id, { name: source.name, source_type: source.source_type, config: cfg });
                              toast.success('Updated cron');
                              queryClient.invalidateQueries('documentSources');
                              queryClient.invalidateQueries('sourcesNextRun');
                            } catch { toast.error('Failed to update'); }
                          }}
                        />
                      </div>
                      {(source as any)?.config?.sync_only_changed === false && (
                        <div className="mt-1 text-xs inline-flex items-center px-2 py-0.5 rounded bg-yellow-100 text-yellow-800">Full sync mode</div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <SyncButtonWithConfirm source={source} onSync={(forceFull) => syncSourceMutation.mutate({ id: source.id, forceFull })} loading={syncSourceMutation.isLoading} />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => dryRunMutation.mutate({ id: source.id })}
                    icon={<Play className="w-4 h-4" />}
                  >
                    Dry Run
                  </Button>
                  {(source as any)?.is_syncing && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={async () => {
                        try { const res = await apiClient.cancelSourceSync(source.id); toast.success('Cancel requested'); } catch { toast.error('Failed to request cancel'); }
                      }}
                    >
                      Cancel
                    </Button>
                  )}
                  {dryRunCache[source.id] && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => { setDryRunResult(dryRunCache[source.id]); setDryRunOpen(true); }}
                    >
                      View Sample
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try { const res = await apiClient.getSourceSyncLogs(source.id, 20); setHistoryItems(res.items || []); setHistoryOpen(true); } catch { toast.error('Failed to load history'); }
                    }}
                  >
                    History
                  </Button>
                  {source.last_error && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => syncSourceMutation.mutate({ id: source.id })}
                    icon={<RefreshCw className="w-4 h-4" />}
                  >
                    Retry
                  </Button>
                  )}
                  {source.last_error && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => clearErrorMutation.mutate(source.id)}
                    >
                      Clear Error
                    </Button>
                  )}
                </div>
              </div>

              {/* Configuration Preview */}
              <div className="mt-4 p-3 bg-gray-50 rounded border">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Configuration</h4>
                <pre className="text-xs text-gray-600 overflow-auto">
                  {JSON.stringify(source.config, null, 2)}
                </pre>
              </div>

              {/* Last error (if any) */}
              {source.last_error && (
                <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded">
                  <div className="text-sm text-red-700 font-medium">Last Error</div>
                  <pre className="text-xs text-red-700 overflow-auto whitespace-pre-wrap mt-1">{source.last_error}</pre>
                </div>
              )}

              {/* Ingestion progress */}
              {ingProgress[source.id] && (
                <div className="mt-3">
                  <div className="flex justify-between text-sm text-gray-700 mb-1">
                    <span>{ingProgress[source.id].status || 'Syncing...'}</span>
                    <span>{ingProgress[source.id].progress ?? 0}%</span>
                  </div>
                  <ProgressBar value={ingProgress[source.id].progress ?? 0} />
                  <div className="flex items-center justify-between text-xs text-gray-500 mt-1">
                    <div>
                      {(ingProgress[source.id].current !== undefined) && (ingProgress[source.id].total !== undefined) && (
                        <span>{ingProgress[source.id].current}/{ingProgress[source.id].total}</span>
                      )}
                    </div>
                    <div>
                      {typeof ingProgress[source.id].remaining_formatted === 'string' && (
                        <span>ETA: {ingProgress[source.id].remaining_formatted}</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Dry-run results modal */}
      <AlertModal
        isOpen={dryRunOpen}
        onClose={() => setDryRunOpen(false)}
        title={dryRunResult?.source_name ? `Dry Run — ${dryRunResult.source_name}` : 'Dry Run Results'}
        message={(
          <div>
            {/* Toggles to refine dry run */}
            <div className="flex flex-wrap gap-4 mb-3 text-sm items-center">
              <label className="inline-flex items-center gap-1"><input type="checkbox" checked={dryRunOverrides.include_files} onChange={(e)=> setDryRunOverrides(prev => ({ ...prev, include_files: e.target.checked }))}/> Files</label>
              <label className="inline-flex items-center gap-1"><input type="checkbox" checked={dryRunOverrides.include_issues} onChange={(e)=> setDryRunOverrides(prev => ({ ...prev, include_issues: e.target.checked }))}/> Issues</label>
              <label className="inline-flex items-center gap-1"><input type="checkbox" checked={dryRunOverrides.include_prs} onChange={(e)=> setDryRunOverrides(prev => ({ ...prev, include_prs: e.target.checked }))}/> PRs</label>
              <label className="inline-flex items-center gap-1"><input type="checkbox" checked={dryRunOverrides.include_mrs} onChange={(e)=> setDryRunOverrides(prev => ({ ...prev, include_mrs: e.target.checked }))}/> MRs</label>
              <label className="inline-flex items-center gap-1"><input type="checkbox" checked={dryRunOverrides.include_wiki} onChange={(e)=> setDryRunOverrides(prev => ({ ...prev, include_wiki: e.target.checked }))}/> Wiki</label>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  const srcId = (dryRunResult as any)?.source_id as string | undefined;
                  if (!srcId) return;
                  dryRunMutation.mutate({ id: srcId, overrides: {
                    include_files: dryRunOverrides.include_files,
                    include_issues: dryRunOverrides.include_issues,
                    include_pull_requests: dryRunOverrides.include_prs,
                    include_merge_requests: dryRunOverrides.include_mrs,
                    include_wiki: dryRunOverrides.include_wiki,
                  }});
                }}
              >Refresh</Button>
            </div>
            <div className="text-sm text-gray-700 mb-3">
              <div>Total candidates: <span className="font-medium">{dryRunResult?.total ?? 0}</span></div>
              <div>Estimated new: <span className="font-medium">{dryRunResult?.estimated_new ?? 0}</span></div>
              <div>Estimated existing: <span className="font-medium">{dryRunResult?.estimated_existing ?? 0}</span></div>
              <div>Mode: <span className="font-medium">{dryRunResult?.mode || 'full'}</span></div>
              {dryRunResult?.by_type && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {Object.entries(dryRunResult.by_type as any).map(([k,v]) => {
                    const label = ((): string => {
                      const t = (k || '').toString().toLowerCase();
                      if (t.includes('repository') || t.includes('file')) return 'Files';
                      if (t === 'issue') return 'Issues';
                      if (t === 'pull_request' || t === 'merge_request') return 'PRs/MRs';
                      if (t === 'wiki_file' || t === 'wiki_page') return 'Wiki';
                      return k as string;
                    })();
                    return (
                      <span key={k} className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-800">
                        <span className="font-medium mr-1">{label}</span>
                        <span>{v as any}</span>
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
            {dryRunResult?.sample && dryRunResult.sample.length > 0 ? (
              <div className="max-h-64 overflow-auto border rounded">
                <DryRunSampleTable items={dryRunResult.sample} />
              </div>
            ) : (
              <div className="text-sm text-gray-500">No sample available.</div>
            )}
          </div>
        )}
      />

      {/* Sync history modal */}
      <AlertModal
        isOpen={historyOpen}
        onClose={() => setHistoryOpen(false)}
        title="Recent Syncs"
        message={(
          <div className="max-h-80 overflow-auto">
            {!historyItems || historyItems.length === 0 ? (
              <div className="text-sm text-gray-500">No sync history.</div>
            ) : (
              <table className="min-w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="text-left px-3 py-2">Started</th>
                    <th className="text-left px-3 py-2">Status</th>
                    <th className="text-left px-3 py-2">Processed</th>
                    <th className="text-left px-3 py-2">Created</th>
                    <th className="text-left px-3 py-2">Updated</th>
                    <th className="text-left px-3 py-2">Errors</th>
                    <th className="text-left px-3 py-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {historyItems.map((it, idx) => (
                    <React.Fragment key={idx}>
                      <tr className="border-t">
                        <td className="px-3 py-2 text-gray-700">{it.started_at ? formatDistanceToNow(new Date(it.started_at)) + ' ago' : '—'}</td>
                        <td className="px-3 py-2">
                          <span className={`inline-flex px-2 py-0.5 rounded text-xs ${it.status === 'success' ? 'bg-green-100 text-green-800' : it.status === 'failed' ? 'bg-red-100 text-red-800' : it.status === 'canceled' ? 'bg-yellow-100 text-yellow-800' : 'bg-blue-100 text-blue-800'}`}>
                            {it.status}
                          </span>
                        </td>
                        <td className="px-3 py-2">{it.processed ?? '—'}/{it.total_documents ?? '—'}</td>
                        <td className="px-3 py-2">{it.created ?? 0}</td>
                        <td className="px-3 py-2">{it.updated ?? 0}</td>
                        <td className="px-3 py-2">{it.errors ?? 0}</td>
                        <td className="px-3 py-2">
                          {it.status === 'failed' && (
                            <div className="flex gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={async () => {
                                  try {
                                    await navigator.clipboard.writeText(it.error_message || '');
                                    toast.success('Error copied');
                                  } catch {
                                    toast.error('Copy failed');
                                  }
                                }}
                              >
                                Copy error
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={async () => {
                                  if (!historySourceId) return;
                                  try {
                                    await apiClient.triggerSourceSync(historySourceId, { forceFull: true });
                                    toast.success('Re-run full sync started');
                                  } catch {
                                    toast.error('Failed to start full sync');
                                  }
                                }}
                              >
                                Re-run full sync
                              </Button>
                            </div>
                          )}
                        </td>
                      </tr>
                      {it.error_message && (
                        <tr className="border-b">
                          <td className="px-3 py-2 text-xs text-red-700" colSpan={6}>{it.error_message}</td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            )}
            {historySourceId && (
              <div className="mt-2 flex justify-between items-center">
                <div className="text-xs text-gray-500">Showing {historyItems?.length || 0}</div>
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        const newOffset = Math.max(0, (historyOffset ?? 0) - (historyLimit ?? 20));
                        setHistoryOffset(newOffset);
                        const res = await apiClient.getSourceSyncLogs(historySourceId, historyLimit ?? 20, newOffset);
                        setHistoryItems(res.items || []);
                      } catch { toast.error('Failed'); }
                    }}
                  >
                    Prev
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        const newOffset = (historyOffset ?? 0) + (historyLimit ?? 20);
                        setHistoryOffset(newOffset);
                        const res = await apiClient.getSourceSyncLogs(historySourceId, historyLimit ?? 20, newOffset);
                        setHistoryItems(res.items || []);
                      } catch { toast.error('Failed'); }
                    }}
                  >
                    Next
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={async () => {
                      try {
                        const blob = await apiClient.exportSourceSyncLogsCSV(historySourceId, 1000, 0);
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `sync_logs_${historySourceId}.csv`;
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        URL.revokeObjectURL(url);
                      } catch { toast.error('Export failed'); }
                    }}
                  >
                    Export CSV
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      />
    </div>
  );
};

// Sync button component with confirmation for full sync
const SyncButtonWithConfirm: React.FC<{ source: any; onSync: (forceFull: boolean) => void; loading?: boolean }>
  = ({ source, onSync, loading }) => {
  const [open, setOpen] = React.useState(false);
  const cfg = (source as any)?.config || {};
  const syncOnlyChanged = Boolean(cfg.sync_only_changed ?? true);
  const handleClick = () => {
    if (!syncOnlyChanged) {
      setOpen(true);
    } else {
      onSync(false);
    }
  };
  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={handleClick}
        loading={loading}
        icon={<Play className="w-4 h-4" />}
      >
        Sync
      </Button>
      <ConfirmationModal
        isOpen={open}
        onClose={() => setOpen(false)}
        title="Run Full Sync?"
        message="This will re-index all items for this source. It may take a long time and use significant resources. Proceed?"
        confirmText="Run Full Sync"
        cancelText="Cancel"
        onConfirm={() => { setOpen(false); onSync(true); }}
      />
    </>
  );
};

// Dry Run Sample Table with filter
const DryRunSampleTable: React.FC<{ items: Array<{ title?: string; identifier?: string; type?: string }> }>
  = ({ items }) => {
  const [filter, setFilter] = React.useState<'all' | 'files' | 'issues' | 'prs' | 'wiki'>('all');
  const filtered = React.useMemo(() => {
    if (!items) return [] as any[];
    if (filter === 'all') return items;
    return items.filter((it) => {
      const t = (it.type || '').toLowerCase();
      if (filter === 'files') return t.includes('file');
      if (filter === 'issues') return t === 'issue';
      if (filter === 'prs') return t === 'pull_request' || t === 'merge_request';
      if (filter === 'wiki') return t === 'wiki_file' || t === 'wiki_page';
      return true;
    });
  }, [items, filter]);

  return (
    <div className="w-full">
      <div className="flex items-center justify-between p-2 bg-gray-50 border-b">
        <div className="text-xs text-gray-600">Filter:</div>
        <select
          className="text-xs border rounded px-2 py-1"
          value={filter}
          onChange={(e) => setFilter(e.target.value as any)}
        >
          <option value="all">All</option>
          <option value="files">Files</option>
          <option value="issues">Issues</option>
          <option value="prs">PRs/MRs</option>
          <option value="wiki">Wiki</option>
        </select>
      </div>
      <table className="min-w-full text-xs">
        <thead className="bg-gray-50 sticky top-0">
          <tr>
            <th className="text-left px-3 py-2 font-medium text-gray-700">Title</th>
            <th className="text-left px-3 py-2 font-medium text-gray-700">Type</th>
            <th className="text-left px-3 py-2 font-medium text-gray-700">Identifier</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((it, idx) => (
            <tr key={idx} className="border-t">
              <td className="px-3 py-2 text-gray-800">{it.title || '–'}</td>
              <td className="px-3 py-2 text-gray-600">{it.type || '–'}</td>
              <td className="px-3 py-2 text-gray-500 break-all">{it.identifier || '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Tasks Tab
const TasksTab: React.FC = () => {
  const queryClient = useQueryClient();
  const summarizeAllMutation = useMutation(
    (limit: number) => apiClient.summarizeMissingDocuments(limit),
    {
      onSuccess: (res) => {
        toast.success(`Queued ${res.queued} summaries`);
      },
      onError: (e: any) => {
        toast.error(e?.response?.data?.detail || e?.message || 'Failed to queue summaries');
      }
    }
  );
  const { data: taskStatus, isLoading } = useQuery(
    'taskStatus',
    () => apiClient.getTaskStatus(),
    {
      refetchInterval: 5000, // Refresh every 5 seconds
      refetchOnWindowFocus: false,
    }
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900">Background Tasks</h2>
        <div className="flex items-center gap-2">
          <input id="sum-limit" type="number" min={1} max={5000} defaultValue={500} className="w-24 border rounded px-2 py-1 text-sm" />
          <Button
            onClick={() => {
              const input = document.getElementById('sum-limit') as HTMLInputElement | null;
              const val = input ? parseInt(input.value || '500', 10) : 500;
              summarizeAllMutation.mutate(isNaN(val) ? 500 : val);
            }}
            loading={summarizeAllMutation.isLoading}
            icon={<Play className="w-4 h-4" />}
          >
            Summarize Missing
          </Button>
        </div>
      </div>

      {isLoading ? (
        <LoadingSpinner className="h-32" text="Loading task status..." />
      ) : taskStatus ? (
        <div className="space-y-4">
          {/* Active Tasks */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Active Tasks</h3>
            {taskStatus.active_tasks && typeof taskStatus.active_tasks === 'object' && Object.keys(taskStatus.active_tasks).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(taskStatus.active_tasks).map(([worker, tasks]) => (
                  <div key={worker}>
                    <h4 className="font-medium text-gray-700">{worker}</h4>
                    {Array.isArray(tasks) && tasks.length > 0 ? (
                      tasks.map((task: any, index: number) => (
                        <div key={index} className="ml-4 text-sm text-gray-600">
                          {task.name || task.task || 'Unknown task'} - {task.id || 'N/A'}
                        </div>
                      ))
                    ) : (
                      <p className="ml-4 text-sm text-gray-500">No tasks for this worker</p>
                    )}
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
            {taskStatus.scheduled_tasks && typeof taskStatus.scheduled_tasks === 'object' && Object.keys(taskStatus.scheduled_tasks).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(taskStatus.scheduled_tasks).map(([worker, tasks]) => (
                  <div key={worker}>
                    <h4 className="font-medium text-gray-700">{worker}</h4>
                    {Array.isArray(tasks) && tasks.length > 0 ? (
                      tasks.map((task: any, index: number) => (
                        <div key={index} className="ml-4 text-sm text-gray-600">
                          {task.name || task.task || 'Unknown task'} - {task.id || 'N/A'}
                          {task.eta && <span className="ml-2 text-gray-500">(ETA: {new Date(task.eta).toLocaleString()})</span>}
                        </div>
                      ))
                    ) : (
                      <p className="ml-4 text-sm text-gray-500">No scheduled tasks for this worker</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No scheduled tasks</p>
            )}
          </div>

          {/* Reserved Tasks */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Reserved Tasks</h3>
            {taskStatus.reserved_tasks && typeof taskStatus.reserved_tasks === 'object' && Object.keys(taskStatus.reserved_tasks).length > 0 ? (
              <div className="space-y-2">
                {Object.entries(taskStatus.reserved_tasks).map(([worker, tasks]) => (
                  <div key={worker}>
                    <h4 className="font-medium text-gray-700">{worker}</h4>
                    {Array.isArray(tasks) && tasks.length > 0 ? (
                      tasks.map((task: any, index: number) => (
                        <div key={index} className="ml-4 text-sm text-gray-600">
                          {task.name || task.task || 'Unknown task'} - {task.id || 'N/A'}
                        </div>
                      ))
                    ) : (
                      <p className="ml-4 text-sm text-gray-500">No reserved tasks for this worker</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No reserved tasks</p>
            )}
          </div>

          {/* Ingestion Tasks */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Ingestion Tasks</h3>
            {taskStatus.ingestion_tasks && Array.isArray(taskStatus.ingestion_tasks) && taskStatus.ingestion_tasks.length > 0 ? (
              <div className="space-y-2">
                {taskStatus.ingestion_tasks.map((task: any, index: number) => (
                  <div key={index} className="p-3 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-gray-900">{task.source_name}</div>
                        <div className="text-sm text-gray-600">
                          Type: {task.source_type} | Status: <span className={`font-medium ${task.status === 'syncing' ? 'text-blue-600' : 'text-yellow-600'}`}>{task.status}</span>
                        </div>
                        {task.task_id && (
                          <div className="text-xs text-gray-500 mt-1">Task ID: {task.task_id}</div>
                        )}
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        task.status === 'syncing' ? 'bg-blue-100 text-blue-800' : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {task.status === 'syncing' ? 'Running' : 'Pending'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No active ingestion tasks</p>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <p className="text-yellow-800">Unable to load task status. Please check if Celery workers are running.</p>
        </div>
      )}
    </div>
  );
};

// Personas Tab
interface PersonaFormValues {
  name: string;
  platform_id?: string | null;
  user_id?: string | null;
  description?: string | null;
  avatar_url?: string | null;
  is_active: boolean;
  is_system: boolean;
}

interface PersonasTabProps {
  focusPersonaId?: string;
  onFocusPersonaHandled?: () => void;
}

const PersonasTab: React.FC<PersonasTabProps> = ({ focusPersonaId, onFocusPersonaHandled }) => {
  const queryClient = useQueryClient();
  const [searchInput, setSearchInput] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [page, setPage] = useState(1);
  const pageSize = 20;
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [currentPersona, setCurrentPersona] = useState<Persona | null>(null);
  const [personaToDelete, setPersonaToDelete] = useState<Persona | null>(null);
  const [highlightPersonaId, setHighlightPersonaId] = useState<string | null>(null);
  const [pendingSearchOverride, setPendingSearchOverride] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim());
    }, 400);
    return () => clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    setPage(1);
  }, [debouncedSearch]);

  useEffect(() => {
    if (pendingSearchOverride) {
      setSearchInput(pendingSearchOverride);
      setDebouncedSearch(pendingSearchOverride.trim());
      setPendingSearchOverride(null);
      setPage(1);
    }
  }, [pendingSearchOverride]);

  const { data, isLoading } = useQuery(
    ['personas', debouncedSearch, page],
    () =>
      apiClient.listPersonas({
        search: debouncedSearch || undefined,
        page,
        page_size: pageSize,
        include_inactive: true,
      }),
    { keepPreviousData: true }
  );

  const personas = useMemo(() => data?.items || [], [data]);
  const total = data?.total || 0;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const createPersonaMutation = useMutation(
    (payload: PersonaFormValues) =>
      apiClient.createPersona({
        name: payload.name,
        platform_id: payload.platform_id,
        user_id: payload.user_id,
        description: payload.description,
        avatar_url: payload.avatar_url,
        is_active: payload.is_active,
        is_system: payload.is_system,
      }),
    {
      onSuccess: () => {
        toast.success('Persona created');
        setIsFormOpen(false);
        queryClient.invalidateQueries('personas');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to create persona';
        toast.error(message);
      },
    }
  );

  const updatePersonaMutation = useMutation(
    ({ id, values }: { id: string; values: PersonaFormValues }) =>
      apiClient.updatePersona(id, {
        name: values.name,
        platform_id: values.platform_id,
        user_id: values.user_id,
        description: values.description,
        avatar_url: values.avatar_url,
        is_active: values.is_active,
        is_system: values.is_system,
      }),
    {
      onSuccess: () => {
        toast.success('Persona updated');
        setIsFormOpen(false);
        queryClient.invalidateQueries('personas');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to update persona';
        toast.error(message);
      },
    }
  );

  const deletePersonaMutation = useMutation(
    (personaId: string) => apiClient.deletePersona(personaId),
    {
      onSuccess: () => {
        toast.success('Persona deleted');
        setPersonaToDelete(null);
        queryClient.invalidateQueries('personas');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to delete persona';
        toast.error(message);
      },
    }
  );

  const openCreateModal = () => {
    setCurrentPersona(null);
    setIsFormOpen(true);
  };

  const openEditModal = (persona: Persona) => {
    setCurrentPersona(persona);
    setIsFormOpen(true);
  };

  const handlePersonaSubmit = (values: PersonaFormValues) => {
    if (currentPersona) {
      updatePersonaMutation.mutate({ id: currentPersona.id, values });
    } else {
      createPersonaMutation.mutate(values);
    }
  };

  const confirmDelete = () => {
    if (personaToDelete) {
      deletePersonaMutation.mutate(personaToDelete.id);
    }
  };

  const isSaving = createPersonaMutation.isLoading || updatePersonaMutation.isLoading;
  useEffect(() => {
    let cancelled = false;
    if (focusPersonaId) {
      apiClient
        .getPersona(focusPersonaId)
        .then((persona) => {
          if (cancelled) return;
          setCurrentPersona(persona);
          setIsFormOpen(true);
          setHighlightPersonaId(persona.id);
          setPendingSearchOverride(persona.name);
        })
        .catch(() => {
          if (!cancelled) {
            toast.error('Persona not found');
          }
        })
        .finally(() => {
          if (!cancelled) {
            onFocusPersonaHandled?.();
          }
        });
    }
    return () => {
      cancelled = true;
    };
  }, [focusPersonaId, onFocusPersonaHandled]);

  useEffect(() => {
    if (!highlightPersonaId) return;
    const row = document.getElementById(`persona-row-${highlightPersonaId}`);
    if (row) {
      row.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [highlightPersonaId, personas]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex-1">
          <Input
            label="Search personas"
            value={searchInput}
            onChange={(e) => {
              if (highlightPersonaId) {
                setHighlightPersonaId(null);
              }
              setSearchInput(e.target.value);
            }}
            placeholder="Search by name or platform identifier"
          />
        </div>
        <Button onClick={openCreateModal}>New Persona</Button>
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        {isLoading ? (
          <div className="p-6">
            <LoadingSpinner text="Loading personas..." />
          </div>
        ) : personas.length === 0 ? (
          <div className="p-6 text-center text-gray-600">
            {debouncedSearch ? 'No personas match your search.' : 'No personas found yet.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Platform ID</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Linked User</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Updated</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {personas.map((persona) => {
                  const isHighlighted = highlightPersonaId === persona.id;
                  return (
                    <tr
                      key={persona.id}
                      id={`persona-row-${persona.id}`}
                      className={isHighlighted ? 'bg-primary-50/70' : undefined}
                    >
                    <td className="px-4 py-3 text-sm text-gray-900">
                      <div className="flex flex-col">
                        <span className="font-medium flex items-center gap-2">
                          <UserCircle2 className="w-4 h-4 text-primary-600" />
                          {persona.name}
                        </span>
                        {persona.description && (
                          <span className="text-xs text-gray-500 truncate">{persona.description}</span>
                        )}
                        {persona.is_system && (
                          <span className="mt-1 inline-flex h-5 items-center rounded-full bg-blue-50 px-2 text-xs font-medium text-blue-700">
                            System
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">{persona.platform_id || '—'}</td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {persona.user_id ? (
                        <span className="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">{persona.user_id}</span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                          persona.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {persona.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {persona.updated_at ? `${formatDistanceToNow(new Date(persona.updated_at))} ago` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-sm space-x-2">
                      <Button variant="ghost" size="sm" onClick={() => openEditModal(persona)}>
                        Edit
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-600 hover:text-red-700"
                        onClick={() => setPersonaToDelete(persona)}
                      >
                        Delete
                      </Button>
                    </td>
                  </tr>
                );
                })}
              </tbody>
            </table>
          </div>
        )}

        {personas.length > 0 && (
          <div className="flex items-center justify-between px-4 py-3 text-sm text-gray-600 border-t">
            <span>
              Showing {Math.min((page - 1) * pageSize + 1, total)}-
              {Math.min(page * pageSize, total)} of {total}
            </span>
            <div className="space-x-2">
              <Button
                variant="ghost"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              >
                Previous
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </div>

      {isFormOpen && (
        <PersonaFormModal
          persona={currentPersona}
          isSaving={isSaving}
          onClose={() => setIsFormOpen(false)}
          onSubmit={handlePersonaSubmit}
        />
      )}

      <ConfirmationModal
        isOpen={!!personaToDelete}
        title="Delete Persona"
        message={`Delete persona "${personaToDelete?.name}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deletePersonaMutation.isLoading}
        onClose={() => setPersonaToDelete(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
};

interface PersonaFormModalProps {
  persona: Persona | null;
  onClose: () => void;
  onSubmit: (values: PersonaFormValues) => void;
  isSaving: boolean;
}

const PersonaFormModal: React.FC<PersonaFormModalProps> = ({ persona, onClose, onSubmit, isSaving }) => {
  const [formValues, setFormValues] = useState<PersonaFormValues>({
    name: persona?.name || '',
    platform_id: persona?.platform_id || '',
    user_id: persona?.user_id || '',
    description: persona?.description || '',
    avatar_url: persona?.avatar_url || '',
    is_active: persona?.is_active ?? true,
    is_system: persona?.is_system ?? false,
  });

  useEffect(() => {
    setFormValues({
      name: persona?.name || '',
      platform_id: persona?.platform_id || '',
      user_id: persona?.user_id || '',
      description: persona?.description || '',
      avatar_url: persona?.avatar_url || '',
      is_active: persona?.is_active ?? true,
      is_system: persona?.is_system ?? false,
    });
  }, [persona]);

  const handleChange = (field: keyof PersonaFormValues, value: any) => {
    setFormValues((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit({
      ...formValues,
      name: formValues.name.trim(),
      platform_id: formValues.platform_id?.trim() || null,
      user_id: formValues.user_id?.trim() || null,
      description: formValues.description?.trim() || null,
      avatar_url: formValues.avatar_url?.trim() || null,
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">{persona ? 'Edit Persona' : 'Create Persona'}</h3>
          <button className="text-gray-400 hover:text-gray-600" onClick={onClose}>
            <XCircle className="w-5 h-5" />
          </button>
        </div>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <Input
            label="Name"
            value={formValues.name}
            onChange={(e) => handleChange('name', e.target.value)}
            required
          />
          <Input
            label="Platform ID"
            value={formValues.platform_id || ''}
            onChange={(e) => handleChange('platform_id', e.target.value)}
            placeholder="Optional unique identifier"
          />
          <Input
            label="Linked User ID"
            value={formValues.user_id || ''}
            onChange={(e) => handleChange('user_id', e.target.value)}
            placeholder="Optional user UUID"
          />
          <Input
            label="Avatar URL"
            value={formValues.avatar_url || ''}
            onChange={(e) => handleChange('avatar_url', e.target.value)}
            placeholder="https://example.com/avatar.png"
          />
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
              rows={3}
              value={formValues.description || ''}
              onChange={(e) => handleChange('description', e.target.value)}
              placeholder="Optional description for this persona"
            />
          </div>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                checked={formValues.is_active}
                onChange={(e) => handleChange('is_active', e.target.checked)}
              />
              <span>Active</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                checked={formValues.is_system}
                onChange={(e) => handleChange('is_system', e.target.checked)}
              />
              <span>System persona</span>
            </label>
          </div>
          <div className="flex justify-end space-x-3 pt-2">
            <Button type="button" variant="ghost" onClick={onClose} disabled={isSaving}>
              Cancel
            </Button>
            <Button type="submit" loading={isSaving} disabled={formValues.name.trim().length === 0}>
              {persona ? 'Save Changes' : 'Create Persona'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

// ========================================
// Agents Tab
// ========================================

interface AgentFormValues {
  name: string;
  display_name: string;
  description: string | null;
  system_prompt: string;
  capabilities: string[];
  tool_whitelist: string[] | null;
  priority: number;
  is_active: boolean;
}

const AgentsTab: React.FC = () => {
  const queryClient = useQueryClient();
  const [searchInput, setSearchInput] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [currentAgent, setCurrentAgent] = useState<AgentDefinition | null>(null);
  const [agentToDelete, setAgentToDelete] = useState<AgentDefinitionSummary | null>(null);
  const [agentLoadingId, setAgentLoadingId] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim());
    }, 400);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const { data, isLoading } = useQuery(
    ['agentDefinitions', debouncedSearch],
    () => apiClient.listAgentDefinitions({ search: debouncedSearch || undefined, active_only: false }),
    { keepPreviousData: true }
  );

  const { data: capabilitiesData } = useQuery(
    ['agentCapabilities'],
    () => apiClient.listAgentCapabilities()
  );

  const { data: toolsData } = useQuery(
    ['agentTools'],
    () => apiClient.listAgentTools()
  );

  const agents = useMemo(() => data?.agents || [], [data]);
  const capabilities = useMemo(() => capabilitiesData?.capabilities || [], [capabilitiesData]);
  const availableTools = useMemo(() => (toolsData?.tools || []).map(t => t.name), [toolsData]);

  const loadAgentMutation = useMutation(
    (agentId: string) => apiClient.getAgentDefinition(agentId),
    {
      onMutate: (agentId) => setAgentLoadingId(agentId),
      onSuccess: (agent) => {
        setCurrentAgent(agent);
        setIsFormOpen(true);
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to load agent';
        toast.error(message);
      },
      onSettled: () => setAgentLoadingId(null),
    }
  );

  const createAgentMutation = useMutation(
    (payload: AgentDefinitionCreate) => apiClient.createAgentDefinition(payload),
    {
      onSuccess: () => {
        toast.success('Agent created');
        setIsFormOpen(false);
        queryClient.invalidateQueries('agentDefinitions');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to create agent';
        toast.error(message);
      },
    }
  );

  const updateAgentMutation = useMutation(
    ({ id, values }: { id: string; values: AgentDefinitionUpdate }) =>
      apiClient.updateAgentDefinition(id, values),
    {
      onSuccess: () => {
        toast.success('Agent updated');
        setIsFormOpen(false);
        queryClient.invalidateQueries('agentDefinitions');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to update agent';
        toast.error(message);
      },
    }
  );

  const deleteAgentMutation = useMutation(
    (agentId: string) => apiClient.deleteAgentDefinition(agentId),
    {
      onSuccess: () => {
        toast.success('Agent deleted');
        setAgentToDelete(null);
        queryClient.invalidateQueries('agentDefinitions');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to delete agent';
        toast.error(message);
      },
    }
  );

  const duplicateAgentMutation = useMutation(
    (agentId: string) => apiClient.duplicateAgentDefinition(agentId),
    {
      onSuccess: () => {
        toast.success('Agent duplicated');
        queryClient.invalidateQueries('agentDefinitions');
      },
      onError: (error: any) => {
        const message = error?.response?.data?.detail || error?.message || 'Failed to duplicate agent';
        toast.error(message);
      },
    }
  );

  const openCreateModal = () => {
    setCurrentAgent(null);
    setIsFormOpen(true);
  };

  const openEditModal = (agent: AgentDefinitionSummary) => {
    loadAgentMutation.mutate(agent.id);
  };

  const handleAgentSubmit = (values: AgentFormValues) => {
    if (currentAgent) {
      if (currentAgent.is_system) {
        updateAgentMutation.mutate({
          id: currentAgent.id,
          values: {
            priority: values.priority,
            is_active: values.is_active,
          },
        });
        return;
      }

      updateAgentMutation.mutate({
        id: currentAgent.id,
        values: {
          display_name: values.display_name,
          description: values.description,
          system_prompt: values.system_prompt,
          capabilities: values.capabilities,
          tool_whitelist: values.tool_whitelist,
          priority: values.priority,
          is_active: values.is_active,
        },
      });
    } else {
      createAgentMutation.mutate({
        name: values.name,
        display_name: values.display_name,
        description: values.description,
        system_prompt: values.system_prompt,
        capabilities: values.capabilities,
        tool_whitelist: values.tool_whitelist,
        priority: values.priority,
        is_active: values.is_active,
      });
    }
  };

  const confirmDelete = () => {
    if (agentToDelete) {
      deleteAgentMutation.mutate(agentToDelete.id);
    }
  };

  const isSaving = createAgentMutation.isLoading || updateAgentMutation.isLoading;

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="flex-1">
          <Input
            label="Search agents"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Search by name or display name"
          />
        </div>
        <Button onClick={openCreateModal}>New Agent</Button>
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        {isLoading ? (
          <div className="p-6">
            <LoadingSpinner text="Loading agents..." />
          </div>
        ) : agents.length === 0 ? (
          <div className="p-6 text-center text-gray-600">
            {debouncedSearch ? 'No agents match your search.' : 'No agents found yet.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Capabilities</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Priority</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Updated</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {agents.map((agent) => (
                  <tr key={agent.id}>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      <div className="flex flex-col">
                        <span className="font-medium flex items-center gap-2">
                          <Bot className="w-4 h-4 text-primary-600" />
                          {agent.display_name}
                        </span>
                        <span className="text-xs text-gray-500 font-mono">{agent.name}</span>
                        {agent.description && (
                          <span className="text-xs text-gray-500 truncate max-w-xs">{agent.description}</span>
                        )}
                        {agent.is_system && (
                          <span className="mt-1 inline-flex h-5 items-center rounded-full bg-blue-50 px-2 text-xs font-medium text-blue-700 w-fit">
                            System
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {agent.capabilities.slice(0, 3).map((cap) => (
                          <span
                            key={cap}
                            className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600"
                          >
                            {cap}
                          </span>
                        ))}
                        {agent.capabilities.length > 3 && (
                          <span className="text-xs text-gray-400">+{agent.capabilities.length - 3}</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-100 text-primary-700 font-medium text-sm">
                        {agent.priority}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
                          agent.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                        }`}
                      >
                        {agent.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {agent.updated_at ? `${formatDistanceToNow(new Date(agent.updated_at))} ago` : '—'}
                    </td>
                    <td className="px-4 py-3 text-right text-sm space-x-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => openEditModal(agent)}
                        loading={agentLoadingId === agent.id}
                        disabled={!!agentLoadingId && agentLoadingId !== agent.id}
                      >
                        Edit
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => duplicateAgentMutation.mutate(agent.id)}
                        disabled={duplicateAgentMutation.isLoading}
                        title="Duplicate agent"
                      >
                        <Copy className="w-4 h-4" />
                      </Button>
                      {!agent.is_system && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 hover:text-red-700"
                          onClick={() => setAgentToDelete(agent)}
                        >
                          Delete
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {isFormOpen && (
        <AgentFormModal
          agent={currentAgent}
          capabilities={capabilities}
          availableTools={availableTools}
          isSaving={isSaving}
          onClose={() => {
            setIsFormOpen(false);
            setCurrentAgent(null);
          }}
          onSubmit={handleAgentSubmit}
        />
      )}

      <ConfirmationModal
        isOpen={!!agentToDelete}
        title="Delete Agent"
        message={`Delete agent "${agentToDelete?.display_name}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        variant="danger"
        isLoading={deleteAgentMutation.isLoading}
        onClose={() => setAgentToDelete(null)}
        onConfirm={confirmDelete}
      />
    </div>
  );
};

interface AgentFormModalProps {
  agent: AgentDefinition | null;
  capabilities: CapabilityInfo[];
  availableTools: string[];
  onClose: () => void;
  onSubmit: (values: AgentFormValues) => void;
  isSaving: boolean;
}

const AgentFormModal: React.FC<AgentFormModalProps> = ({
  agent,
  capabilities,
  availableTools,
  onClose,
  onSubmit,
  isSaving,
}) => {
  const [formValues, setFormValues] = useState<AgentFormValues>({
    name: agent?.name || '',
    display_name: agent?.display_name || '',
    description: agent?.description || null,
    system_prompt: agent?.system_prompt || '',
    capabilities: agent?.capabilities || [],
    tool_whitelist: agent?.tool_whitelist || null,
    priority: agent?.priority ?? 50,
    is_active: agent?.is_active ?? true,
  });

  const [useAllTools, setUseAllTools] = useState(agent?.tool_whitelist === null);

  useEffect(() => {
    setFormValues({
      name: agent?.name || '',
      display_name: agent?.display_name || '',
      description: agent?.description || null,
      system_prompt: agent?.system_prompt || '',
      capabilities: agent?.capabilities || [],
      tool_whitelist: agent?.tool_whitelist || null,
      priority: agent?.priority ?? 50,
      is_active: agent?.is_active ?? true,
    });
    setUseAllTools(agent?.tool_whitelist === null);
  }, [agent]);

  const handleChange = (field: keyof AgentFormValues, value: any) => {
    setFormValues((prev) => ({ ...prev, [field]: value }));
  };

  const toggleCapability = (capName: string) => {
    setFormValues((prev) => ({
      ...prev,
      capabilities: prev.capabilities.includes(capName)
        ? prev.capabilities.filter((c) => c !== capName)
        : [...prev.capabilities, capName],
    }));
  };

  const toggleTool = (toolName: string) => {
    const current = formValues.tool_whitelist || [];
    setFormValues((prev) => ({
      ...prev,
      tool_whitelist: current.includes(toolName)
        ? current.filter((t) => t !== toolName)
        : [...current, toolName],
    }));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit({
      ...formValues,
      name: formValues.name.trim(),
      display_name: formValues.display_name.trim(),
      description: formValues.description?.trim() || null,
      system_prompt: formValues.system_prompt.trim(),
      tool_whitelist: useAllTools ? null : formValues.tool_whitelist,
    });
  };

  const isSystem = agent?.is_system ?? false;
  const requiresPrompt = !agent || !isSystem;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            {agent ? 'Edit Agent' : 'Create Agent'}
            {isSystem && (
              <span className="ml-2 text-sm font-normal text-blue-600">(System Agent - Limited Editing)</span>
            )}
          </h3>
          <button className="text-gray-400 hover:text-gray-600" onClick={onClose}>
            <XCircle className="w-5 h-5" />
          </button>
        </div>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Name (slug)"
              value={formValues.name}
              onChange={(e) => handleChange('name', e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_'))}
              placeholder="my_agent"
              required
              disabled={!!agent}
              pattern="^[a-z][a-z0-9_]*$"
              title="Lowercase letters, numbers, and underscores only. Must start with a letter."
            />
            <Input
              label="Display Name"
              value={formValues.display_name}
              onChange={(e) => handleChange('display_name', e.target.value)}
              placeholder="My Agent"
              required
              disabled={isSystem}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
              rows={2}
              value={formValues.description || ''}
              onChange={(e) => handleChange('description', e.target.value)}
              placeholder="Brief description of what this agent does"
              disabled={isSystem}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">System Prompt</label>
            <textarea
              className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm font-mono"
              rows={8}
              value={formValues.system_prompt}
              onChange={(e) => handleChange('system_prompt', e.target.value)}
              placeholder="You are a specialized agent that..."
              required
              disabled={isSystem}
            />
            <p className="mt-1 text-xs text-gray-500">
              Define the agent's personality, behavior, and instructions.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Capabilities</label>
            <div className="flex flex-wrap gap-2">
              {capabilities.map((cap) => (
                <button
                  key={cap.name}
                  type="button"
                  onClick={() => !isSystem && toggleCapability(cap.name)}
                  disabled={isSystem}
                  className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                    formValues.capabilities.includes(cap.name)
                      ? 'bg-primary-100 border-primary-300 text-primary-700'
                      : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100'
                  } ${isSystem ? 'opacity-60 cursor-not-allowed' : ''}`}
                  title={cap.description}
                >
                  {cap.name}
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-gray-500">
              Capabilities determine when this agent is selected for routing.
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">Tool Access</label>
              <label className="flex items-center space-x-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  checked={useAllTools}
                  onChange={(e) => {
                    setUseAllTools(e.target.checked);
                    if (e.target.checked) {
                      handleChange('tool_whitelist', null);
                    } else {
                      handleChange('tool_whitelist', []);
                    }
                  }}
                  disabled={isSystem}
                />
                <span>All tools</span>
              </label>
            </div>
            {!useAllTools && (
              <div className="max-h-40 overflow-y-auto border rounded-lg p-2">
                <div className="flex flex-wrap gap-2">
                  {availableTools.map((tool) => (
                    <button
                      key={tool}
                      type="button"
                      onClick={() => !isSystem && toggleTool(tool)}
                      disabled={isSystem}
                      className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                        (formValues.tool_whitelist || []).includes(tool)
                          ? 'bg-green-100 border-green-300 text-green-700'
                          : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100'
                      } ${isSystem ? 'opacity-60 cursor-not-allowed' : ''}`}
                    >
                      {tool}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <p className="mt-1 text-xs text-gray-500">
              {useAllTools ? 'Agent has access to all available tools.' : `${(formValues.tool_whitelist || []).length} tools selected.`}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Priority: {formValues.priority}
            </label>
            <input
              type="range"
              min="1"
              max="100"
              value={formValues.priority}
              onChange={(e) => handleChange('priority', Number(e.target.value))}
              className="w-full"
            />
            <p className="mt-1 text-xs text-gray-500">
              Higher priority agents are preferred when multiple agents match.
            </p>
          </div>

          <div className="flex items-center">
            <label className="flex items-center space-x-2 text-sm text-gray-700">
              <input
                type="checkbox"
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                checked={formValues.is_active}
                onChange={(e) => handleChange('is_active', e.target.checked)}
              />
              <span>Active (available for routing)</span>
            </label>
          </div>

          <div className="flex justify-end space-x-3 pt-4 border-t">
            <Button type="button" variant="ghost" onClick={onClose} disabled={isSaving}>
              Cancel
            </Button>
            <Button
              type="submit"
              loading={isSaving}
              disabled={
                formValues.name.trim().length === 0 ||
                formValues.display_name.trim().length === 0 ||
                (requiresPrompt && formValues.system_prompt.trim().length < 10)
              }
            >
              {agent ? 'Save Changes' : 'Create Agent'}
            </Button>
          </div>
        </form>
      </div>
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
