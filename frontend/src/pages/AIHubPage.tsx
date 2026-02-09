/**
 * AI Hub Page
 *
 * Train custom models and LoRA adapters on user data.
 * Tabs: Datasets | Training | Models
 */

import React, { useEffect, useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Database,
  MessageCircle,
  Brain,
  Boxes,
  Plus,
  Trash2,
  Play,
  Square,
  Download,
  Upload,
  Eye,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Settings,
  Cpu,
  HardDrive,
  Zap,
  FileText,
  X,
  Search,
  ChevronRight,
  BarChart3,
  Rocket,
  Activity,
  Sparkles,
  Flag,
  Edit3,
  Check,
  Server,
  Layers,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../services/api';
import type {
  TrainingDataset,
  TrainingDatasetCreate,
  DatasetType,
  DatasetFormat,
  DatasetStatus,
  TrainingJob,
  TrainingJobCreate,
  TrainingJobStatus,
  TrainingMethod,
  TrainingBackend,
  ModelAdapter,
  AdapterStatus,
  BaseModelInfo,
  HyperparametersConfig,
} from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';

// Status configurations
const DATASET_STATUS_CONFIG: Record<DatasetStatus, { color: string; bgColor: string; icon: React.ComponentType<any> }> = {
  draft: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: Edit3 },
  validating: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Loader2 },
  ready: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle2 },
  error: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  archived: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: X },
};

type DatasetPresetId = 'repro_checklist' | 'perf_regression_triage' | 'gap_analysis_hypotheses';

type EnabledDatasetPreset = { id: string; name: string; description: string; dataset_type: DatasetType };

const JOB_STATUS_CONFIG: Record<TrainingJobStatus, { color: string; bgColor: string; icon: React.ComponentType<any> }> = {
  pending: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock },
  queued: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Layers },
  preparing: { color: 'text-indigo-700', bgColor: 'bg-indigo-100', icon: Download },
  training: { color: 'text-purple-700', bgColor: 'bg-purple-100', icon: Loader2 },
  saving: { color: 'text-sky-700', bgColor: 'bg-sky-100', icon: HardDrive },
  completed: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle2 },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  cancelled: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: X },
};

const ADAPTER_STATUS_CONFIG: Record<AdapterStatus, { color: string; bgColor: string; icon: React.ComponentType<any> }> = {
  training: { color: 'text-purple-700', bgColor: 'bg-purple-100', icon: Loader2 },
  ready: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle2 },
  deploying: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Loader2 },
  deployed: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Rocket },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  archived: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: X },
};

type TabType = 'datasets' | 'training' | 'models';

const AIHubPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('datasets');
  const [showAIScientist, setShowAIScientist] = useState(false);
  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();

  // Deep-link support: /ai-hub?tab=training&job=<id>
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tab = params.get('tab') as TabType | null;
    if (tab === 'datasets' || tab === 'training' || tab === 'models') {
      setActiveTab(tab);
    }
  }, [location.search]);

  const setTabAndPersist = (tab: TabType) => {
    setActiveTab(tab);
    const params = new URLSearchParams(location.search);
    params.set('tab', tab);
    // Keep job/createJobForDataset params when switching into training; drop job when leaving it.
    if (tab !== 'training') {
      params.delete('job');
      params.delete('createJobForDataset');
    }
    navigate({ pathname: '/ai-hub', search: `?${params.toString()}` }, { replace: true });
  };

  // Fetch stats for header
  const { data: trainingStats } = useQuery(
    ['training-stats'],
    () => apiClient.getTrainingStats(),
    { refetchInterval: 10000 }
  );

  const { data: adapterStats } = useQuery(
    ['adapter-stats'],
    () => apiClient.getModelAdapterStats()
  );

  const tabs: { id: TabType; label: string; icon: React.ComponentType<any>; count?: number }[] = [
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'training', label: 'Training', icon: Brain, count: trainingStats?.running_jobs },
    { id: 'models', label: 'Models', icon: Boxes },
  ];

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">AI Hub</h1>
            <p className="text-gray-500">Train custom models and LoRA adapters on your data</p>
          </div>
          <div className="flex items-center gap-4">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setShowAIScientist(true)}
              title="Propose enabled presets/evals + happy-path demos"
            >
              <Sparkles className="w-4 h-4 mr-2" />
              AI Scientist
            </Button>
            {/* Quick stats */}
            <div className="flex items-center gap-6 text-sm">
              {trainingStats && (
                <>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-gray-600">{trainingStats.running_jobs} training</span>
                  </div>
                  <div className="text-gray-600">
                    {trainingStats.completed_jobs} completed
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-6 border-b border-gray-200">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setTabAndPersist(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
                {tab.count !== undefined && tab.count > 0 && (
                  <span className="ml-1 px-2 py-0.5 text-xs bg-primary-100 text-primary-700 rounded-full">
                    {tab.count}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0">
        {activeTab === 'datasets' && <DatasetsTab />}
        {activeTab === 'training' && <TrainingTab />}
        {activeTab === 'models' && <ModelsTab />}
      </div>

      {showAIScientist && <AIScientistModal onClose={() => setShowAIScientist(false)} />}
    </div>
  );
};

const AIScientistModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [workflows, setWorkflows] = useState<{ triage: boolean; extraction: boolean; literature: boolean }>({
    triage: true,
    extraction: true,
    literature: true,
  });
  const [applyNow, setApplyNow] = useState(false);
  const [customerContext, setCustomerContext] = useState('');
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();

  const { data: templatesData } = useQuery(
    ['agent-job-templates', 'ai-hub'],
    () => apiClient.listAgentJobTemplates('ai_hub'),
    { staleTime: 60000 }
  );

  const handleCreate = async () => {
    const template = (templatesData as any)?.templates?.find((t: any) => t.name === 'ai_hub_scientist_propose_bundle')
      || (templatesData as any)?.templates?.[0];
    if (!template?.id) {
      toast.error('AI Scientist template not available');
      return;
    }

    const selected = Object.entries(workflows)
      .filter(([, v]) => v)
      .map(([k]) => k);
    if (selected.length === 0) {
      toast.error('Select at least one workflow');
      return;
    }

    setCreating(true);
    try {
      const job = await apiClient.createAgentJobFromTemplate({
        template_id: template.id,
        name: `AI Scientist — ${new Date().toLocaleDateString()}`,
        config: { workflows: selected, apply: applyNow, customer_context: customerContext.trim() || undefined },
        start_immediately: true,
      } as any);
      toast.success('AI Scientist job started');
      onClose();
      navigate(`/autonomous-agents?job=${encodeURIComponent(job.id)}`);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to start AI Scientist job');
    } finally {
      setCreating(false);
    }
  };

  const toggle = (key: 'triage' | 'extraction' | 'literature') => {
    setWorkflows((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
        <div className="p-6 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">AI Scientist</h2>
            <p className="text-sm text-gray-500">Propose enabled presets/evals + happy-path demos</p>
          </div>
          <button className="text-gray-500 hover:text-gray-700" onClick={onClose}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">Workflows</div>
            <div className="space-y-2 text-sm text-gray-700">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={workflows.triage} onChange={() => toggle('triage')} />
                Perf / Investigation triage
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={workflows.extraction} onChange={() => toggle('extraction')} />
                Evidence / experiment extraction
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={workflows.literature} onChange={() => toggle('literature')} />
                Literature triage + gap analysis
              </label>
            </div>
          </div>

          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">Customer context (optional)</div>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              rows={3}
              value={customerContext}
              onChange={(e) => setCustomerContext(e.target.value)}
              placeholder="E.g., 'Applied robotics lab. Weekly literature triage + experiment writeups. Biggest pain: reproducibility and comparing results across hardware.'"
            />
            <div className="mt-1 text-xs text-gray-500">
              Used to pick the most relevant presets/evals from enabled plugins and suggest missing ones.
            </div>
          </div>

          <div className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-lg p-3">
            <div>
              <div className="text-sm font-medium text-gray-800">Apply allowlists</div>
              <div className="text-xs text-gray-500">Writes enabled presets/evals (admin only)</div>
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={applyNow} onChange={(e) => setApplyNow(e.target.checked)} />
              Apply now
            </label>
          </div>
        </div>

        <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={creating}>
            {creating ? 'Starting…' : 'Propose Bundle'}
          </Button>
        </div>
      </div>
    </div>
  );
};

// ==================== Datasets Tab ====================

const DatasetsTab: React.FC = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<TrainingDataset | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data: datasetsData, isLoading, refetch } = useQuery(
    ['training-datasets', statusFilter],
    () => apiClient.listTrainingDatasets({
      status: statusFilter || undefined,
      page_size: 50,
    }),
    { refetchInterval: 10000 }
  );

  const deleteMutation = useMutation(
    (datasetId: string) => apiClient.deleteTrainingDataset(datasetId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['training-datasets']);
        toast.success('Dataset deleted');
        setSelectedDataset(null);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
      },
    }
  );

  const validateMutation = useMutation(
    (datasetId: string) => apiClient.validateDataset(datasetId),
    {
      onSuccess: (result) => {
        queryClient.invalidateQueries(['training-datasets']);
        if (result.is_valid) {
          toast.success('Dataset is valid and ready for training');
        } else {
          const errorSummary =
            result.errors?.map((e) => (e.code ? `${e.code}: ${e.message}` : e.message)).join(', ') ||
            'Unknown error';
          toast.error(`Validation failed: ${errorSummary}`);
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Validation failed');
      },
    }
  );

  const GenerateDatasetModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    const [readingListId, setReadingListId] = useState('');
    const [presetId, setPresetId] = useState<string>('');
    const [name, setName] = useState('Generated Dataset');
    const [description, setDescription] = useState('');
    const [samplesPerDoc, setSamplesPerDoc] = useState(5);
    const [autoValidate, setAutoValidate] = useState(true);
    const [extraInstructions, setExtraInstructions] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [goToTrainingAfter, setGoToTrainingAfter] = useState(true);

    const { data: readingLists } = useQuery(['reading-lists', 'ai-hub'], () =>
      apiClient.listReadingLists({ limit: 200, offset: 0 })
    );

    const { data: enabledPresetsData } = useQuery(
      ['ai-hub', 'dataset-presets', 'enabled'],
      () => apiClient.listEnabledDatasetPresets(),
      { staleTime: 60000 }
    );

    const { data: readingList } = useQuery(
      ['reading-list', readingListId],
      () => apiClient.getReadingList(readingListId),
      { enabled: !!readingListId }
    );

    const preset = useMemo(() => {
      const presets: EnabledDatasetPreset[] = enabledPresetsData?.presets || [];
      return presets.find((p) => p.id === presetId) || presets[0] || null;
    }, [enabledPresetsData?.presets, presetId]);

    useEffect(() => {
      if (presetId) return;
      const presets: EnabledDatasetPreset[] = enabledPresetsData?.presets || [];
      if (presets.length === 0) return;
      const preferred = presets.find((p) => p.id === 'perf_regression_triage_v1');
      setPresetId((preferred || presets[0]).id);
    }, [enabledPresetsData?.presets, presetId]);

    useEffect(() => {
      if (!readingListId) return;
      const rlName =
        (readingLists?.items || []).find((x: any) => x.id === readingListId)?.name ||
        readingList?.name ||
        'Reading List';
      setName(`RL: ${rlName} — ${preset?.name || 'Preset'}`);
      if (!description.trim()) {
        setDescription(`Generated from reading list "${rlName}" using preset "${preset?.name || presetId}".`);
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [readingListId, presetId, preset?.name, readingList?.name, readingLists?.items]);

    const handleSubmit = async () => {
      if (!readingListId) {
        toast.error('Select a reading list');
        return;
      }
      if (!presetId) {
        toast.error('Select a preset');
        return;
      }
      const docIds: string[] = (readingList?.items || [])
        .map((it: any) => it.document_id)
        .filter(Boolean);
      if (docIds.length === 0) {
        toast.error('Reading list has no documents');
        return;
      }
      if (!name.trim()) {
        toast.error('Enter a dataset name');
        return;
      }

      setIsSubmitting(true);
      try {
        const created = await apiClient.generateDatasetFromDocuments({
          name: name.trim(),
          description: description || undefined,
          document_ids: docIds,
          dataset_type: (preset?.dataset_type as DatasetType) || 'instruction',
          samples_per_document: samplesPerDoc,
          preset_id: presetId,
          extra_instructions: extraInstructions.trim() ? extraInstructions.trim() : undefined,
        });

        toast.success('Dataset generated');
        queryClient.invalidateQueries(['training-datasets']);
        setSelectedDataset(created);

        if (autoValidate) {
          try {
            await validateMutation.mutateAsync(created.id);
          } catch (e) {
            // validation toast handled by mutation
          } finally {
            queryClient.invalidateQueries(['training-datasets']);
          }
        }

        onClose();

        if (goToTrainingAfter) {
          const params = new URLSearchParams();
          params.set('tab', 'training');
          params.set('createJobForDataset', created.id);
          navigate({ pathname: '/ai-hub', search: `?${params.toString()}` });
        }
      } catch (error: any) {
        toast.error(error.message || 'Failed to generate dataset');
      } finally {
        setIsSubmitting(false);
      }
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Generate Dataset (Research Presets)</h2>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="w-5 h-5" />
              </Button>
            </div>
            <p className="text-sm text-gray-500 mt-1">
              Turn a Reading List into a training dataset tailored for a CPU/compiler research lab.
            </p>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Reading List *</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={readingListId}
                onChange={(e) => setReadingListId(e.target.value)}
              >
                <option value="">Select a reading list</option>
                {(readingLists?.items || []).map((rl: any) => (
                  <option key={rl.id} value={rl.id}>
                    {rl.name}
                  </option>
                ))}
              </select>
              {readingListId && (
                <p className="text-xs text-gray-500 mt-1">
                  {((readingList?.items || []).length || 0).toLocaleString()} documents
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Preset</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={presetId}
                onChange={(e) => setPresetId(e.target.value)}
              >
                <option value="">Select a preset</option>
                {(enabledPresetsData?.presets || []).map((p: any) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                    {p.id === 'perf_regression_triage_v1' ? ' (Recommended)' : ''}
                  </option>
                ))}
              </select>
              {preset?.description && <p className="text-xs text-gray-500 mt-1">{preset.description}</p>}
              {(enabledPresetsData?.presets || []).length === 0 && (
                <p className="text-xs text-gray-500 mt-1">No presets enabled for this deployment.</p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Samples per document</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={samplesPerDoc}
                  onChange={(e) => setSamplesPerDoc(parseInt(e.target.value, 10) || 1)}
                />
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={autoValidate}
                    onChange={(e) => setAutoValidate(e.target.checked)}
                  />
                  Auto-validate after generation
                </label>
              </div>
            </div>

            <div className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-lg p-3">
              <div>
                <div className="text-sm font-medium text-gray-800">Next step</div>
                <div className="text-xs text-gray-500">Jump to Training and prefill the job wizard</div>
              </div>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={goToTrainingAfter}
                  onChange={(e) => setGoToTrainingAfter(e.target.checked)}
                />
                Continue to Training
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Dataset Name *</label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={2}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Extra constraints (optional)</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  value={extraInstructions}
                  onChange={(e) => setExtraInstructions(e.target.value)}
                  placeholder="E.g., include measurements; mention confounders; cite evidence; keep answers under 250 tokens…"
                />
              </div>
            </div>

          <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSubmit} disabled={isSubmitting}>
              {isSubmitting ? 'Generating…' : 'Generate Dataset'}
            </Button>
          </div>
        </div>
      </div>
    );
  };

  const DatasetCard: React.FC<{ dataset: TrainingDataset }> = ({ dataset }) => {
    const statusConfig = DATASET_STATUS_CONFIG[dataset.status] || DATASET_STATUS_CONFIG.draft;
    const StatusIcon = statusConfig.icon;
    const isValidating = dataset.status === 'validating';

    return (
      <div
        className={`bg-white border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
          selectedDataset?.id === dataset.id ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
        }`}
        onClick={() => setSelectedDataset(dataset)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 truncate max-w-[200px]">{dataset.name}</h3>
              <p className="text-xs text-gray-500 capitalize">{dataset.dataset_type.replace('_', ' ')}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
            <StatusIcon className={`w-3 h-3 ${isValidating ? 'animate-spin' : ''}`} />
            <span className="capitalize">{dataset.status}</span>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-gray-50 rounded px-2 py-1">
            <p className="text-xs text-gray-500">Samples</p>
            <p className="font-semibold text-sm">{dataset.sample_count.toLocaleString()}</p>
          </div>
          <div className="bg-gray-50 rounded px-2 py-1">
            <p className="text-xs text-gray-500">Tokens</p>
            <p className="font-semibold text-sm">{(dataset.token_count / 1000).toFixed(1)}K</p>
          </div>
          <div className="bg-gray-50 rounded px-2 py-1">
            <p className="text-xs text-gray-500">Format</p>
            <p className="font-semibold text-sm uppercase">{dataset.format}</p>
          </div>
        </div>

        {/* Validation badge */}
        {dataset.is_validated && (
          <div className="mt-3 flex items-center gap-1 text-xs text-green-600">
            <CheckCircle2 className="w-3 h-3" />
            <span>Validated</span>
          </div>
        )}
      </div>
    );
  };

  const DatasetDetailPanel: React.FC<{ dataset: TrainingDataset }> = ({ dataset }) => {
    const statusConfig = DATASET_STATUS_CONFIG[dataset.status] || DATASET_STATUS_CONFIG.draft;
    const StatusIcon = statusConfig.icon;

    return (
      <div className="bg-white border border-gray-200 rounded-lg h-full overflow-hidden flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
                <Database className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">{dataset.name}</h2>
                <p className="text-sm text-gray-500 capitalize">{dataset.dataset_type.replace('_', ' ')}</p>
              </div>
            </div>
            <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${statusConfig.bgColor} ${statusConfig.color}`}>
              <StatusIcon className="w-4 h-4" />
              <span className="font-medium capitalize">{dataset.status}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-3">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => validateMutation.mutate(dataset.id)}
              disabled={validateMutation.isLoading || dataset.status === 'validating'}
            >
              <Check className="w-4 h-4 mr-1" />
              Validate
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={async () => {
                try {
                  const result = await apiClient.exportDataset(dataset.id);
                  toast.success(`Exported to ${result.file_path}`);
                } catch (error: any) {
                  toast.error(error.message || 'Export failed');
                }
              }}
            >
              <Download className="w-4 h-4 mr-1" />
              Export
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (window.confirm('Delete this dataset?')) {
                  deleteMutation.mutate(dataset.id);
                }
              }}
              disabled={deleteMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {dataset.description && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-1">Description</h3>
              <p className="text-sm text-gray-600">{dataset.description}</p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Samples</p>
              <p className="text-2xl font-bold">{dataset.sample_count.toLocaleString()}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Tokens</p>
              <p className="text-2xl font-bold">{dataset.token_count.toLocaleString()}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Format</p>
              <p className="text-lg font-semibold uppercase">{dataset.format}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Size</p>
              <p className="text-lg font-semibold">
                {dataset.file_size ? `${(dataset.file_size / 1024 / 1024).toFixed(2)} MB` : '-'}
              </p>
            </div>
          </div>

          {dataset.validation_errors && dataset.validation_errors.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-red-700 mb-2">Validation Errors</h3>
              <div className="bg-red-50 rounded-lg p-3">
                <ul className="text-sm text-red-600 space-y-1 list-disc list-inside">
                  {dataset.validation_errors.map((validationError, idx) => (
                    <li key={idx}>
                      {validationError.code ? `${validationError.code}: ${validationError.message}` : validationError.message}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          <div className="text-xs text-gray-500 space-y-1">
            <p>Created: {new Date(dataset.created_at).toLocaleString()}</p>
            <p>
              Updated: {dataset.updated_at ? new Date(dataset.updated_at).toLocaleString() : '-'}
            </p>
            <p>Version: {dataset.version}</p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-3">
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All Status</option>
            <option value="draft">Draft</option>
            <option value="validating">Validating</option>
            <option value="ready">Ready</option>
            <option value="error">Error</option>
            <option value="archived">Archived</option>
          </select>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" onClick={() => setShowGenerateModal(true)}>
            <Zap className="w-4 h-4 mr-2" />
            Generate (Presets)
          </Button>
          <Button onClick={() => setShowCreateModal(true)}>
            <Plus className="w-4 h-4 mr-2" />
            New Dataset
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        <div className="w-2/3 overflow-y-auto">
          {isLoading ? (
            <div className="flex justify-center items-center h-full">
              <LoadingSpinner />
            </div>
          ) : datasetsData?.datasets.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <Database className="w-12 h-12 mb-3 text-gray-400" />
              <p className="text-lg font-medium">No datasets yet</p>
              <p className="text-sm">Create a dataset to start training models</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {datasetsData?.datasets.map((dataset) => (
                <DatasetCard key={dataset.id} dataset={dataset} />
              ))}
            </div>
          )}
        </div>

        <div className="w-1/3">
          {selectedDataset ? (
            <DatasetDetailPanel dataset={selectedDataset} />
          ) : (
            <div className="bg-gray-50 border border-gray-200 rounded-lg h-full flex flex-col items-center justify-center text-gray-500">
              <Eye className="w-10 h-10 mb-3 text-gray-400" />
              <p className="font-medium">Select a dataset</p>
              <p className="text-sm">Click on a dataset to view details</p>
            </div>
          )}
        </div>
      </div>

      {showCreateModal && (
        <CreateDatasetModal onClose={() => setShowCreateModal(false)} />
      )}
      {showGenerateModal && (
        <GenerateDatasetModal onClose={() => setShowGenerateModal(false)} />
      )}
    </div>
  );
};

// Create Dataset Modal
const CreateDatasetModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [datasetType, setDatasetType] = useState<DatasetType>('instruction');
  const [format, setFormat] = useState<DatasetFormat>('alpaca');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const queryClient = useQueryClient();

  const handleSubmit = async () => {
    if (!name.trim()) {
      toast.error('Enter a name');
      return;
    }

    setIsSubmitting(true);
    try {
      await apiClient.createTrainingDataset({
        name,
        description: description || undefined,
        dataset_type: datasetType,
        format,
      });
      toast.success('Dataset created');
      queryClient.invalidateQueries(['training-datasets']);
      onClose();
    } catch (error: any) {
      toast.error(error.message || 'Failed to create dataset');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Create Dataset</h2>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name *</label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Training Dataset"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              rows={2}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
            <select
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={datasetType}
              onChange={(e) => setDatasetType(e.target.value as DatasetType)}
            >
              <option value="instruction">Instruction Following</option>
              <option value="conversation">Conversation</option>
              <option value="completion">Text Completion</option>
              <option value="qa">Question & Answer</option>
              <option value="classification">Classification</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Format</label>
            <select
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              value={format}
              onChange={(e) => setFormat(e.target.value as DatasetFormat)}
            >
              <option value="alpaca">Alpaca</option>
              <option value="sharegpt">ShareGPT</option>
              <option value="openai">OpenAI</option>
              <option value="completion">Completion</option>
              <option value="custom">Custom</option>
            </select>
          </div>
        </div>

        <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting ? 'Creating...' : 'Create Dataset'}
          </Button>
        </div>
      </div>
    </div>
  );
};

// ==================== Training Tab ====================

const TrainingTab: React.FC = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();

  const urlJobId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('job');
  }, [location.search]);

  const urlCreateJobForDatasetId = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get('createJobForDataset');
  }, [location.search]);

  const { data: jobsData, isLoading, refetch } = useQuery(
    ['training-jobs', statusFilter],
    () => apiClient.listTrainingJobs({
      status: statusFilter || undefined,
      page_size: 50,
    }),
    { refetchInterval: 5000 }
  );

  const { data: stats } = useQuery(['training-stats'], () => apiClient.getTrainingStats());

  const startMutation = useMutation(
    (jobId: string) => apiClient.startTrainingJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['training-jobs']);
        toast.success('Training started');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to start training');
      },
    }
  );

  const cancelMutation = useMutation(
    (jobId: string) => apiClient.cancelTrainingJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['training-jobs']);
        toast.success('Training cancelled');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to cancel training');
      },
    }
  );

  const deleteMutation = useMutation(
    (jobId: string) => apiClient.deleteTrainingJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['training-jobs']);
        toast.success('Job deleted');
        setSelectedJob(null);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
      },
    }
  );

  // Deep-link: auto-select a job by query param.
  useEffect(() => {
    if (!urlJobId) return;
    const match = jobsData?.jobs?.find((j) => j.id === urlJobId);
    if (match) setSelectedJob(match);
  }, [urlJobId, jobsData?.jobs]);

  // Deep-link: open Create Training wizard with dataset preselected.
  useEffect(() => {
    if (!urlCreateJobForDatasetId) return;
    setShowCreateModal(true);
  }, [urlCreateJobForDatasetId]);

  const setJobParam = (jobId: string) => {
    const params = new URLSearchParams(location.search);
    params.set('tab', 'training');
    params.set('job', jobId);
    params.delete('createJobForDataset');
    navigate({ pathname: '/ai-hub', search: `?${params.toString()}` }, { replace: true });
  };

  // Real-time progress updates for the selected job
  useEffect(() => {
    if (!selectedJob) return;

    let ws: WebSocket | null = null;

    try {
      ws = apiClient.createTrainingJobProgressWebSocket(selectedJob.id);
    } catch (e) {
      return;
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (!msg || (msg.type !== 'progress' && msg.type !== 'connected')) return;

        const applyUpdate = (job: TrainingJob): TrainingJob => {
          const next: TrainingJob = {
            ...job,
            status: msg.status ?? job.status,
            progress: msg.progress ?? job.progress,
            current_step: msg.current_step ?? job.current_step,
            total_steps: msg.total_steps ?? job.total_steps,
            current_epoch: msg.current_epoch ?? job.current_epoch,
            total_epochs: msg.total_epochs ?? job.total_epochs,
          };

          if (msg.current_loss !== undefined || msg.learning_rate !== undefined) {
            next.training_metrics = {
              ...(job.training_metrics || {}),
              current_loss: msg.current_loss ?? job.training_metrics?.current_loss,
              learning_rate: msg.learning_rate ?? job.training_metrics?.learning_rate,
            };
          }

          return next;
        };

        setSelectedJob((prev) => {
          if (!prev || prev.id !== selectedJob.id) return prev;
          return applyUpdate(prev);
        });

        queryClient.setQueryData(['training-jobs', statusFilter], (oldData: any) => {
          if (!oldData?.jobs) return oldData;
          return {
            ...oldData,
            jobs: oldData.jobs.map((j: TrainingJob) => (j.id === selectedJob.id ? applyUpdate(j) : j)),
          };
        });

        if (msg.type === 'progress' && ['completed', 'failed', 'cancelled'].includes(msg.status)) {
          queryClient.invalidateQueries(['training-jobs']);
          queryClient.invalidateQueries(['training-stats']);
          queryClient.invalidateQueries(['model-adapters']);
          queryClient.invalidateQueries(['adapter-stats']);
        }
      } catch (e) {
        // ignore malformed messages
      }
    };

    return () => {
      try {
        ws?.close();
      } catch (e) {
        // ignore
      }
    };
  }, [selectedJob?.id, statusFilter, queryClient]);

  const JobCard: React.FC<{ job: TrainingJob }> = ({ job }) => {
    const statusConfig = JOB_STATUS_CONFIG[job.status] || JOB_STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const isRunning = job.status === 'preparing' || job.status === 'training' || job.status === 'saving';

    return (
      <div
        className={`bg-white border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
          selectedJob?.id === job.id ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
        }`}
        onClick={() => {
          setSelectedJob(job);
          setJobParam(job.id);
        }}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
              <Brain className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 truncate max-w-[200px]">{job.name}</h3>
              <p className="text-xs text-gray-500">{job.base_model}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
            <StatusIcon className={`w-3 h-3 ${isRunning ? 'animate-spin' : ''}`} />
            <span className="capitalize">{job.status}</span>
          </div>
        </div>

        {/* Progress */}
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
            <span>Progress</span>
            <span>{job.progress}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                job.status === 'completed' ? 'bg-green-500' :
                job.status === 'failed' ? 'bg-red-500' :
                'bg-purple-500'
              }`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>

        {/* Training info */}
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <Activity className="w-3 h-3" />
            {job.training_method}
          </span>
          {job.current_epoch !== undefined && job.total_epochs && (
            <span>Epoch {job.current_epoch}/{job.total_epochs}</span>
          )}
        </div>
      </div>
    );
  };

  const JobDetailPanel: React.FC<{ job: TrainingJob }> = ({ job }) => {
    const statusConfig = JOB_STATUS_CONFIG[job.status] || JOB_STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const isRunning = job.status === 'preparing' || job.status === 'training' || job.status === 'saving';
    const canStart = job.status === 'pending';
    const canCancel = job.status === 'queued' || job.status === 'preparing' || job.status === 'training';

    return (
      <div className="bg-white border border-gray-200 rounded-lg h-full overflow-hidden flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                <Brain className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">{job.name}</h2>
                <p className="text-sm text-gray-500">{job.base_model}</p>
              </div>
            </div>
            <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${statusConfig.bgColor} ${statusConfig.color}`}>
              <StatusIcon className={`w-4 h-4 ${isRunning ? 'animate-spin' : ''}`} />
              <span className="font-medium capitalize">{job.status}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-3">
            {canStart && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => startMutation.mutate(job.id)}
                disabled={startMutation.isLoading}
              >
                <Play className="w-4 h-4 mr-1" />
                Start
              </Button>
            )}
            {canCancel && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => cancelMutation.mutate(job.id)}
                disabled={cancelMutation.isLoading}
              >
                <Square className="w-4 h-4 mr-1" />
                Cancel
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (window.confirm('Delete this job?')) {
                  deleteMutation.mutate(job.id);
                }
              }}
              disabled={isRunning || job.status === 'queued' || deleteMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {/* Progress */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Training Progress</h3>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  job.status === 'completed' ? 'bg-green-500' :
                  job.status === 'failed' ? 'bg-red-500' :
                  'bg-purple-500'
                }`}
                style={{ width: `${job.progress}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{job.progress}% complete</span>
              {job.current_step !== undefined && job.total_steps && (
                <span>Step {job.current_step}/{job.total_steps}</span>
              )}
            </div>
          </div>

          {/* Config */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Method</p>
              <p className="font-semibold capitalize">{job.training_method}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Backend</p>
              <p className="font-semibold capitalize">{job.training_backend}</p>
            </div>
            {job.current_epoch !== undefined && (
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-500">Epoch</p>
                <p className="font-semibold">{job.current_epoch} / {job.total_epochs}</p>
              </div>
            )}
          </div>

          {/* Hyperparameters */}
          {job.hyperparameters && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Hyperparameters</h3>
              <div className="bg-gray-50 rounded-lg p-3 text-sm">
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(job.hyperparameters).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-500">{key}:</span>
                      <span className="font-medium">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Metrics */}
          {job.training_metrics && Object.keys(job.training_metrics).length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Training Metrics</h3>
              <div className="grid grid-cols-2 gap-2">
                {job.training_metrics.current_loss !== undefined && (
                  <div className="bg-blue-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-blue-600">Current Loss</p>
                    <p className="text-lg font-bold text-blue-700">{job.training_metrics.current_loss.toFixed(4)}</p>
                  </div>
                )}
                {job.training_metrics.best_loss !== undefined && (
                  <div className="bg-green-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-green-600">Best Loss</p>
                    <p className="text-lg font-bold text-green-700">{job.training_metrics.best_loss.toFixed(4)}</p>
                  </div>
                )}
                {job.training_metrics.learning_rate !== undefined && (
                  <div className="bg-purple-50 rounded-lg p-3 text-center">
                    <p className="text-xs text-purple-600">Learning Rate</p>
                    <p className="text-lg font-bold text-purple-700">{job.training_metrics.learning_rate.toExponential(2)}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error */}
          {job.error && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-red-700 mb-1">Error</h3>
              <p className="text-sm text-red-600 bg-red-50 rounded-lg p-3">{job.error}</p>
            </div>
          )}

          <div className="text-xs text-gray-500 space-y-1">
            <p>Created: {new Date(job.created_at).toLocaleString()}</p>
            {job.started_at && <p>Started: {new Date(job.started_at).toLocaleString()}</p>}
            {job.completed_at && <p>Completed: {new Date(job.completed_at).toLocaleString()}</p>}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-xs text-gray-500">Total Jobs</p>
            <p className="text-2xl font-bold">{stats.total_jobs}</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-xs text-gray-500">Running</p>
            <p className="text-2xl font-bold text-purple-600">{stats.running_jobs}</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-xs text-gray-500">Completed</p>
            <p className="text-2xl font-bold text-green-600">{stats.completed_jobs}</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-xs text-gray-500">Failed</p>
            <p className="text-2xl font-bold text-red-600">{stats.failed_jobs}</p>
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-3">
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="queued">Queued</option>
            <option value="preparing">Preparing</option>
            <option value="training">Training</option>
            <option value="saving">Saving</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="w-4 h-4 mr-2" />
          New Training Job
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        <div className="w-2/3 overflow-y-auto">
          {isLoading ? (
            <div className="flex justify-center items-center h-full">
              <LoadingSpinner />
            </div>
          ) : jobsData?.jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <Brain className="w-12 h-12 mb-3 text-gray-400" />
              <p className="text-lg font-medium">No training jobs yet</p>
              <p className="text-sm">Create a job to start training models</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {jobsData?.jobs.map((job) => (
                <JobCard key={job.id} job={job} />
              ))}
            </div>
          )}
        </div>

        <div className="w-1/3">
          {selectedJob ? (
            <JobDetailPanel job={selectedJob} />
          ) : (
            <div className="bg-gray-50 border border-gray-200 rounded-lg h-full flex flex-col items-center justify-center text-gray-500">
              <Eye className="w-10 h-10 mb-3 text-gray-400" />
              <p className="font-medium">Select a job</p>
              <p className="text-sm">Click on a job to view details</p>
            </div>
          )}
        </div>
      </div>

      {showCreateModal && (
        <CreateTrainingModal
          onClose={() => {
            setShowCreateModal(false);
            const params = new URLSearchParams(location.search);
            params.delete('createJobForDataset');
            navigate({ pathname: '/ai-hub', search: `?${params.toString()}` }, { replace: true });
          }}
          initialDatasetId={urlCreateJobForDatasetId || undefined}
          initialStep={urlCreateJobForDatasetId ? 'model' : undefined}
          onCreated={(job) => {
            queryClient.invalidateQueries(['training-jobs']);
            setSelectedJob(job);
            setJobParam(job.id);
          }}
        />
      )}
    </div>
  );
};

// Create Training Modal
const CreateTrainingModal: React.FC<{
  onClose: () => void;
  initialDatasetId?: string;
  initialStep?: 'dataset' | 'model' | 'config';
  onCreated?: (job: TrainingJob) => void;
}> = ({ onClose, initialDatasetId, initialStep, onCreated }) => {
  const [step, setStep] = useState<'dataset' | 'model' | 'config'>(initialStep || 'dataset');
  const [selectedDataset, setSelectedDataset] = useState<string>(initialDatasetId || '');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [baseModel, setBaseModel] = useState('');
  const [method, setMethod] = useState<TrainingMethod>('lora');
  const [backend, setBackend] = useState<TrainingBackend>('local');
  const [hyperparameters, setHyperparameters] = useState<HyperparametersConfig>({
    learning_rate: 2e-4,
    num_epochs: 3,
    batch_size: 4,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [nameTouched, setNameTouched] = useState(false);
  const [descriptionTouched, setDescriptionTouched] = useState(false);
  const [baseModelTouched, setBaseModelTouched] = useState(false);
  const [methodTouched, setMethodTouched] = useState(false);
  const queryClient = useQueryClient();

  const { data: datasetsData } = useQuery(
    ['training-datasets-for-job'],
    () => apiClient.listTrainingDatasets({ status: 'ready', page_size: 100 })
  );

  const { data: baseModelsData } = useQuery(
    ['base-models'],
    () => apiClient.getAvailableBaseModels()
  );

  const { data: backendsData } = useQuery(
    ['training-backends'],
    () => apiClient.getAvailableTrainingBackends()
  );

  useEffect(() => {
    const firstBackend = backendsData?.backends?.[0]?.name as TrainingBackend | undefined;
    if (firstBackend && backend !== firstBackend && !backendsData?.backends?.some((b) => b.name === backend)) {
      setBackend(firstBackend);
    }
  }, [backendsData?.backends, backend]);

  const selectedDatasetObj = useMemo(() => {
    return datasetsData?.datasets?.find((d) => d.id === selectedDataset) || null;
  }, [datasetsData?.datasets, selectedDataset]);

  // If the dataset was preselected via deep-link, ensure it's still selected once the query resolves.
  useEffect(() => {
    if (!initialDatasetId) return;
    if (!datasetsData?.datasets?.some((d) => d.id === initialDatasetId)) return;
    setSelectedDataset(initialDatasetId);
  }, [initialDatasetId, datasetsData?.datasets]);

  // Auto-fill job name/description from dataset (only if user hasn't typed their own yet).
  useEffect(() => {
    if (!selectedDatasetObj) return;
    const datasetName = selectedDatasetObj.name || 'Dataset';
    const lower = datasetName.toLowerCase();
    const looksLikePerfTriage = lower.includes('perf') && lower.includes('triage');
    const suggestedName = looksLikePerfTriage ? `Perf Triage — ${datasetName}` : `Train — ${datasetName}`;
    const suggestedDescription = `Trained on dataset "${datasetName}".`;

    if (!nameTouched && !name.trim()) setName(suggestedName);
    if (!descriptionTouched && !description.trim()) setDescription(suggestedDescription);
  }, [selectedDatasetObj, nameTouched, descriptionTouched, name, description]);

  // Prefer simulated backend for demos when available.
  useEffect(() => {
    const hasSimulated = backendsData?.backends?.some((b) => b.name === 'simulated' && b.is_available);
    if (!hasSimulated) return;
    if (backend === 'simulated') return;
    setBackend('simulated');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendsData?.backends]);

  // Auto-pick a recommended base model for demos (smallest available model).
  useEffect(() => {
    if (baseModelTouched) return;
    if (baseModel) return;
    const models = (baseModelsData?.models || []).filter((m) => m.is_available);
    if (models.length === 0) return;

    const pick = [...models].sort((a, b) => {
      const as = a.size_gb ?? Number.POSITIVE_INFINITY;
      const bs = b.size_gb ?? Number.POSITIVE_INFINITY;
      if (as !== bs) return as - bs;
      return (a.name || '').localeCompare(b.name || '');
    })[0];

    if (pick?.name) setBaseModel(pick.name);
  }, [baseModelsData?.models, baseModelTouched, baseModel]);

  // Auto-pick training method based on model size (prefer QLoRA for larger models).
  useEffect(() => {
    if (methodTouched) return;
    if (!baseModel) return;
    const model = (baseModelsData?.models || []).find((m) => m.name === baseModel);
    const nameLower = baseModel.toLowerCase();

    const looksLargeByName =
      nameLower.includes('7b') ||
      nameLower.includes('8b') ||
      nameLower.includes('13b') ||
      nameLower.includes('14b') ||
      nameLower.includes('34b') ||
      nameLower.includes('70b');

    const looksLargeBySize = (model?.size_gb ?? 0) >= 10;

    const shouldPreferQlora = looksLargeByName || looksLargeBySize;
    const desired: TrainingMethod = shouldPreferQlora ? 'qlora' : 'lora';

    if (method !== desired) setMethod(desired);
  }, [baseModel, baseModelsData?.models, methodTouched, method]);

  const handleSubmit = async () => {
    if (!selectedDataset) {
      toast.error('Select a dataset');
      return;
    }
    if (!baseModel) {
      toast.error('Select a base model');
      return;
    }
    if (!name.trim()) {
      toast.error('Enter a name');
      return;
    }

    setIsSubmitting(true);
    try {
      const createdJob = await apiClient.createTrainingJob({
        name,
        description: description || undefined,
        training_method: method,
        training_backend: backend,
        base_model: baseModel,
        dataset_id: selectedDataset,
        hyperparameters,
      });
      toast.success('Training job created');
      queryClient.invalidateQueries(['training-jobs']);
      onCreated?.(createdJob);
      onClose();
    } catch (error: any) {
      toast.error(error.message || 'Failed to create job');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Create Training Job</h2>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Steps */}
          <div className="flex items-center gap-4 mt-4">
            {['dataset', 'model', 'config'].map((s, idx) => (
              <div
                key={s}
                className={`flex items-center gap-2 ${step === s ? 'text-primary-600' : 'text-gray-400'}`}
              >
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                    step === s ? 'bg-primary-600 text-white' : 'bg-gray-200'
                  }`}
                >
                  {idx + 1}
                </div>
                <span className="text-sm capitalize">{s}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {step === 'dataset' && (
            <div className="space-y-2">
              <p className="text-sm text-gray-600 mb-4">Select a validated dataset for training</p>
              {datasetsData?.datasets.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Database className="w-10 h-10 mx-auto mb-2 text-gray-400" />
                  <p>No ready datasets available</p>
                  <p className="text-sm">Create and validate a dataset first</p>
                </div>
              ) : (
                datasetsData?.datasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                      selectedDataset === dataset.id
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedDataset(dataset.id)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">{dataset.name}</h4>
                        <p className="text-sm text-gray-500">
                          {dataset.sample_count.toLocaleString()} samples • {(dataset.token_count / 1000).toFixed(1)}K tokens
                        </p>
                      </div>
                      {selectedDataset === dataset.id && (
                        <CheckCircle2 className="w-5 h-5 text-primary-600" />
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {step === 'model' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Base Model *</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={baseModel}
                  onChange={(e) => {
                    setBaseModelTouched(true);
                    setBaseModel(e.target.value);
                  }}
                >
                  <option value="">Select a model</option>
                  {baseModelsData?.models.map((model) => (
                    <option key={model.name} value={model.name} disabled={!model.is_available}>
                      {model.display_name} {model.parameters && `(${model.parameters})`}
                      {!model.is_available && ' - Not available'}
                    </option>
                  ))}
                </select>
                {!baseModelTouched && baseModel && (
                  <p className="text-xs text-gray-500 mt-2">Recommended default selected. You can change it.</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Training Method</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={method}
                  onChange={(e) => {
                    setMethodTouched(true);
                    setMethod(e.target.value as TrainingMethod);
                  }}
                >
                  <option value="lora">LoRA (Recommended)</option>
                  <option value="qlora">QLoRA (Memory Efficient)</option>
                  <option value="full_finetune" disabled>
                    Full fine-tuning (Not supported yet)
                  </option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Backend</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={backend}
                  onChange={(e) => setBackend(e.target.value as TrainingBackend)}
                >
                  {(backendsData?.backends || []).length === 0 ? (
                    <option value="local">No backends available</option>
                  ) : (
                    backendsData?.backends.map((b) => (
                      <option key={b.name} value={b.name} disabled={!b.is_available}>
                        {b.display_name}
                        {!b.is_available && ' - Not available'}
                      </option>
                    ))
                  )}
                </select>
                {backend === 'simulated' && (
                  <p className="text-xs text-gray-500 mt-2">
                    Simulated backend creates a demo adapter (no real weights) but supports real-time progress + model registry.
                  </p>
                )}
              </div>
            </div>
          )}

          {step === 'config' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Job Name *</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={name}
                  onChange={(e) => {
                    setNameTouched(true);
                    setName(e.target.value);
                  }}
                  placeholder="My Training Job"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={2}
                  value={description}
                  onChange={(e) => {
                    setDescriptionTouched(true);
                    setDescription(e.target.value);
                  }}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Hyperparameters</label>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-500">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={hyperparameters.learning_rate}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        learning_rate: parseFloat(e.target.value),
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Epochs</label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={hyperparameters.num_epochs}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        num_epochs: parseInt(e.target.value),
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Batch Size</label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={hyperparameters.batch_size}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        batch_size: parseInt(e.target.value),
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">LoRA Rank (r)</label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={hyperparameters.lora_r}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        lora_r: parseInt(e.target.value),
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">LoRA Alpha</label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={hyperparameters.lora_alpha}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        lora_alpha: parseInt(e.target.value),
                      })}
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">LoRA Dropout</label>
                    <input
                      type="number"
                      step="0.01"
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      value={hyperparameters.lora_dropout}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        lora_dropout: parseFloat(e.target.value),
                      })}
                    />
                  </div>
                </div>
              </div>

              {/* Summary */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Summary</h4>
                <div className="text-sm text-gray-600 space-y-1">
                  <p><span className="text-gray-500">Base Model:</span> {baseModel}</p>
                  <p><span className="text-gray-500">Method:</span> {method.toUpperCase()}</p>
                  <p><span className="text-gray-500">Backend:</span> {backend}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-gray-200 flex justify-between">
          <Button
            variant="secondary"
            onClick={() => {
              if (step === 'model') setStep('dataset');
              else if (step === 'config') setStep('model');
              else onClose();
            }}
          >
            {step === 'dataset' ? 'Cancel' : 'Back'}
          </Button>

          <Button
            onClick={() => {
              if (step === 'dataset') {
                if (!selectedDataset) {
                  toast.error('Select a dataset');
                  return;
                }
                setStep('model');
              } else if (step === 'model') {
                if (!baseModel) {
                  toast.error('Select a base model');
                  return;
                }
                setStep('config');
              } else {
                handleSubmit();
              }
            }}
            disabled={isSubmitting}
          >
            {step === 'config' ? (isSubmitting ? 'Creating...' : 'Create Job') : 'Next'}
          </Button>
        </div>
      </div>
    </div>
  );
};

// ==================== Models Tab ====================

const ModelsTab: React.FC = () => {
  const [selectedAdapter, setSelectedAdapter] = useState<ModelAdapter | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [testPrompt, setTestPrompt] = useState('');
  const [testResult, setTestResult] = useState<string>('');
  const [isTesting, setIsTesting] = useState(false);
  const [isUsingInChat, setIsUsingInChat] = useState(false);
  const [selectedEvalTemplateId, setSelectedEvalTemplateId] = useState<string>('');
  const [judgeModel, setJudgeModel] = useState<string>('');
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const { data: adaptersData, isLoading, refetch } = useQuery(
    ['model-adapters', statusFilter],
    () => apiClient.listModelAdapters({
      status: statusFilter || undefined,
      page_size: 50,
    }),
    { refetchInterval: 10000 }
  );

  const { data: deployedModels } = useQuery(
    ['deployed-models'],
    () => apiClient.getDeployedModels()
  );

  const { data: evalTemplatesData } = useQuery(
    ['training-eval-templates'],
    () => apiClient.listTrainingEvalTemplates(),
    { staleTime: 60000 }
  );

  useEffect(() => {
    if (selectedEvalTemplateId) return;
    const templates = evalTemplatesData?.templates || [];
    if (templates.length === 0) return;
    const preferred = templates.find((t: any) => t.id === 'perf_regression_triage_v1');
    setSelectedEvalTemplateId((preferred || templates[0]).id);
  }, [evalTemplatesData?.templates, selectedEvalTemplateId]);

  const deployMutation = useMutation(
    (adapterId: string) => apiClient.deployModelAdapter(adapterId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['model-adapters']);
        queryClient.invalidateQueries(['deployed-models']);
        toast.success('Model deployed to Ollama');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Deploy failed');
      },
    }
  );

  const undeployMutation = useMutation(
    (adapterId: string) => apiClient.undeployModelAdapter(adapterId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['model-adapters']);
        queryClient.invalidateQueries(['deployed-models']);
        toast.success('Model undeployed');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Undeploy failed');
      },
    }
  );

  const deleteMutation = useMutation(
    (adapterId: string) => apiClient.deleteModelAdapter(adapterId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['model-adapters']);
        toast.success('Adapter deleted');
        setSelectedAdapter(null);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
      },
    }
  );

  const runEvalMutation = useMutation(
    (payload: { adapter_id: string; template_id: string; judge_model?: string | null }) =>
      apiClient.runTrainingEval(payload),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['model-adapters']);
        toast.success('Eval completed');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Eval failed');
      },
    }
  );

  const handleTest = async () => {
    if (!selectedAdapter || !testPrompt.trim()) return;

    setIsTesting(true);
    setTestResult('');
    try {
      const result = await apiClient.testModelAdapter(selectedAdapter.id, {
        prompt: testPrompt,
        max_tokens: 256,
        temperature: 0.7,
      });
      setTestResult(result.response);
    } catch (error: any) {
      toast.error(error.message || 'Test failed');
    } finally {
      setIsTesting(false);
    }
  };

  const handleUseInChat = async (adapter: ModelAdapter) => {
    setIsUsingInChat(true);
    try {
      let current = adapter;
      if (!current.is_deployed) {
        current = await apiClient.deployModelAdapter(current.id);
        queryClient.invalidateQueries(['model-adapters']);
        queryClient.invalidateQueries(['deployed-models']);
        setSelectedAdapter(current);
      }

      const ollamaModelName =
        current.deployment_config?.ollama_model_name || (current as any).ollama_model_name;
      if (!ollamaModelName) {
        throw new Error('Deployed model name not available');
      }

      const title = `AI Hub — ${current.display_name || current.name}`;
      const session = await apiClient.createChatSession(title, {
        llm_task_providers: { chat: 'ollama' },
        llm_task_models: { chat: ollamaModelName },
        ai_hub: { adapter_id: current.id },
      });
      navigate(`/chat/${session.id}`);
    } catch (e: any) {
      toast.error(e?.message || 'Failed to start chat');
    } finally {
      setIsUsingInChat(false);
    }
  };

  const AdapterCard: React.FC<{ adapter: ModelAdapter }> = ({ adapter }) => {
    const statusConfig = ADAPTER_STATUS_CONFIG[adapter.status] || ADAPTER_STATUS_CONFIG.ready;
    const StatusIcon = statusConfig.icon;
    const isTraining = adapter.status === 'training';

    return (
      <div
        className={`bg-white border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
          selectedAdapter?.id === adapter.id ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
        }`}
        onClick={() => setSelectedAdapter(adapter)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-green-100 text-green-600">
              <Boxes className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 truncate max-w-[200px]">
                {adapter.display_name || adapter.name}
              </h3>
              <p className="text-xs text-gray-500">{adapter.base_model}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
            <StatusIcon className={`w-3 h-3 ${isTraining ? 'animate-spin' : ''}`} />
            <span className="capitalize">{adapter.status}</span>
          </div>
        </div>

        <div className="flex items-center gap-3 mb-3">
          <span className="text-xs text-gray-500">
            {adapter.adapter_type.toUpperCase()}
          </span>
          {adapter.adapter_size && (
            <span className="text-xs text-gray-500">
              {(adapter.adapter_size / 1024 / 1024).toFixed(1)} MB
            </span>
          )}
        </div>

        {/* Deploy status */}
        {adapter.is_deployed && (
          <div className="flex items-center gap-1 text-xs text-blue-600 bg-blue-50 rounded px-2 py-1">
            <Rocket className="w-3 h-3" />
            <span>Deployed to Ollama</span>
          </div>
        )}

        {/* Tags */}
        {adapter.tags && adapter.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {adapter.tags.slice(0, 3).map((tag, idx) => (
              <span key={idx} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  const AdapterDetailPanel: React.FC<{ adapter: ModelAdapter }> = ({ adapter }) => {
    const statusConfig = ADAPTER_STATUS_CONFIG[adapter.status] || ADAPTER_STATUS_CONFIG.ready;
    const StatusIcon = statusConfig.icon;
    const canDeploy = adapter.status === 'ready' && !adapter.is_deployed;
    const canUndeploy = adapter.is_deployed;
    const evalRuns = (adapter.training_metrics as any)?.eval_runs;
    const lastEval = Array.isArray(evalRuns) && evalRuns.length > 0 ? evalRuns[evalRuns.length - 1] : null;

    return (
      <div className="bg-white border border-gray-200 rounded-lg h-full overflow-hidden flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-100 text-green-600">
                <Boxes className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">{adapter.display_name || adapter.name}</h2>
                <p className="text-sm text-gray-500">{adapter.base_model}</p>
              </div>
            </div>
            <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${statusConfig.bgColor} ${statusConfig.color}`}>
              <StatusIcon className="w-4 h-4" />
              <span className="font-medium capitalize">{adapter.status}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-3">
            {canDeploy && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => deployMutation.mutate(adapter.id)}
                disabled={deployMutation.isLoading}
              >
                <Rocket className="w-4 h-4 mr-1" />
                Deploy
              </Button>
            )}
            {canUndeploy && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => undeployMutation.mutate(adapter.id)}
                disabled={undeployMutation.isLoading}
              >
                <Server className="w-4 h-4 mr-1" />
                Undeploy
              </Button>
            )}
            <Button
              size="sm"
              variant="secondary"
              onClick={() => handleUseInChat(adapter)}
              disabled={isUsingInChat || deployMutation.isLoading}
              title="Creates a new chat session that uses this model for the chat task"
            >
              <MessageCircle className="w-4 h-4 mr-1" />
              Use in Chat
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (window.confirm('Delete this adapter?')) {
                  deleteMutation.mutate(adapter.id);
                }
              }}
              disabled={adapter.is_deployed || deleteMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {adapter.description && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-1">Description</h3>
              <p className="text-sm text-gray-600">{adapter.description}</p>
            </div>
          )}

          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Type</p>
              <p className="font-semibold uppercase">{adapter.adapter_type}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Size</p>
              <p className="font-semibold">
                {adapter.adapter_size ? `${(adapter.adapter_size / 1024 / 1024).toFixed(1)} MB` : '-'}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Version</p>
              <p className="font-semibold">{adapter.version}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500">Usage Count</p>
              <p className="font-semibold">{adapter.usage_count}</p>
            </div>
          </div>

          {/* Training metrics */}
          {adapter.training_metrics && Object.keys(adapter.training_metrics).length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Training Metrics</h3>
              <div className="bg-gray-50 rounded-lg p-3 text-sm">
                {Object.entries(adapter.training_metrics).filter(([k]) => k !== 'eval_runs').map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-500">{key}:</span>
                    <span className="font-medium">{typeof value === 'number' ? value.toFixed(4) : String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Eval (pluggable templates) */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Eval</h3>
            {lastEval && (
              <div className="mb-3 bg-gray-50 border border-gray-200 rounded-lg p-3 text-sm">
                <div className="flex items-center justify-between">
                  <div className="font-medium text-gray-800">Last eval</div>
                  <div className="text-gray-700">
                    Score: <span className="font-semibold">{Number(lastEval.avg_score).toFixed(2)}</span> / 5
                  </div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {lastEval.template_id} • {lastEval.num_cases} cases • judge: {lastEval.judge_model}
                </div>
              </div>
            )}

            <div className="space-y-2">
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={selectedEvalTemplateId}
                onChange={(e) => setSelectedEvalTemplateId(e.target.value)}
              >
                <option value="">Select an eval template</option>
                {(evalTemplatesData?.templates || []).map((t: any) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
              </select>

              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                placeholder={`Judge model (default: ${adapter.base_model})`}
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
              />

              <Button
                size="sm"
                variant="primary"
                onClick={() => {
                  if (!selectedEvalTemplateId) {
                    toast.error('Select an eval template');
                    return;
                  }
                  runEvalMutation.mutate({
                    adapter_id: adapter.id,
                    template_id: selectedEvalTemplateId,
                    judge_model: judgeModel.trim() ? judgeModel.trim() : null,
                  });
                }}
                disabled={!adapter.is_deployed || runEvalMutation.isLoading}
              >
                {runEvalMutation.isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                    Running…
                  </>
                ) : (
                  <>
                    <Activity className="w-4 h-4 mr-1" />
                    Run Eval
                  </>
                )}
              </Button>
              {!adapter.is_deployed && (
                <p className="text-xs text-gray-500">Deploy the adapter before running eval.</p>
              )}
            </div>
          </div>

          {/* Test inference */}
          {adapter.is_deployed && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Test Inference</h3>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm mb-2"
                rows={2}
                placeholder="Enter a test prompt..."
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
              />
              <Button
                size="sm"
                onClick={handleTest}
                disabled={isTesting || !testPrompt.trim()}
              >
                {isTesting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                    Testing...
                  </>
                ) : (
                  <>
                    <Zap className="w-4 h-4 mr-1" />
                    Run Test
                  </>
                )}
              </Button>
              {testResult && (
                <div className="mt-3 bg-gray-50 rounded-lg p-3 text-sm">
                  <p className="text-xs text-gray-500 mb-1">Response:</p>
                  <p className="whitespace-pre-wrap">{testResult}</p>
                </div>
              )}
            </div>
          )}

          <div className="text-xs text-gray-500 space-y-1">
            <p>Created: {new Date(adapter.created_at).toLocaleString()}</p>
            <p>
              Updated: {adapter.updated_at ? new Date(adapter.updated_at).toLocaleString() : '-'}
            </p>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Deployed models indicator */}
      {deployedModels && deployedModels.models.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
          <div className="flex items-center gap-2">
            <Rocket className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-700 font-medium">
              {deployedModels.models.length} model(s) deployed to Ollama
            </span>
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex gap-3">
          <select
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="">All Status</option>
            <option value="training">Training</option>
            <option value="ready">Ready</option>
            <option value="deployed">Deployed</option>
            <option value="failed">Failed</option>
          </select>
          <Button variant="ghost" size="sm" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        <div className="w-2/3 overflow-y-auto">
          {isLoading ? (
            <div className="flex justify-center items-center h-full">
              <LoadingSpinner />
            </div>
          ) : adaptersData?.adapters.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <Boxes className="w-12 h-12 mb-3 text-gray-400" />
              <p className="text-lg font-medium">No trained models yet</p>
              <p className="text-sm">Complete a training job to see models here</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {adaptersData?.adapters.map((adapter) => (
                <AdapterCard key={adapter.id} adapter={adapter} />
              ))}
            </div>
          )}
        </div>

        <div className="w-1/3">
          {selectedAdapter ? (
            <AdapterDetailPanel adapter={selectedAdapter} />
          ) : (
            <div className="bg-gray-50 border border-gray-200 rounded-lg h-full flex flex-col items-center justify-center text-gray-500">
              <Eye className="w-10 h-10 mb-3 text-gray-400" />
              <p className="font-medium">Select a model</p>
              <p className="text-sm">Click on a model to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIHubPage;
