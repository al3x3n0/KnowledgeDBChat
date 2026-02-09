/**
 * Autonomous Agents Page
 *
 * Manage and monitor autonomous agent jobs that run independently
 * to accomplish goals like research, monitoring, and analysis.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Bot,
  Play,
  Pause,
  XCircle,
  RotateCcw,
  Plus,
  Trash2,
  Eye,
  Clock,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Search,
  BookOpen,
  Activity,
  BarChart3,
  FileText,
  ChevronRight,
  RefreshCw,
  Target,
  Zap,
  Calendar,
  Cpu,
  MessageSquare,
  Settings,
  Layers,
  Link2,
  GitBranch,
  Download,
  FileDown,
  Brain,
  Sparkles,
  Lightbulb,
  Inbox,
  ThumbsUp,
  ThumbsDown,
} from 'lucide-react';
import toast from 'react-hot-toast';
import { apiClient } from '../services/api';
import type {
  AgentJob,
  AgentJobCreate,
  AgentJobFromTemplate,
  AgentJobTemplate,
  AgentJobStats,
  AgentJobStatus,
  AgentJobType,
  AgentJobChainDefinition,
  AgentJobChainStatus,
  AgentJobFromChainCreate,
  ResearchInboxItem,
  ResearchInboxItemStatus,
} from '../types';
import Button from '../components/common/Button';
import LoadingSpinner from '../components/common/LoadingSpinner';

// Job type icons and labels
const JOB_TYPE_CONFIG: Record<AgentJobType, { icon: React.ComponentType<any>; label: string; color: string }> = {
  research: { icon: BookOpen, label: 'Research', color: 'text-blue-600 bg-blue-100' },
  monitor: { icon: Activity, label: 'Monitor', color: 'text-green-600 bg-green-100' },
  analysis: { icon: BarChart3, label: 'Analysis', color: 'text-purple-600 bg-purple-100' },
  synthesis: { icon: Layers, label: 'Synthesis', color: 'text-orange-600 bg-orange-100' },
  knowledge_expansion: { icon: Zap, label: 'Knowledge Expansion', color: 'text-yellow-600 bg-yellow-100' },
  data_analysis: { icon: BarChart3, label: 'Data Analysis', color: 'text-indigo-600 bg-indigo-100' },
  custom: { icon: Settings, label: 'Custom', color: 'text-gray-600 bg-gray-100' },
};

// Status badges
const STATUS_CONFIG: Record<AgentJobStatus, { color: string; bgColor: string; icon: React.ComponentType<any> }> = {
  pending: { color: 'text-yellow-700', bgColor: 'bg-yellow-100', icon: Clock },
  running: { color: 'text-blue-700', bgColor: 'bg-blue-100', icon: Loader2 },
  paused: { color: 'text-orange-700', bgColor: 'bg-orange-100', icon: Pause },
  completed: { color: 'text-green-700', bgColor: 'bg-green-100', icon: CheckCircle2 },
  failed: { color: 'text-red-700', bgColor: 'bg-red-100', icon: AlertCircle },
  cancelled: { color: 'text-gray-700', bgColor: 'bg-gray-100', icon: XCircle },
};

const AutonomousAgentsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'jobs' | 'templates' | 'chains' | 'inbox' | 'create'>('jobs');
  const [selectedJob, setSelectedJob] = useState<AgentJob | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [swarmOnlyFilter, setSwarmOnlyFilter] = useState<boolean>(false);
  const [swarmSortBy, setSwarmSortBy] = useState<string>('created_desc');
  const [swarmMinConsensus, setSwarmMinConsensus] = useState<number>(0);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCustomerResearchModal, setShowCustomerResearchModal] = useState(false);
  const [showInboxMonitorModal, setShowInboxMonitorModal] = useState(false);
  const [showMonitorProfilesModal, setShowMonitorProfilesModal] = useState(false);
  const [createFromTemplate, setCreateFromTemplate] = useState<AgentJobTemplate | null>(null);
  const [startFromChain, setStartFromChain] = useState<AgentJobChainDefinition | null>(null);
  const [selectedChainStatus, setSelectedChainStatus] = useState<AgentJobChainStatus | null>(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportingJob, setExportingJob] = useState<AgentJob | null>(null);

  const [inboxStatusFilter, setInboxStatusFilter] = useState<ResearchInboxItemStatus | ''>('');
  const [inboxTypeFilter, setInboxTypeFilter] = useState<string>('');
  const [inboxSearch, setInboxSearch] = useState<string>('');
  const [selectedInboxIds, setSelectedInboxIds] = useState<Record<string, boolean>>({});

  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();
  const [creatingPluginId, setCreatingPluginId] = useState<string | null>(null);
  const [enableAfterCreate, setEnableAfterCreate] = useState(true);

  // Fetch jobs
  const { data: jobsData, isLoading: jobsLoading, refetch: refetchJobs } = useQuery(
    ['agent-jobs', statusFilter, typeFilter, swarmOnlyFilter, swarmSortBy, swarmMinConsensus],
    () => apiClient.listAgentJobs({
      status: statusFilter || undefined,
      job_type: typeFilter || undefined,
      swarm_only: swarmOnlyFilter || undefined,
      swarm_min_consensus: swarmMinConsensus > 0 ? swarmMinConsensus : undefined,
      sort_by: swarmSortBy || undefined,
      page_size: 50,
    }),
    {
      refetchInterval: 10000, // Auto-refresh every 10 seconds
    }
  );

  // Deep-link: /autonomous-agents?job=<id>
  useEffect(() => {
    const jobId = new URLSearchParams(location.search).get('job');
    if (!jobId) return;
    const jobs = (jobsData as any)?.jobs || [];
    const match = jobs.find((j: any) => String(j.id) === String(jobId));
    if (match) {
      setSelectedJob(match);
      setActiveTab('jobs');
    }
  }, [location.search, jobsData]);

  // Fetch stats
  const { data: stats } = useQuery(
    ['agent-jobs-stats'],
    () => apiClient.getAgentJobStats(),
    {
      refetchInterval: 30000,
    }
  );

  // Fetch templates
  const { data: templatesData } = useQuery(
    ['agent-job-templates'],
    () => apiClient.listAgentJobTemplates()
  );

  // Fetch chain definitions
  const { data: chainsData } = useQuery(
    ['agent-job-chains'],
    () => apiClient.listChainDefinitions()
  );

  // Research Inbox
  const { data: inboxStats } = useQuery(
    ['research-inbox-stats'],
    () => apiClient.getResearchInboxStats(),
    {
      refetchInterval: 20000,
    }
  );

  const { data: inboxData, isLoading: inboxLoading, refetch: refetchInbox } = useQuery(
    ['research-inbox', inboxStatusFilter, inboxTypeFilter, inboxSearch],
    () =>
      apiClient.listResearchInboxItems({
        status: inboxStatusFilter || undefined,
        item_type: inboxTypeFilter || undefined,
        q: inboxSearch.trim() || undefined,
        limit: 100,
        offset: 0,
      }),
    {
      enabled: activeTab === 'inbox',
      refetchInterval: 15000,
    }
  );

  const { data: monitorProfiles, isLoading: monitorProfilesLoading, refetch: refetchMonitorProfiles } = useQuery(
    ['research-monitor-profiles'],
    () => apiClient.listResearchMonitorProfiles(),
    {
      enabled: showMonitorProfilesModal,
      staleTime: 30000,
    }
  );

  // View chain status
  const viewChainStatus = async (jobId: string) => {
    try {
      const status = await apiClient.getChainStatus(jobId);
      setSelectedChainStatus(status);
    } catch (error) {
      console.error('Failed to load chain status:', error);
      toast.error('Failed to load chain status');
    }
  };

  const chainExperimentStopInfo = useMemo(() => {
    const cs = selectedChainStatus;
    if (!cs || !Array.isArray(cs.jobs)) return null;

    let found: any = null;
    for (let i = cs.jobs.length - 1; i >= 0; i--) {
      const job: any = cs.jobs[i];
      const results = job?.results;
      const stop = results?.experiment_loop_stop;
      if (stop && typeof stop === 'object') {
        found = { stop, job };
        break;
      }
    }
    if (!found) return null;

    let noteId: string | null = null;
    for (const j of cs.jobs as any[]) {
      const cfg = j?.config;
      const id = String(cfg?.research_note_id || cfg?.note_id || '').trim();
      if (id) {
        noteId = id;
        break;
      }
    }

    return {
      reason: String(found.stop?.reason || '').trim(),
      atRunId: String(found.stop?.at_run_id || '').trim(),
      stoppedByJobId: String(found.job?.id || '').trim(),
      noteId,
    };
  }, [selectedChainStatus]);

  // Mutations
  const actionMutation = useMutation(
    ({ jobId, action }: { jobId: string; action: 'pause' | 'resume' | 'cancel' | 'restart' }) =>
      apiClient.performAgentJobAction(jobId, action),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success(`Job ${job.status}`);
        if (selectedJob?.id === job.id) {
          setSelectedJob(job);
        }
      },
      onError: (error: any) => {
        toast.error(error.message || 'Action failed');
      },
    }
  );

  const deleteMutation = useMutation(
    (jobId: string) => apiClient.deleteAgentJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Job deleted');
        setSelectedJob(null);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Delete failed');
      },
    }
  );

  const updateInboxItemMutation = useMutation(
    ({ itemId, data }: { itemId: string; data: { status?: ResearchInboxItemStatus; feedback?: string; metadata_patch?: Record<string, any> } }) =>
      apiClient.updateResearchInboxItem(itemId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Update failed');
      },
    }
  );

  const bulkUpdateInboxMutation = useMutation(
    ({ itemIds, data }: { itemIds: string[]; data: { status?: ResearchInboxItemStatus; feedback?: string } }) =>
      apiClient.bulkUpdateResearchInboxItems({ item_ids: itemIds, ...data }),
    {
      onSuccess: (res) => {
        queryClient.invalidateQueries(['research-inbox']);
        queryClient.invalidateQueries(['research-inbox-stats']);
        setSelectedInboxIds({});
        toast.success(`Updated ${res.updated} items`);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Bulk update failed');
      },
    }
  );

  const upsertMonitorProfileMutation = useMutation(
    (data: { customer?: string; muted_tokens?: string[]; muted_patterns?: string[]; notes?: string; merge_lists?: boolean }) =>
      apiClient.upsertResearchMonitorProfile(data),
    {
      onSuccess: () => {
        toast.success('Monitor profile updated');
        queryClient.invalidateQueries(['research-monitor-profiles']);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to update monitor profile');
      },
    }
  );

  const extractReposMutation = useMutation(
    (itemId: string) => apiClient.extractReposForInboxItem(itemId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['research-inbox']);
        toast.success('Repo links extracted');
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to extract repos');
      },
    }
  );

  const ingestRepoMutation = useMutation(
    (payload: { provider: 'github' | 'gitlab'; repo: string }) =>
      apiClient.requestGitRepository({
        provider: payload.provider,
        repositories: [payload.repo],
        include_files: true,
        include_issues: false,
        include_pull_requests: false,
        include_wiki: false,
        incremental_files: true,
        use_gitignore: true,
        max_pages: 5,
        auto_sync: true,
      }),
    {
      onSuccess: (src) => {
        toast.success(`Repo ingestion started: ${src.name}`);
        // show in documents sources list
        navigate(`/documents`);
      },
      onError: (error: any) => {
        toast.error(error?.response?.data?.detail || error.message || 'Failed to ingest repo');
      },
    }
  );

  const runPaperRepoCodeAgent = async (item: ResearchInboxItem) => {
    try {
      // Ensure repos are extracted
      let repos = (item.metadata as any)?.repos;
      if (!Array.isArray(repos) || repos.length === 0) {
        const res = await apiClient.extractReposForInboxItem(item.id);
        repos = res.repos;
      }
      const githubRepos = (repos || []).filter((r: any) => String(r?.provider) === 'github');
      if (githubRepos.length === 0) {
        toast.error('No GitHub repos found for this paper yet');
        return;
      }
      const defaultRepo = String(githubRepos[0].repo || '').trim();
      let chosenRepo = defaultRepo;
      if (githubRepos.length > 1) {
        const options = githubRepos
          .map((r: any, idx: number) => `${idx + 1}) ${String(r?.repo || '').trim()}`)
          .filter((s: string) => !!s && s.length > 4)
          .slice(0, 12)
          .join('\n');
        chosenRepo = (window.prompt(`Multiple GitHub repos found:\n${options}\n\nEnter repo to ingest:`, defaultRepo) || '').trim();
        if (!chosenRepo) return;
      }

      const goal = `Implement or integrate the paper's repository changes relevant to our product. Start from the ingested repo and produce a minimal patch.\n\nPaper: ${item.title}\n\nAbstract:\n${item.summary || ''}`.slice(
        0,
        1600
      );

      const chains = ((chainsData as any)?.chains || []) as any[];
      const chain = chains.find((c: any) => c?.name === 'arxiv_repo_code_patch_chain');
      if (!chain?.id) {
        toast.error('Chain definition not found (arxiv_repo_code_patch_chain)');
        return;
      }

      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: `Paper→Repo→Code — ${new Date().toLocaleDateString()}`,
        variables: {
          inbox_item_id: item.id,
          provider: 'github',
          repo: chosenRepo,
          goal,
        },
        config_overrides: {
          inbox_item_id: item.id,
          provider: 'github',
          repo: chosenRepo,
          // help code patch proposer pick relevant files
          search_query: `${item.title}\n${item.summary || ''}`.slice(0, 500),
        },
        start_immediately: true,
      });
      setActiveTab('jobs');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to start chain');
    }
  };

  const runPaperAlgorithmProject = async (item: ResearchInboxItem, requestedBehavioralCheck: boolean, entrypoint: string) => {
    try {
      const chains = ((chainsData as any)?.chains || []) as any[];
      let preferredChainName = 'arxiv_algorithm_project_chain';
      let repos = (item.metadata as any)?.repos;
      if (!Array.isArray(repos) || repos.length === 0) {
        try {
          const res = await apiClient.extractReposForInboxItem(item.id);
          repos = res.repos;
        } catch {
          repos = repos || [];
        }
      }
      const hasGithubRepo = Array.isArray(repos) && repos.some((r: any) => String(r?.provider || '').toLowerCase() === 'github');
      if (hasGithubRepo) preferredChainName = 'arxiv_repo_algorithm_project_chain';

      const chain = chains.find((c: any) => c?.name === preferredChainName);
      if (!chain?.id) {
        toast.error(`Chain definition not found (${preferredChainName})`);
        return;
      }
      const allowBehavioral = !!requestedBehavioralCheck && unsafeExecBadge.status === 'ready';
      if (requestedBehavioralCheck && !allowBehavioral) {
        toast('Behavioral demo run requested, but server is not ready (see badge)');
      }
      const ep = String(entrypoint || 'demo.py').trim() || 'demo.py';
      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: `Paper→Algorithm — ${new Date().toLocaleDateString()}`,
        variables: { inbox_item_id: item.id },
        config_overrides: {
          inbox_item_id: item.id,
          language: 'python',
          include_tests: true,
          behavioral_check: allowBehavioral,
          entrypoint: ep,
        },
        start_immediately: true,
      });
      setActiveTab('jobs');
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to start algorithm implementation');
    }
  };

  const createMutation = useMutation(
    (data: AgentJobCreate) => apiClient.createAgentJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Job created');
        setShowCreateModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Create failed');
      },
    }
  );

  const createInboxMonitorMutation = useMutation(
    (data: AgentJobCreate) => apiClient.createAgentJob(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Monitor created');
        setShowInboxMonitorModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Create failed');
      },
    }
  );

  const createFromTemplateMutation = useMutation(
    (data: AgentJobFromTemplate) => apiClient.createAgentJobFromTemplate(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Job created from template');
        setCreateFromTemplate(null);
        setShowCustomerResearchModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Create failed');
      },
    }
  );

  const createFromChainMutation = useMutation(
    (data: AgentJobFromChainCreate) => apiClient.createJobFromChain(data),
    {
      onSuccess: (job) => {
        queryClient.invalidateQueries(['agent-jobs']);
        queryClient.invalidateQueries(['agent-jobs-stats']);
        toast.success('Chain started');
        setStartFromChain(null);
        setShowCustomerResearchModal(false);
        setActiveTab('jobs');
        setSelectedJob(job);
      },
      onError: (error: any) => {
        toast.error(error.message || 'Failed to start chain');
      },
    }
  );

  const [paperAlgoRunDemo, setPaperAlgoRunDemo] = useState<Record<string, boolean>>({});
  const [paperAlgoEntrypoint, setPaperAlgoEntrypoint] = useState<Record<string, string>>({});
  const [paperAlgoEntrypointSavedAt, setPaperAlgoEntrypointSavedAt] = useState<Record<string, string>>({});
  const [paperAlgoEntrypointSaving, setPaperAlgoEntrypointSaving] = useState<Record<string, boolean>>({});
  const [paperAlgoEntrypointError, setPaperAlgoEntrypointError] = useState<Record<string, string>>({});

  const { data: myPreferences } = useQuery(['me-preferences'], () => apiClient.getMyPreferences(), {
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  });
  const updateMyPreferencesMutation = useMutation((updates: any) => apiClient.updateMyPreferences(updates), {
    onSuccess: () => {
      queryClient.invalidateQueries(['me-preferences']);
      toast.success('Preferences updated');
    },
    onError: (e: any) => {
      toast.error(e?.response?.data?.detail || e?.message || 'Failed to update preferences');
    },
  });
  const paperAlgoDefaultRunDemoCheck = (myPreferences as any)?.paper_algo_default_run_demo_check === true;

  const { data: unsafeExecAvailability } = useQuery(
    ['unsafe-exec-availability'],
    () => apiClient.getUnsafeExecAvailability(),
    { staleTime: 30_000, refetchOnWindowFocus: false }
  );

  const unsafeExecBadge = useMemo(() => {
    const avail: any = unsafeExecAvailability as any;
    const enabled = !!avail?.enabled;
    const backend = String(avail?.backend || 'subprocess');
    const dockerOk = backend !== 'docker' || (avail?.docker?.available === true && avail?.docker?.image_present === true);
    const status: 'ready' | 'blocked' | 'off' = enabled && dockerOk ? 'ready' : enabled ? 'blocked' : 'off';
    const label =
      status === 'ready'
        ? 'demo-check ready'
        : status === 'blocked'
          ? 'demo-check not ready'
          : 'demo-check off';
    const title =
      status === 'ready'
        ? `Behavioral demo check available (backend: ${backend})`
        : status === 'blocked'
          ? `Behavioral demo check enabled but not ready (backend: ${backend})`
          : 'Behavioral demo check disabled on server';
    const color =
      status === 'ready' ? 'bg-green-500' : status === 'blocked' ? 'bg-amber-500' : 'bg-gray-400';
    return { status, label, title, color };
  }, [unsafeExecAvailability]);

  const paperAlgoDefaultToggleTitle =
    unsafeExecBadge.status === 'ready'
      ? 'Set the default for new items in this session'
      : 'Server not ready for demo checks (see badge)';

  const normalizeEntrypoint = (raw: string): { ok: boolean; value: string; error?: string } => {
    let v = String(raw || '').trim();
    if (!v) return { ok: true, value: 'demo.py' };
    v = v.replace(/\\/g, '/');
    while (v.startsWith('./')) v = v.slice(2);
    if (v.startsWith('/') || v.startsWith('~') || v.includes(':')) return { ok: false, value: v, error: 'Absolute paths not allowed' };
    if (v.split('/').some((p) => p === '..')) return { ok: false, value: v, error: "'..' not allowed" };
    if (/\s/.test(v)) return { ok: false, value: v, error: 'Whitespace not allowed' };
    if (!v.endsWith('.py')) return { ok: false, value: v, error: 'Must end with .py' };
    if (!/^[A-Za-z0-9._/\\-]+$/.test(v)) return { ok: false, value: v, error: 'Invalid characters' };
    return { ok: true, value: v };
  };

  // Format time duration
  const formatDuration = (startedAt?: string, completedAt?: string) => {
    if (!startedAt) return '-';
    const start = new Date(startedAt);
    const end = completedAt ? new Date(completedAt) : new Date();
    const diff = Math.floor((end.getTime() - start.getTime()) / 1000);
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
    return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m`;
  };

  // Render stats card
  const StatsCard: React.FC<{ title: string; value: string | number; icon: React.ComponentType<any>; color: string }> = ({
    title,
    value,
    icon: Icon,
    color,
  }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-2xl font-semibold mt-1">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );

  // Render job card
  const JobCard: React.FC<{ job: AgentJob }> = ({ job }) => {
    const typeConfig = JOB_TYPE_CONFIG[job.job_type as AgentJobType] || JOB_TYPE_CONFIG.custom;
    const statusConfig = STATUS_CONFIG[job.status as AgentJobStatus] || STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;
    const rawFanIn = (job.results as any)?.swarm_fan_in;
    const swarmSummary = ((job as any)?.swarm_summary && typeof (job as any)?.swarm_summary === 'object')
      ? (job as any).swarm_summary
      : null;
    const goalContractSummary = ((job as any)?.goal_contract_summary && typeof (job as any)?.goal_contract_summary === 'object')
      ? (job as any).goal_contract_summary
      : (((job.results as any)?.goal_contract && typeof (job.results as any)?.goal_contract === 'object') ? (job.results as any).goal_contract : null);
    const contractEnabled = Boolean(goalContractSummary?.enabled);
    const contractSatisfied = contractEnabled ? Boolean(goalContractSummary?.satisfied) : true;
    const contractMissingCount = Number(
      goalContractSummary?.missing_count ??
      (Array.isArray(goalContractSummary?.missing) ? goalContractSummary.missing.length : 0)
    );
    const approvalCheckpoint = ((job as any)?.approval_checkpoint && typeof (job as any)?.approval_checkpoint === 'object')
      ? (job as any).approval_checkpoint
      : (((job.results as any)?.approval_checkpoint && typeof (job.results as any)?.approval_checkpoint === 'object')
          ? (job.results as any).approval_checkpoint
          : (((job.results as any)?.execution_strategy?.approval_checkpoints?.pending && typeof (job.results as any)?.execution_strategy?.approval_checkpoints?.pending === 'object')
              ? (job.results as any).execution_strategy.approval_checkpoints.pending
              : null));
    const hasSwarm = Boolean(swarmSummary || (rawFanIn && typeof rawFanIn === 'object'));
    const consensusCount = Number(
      swarmSummary?.consensus_count ??
      (Array.isArray(rawFanIn?.consensus_findings) ? rawFanIn.consensus_findings.length : 0)
    );
    const conflictCount = Number(
      swarmSummary?.conflict_count ??
      (Array.isArray(rawFanIn?.conflicts) ? rawFanIn.conflicts.length : 0)
    );
    const confidenceOverall = Number(
      swarmSummary?.confidence?.overall ??
      rawFanIn?.confidence?.overall ??
      0
    );

    return (
      <div
        className={`bg-white border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
          selectedJob?.id === job.id ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
        }`}
        onClick={() => setSelectedJob(job)}
      >
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-lg ${typeConfig.color}`}>
              <TypeIcon className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-medium text-gray-900 truncate max-w-[200px]">{job.name}</h3>
              <p className="text-xs text-gray-500">{typeConfig.label}</p>
            </div>
          </div>
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}>
            <StatusIcon className={`w-3 h-3 ${job.status === 'running' ? 'animate-spin' : ''}`} />
            <span className="capitalize">{job.status}</span>
          </div>
        </div>

        {/* Progress bar */}
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
                'bg-primary-500'
              }`}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>

        {/* Current phase */}
        {job.current_phase && (
          <p className="text-xs text-gray-600 mb-2 truncate">
            <span className="font-medium">Phase:</span> {job.current_phase}
          </p>
        )}

        {(approvalCheckpoint || contractEnabled) && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            {approvalCheckpoint && job.status === 'paused' && (
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-100">
                Awaiting approval
              </span>
            )}
            {contractEnabled && (
              <span
                className={`px-2 py-0.5 rounded-full text-[11px] font-medium border ${
                  contractSatisfied
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-100'
                    : 'bg-amber-50 text-amber-700 border-amber-100'
                }`}
              >
                {contractSatisfied ? 'Contract satisfied' : `Contract missing ${Math.max(0, contractMissingCount)}`}
              </span>
            )}
          </div>
        )}

        {/* Swarm quick chips */}
        {hasSwarm && (
          <div className="flex flex-wrap items-center gap-1.5 mt-1 mb-2">
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-100 hover:bg-indigo-100"
              title="Filter to swarm jobs"
              onClick={(e) => {
                e.stopPropagation();
                setSwarmOnlyFilter(true);
              }}
            >
              Swarm
            </button>
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] bg-emerald-50 text-emerald-700 border border-emerald-100 hover:bg-emerald-100"
              title="Sort by consensus and set minimum consensus threshold"
              onClick={(e) => {
                e.stopPropagation();
                setSwarmOnlyFilter(true);
                setSwarmSortBy('swarm_consensus_desc');
                setSwarmMinConsensus(Math.max(1, consensusCount));
              }}
            >
              Consensus {consensusCount}
            </button>
            <button
              type="button"
              className="px-2 py-0.5 rounded-full text-[11px] bg-amber-50 text-amber-700 border border-amber-100 hover:bg-amber-100"
              title="Sort by conflicts in swarm jobs"
              onClick={(e) => {
                e.stopPropagation();
                setSwarmOnlyFilter(true);
                setSwarmSortBy('swarm_conflicts_desc');
              }}
            >
              Conflicts {conflictCount}
            </button>
            {confidenceOverall > 0 && (
              <button
                type="button"
                className="px-2 py-0.5 rounded-full text-[11px] bg-sky-50 text-sky-700 border border-sky-100 hover:bg-sky-100"
                title="Sort by swarm confidence"
                onClick={(e) => {
                  e.stopPropagation();
                  setSwarmOnlyFilter(true);
                  setSwarmSortBy('swarm_confidence_desc');
                }}
              >
                Confidence {(confidenceOverall * 100).toFixed(0)}%
              </button>
            )}
          </div>
        )}

        {/* Chain indicator */}
        {(job.parent_job_id || job.chain_config) && (
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-100">
            <GitBranch className="w-3 h-3 text-purple-500" />
            <span className="text-xs text-purple-600">
              {job.parent_job_id ? `Step ${job.chain_depth + 1} in chain` : 'Chain root'}
              {job.chain_triggered && ' • Children triggered'}
            </span>
            <button
              className="ml-auto text-xs text-purple-600 hover:text-purple-800 flex items-center gap-1"
              onClick={(e) => {
                e.stopPropagation();
                viewChainStatus(job.id);
              }}
            >
              <Link2 className="w-3 h-3" />
              View Chain
            </button>
          </div>
        )}

        {/* Stats row */}
        <div className="flex items-center gap-4 text-xs text-gray-500 mt-2">
          <span className="flex items-center gap-1">
            <RefreshCw className="w-3 h-3" />
            {job.iteration}/{job.max_iterations}
          </span>
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3" />
            {job.tool_calls_used}/{job.max_tool_calls}
          </span>
          <span className="flex items-center gap-1">
            <MessageSquare className="w-3 h-3" />
            {job.llm_calls_used}/{job.max_llm_calls}
          </span>
        </div>
      </div>
    );
  };

  // Render job detail panel
  const JobDetailPanel: React.FC<{ job: AgentJob }> = ({ job }) => {
    const [logData, setLogData] = useState<{ entries: Array<Record<string, any>>; total: number } | null>(null);
    const [loadingLog, setLoadingLog] = useState(false);
    const [memoriesData, setMemoriesData] = useState<{
      memories: Array<{
        id: string;
        type: string;
        content: string;
        importance_score: number;
        tags: string[];
      }>;
      total: number;
    } | null>(null);
    const [loadingMemories, setLoadingMemories] = useState(false);
    const [showMemories, setShowMemories] = useState(false);
    const [extractingMemories, setExtractingMemories] = useState(false);

    const typeConfig = JOB_TYPE_CONFIG[job.job_type as AgentJobType] || JOB_TYPE_CONFIG.custom;
    const statusConfig = STATUS_CONFIG[job.status as AgentJobStatus] || STATUS_CONFIG.pending;
    const StatusIcon = statusConfig.icon;
    const TypeIcon = typeConfig.icon;
    const aiHubBundle = (job.results as any)?.ai_hub_bundle;
    const customerProfile = (job.results as any)?.customer_profile;
    const customerContext = (job.results as any)?.customer_context;
    const researchBundle = (job.results as any)?.research_bundle;
    const executiveDigest = (((job as any)?.executive_digest && typeof (job as any)?.executive_digest === 'object')
      ? (job as any).executive_digest
      : (((job.results as any)?.executive_digest && typeof (job.results as any)?.executive_digest === 'object')
          ? (job.results as any).executive_digest
          : null));
    const goalContractSummary = (((job as any)?.goal_contract_summary && typeof (job as any)?.goal_contract_summary === 'object')
      ? (job as any).goal_contract_summary
      : (((job.results as any)?.goal_contract && typeof (job.results as any)?.goal_contract === 'object')
          ? (job.results as any).goal_contract
          : null));
    const approvalCheckpoint = (((job as any)?.approval_checkpoint && typeof (job as any)?.approval_checkpoint === 'object')
      ? (job as any).approval_checkpoint
      : (((job.results as any)?.approval_checkpoint && typeof (job.results as any)?.approval_checkpoint === 'object')
          ? (job.results as any).approval_checkpoint
          : (((job.results as any)?.execution_strategy?.approval_checkpoints?.pending && typeof (job.results as any)?.execution_strategy?.approval_checkpoints?.pending === 'object')
              ? (job.results as any).execution_strategy.approval_checkpoints.pending
              : null)));
    const swarmSummary = useMemo(() => {
      const fromApi = (job as any)?.swarm_summary;
      if (fromApi && typeof fromApi === 'object') return fromApi as any;
      const fanIn = (job.results as any)?.swarm_fan_in;
      if (!fanIn || typeof fanIn !== 'object') return null;
      return {
        enabled: true,
        configured: true,
        fan_in_enabled: true,
        fan_in_group_id: String(fanIn?.fan_in_group_id || ''),
        roles: Array.isArray(fanIn?.roles) ? fanIn.roles : [],
        role_count: Array.isArray(fanIn?.roles) ? fanIn.roles.length : 0,
        expected_siblings: Number(fanIn?.expected_siblings || 0),
        received_siblings: Number(fanIn?.received_siblings || 0),
        terminal_siblings: Number(fanIn?.terminal_siblings || 0),
        consensus_count: Array.isArray(fanIn?.consensus_findings) ? fanIn.consensus_findings.length : 0,
        consensus_findings: (Array.isArray(fanIn?.consensus_findings) ? fanIn.consensus_findings : [])
          .map((r: any) => String(r?.finding || ''))
          .filter(Boolean),
        conflict_count: Array.isArray(fanIn?.conflicts) ? fanIn.conflicts.length : 0,
        conflicts: Array.isArray(fanIn?.conflicts) ? fanIn.conflicts : [],
        action_plan: Array.isArray(fanIn?.action_plan) ? fanIn.action_plan : [],
        confidence: fanIn?.confidence && typeof fanIn.confidence === 'object' ? fanIn.confidence : {},
      } as any;
    }, [job]);
    const [feedbackReasons, setFeedbackReasons] = useState<Record<string, string>>({});
    const [bulkReason, setBulkReason] = useState('');
    const [bulkSubmitting, setBulkSubmitting] = useState(false);
    const [detailsOpen, setDetailsOpen] = useState<Record<string, boolean>>({});
    const canSaveAsPlaybook = Boolean((job as any)?.chain_config?.child_jobs?.length) || Boolean((job as any)?.root_job_id) || Boolean((job as any)?.parent_job_id);

    const saveAsPlaybookMutation = useMutation(
      () => apiClient.saveAgentJobAsChain(String(job.id), {}),
      {
        onSuccess: () => {
          toast.success('Saved as playbook');
          queryClient.invalidateQueries(['agent-job-chains']);
          setActiveTab('chains');
        },
        onError: (e: any) => {
          toast.error(e?.message || 'Failed to save playbook');
        },
      }
    );

    const { data: feedbackData } = useQuery(
      ['agent-job', job.id, 'ai-hub', 'recommendation-feedback'],
      () => apiClient.listAIHubRecommendationFeedback(String(job.id)),
      { enabled: !!aiHubBundle, staleTime: 15000 }
    );

    const feedbackIndex = useMemo(() => {
      const idx: Record<string, any> = {};
      const items = (feedbackData as any)?.items || [];
      for (const it of items) {
        const key = `${it.workflow}:${it.item_type}:${it.item_id}`;
        idx[key] = it;
      }
      return idx;
    }, [feedbackData]);

    const applyAIHubBundle = async () => {
      const evalIds: string[] = aiHubBundle?.enabled_eval_templates || [];
      const presetIds: string[] = aiHubBundle?.enabled_dataset_presets || [];
      try {
        await apiClient.setEnabledAIHubEvalTemplates({ enabled: evalIds });
        await apiClient.setEnabledAIHubDatasetPresets({ enabled: presetIds });
        toast.success('AI Hub bundle applied');
        navigate('/ai-hub?tab=datasets');
      } catch (e: any) {
        toast.error(e?.message || 'Failed to apply bundle (admin required)');
      }
    };

    const copyText = async (text: string, label: string) => {
      try {
        await navigator.clipboard.writeText(text);
        toast.success(`${label} copied`);
      } catch (e) {
        toast.error(`Failed to copy ${label}`);
      }
    };

    const envText = aiHubBundle?.env
      ? [
          `AI_HUB_DATASET_ENABLED_PRESET_IDS=${aiHubBundle.env.AI_HUB_DATASET_ENABLED_PRESET_IDS || ''}`,
          `AI_HUB_EVAL_ENABLED_TEMPLATE_IDS=${aiHubBundle.env.AI_HUB_EVAL_ENABLED_TEMPLATE_IDS || ''}`,
        ].join('\n')
      : '';

    const documentArtifact = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.find((a) => a?.type === 'document' && (a?.id || a?.document_id));
    }, [job.output_artifacts]);

    const readingListArtifact = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.find((a) => a?.type === 'reading_list' && a?.id);
    }, [job.output_artifacts]);

    const arxivSourceArtifacts = useMemo(() => {
      const arts = (job.output_artifacts as any[]) || [];
      return arts.filter((a) => a?.type === 'document_source' && (a?.source_type === 'arxiv' || a?.sourceType === 'arxiv'));
    }, [job.output_artifacts]);

    const codePatchProposal = useMemo(() => {
      const fromResults = (job.results as any)?.code_patch;
      if (fromResults?.proposal_id) {
        return {
          proposal_id: String(fromResults.proposal_id),
          title: String(fromResults.title || 'Code Patch Proposal'),
          summary: fromResults.summary ? String(fromResults.summary) : '',
        };
      }
      const arts = (job.output_artifacts as any[]) || [];
      const art = arts.find((a) => a?.type === 'code_patch_proposal' && a?.id);
      if (art?.id) {
        return { proposal_id: String(art.id), title: String(art.title || 'Code Patch Proposal'), summary: '' };
      }
      return null;
    }, [job.output_artifacts, job.results]);

    const codePatchProposals = useMemo(() => {
      const seen = new Set<string>();
      const out: Array<{ proposal_id: string; title: string; summary: string }> = [];
      const hist = (job.results as any)?.code_patches;
      if (Array.isArray(hist)) {
        for (const p of hist) {
          const id = String(p?.proposal_id || '').trim();
          if (!id || seen.has(id)) continue;
          seen.add(id);
          out.push({
            proposal_id: id,
            title: String(p?.title || 'Code Patch Proposal'),
            summary: p?.summary ? String(p.summary) : '',
          });
        }
      }
      const cur = (job.results as any)?.code_patch;
      if (cur?.proposal_id) {
        const id = String(cur.proposal_id).trim();
        if (id && !seen.has(id)) {
          out.push({
            proposal_id: id,
            title: String(cur?.title || 'Code Patch Proposal'),
            summary: cur?.summary ? String(cur.summary) : '',
          });
        }
      }
      return out;
    }, [job.results]);

    const experimentRuns = useMemo(() => {
      const out: any[] = [];
      const hist = (job.results as any)?.experiment_runs;
      if (Array.isArray(hist)) out.push(...hist);
      const cur = (job.results as any)?.experiment_run;
      if (cur && typeof cur === 'object') out.push(cur);
      return out.filter(Boolean).slice(-5);
    }, [job.results]);

    const codePatchApply = useMemo(() => {
      const v = (job.results as any)?.code_patch_apply;
      if (v && typeof v === 'object') return v as any;
      return null;
    }, [job.results]);

    const codePatchKbApply = useMemo(() => {
      const v = (job.results as any)?.code_patch_kb_apply;
      if (v && typeof v === 'object') return v as any;
      return null;
    }, [job.results]);

    const generatedProject = useMemo(() => {
      const fromResults = (job.results as any)?.generated_project;
      if (fromResults?.source_id) {
        const behavioral = fromResults?.sanity_check?.behavioral;
        return {
          source_id: String(fromResults.source_id),
          source_name: String(fromResults.source_name || 'Generated project'),
          project_name: String(fromResults.project_name || fromResults.source_name || 'Generated project'),
          entrypoint: String(fromResults.entrypoint || 'demo.py'),
          file_count: Number(fromResults.file_count || 0),
          sanity_ok: fromResults?.sanity_check?.ok === true,
          sanity_errors_count: Array.isArray(fromResults?.sanity_check?.syntax_errors) ? fromResults.sanity_check.syntax_errors.length : 0,
          behavioral,
        };
      }
      const arts = (job.output_artifacts as any[]) || [];
      const art = arts.find((a) => a?.type === 'generated_project' && a?.source_id);
      if (art?.source_id) {
        return {
          source_id: String(art.source_id),
          source_name: String(art.title || 'Generated project'),
          project_name: String(art.title || 'Generated project'),
          entrypoint: 'demo.py',
          file_count: 0,
          sanity_ok: false,
          sanity_errors_count: 0,
          behavioral: null,
        };
      }
      return null;
    }, [job.output_artifacts, job.results]);

    const demoCheck = useMemo(() => {
      const fromResults = (job.results as any)?.demo_check;
      if (fromResults?.source_id) {
        return {
          source_id: String(fromResults.source_id),
          source_name: String(fromResults.source_name || ''),
          entrypoint: String(fromResults.entrypoint || 'demo.py'),
          ok: fromResults.ok === true,
          behavioral: fromResults.behavioral,
        };
      }
      return null;
    }, [job.results]);

    const { data: recentImports } = useQuery(
      ['arxiv-imports', 'recent'],
      () => apiClient.listArxivImports({ limit: 50, offset: 0 }),
      { staleTime: 30000 }
    );

    const arxivImportsFallback = useMemo(() => {
      if (arxivSourceArtifacts.length > 0) return [];
      const ids = new Set<string>();
      const arts = (job.output_artifacts as any[]) || [];
      for (const a of arts) {
        if (a?.type === 'arxiv_ingest_requested' && a?.source_id) ids.add(String(a.source_id));
      }
      const items = (recentImports as any)?.items || [];
      const found: any[] = [];
      for (const it of items) {
        if (ids.has(String(it?.id))) found.push(it);
      }
      return found;
    }, [arxivSourceArtifacts.length, job.output_artifacts, recentImports]);

    const desiredReadingListName = useMemo(() => {
      const cfgName = (job.config as any)?.reading_list_name;
      if (typeof cfgName === 'string' && cfgName.trim()) return cfgName.trim();
      return '';
    }, [job.config]);

    const { data: readingListsLookup } = useQuery(
      ['reading-lists', 'lookup', desiredReadingListName],
      () => apiClient.listReadingLists({ limit: 200, offset: 0 }),
      { enabled: !!desiredReadingListName && !readingListArtifact?.id, staleTime: 30000 }
    );

    const readingListByName = useMemo(() => {
      if (!desiredReadingListName) return null;
      const items = (readingListsLookup as any)?.items || [];
      const match = items.find((x: any) => String(x?.name || '').trim() === desiredReadingListName);
      return match || null;
    }, [readingListsLookup, desiredReadingListName]);

    const openDocument = (docId: string) => {
      if (!docId) return;
      navigate('/documents', { state: { openDocId: String(docId) } });
    };

    const openReadingList = (rlId: string) => {
      if (!rlId) return;
      navigate(`/reading-lists/${encodeURIComponent(String(rlId))}`);
    };

    const createPlugin = async (pluginType: 'dataset_preset' | 'eval_template', plugin: any) => {
      if (!plugin?.id) {
        toast.error('Plugin is missing id');
        return;
      }
      setCreatingPluginId(String(plugin.id));
      try {
        const res = await apiClient.createAIHubPlugin({ plugin_type: pluginType, plugin, overwrite: false });
        toast.success(`Created ${pluginType}: ${res.plugin_id}`);
        if (res.warnings && res.warnings.length > 0) {
          toast(res.warnings.join(' '), { duration: 6000 });
        }
        queryClient.invalidateQueries(['admin', 'ai-hub', 'eval-templates', 'all']);
        queryClient.invalidateQueries(['admin', 'ai-hub', 'dataset-presets', 'all']);

        if (enableAfterCreate) {
          if (pluginType === 'dataset_preset') {
            const current = await apiClient.getEnabledAIHubDatasetPresets();
            const enabled = (current as any)?.enabled || [];
            if (Array.isArray(enabled) && enabled.length > 0) {
              if (!enabled.includes(res.plugin_id)) {
                await apiClient.setEnabledAIHubDatasetPresets({ enabled: [...enabled, res.plugin_id] });
                toast.success('Preset enabled');
                queryClient.invalidateQueries(['admin', 'ai-hub', 'dataset-presets', 'enabled']);
                queryClient.invalidateQueries(['ai-hub', 'dataset-presets', 'enabled']);
              }
            } else {
              toast('Preset created (all presets currently enabled)', { duration: 4000 });
            }
          } else {
            const current = await apiClient.getEnabledAIHubEvalTemplates();
            const enabled = (current as any)?.enabled || [];
            if (Array.isArray(enabled) && enabled.length > 0) {
              if (!enabled.includes(res.plugin_id)) {
                await apiClient.setEnabledAIHubEvalTemplates({ enabled: [...enabled, res.plugin_id] });
                toast.success('Eval template enabled');
                queryClient.invalidateQueries(['admin', 'ai-hub', 'eval-templates', 'enabled']);
                queryClient.invalidateQueries(['training-eval-templates']);
              }
            } else {
              toast('Eval created (all eval templates currently enabled)', { duration: 4000 });
            }
          }
        }
      } catch (e: any) {
        const msg =
          e?.response?.data?.detail || e?.message || 'Failed to create plugin (admin required)';
        toast.error(msg);
      } finally {
        setCreatingPluginId(null);
      }
    };

    const submitFeedback = async (payload: {
      workflow: 'triage' | 'extraction' | 'literature';
      item_type: 'dataset_preset' | 'eval_template';
      item_id: string;
      decision: 'accept' | 'reject';
    }) => {
      const reasonKey = `${payload.workflow}:${payload.item_type}:${payload.item_id}`;
      const reason = (feedbackReasons[reasonKey] || '').trim();
      try {
        await apiClient.submitAIHubRecommendationFeedback(String(job.id), {
          ...payload,
          reason: reason || undefined,
        } as any);
        toast.success('Feedback saved');
        queryClient.invalidateQueries(['agent-job', job.id, 'ai-hub', 'recommendation-feedback']);
      } catch (e: any) {
        toast.error(e?.response?.data?.detail || e?.message || 'Failed to save feedback');
      }
    };

    const bulkDecision = async (decision: 'accept' | 'reject') => {
      if (!aiHubBundle || !Array.isArray(aiHubBundle.selection_rationale) || aiHubBundle.selection_rationale.length === 0) {
        return;
      }
      setBulkSubmitting(true);
      try {
        const reason = bulkReason.trim();
        for (const rec of aiHubBundle.selection_rationale) {
          const itemType = rec?.type === 'dataset_preset' ? 'dataset_preset' : 'eval_template';
          const workflow = rec?.workflow as 'triage' | 'extraction' | 'literature';
          const itemId = rec?.id;
          if (!workflow || !itemId) continue;
          await apiClient.submitAIHubRecommendationFeedback(String(job.id), {
            workflow,
            item_type: itemType as any,
            item_id: itemId,
            decision,
            reason: reason || undefined,
          } as any);
        }
        toast.success(`Saved ${decision} for all`);
        queryClient.invalidateQueries(['agent-job', job.id, 'ai-hub', 'recommendation-feedback']);
      } catch (e: any) {
        toast.error(e?.response?.data?.detail || e?.message || 'Failed to save bulk feedback');
      } finally {
        setBulkSubmitting(false);
      }
    };

    const loadLog = useCallback(async () => {
      setLoadingLog(true);
      try {
        const data = await apiClient.getAgentJobLog(job.id, 20);
        setLogData(data);
      } catch (error) {
        console.error('Failed to load log:', error);
      }
      setLoadingLog(false);
    }, [job.id]);

    const loadMemories = useCallback(async () => {
      setLoadingMemories(true);
      try {
        const data = await apiClient.getJobMemories(job.id);
        setMemoriesData(data);
      } catch (error) {
        console.error('Failed to load memories:', error);
      }
      setLoadingMemories(false);
    }, [job.id]);

    const handleExtractMemories = async () => {
      setExtractingMemories(true);
      try {
        const result = await apiClient.extractJobMemories(job.id);
        toast.success(`Extracted ${result.memories_created} memories`);
        await loadMemories();
      } catch (error: any) {
        console.error('Failed to extract memories:', error);
        toast.error(error.message || 'Failed to extract memories');
      }
      setExtractingMemories(false);
    };

    useEffect(() => {
      loadLog();
      loadMemories();
    }, [loadLog, loadMemories]);

    const getMemoryIcon = (type: string) => {
      switch (type) {
        case 'finding': return <Search className="w-3 h-3" />;
        case 'insight': return <Lightbulb className="w-3 h-3" />;
        case 'pattern': return <Layers className="w-3 h-3" />;
        case 'lesson': return <BookOpen className="w-3 h-3" />;
        default: return <Brain className="w-3 h-3" />;
      }
    };

    const getMemoryColor = (type: string) => {
      switch (type) {
        case 'finding': return 'text-blue-600 bg-blue-100';
        case 'insight': return 'text-purple-600 bg-purple-100';
        case 'pattern': return 'text-orange-600 bg-orange-100';
        case 'lesson': return 'text-green-600 bg-green-100';
        default: return 'text-gray-600 bg-gray-100';
      }
    };

    return (
      <div className="bg-white border border-gray-200 rounded-lg h-full overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${typeConfig.color}`}>
                <TypeIcon className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">{job.name}</h2>
                <p className="text-sm text-gray-500">{typeConfig.label}</p>
              </div>
            </div>
            <div className={`flex items-center gap-1 px-3 py-1.5 rounded-full ${statusConfig.bgColor} ${statusConfig.color}`}>
              <StatusIcon className={`w-4 h-4 ${job.status === 'running' ? 'animate-spin' : ''}`} />
              <span className="font-medium capitalize">{job.status}</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 mt-3">
            {job.status === 'running' && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'pause' })}
                disabled={actionMutation.isLoading}
              >
                <Pause className="w-4 h-4 mr-1" />
                Pause
              </Button>
            )}
            {job.status === 'paused' && (
              <Button
                size="sm"
                variant="primary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'resume' })}
                disabled={actionMutation.isLoading}
              >
                <Play className="w-4 h-4 mr-1" />
                Resume
              </Button>
            )}
            {['pending', 'running', 'paused'].includes(job.status) && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'cancel' })}
                disabled={actionMutation.isLoading}
              >
                <XCircle className="w-4 h-4 mr-1" />
                Cancel
              </Button>
            )}
            {['completed', 'failed', 'cancelled'].includes(job.status) && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => actionMutation.mutate({ jobId: job.id, action: 'restart' })}
                disabled={actionMutation.isLoading}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Restart
              </Button>
            )}
            {/* Export button - available for completed or failed jobs with results */}
            {['completed', 'failed'].includes(job.status) && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  setExportingJob(job);
                  setShowExportModal(true);
                }}
              >
                <Download className="w-4 h-4 mr-1" />
                Export
              </Button>
            )}
            {canSaveAsPlaybook && (
              <Button
                size="sm"
                variant="secondary"
                onClick={() => saveAsPlaybookMutation.mutate()}
                disabled={saveAsPlaybookMutation.isLoading}
                title="Save this job chain as a reusable playbook (chain definition)"
              >
                <GitBranch className="w-4 h-4 mr-1" />
                {saveAsPlaybookMutation.isLoading ? 'Saving…' : 'Save playbook'}
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (window.confirm('Are you sure you want to delete this job?')) {
                  deleteMutation.mutate(job.id);
                }
              }}
              disabled={job.status === 'running' || deleteMutation.isLoading}
            >
              <Trash2 className="w-4 h-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Goal */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-1 flex items-center gap-1">
              <Target className="w-4 h-4" />
              Goal
            </h3>
            <p className="text-sm text-gray-600 bg-gray-50 rounded-lg p-3">{job.goal}</p>
          </div>

          {/* Progress */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Progress</h3>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  job.status === 'completed' ? 'bg-green-500' :
                  job.status === 'failed' ? 'bg-red-500' :
                  'bg-primary-500'
                }`}
                style={{ width: `${job.progress}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{job.progress}% complete</span>
              {job.current_phase && <span>{job.current_phase}</span>}
            </div>
          </div>

          {/* Resource usage */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Resource Usage</h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500">Iterations</p>
                <p className="text-lg font-semibold">{job.iteration}/{job.max_iterations}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500">Tool Calls</p>
                <p className="text-lg font-semibold">{job.tool_calls_used}/{job.max_tool_calls}</p>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500">LLM Calls</p>
                <p className="text-lg font-semibold">{job.llm_calls_used}/{job.max_llm_calls}</p>
              </div>
            </div>
          </div>

          {/* Timing */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Timing</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-500">Created:</span>
                <span className="ml-2">{new Date(job.created_at).toLocaleString()}</span>
              </div>
              {job.started_at && (
                <div>
                  <span className="text-gray-500">Started:</span>
                  <span className="ml-2">{new Date(job.started_at).toLocaleString()}</span>
                </div>
              )}
              {job.completed_at && (
                <div>
                  <span className="text-gray-500">Completed:</span>
                  <span className="ml-2">{new Date(job.completed_at).toLocaleString()}</span>
                </div>
              )}
              <div>
                <span className="text-gray-500">Duration:</span>
                <span className="ml-2">{formatDuration(job.started_at, job.completed_at)}</span>
              </div>
            </div>
          </div>

          {/* Error */}
          {job.error && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-red-700 mb-1">Error</h3>
              <p className="text-sm text-red-600 bg-red-50 rounded-lg p-3">{job.error}</p>
            </div>
          )}

          {/* Results summary */}
          {job.results && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Results Summary</h3>
              <div className="bg-gray-50 rounded-lg p-3">
                {job.results.summary && (
                  <p className="text-sm text-gray-600 mb-2">{job.results.summary}</p>
                )}
                <div className="flex gap-4 text-sm text-gray-500">
                  {job.results.findings_count !== undefined && (
                    <span>Findings: {job.results.findings_count}</span>
                  )}
                  {job.results.actions_count !== undefined && (
                    <span>Actions: {job.results.actions_count}</span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Approval checkpoint */}
          {approvalCheckpoint && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Approval Checkpoint</h3>
              <div className="bg-rose-50 border border-rose-100 rounded-lg p-3">
                <p className="text-sm text-rose-800">
                  {String(approvalCheckpoint?.message || 'Human approval required before next action.')}
                </p>
                <div className="mt-2 flex flex-wrap gap-3 text-xs text-rose-700">
                  {approvalCheckpoint?.iteration !== undefined && (
                    <span>Iteration: {Number(approvalCheckpoint.iteration || 0)}</span>
                  )}
                  {approvalCheckpoint?.action?.tool && (
                    <span>Next tool: {String(approvalCheckpoint.action.tool)}</span>
                  )}
                  {approvalCheckpoint?.created_at && (
                    <span>Created: {new Date(String(approvalCheckpoint.created_at)).toLocaleString()}</span>
                  )}
                </div>
                {Array.isArray(approvalCheckpoint?.reasons) && approvalCheckpoint.reasons.length > 0 && (
                  <ul className="mt-2 text-xs text-rose-700 space-y-1">
                    {approvalCheckpoint.reasons.slice(0, 6).map((reason: string, idx: number) => (
                      <li key={`${idx}-${reason.slice(0, 24)}`}>- {reason}</li>
                    ))}
                  </ul>
                )}
                {job.status === 'paused' && (
                  <div className="mt-2 text-xs text-rose-700">
                    Use <span className="font-medium">Resume</span> to continue execution.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Goal contract */}
          {goalContractSummary && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Goal Contract</h3>
              <div className="bg-amber-50 border border-amber-100 rounded-lg p-3">
                <div className="flex flex-wrap gap-3 text-xs text-amber-700 mb-2">
                  <span>Enabled: {goalContractSummary?.enabled ? 'yes' : 'no'}</span>
                  <span>Satisfied: {goalContractSummary?.satisfied ? 'yes' : 'no'}</span>
                  {goalContractSummary?.strict_completion !== undefined && (
                    <span>Strict: {goalContractSummary.strict_completion ? 'yes' : 'no'}</span>
                  )}
                  {goalContractSummary?.satisfied_iteration ? (
                    <span>Satisfied at iteration: {Number(goalContractSummary.satisfied_iteration || 0)}</span>
                  ) : null}
                </div>
                {Array.isArray(goalContractSummary?.missing) && goalContractSummary.missing.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-amber-800 mb-1">Missing requirements</div>
                    <ul className="text-xs text-amber-700 space-y-1">
                      {goalContractSummary.missing.slice(0, 8).map((m: string, idx: number) => (
                        <li key={`${idx}-${m.slice(0, 24)}`}>- {m}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Executive digest */}
          {executiveDigest && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Executive Digest</h3>
              <div className="bg-sky-50 border border-sky-100 rounded-lg p-3 space-y-2">
                {executiveDigest?.outcome && (
                  <p className="text-sm text-sky-800">{String(executiveDigest.outcome)}</p>
                )}
                {executiveDigest?.metrics && typeof executiveDigest.metrics === 'object' && (
                  <div className="flex flex-wrap gap-3 text-xs text-sky-700">
                    <span>Progress: {Number((executiveDigest.metrics as any).goal_progress || 0)}%</span>
                    <span>Iterations: {Number((executiveDigest.metrics as any).iterations || 0)}</span>
                    <span>Findings: {Number((executiveDigest.metrics as any).findings_count || 0)}</span>
                    <span>Artifacts: {Number((executiveDigest.metrics as any).artifacts_count || 0)}</span>
                  </div>
                )}
                {Array.isArray(executiveDigest?.key_findings) && executiveDigest.key_findings.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-sky-800 mb-1">Key findings</div>
                    <ul className="text-xs text-sky-700 space-y-1">
                      {executiveDigest.key_findings.slice(0, 5).map((f: string, idx: number) => (
                        <li key={`${idx}-${f.slice(0, 24)}`}>- {f}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(executiveDigest?.risks) && executiveDigest.risks.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-sky-800 mb-1">Risks</div>
                    <ul className="text-xs text-sky-700 space-y-1">
                      {executiveDigest.risks.slice(0, 4).map((r: string, idx: number) => (
                        <li key={`${idx}-${r.slice(0, 24)}`}>- {r}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(executiveDigest?.next_actions) && executiveDigest.next_actions.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-sky-800 mb-1">Next actions</div>
                    <ul className="text-xs text-sky-700 space-y-1">
                      {executiveDigest.next_actions.slice(0, 4).map((step: string, idx: number) => (
                        <li key={`${idx}-${step.slice(0, 24)}`}>- {step}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Swarm summary */}
          {swarmSummary && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <Layers className="w-4 h-4" />
                Swarm Summary
              </h3>
              <div className="bg-slate-50 rounded-lg p-3 space-y-2">
                <div className="flex flex-wrap gap-3 text-xs text-slate-600">
                  <span>Roles: {Number(swarmSummary?.role_count || 0)}</span>
                  <span>Siblings: {Number(swarmSummary?.terminal_siblings || 0)}/{Number(swarmSummary?.expected_siblings || 0)}</span>
                  <span>Consensus: {Number(swarmSummary?.consensus_count || 0)}</span>
                  <span>Conflicts: {Number(swarmSummary?.conflict_count || 0)}</span>
                  {swarmSummary?.confidence?.overall !== undefined && (
                    <span>Confidence: {(Number(swarmSummary.confidence.overall) * 100).toFixed(0)}%</span>
                  )}
                </div>
                {Array.isArray(swarmSummary?.roles) && swarmSummary.roles.length > 0 && (
                  <div className="text-xs text-slate-700">
                    Roles: {swarmSummary.roles.slice(0, 8).join(', ')}
                  </div>
                )}
                {Array.isArray(swarmSummary?.consensus_findings) && swarmSummary.consensus_findings.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Top consensus findings</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.consensus_findings.slice(0, 4).map((finding: string, idx: number) => (
                        <li key={`${idx}-${finding.slice(0, 24)}`}>- {finding}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(swarmSummary?.conflicts) && swarmSummary.conflicts.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Conflicts</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.conflicts.slice(0, 3).map((c: any, idx: number) => (
                        <li key={`${idx}-${String(c?.type || 'conflict')}`}>
                          - {String(c?.description || c?.type || 'Conflict').slice(0, 220)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {Array.isArray(swarmSummary?.action_plan) && swarmSummary.action_plan.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-slate-700 mb-1">Action plan</div>
                    <ul className="text-xs text-slate-700 space-y-1">
                      {swarmSummary.action_plan.slice(0, 4).map((step: any, idx: number) => (
                        <li key={`${idx}-${String(step?.action || 'step')}`}>
                          - {String(step?.action || '').slice(0, 220)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Customer research context */}
          {(customerProfile || customerContext || documentArtifact || readingListArtifact || arxivSourceArtifacts.length > 0) && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <Brain className="w-4 h-4" />
                Customer Research
              </h3>
              <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-3">
                {customerProfile?.name && (
                  <div className="text-sm text-gray-800">
                    <span className="text-gray-500">Profile:</span> {customerProfile.name}
                  </div>
                )}
                {Array.isArray(customerProfile?.keywords) && customerProfile.keywords.length > 0 && (
                  <div className="text-xs text-gray-600">
                    <span className="text-gray-500">Keywords:</span> {customerProfile.keywords.slice(0, 20).join(', ')}
                  </div>
                )}
                {customerContext && (
                  <div className="text-xs text-gray-600 whitespace-pre-wrap">
                    <span className="text-gray-500">Context:</span> {String(customerContext).slice(0, 1200)}
                  </div>
                )}

                {researchBundle && (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">
                      Research bundle ({(researchBundle?.top_documents || []).length} docs • {(researchBundle?.top_papers || []).length} papers •{' '}
                      {(researchBundle?.key_insights || []).length} insights)
                    </summary>
                    <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="bg-white border border-gray-200 rounded p-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Top documents</div>
                        <div className="space-y-1">
                          {(researchBundle?.top_documents || []).slice(0, 6).map((d: any) => (
                            <div key={String(d?.id)} className="flex items-start justify-between gap-2">
                              <div className="text-xs text-gray-700 min-w-0">
                                <div className="truncate">{d?.title || d?.id}</div>
                                <div className="text-gray-500 font-mono truncate">{String(d?.id || '')}</div>
                              </div>
                              {d?.id && (
                                <div className="shrink-0 flex gap-1">
                                  <Button size="sm" variant="secondary" onClick={() => openDocument(String(d.id))}>
                                    Open
                                  </Button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white border border-gray-200 rounded p-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Key insights</div>
                        <div className="space-y-1">
                          {(researchBundle?.key_insights || []).slice(0, 8).map((it: any, idx: number) => (
                            <div key={String(it?.id || idx)} className="text-xs text-gray-700">
                              <div className="truncate">{it?.title || '(untitled insight)'}</div>
                              {(it?.category || it?.confidence !== undefined) && (
                                <div className="text-gray-500">
                                  {it?.category ? String(it.category) : ''}
                                  {it?.confidence !== undefined ? ` • ${(Number(it.confidence) * 100).toFixed(0)}%` : ''}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white border border-gray-200 rounded p-2 md:col-span-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Top papers</div>
                        <div className="space-y-1">
                          {(researchBundle?.top_papers || []).slice(0, 6).map((p: any, idx: number) => (
                            <div key={String(p?.arxiv_id || idx)} className="flex items-start justify-between gap-2">
                              <div className="text-xs text-gray-700 min-w-0">
                                <div className="truncate">{p?.title || p?.arxiv_id}</div>
                                <div className="text-gray-500 font-mono truncate">{String(p?.arxiv_id || '')}</div>
                              </div>
                              {p?.arxiv_id && (
                                <div className="shrink-0">
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(p.arxiv_id), 'arXiv ID')}>
                                    Copy ID
                                  </Button>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    {Array.isArray(researchBundle?.suggested_queries) && researchBundle.suggested_queries.length > 0 && (
                      <div className="mt-3 bg-white border border-gray-200 rounded p-2">
                        <div className="text-xs font-medium text-gray-800 mb-1">Suggested queries</div>
                        <div className="text-xs text-gray-600 whitespace-pre-wrap">
                          {researchBundle.suggested_queries.slice(0, 8).map((q: string) => `- ${q}`).join('\n')}
                        </div>
                        <div className="pt-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => copyText((researchBundle.suggested_queries || []).join('\n'), 'Suggested queries')}
                          >
                            Copy queries
                          </Button>
                        </div>
                      </div>
                    )}
                  </details>
                )}

                {(documentArtifact?.id || documentArtifact?.document_id) && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700">
                      <div className="font-medium text-gray-800">Brief document</div>
                      <div className="text-gray-600 font-mono">
                        {String(documentArtifact.id || documentArtifact.document_id)}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => openDocument(String(documentArtifact.id || documentArtifact.document_id))}
                      >
                        Open
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyText(String(documentArtifact.id || documentArtifact.document_id), 'Document ID')}
                      >
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {codePatchProposal?.proposal_id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Code patch</div>
                      <div className="text-gray-600 truncate">{codePatchProposal.title}</div>
                      <div className="text-gray-600 font-mono truncate">{codePatchProposal.proposal_id}</div>
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() =>
                          apiClient.downloadCodePatchProposal(codePatchProposal.proposal_id, codePatchProposal.title)
                        }
                      >
                        Download
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={async () => {
                          const ok = window.confirm(
                            'Apply this patch to KnowledgeDB code documents now? This updates the stored file contents.'
                          );
                          if (!ok) return;
                          try {
                            const res = await apiClient.applyCodePatchProposal(codePatchProposal.proposal_id);
                            if ((res.errors || []).length > 0) {
                              toast.error(`Applied with errors: ${(res.errors || []).length}`);
                            } else {
                              toast.success('Patch applied to KB');
                            }
                            queryClient.invalidateQueries(['agent-jobs']);
                          } catch (e: any) {
                            toast.error(e?.response?.data?.detail || e?.message || 'Failed to apply patch');
                          }
                        }}
                      >
                        Apply to KB
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyText(codePatchProposal.proposal_id, 'Proposal ID')}
                      >
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {codePatchProposals.length > 1 ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Code patch history</summary>
                    <div className="mt-2 space-y-2">
                      {codePatchProposals.map((p) => (
                        <div key={p.proposal_id} className="flex items-center justify-between gap-3 bg-white border border-gray-200 rounded-lg p-2">
                          <div className="text-xs text-gray-700 min-w-0">
                            <div className="text-gray-600 truncate">{p.title}</div>
                            <div className="text-gray-600 font-mono truncate">{p.proposal_id}</div>
                          </div>
                          <div className="flex gap-2 shrink-0">
                            <Button size="sm" variant="secondary" onClick={() => apiClient.downloadCodePatchProposal(p.proposal_id, p.title)}>
                              Download
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={async () => {
                                const ok = window.confirm(
                                  'Apply this patch to KnowledgeDB code documents now? This updates the stored file contents.'
                                );
                                if (!ok) return;
                                try {
                                  const res = await apiClient.applyCodePatchProposal(p.proposal_id);
                                  if ((res.errors || []).length > 0) toast.error(`Applied with errors: ${(res.errors || []).length}`);
                                  else toast.success('Patch applied to KB');
                                  queryClient.invalidateQueries(['agent-jobs']);
                                } catch (e: any) {
                                  toast.error(e?.response?.data?.detail || e?.message || 'Failed to apply patch');
                                }
                              }}
                            >
                              Apply to KB
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => copyText(p.proposal_id, 'Proposal ID')}>
                              Copy ID
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                ) : null}

                {experimentRuns.length > 0 ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Experiment runs</summary>
                    <div className="mt-2 space-y-2">
                      {experimentRuns.map((er, idx) => {
                        const okVal = er?.ok;
                        const label = okVal === true ? 'PASS' : okVal === false ? 'FAIL' : 'SKIP';
                        const labelClass = okVal === true ? 'text-green-700' : okVal === false ? 'text-red-700' : 'text-amber-700';
                        const cmds = Array.isArray(er?.commands) ? er.commands : [];
                        const pid = String(er?.proposal_id || '').trim();
                        return (
                          <div key={idx} className="bg-white border border-gray-200 rounded-lg p-2">
                            <div className="flex items-center justify-between gap-2 text-xs">
                              <div className="text-gray-700 min-w-0">
                                <span className={`font-medium ${labelClass}`}>{label}</span>
                                {er?.source_name ? <span className="text-gray-500"> — {String(er.source_name)}</span> : null}
                                {pid ? <span className="text-gray-500"> • </span> : null}
                                {pid ? <span className="text-gray-500 font-mono truncate">{pid}</span> : null}
                              </div>
                              {cmds.length > 0 ? <div className="text-gray-500">{cmds.length} cmd(s)</div> : null}
                            </div>
                            {cmds.length > 0 ? (
                              <div className="mt-1 text-[11px] text-gray-600 font-mono whitespace-pre-wrap">
                                {cmds.slice(0, 6).join('\n')}
                              </div>
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  </details>
                ) : null}

                {codePatchApply ? (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Patch apply (sandbox)</div>
                      <div className="text-gray-600">
                        applied: {Array.isArray(codePatchApply.applied) ? codePatchApply.applied.length : 0} • errors:{' '}
                        {Array.isArray(codePatchApply.errors) ? codePatchApply.errors.length : 0}
                      </div>
                      {codePatchApply.proposal_id ? (
                        <div className="text-gray-600 font-mono truncate">{String(codePatchApply.proposal_id)}</div>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                {codePatchKbApply ? (
                  <details className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <summary className="cursor-pointer text-xs font-medium text-gray-800">Patch apply (Knowledge DB)</summary>
                    <div className="mt-2 space-y-2 text-xs text-gray-700">
                      <div className="text-gray-600">
                        {codePatchKbApply.enabled === false
                          ? 'skipped'
                          : codePatchKbApply.dry_run
                            ? `dry-run — ok: ${String(codePatchKbApply.ok)}`
                            : `applied: ${String(codePatchKbApply.did_apply)} — ok: ${String(codePatchKbApply.ok)}`}
                        {' • '}
                        errors: {Array.isArray(codePatchKbApply.errors) ? codePatchKbApply.errors.length : 0}
                        {' • '}
                        files: {Array.isArray(codePatchKbApply.applied_files) ? codePatchKbApply.applied_files.length : 0}
                      </div>
                      {codePatchKbApply.blocked_reason ? (
                        <div className="text-yellow-800 bg-yellow-50 border border-yellow-200 rounded px-2 py-1">
                          Blocked: {String(codePatchKbApply.blocked_reason)}
                        </div>
                      ) : null}
                      {codePatchKbApply.proposal_strategy ? (
                        <div className="text-gray-500">strategy: {String(codePatchKbApply.proposal_strategy)}</div>
                      ) : null}
                      {codePatchKbApply.proposal_id ? (
                        <div className="text-gray-600 font-mono truncate">{String(codePatchKbApply.proposal_id)}</div>
                      ) : null}

                      {Array.isArray(codePatchKbApply.applied_files) && codePatchKbApply.applied_files.length > 0 ? (
                        <div className="space-y-1">
                          <div className="font-medium text-gray-800">Applied files</div>
                          <div className="space-y-1">
                            {codePatchKbApply.applied_files.slice(0, 50).map((f: any, i: number) => (
                              <div key={String(f?.document_id || f?.path || i)} className="flex items-center justify-between gap-2 bg-white border border-gray-200 rounded px-2 py-1">
                                <div className="min-w-0">
                                  <div className="text-gray-600 font-mono truncate">{String(f?.path || '(unknown path)')}</div>
                                  {f?.document_id ? (
                                    <div className="text-gray-500 font-mono truncate">{String(f.document_id)}</div>
                                  ) : null}
                                </div>
                                <div className="flex gap-2 shrink-0">
                                  {f?.document_id ? (
                                    <Button size="sm" variant="secondary" onClick={() => openDocument(String(f.document_id))}>
                                      Open
                                    </Button>
                                  ) : null}
                                  {f?.document_id ? (
                                    <Button size="sm" variant="ghost" onClick={() => copyText(String(f.document_id), 'Document ID')}>
                                      Copy ID
                                    </Button>
                                  ) : null}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}

                      {Array.isArray(codePatchKbApply.errors) && codePatchKbApply.errors.length > 0 ? (
                        <div className="space-y-1">
                          <div className="font-medium text-gray-800">Errors</div>
                          <div className="space-y-1">
                            {codePatchKbApply.errors.slice(0, 50).map((e: any, i: number) => (
                              <div key={String(e?.path || i)} className="bg-white border border-red-200 rounded px-2 py-1">
                                <div className="text-red-800 font-mono">{String(e?.path || '(unknown file)')}</div>
                                <div className="text-red-700">{String(e?.error || e?.message || '')}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </details>
                ) : null}

                {generatedProject?.source_id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Generated project</div>
                      <div className="text-gray-600 truncate">{generatedProject.project_name}</div>
                      <div className="text-gray-600 font-mono truncate">{generatedProject.source_id}</div>
                      {generatedProject.file_count ? (
                        <div className="text-gray-500">{generatedProject.file_count} files</div>
                      ) : null}
                      {generatedProject.sanity_errors_count ? (
                        <div className="text-red-600">Syntax errors: {generatedProject.sanity_errors_count}</div>
                      ) : generatedProject.sanity_ok ? (
                        <div className="text-green-700">Sanity check: OK</div>
                      ) : null}
                      {generatedProject.sanity_ok && generatedProject.behavioral?.enabled === false ? (
                        <div className="text-amber-700">Behavioral check: skipped (server disabled)</div>
                      ) : generatedProject.behavioral?.ran ? (
                        generatedProject.behavioral?.ok ? (
                          <div className="text-green-700">Behavioral check: OK</div>
                        ) : (
                          <div className="text-red-600">Behavioral check: failed</div>
                        )
                      ) : null}
                      {generatedProject.behavioral?.ran ? (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-gray-700">Behavior details</summary>
                          <div className="mt-2 space-y-2">
                            <div className="text-gray-700">
                              Backend: <span className="font-mono">{String(generatedProject.behavioral.backend || '')}</span>
                              {typeof generatedProject.behavioral.duration_ms === 'number' ? (
                                <span className="ml-2">({generatedProject.behavioral.duration_ms}ms)</span>
                              ) : null}
                              {generatedProject.behavioral.timed_out ? <span className="ml-2 text-red-600">timeout</span> : null}
                            </div>
                            {generatedProject.behavioral.error ? (
                              <div className="text-red-700">Error: {String(generatedProject.behavioral.error)}</div>
                            ) : null}
                            {typeof generatedProject.behavioral.exit_code === 'number' ? (
                              <div className="text-gray-700">
                                Exit code: <span className="font-mono">{String(generatedProject.behavioral.exit_code)}</span>
                              </div>
                            ) : null}
                            {typeof generatedProject.behavioral.stdout === 'string' && generatedProject.behavioral.stdout.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stdout</div>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => copyText(String(generatedProject.behavioral.stdout || ''), 'stdout')}
                                  >
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(generatedProject.behavioral.stdout)}
                                </pre>
                              </div>
                            ) : null}
                            {typeof generatedProject.behavioral.stderr === 'string' && generatedProject.behavioral.stderr.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stderr</div>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => copyText(String(generatedProject.behavioral.stderr || ''), 'stderr')}
                                  >
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(generatedProject.behavioral.stderr)}
                                </pre>
                              </div>
                            ) : null}
                          </div>
                        </details>
                      ) : null}
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() =>
                          apiClient.downloadDocumentSourceZip(generatedProject.source_id, generatedProject.project_name)
                        }
                      >
                        Download ZIP
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => navigate('/documents', { state: { selectedSourceId: generatedProject.source_id } })}
                      >
                        Open
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={unsafeExecBadge.status !== 'ready' || createMutation.isLoading}
                        title={
                          unsafeExecBadge.status === 'ready'
                            ? 'Run sandboxed demo check again'
                            : 'Demo check not available (see badge on Implement Algorithm)'
                        }
                        onClick={() =>
                          createMutation.mutate({
                            name: `Demo check — ${generatedProject.project_name}`.slice(0, 120),
                            job_type: 'monitor' as any,
                            goal: `Run demo check (${generatedProject.project_name})`,
                            config: {
                              deterministic_runner: 'generated_project_demo_check',
                              source_id: generatedProject.source_id,
                              entrypoint: generatedProject.entrypoint || 'demo.py',
                            },
                            max_iterations: 1,
                            max_tool_calls: 0,
                            max_llm_calls: 0,
                            max_runtime_minutes: 5,
                            start_immediately: true,
                          })
                        }
                      >
                        Re-run demo
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => copyText(generatedProject.source_id, 'Source ID')}>
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {demoCheck?.source_id && (
                  <div className="flex items-start justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700 min-w-0">
                      <div className="font-medium text-gray-800">Demo check</div>
                      <div className="text-gray-600 font-mono truncate">{demoCheck.source_id}</div>
                      <div className={demoCheck.ok ? 'text-green-700' : 'text-red-600'}>
                        {demoCheck.ok ? 'OK' : 'FAILED'} • {demoCheck.entrypoint}
                      </div>
                      {demoCheck.behavioral?.ran ? (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-gray-700">Details</summary>
                          <div className="mt-2 space-y-2">
                            <div className="text-gray-700">
                              Backend: <span className="font-mono">{String(demoCheck.behavioral.backend || '')}</span>
                              {typeof demoCheck.behavioral.duration_ms === 'number' ? (
                                <span className="ml-2">({demoCheck.behavioral.duration_ms}ms)</span>
                              ) : null}
                              {demoCheck.behavioral.timed_out ? <span className="ml-2 text-red-600">timeout</span> : null}
                            </div>
                            {demoCheck.behavioral.error ? (
                              <div className="text-red-700">Error: {String(demoCheck.behavioral.error)}</div>
                            ) : null}
                            {typeof demoCheck.behavioral.exit_code === 'number' ? (
                              <div className="text-gray-700">
                                Exit code: <span className="font-mono">{String(demoCheck.behavioral.exit_code)}</span>
                              </div>
                            ) : null}
                            {typeof demoCheck.behavioral.stdout === 'string' && demoCheck.behavioral.stdout.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stdout</div>
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(demoCheck.behavioral.stdout || ''), 'stdout')}>
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(demoCheck.behavioral.stdout)}
                                </pre>
                              </div>
                            ) : null}
                            {typeof demoCheck.behavioral.stderr === 'string' && demoCheck.behavioral.stderr.trim() ? (
                              <div>
                                <div className="flex items-center justify-between">
                                  <div className="text-gray-700">stderr</div>
                                  <Button size="sm" variant="ghost" onClick={() => copyText(String(demoCheck.behavioral.stderr || ''), 'stderr')}>
                                    Copy
                                  </Button>
                                </div>
                                <pre className="mt-1 p-2 bg-white border border-gray-200 rounded whitespace-pre-wrap max-h-48 overflow-auto">
                                  {String(demoCheck.behavioral.stderr)}
                                </pre>
                              </div>
                            ) : null}
                          </div>
                        </details>
                      ) : null}
                    </div>
                    <div className="flex gap-2 shrink-0">
                      <Button size="sm" variant="ghost" onClick={() => copyText(demoCheck.source_id, 'Source ID')}>
                        Copy source
                      </Button>
                    </div>
                  </div>
                )}

                {readingListArtifact?.id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700">
                      <div className="font-medium text-gray-800">Reading list</div>
                      <div className="text-gray-600">
                        {readingListArtifact.name || 'Reading List'} •{' '}
                        <span className="font-mono">{String(readingListArtifact.id)}</span>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="secondary" onClick={() => openReadingList(String(readingListArtifact.id))}>
                        Open
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => copyText(String(readingListArtifact.id), 'Reading list ID')}>
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {!readingListArtifact?.id && readingListByName?.id && (
                  <div className="flex items-center justify-between gap-3 bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs text-gray-700">
                      <div className="font-medium text-gray-800">Reading list</div>
                      <div className="text-gray-600">
                        {readingListByName.name || 'Reading List'} •{' '}
                        <span className="font-mono">{String(readingListByName.id)}</span>
                      </div>
                      <div className="text-gray-500">Resolved by name from job config</div>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="secondary" onClick={() => openReadingList(String(readingListByName.id))}>
                        Open
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => copyText(String(readingListByName.id), 'Reading list ID')}>
                        Copy ID
                      </Button>
                    </div>
                  </div>
                )}

                {(arxivSourceArtifacts.length > 0 || arxivImportsFallback.length > 0) && (
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                    <div className="text-xs font-medium text-gray-800 mb-2">arXiv imports</div>
                    <div className="space-y-2">
                      {(arxivSourceArtifacts.length > 0 ? arxivSourceArtifacts : arxivImportsFallback).slice(0, 5).map((s: any, idx: number) => (
                        <div key={`${s.id || idx}`} className="flex items-center justify-between gap-3">
                          <div className="text-xs text-gray-700">
                            <div className="text-gray-800">{s.name || 'ArXiv Import'}</div>
                            <div className="text-gray-600 font-mono">{String(s.id)}</div>
                          </div>
                          <div className="flex gap-2">
                            <Button size="sm" variant="secondary" onClick={() => navigate(`/papers?source_id=${encodeURIComponent(String(s.id))}`)}>
                              Open Papers
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => copyText(String(s.id), 'Source ID')}>
                              Copy ID
                            </Button>
                          </div>
                        </div>
                      ))}
                      {(arxivSourceArtifacts.length > 5 || arxivImportsFallback.length > 5) && (
                        <div className="text-xs text-gray-500">
                          +{(arxivSourceArtifacts.length > 0 ? arxivSourceArtifacts.length : arxivImportsFallback.length) - 5} more imports
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* AI Hub bundle proposal */}
          {aiHubBundle && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <Sparkles className="w-4 h-4" />
                AI Hub Bundle
              </h3>
              <div className="bg-white border border-gray-200 rounded-lg p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium text-gray-900">{aiHubBundle.bundle_name || 'Bundle'}</div>
                    <div className="text-xs text-gray-500">
                      Presets: {(aiHubBundle.enabled_dataset_presets || []).length} • Evals: {(aiHubBundle.enabled_eval_templates || []).length}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" onClick={applyAIHubBundle}>
                      Apply to AI Hub
                    </Button>
                    <Button size="sm" variant="secondary" onClick={() => navigate('/ai-hub?tab=datasets')}>
                      Open AI Hub
                    </Button>
                  </div>
                </div>

                <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-gray-50 rounded p-2">
                    <div className="font-medium text-gray-700 mb-1">Enabled Dataset Presets</div>
                    <div className="text-gray-600 break-words">
                      {(aiHubBundle.enabled_dataset_presets || []).join(', ') || '(none)'
                    }</div>
                  </div>
                  <div className="bg-gray-50 rounded p-2">
                    <div className="font-medium text-gray-700 mb-1">Enabled Eval Templates</div>
                    <div className="text-gray-600 break-words">
                      {(aiHubBundle.enabled_eval_templates || []).join(', ') || '(none)'
                    }</div>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => copyText(JSON.stringify(aiHubBundle, null, 2), 'Bundle JSON')}
                  >
                    Copy Bundle JSON
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => copyText(envText, 'Env Vars')}
                    disabled={!envText}
                    title="Use these for env-based configuration if you can’t apply via admin"
                  >
                    Copy Env Vars
                  </Button>
                </div>

                {Array.isArray(aiHubBundle.recommended_new_plugins) &&
                  aiHubBundle.recommended_new_plugins.length > 0 && (
                    <div className="mt-4">
                      <div className="text-sm font-medium text-gray-800 mb-2">Recommended new plugins</div>
                      <label className="flex items-center gap-2 text-xs text-gray-600 mb-2">
                        <input
                          type="checkbox"
                          checked={enableAfterCreate}
                          onChange={(e) => setEnableAfterCreate(e.target.checked)}
                        />
                        Enable after create (only affects allowlist mode; no-op if “all enabled”)
                      </label>
                      <div className="space-y-2">
                        {aiHubBundle.recommended_new_plugins.map((rec: any, idx: number) => {
                          const skeleton = rec?.skeleton;
                          const pluginType =
                            rec?.type === 'dataset_preset' ? ('dataset_preset' as const) : ('eval_template' as const);
                          const suggestedId = rec?.id_suggestion || skeleton?.id || `plugin_${idx}`;
                          const plugin = {
                            ...(skeleton || {}),
                            id: suggestedId,
                            name: rec?.name_suggestion || skeleton?.name || suggestedId,
                          };
                          return (
                            <div key={`${pluginType}:${suggestedId}:${idx}`} className="border border-gray-200 rounded-lg p-3 bg-white">
                              <div className="flex items-start justify-between gap-3">
                                <div>
                                  <div className="text-sm font-medium text-gray-900">
                                    {pluginType === 'dataset_preset' ? 'Dataset Preset' : 'Eval Template'} • {rec?.workflow || 'workflow'}
                                  </div>
                                  <div className="text-xs text-gray-500 mt-1">
                                    Suggested id: <span className="font-mono">{suggestedId}</span>
                                  </div>
                                  {rec?.why && <div className="text-xs text-gray-600 mt-1">{rec.why}</div>}
                                </div>
                                <div className="flex gap-2">
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => copyText(JSON.stringify(plugin, null, 2), 'Plugin JSON')}
                                  >
                                    Copy JSON
                                  </Button>
                                  <Button
                                    size="sm"
                                    onClick={() => createPlugin(pluginType, plugin)}
                                    disabled={creatingPluginId === String(plugin.id)}
                                    title="Admin: persist this plugin JSON to disk"
                                  >
                                    {creatingPluginId === String(plugin.id) ? 'Creating…' : 'Create Plugin'}
                                  </Button>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        After creating, enable it in `Admin → AI Hub` (or rerun AI Scientist and Apply).
                      </div>
                    </div>
                  )}

                {Array.isArray(aiHubBundle.selection_rationale) && aiHubBundle.selection_rationale.length > 0 && (
                  <div className="mt-4">
                    <div className="text-sm font-medium text-gray-800 mb-2">Learning loop (accept/reject)</div>
                    <div className="mb-3 border border-gray-200 rounded-lg p-3 bg-gray-50">
                      <div className="text-xs text-gray-600 mb-2">Bulk actions</div>
                      <div className="flex flex-wrap gap-2 items-center">
                        <input
                          className="flex-1 min-w-[220px] border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          value={bulkReason}
                          onChange={(e) => setBulkReason(e.target.value)}
                          placeholder="Optional shared reason (applies to all)"
                        />
                        <Button size="sm" onClick={() => bulkDecision('accept')} disabled={bulkSubmitting}>
                          {bulkSubmitting ? 'Saving…' : 'Accept all'}
                        </Button>
                        <Button size="sm" variant="secondary" onClick={() => bulkDecision('reject')} disabled={bulkSubmitting}>
                          {bulkSubmitting ? 'Saving…' : 'Reject all'}
                        </Button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      {aiHubBundle.selection_rationale.map((rec: any, idx: number) => {
                        const itemType = rec?.type === 'dataset_preset' ? 'dataset_preset' : 'eval_template';
                        const workflow = rec?.workflow as 'triage' | 'extraction' | 'literature';
                        const itemId = rec?.id;
                        const key = `${workflow}:${itemType}:${itemId}`;
                        const existing = feedbackIndex[key];
                        const isOpen = Boolean(detailsOpen[key]);
                        return (
                          <div key={`${key}:${idx}`} className="border border-gray-200 rounded-lg p-3 bg-white">
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <div className="text-sm font-medium text-gray-900">
                                  {workflow} • {itemType === 'dataset_preset' ? 'Preset' : 'Eval'} •{' '}
                                  <span className="font-mono">{itemId}</span>
                                </div>
                                {Array.isArray(rec?.matched_terms) && rec.matched_terms.length > 0 && (
                                  <div className="text-xs text-gray-500 mt-1">
                                    Matched: {rec.matched_terms.slice(0, 8).join(', ')}
                                  </div>
                                )}
                                {(rec?.feedback_accepts !== undefined || rec?.feedback_rejects !== undefined) && (
                                  <div className="text-xs text-gray-500 mt-1">
                                    Feedback: +{Number(rec.feedback_accepts || 0)} / -{Number(rec.feedback_rejects || 0)}
                                    {rec?.feedback_bias !== undefined && (
                                      <> • bias {Number(rec.feedback_bias || 0) >= 0 ? '+' : ''}{Number(rec.feedback_bias || 0)}</>
                                    )}
                                    {rec?.base_score !== undefined && (
                                      <> • base {Number(rec.base_score || 0)}</>
                                    )}
                                  </div>
                                )}
                                {existing?.decision && (
                                  <div className="text-xs text-gray-600 mt-1">
                                    Your last decision: <span className="font-medium">{existing.decision}</span>
                                  </div>
                                )}
                              </div>
                              <div className="flex gap-2">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => setDetailsOpen((prev) => ({ ...prev, [key]: !prev[key] }))}
                                >
                                  {isOpen ? 'Hide' : 'Why'}
                                </Button>
                                <Button
                                  size="sm"
                                  variant={existing?.decision === 'accept' ? 'primary' : 'secondary'}
                                  onClick={() =>
                                    submitFeedback({
                                      workflow,
                                      item_type: itemType as any,
                                      item_id: itemId,
                                      decision: 'accept',
                                    })
                                  }
                                >
                                  Accept
                                </Button>
                                <Button
                                  size="sm"
                                  variant={existing?.decision === 'reject' ? 'primary' : 'secondary'}
                                  onClick={() =>
                                    submitFeedback({
                                      workflow,
                                      item_type: itemType as any,
                                      item_id: itemId,
                                      decision: 'reject',
                                    })
                                  }
                                >
                                  Reject
                                </Button>
                              </div>
                            </div>
                            {isOpen && (
                              <div className="mt-3 bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs text-gray-700 space-y-1">
                                <div>
                                  Score: <span className="font-medium">{Number(rec.score || 0)}</span>{' '}
                                  (base {Number(rec.base_score || 0)} + bias {Number(rec.feedback_bias || 0) >= 0 ? '+' : ''}{Number(rec.feedback_bias || 0)})
                                </div>
                                {Array.isArray(rec?.matched_terms) && rec.matched_terms.length > 0 && (
                                  <div>
                                    Matched terms: <span className="text-gray-600">{rec.matched_terms.join(', ')}</span>
                                  </div>
                                )}
                                {Array.isArray((aiHubBundle as any)?.customer_keywords) && (
                                  <div>
                                    Customer keywords: <span className="text-gray-600">{(aiHubBundle as any).customer_keywords.slice(0, 12).join(', ')}</span>
                                  </div>
                                )}
                                <div className="pt-2 flex gap-2">
                                  <Button
                                    size="sm"
                                    variant="secondary"
                                    onClick={() => copyText(JSON.stringify(rec, null, 2), 'Rationale JSON')}
                                  >
                                    Copy rationale
                                  </Button>
                                </div>
                              </div>
                            )}
                            <div className="mt-2">
                              <label className="block text-xs font-medium text-gray-700 mb-1">Reason (optional)</label>
                              <input
                                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                                value={feedbackReasons[key] ?? existing?.reason ?? ''}
                                onChange={(e) => setFeedbackReasons((prev) => ({ ...prev, [key]: e.target.value }))}
                                placeholder="E.g., 'Not relevant to our tooling' or 'Great default for weekly triage'"
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                      Feedback is stored per customer profile and will bias future AI Scientist recommendations.
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Output Artifacts - Charts and Diagrams */}
          {job.output_artifacts && job.output_artifacts.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                <BarChart3 className="w-4 h-4" />
                Generated Visualizations ({job.output_artifacts.filter(a => a.type === 'chart' || a.type === 'diagram').length})
              </h3>
              <div className="space-y-3">
                {job.output_artifacts
                  .filter((artifact: any) => artifact.type === 'chart' || artifact.type === 'diagram')
                  .map((artifact: any, idx: number) => (
                    <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden">
                      <div className="px-3 py-2 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
                        <span className="text-xs font-medium text-gray-600 flex items-center gap-1">
                          {artifact.type === 'chart' ? (
                            <BarChart3 className="w-3 h-3" />
                          ) : (
                            <Layers className="w-3 h-3" />
                          )}
                          {artifact.tool || artifact.type}
                          {artifact.format && ` (${artifact.format})`}
                        </span>
                        {artifact.edit_url && (
                          <a
                            href={artifact.edit_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-primary-600 hover:text-primary-800"
                          >
                            Edit in Draw.io
                          </a>
                        )}
                      </div>
                      {artifact.image_base64 && (
                        <div className="p-2 bg-white">
                          <img
                            src={`data:${artifact.mime_type || 'image/png'};base64,${artifact.image_base64}`}
                            alt={artifact.tool || 'Visualization'}
                            className="max-w-full h-auto mx-auto"
                            style={{ maxHeight: '300px' }}
                          />
                        </div>
                      )}
                      {artifact.code && artifact.format === 'mermaid' && (
                        <div className="p-2 bg-gray-50 text-gray-900 overflow-x-auto border-t border-gray-200">
                          <pre className="text-xs font-mono whitespace-pre-wrap">{artifact.code}</pre>
                        </div>
                      )}
                      {artifact.code && artifact.format === 'graphviz' && (
                        <div className="p-2 bg-gray-50 text-gray-900 overflow-x-auto border-t border-gray-200">
                          <pre className="text-xs font-mono whitespace-pre-wrap">{artifact.code}</pre>
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Job Memories */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-700 flex items-center gap-1">
                <Brain className="w-4 h-4" />
                Memories
                {memoriesData && memoriesData.total > 0 && (
                  <span className="ml-1 text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
                    {memoriesData.total}
                  </span>
                )}
              </h3>
              <div className="flex items-center gap-2">
                {['completed', 'failed'].includes(job.status) && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleExtractMemories}
                    disabled={extractingMemories}
                    title="Extract memories from job results"
                  >
                    {extractingMemories ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Sparkles className="w-3 h-3" />
                    )}
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setShowMemories(!showMemories)}
                >
                  {showMemories ? 'Hide' : 'Show'}
                </Button>
              </div>
            </div>

            {showMemories && (
              <div className="border border-purple-200 rounded-lg p-3 bg-purple-50">
                {loadingMemories ? (
                  <div className="flex justify-center py-4">
                    <LoadingSpinner size="sm" />
                  </div>
                ) : memoriesData && memoriesData.memories.length > 0 ? (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {memoriesData.memories.map((memory) => (
                      <div
                        key={memory.id}
                        className="bg-white rounded-lg p-2 border border-purple-100"
                      >
                        <div className="flex items-start gap-2">
                          <div className={`p-1 rounded ${getMemoryColor(memory.type)}`}>
                            {getMemoryIcon(memory.type)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-medium text-purple-700 uppercase">
                                {memory.type}
                              </span>
                              <span className="text-xs text-gray-400">
                                {(memory.importance_score * 100).toFixed(0)}% importance
                              </span>
                            </div>
                            <p className="text-xs text-gray-700">{memory.content}</p>
                            {memory.tags && memory.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-1">
                                {memory.tags.slice(0, 4).map((tag, idx) => (
                                  <span
                                    key={idx}
                                    className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-4">
                    <Brain className="w-8 h-8 text-purple-300 mx-auto mb-2" />
                    <p className="text-sm text-purple-600">No memories extracted yet</p>
                    {['completed', 'failed'].includes(job.status) && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="mt-2 text-purple-600"
                        onClick={handleExtractMemories}
                        disabled={extractingMemories}
                      >
                        <Sparkles className="w-3 h-3 mr-1" />
                        Extract Memories
                      </Button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Execution log */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-gray-700">Execution Log</h3>
              <Button size="sm" variant="ghost" onClick={loadLog} disabled={loadingLog}>
                <RefreshCw className={`w-3 h-3 mr-1 ${loadingLog ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
            {loadingLog ? (
              <div className="flex justify-center py-4">
                <LoadingSpinner size="sm" />
              </div>
            ) : logData && logData.entries.length > 0 ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {logData.entries.map((entry, idx) => (
                  <div key={idx} className="text-xs bg-gray-50 rounded p-2">
                    <div className="flex items-center justify-between text-gray-500 mb-1">
                      <span className="font-medium">
                        Iteration {entry.iteration} - {entry.phase}
                      </span>
                      <span>{entry.timestamp}</span>
                    </div>
                    {entry.action && <p className="text-gray-600">Action: {entry.action}</p>}
                    {entry.thought && <p className="text-gray-600 truncate">Thought: {entry.thought}</p>}
                    {entry.error && <p className="text-red-600">Error: {entry.error}</p>}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">No log entries yet</p>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Render template card
  const TemplateCard: React.FC<{ template: AgentJobTemplate }> = ({ template }) => {
    const typeConfig = JOB_TYPE_CONFIG[template.job_type as AgentJobType] || JOB_TYPE_CONFIG.custom;
    const TypeIcon = typeConfig.icon;

    return (
      <div
        className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
        onClick={() => setCreateFromTemplate(template)}
      >
        <div className="flex items-start gap-3 mb-3">
          <div className={`p-2 rounded-lg ${typeConfig.color}`}>
            <TypeIcon className="w-5 h-5" />
          </div>
          <div className="flex-1">
            <h3 className="font-medium text-gray-900">{template.display_name}</h3>
            <p className="text-sm text-gray-500">{template.category}</p>
          </div>
          {template.is_system && (
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">System</span>
          )}
        </div>
        <p className="text-sm text-gray-600 mb-3 line-clamp-2">{template.description}</p>
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span>Max {template.default_max_iterations} iterations</span>
          <span>{template.default_max_runtime_minutes} min runtime</span>
        </div>
      </div>
    );
  };

  // Create job modal
  const CreateJobModal: React.FC = () => {
    const [formData, setFormData] = useState<Partial<AgentJobCreate>>({
      name: '',
      job_type: 'research',
      goal: '',
      max_iterations: 50,
      max_tool_calls: 200,
      max_llm_calls: 100,
      max_runtime_minutes: 30,
      start_immediately: true,
      config: {},
    });

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      if (!formData.name || !formData.goal) {
        toast.error('Name and goal are required');
        return;
      }
      createMutation.mutate(formData as AgentJobCreate);
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <h2 className="text-lg font-semibold mb-4">Create Autonomous Job</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="My Research Job"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Job Type</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={formData.job_type}
                  onChange={(e) => setFormData({ ...formData, job_type: e.target.value as AgentJobType })}
                >
                  <option value="research">Research</option>
                  <option value="analysis">Analysis</option>
                  <option value="data_analysis">Data Analysis (ETL, Charts, Diagrams)</option>
                  <option value="monitor">Monitor</option>
                  <option value="synthesis">Synthesis</option>
                  <option value="knowledge_expansion">Knowledge Expansion</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={4}
                  value={formData.goal}
                  onChange={(e) => setFormData({ ...formData, goal: e.target.value })}
                  placeholder="Research the latest developments in transformer architectures..."
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Max Iterations</label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={formData.max_iterations}
                    onChange={(e) => setFormData({ ...formData, max_iterations: parseInt(e.target.value) })}
                    min={1}
                    max={1000}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Max Runtime (min)</label>
                  <input
                    type="number"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={formData.max_runtime_minutes}
                    onChange={(e) => setFormData({ ...formData, max_runtime_minutes: parseInt(e.target.value) })}
                    min={1}
                    max={480}
                  />
                </div>
              </div>


              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-700 mb-2">LLM Routing (optional)</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Tier</label>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={String((formData.config as any)?.llm_tier || '')}
                      onChange={(e) => {
                        const tier = e.target.value;
                        const cfg = { ...((formData.config as any) || {}) };
                        if (!tier) {
                          delete (cfg as any).llm_tier;
                        } else {
                          (cfg as any).llm_tier = tier;
                        }
                        setFormData({ ...formData, config: cfg });
                      }}
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
                      value={String(((formData.config as any)?.llm_fallback_tiers || []).join(', '))}
                      onChange={(e) => {
                        const raw = e.target.value;
                        const arr = raw
                          .split(',')
                          .map((s) => s.trim())
                          .filter(Boolean);
                        const cfg = { ...((formData.config as any) || {}) };
                        if (arr.length === 0) {
                          delete (cfg as any).llm_fallback_tiers;
                        } else {
                          (cfg as any).llm_fallback_tiers = arr;
                        }
                        setFormData({ ...formData, config: cfg });
                      }}
                      placeholder="balanced, fast"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Timeout (sec)</label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={String((formData.config as any)?.llm_timeout_seconds ?? '')}
                      onChange={(e) => {
                        const v = e.target.value;
                        const cfg = { ...((formData.config as any) || {}) };
                        if (!v) {
                          delete (cfg as any).llm_timeout_seconds;
                        } else {
                          (cfg as any).llm_timeout_seconds = parseInt(v, 10);
                        }
                        setFormData({ ...formData, config: cfg });
                      }}
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
                      value={String((formData.config as any)?.llm_max_tokens_cap ?? '')}
                      onChange={(e) => {
                        const v = e.target.value;
                        const cfg = { ...((formData.config as any) || {}) };
                        if (!v) {
                          delete (cfg as any).llm_max_tokens_cap;
                        } else {
                          (cfg as any).llm_max_tokens_cap = parseInt(v, 10);
                        }
                        setFormData({ ...formData, config: cfg });
                      }}
                      min={64}
                      max={20000}
                      placeholder="2000"
                    />
                  </div>
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  Uses feature flags <span className="font-mono">llm_provider_* / llm_model_*</span> for tier resolution; falls back on failures.
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="start_immediately"
                  checked={formData.start_immediately}
                  onChange={(e) => setFormData({ ...formData, start_immediately: e.target.checked })}
                />
                <label htmlFor="start_immediately" className="text-sm text-gray-700">
                  Start immediately
                </label>
              </div>

              <div className="flex justify-end gap-3 pt-4 border-t">
                <Button type="button" variant="secondary" onClick={() => setShowCreateModal(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={createMutation.isLoading}>
                  {createMutation.isLoading ? 'Creating...' : 'Create Job'}
                </Button>
              </div>
            </form>
          </div>
        </div>
      </div>
    );
  };

  // Create from template modal
  const CreateFromTemplateModal: React.FC<{ template: AgentJobTemplate }> = ({ template }) => {
    const [name, setName] = useState(`${template.display_name} - ${new Date().toLocaleDateString()}`);
    const [goal, setGoal] = useState(template.default_goal || '');
    const [configText, setConfigText] = useState(
      template.default_config ? JSON.stringify(template.default_config, null, 2) : ''
    );
    const isCodePatchTemplate = template.name === 'code_patch_proposer';
    const [selectedTargetSourceId, setSelectedTargetSourceId] = useState<string>('');

    const { data: documentSources } = useQuery(
      ['document-sources', 'all'],
      () => apiClient.getDocumentSources(),
      { staleTime: 30000 }
    );

    const codeSources = useMemo(() => {
      const items = (documentSources || []) as any[];
      return items.filter((s) => ['github', 'gitlab'].includes(String(s?.source_type || s?.sourceType || '').toLowerCase()));
    }, [documentSources]);

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      if (!name) {
        toast.error('Name is required');
        return;
      }
      let parsedConfig: any | undefined = undefined;
      if (configText.trim()) {
        try {
          parsedConfig = JSON.parse(configText);
        } catch (err) {
          toast.error('Config must be valid JSON');
          return;
        }
      }
      createFromTemplateMutation.mutate({
        template_id: template.id,
        name,
        goal: goal !== template.default_goal ? goal : undefined,
        config: parsedConfig,
      });
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <h2 className="text-lg font-semibold mb-1">Create from Template</h2>
            <p className="text-sm text-gray-500 mb-4">{template.display_name}</p>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Job Name</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={4}
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                />
              </div>

              {isCodePatchTemplate && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Target code source</label>
                  <select
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={selectedTargetSourceId}
                    onChange={(e) => {
                      const id = e.target.value;
                      setSelectedTargetSourceId(id);
                      try {
                        const obj = configText.trim() ? JSON.parse(configText) : {};
                        const next = { ...(obj || {}), target_source_id: id };
                        setConfigText(JSON.stringify(next, null, 2));
                      } catch {
                        setConfigText(JSON.stringify({ target_source_id: id }, null, 2));
                      }
                    }}
                  >
                    <option value="">Select a git source…</option>
                    {codeSources.map((s: any) => (
                      <option key={String(s.id)} value={String(s.id)}>
                        {String(s.name || s.id)}
                      </option>
                    ))}
                  </select>
                  <div className="mt-1 text-xs text-gray-500">
                    This should be a GitHub/GitLab document source (code ingested into the KB).
                  </div>
                </div>
              )}

              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-2">Template Configuration</p>
                <div className="flex gap-4 text-sm text-gray-600">
                  <span>Max {template.default_max_iterations} iterations</span>
                  <span>{template.default_max_runtime_minutes} min runtime</span>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-700 mb-2">LLM Routing (applies to config JSON)</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Tier</label>
                    <select
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={(() => {
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          return String((obj as any)?.llm_tier || '');
                        } catch {
                          return '';
                        }
                      })()}
                      onChange={(e) => {
                        const tier = e.target.value;
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          const next = { ...(obj || {}) } as any;
                          if (!tier) delete next.llm_tier;
                          else next.llm_tier = tier;
                          setConfigText(JSON.stringify(next, null, 2));
                        } catch {
                          const next: any = {};
                          if (tier) next.llm_tier = tier;
                          setConfigText(JSON.stringify(next, null, 2));
                        }
                      }}
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
                      value={(() => {
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          return String(((obj as any)?.llm_fallback_tiers || []).join(', '));
                        } catch {
                          return '';
                        }
                      })()}
                      onChange={(e) => {
                        const raw = e.target.value;
                        const arr = raw
                          .split(',')
                          .map((s) => s.trim())
                          .filter(Boolean);
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          const next = { ...(obj || {}) } as any;
                          if (arr.length === 0) delete next.llm_fallback_tiers;
                          else next.llm_fallback_tiers = arr;
                          setConfigText(JSON.stringify(next, null, 2));
                        } catch {
                          const next: any = {};
                          if (arr.length) next.llm_fallback_tiers = arr;
                          setConfigText(JSON.stringify(next, null, 2));
                        }
                      }}
                      placeholder="balanced, fast"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Timeout (sec)</label>
                    <input
                      type="number"
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                      value={(() => {
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          const v = (obj as any)?.llm_timeout_seconds;
                          return v === undefined || v === null ? '' : String(v);
                        } catch {
                          return '';
                        }
                      })()}
                      onChange={(e) => {
                        const v = e.target.value;
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          const next = { ...(obj || {}) } as any;
                          if (!v) delete next.llm_timeout_seconds;
                          else next.llm_timeout_seconds = parseInt(v, 10);
                          setConfigText(JSON.stringify(next, null, 2));
                        } catch {
                          const next: any = {};
                          if (v) next.llm_timeout_seconds = parseInt(v, 10);
                          setConfigText(JSON.stringify(next, null, 2));
                        }
                      }}
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
                      value={(() => {
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          const v = (obj as any)?.llm_max_tokens_cap;
                          return v === undefined || v === null ? '' : String(v);
                        } catch {
                          return '';
                        }
                      })()}
                      onChange={(e) => {
                        const v = e.target.value;
                        try {
                          const obj = configText.trim() ? JSON.parse(configText) : {};
                          const next = { ...(obj || {}) } as any;
                          if (!v) delete next.llm_max_tokens_cap;
                          else next.llm_max_tokens_cap = parseInt(v, 10);
                          setConfigText(JSON.stringify(next, null, 2));
                        } catch {
                          const next: any = {};
                          if (v) next.llm_max_tokens_cap = parseInt(v, 10);
                          setConfigText(JSON.stringify(next, null, 2));
                        }
                      }}
                      min={64}
                      max={20000}
                      placeholder="2000"
                    />
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Config (JSON)</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono"
                  rows={6}
                  value={configText}
                  onChange={(e) => setConfigText(e.target.value)}
                  placeholder='{"key":"value"}'
                />
                {template.name === 'code_patch_proposer' && (
                  <div className="mt-1 text-xs text-gray-500">
                    Required: <span className="font-mono">target_source_id</span> (UUID of a git document source).
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-3 pt-4 border-t">
                <Button type="button" variant="secondary" onClick={() => setCreateFromTemplate(null)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={createFromTemplateMutation.isLoading}>
                  {createFromTemplateMutation.isLoading ? 'Creating...' : 'Create Job'}
                </Button>
              </div>
            </form>
          </div>
        </div>
      </div>
    );
  };

  const InboxMonitorModal: React.FC = () => {
    const [name, setName] = useState('Research Inbox Monitor');
    const [customer, setCustomer] = useState('');
    const [customerContext, setCustomerContext] = useState('');
    const [intervalMinutes, setIntervalMinutes] = useState(60);
    const [maxDocuments, setMaxDocuments] = useState(8);
    const [maxPapers, setMaxPapers] = useState(8);
    const [includeDocuments, setIncludeDocuments] = useState(true);
    const [includeArxiv, setIncludeArxiv] = useState(true);
    const [monitorQueriesText, setMonitorQueriesText] = useState('');
    const [persistArtifacts, setPersistArtifacts] = useState(false);
    const [autoAddToReadingList, setAutoAddToReadingList] = useState(false);
    const [readingListName, setReadingListName] = useState('Research Inbox');
    const [runImmediately, setRunImmediately] = useState(true);

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      const preferSources: string[] = [];
      if (includeDocuments) preferSources.push('documents');
      if (includeArxiv) preferSources.push('arxiv');

      const monitorQueries = monitorQueriesText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);

      const goal =
        (customer || '').trim().length > 0
          ? `Continuously monitor for customer-relevant updates (${customer.trim()}) and file them into the Research Inbox.`
          : 'Continuously monitor for customer-relevant updates and file them into the Research Inbox.';

      createInboxMonitorMutation.mutate({
        name,
        job_type: 'monitor',
        goal,
        config: {
          deterministic_runner: 'research_inbox_monitor',
          customer: customer.trim(),
          customer_context: customerContext,
          prefer_sources: preferSources,
          monitor_queries: monitorQueries,
          max_documents: maxDocuments,
          max_papers: maxPapers,
          interval_minutes: intervalMinutes,
          persist_artifacts: persistArtifacts,
          auto_add_to_reading_list: autoAddToReadingList,
          reading_list_name: readingListName,
        },
        schedule_type: 'continuous',
        start_immediately: runImmediately,
      });
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-xl max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold">Create Research Inbox Monitor</h2>
                <p className="text-sm text-gray-500">Runs continuously and files new items into your inbox</p>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setShowInboxMonitorModal(false)}>
                <XCircle className="w-5 h-5" />
              </Button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Interval (minutes)</label>
                  <input
                    type="number"
                    min={1}
                    max={1440}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={intervalMinutes}
                    onChange={(e) => setIntervalMinutes(parseInt(e.target.value || '60', 10))}
                  />
                </div>
                <div className="flex items-center gap-2 pt-6">
                  <input
                    type="checkbox"
                    checked={runImmediately}
                    onChange={(e) => setRunImmediately(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Run immediately</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Customer (optional)</label>
                <input
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={customer}
                  onChange={(e) => setCustomer(e.target.value)}
                  placeholder="Acme Corp"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Customer context (optional)</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  value={customerContext}
                  onChange={(e) => setCustomerContext(e.target.value)}
                  placeholder="What we care about, constraints, success metrics..."
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeDocuments}
                    onChange={(e) => setIncludeDocuments(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Search internal documents</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeArxiv}
                    onChange={(e) => setIncludeArxiv(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Search arXiv</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Max KB docs / run</label>
                  <input
                    type="number"
                    min={0}
                    max={50}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={maxDocuments}
                    onChange={(e) => setMaxDocuments(parseInt(e.target.value || '8', 10))}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Max papers / run</label>
                  <input
                    type="number"
                    min={0}
                    max={50}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={maxPapers}
                    onChange={(e) => setMaxPapers(parseInt(e.target.value || '8', 10))}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Monitor queries (optional, one per line)</label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  rows={3}
                  value={monitorQueriesText}
                  onChange={(e) => setMonitorQueriesText(e.target.value)}
                  placeholder="e.g.\nLLM safety evaluation\ncustomer SLA latency"
                />
                <p className="text-xs text-gray-500 mt-1">If empty, queries are derived from goal + customer profile/context.</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={persistArtifacts}
                    onChange={(e) => setPersistArtifacts(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Persist weekly brief doc</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={autoAddToReadingList}
                    onChange={(e) => setAutoAddToReadingList(e.target.checked)}
                  />
                  <span className="text-sm text-gray-700">Auto-add docs to reading list</span>
                </div>
              </div>

              {autoAddToReadingList ? (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Reading list name</label>
                  <input
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={readingListName}
                    onChange={(e) => setReadingListName(e.target.value)}
                  />
                </div>
              ) : null}

              <div className="flex justify-end gap-3 pt-4 border-t">
                <Button type="button" variant="secondary" onClick={() => setShowInboxMonitorModal(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={createInboxMonitorMutation.isLoading}>
                  {createInboxMonitorMutation.isLoading ? 'Creating…' : 'Create Monitor'}
                </Button>
              </div>
            </form>
          </div>
        </div>
      </div>
    );
  };

  const MonitorProfilesModal: React.FC = () => {
    const profiles = (monitorProfiles || []) as any[];
    const [selectedCustomer, setSelectedCustomer] = useState<string>('');
    const selected = useMemo(() => {
      const key = (selectedCustomer || '').trim();
      if (!key) {
        return profiles.find((p: any) => !p?.customer) || null;
      }
      return profiles.find((p: any) => String(p?.customer || '') === key) || null;
    }, [profiles, selectedCustomer]);

    const [mutedTokensText, setMutedTokensText] = useState<string>('');
    const [mutedPatternsText, setMutedPatternsText] = useState<string>('');
    const [notes, setNotes] = useState<string>('');

    useEffect(() => {
      const mt = Array.isArray(selected?.muted_tokens) ? selected.muted_tokens : [];
      const mp = Array.isArray(selected?.muted_patterns) ? selected.muted_patterns : [];
      setMutedTokensText(mt.join('\n'));
      setMutedPatternsText(mp.join('\n'));
      setNotes(String(selected?.notes || ''));
    }, [selected?.id]);

    const tokenScores = (selected?.token_scores || {}) as Record<string, number>;
    const topPositive = Object.entries(tokenScores)
      .filter(([, v]) => typeof v === 'number' && v > 0)
      .sort((a, b) => (b[1] as number) - (a[1] as number))
      .slice(0, 8);
    const topNegative = Object.entries(tokenScores)
      .filter(([, v]) => typeof v === 'number' && v < 0)
      .sort((a, b) => (a[1] as number) - (b[1] as number))
      .slice(0, 8);

    const handleSave = () => {
      const customer = (selectedCustomer || '').trim() || (selected?.customer ? String(selected.customer) : '');
      const muted_tokens = mutedTokensText
        .split('\n')
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean);
      const muted_patterns = mutedPatternsText
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean);

      upsertMonitorProfileMutation.mutate({
        customer: customer || undefined,
        muted_tokens,
        muted_patterns,
        notes: notes || undefined,
        merge_lists: false,
      } as any);
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold">Monitor Profiles</h2>
              <p className="text-sm text-gray-500">Manage mutes and inspect learned tokens (per customer)</p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" onClick={() => refetchMonitorProfiles()}>
                <RefreshCw className="w-4 h-4" />
              </Button>
              <Button variant="ghost" onClick={() => setShowMonitorProfilesModal(false)}>
                <XCircle className="w-5 h-5" />
              </Button>
            </div>
          </div>

          <div className="flex flex-1 min-h-0">
            <div className="w-1/3 border-r border-gray-200 p-4 overflow-y-auto">
              <div className="text-sm font-medium text-gray-900 mb-2">Profiles</div>
              {monitorProfilesLoading ? (
                <div className="py-6 flex justify-center">
                  <LoadingSpinner />
                </div>
              ) : (
                <div className="space-y-2">
                  <button
                    className={`w-full text-left px-3 py-2 rounded border ${
                      !selectedCustomer ? 'border-primary-300 bg-primary-50' : 'border-gray-200 hover:bg-gray-50'
                    }`}
                    onClick={() => setSelectedCustomer('')}
                  >
                    <div className="text-sm font-medium text-gray-900">Global</div>
                    <div className="text-xs text-gray-500">Applies when customer is empty</div>
                  </button>
                  {profiles
                    .filter((p: any) => !!p?.customer)
                    .map((p: any) => (
                      <button
                        key={p.id}
                        className={`w-full text-left px-3 py-2 rounded border ${
                          selectedCustomer === String(p.customer)
                            ? 'border-primary-300 bg-primary-50'
                            : 'border-gray-200 hover:bg-gray-50'
                        }`}
                        onClick={() => setSelectedCustomer(String(p.customer))}
                      >
                        <div className="text-sm font-medium text-gray-900 truncate">{String(p.customer)}</div>
                        <div className="text-xs text-gray-500">Updated: {new Date(p.updated_at).toLocaleString()}</div>
                      </button>
                    ))}
                  <div className="mt-4">
                    <div className="text-xs text-gray-500 mb-1">Create / select customer</div>
                    <input
                      className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                      placeholder="Customer tag (e.g. Acme)"
                      value={selectedCustomer}
                      onChange={(e) => setSelectedCustomer(e.target.value)}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="w-2/3 p-4 overflow-y-auto">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="text-sm font-medium text-gray-900">
                    {selectedCustomer ? `Customer: ${selectedCustomer}` : 'Global profile'}
                  </div>
                  <div className="text-xs text-gray-500">
                    Learned tokens come from accept/reject; mutes are applied immediately to monitors.
                  </div>
                </div>
                <Button
                  variant="secondary"
                  onClick={handleSave}
                  disabled={upsertMonitorProfileMutation.isLoading}
                >
                  {upsertMonitorProfileMutation.isLoading ? 'Saving…' : 'Save'}
                </Button>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="bg-gray-50 border border-gray-200 rounded p-3">
                  <div className="text-xs font-medium text-gray-700 mb-2">Top positive tokens</div>
                  {topPositive.length === 0 ? (
                    <div className="text-xs text-gray-500">No learned positives yet.</div>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {topPositive.map(([t, v]) => (
                        <span key={t} className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                          {t} (+{v})
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <div className="bg-gray-50 border border-gray-200 rounded p-3">
                  <div className="text-xs font-medium text-gray-700 mb-2">Top negative tokens</div>
                  {topNegative.length === 0 ? (
                    <div className="text-xs text-gray-500">No learned negatives yet.</div>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {topNegative.map(([t, v]) => (
                        <span key={t} className="text-xs bg-red-100 text-red-800 px-2 py-1 rounded">
                          {t} ({v})
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Muted tokens (one per line)</label>
                  <textarea
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                    rows={10}
                    value={mutedTokensText}
                    onChange={(e) => setMutedTokensText(e.target.value)}
                    placeholder="e.g.\nbenchmark\nnewsletter"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Muted phrases (substring match)</label>
                  <textarea
                    className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                    rows={10}
                    value={mutedPatternsText}
                    onChange={(e) => setMutedPatternsText(e.target.value)}
                    placeholder="e.g.\nweekly roundup\ncall for papers"
                  />
                </div>
              </div>
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                <textarea
                  className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                  rows={3}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const CustomerResearchModal: React.FC = () => {
    const [name, setName] = useState(`Customer Research — ${new Date().toLocaleDateString()}`);
    const [goal, setGoal] = useState('');
    const [customerContext, setCustomerContext] = useState('');
    const [persistArtifacts, setPersistArtifacts] = useState(false);
    const [addToReadingList, setAddToReadingList] = useState(true);
    const [readingListName, setReadingListName] = useState('Customer Research');
    const [runDeepDive, setRunDeepDive] = useState(false);
    const [sourcePreference, setSourcePreference] = useState<'documents_first' | 'balanced' | 'arxiv_first'>('documents_first');
    const [maxDocuments, setMaxDocuments] = useState(12);
    const [maxPapers, setMaxPapers] = useState(8);

    const templates = (templatesData as any)?.templates || [];
    const deepDiveTemplate = templates.find((t: any) => t?.name === 'customer_research_scout_deep_dive');
    const scoutTemplate = templates.find((t: any) => t?.name === 'customer_research_scout');
    const template =
      (runDeepDive ? deepDiveTemplate : null) ||
      scoutTemplate ||
      deepDiveTemplate ||
      templates.find((t: any) => t?.category === 'research');

    const deepDiveChain = useMemo(() => {
      const chains = (chainsData as any)?.chains || [];
      return chains.find((c: any) => c?.name === 'customer_research_scout_deep_dive_chain') || null;
    }, [chainsData]);

    const preferSources =
      sourcePreference === 'documents_first'
        ? ['documents', 'arxiv']
        : sourcePreference === 'arxiv_first'
          ? ['arxiv', 'documents']
          : ['documents', 'arxiv'];

    const handleCreate = (e: React.FormEvent) => {
      e.preventDefault();
      if (!template?.id) {
        toast.error('Customer research template not available');
        return;
      }
      if (!goal.trim()) {
        toast.error('Goal is required');
        return;
      }

      const chainConfig = runDeepDive && !deepDiveTemplate
        ? {
            trigger_condition: 'on_complete' as const,
            inherit_results: true,
            inherit_config: true,
            child_jobs: [
              {
                name: 'Customer Research — Deep Dive',
                job_type: 'research' as const,
                goal:
                  'Deep-dive using inherited results from the scout job. Focus on the top internal documents and any high-signal papers. ' +
                  'Output: (1) 3-5 hypotheses, (2) risks/unknowns, (3) minimal experiment plan (metrics + timeline), ' +
                  'and (4) a short brief document.',
                config: {
                  prefer_sources: ['documents'],
                  max_documents: 6,
                  max_papers: 0,
                },
                max_iterations: 6,
                max_tool_calls: 40,
                max_llm_calls: 12,
                max_runtime_minutes: 10,
              },
            ],
          }
        : undefined;

      // Prefer starting from the built-in chain definition if available: it creates a first job and chains the deep dive.
      if (runDeepDive && deepDiveChain?.id) {
        createFromChainMutation.mutate({
          chain_definition_id: String(deepDiveChain.id),
          name_prefix: name.trim(),
          variables: { goal: goal.trim() },
          config_overrides: {
            customer_context: customerContext.trim() || undefined,
            persist_artifacts: !!persistArtifacts,
            reading_list_name: addToReadingList ? (readingListName.trim() || 'Customer Research') : undefined,
            prefer_sources: preferSources,
            max_documents: Math.max(1, Math.min(200, Number(maxDocuments) || 12)),
            max_papers: Math.max(1, Math.min(200, Number(maxPapers) || 8)),
          },
          start_immediately: true,
        } as any);
        return;
      }

      createFromTemplateMutation.mutate({
        template_id: template.id,
        name: name.trim(),
        goal: goal.trim(),
        start_immediately: true,
        chain_config: chainConfig as any,
        config: {
          customer_context: customerContext.trim() || undefined,
          persist_artifacts: !!persistArtifacts,
          reading_list_name: addToReadingList ? (readingListName.trim() || 'Customer Research') : undefined,
          prefer_sources: preferSources,
          max_documents: Math.max(1, Math.min(200, Number(maxDocuments) || 12)),
          max_papers: Math.max(1, Math.min(200, Number(maxPapers) || 8)),
        },
      });
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl">
          <div className="p-6 border-b border-gray-200 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary-100 text-primary-600">
                <Brain className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Customer Research</h2>
                <p className="text-sm text-gray-500">
                  Uses the deployment customer profile + optional context to run a tailored research loop.
                </p>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={() => setShowCustomerResearchModal(false)}>
              <XCircle className="w-5 h-5" />
            </Button>
          </div>

          <form onSubmit={handleCreate} className="p-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Job Name</label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Source preference</label>
                <select
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={sourcePreference}
                  onChange={(e) => setSourcePreference(e.target.value as any)}
                >
                  <option value="documents_first">Prefer internal documents first</option>
                  <option value="balanced">Balanced</option>
                  <option value="arxiv_first">Prefer arXiv first</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Goal</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="E.g., 'Summarize the state of the art on X and propose 3 experiments tailored to our constraints.'"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Customer context (optional)</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                rows={3}
                value={customerContext}
                onChange={(e) => setCustomerContext(e.target.value)}
                placeholder="Any extra details not captured in the deployment customer profile."
              />
              <div className="mt-1 text-xs text-gray-500">
                This is combined with the deployment customer profile (Admin → AI Hub) when generating the plan.
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max internal docs</label>
                <input
                  type="number"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={maxDocuments}
                  onChange={(e) => setMaxDocuments(parseInt(e.target.value || '0', 10))}
                  min={1}
                  max={200}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Max papers</label>
                <input
                  type="number"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={maxPapers}
                  onChange={(e) => setMaxPapers(parseInt(e.target.value || '0', 10))}
                  min={1}
                  max={200}
                />
              </div>
            </div>

            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={addToReadingList}
                  onChange={(e) => setAddToReadingList(e.target.checked)}
                />
                Allow adding relevant documents to a reading list
              </label>
              {addToReadingList && (
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Reading list name</label>
                  <input
                    type="text"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    value={readingListName}
                    onChange={(e) => setReadingListName(e.target.value)}
                  />
                </div>
              )}
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={persistArtifacts}
                  onChange={(e) => setPersistArtifacts(e.target.checked)}
                />
                Allow saving a brief document (optional)
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={runDeepDive}
                  onChange={(e) => setRunDeepDive(e.target.checked)}
                />
                Auto-run a deep-dive follow-up job
              </label>
              <div className="text-xs text-gray-500">
                If enabled, the agent may create a short “Customer Research Brief” document in the knowledge base.
              </div>
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => setShowCustomerResearchModal(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={createFromTemplateMutation.isLoading}>
                {createFromTemplateMutation.isLoading ? 'Starting...' : 'Start Research Job'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  const StartChainModal: React.FC<{ chain: AgentJobChainDefinition }> = ({ chain }) => {
    const defaultPrefix = `${chain.display_name} — ${new Date().toLocaleDateString()}`;
    const [namePrefix, setNamePrefix] = useState(defaultPrefix);
    const [startImmediately, setStartImmediately] = useState(true);
    const [configOverridesRaw, setConfigOverridesRaw] = useState<string>('');
    const [showAdvanced, setShowAdvanced] = useState(false);

    const variableKeys = useMemo(() => {
      const keys = new Set<string>();
      const steps = (chain as any)?.chain_steps || [];
      const re = /\{([a-zA-Z0-9_]+)\}/g;
      for (const s of steps) {
        const tmpl = String(s?.goal_template || '');
        let m: RegExpExecArray | null;
        while ((m = re.exec(tmpl)) !== null) {
          if (m[1]) keys.add(m[1]);
        }
      }
      return Array.from(keys).sort();
    }, [chain]);

    const [variables, setVariables] = useState<Record<string, string>>(() => {
      const initial: Record<string, string> = {};
      for (const k of variableKeys) initial[k] = '';
      return initial;
    });

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      if (!namePrefix.trim()) {
        toast.error('Name prefix is required');
        return;
      }
      const payloadVars: Record<string, string> = {};
      for (const k of variableKeys) {
        const v = String(variables[k] || '').trim();
        if (v) payloadVars[k] = v;
      }

      let config_overrides: Record<string, any> | undefined = undefined;
      const raw = (configOverridesRaw || '').trim();
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            config_overrides = parsed;
          } else {
            toast.error('Config overrides must be a JSON object');
            return;
          }
        } catch {
          toast.error('Invalid JSON in config overrides');
          return;
        }
      }
      createFromChainMutation.mutate({
        chain_definition_id: chain.id,
        name_prefix: namePrefix.trim(),
        variables: payloadVars,
        config_overrides,
        start_immediately: startImmediately,
      });
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl w-full max-w-lg">
          <div className="p-6 border-b border-gray-200 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                <GitBranch className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Start Chain</h2>
                <p className="text-sm text-gray-500">{chain.display_name}</p>
              </div>
            </div>
            <Button variant="ghost" size="sm" onClick={() => setStartFromChain(null)}>
              <XCircle className="w-5 h-5" />
            </Button>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name prefix</label>
              <input
                type="text"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={namePrefix}
                onChange={(e) => setNamePrefix(e.target.value)}
              />
              <div className="mt-1 text-xs text-gray-500">
                Used to name each step job (e.g., “{namePrefix} - Step 1”).
              </div>
            </div>

            {variableKeys.length > 0 ? (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                <div className="text-sm font-medium text-gray-800 mb-2">Variables</div>
                <div className="space-y-2">
                  {variableKeys.map((k) => (
                    <div key={k}>
                      <label className="block text-xs font-medium text-gray-600 mb-1">{k}</label>
                      <input
                        type="text"
                        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                        value={variables[k] || ''}
                        onChange={(e) => setVariables((prev) => ({ ...prev, [k]: e.target.value }))}
                        placeholder={`Value for {${k}}`}
                      />
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-sm text-gray-600 bg-gray-50 border border-gray-200 rounded-lg p-3">
                This chain has no variables.
              </div>
            )}

            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={startImmediately}
                onChange={(e) => setStartImmediately(e.target.checked)}
              />
              Start immediately
            </label>

            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
              <button
                type="button"
                className="text-sm font-medium text-gray-800"
                onClick={() => setShowAdvanced((v) => !v)}
              >
                Advanced: config overrides (JSON)
              </button>
              {showAdvanced && (
                <div className="mt-2 space-y-2">
                  <div className="text-xs text-gray-500">
                    Optional. Passed as <span className="font-mono">config_overrides</span> to the chain start request.
                  </div>
                  <textarea
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-xs font-mono"
                    rows={6}
                    value={configOverridesRaw}
                    onChange={(e) => setConfigOverridesRaw(e.target.value)}
                    placeholder='{"latex_project_id":"...","target_source_id":"...","search_query":"..."}'
                  />
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t">
              <Button type="button" variant="secondary" onClick={() => setStartFromChain(null)}>
                Cancel
              </Button>
              <Button type="submit" disabled={createFromChainMutation.isLoading}>
                {createFromChainMutation.isLoading ? 'Starting…' : 'Start Chain'}
              </Button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Autonomous Agents</h1>
          <p className="text-gray-500">Manage background jobs that work autonomously toward goals</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowCustomerResearchModal(true)}>
            <Brain className="w-4 h-4 mr-2" />
            Customer Research
          </Button>
          <Button onClick={() => setShowCreateModal(true)}>
            <Plus className="w-4 h-4 mr-2" />
            New Job
          </Button>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-5 gap-4 mb-6">
          <StatsCard title="Total Jobs" value={stats.total_jobs} icon={FileText} color="bg-gray-100 text-gray-600" />
          <StatsCard title="Running" value={stats.running_jobs} icon={Play} color="bg-blue-100 text-blue-600" />
          <StatsCard title="Completed" value={stats.completed_jobs} icon={CheckCircle2} color="bg-green-100 text-green-600" />
          <StatsCard title="Failed" value={stats.failed_jobs} icon={AlertCircle} color="bg-red-100 text-red-600" />
          <StatsCard
            title="Success Rate"
            value={stats.success_rate ? `${(stats.success_rate * 100).toFixed(0)}%` : '-'}
            icon={BarChart3}
            color="bg-purple-100 text-purple-600"
          />
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-4 mb-4 border-b border-gray-200">
        <button
          className={`pb-2 px-1 text-sm font-medium ${
            activeTab === 'jobs'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('jobs')}
        >
          My Jobs
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'inbox'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('inbox')}
        >
          <Inbox className="w-4 h-4" />
          Research Inbox
          {inboxStats?.new ? (
            <span className="ml-1 text-xs bg-primary-100 text-primary-700 px-1.5 py-0.5 rounded">
              {inboxStats.new}
            </span>
          ) : null}
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium ${
            activeTab === 'templates'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('templates')}
        >
          Templates
        </button>
        <button
          className={`pb-2 px-1 text-sm font-medium flex items-center gap-1 ${
            activeTab === 'chains'
              ? 'text-primary-600 border-b-2 border-primary-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
          onClick={() => setActiveTab('chains')}
        >
          <GitBranch className="w-4 h-4" />
          Job Chains
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 flex gap-6 min-h-0">
        {activeTab === 'jobs' && (
          <>
            {/* Jobs list */}
            <div className="w-2/3 flex flex-col">
              {/* Filters */}
              <div className="flex gap-3 mb-4">
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <option value="">All Status</option>
                  <option value="pending">Pending</option>
                  <option value="running">Running</option>
                  <option value="paused">Paused</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value)}
                >
                  <option value="">All Types</option>
                  <option value="research">Research</option>
                  <option value="analysis">Analysis</option>
                  <option value="data_analysis">Data Analysis</option>
                  <option value="monitor">Monitor</option>
                  <option value="synthesis">Synthesis</option>
                  <option value="knowledge_expansion">Knowledge Expansion</option>
                  <option value="custom">Custom</option>
                </select>
                <label className="inline-flex items-center gap-2 text-sm text-gray-700 px-2">
                  <input
                    type="checkbox"
                    className="rounded border-gray-300"
                    checked={swarmOnlyFilter}
                    onChange={(e) => setSwarmOnlyFilter(Boolean(e.target.checked))}
                  />
                  Swarm only
                </label>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={swarmSortBy}
                  onChange={(e) => setSwarmSortBy(e.target.value)}
                >
                  <option value="created_desc">Newest first</option>
                  <option value="created_asc">Oldest first</option>
                  <option value="swarm_confidence_desc">Swarm confidence</option>
                  <option value="swarm_consensus_desc">Swarm consensus</option>
                  <option value="swarm_conflicts_desc">Swarm conflicts</option>
                </select>
                <select
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  value={String(swarmMinConsensus)}
                  onChange={(e) => setSwarmMinConsensus(Number(e.target.value || 0))}
                >
                  <option value="0">Any consensus</option>
                  <option value="1">Consensus &ge; 1</option>
                  <option value="2">Consensus &ge; 2</option>
                  <option value="3">Consensus &ge; 3</option>
                  <option value="5">Consensus &ge; 5</option>
                  <option value="8">Consensus &ge; 8</option>
                </select>
                {(swarmOnlyFilter || swarmSortBy !== 'created_desc' || swarmMinConsensus > 0) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setSwarmOnlyFilter(false);
                      setSwarmSortBy('created_desc');
                      setSwarmMinConsensus(0);
                    }}
                    title="Reset swarm-specific filters"
                  >
                    <XCircle className="w-4 h-4 mr-1" />
                    Clear Swarm
                  </Button>
                )}
                <Button variant="ghost" size="sm" onClick={() => refetchJobs()}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>

              {/* Jobs grid */}
              {jobsLoading ? (
                <div className="flex justify-center items-center flex-1">
                  <LoadingSpinner />
                </div>
              ) : jobsData?.jobs.length === 0 ? (
                <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                  <Bot className="w-12 h-12 mb-3 text-gray-400" />
                  <p className="text-lg font-medium">No jobs yet</p>
                  <p className="text-sm">Create a new job or use a template to get started</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4 overflow-y-auto flex-1">
                  {jobsData?.jobs.map((job) => (
                    <JobCard key={job.id} job={job} />
                  ))}
                </div>
              )}
            </div>

            {/* Detail panel */}
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
          </>
        )}

        {activeTab === 'templates' && (
          <div className="w-full">
            <p className="text-sm text-gray-500 mb-4">
              Choose a template to quickly create a pre-configured autonomous job
            </p>
            {templatesData?.templates.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                <FileText className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">No templates available</p>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                {templatesData?.templates.map((template) => (
                  <TemplateCard key={template.id} template={template} />
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'chains' && (
          <div className="w-full">
            <p className="text-sm text-gray-500 mb-4">
              Job chains allow you to create multi-step workflows where jobs automatically trigger subsequent jobs on completion
            </p>
            {chainsData?.chains.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                <GitBranch className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">No chain definitions yet</p>
                <p className="text-sm">Chain definitions allow you to create multi-step workflows</p>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                {chainsData?.chains.map((chain) => (
                  <div
                    key={chain.id}
                    className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start gap-3 mb-3">
                      <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                        <GitBranch className="w-5 h-5" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900">{chain.display_name}</h3>
                        <p className="text-sm text-gray-500">{chain.chain_steps.length} steps</p>
                      </div>
                      {chain.is_system && (
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">System</span>
                      )}
                    </div>
                    {chain.description && (
                      <p className="text-sm text-gray-600 mb-3 line-clamp-2">{chain.description}</p>
                    )}
                    <div className="flex flex-wrap gap-2 mb-3">
                      {chain.chain_steps.slice(0, 3).map((step, idx) => (
                        <span key={idx} className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                          {step.step_name}
                        </span>
                      ))}
                      {chain.chain_steps.length > 3 && (
                        <span className="text-xs text-gray-500">+{chain.chain_steps.length - 3} more</span>
                      )}
                    </div>
                    <Button
                      size="sm"
                      variant="secondary"
                      className="w-full"
                      onClick={() => {
                        setStartFromChain(chain);
                      }}
                    >
                      <Play className="w-3 h-3 mr-1" />
                      Start Chain
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'inbox' && (
          <div className="w-full flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3 text-sm text-gray-600">
                <span className="font-medium text-gray-900">Research Inbox</span>
                <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded">Total: {inboxStats?.total ?? '-'}</span>
                <span className="bg-primary-100 text-primary-700 px-2 py-1 rounded">New: {inboxStats?.new ?? '-'}</span>
                <span className="bg-green-100 text-green-700 px-2 py-1 rounded">Accepted: {inboxStats?.accepted ?? '-'}</span>
                <span className="bg-red-100 text-red-700 px-2 py-1 rounded">Rejected: {inboxStats?.rejected ?? '-'}</span>
              </div>
              <div className="flex gap-2">
                <label className="flex items-center gap-2 text-xs text-gray-600 select-none" title={paperAlgoDefaultToggleTitle}>
                  <input
                    type="checkbox"
                    className="h-3 w-3"
                    checked={paperAlgoDefaultRunDemoCheck}
                    disabled={updateMyPreferencesMutation.isLoading}
                    onChange={(e) => updateMyPreferencesMutation.mutate({ paper_algo_default_run_demo_check: e.target.checked })}
                  />
                  <span>Default: Run demo check</span>
                </label>
                <Button variant="secondary" onClick={() => setShowInboxMonitorModal(true)}>
                  <Activity className="w-4 h-4 mr-2" />
                  Create Monitor
                </Button>
                <Button variant="secondary" onClick={() => setShowMonitorProfilesModal(true)}>
                  <Settings className="w-4 h-4 mr-2" />
                  Profiles
                </Button>
                <Button variant="ghost" onClick={() => refetchInbox()}>
                  <RefreshCw className="w-4 h-4" />
                </Button>
              </div>
            </div>

            <div className="flex gap-3 mb-4">
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={inboxStatusFilter}
                onChange={(e) => setInboxStatusFilter(e.target.value as any)}
              >
                <option value="">All Status</option>
                <option value="new">New</option>
                <option value="accepted">Accepted</option>
                <option value="rejected">Rejected</option>
              </select>
              <select
                className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={inboxTypeFilter}
                onChange={(e) => setInboxTypeFilter(e.target.value)}
              >
                <option value="">All Types</option>
                <option value="document">Document</option>
                <option value="arxiv">arXiv</option>
              </select>
                <div className="flex-1 relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    className="w-full border border-gray-300 rounded-lg pl-9 pr-3 py-2 text-sm"
                    placeholder="Search inbox items…"
                    value={inboxSearch}
                    onChange={(e) => setInboxSearch(e.target.value)}
                  />
                </div>
            </div>

            {(() => {
              const items = (inboxData?.items || []) as ResearchInboxItem[];
              const selectedIds = Object.keys(selectedInboxIds).filter((id) => selectedInboxIds[id]);
              const allSelected = items.length > 0 && selectedIds.length === items.length;
              if (items.length === 0) return null;
              return (
                <div className="flex items-center justify-between mb-3 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={allSelected}
                      onChange={(e) => {
                        const next: Record<string, boolean> = {};
                        if (e.target.checked) {
                          items.forEach((it) => (next[it.id] = true));
                        }
                        setSelectedInboxIds(next);
                      }}
                    />
                    <span className="text-sm text-gray-700">
                      Selected: {selectedIds.length}/{items.length}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={selectedIds.length === 0 || createMutation.isLoading || createFromChainMutation.isLoading}
                      onClick={() => {
                        const selectedItems = items.filter((it) => selectedInboxIds[it.id]);
                        if (selectedItems.length === 0) return;

                        const goal =
                          window.prompt(
                            'Follow-up research goal:',
                            'Deep-dive on the selected Research Inbox items and propose concrete next steps (hypotheses + experiment plan).'
                          ) || '';
                        if (!goal.trim()) return;

                        const docItems = selectedItems.filter((it) => it.item_type === 'document');
                        const paperItems = selectedItems.filter((it) => it.item_type === 'arxiv');

                        const top_documents = docItems.slice(0, 20).map((d) => ({
                          id: d.item_key,
                          title: d.title,
                          url: d.url,
                          score: null,
                          source: 'inbox',
                        }));
                        const top_papers = paperItems.slice(0, 20).map((p) => ({
                          id: p.item_key,
                          title: p.title,
                          url: p.url,
                          score: null,
                          source: 'inbox',
                        }));

                        const parent_findings = selectedItems.slice(0, 50).map((it) => ({
                          type: it.item_type === 'arxiv' ? 'paper' : 'document',
                          title: it.title,
                          id: it.item_key,
                          url: it.url,
                          snippet: it.summary,
                        }));

                        const customers = Array.from(new Set(selectedItems.map((it) => it.customer).filter(Boolean))) as string[];
                        const customerContextHint =
                          customers.length === 1 ? `Customer: ${customers[0]}` : customers.length > 1 ? `Customers: ${customers.join(', ')}` : '';

                        const chains = ((chainsData as any)?.chains || []) as any[];
                        const deepDiveChain =
                          chains.find((c: any) => c?.name === 'customer_research_scout_deep_dive_chain') || null;

                        if (deepDiveChain?.id) {
                          createFromChainMutation.mutate({
                            chain_definition_id: deepDiveChain.id,
                            name_prefix: `Inbox Research — ${new Date().toLocaleDateString()}`,
                            variables: { goal: goal.trim() },
                            config_overrides: {
                              customer_context: customerContextHint,
                              prefer_sources: ['documents', 'arxiv'],
                              max_documents: 12,
                              max_papers: 8,
                              persist_artifacts: false,
                              reading_list_name: 'Customer Research',
                              inherited_data: {
                                parent_results: {
                                  summary: `Seeded from ${selectedItems.length} Research Inbox items.`,
                                  research_bundle: {
                                    top_documents,
                                    top_papers,
                                    insights: [],
                                    next_steps: [],
                                    artifacts: [],
                                  },
                                  inbox_items: selectedItems,
                                },
                                parent_findings,
                              },
                            },
                            start_immediately: true,
                          });
                        } else {
                          // Fallback: single research job
                          createMutation.mutate({
                            name: `Inbox Research — ${new Date().toLocaleDateString()}`,
                            job_type: 'research',
                            goal: goal.trim(),
                            config: {
                              customer_context: customerContextHint,
                              prefer_sources: ['documents', 'arxiv'],
                              max_documents: 12,
                              max_papers: 8,
                              persist_artifacts: false,
                              reading_list_name: 'Customer Research',
                              inherited_data: {
                                parent_results: {
                                  summary: `Seeded from ${selectedItems.length} Research Inbox items.`,
                                  research_bundle: {
                                    top_documents,
                                    top_papers,
                                    insights: [],
                                    next_steps: [],
                                    artifacts: [],
                                  },
                                  inbox_items: selectedItems,
                                },
                                parent_findings,
                              },
                            },
                            start_immediately: true,
                          });
                        }
                        setActiveTab('jobs');
                      }}
                    >
                      <Sparkles className="w-4 h-4 mr-1" />
                      Research Selected
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={selectedIds.length === 0 || bulkUpdateInboxMutation.isLoading}
                      onClick={() => bulkUpdateInboxMutation.mutate({ itemIds: selectedIds, data: { status: 'accepted' } })}
                    >
                      <ThumbsUp className="w-4 h-4 mr-1" />
                      Accept Selected
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={selectedIds.length === 0 || bulkUpdateInboxMutation.isLoading}
                      onClick={() => {
                        const feedback = window.prompt('Reject reason (optional):') || undefined;
                        bulkUpdateInboxMutation.mutate({ itemIds: selectedIds, data: { status: 'rejected', feedback } });
                      }}
                    >
                      <ThumbsDown className="w-4 h-4 mr-1" />
                      Reject Selected
                    </Button>
                  </div>
                </div>
              );
            })()}

            {inboxLoading ? (
              <div className="flex justify-center items-center flex-1">
                <LoadingSpinner />
              </div>
            ) : (inboxData?.items || []).length === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 text-gray-500">
                <Inbox className="w-12 h-12 mb-3 text-gray-400" />
                <p className="text-lg font-medium">Inbox is empty</p>
                <p className="text-sm">Create a monitor or run customer research to discover items</p>
              </div>
            ) : (
              <div className="space-y-3 overflow-y-auto flex-1 pr-1">
                {(inboxData?.items || []).map((item: ResearchInboxItem) => (
                  <div key={item.id} className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={!!selectedInboxIds[item.id]}
                            onChange={(e) => setSelectedInboxIds((prev) => ({ ...prev, [item.id]: e.target.checked }))}
                          />
                          <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                            {item.item_type}
                          </span>
                          <span
                            className={`text-xs px-2 py-1 rounded ${
                              item.status === 'new'
                                ? 'bg-primary-100 text-primary-700'
                                : item.status === 'accepted'
                                ? 'bg-green-100 text-green-700'
                                : 'bg-red-100 text-red-700'
                            }`}
                          >
                            {item.status}
                          </span>
                          {item.customer ? (
                            <span className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded">
                              {item.customer}
                            </span>
                          ) : null}
                        </div>
                        <h3 className="mt-2 font-medium text-gray-900 truncate">{item.title}</h3>
                        {item.summary ? (
                          <p className="text-sm text-gray-600 mt-1 line-clamp-2">{item.summary}</p>
                        ) : null}
                        {(item.metadata as any)?.query ? (
                          <p className="text-xs text-gray-500 mt-2">
                            Why: matched query “{String((item.metadata as any).query).slice(0, 140)}”
                            {(item.metadata as any)?.bias?.source ? (
                              <span className="ml-2 bg-gray-100 text-gray-700 px-2 py-0.5 rounded">
                                bias: {String((item.metadata as any).bias.source)}
                              </span>
                            ) : null}
                          </p>
                        ) : null}
                        <div className="text-xs text-gray-500 mt-2 flex flex-wrap gap-x-4 gap-y-1">
                          <span>Discovered: {new Date(item.discovered_at).toLocaleString()}</span>
                          {item.published_at ? (
                            <span>Published: {new Date(item.published_at).toLocaleDateString()}</span>
                          ) : null}
                        </div>
                      </div>
                      <div className="flex flex-col gap-2 shrink-0">
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={item.status === 'accepted' || updateInboxItemMutation.isLoading}
                          onClick={() => updateInboxItemMutation.mutate({ itemId: item.id, data: { status: 'accepted' } })}
                        >
                          <ThumbsUp className="w-4 h-4 mr-1" />
                          Accept
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={item.status === 'rejected' || updateInboxItemMutation.isLoading}
                          onClick={() => {
                            const feedback = window.prompt('Reject reason (optional):') || undefined;
                            updateInboxItemMutation.mutate({ itemId: item.id, data: { status: 'rejected', feedback } });
                          }}
                        >
                          <ThumbsDown className="w-4 h-4 mr-1" />
                          Reject
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={upsertMonitorProfileMutation.isLoading}
                          onClick={() => {
                            const suggested = (item.title || '').split(/[^a-zA-Z0-9_-]+/).find((t) => t && t.length >= 4) || '';
                            const token = (window.prompt('Mute token (prevents future suggestions):', suggested) || '').trim().toLowerCase();
                            if (!token) return;
                            upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_tokens: [token], merge_lists: true });
                          }}
                        >
                          Mute token
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={upsertMonitorProfileMutation.isLoading}
                          onClick={() => {
                            const phrase = (window.prompt('Mute phrase (substring match):', item.title || '') || '').trim();
                            if (!phrase) return;
                            upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_patterns: [phrase], merge_lists: true });
                          }}
                        >
                          Mute phrase
                        </Button>
                        {(item.metadata as any)?.query ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={upsertMonitorProfileMutation.isLoading}
                            onClick={() => {
                              const q = String((item.metadata as any).query || '').trim();
                              if (!q) return;
                              upsertMonitorProfileMutation.mutate({ customer: item.customer || undefined, muted_patterns: [q], merge_lists: true });
                            }}
                          >
                            Mute query
                          </Button>
                        ) : null}
                        {item.item_type === 'arxiv' ? (
                          <>
                            {Array.isArray((item.metadata as any)?.repos) && (item.metadata as any).repos.length > 0 ? (
                              <>
                                <div className="text-xs text-gray-500 mt-1">Repos</div>
                                {((item.metadata as any).repos as any[]).slice(0, 2).map((r: any, idx: number) => (
                                  <Button
                                    key={idx}
                                    size="sm"
                                    variant="secondary"
                                    disabled={ingestRepoMutation.isLoading || String(r?.provider) !== 'github'}
                                    title={String(r?.provider) === 'github' ? 'Ingest this repo' : 'GitLab ingestion requires a token (use Documents → Repos)'}
                                    onClick={() => ingestRepoMutation.mutate({ provider: 'github', repo: String(r.repo) })}
                                  >
                                    Ingest {String(r?.provider || 'repo')}
                                  </Button>
                                ))}
                              </>
                            ) : (
                              <Button
                                size="sm"
                                variant="secondary"
                                disabled={extractReposMutation.isLoading}
                                onClick={() => extractReposMutation.mutate(item.id)}
                              >
                                Find repos
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={createFromChainMutation.isLoading}
                              onClick={() => runPaperRepoCodeAgent(item)}
                            >
                              Code Agent on Repo
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={createFromChainMutation.isLoading}
                              onClick={() => {
                                const persistedEp = String((item.metadata as any)?.paper_algo_entrypoint || '').trim();
                                const ep = (paperAlgoEntrypoint[item.id] ?? persistedEp ?? 'demo.py') || 'demo.py';
                                runPaperAlgorithmProject(item, paperAlgoRunDemo[item.id] ?? paperAlgoDefaultRunDemoCheck, ep);
                              }}
                            >
                              <span className="inline-flex items-center gap-2">
                                <span>Implement Algorithm</span>
                                <span className="inline-flex items-center gap-1" title={unsafeExecBadge.title}>
                                  <span className={`inline-block w-2 h-2 rounded-full ${unsafeExecBadge.color}`} />
                                  <span className="text-[10px] text-gray-600">{unsafeExecBadge.label}</span>
                                </span>
                              </span>
                            </Button>
                            <label
                              className="flex items-center gap-1 text-xs text-gray-600 select-none"
                              title={
                                unsafeExecBadge.status === 'ready'
                                  ? 'Run a sandboxed demo.py check after generating the project'
                                  : unsafeExecBadge.title
                              }
                            >
                              {(() => {
                                const persisted = (item.metadata as any)?.paper_algo_run_demo_check;
                                const checked =
                                  typeof persisted === 'boolean'
                                    ? persisted
                                    : paperAlgoRunDemo[item.id] ?? paperAlgoDefaultRunDemoCheck;
                                return (
                              <input
                                type="checkbox"
                                className="h-3 w-3"
                                checked={checked}
                                disabled={unsafeExecBadge.status !== 'ready' || updateInboxItemMutation.isLoading}
                                onChange={(e) => {
                                  const v = e.target.checked;
                                  setPaperAlgoRunDemo((prev) => ({ ...prev, [item.id]: v }));
                                  updateInboxItemMutation.mutate({
                                    itemId: item.id,
                                    data: { metadata_patch: { paper_algo_run_demo_check: v } },
                                  });
                                }}
                              />
                                );
                              })()}
                              <span>Run demo check</span>
                            </label>
                            <input
                              className={`border rounded px-2 py-1 text-xs w-36 ${
                                paperAlgoEntrypointError[item.id] ? 'border-red-400' : 'border-gray-200'
                              }`}
                              placeholder="demo.py"
                              value={
                                paperAlgoEntrypoint[item.id] ??
                                (String((item.metadata as any)?.paper_algo_entrypoint || '').trim() || 'demo.py')
                              }
                              onChange={(e) => {
                                const raw = e.target.value;
                                setPaperAlgoEntrypoint((prev) => ({ ...prev, [item.id]: raw }));
                                const check = normalizeEntrypoint(raw);
                                setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: check.ok ? '' : String(check.error || 'Invalid') }));
                              }}
                              onBlur={async () => {
                                const raw =
                                  paperAlgoEntrypoint[item.id] ??
                                  (String((item.metadata as any)?.paper_algo_entrypoint || '').trim() || 'demo.py');
                                const check = normalizeEntrypoint(raw);
                                setPaperAlgoEntrypoint((prev) => ({ ...prev, [item.id]: check.value }));
                                setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: check.ok ? '' : String(check.error || 'Invalid') }));
                                if (!check.ok) {
                                  toast.error(`Invalid entrypoint: ${check.error || 'Invalid'}`);
                                  return;
                                }
                                setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: true }));
                                try {
                                  await apiClient.updateResearchInboxItem(item.id, {
                                    metadata_patch: { paper_algo_entrypoint: check.value },
                                  } as any);
                                  queryClient.invalidateQueries(['research-inbox']);
                                  setPaperAlgoEntrypointSavedAt((prev) => ({ ...prev, [item.id]: new Date().toISOString() }));
                                } catch (e: any) {
                                  toast.error(e?.response?.data?.detail || e?.message || 'Failed to save entrypoint');
                                } finally {
                                  setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: false }));
                                }
                              }}
                              title={
                                paperAlgoEntrypointError[item.id]
                                  ? `Entrypoint invalid: ${paperAlgoEntrypointError[item.id]}`
                                  : 'Demo entrypoint path (persisted per paper)'
                              }
                            />
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={paperAlgoEntrypointSaving[item.id] || updateInboxItemMutation.isLoading}
                              title="Reset entrypoint override to default (demo.py)"
                              onClick={async () => {
                                setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: true }));
                                try {
                                  await apiClient.updateResearchInboxItem(item.id, {
                                    metadata_patch: { paper_algo_entrypoint: null },
                                  } as any);
                                  setPaperAlgoEntrypoint((prev) => {
                                    const next = { ...prev };
                                    delete next[item.id];
                                    return next;
                                  });
                                  setPaperAlgoEntrypointError((prev) => ({ ...prev, [item.id]: '' }));
                                  setPaperAlgoEntrypointSavedAt((prev) => ({ ...prev, [item.id]: new Date().toISOString() }));
                                  queryClient.invalidateQueries(['research-inbox']);
                                  toast.success('Entrypoint reset to default');
                                } catch (e: any) {
                                  toast.error(e?.response?.data?.detail || e?.message || 'Failed to reset entrypoint');
                                } finally {
                                  setPaperAlgoEntrypointSaving((prev) => ({ ...prev, [item.id]: false }));
                                }
                              }}
                            >
                              Reset
                            </Button>
                            <span className="text-[10px] text-gray-500 min-w-[42px]">
                              {paperAlgoEntrypointSaving[item.id]
                                ? 'saving…'
                                : paperAlgoEntrypointSavedAt[item.id]
                                  ? 'saved'
                                  : ''}
                            </span>
                          </>
                        ) : null}
                        {item.item_type === 'document' ? (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => navigate(`/search?q=${encodeURIComponent(item.title || item.item_key)}`)}
                          >
                            <Search className="w-4 h-4 mr-1" />
                            Search
                          </Button>
                        ) : null}
                        {item.url ? (
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1 justify-center"
                          >
                            <Link2 className="w-4 h-4" />
                            Open
                          </a>
                        ) : null}
                      </div>
                    </div>
                    {item.feedback ? (
                      <div className="mt-3 text-xs text-gray-600 bg-gray-50 border border-gray-100 rounded p-2">
                        Feedback: {item.feedback}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modals */}
      {showCreateModal && <CreateJobModal />}
      {createFromTemplate && <CreateFromTemplateModal template={createFromTemplate} />}
      {showCustomerResearchModal && <CustomerResearchModal />}
      {showInboxMonitorModal && <InboxMonitorModal />}
      {showMonitorProfilesModal && <MonitorProfilesModal />}
      {startFromChain && <StartChainModal chain={startFromChain} />}

      {/* Chain Status Modal */}
      {selectedChainStatus && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-100 text-purple-600">
                  <GitBranch className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Chain Status</h2>
                  <p className="text-sm text-gray-500">
                    Step {selectedChainStatus.current_step + 1} of {selectedChainStatus.total_steps} •{' '}
                    <span className="capitalize">{selectedChainStatus.status}</span>
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setSelectedChainStatus(null)}>
                <XCircle className="w-5 h-5" />
              </Button>
            </div>

            {chainExperimentStopInfo ? (
              <div className="px-4 py-3 border-b border-gray-200 bg-amber-50">
                <div className="flex items-start justify-between gap-3">
                  <div className="text-sm text-amber-900">
                    <div className="font-medium">Experiment loop stopped early</div>
                    <div className="text-xs text-amber-800 mt-1">
                      Reason: <span className="font-mono">{chainExperimentStopInfo.reason || 'unknown'}</span>
                      {chainExperimentStopInfo.atRunId ? (
                        <>
                          {' '}
                          • Run: <span className="font-mono">{chainExperimentStopInfo.atRunId}</span>
                        </>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {chainExperimentStopInfo.noteId ? (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          setSelectedChainStatus(null);
                          navigate(`/research-notes?note=${encodeURIComponent(chainExperimentStopInfo.noteId || '')}`);
                        }}
                      >
                        Open note
                      </Button>
                    ) : null}
                    {chainExperimentStopInfo.stoppedByJobId ? (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => {
                          setSelectedChainStatus(null);
                          navigate(`/autonomous-agents?job=${encodeURIComponent(chainExperimentStopInfo.stoppedByJobId || '')}`);
                        }}
                      >
                        Open job
                      </Button>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}

            {/* Progress bar */}
            <div className="px-4 py-3 border-b border-gray-200">
              <div className="flex items-center justify-between text-sm text-gray-500 mb-1">
                <span>Overall Progress</span>
                <span>{selectedChainStatus.overall_progress}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    selectedChainStatus.status === 'completed'
                      ? 'bg-green-500'
                      : selectedChainStatus.status === 'failed'
                      ? 'bg-red-500'
                      : 'bg-purple-500'
                  }`}
                  style={{ width: `${selectedChainStatus.overall_progress}%` }}
                />
              </div>
            </div>

            {/* Jobs list */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="space-y-3">
                {selectedChainStatus.jobs.map((job, index) => {
                  const statusConfig = STATUS_CONFIG[job.status as AgentJobStatus] || STATUS_CONFIG.pending;
                  const StatusIcon = statusConfig.icon;
                  const isCurrentStep = index === selectedChainStatus.current_step;

                  return (
                    <div
                      key={job.id}
                      className={`border rounded-lg p-3 ${
                        isCurrentStep ? 'border-purple-500 bg-purple-50' : 'border-gray-200'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-gray-200 text-xs font-medium">
                            {index + 1}
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-900">{job.name}</h4>
                            <p className="text-xs text-gray-500">{job.job_type}</p>
                          </div>
                        </div>
                        <div
                          className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${statusConfig.bgColor} ${statusConfig.color}`}
                        >
                          <StatusIcon className={`w-3 h-3 ${job.status === 'running' ? 'animate-spin' : ''}`} />
                          <span className="capitalize">{job.status}</span>
                        </div>
                      </div>

                      {/* Mini progress bar */}
                      <div className="mt-2">
                        <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              job.status === 'completed'
                                ? 'bg-green-500'
                                : job.status === 'failed'
                                ? 'bg-red-500'
                                : 'bg-purple-500'
                            }`}
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                      </div>

                      {job.error && (
                        <p className="mt-2 text-xs text-red-600 bg-red-50 rounded p-2">{job.error}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="p-4 border-t border-gray-200 flex justify-end">
              <Button variant="secondary" onClick={() => setSelectedChainStatus(null)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      {showExportModal && exportingJob && (
        <ExportModal
          job={exportingJob}
          onClose={() => {
            setShowExportModal(false);
            setExportingJob(null);
          }}
        />
      )}
    </div>
  );
};

// Export Modal Component
const ExportModal: React.FC<{ job: AgentJob; onClose: () => void }> = ({ job, onClose }) => {
  const [format, setFormat] = useState<'docx' | 'pdf' | 'pptx'>('docx');
  const [style, setStyle] = useState<'professional' | 'technical' | 'casual'>('professional');
  const [includeLog, setIncludeLog] = useState(false);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [enhance, setEnhance] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      await apiClient.downloadJobExport(job.id, job.name, format, {
        style,
        includeLog,
        includeMetadata,
        enhance,
      });
      toast.success(`Exported as ${format.toUpperCase()}${enhance ? ' (AI-enhanced)' : ''}`);
      onClose();
    } catch (error: any) {
      console.error('Export failed:', error);
      toast.error(error.message || 'Export failed');
    } finally {
      setIsExporting(false);
    }
  };

  const formatOptions = [
    { value: 'docx', label: 'Word Document', icon: FileText, description: 'DOCX format, editable' },
    { value: 'pdf', label: 'PDF Document', icon: FileText, description: 'PDF format, universal' },
    { value: 'pptx', label: 'Presentation', icon: FileDown, description: 'PowerPoint slides' },
  ];

  const styleOptions = [
    { value: 'professional', label: 'Professional', description: 'Clean corporate look' },
    { value: 'technical', label: 'Technical', description: 'Developer-focused' },
    { value: 'casual', label: 'Casual', description: 'Friendly and approachable' },
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-blue-100 text-blue-600">
              <Download className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Export Results</h2>
              <p className="text-sm text-gray-500">Export "{job.name}" results</p>
            </div>
          </div>

          <div className="space-y-4">
            {/* Format selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Format</label>
              <div className="grid grid-cols-3 gap-2">
                {formatOptions.map((option) => {
                  const Icon = option.icon;
                  return (
                    <button
                      key={option.value}
                      className={`p-3 border rounded-lg text-center transition-colors ${
                        format === option.value
                          ? 'border-primary-500 bg-primary-50 text-primary-700'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setFormat(option.value as any)}
                    >
                      <Icon className="w-5 h-5 mx-auto mb-1" />
                      <span className="text-xs font-medium block">{option.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Style selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Style</label>
              <select
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                value={style}
                onChange={(e) => setStyle(e.target.value as any)}
              >
                {styleOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label} - {option.description}
                  </option>
                ))}
              </select>
            </div>

            {/* Options */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">Options</label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={includeMetadata}
                  onChange={(e) => setIncludeMetadata(e.target.checked)}
                  className="rounded"
                />
                <span>Include job metadata and statistics</span>
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={includeLog}
                  onChange={(e) => setIncludeLog(e.target.checked)}
                  className="rounded"
                />
                <span>Include execution log</span>
              </label>
            </div>

            {/* AI Enhancement */}
            <div className="border border-purple-200 rounded-lg p-3 bg-purple-50">
              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={enhance}
                  onChange={(e) => setEnhance(e.target.checked)}
                  className="rounded mt-0.5 border-purple-300"
                />
                <div>
                  <span className="text-sm font-medium text-purple-900 flex items-center gap-1">
                    <Zap className="w-4 h-4" />
                    AI-Enhanced Report
                  </span>
                  <p className="text-xs text-purple-700 mt-0.5">
                    Uses AI to generate an executive summary, key insights, and recommendations.
                    Takes longer to generate.
                  </p>
                </div>
              </label>
            </div>

            {/* Job summary */}
            <div className="bg-gray-50 rounded-lg p-3">
              <p className="text-xs text-gray-500 mb-1">Export preview</p>
              <div className="text-sm space-y-1">
                <p><span className="text-gray-500">Status:</span> {job.status}</p>
                <p><span className="text-gray-500">Progress:</span> {job.progress}%</p>
                {job.results?.findings_count !== undefined && (
                  <p><span className="text-gray-500">Findings:</span> {job.results.findings_count}</p>
                )}
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
            <Button variant="secondary" onClick={onClose} disabled={isExporting}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Export {format.toUpperCase()}
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AutonomousAgentsPage;
