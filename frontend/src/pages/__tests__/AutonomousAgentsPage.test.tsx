import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import type { RenderResult } from '@testing-library/react';
import AutonomousAgentsPage from '../AutonomousAgentsPage';
import {
  buildBugTriageSwarmQuickStartPayload,
  buildDomainResearchQuickStartPayload,
  buildRepoBugTriageQuickStartPayload,
  DEFAULT_VALIDATION_POLICY,
  findUnsafeQuickStartCommands,
  parseQuickStartCommands,
} from '../autonomousAgentQuickStarts';
import { AuthProvider } from '../../contexts/AuthContext';
import type { AgentJob } from '../../types';

const mockUseAuth = jest.fn();

jest.mock('../../contexts/AuthContext', () => ({
  ...jest.requireActual('../../contexts/AuthContext'),
  useAuth: () => mockUseAuth(),
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

const mockSocket = () =>
  ({
    onmessage: null,
    onerror: null,
    close: jest.fn(),
  } as any);

jest.mock('../../services/api', () => ({
  apiClient: {
    listAgentJobs: jest.fn(),
    getAgentJob: jest.fn(),
    getAgentJobStats: jest.fn(),
    getAgentJobSwarmAnalytics: jest.fn(),
    getAgentJobSwarmOutcomeAnalytics: jest.fn(),
    getAgentCheckpointQueue: jest.fn(),
    getAgentDecisionTrace: jest.fn(),
    getAgentDecisionTraceAnalytics: jest.fn(),
    downloadAgentDecisionTraceExport: jest.fn(),
    actionAgentDecisionTraceEvent: jest.fn(),
    listAgentDecisionTraceViews: jest.fn(),
    createAgentDecisionTraceView: jest.fn(),
    updateAgentDecisionTraceView: jest.fn(),
    deleteAgentDecisionTraceView: jest.fn(),
    actionAgentCheckpointQueueFollowUp: jest.fn(),
    bulkActionAgentCheckpointQueueFollowUp: jest.fn(),
    bulkRelaunchInboxFollowUp: jest.fn(),
    listResearchInboxItems: jest.fn(),
    getResearchMonitorAnalytics: jest.fn(),
    getResearchMonitorPolicyEvaluation: jest.fn(),
    getResearchMonitorCustomerRebalanceEvaluation: jest.fn(),
    updateResearchMonitorPolicy: jest.fn(),
    rollbackResearchMonitorPolicy: jest.fn(),
    simulateResearchMonitorPolicy: jest.fn(),
    updateResearchMonitorCustomerBudget: jest.fn(),
    previewResearchMonitorCustomerRebalance: jest.fn(),
    applyResearchMonitorCustomerRebalance: jest.fn(),
    listChainDefinitions: jest.fn(),
    saveAgentJobAsChain: jest.fn(),
    listAgentJobTemplates: jest.fn(),
    getDocumentSources: jest.fn(),
    getResearchInboxStats: jest.fn(),
    getMyPreferences: jest.fn(),
    updateMyPreferences: jest.fn(),
    getUnsafeExecAvailability: jest.fn(),
    createAgentJobProgressWebSocket: jest.fn(),
    listArxivImports: jest.fn(),
    listReadingLists: jest.fn(),
    getAgentJobLog: jest.fn(),
    getAgentJobStepEvents: jest.fn(),
    getJobMemories: jest.fn(),
    performAgentJobAction: jest.fn(),
    promoteDomainResearchAgentJob: jest.fn(),
    getAgentJobRelaunchLineage: jest.fn(),
    quickStartBugTriageSwarmJob: jest.fn(),
    quickStartBuildBreakSwarmJob: jest.fn(),
    quickStartFrontendRegressionSwarmJob: jest.fn(),
    quickStartDomainResearchJob: jest.fn(),
    quickStartRepoBugTriageJob: jest.fn(),
    listCodingSwarmProfiles: jest.fn(),
    createCodingSwarmProfile: jest.fn(),
    updateCodingSwarmProfile: jest.fn(),
    deleteCodingSwarmProfile: jest.fn(),
    listCollaborationUsers: jest.fn(),
    listDomainResearchProfiles: jest.fn(),
    createDomainResearchProfile: jest.fn(),
    updateDomainResearchProfile: jest.fn(),
    performDomainResearchProfileAction: jest.fn(),
    actOnDomainResearchOpportunity: jest.fn(),
    listResearchPortfolios: jest.fn(),
    createResearchPortfolio: jest.fn(),
    updateResearchPortfolio: jest.fn(),
    performResearchPortfolioAction: jest.fn(),
    actOnResearchPortfolioOpportunity: jest.fn(),
    listScientificSandboxProfiles: jest.fn(),
    createScientificSandboxProfile: jest.fn(),
    updateScientificSandboxProfile: jest.fn(),
    deleteScientificSandboxProfile: jest.fn(),
    createSynthesisJob: jest.fn(),
    saveSynthesisJobAsResearchNote: jest.fn(),
    listResearchNotes: jest.fn(),
    getResearchNote: jest.fn(),
    listCodingBacklogItems: jest.fn(),
    createCodingBacklogItem: jest.fn(),
    performCodingBacklogAction: jest.fn(),
    downloadCodePatchProposal: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;

const makeJob = (overrides: Partial<AgentJob> = {}): AgentJob => ({
  id: 'job-1',
  name: 'Autonomous Runtime Job',
  goal: 'Inspect repo health',
  job_type: 'research',
  user_id: 'user-1',
  status: 'running',
  progress: 42,
  current_phase: 'observe',
  phase_details: 'Inspecting execution graph',
  iteration: 4,
  max_iterations: 20,
  max_tool_calls: 40,
  max_llm_calls: 20,
  max_runtime_minutes: 15,
  tool_calls_used: 8,
  llm_calls_used: 4,
  tokens_used: 1200,
  error_count: 0,
  chain_depth: 0,
  chain_triggered: false,
    experiment_run: {
      source_id: 'repo-123',
      source_name: 'Knowledge Repo',
      ok: true,
      commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
      verification_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
      bootstrap_commands: ['npm --prefix frontend install'],
      fallback_commands: ['python3 -m pytest -q backend/tests'],
      phases: ['primary', 'bootstrap', 'retry_primary'],
      final_phase: 'retry_primary',
      failed_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
      bootstrap_attempted: true,
      bootstrap_ok: true,
      bootstrap_used: true,
      fallback_attempted: false,
      fallback_ok: null,
      fallback_used: false,
      inferred_project_profile: {
        detected_stack: ['node', 'python'],
      },
    },
    results: {
      execution_strategy: {
        execution_graph: {
          graph_health: {
          status: 'warning',
          severity_score: 18,
          blocked_ratio: 0.25,
          reasons: ['verification debt building'],
        },
        dag_stats: {
          total_nodes: 9,
          total_edges: 8,
          critical_path_length: 4,
          blocked_nodes: 2,
          root_nodes: 1,
          leaf_nodes: 3,
          orphan_nodes: 0,
          has_cycle: false,
        },
        verification_actions: [{ id: 'v1' }, { id: 'v2' }],
        summarization_actions: [{ id: 's1' }],
        recommended_actions: ['Re-ground on failing verification output'],
      },
      scope_observability: {
        resolved_scope_id: 'repo-123',
        scope_source: 'config.source_id',
        events: [
          {
            type: 'resolved_scope',
            source_id: 'repo-123',
            scope_source: 'config.source_id',
          },
          {
            type: 'scope_guard_blocked',
            source_id: 'repo-999',
            scope_source: 'config.source_id',
          },
        ],
      },
      operator_interventions: [
        {
          action: 'pause',
          actor_user_id: 'user-1',
          at: '2026-03-10T00:30:00Z',
          note: 'Paused for manual inspection',
          job_status_before: 'running',
          job_status_after: 'paused',
          outcome_status: 'superseded',
        },
        {
          action: 'restart',
          actor_user_id: 'user-1',
          at: '2026-03-10T01:00:00Z',
          note: 'Retry after fallback failure',
          job_status_before: 'failed',
          job_status_after: 'pending',
          outcome_status: 'applied',
          outcome_reason: 'Job resumed after intervention',
          metadata: {
            tool: 'python -m pytest -q backend/tests',
            plan_step_id: 'verify_1',
          },
        },
      ],
      },
    },
    output_artifacts: [
      {
        type: 'document',
        id: 'doc-1',
        title: 'Run report',
      },
    ],
    created_at: '2026-03-10T00:00:00Z',
    ...overrides,
  });

const defaultDocumentSources = [
  { id: 'repo-source-1', name: 'Knowledge Repo', source_type: 'github' },
  { id: 'repo-source-2', name: 'Frontend Repo', source_type: 'gitlab' },
];

const renderedViews: RenderResult[] = [];
let consoleErrorSpy: jest.SpyInstance | null = null;
let consoleWarnSpy: jest.SpyInstance | null = null;

const flushMockPromises = async () => {
  await act(async () => {
    for (let pass = 0; pass < 3; pass += 1) {
      await Promise.resolve();
      const pendingCalls = Object.values(apiClient as Record<string, any>)
        .filter((candidate) => jest.isMockFunction(candidate))
        .flatMap((mockFn: any) => mockFn.mock.results || [])
        .map((result: any) => result?.value)
        .filter((value: any) => value && typeof value.then === 'function');
      if (pendingCalls.length === 0) {
        continue;
      }
      await Promise.allSettled(pendingCalls);
    }
  });
};

const renderWithProviders = async (
  initialEntry: string = '/autonomous-agents',
  options?: { documentSources?: typeof defaultDocumentSources }
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });
  if (options?.documentSources) {
    queryClient.setQueryData(['document-sources', 'all'], options.documentSources);
  }

  const view = render(
    <MemoryRouter
      initialEntries={[initialEntry]}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <AutonomousAgentsPage />
        </AuthProvider>
      </QueryClientProvider>
    </MemoryRouter>
  );
  renderedViews.push(view);
  await flushMockPromises();
  return view;
};

const expectJobHeading = async (name: string) => {
  await waitFor(() => {
    const titles = screen.getAllByRole('heading', { level: 3 }).map((node) => node.textContent || '');
    expect(titles).toContain(name);
  });
};

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe('AutonomousAgentsPage', () => {
  beforeEach(() => {
    const shouldIgnoreConsoleMessage = (args: unknown[]) =>
      String(args[0] || '').includes('Warning: An update to JobDetailPanel inside a test was not wrapped in act');
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      if (shouldIgnoreConsoleMessage(args)) {
        return;
      }
      // eslint-disable-next-line no-console
      jest.requireActual('console').error(...args);
    });
    consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation((...args: unknown[]) => {
      if (shouldIgnoreConsoleMessage(args)) {
        return;
      }
      // eslint-disable-next-line no-console
      jest.requireActual('console').warn(...args);
    });
    mockUseAuth.mockReturnValue({
      user: { id: 'user-1', username: 'testuser', role: 'user' },
      loading: false,
    });
    const job = makeJob();
    const cleanJob = makeJob({
      id: 'job-2',
      name: 'Clean Scope Job',
      experiment_run: null,
      experiment_runs: [],
      results: {
        execution_strategy: {
          execution_graph: {
            graph_health: {
              status: 'ok',
              severity_score: 0,
              blocked_ratio: 0,
              reasons: [],
            },
            dag_stats: {
              total_nodes: 4,
              total_edges: 3,
              critical_path_length: 2,
              blocked_nodes: 0,
              root_nodes: 1,
              leaf_nodes: 1,
              orphan_nodes: 0,
              has_cycle: false,
            },
            verification_actions: [],
            summarization_actions: [],
            recommended_actions: [],
          },
          scope_observability: {
            resolved_scope_id: 'repo-456',
            scope_source: 'config.source_id',
            events: [
              {
                type: 'resolved_scope',
                source_id: 'repo-456',
                scope_source: 'config.source_id',
              },
            ],
          },
        },
      },
    });
    const fallbackJob = makeJob({
      id: 'job-3',
      name: 'Fallback Recovery Job',
      experiment_run: {
        source_id: 'repo-789',
        source_name: 'Fallback Repo',
        ok: false,
        commands: ['python3 -m pytest -q backend/tests'],
        verification_commands: ['npm --prefix frontend test'],
        bootstrap_commands: ['npm --prefix frontend install'],
        fallback_commands: ['python3 -m pytest -q backend/tests'],
        phases: ['primary', 'bootstrap', 'fallback'],
        final_phase: 'fallback',
        failed_commands: ['npm --prefix frontend test'],
        bootstrap_attempted: true,
        bootstrap_ok: false,
        bootstrap_used: true,
        fallback_attempted: true,
        fallback_ok: true,
        fallback_used: true,
        inferred_project_profile: {
          detected_stack: ['python'],
        },
      },
      results: {
        execution_strategy: {
          execution_graph: {
            graph_health: {
              status: 'critical',
              severity_score: 30,
              blocked_ratio: 0.5,
              reasons: ['verification fallback engaged'],
            },
            dag_stats: {
              total_nodes: 6,
              total_edges: 5,
              critical_path_length: 3,
              blocked_nodes: 2,
              root_nodes: 1,
              leaf_nodes: 2,
              orphan_nodes: 0,
              has_cycle: false,
            },
            verification_actions: [{ id: 'v3' }],
            summarization_actions: [],
            recommended_actions: [],
          },
          scope_observability: {
            resolved_scope_id: 'repo-789',
            scope_source: 'config.source_id',
            events: [
              {
                type: 'resolved_scope',
                source_id: 'repo-789',
                scope_source: 'config.source_id',
              },
            ],
          },
        },
      },
    });
    const unresolvedRecoveryJob = makeJob({
      id: 'job-4',
      name: 'Unresolved Recovery Job',
      experiment_run: {
        source_id: 'repo-999',
        source_name: 'Broken Repo',
        ok: false,
        commands: ['python3 -m pytest -q backend/tests'],
        verification_commands: ['npm --prefix frontend test'],
        bootstrap_commands: ['npm --prefix frontend install'],
        fallback_commands: ['python3 -m pytest -q backend/tests'],
        phases: ['primary', 'bootstrap', 'fallback'],
        final_phase: 'fallback',
        failed_commands: ['npm --prefix frontend test'],
        bootstrap_attempted: true,
        bootstrap_ok: false,
        bootstrap_used: true,
        fallback_attempted: true,
        fallback_ok: false,
        fallback_used: true,
        inferred_project_profile: {
          detected_stack: ['python'],
        },
      },
      results: {
        execution_strategy: {
          execution_graph: {
            graph_health: {
              status: 'critical',
              severity_score: 42,
              blocked_ratio: 0.75,
              reasons: ['fallback verification still failing'],
            },
            dag_stats: {
              total_nodes: 7,
              total_edges: 6,
              critical_path_length: 4,
              blocked_nodes: 3,
              root_nodes: 1,
              leaf_nodes: 2,
              orphan_nodes: 0,
              has_cycle: false,
            },
            verification_actions: [{ id: 'v4' }],
            summarization_actions: [],
            recommended_actions: ['Inspect failing fallback output'],
          },
          scope_observability: {
            resolved_scope_id: 'repo-999',
            scope_source: 'config.source_id',
            events: [
              {
                type: 'resolved_scope',
                source_id: 'repo-999',
                scope_source: 'config.source_id',
              },
            ],
          },
        },
      },
    });
    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [job, cleanJob, fallbackJob, unresolvedRecoveryJob],
      total: 4,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValue(job);
    apiClient.createAgentJobProgressWebSocket.mockReturnValue(mockSocket());
    apiClient.getAgentJobStats.mockResolvedValue({
      total_jobs: 1,
      running_jobs: 1,
      pending_jobs: 0,
      completed_jobs: 0,
      failed_jobs: 0,
      total_iterations: 4,
      total_tool_calls: 8,
      total_llm_calls: 4,
    });
    apiClient.getAgentJobSwarmAnalytics.mockResolvedValue({
      preset_rows: [],
      totals: {
        total_runs: 0,
        repair_handoff_runs: 0,
        review_needed_runs: 0,
        avg_confidence: null,
      },
      filters: {},
    });
    apiClient.getAgentJobSwarmOutcomeAnalytics.mockResolvedValue({
      preset_rows: [],
      cases: [],
      totals: {
        total_swarm_roots: 0,
        repair_handoff_runs: 0,
        verified_fix_runs: 0,
        backlog_routed_runs: 0,
        avg_handoff_minutes: null,
      },
      filters: {},
    });
    apiClient.listChainDefinitions.mockResolvedValue({ chains: [], total: 0 });
    apiClient.listAgentJobTemplates.mockResolvedValue({
      templates: [
        {
          id: 'tpl-claude',
          name: 'claude_code_backend',
          display_name: 'Code Agent: Claude-Code Backend Loop',
          job_type: 'analysis',
          default_max_iterations: 1,
          default_max_tool_calls: 0,
          default_max_llm_calls: 2,
          default_max_runtime_minutes: 20,
          is_system: true,
          is_active: true,
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
          category: 'code',
          default_config: {
            deterministic_runner: 'code_patch_proposer',
          },
        },
      ],
      total: 1,
    });
    apiClient.getDocumentSources.mockResolvedValue(defaultDocumentSources);
    apiClient.listCodingSwarmProfiles.mockResolvedValue({
      items: [],
      total: 0,
      limit: 200,
      offset: 0,
    });
    apiClient.listCodingBacklogItems.mockResolvedValue({
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
    });
    apiClient.listDomainResearchProfiles.mockResolvedValue({
      items: [],
      total: 0,
    });
    apiClient.listResearchPortfolios.mockResolvedValue({
      items: [],
      total: 0,
    });
    apiClient.listScientificSandboxProfiles.mockResolvedValue({
      items: [
        {
          id: 'scientific-compiler-sandbox',
          name: 'Compiler Validation Sandbox',
          description: 'Compiler sandbox',
          track_type: 'compiler',
          backend: 'docker',
          docker_image: 'ghcr.io/knowledgedb/compiler-research:latest',
          timeout_seconds: 1200,
          resource_caps: { memory_mb: 4096, cpus: 2, pids_limit: 256 },
          allowed_benchmark_families: ['compiler_regression'],
          allowed_perf_collectors: ['benchmark_output', 'compile_time', 'artifact_diff'],
          required_capabilities: ['repo_reconstruction'],
          toolchains: ['clang', 'pytest'],
          budget_limit_default: 35,
          enabled: true,
          system_managed: true,
          is_default: true,
        },
        {
          id: 'scientific-microarchitecture-sandbox',
          name: 'Microarchitecture Validation Sandbox',
          description: 'Microarchitecture sandbox',
          track_type: 'microarchitecture',
          backend: 'docker',
          docker_image: 'ghcr.io/knowledgedb/microarch-research:latest',
          timeout_seconds: 1200,
          resource_caps: { memory_mb: 4096, cpus: 2, pids_limit: 256 },
          allowed_benchmark_families: ['perf_counter_regression'],
          allowed_perf_collectors: ['perf_stat', 'benchmark_output'],
          required_capabilities: ['repo_reconstruction', 'perf_counters'],
          toolchains: ['python', 'perf'],
          budget_limit_default: 40,
          enabled: true,
          system_managed: true,
          is_default: true,
        },
        {
          id: 'scientific-generic-sandbox',
          name: 'Scientific Validation Sandbox',
          description: 'Generic sandbox',
          track_type: 'generic',
          backend: 'docker',
          docker_image: 'python:3.11-slim',
          timeout_seconds: 900,
          resource_caps: { memory_mb: 2048, cpus: 1.5, pids_limit: 192 },
          allowed_benchmark_families: ['generic_validation'],
          allowed_perf_collectors: ['benchmark_output'],
          required_capabilities: ['repo_reconstruction'],
          toolchains: ['python', 'pytest'],
          budget_limit_default: 25,
          enabled: true,
          system_managed: true,
          is_default: true,
        },
      ],
      total: 3,
    });
    apiClient.createScientificSandboxProfile.mockImplementation(async (payload: any) => ({
      ...payload,
      system_managed: false,
      created_by_user_id: 'user-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:00:00Z',
    }));
    apiClient.updateScientificSandboxProfile.mockImplementation(async (_profileId: string, payload: any) => ({
      id: 'custom-generic-sandbox',
      name: payload.name ?? 'Custom Generic Sandbox',
      description: payload.description ?? null,
      track_type: payload.track_type ?? 'generic',
      backend: payload.backend ?? 'docker',
      docker_image: payload.docker_image ?? 'python:3.11-slim',
      timeout_seconds: payload.timeout_seconds ?? 900,
      resource_caps: payload.resource_caps ?? { memory_mb: 2048, cpus: 1.5, pids_limit: 192 },
      allowed_benchmark_families: payload.allowed_benchmark_families ?? ['generic_validation'],
      allowed_perf_collectors: payload.allowed_perf_collectors ?? ['benchmark_output'],
      required_capabilities: payload.required_capabilities ?? ['repo_reconstruction'],
      toolchains: payload.toolchains ?? ['python', 'pytest'],
      budget_limit_default: payload.budget_limit_default ?? 25,
      enabled: payload.enabled ?? true,
      system_managed: false,
      is_default: payload.is_default ?? false,
      created_by_user_id: 'user-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.deleteScientificSandboxProfile.mockResolvedValue(undefined);
    apiClient.createDomainResearchProfile.mockImplementation(async (payload: any) => ({
      id: 'profile-1',
      user_id: 'user-1',
      title: payload.title,
      domain: payload.domain,
      objective: payload.objective,
      customer_context: payload.customer_context ?? null,
      status: payload.start_immediately === false ? 'draft' : 'running',
      source_scope: payload.source_scope ?? 'kb_plus_arxiv',
      track_type: payload.track_type ?? 'generic',
      automation_profile: payload.automation_profile ?? 'balanced',
      automation_policy: payload.automation_policy ?? {},
      effective_policy: payload.automation_policy ?? {},
      monitor_queries: payload.monitor_queries ?? [],
      repo_source_ids: payload.repo_source_ids ?? [],
      benchmark_queries: payload.benchmark_queries ?? [],
      report_format: payload.report_format ?? 'brief_and_report',
      interval_minutes: payload.interval_minutes ?? 1440,
      persist_artifacts: payload.persist_artifacts ?? true,
      auto_launch_follow_up: payload.auto_launch_follow_up ?? true,
      auto_create_experiment_plans: payload.auto_create_experiment_plans ?? true,
      confidence_threshold: payload.confidence_threshold ?? 0.7,
      max_documents: payload.max_documents ?? 10,
      max_papers: payload.max_papers ?? 8,
      latest_summary: null,
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      latest_run_job_id: 'job-domain-1',
      active_job_id: 'job-domain-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:00:00Z',
      started_at: '2026-03-24T12:00:00Z',
      paused_at: null,
      last_run_at: null,
    }));
    apiClient.updateDomainResearchProfile.mockImplementation(async (profileId: string, payload: any) => ({
      id: profileId,
      user_id: 'user-1',
      title: 'Compiler Frontier',
      domain: 'Compiler',
      objective: 'Track compiler opportunities',
      status: 'running',
      source_scope: 'kb_plus_arxiv_plus_repo',
      track_type: 'compiler',
      research_mode: 'literature_to_hypothesis',
      report_format: 'brief_and_report',
      automation_profile: payload.automation_profile ?? 'balanced',
      automation_policy: payload.automation_policy ?? {},
      effective_policy: payload.automation_policy ?? {},
      interval_minutes: 1440,
      persist_artifacts: true,
      auto_launch_follow_up: payload.automation_policy?.auto_launch_follow_up ?? true,
      auto_create_experiment_plans: payload.automation_policy?.auto_create_experiment_plans ?? true,
      confidence_threshold: payload.automation_policy?.confidence_threshold ?? 0.7,
      max_documents: 10,
      max_papers: 8,
      latest_summary: { effective_policy: payload.automation_policy ?? {} },
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      latest_validation_run_ids: [],
      latest_run_job_id: 'job-domain-1',
      active_job_id: 'job-domain-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.createResearchPortfolio.mockImplementation(async (payload: any) => ({
      id: 'portfolio-1',
      user_id: 'user-1',
      title: payload.title,
      objective: payload.objective,
      status: payload.start_immediately === false ? 'draft' : 'running',
      linked_profile_ids: payload.linked_profile_ids ?? [],
      automation_profile: payload.automation_profile ?? 'balanced',
      automation_policy: payload.automation_policy ?? {},
      opportunities: [],
      latest_summary: {},
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      child_job_ids: [],
      active_job_id: 'job-portfolio-1',
      latest_run_job_id: 'job-portfolio-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:00:00Z',
      started_at: '2026-03-24T12:00:00Z',
      paused_at: null,
      last_run_at: null,
    }));
    apiClient.updateResearchPortfolio.mockImplementation(async (portfolioId: string, payload: any) => ({
      id: portfolioId,
      user_id: 'user-1',
      title: 'Scientific Fleet',
      objective: 'Rank and validate scientific opportunities',
      status: 'running',
      linked_profile_ids: ['profile-1'],
      automation_profile: payload.automation_profile ?? 'balanced',
      automation_policy: payload.automation_policy ?? {},
      effective_policy: payload.automation_policy ?? {},
      opportunities: [],
      latest_summary: { effective_policy: payload.automation_policy ?? {} },
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      child_job_ids: [],
      active_job_id: 'job-portfolio-1',
      latest_run_job_id: 'job-portfolio-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.promoteDomainResearchAgentJob.mockImplementation(async (jobId: string, payload: any) => ({
      source_job_id: jobId,
      promotion_status: payload.target_mode === 'profile_with_portfolio' ? 'promoted_to_profile_and_portfolio' : 'promoted_to_profile',
      domain_research_profile_id: 'profile-promoted-1',
      research_portfolio_id: payload.target_mode === 'profile_with_portfolio' ? (payload.portfolio_id || 'portfolio-promoted-1') : null,
      profile: {
        id: 'profile-promoted-1',
        title: payload.profile?.title || 'Promoted Monitor',
        status: payload.start_profile_now === false ? 'draft' : 'running',
      },
      portfolio: payload.target_mode === 'profile_with_portfolio'
        ? {
            id: payload.portfolio_id || 'portfolio-promoted-1',
            title: payload.portfolio?.title || 'Promoted Fleet',
            status: payload.run_portfolio_now ? 'running' : 'draft',
          }
        : null,
      source_job: {
        ...makeJob({
          id: jobId,
          status: 'completed',
          launch_mode: 'quick_start_domain_research',
          promotion_status: payload.target_mode === 'profile_with_portfolio' ? 'promoted_to_profile_and_portfolio' : 'promoted_to_profile',
          promoted_domain_research_profile_id: 'profile-promoted-1',
          promoted_research_portfolio_id: payload.target_mode === 'profile_with_portfolio' ? (payload.portfolio_id || 'portfolio-promoted-1') : null,
        }),
      },
    }));
    apiClient.performDomainResearchProfileAction.mockImplementation(async (profileId: string, payload: any) => ({
      id: profileId,
      user_id: 'user-1',
      title: 'Domain Monitor',
      domain: 'Retrieval',
      objective: 'Track retrieval opportunities',
      status: payload.action === 'pause' ? 'paused' : payload.action === 'cancel' ? 'cancelled' : 'running',
      source_scope: 'kb_plus_arxiv',
      track_type: 'generic',
      monitor_queries: ['retrieval benchmarks'],
      repo_source_ids: [],
      benchmark_queries: [],
      report_format: 'brief_and_report',
      automation_profile: 'balanced',
      automation_policy: {},
      effective_policy: {},
      interval_minutes: 1440,
      persist_artifacts: true,
      auto_launch_follow_up: true,
      auto_create_experiment_plans: true,
      confidence_threshold: 0.7,
      max_documents: 10,
      max_papers: 8,
      latest_summary: {},
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      latest_run_job_id: 'job-domain-1',
      active_job_id: 'job-domain-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.actOnDomainResearchOpportunity.mockImplementation(async (profileId: string, opportunityId: string, payload: any) => ({
      id: profileId,
      user_id: 'user-1',
      title: 'Domain Monitor',
      domain: 'Retrieval',
      objective: 'Track retrieval opportunities',
      status: 'running',
      source_scope: 'kb_plus_arxiv',
      track_type: 'generic',
      research_mode: 'literature_to_hypothesis',
      monitor_queries: ['retrieval benchmarks'],
      repo_source_ids: [],
      benchmark_queries: [],
      report_format: 'brief_and_report',
      automation_profile: 'balanced',
      automation_policy: {},
      effective_policy: {},
      interval_minutes: 1440,
      persist_artifacts: true,
      auto_launch_follow_up: true,
      auto_create_experiment_plans: true,
      confidence_threshold: 0.7,
      max_documents: 10,
      max_papers: 8,
      opportunities: [
        {
          opportunity_id: opportunityId,
          canonical_key: 'retrieval_eval_gap',
          title: 'Retrieval eval gap',
          hypothesis: 'Missing grounded eval baseline',
          stage: payload.action === 'suppress'
            ? 'suppressed'
            : payload.action === 'create_plan'
              ? 'planned'
              : payload.action === 'launch_validation' || payload.action === 'materialize_experiment'
                ? 'validating'
                : 'accepted',
          decision_state: payload.action === 'suppress' ? 'suppressed' : 'accepted',
          decision_source: 'operator',
          operator_note: payload.operator_note ?? null,
          confidence: 0.82,
          novelty: 0.76,
          readiness: 0.74,
          linked_experiment_plan_ids: payload.action === 'create_plan' || payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? ['plan-1'] : [],
          linked_validation_run_ids: payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? ['run-1'] : [],
          latest_experiment_plan_id: payload.action === 'create_plan' || payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? 'plan-1' : null,
          latest_validation_run_id: payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? 'run-1' : null,
          latest_validation_job_id: payload.action === 'materialize_experiment' ? 'job-validation-1' : null,
          latest_validation_status: payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? 'queued' : null,
          latest_validation_blocked_reason_code: null,
          child_job_ids: payload.action === 'launch_follow_up' ? ['job-follow-1'] : [],
        },
      ],
      latest_summary: {},
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      latest_run_job_id: 'job-domain-1',
      active_job_id: 'job-domain-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.performResearchPortfolioAction.mockImplementation(async (portfolioId: string, payload: any) => ({
      id: portfolioId,
      user_id: 'user-1',
      title: 'Fleet Portfolio',
      objective: 'Rank and validate research opportunities',
      status: payload.action === 'pause' ? 'paused' : payload.action === 'cancel' ? 'cancelled' : 'running',
      linked_profile_ids: ['profile-1'],
      automation_policy: {},
      opportunities: [],
      latest_summary: {},
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      child_job_ids: [],
      active_job_id: 'job-portfolio-1',
      latest_run_job_id: 'job-portfolio-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.actOnResearchPortfolioOpportunity.mockImplementation(async (portfolioId: string, opportunityId: string, payload: any) => ({
      id: portfolioId,
      user_id: 'user-1',
      title: 'Fleet Portfolio',
      objective: 'Rank and validate research opportunities',
      status: 'running',
      linked_profile_ids: ['profile-1'],
      automation_policy: {},
      opportunities: [
        {
          opportunity_id: opportunityId,
          canonical_key: 'compiler_hotspot',
          title: 'Compiler hotspot',
          hypothesis: 'Scheduler bottleneck',
          stage: payload.action === 'suppress'
            ? 'suppressed'
            : payload.action === 'create_plan'
              ? 'planned'
              : payload.action === 'launch_validation' || payload.action === 'materialize_experiment'
                ? 'validating'
                : 'accepted',
          decision_state: payload.action === 'suppress' ? 'suppressed' : 'accepted',
          decision_source: 'operator',
          operator_note: payload.operator_note ?? null,
          confidence: 0.88,
          novelty: 0.71,
          readiness: 0.79,
          linked_experiment_plan_ids: payload.action === 'create_plan' || payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? ['plan-1'] : [],
          linked_validation_run_ids: payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? ['run-1'] : [],
          latest_experiment_plan_id: payload.action === 'create_plan' || payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? 'plan-1' : null,
          latest_validation_run_id: payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? 'run-1' : null,
          latest_validation_job_id: payload.action === 'materialize_experiment' ? 'job-validation-1' : null,
          latest_validation_status: payload.action === 'launch_validation' || payload.action === 'materialize_experiment' ? 'queued' : null,
          latest_validation_blocked_reason_code: null,
          child_job_ids: payload.action === 'launch_follow_up' ? ['job-follow-1'] : [],
        },
      ],
      latest_summary: {},
      latest_note_ids: [],
      latest_experiment_plan_ids: [],
      child_job_ids: [],
      active_job_id: 'job-portfolio-1',
      latest_run_job_id: 'job-portfolio-1',
      created_at: '2026-03-24T12:00:00Z',
      updated_at: '2026-03-24T12:05:00Z',
    }));
    apiClient.createCodingBacklogItem.mockImplementation(async (payload: any) => ({
      id: 'backlog-1',
      user_id: 'user-1',
      source_id: payload.source_id,
      title: payload.title,
      portfolio_goal: payload.portfolio_goal,
      status: 'running',
      priority: payload.priority ?? 50,
      scope: payload.scope ?? 'auto',
      failure_symptom: payload.failure_symptom ?? null,
      error_output: payload.error_output ?? null,
      file_paths: payload.file_paths ?? [],
      commands: payload.commands ?? [],
      auto_apply_enabled: payload.auto_apply_enabled ?? true,
      require_patch_pr: payload.require_patch_pr ?? false,
      policy: payload.policy ?? {},
      decomposition: { strategy: 'portfolio_goal', slices_planned: [], completed_slices: [] },
      child_job_ids: [],
      latest_summary: {},
      orchestrator_job_id: 'job-backlog-1',
      current_job_id: 'job-backlog-1',
      latest_apply_job_id: null,
      latest_proposal_id: null,
      started_at: '2026-03-23T12:00:00Z',
      completed_at: null,
      created_at: '2026-03-23T12:00:00Z',
      updated_at: '2026-03-23T12:00:00Z',
    }));
    apiClient.performCodingBacklogAction.mockImplementation(async (itemId: string, payload: any) => ({
      id: itemId,
      user_id: 'user-1',
      source_id: 'repo-source-1',
      title: 'Backlog Item',
      portfolio_goal: 'Repair the unstable document save flow',
      status: payload?.action === 'pause' ? 'paused' : payload?.action === 'cancel' ? 'cancelled' : payload?.action === 'create_patch_pr' ? 'completed' : 'running',
      priority: 50,
      scope: 'frontend',
      failure_symptom: 'Save flow intermittently 500s',
      error_output: null,
      file_paths: [],
      commands: [],
      auto_apply_enabled: true,
      require_patch_pr: false,
      policy: { max_auto_retries: 1 },
      decomposition: { strategy: 'portfolio_goal', planned_slices: [], completed_slices: [] },
      child_job_ids: [],
      latest_summary: {},
      orchestrator_job_id: 'job-backlog-1',
      current_job_id: 'job-backlog-1',
      latest_apply_job_id: null,
      latest_proposal_id: null,
      started_at: '2026-03-23T12:00:00Z',
      completed_at: null,
      created_at: '2026-03-23T12:00:00Z',
      updated_at: '2026-03-23T12:00:00Z',
    }));
    apiClient.getAgentCheckpointQueue.mockResolvedValue({
      items: [],
      total: 0,
      approvals: 0,
      recoveries: 0,
      follow_ups: 0,
      by_type: {},
      by_status: {},
      by_customer: {},
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
      by_source_kind: { job: 3 },
      by_decision_type: { job_recovery_queued: 2 },
      by_status: {},
      by_customer: {},
      by_severity: {},
      by_actor_mode: {},
      by_triage_status: {},
      by_assignee: {},
      by_escalation_state: {},
      overdue_count: 0,
      has_more: false,
    });
    apiClient.getAgentDecisionTraceAnalytics.mockResolvedValue({
      window_days: 7,
      total: 0,
      by_source_kind: {},
      by_triage_status: {},
      top_decision_types: [],
      top_reason_labels: [],
      top_queue_reasons: [],
      daily_trend: [
        { day: '2026-03-11', count: 0 },
        { day: '2026-03-12', count: 0 },
        { day: '2026-03-13', count: 0 },
        { day: '2026-03-14', count: 0 },
        { day: '2026-03-15', count: 0 },
        { day: '2026-03-16', count: 0 },
        { day: '2026-03-17', count: 0 },
      ],
    });
    apiClient.downloadAgentDecisionTraceExport.mockResolvedValue(undefined);
    apiClient.listAgentDecisionTraceViews.mockResolvedValue({ items: [], total: 0 });
    apiClient.createAgentDecisionTraceView.mockImplementation(async (payload: any) => ({
      id: 'trace-view-1',
      user_id: 'user-1',
      created_at: '2026-03-17T12:00:00Z',
      updated_at: '2026-03-17T12:00:00Z',
      is_default: false,
      ...payload,
    }));
    apiClient.updateAgentDecisionTraceView.mockImplementation(async (_viewId: string, payload: any) => ({
      id: 'trace-view-1',
      user_id: 'user-1',
      name: payload?.name || 'Updated Trace View',
      filters: payload?.filters || {},
      is_default: Boolean(payload?.is_default),
      created_at: '2026-03-17T12:00:00Z',
      updated_at: '2026-03-18T12:00:00Z',
    }));
    apiClient.deleteAgentDecisionTraceView.mockResolvedValue(undefined);
    apiClient.actionAgentDecisionTraceEvent.mockImplementation(async (_eventId: string, payload: any) => ({
      event: {
        event_id: 'trace-event-1',
        event_type: 'job_recovery_queued',
        event_time: '2026-03-17T12:00:00Z',
        source_kind: 'job',
        decision_type: payload.action,
        summary: 'Trace event',
        triage_status: payload.action === 'resolve' ? 'resolved' : payload.action === 'start_investigation' ? 'investigating' : 'acknowledged',
        pinned: payload.action === 'toggle_pin',
        owner_user_id: 'user-1',
        owner_label: 'Test User',
        assigned_to_user_id: payload.assigned_to_user_id || null,
        assignee_label: payload.assigned_to_user_id === 'user-2' ? 'Reviewer User' : null,
        due_at: payload.due_at || null,
        escalation_state: payload.action === 'set_due_at' ? 'escalated' : 'none',
      },
    }));
    apiClient.getResearchMonitorAnalytics.mockResolvedValue({
      generated_at: '2026-03-17T12:00:00Z',
      totals: {
        total_monitors: 2,
        discovered_count: 12,
        accepted_count: 7,
        rejected_count: 3,
        auto_launched_count: 3,
        approval_launched_count: 1,
        blocked_count: 2,
        follow_up_completed_count: 3,
        follow_up_failed_count: 1,
        follow_up_cancelled_count: 1,
        strong_monitors: 1,
        mixed_monitors: 0,
        weak_monitors: 1,
      },
      customers: [
        {
          customer: 'Acme',
          monitor_count: 1,
          strong_monitor_count: 1,
          mixed_monitor_count: 0,
          weak_monitor_count: 0,
          auto_launch_used_24h: 2,
          auto_launch_capacity_24h: 3,
          approval_queue_used_24h: 0,
          approval_queue_capacity_24h: 6,
          alert_used_24h: 1,
          alert_capacity_24h: 4,
          backlog_used: 0,
          backlog_capacity: 8,
          throttled_monitor_count: 0,
          customer_budget: {
            auto_launch_limit_24h: 6,
            approval_queue_limit_24h: 10,
            alert_limit_24h: 5,
            queue_backlog_cap: 12,
          },
          customer_budget_usage: {
            auto_launch_count_24h: 2,
            approval_queue_count_24h: 0,
            alert_count_24h: 1,
            queue_backlog_count: 0,
          },
          customer_budget_remaining: {
            auto_launch_count_24h: 4,
            approval_queue_count_24h: 10,
            alert_count_24h: 4,
            queue_backlog_count: 12,
          },
          customer_budget_throttle_state: 'normal',
          customer_budget_throttle_reasons: [],
          accepted_count: 6,
          blocked_count: 0,
          follow_up_completed_count: 3,
          follow_up_failed_count: 1,
          follow_up_cancelled_count: 0,
          portfolio_status: 'normal',
          portfolio_reasons: [],
          top_launch_monitors: [{ monitor_job_id: 'monitor-1', monitor_name: 'Acme Monitor', customer: 'Acme', value: 2, throttle_state: null }],
          top_backlog_monitors: [],
          top_alert_monitors: [{ monitor_job_id: 'monitor-1', monitor_name: 'Acme Monitor', customer: 'Acme', value: 1, throttle_state: null }],
          throttled_monitors: [],
          rebalance_guidance_status: 'none',
          rebalance_guidance_reasons: [],
          rebalance_guidance_summary: null,
          rebalance_guidance_changes: [],
          latest_rebalance_evaluation_status: undefined,
          latest_rebalance_evaluation_sample_count: 0,
          latest_rebalance_evaluation_target_count: 0,
          latest_rebalance_evaluation_reasons: [],
          recent_rebalance_history: [],
        },
        {
          customer: 'Beta',
          monitor_count: 1,
          strong_monitor_count: 0,
          mixed_monitor_count: 0,
          weak_monitor_count: 1,
          auto_launch_used_24h: 0,
          auto_launch_capacity_24h: 3,
          approval_queue_used_24h: 1,
          approval_queue_capacity_24h: 6,
          alert_used_24h: 2,
          alert_capacity_24h: 4,
          backlog_used: 2,
          backlog_capacity: 8,
          throttled_monitor_count: 1,
          customer_budget: {
            auto_launch_limit_24h: 2,
            approval_queue_limit_24h: 2,
            alert_limit_24h: 2,
            queue_backlog_cap: 2,
          },
          customer_budget_usage: {
            auto_launch_count_24h: 0,
            approval_queue_count_24h: 1,
            alert_count_24h: 2,
            queue_backlog_count: 2,
          },
          customer_budget_remaining: {
            auto_launch_count_24h: 2,
            approval_queue_count_24h: 1,
            alert_count_24h: 0,
            queue_backlog_count: 0,
          },
          customer_budget_throttle_state: 'manual_only_clamped',
          customer_budget_throttle_reasons: ['Shared customer backlog cap is exhausted.'],
          accepted_count: 1,
          blocked_count: 2,
          follow_up_completed_count: 0,
          follow_up_failed_count: 0,
          follow_up_cancelled_count: 1,
          portfolio_status: 'monitor_throttled',
          portfolio_reasons: ['1 monitor(s) are currently throttled.'],
          top_launch_monitors: [],
          top_backlog_monitors: [{ monitor_job_id: 'monitor-2', monitor_name: 'Beta Watch', customer: 'Beta', value: 2, throttle_state: 'manual_only_clamped' }],
          top_alert_monitors: [{ monitor_job_id: 'monitor-2', monitor_name: 'Beta Watch', customer: 'Beta', value: 2, throttle_state: 'manual_only_clamped' }],
          throttled_monitors: [{ monitor_job_id: 'monitor-2', monitor_name: 'Beta Watch', customer: 'Beta', value: 2, throttle_state: 'manual_only_clamped' }],
          rebalance_guidance_status: 'actionable',
          rebalance_guidance_reasons: ['Beta Watch is consuming the most customer budget pressure.', 'Acme Monitor has stronger spare headroom.'],
          rebalance_guidance_summary: 'Shift budget headroom from Beta Watch to Acme Monitor.',
          rebalance_guidance_changes: [
            {
              monitor_job_id: 'monitor-2',
              monitor_name: 'Beta Watch',
              customer: 'Beta',
              current_budget: {
                auto_launch_limit_24h: 3,
                approval_queue_limit_24h: 6,
                alert_limit_24h: 4,
                queue_backlog_cap: 8,
              },
              proposed_budget: {
                auto_launch_limit_24h: 2,
                approval_queue_limit_24h: 5,
                alert_limit_24h: 3,
                queue_backlog_cap: 7,
              },
              delta_budget: {
                auto_launch_limit_24h: -1,
                approval_queue_limit_24h: -1,
                alert_limit_24h: -1,
                queue_backlog_cap: -1,
              },
              reasons: ['Reduce auto-launch headroom on Beta Watch.'],
            },
            {
              monitor_job_id: 'monitor-1',
              monitor_name: 'Acme Monitor',
              customer: 'Acme',
              current_budget: {
                auto_launch_limit_24h: 3,
                approval_queue_limit_24h: 6,
                alert_limit_24h: 4,
                queue_backlog_cap: 8,
              },
              proposed_budget: {
                auto_launch_limit_24h: 4,
                approval_queue_limit_24h: 7,
                alert_limit_24h: 5,
                queue_backlog_cap: 9,
              },
              delta_budget: {
                auto_launch_limit_24h: 1,
                approval_queue_limit_24h: 1,
                alert_limit_24h: 1,
                queue_backlog_cap: 1,
              },
              reasons: ['Reassign auto-launch headroom to Acme Monitor.'],
            },
          ],
          latest_rebalance_evaluation_status: 'mixed',
          latest_rebalance_evaluation_sample_count: 4,
          latest_rebalance_evaluation_target_count: 8,
          latest_rebalance_evaluation_reasons: ['Queue backlog pressure declined'],
          recent_rebalance_history: [
            {
              id: 'rebalance-1',
              at: '2026-03-17T09:30:00Z',
              actor_user_id: 'user-1',
              change_source: 'customer_rebalance_guidance',
              change_reason: 'Shift budget headroom from Beta Watch to Acme Monitor.',
              changes: [
                {
                  monitor_job_id: 'monitor-2',
                  monitor_name: 'Beta Watch',
                  customer: 'Beta',
                  current_budget: {
                    auto_launch_limit_24h: 3,
                    approval_queue_limit_24h: 6,
                    alert_limit_24h: 4,
                    queue_backlog_cap: 8,
                  },
                  proposed_budget: {
                    auto_launch_limit_24h: 2,
                    approval_queue_limit_24h: 5,
                    alert_limit_24h: 3,
                    queue_backlog_cap: 7,
                  },
                  delta_budget: {
                    auto_launch_limit_24h: -1,
                    approval_queue_limit_24h: -1,
                    alert_limit_24h: -1,
                    queue_backlog_cap: -1,
                  },
                  reasons: ['Reduce auto-launch headroom on Beta Watch.'],
                },
              ],
              before_capacity: {
                auto_launch_limit_24h: 6,
                approval_queue_limit_24h: 12,
                alert_limit_24h: 8,
                queue_backlog_cap: 16,
              },
              after_capacity: {
                auto_launch_limit_24h: 6,
                approval_queue_limit_24h: 12,
                alert_limit_24h: 8,
                queue_backlog_cap: 16,
              },
              evaluation_target_count: 8,
              evaluation_state: 'active',
              evaluation_status: 'mixed',
              evaluation_sample_count: 4,
              evaluation_reasons: ['Queue backlog pressure declined'],
              before_counts: {
                accepted_count: 4,
                blocked_count: 2,
                follow_up_completed_count: 0,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 1,
                auto_launch_used_24h: 0,
                approval_queue_used_24h: 1,
                alert_used_24h: 2,
                backlog_used: 3,
                throttled_monitor_count: 1,
              },
              after_counts: {
                accepted_count: 4,
                blocked_count: 1,
                follow_up_completed_count: 1,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 0,
                auto_launch_used_24h: 0,
                approval_queue_used_24h: 1,
                alert_used_24h: 2,
                backlog_used: 2,
                throttled_monitor_count: 1,
              },
              delta_counts: {
                accepted_count: 0,
                blocked_count: -1,
                follow_up_completed_count: 1,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: -1,
                auto_launch_used_24h: 0,
                approval_queue_used_24h: 0,
                alert_used_24h: 0,
                backlog_used: -1,
                throttled_monitor_count: 0,
              },
            },
          ],
        },
      ],
      monitors: [
        {
          monitor_job_id: 'monitor-1',
          monitor_name: 'Acme Monitor',
          monitor_job_type: 'monitor',
          customer: 'Acme',
          discovered_count: 8,
          accepted_count: 6,
          rejected_count: 1,
          acceptance_rate: 75,
          auto_launched_count: 3,
          approval_launched_count: 1,
          queued_for_approval_count: 0,
          manual_only_count: 1,
          blocked_count: 0,
          follow_up_completed_count: 3,
          follow_up_failed_count: 1,
          follow_up_cancelled_count: 0,
          relaunch_count: 1,
          health_score: 81,
          health_bucket: 'strong',
          health_reasons: ['High acceptance rate', 'Launched follow-ups are completing reliably'],
          automation_profile: 'balanced',
          automation_policy: {
            follow_up_review_mode: 'manual_only',
            allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
          },
          effective_policy: {
            follow_up_review_mode: 'manual_only',
            allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
          },
          autonomy_budget: {
            auto_launch_limit_24h: 3,
            approval_queue_limit_24h: 6,
            alert_limit_24h: 4,
            queue_backlog_cap: 8,
          },
          budget_usage: {
            auto_launch_count_24h: 2,
            approval_queue_count_24h: 0,
            alert_count_24h: 1,
            queue_backlog_count: 0,
          },
          budget_remaining: {
            auto_launch_count_24h: 1,
            approval_queue_count_24h: 6,
            alert_count_24h: 3,
            queue_backlog_count: 8,
          },
          budget_throttle_state: 'normal',
          budget_throttle_reasons: [],
          budget_history_count: 1,
          latest_budget_changed_at: '2026-03-17T08:00:00Z',
          latest_budget_change_source: 'customer_rebalance_guidance',
          latest_budget_actor_user_id: 'user-1',
          latest_budget_change_reason: 'Customer rebalance guidance for Acme',
          recommended_policy_mode: 'auto_launch_safe',
          recommended_allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
          policy_reasons: ['Discovery quality and follow-up outcomes support safe auto-launch'],
          policy_confidence: 'high',
          policy_history_count: 2,
          latest_policy_changed_at: '2026-03-16T09:00:00Z',
          latest_policy_change_source: 'guided_recommendation',
          latest_policy_actor_user_id: 'user-1',
          latest_policy_evaluation_status: 'improving',
          latest_policy_evaluation_sample_count: 4,
          latest_policy_evaluation_target_count: 8,
          latest_policy_evaluation_reasons: ['Completion rate improved from 25.0% to 75.0%'],
          policy_guardrail_status: null,
          policy_guardrail_action: null,
          policy_guardrail_reasons: [],
          policy_guardrail_target_history_entry_id: null,
          policy_guardrail_target_policy: null,
          policy_mode_counts: { auto_launch_safe: 4, manual_only: 1 },
          recent_policy_history: [
            {
              id: 'history-1',
              at: '2026-03-16T09:00:00Z',
              actor_user_id: 'user-1',
              change_source: 'guided_recommendation',
              change_reason: 'Strong autonomy health',
              previous_follow_up_autonomy: {
                mode: 'manual_only',
                allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
              },
              next_follow_up_autonomy: {
                mode: 'auto_launch_safe',
                allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
              },
              analytics_context: { health_bucket: 'strong', policy_confidence: 'high' },
              evaluation_target_count: 8,
              evaluation_state: 'active',
              evaluation_status: 'improving',
              evaluation_sample_count: 4,
              evaluation_reasons: ['Completion rate improved from 25.0% to 75.0%'],
              before_counts: {
                accepted_count: 4,
                auto_launched_count: 1,
                approval_launched_count: 0,
                queued_for_approval_count: 0,
                manual_only_count: 2,
                blocked_count: 1,
                follow_up_completed_count: 1,
                follow_up_failed_count: 1,
                follow_up_cancelled_count: 0,
              },
              after_counts: {
                accepted_count: 4,
                auto_launched_count: 3,
                approval_launched_count: 1,
                queued_for_approval_count: 0,
                manual_only_count: 0,
                blocked_count: 0,
                follow_up_completed_count: 3,
                follow_up_failed_count: 1,
                follow_up_cancelled_count: 0,
              },
              delta_counts: {
                accepted_count: 0,
                auto_launched_count: 2,
                approval_launched_count: 1,
                queued_for_approval_count: 0,
                manual_only_count: -2,
                blocked_count: -1,
                follow_up_completed_count: 2,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 0,
              },
            },
            {
              id: 'history-0',
              at: '2026-03-14T09:00:00Z',
              actor_user_id: 'user-1',
              change_source: 'create_monitor',
              change_reason: 'Initial monitor setup',
              previous_follow_up_autonomy: {
                mode: 'manual_only',
                allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
              },
              next_follow_up_autonomy: {
                mode: 'manual_only',
                allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
              },
              analytics_context: {},
              evaluation_target_count: 8,
              evaluation_state: 'active',
              evaluation_status: 'insufficient_data',
              evaluation_sample_count: 1,
              evaluation_reasons: ['Only 1 accepted signal(s) observed after this policy change'],
              before_counts: {
                accepted_count: 0,
                auto_launched_count: 0,
                approval_launched_count: 0,
                queued_for_approval_count: 0,
                manual_only_count: 0,
                blocked_count: 0,
                follow_up_completed_count: 0,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 0,
              },
              after_counts: {
                accepted_count: 1,
                auto_launched_count: 0,
                approval_launched_count: 0,
                queued_for_approval_count: 0,
                manual_only_count: 1,
                blocked_count: 0,
                follow_up_completed_count: 0,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 0,
              },
              delta_counts: {
                accepted_count: 1,
                auto_launched_count: 0,
                approval_launched_count: 0,
                queued_for_approval_count: 0,
                manual_only_count: 1,
                blocked_count: 0,
                follow_up_completed_count: 0,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 0,
              },
            },
          ],
          top_recommendations: [
            {
              recommendation_key: 'deep_dive_chain',
              launch_count: 4,
              auto_launch_count: 3,
              approval_launch_count: 1,
              blocked_count: 0,
              completed_count: 3,
              failed_count: 1,
              cancelled_count: 0,
              success_rate: 75,
              score_trend: 'positive',
              monitor_count: 1,
            },
          ],
        },
        {
          monitor_job_id: 'monitor-2',
          monitor_name: 'Beta Watch',
          monitor_job_type: 'monitor',
          customer: 'Beta',
          discovered_count: 4,
          accepted_count: 1,
          rejected_count: 2,
          acceptance_rate: 25,
          auto_launched_count: 0,
          approval_launched_count: 0,
          queued_for_approval_count: 1,
          manual_only_count: 1,
          blocked_count: 2,
          follow_up_completed_count: 0,
          follow_up_failed_count: 0,
          follow_up_cancelled_count: 1,
          relaunch_count: 0,
          health_score: 24,
          health_bucket: 'weak',
          health_reasons: ['Low acceptance rate', 'Many accepted items are blocked by policy'],
          automation_profile: 'balanced',
          automation_policy: {
            follow_up_review_mode: 'queue_for_approval',
            allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
          },
          effective_policy: {
            follow_up_review_mode: 'queue_for_approval',
            allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
          },
          autonomy_budget: {
            auto_launch_limit_24h: 3,
            approval_queue_limit_24h: 6,
            alert_limit_24h: 4,
            queue_backlog_cap: 8,
          },
          budget_usage: {
            auto_launch_count_24h: 0,
            approval_queue_count_24h: 1,
            alert_count_24h: 2,
            queue_backlog_count: 2,
          },
          budget_remaining: {
            auto_launch_count_24h: 3,
            approval_queue_count_24h: 5,
            alert_count_24h: 2,
            queue_backlog_count: 6,
          },
          budget_throttle_state: 'manual_only_clamped',
          budget_throttle_reasons: ['Queue backlog cap reached for this monitor.'],
          budget_history_count: 0,
          latest_budget_changed_at: undefined,
          latest_budget_change_source: undefined,
          latest_budget_actor_user_id: undefined,
          latest_budget_change_reason: undefined,
          recommended_policy_mode: 'manual_only',
          recommended_allowed_recommendations: ['deep_dive_chain'],
          policy_reasons: ['Recent follow-up outcomes are too weak for autonomy'],
          policy_confidence: 'medium',
          policy_history_count: 1,
          latest_policy_changed_at: '2026-03-15T08:00:00Z',
          latest_policy_change_source: 'manual_override',
          latest_policy_actor_user_id: 'user-1',
          latest_policy_evaluation_status: 'degrading',
          latest_policy_evaluation_sample_count: 3,
          latest_policy_evaluation_target_count: 8,
          latest_policy_evaluation_reasons: ['More accepted items are getting blocked by policy'],
          policy_guardrail_status: 'active',
          policy_guardrail_action: 'rollback',
          policy_guardrail_reasons: ['More accepted items are getting blocked by policy', 'Apply a more conservative policy until outcomes recover'],
          policy_guardrail_target_history_entry_id: 'history-2',
          policy_guardrail_target_policy: {
            follow_up_review_mode: 'queue_for_approval',
            allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
          },
          policy_mode_counts: { queue_for_approval: 1, manual_only: 1 },
          recent_policy_history: [
            {
              id: 'history-2',
              at: '2026-03-15T08:00:00Z',
              actor_user_id: 'user-1',
              change_source: 'manual_override',
              change_reason: 'Downgraded after failures',
              previous_follow_up_autonomy: {
                mode: 'queue_for_approval',
                allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
              },
              next_follow_up_autonomy: {
                mode: 'manual_only',
                allowed_recommendations: ['deep_dive_chain'],
              },
              analytics_context: { health_bucket: 'weak', policy_confidence: 'medium' },
              evaluation_target_count: 8,
              evaluation_state: 'active',
              evaluation_status: 'degrading',
              evaluation_sample_count: 3,
              evaluation_reasons: ['More accepted items are getting blocked by policy'],
              before_counts: {
                accepted_count: 3,
                auto_launched_count: 1,
                approval_launched_count: 0,
                queued_for_approval_count: 0,
                manual_only_count: 1,
                blocked_count: 0,
                follow_up_completed_count: 1,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 0,
              },
              after_counts: {
                accepted_count: 3,
                auto_launched_count: 0,
                approval_launched_count: 0,
                queued_for_approval_count: 1,
                manual_only_count: 1,
                blocked_count: 2,
                follow_up_completed_count: 0,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 1,
              },
              delta_counts: {
                accepted_count: 0,
                auto_launched_count: -1,
                approval_launched_count: 0,
                queued_for_approval_count: 1,
                manual_only_count: 0,
                blocked_count: 2,
                follow_up_completed_count: -1,
                follow_up_failed_count: 0,
                follow_up_cancelled_count: 1,
              },
            },
          ],
          top_recommendations: [
            {
              recommendation_key: 'single_research_job',
              launch_count: 1,
              auto_launch_count: 0,
              approval_launch_count: 0,
              blocked_count: 1,
              completed_count: 0,
              failed_count: 0,
              cancelled_count: 1,
              success_rate: 0,
              score_trend: 'negative',
              monitor_count: 1,
            },
          ],
        },
      ],
      recommendations: [
        {
          recommendation_key: 'deep_dive_chain',
          launch_count: 4,
          auto_launch_count: 3,
          approval_launch_count: 1,
          blocked_count: 0,
          completed_count: 3,
          failed_count: 1,
          cancelled_count: 0,
          success_rate: 75,
          score_trend: 'positive',
          monitor_count: 1,
        },
      ],
    });
    apiClient.getResearchMonitorPolicyEvaluation.mockResolvedValue({
      monitor_job_id: 'monitor-1',
      history_entry_id: 'history-1',
      evaluation_status: 'improving',
      evaluation_sample_count: 4,
      evaluation_target_count: 8,
      evaluation_reasons: ['Completion rate improved from 25.0% to 75.0%'],
      before_counts: {
        accepted_count: 4,
        auto_launched_count: 1,
        approval_launched_count: 0,
        queued_for_approval_count: 0,
        manual_only_count: 2,
        blocked_count: 1,
        follow_up_completed_count: 1,
        follow_up_failed_count: 1,
        follow_up_cancelled_count: 0,
      },
      after_counts: {
        accepted_count: 4,
        auto_launched_count: 3,
        approval_launched_count: 1,
        queued_for_approval_count: 0,
        manual_only_count: 0,
        blocked_count: 0,
        follow_up_completed_count: 3,
        follow_up_failed_count: 1,
        follow_up_cancelled_count: 0,
      },
      delta_counts: {
        accepted_count: 0,
        auto_launched_count: 2,
        approval_launched_count: 1,
        queued_for_approval_count: 0,
        manual_only_count: -2,
        blocked_count: -1,
        follow_up_completed_count: 2,
        follow_up_failed_count: 0,
        follow_up_cancelled_count: 0,
      },
      sample_items: [
        {
          item_id: 'sample-before-1',
          title: 'Before launch gap',
          period: 'before',
          launch_status: 'blocked',
          outcome_status: undefined,
          recommendation_key: 'deep_dive_chain',
          summary: 'Accepted signal that stayed blocked before the rollout.',
        },
        {
          item_id: 'sample-after-1',
          title: 'After safe launch',
          period: 'after',
          launch_status: 'launched',
          outcome_status: 'completed',
          recommendation_key: 'deep_dive_chain',
          summary: 'Accepted signal that completed after auto-launch.',
        },
      ],
    });
    apiClient.getResearchMonitorCustomerRebalanceEvaluation.mockResolvedValue({
      customer: 'Beta',
      history_entry_id: 'rebalance-1',
      evaluation_status: 'mixed',
      evaluation_sample_count: 4,
      evaluation_target_count: 8,
      evaluation_reasons: ['Queue backlog pressure declined'],
      before_counts: {
        accepted_count: 4,
        blocked_count: 2,
        follow_up_completed_count: 0,
        follow_up_failed_count: 0,
        follow_up_cancelled_count: 1,
        auto_launch_used_24h: 0,
        approval_queue_used_24h: 1,
        alert_used_24h: 2,
        backlog_used: 3,
        throttled_monitor_count: 1,
      },
      after_counts: {
        accepted_count: 4,
        blocked_count: 1,
        follow_up_completed_count: 1,
        follow_up_failed_count: 0,
        follow_up_cancelled_count: 0,
        auto_launch_used_24h: 0,
        approval_queue_used_24h: 1,
        alert_used_24h: 2,
        backlog_used: 2,
        throttled_monitor_count: 1,
      },
      delta_counts: {
        accepted_count: 0,
        blocked_count: -1,
        follow_up_completed_count: 1,
        follow_up_failed_count: 0,
        follow_up_cancelled_count: -1,
        auto_launch_used_24h: 0,
        approval_queue_used_24h: 0,
        alert_used_24h: 0,
        backlog_used: -1,
        throttled_monitor_count: 0,
      },
      sample_items: [
        {
          item_id: 'beta-before-1',
          title: 'Before rebalance pressure',
          period: 'before',
          launch_status: 'blocked',
          outcome_status: undefined,
          recommendation_key: 'single_research_job',
          summary: 'Signal blocked before the rebalance.',
          monitor_job_id: 'monitor-2',
          monitor_name: 'Beta Watch',
        },
        {
          item_id: 'beta-after-1',
          title: 'After rebalance recovery',
          period: 'after',
          launch_status: 'launched',
          outcome_status: 'completed',
          recommendation_key: 'single_research_job',
          summary: 'Signal completed after the rebalance.',
          monitor_job_id: 'monitor-2',
          monitor_name: 'Beta Watch',
        },
      ],
    });
    apiClient.updateResearchMonitorPolicy.mockResolvedValue({
      monitor_job_id: 'monitor-1',
      automation_profile: 'balanced',
      automation_policy: {
        follow_up_review_mode: 'auto_launch_safe',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      effective_policy: {
        follow_up_review_mode: 'auto_launch_safe',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      follow_up_autonomy: {
        mode: 'auto_launch_safe',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      latest_history_entry: {
        id: 'history-new',
        at: '2026-03-17T12:00:00Z',
        actor_user_id: 'user-1',
        change_source: 'guided_recommendation',
        previous_follow_up_autonomy: {
          mode: 'manual_only',
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        next_follow_up_autonomy: {
          mode: 'auto_launch_safe',
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        analytics_context: {},
      },
      policy_history_count: 3,
    });
    apiClient.rollbackResearchMonitorPolicy.mockResolvedValue({
      monitor_job_id: 'monitor-1',
      automation_profile: 'balanced',
      automation_policy: {
        follow_up_review_mode: 'manual_only',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      effective_policy: {
        follow_up_review_mode: 'manual_only',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      follow_up_autonomy: {
        mode: 'manual_only',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      latest_history_entry: {
        id: 'history-rollback',
        at: '2026-03-17T12:05:00Z',
        actor_user_id: 'user-1',
        change_source: 'rollback',
        previous_follow_up_autonomy: {
          mode: 'auto_launch_safe',
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        next_follow_up_autonomy: {
          mode: 'manual_only',
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        analytics_context: {},
      },
      policy_history_count: 4,
    });
    apiClient.simulateResearchMonitorPolicy.mockResolvedValue({
      monitor_job_id: 'monitor-1',
      current_policy: {
        mode: 'manual_only',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      proposed_policy: {
        mode: 'auto_launch_safe',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
      },
      history_limit: 25,
      baseline_counts: {
        auto_launch_safe_count: 0,
        queue_for_approval_count: 0,
        manual_only_count: 4,
        blocked_count: 1,
        insufficient_context_count: 0,
      },
      simulated_counts: {
        auto_launch_safe_count: 3,
        queue_for_approval_count: 0,
        manual_only_count: 0,
        blocked_count: 2,
        insufficient_context_count: 0,
      },
      delta_counts: {
        auto_launch_safe_count: 3,
        queue_for_approval_count: 0,
        manual_only_count: -4,
        blocked_count: 1,
        insufficient_context_count: 0,
      },
      top_recommendation_deltas: [
        {
          recommendation_key: 'deep_dive_chain',
          baseline_count: 1,
          simulated_count: 4,
          delta_count: 3,
        },
      ],
      sample_items: [
        {
          item_id: 'inbox-1',
          title: 'Accepted signal',
          recommendation_key: 'deep_dive_chain',
          current_outcome: 'manual_only',
          simulated_outcome: 'auto_launch_safe',
          reason: 'Policy would auto-launch this safe bounded follow-up.',
        },
      ],
      insufficient_context_count: 0,
    });
    apiClient.updateResearchMonitorCustomerBudget.mockResolvedValue({
      customer: 'Acme',
      customer_budget: {
        auto_launch_limit_24h: 7,
        approval_queue_limit_24h: 10,
        alert_limit_24h: 5,
        queue_backlog_cap: 12,
      },
    });
    apiClient.previewResearchMonitorCustomerRebalance.mockResolvedValue({
      customer: 'Beta',
      guidance_status: 'actionable',
      guidance_summary: 'Shift budget headroom from Beta Watch to Acme Monitor.',
      guidance_reasons: ['Beta Watch is consuming the most customer budget pressure.'],
      before_capacity: {
        auto_launch_limit_24h: 6,
        approval_queue_limit_24h: 12,
        alert_limit_24h: 8,
        queue_backlog_cap: 16,
      },
      after_capacity: {
        auto_launch_limit_24h: 6,
        approval_queue_limit_24h: 12,
        alert_limit_24h: 8,
        queue_backlog_cap: 16,
      },
      changes: [
        {
          monitor_job_id: 'monitor-2',
          monitor_name: 'Beta Watch',
          customer: 'Beta',
          current_budget: {
            auto_launch_limit_24h: 3,
            approval_queue_limit_24h: 6,
            alert_limit_24h: 4,
            queue_backlog_cap: 8,
          },
          proposed_budget: {
            auto_launch_limit_24h: 2,
            approval_queue_limit_24h: 5,
            alert_limit_24h: 3,
            queue_backlog_cap: 7,
          },
          delta_budget: {
            auto_launch_limit_24h: -1,
            approval_queue_limit_24h: -1,
            alert_limit_24h: -1,
            queue_backlog_cap: -1,
          },
          reasons: ['Reduce auto-launch headroom on Beta Watch.'],
        },
      ],
    });
    apiClient.applyResearchMonitorCustomerRebalance.mockResolvedValue({
      customer: 'Beta',
      updated_monitor_ids: ['monitor-2', 'monitor-1'],
      guidance_status: 'none',
      guidance_summary: null,
      latest_history_entries: [],
    });
    apiClient.getResearchInboxStats.mockResolvedValue({
      total_items: 0,
      new_items: 0,
      accepted_items: 0,
      rejected_items: 0,
    });
    apiClient.actionAgentCheckpointQueueFollowUp.mockResolvedValue({
      ok: true,
      detail: 'Follow-up decision recorded',
    });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValue({
      requested_count: 1,
      applied: 1,
      failed: 0,
      results: [],
    });
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'document',
          item_key: 'doc-1',
          title: 'Accepted signal',
          summary: 'Signal summary',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });
    apiClient.getMyPreferences.mockResolvedValue({});
    apiClient.updateMyPreferences.mockResolvedValue({});
    apiClient.getUnsafeExecAvailability.mockResolvedValue({ enabled: false, backend: 'subprocess' });
    apiClient.listArxivImports.mockResolvedValue({ items: [], total: 0 });
    apiClient.listReadingLists.mockResolvedValue({ items: [], total: 0 });
    apiClient.getAgentJobLog.mockResolvedValue({ entries: [] });
    apiClient.getAgentJobStepEvents.mockResolvedValue({ items: [], total: 0, source: 'results.execution_strategy.step_events' });
    apiClient.getJobMemories.mockResolvedValue({ memories: [], total: 0, page: 1, page_size: 100 });
    apiClient.getAgentJobRelaunchLineage.mockResolvedValue({ parent_job_id: null, ancestors: [], descendants: [] });
    apiClient.quickStartBugTriageSwarmJob.mockResolvedValue(makeJob({ id: 'bug-swarm-1', name: 'Bug Triage Swarm Job' }));
    apiClient.quickStartBuildBreakSwarmJob.mockResolvedValue(makeJob({ id: 'build-swarm-1', name: 'Build Break Swarm Job' }));
    apiClient.quickStartFrontendRegressionSwarmJob.mockResolvedValue(makeJob({ id: 'frontend-swarm-1', name: 'Frontend Regression Swarm Job' }));
    apiClient.createCodingSwarmProfile.mockImplementation(async (payload: any) => ({ id: 'profile-1', user_id: 'user-1', created_at: '2026-03-10T00:00:00Z', updated_at: '2026-03-10T00:00:00Z', status: 'active', is_default: false, ...payload }));
    apiClient.updateCodingSwarmProfile.mockImplementation(async (profileId: string, payload: any) => ({ id: profileId, user_id: 'user-1', source_id: 'repo-source-1', created_at: '2026-03-10T00:00:00Z', updated_at: '2026-03-11T00:00:00Z', title: 'Updated Profile', status: 'active', is_default: false, preset_key: 'bug_triage_swarm', scope_default: 'auto', max_agents: 4, safe_command_policy: 'standard', ...payload }));
    apiClient.deleteCodingSwarmProfile.mockResolvedValue(undefined);
    apiClient.listCollaborationUsers.mockResolvedValue({
      items: [
        {
          id: 'user-1',
          username: 'testuser',
          email: 'test@example.com',
          full_name: 'Test User',
          role: 'user',
          is_active: true,
          is_verified: true,
          login_count: 1,
          created_at: '2026-03-10T00:00:00Z',
        },
        {
          id: 'user-2',
          username: 'reviewer',
          email: 'reviewer@example.com',
          full_name: 'Reviewer User',
          role: 'user',
          is_active: true,
          is_verified: true,
          login_count: 1,
          created_at: '2026-03-10T00:00:00Z',
        },
      ],
      total: 2,
      page: 1,
      page_size: 100,
    });
    apiClient.quickStartDomainResearchJob.mockImplementation(async (payload: any) => ({
      ...job,
      id: 'job-domain-research',
      name: payload?.name || 'Domain Research',
      goal: payload?.objective || job.goal,
      status: 'pending',
      launch_mode: 'quick_start_domain_research',
      config: {
        domain: payload?.domain,
        objective: payload?.objective,
        source_scope: payload?.source_scope,
      },
    }));
    apiClient.quickStartRepoBugTriageJob.mockImplementation(async (payload: any) => ({
      ...job,
      id: 'job-repo-bug',
      name: payload?.name || 'Repo Bug Triage',
      goal: payload?.goal || job.goal,
      status: 'pending',
      launch_mode: 'quick_start_repo_bug_triage',
      config: {
        source_id: payload?.source_id,
        failure_symptom: payload?.failure_symptom,
        scope: payload?.scope,
        search_query: payload?.search_query,
      },
    }));
    apiClient.performAgentJobAction.mockImplementation(async (jobId: string, action: string) => ({
      ...job,
      id: jobId,
      status: action === 'restart' ? 'pending' : job.status,
    }));
  });

  afterEach(async () => {
    await flushMockPromises();
    while (renderedViews.length > 0) {
      const view = renderedViews.pop();
      view?.unmount();
    }
    consoleErrorSpy?.mockRestore();
    consoleErrorSpy = null;
    consoleWarnSpy?.mockRestore();
    consoleWarnSpy = null;
    jest.clearAllMocks();
  });

  it('renders runtime summary chips in the jobs list for a running job', async () => {
    await renderWithProviders('/autonomous-agents');

    await expectJobHeading('Autonomous Runtime Job');
    expect(screen.getByText('Guard Blocks')).toBeInTheDocument();
    expect(screen.getByText('Failed Cmds')).toBeInTheDocument();
    expect(screen.getByText('Open Failures')).toBeInTheDocument();
    expect(screen.getByText('Open Recovery Jobs')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(screen.getByText('Verify 2')).toBeInTheDocument();
      expect(screen.getByText('Scope repo-123')).toBeInTheDocument();
      expect(screen.getByText('Guard blocks 1')).toBeInTheDocument();
      expect(screen.getByText('Recovery open')).toBeInTheDocument();
      expect(screen.getByText(/fallback verification still failing/i)).toBeInTheDocument();
      expect(screen.getByText(/Inspect failing fallback output/i)).toBeInTheDocument();
      expect(screen.getAllByText('Final retry_primary').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Bootstrap ok').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Failed cmds 1').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Verify cmds 1').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Repo Knowledge Repo').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Last restart (failed -> pending)').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Outcome applied').length).toBeGreaterThan(0);
      expect(screen.getByText('Pause')).toBeInTheDocument();
      expect(screen.getAllByText('Copy failed command').length).toBeGreaterThan(0);
    });

    fireEvent.click(screen.getByText('Pause'));

    await waitFor(() => {
      expect(apiClient.performAgentJobAction).toHaveBeenCalledWith('job-1', 'pause', expect.any(Object));
    });
  });

  it('shows a system map panel from the page header', async () => {
    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    expect(screen.queryByText('Current operator surface and runtime ownership.')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('System Map'));

    expect(await screen.findByText(/Current operator surface and runtime ownership/i)).toBeInTheDocument();
    expect(screen.getByText(/docs\/ARCHITECTURE_ASCII\.md/i)).toBeInTheDocument();
    expect(screen.getByText(/Canonical autonomy:/i)).toBeInTheDocument();
    expect(screen.getByText(/autonomous_agent_executor/i)).toBeInTheDocument();
  });

  it('renders the decision trace tab with canonical events', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-1',
          event_type: 'follow_up_queued',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'portfolio',
          source_id: 'portfolio-1',
          source_label: 'Scientific Fleet',
          customer: null,
          decision_type: 'follow_up_queued',
          reason_code: 'follow_up_launch_approval',
          status: 'eligible',
          severity: 'medium',
          summary: 'Scientific Fleet: Compiler hotspot is follow up queued',
          operator_note: 'Awaiting human approval',
          before_state: null,
          after_state: { autonomy_state: 'eligible', review_status: 'pending_approval' },
          deep_link: { target_tab: 'fleet', params: { tab: 'fleet', fleetId: 'portfolio-1' }, label: 'Open Scientific Fleet' },
          metadata: { opportunity_id: 'opp-1' },
          triage_status: 'new',
          owner_user_id: 'user-1',
          owner_label: 'Test User',
          assigned_to_user_id: 'user-2',
          assignee_label: 'Reviewer User',
          due_at: '2026-03-23T12:00:00Z',
          escalation_state: 'warning',
          escalation_reason: 'pinned_stale',
          pinned: true,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { portfolio: 1 },
      by_decision_type: { follow_up_queued: 1 },
      by_status: { eligible: 1 },
      by_customer: { Unassigned: 1 },
      by_severity: { medium: 1 },
      by_actor_mode: { operator: 1 },
      by_triage_status: { new: 1 },
      by_assignee: { 'user-2': 1 },
      by_escalation_state: { warning: 1 },
      overdue_count: 1,
      has_more: false,
    });
    apiClient.getAgentDecisionTraceAnalytics.mockResolvedValue({
      window_days: 7,
      total: 1,
      by_source_kind: { portfolio: 1 },
      by_triage_status: { new: 1 },
      top_decision_types: [{ value: 'follow_up_queued', count: 1 }],
      top_reason_labels: [{ value: 'Follow-up launch approval', count: 1 }],
      top_queue_reasons: [{ value: 'follow_up_launch_approval', count: 1 }],
      daily_trend: [
        { day: '2026-03-16', count: 0 },
        { day: '2026-03-17', count: 0 },
        { day: '2026-03-18', count: 0 },
        { day: '2026-03-19', count: 0 },
        { day: '2026-03-20', count: 0 },
        { day: '2026-03-21', count: 0 },
        { day: '2026-03-22', count: 1 },
      ],
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));

    expect(await screen.findByRole('heading', { name: 'Decision Trace' })).toBeInTheDocument();
    expect(await screen.findByText('Trace mix')).toBeInTheDocument();
    expect(screen.getAllByText('follow up queued (1)').length).toBeGreaterThan(0);
    expect(screen.getByText('Reason labels')).toBeInTheDocument();
    expect(screen.getByText('Queue reasons')).toBeInTheDocument();
    expect(screen.getByText('7-day trend')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /JSON/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /CSV/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /CSV/i }));
    await waitFor(() => {
      expect(apiClient.downloadAgentDecisionTraceExport).toHaveBeenCalledWith({
        format: 'csv',
        source_kind: undefined,
        decision_type: undefined,
        customer: undefined,
        status: undefined,
        severity: undefined,
        actor_mode: undefined,
        triage_status: undefined,
        assigned_to_user_id: undefined,
        unassigned_only: undefined,
        escalation_state: undefined,
        pinned: undefined,
        actionable_only: undefined,
        start_at: expect.any(String),
      });
    });
    expect(await screen.findByText('Scientific Fleet: Compiler hotspot is follow up queued')).toBeInTheDocument();
    expect(screen.getByText('Note: Awaiting human approval')).toBeInTheDocument();
    expect(screen.getByText(/Owner:\s*Test User/)).toBeInTheDocument();
    expect(screen.getByText('Assignee: Reviewer User')).toBeInTheDocument();
    expect(screen.getByText('Save Current View')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Trace view name'), { target: { value: 'Team Trace View' } });
    const defaultToggle = screen.getByLabelText('Default trace view');
    expect(defaultToggle).not.toBeChecked();
    fireEvent.click(defaultToggle);
    fireEvent.click(screen.getByText('Save Current View'));
    await waitFor(() => {
      expect(apiClient.createAgentDecisionTraceView).toHaveBeenCalledWith({
        name: 'Team Trace View',
        filters: expect.objectContaining({
          date_range: '7d',
        }),
        is_default: true,
      });
    });
    fireEvent.click(defaultToggle);
    fireEvent.click(screen.getByText('Update View'));
    await waitFor(() => {
      expect(apiClient.updateAgentDecisionTraceView).toHaveBeenCalledWith(
        'trace-view-1',
        expect.objectContaining({
          name: 'Team Trace View',
          filters: expect.objectContaining({
            date_range: '7d',
          }),
          is_default: false,
        })
      );
    });
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);
    expect(apiClient.getAgentDecisionTrace).toHaveBeenCalledWith({
      source_kind: undefined,
      decision_type: undefined,
      customer: undefined,
      status: undefined,
      severity: undefined,
      actor_mode: undefined,
      triage_status: undefined,
      assigned_to_user_id: undefined,
      unassigned_only: undefined,
      escalation_state: undefined,
      pinned: undefined,
      actionable_only: undefined,
      start_at: expect.any(String),
      limit: 50,
      offset: 0,
    });
  });

  it('auto-applies the default trace view and labels it in the selector', async () => {
    apiClient.listAgentDecisionTraceViews.mockResolvedValueOnce({
      items: [
        {
          id: 'trace-view-default',
          user_id: 'user-1',
          name: 'Default Trace View',
          filters: { date_range: '7d' },
          is_default: true,
          created_at: '2026-03-17T12:00:00Z',
          updated_at: '2026-03-17T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));

    expect(await screen.findByRole('option', { name: 'Default Trace View (Default)' })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByLabelText('Trace saved view')).toHaveValue('trace-view-default');
      expect(screen.getByLabelText('Default trace view')).toBeChecked();
    });

    fireEvent.change(screen.getByLabelText('Trace source filter'), { target: { value: 'job' } });
    await waitFor(() => {
      expect(screen.getByLabelText('Trace saved view')).toHaveValue('');
    }, { timeout: 3000 });
  });

  it('restores explicit trace filters from the URL instead of the default trace view', async () => {
    apiClient.listAgentDecisionTraceViews.mockResolvedValueOnce({
      items: [
        {
          id: 'trace-view-default',
          user_id: 'user-1',
          name: 'Default Trace View',
          filters: { source_kind: 'portfolio', date_range: '7d' },
          is_default: true,
          created_at: '2026-03-17T12:00:00Z',
          updated_at: '2026-03-17T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
      by_source_kind: { job: 3, portfolio: 1 },
      by_decision_type: {},
      by_status: {},
      by_customer: {},
      by_severity: {},
      by_actor_mode: {},
      by_triage_status: {},
      by_assignee: {},
      by_escalation_state: {},
      overdue_count: 0,
      has_more: false,
    });
    apiClient.getAgentDecisionTraceAnalytics.mockResolvedValue({
      window_days: 7,
      total: 0,
      by_source_kind: {},
      by_triage_status: {},
      top_decision_types: [],
      top_reason_labels: [],
      top_queue_reasons: [],
      daily_trend: [],
    });

    await renderWithProviders('/autonomous-agents?tab=trace&trace_source_kind=job&trace_decision_type=job_recovery_queued&trace_pinned=true&trace_date_range=30d');

    expect(await screen.findByRole('heading', { name: 'Decision Trace' })).toBeInTheDocument();
    await waitFor(() => {
      expect(apiClient.getAgentDecisionTrace).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.getByLabelText('Trace saved view')).toHaveValue('');
      expect(apiClient.getAgentDecisionTrace).toHaveBeenCalledWith(
        expect.objectContaining({
          source_kind: 'job',
          decision_type: 'job_recovery_queued',
          pinned: true,
          start_at: expect.any(String),
        })
      );
    });

    const clipboardWriteText = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: clipboardWriteText },
      configurable: true,
    });
    fireEvent.click(screen.getByRole('button', { name: /Copy Trace Link/i }));
    await waitFor(() => {
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('tab=trace'));
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('trace_source_kind=job'));
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('trace_decision_type=job_recovery_queued'));
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('trace_pinned=true'));
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('trace_date_range=30d'));
    });
  });

  it('opens a trace permalink event without letting the default view override it', async () => {
    apiClient.listAgentDecisionTraceViews.mockResolvedValueOnce({
      items: [
        {
          id: 'trace-view-default',
          user_id: 'user-1',
          name: 'Default Trace View',
          filters: { source_kind: 'portfolio', date_range: '7d' },
          is_default: true,
          created_at: '2026-03-17T12:00:00Z',
          updated_at: '2026-03-17T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-recovery-1',
          event_type: 'job_recovery_queued',
          event_time: '2026-03-16T09:30:00Z',
          source_kind: 'job',
          source_id: 'job-scheduler-1',
          source_label: 'Recovery Job',
          customer: null,
          decision_type: 'job_recovery_queued',
          reason_code: 'execution_failure',
          reason_label: 'Execution failure',
          scheduler_state: {
            last_run_status: 'failed',
            failure_streak: 2,
            queue_reason: 'execution_failure',
            last_scheduled_at: '2026-03-16T09:00:00Z',
            last_dispatched_at: '2026-03-16T09:05:00Z',
            current_run_started_at: '2026-03-16T09:06:00Z',
            backoff_until: '2026-03-16T12:00:00Z',
            backoff_seconds: 1800,
          },
          status: 'failed',
          severity: 'high',
          actor_mode: 'autonomous',
          summary: 'Recovery Job: queued for scheduler recovery',
          before_state: null,
          after_state: null,
          deep_link: { target_tab: 'queue', params: { tab: 'queue', job: 'job-scheduler-1' }, label: 'Open Checkpoint Queue' },
          metadata: { reason_label: 'Execution failure' },
          is_derived: true,
          record_origin: 'derived_fallback',
          triage_status: 'new',
          pinned: false,
          escalation_state: 'none',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { job: 1 },
      by_decision_type: { job_recovery_queued: 1 },
      by_status: { failed: 1 },
      by_customer: {},
      by_severity: { high: 1 },
      by_actor_mode: { autonomous: 1 },
      by_triage_status: { new: 1 },
      by_assignee: {},
      by_escalation_state: {},
      overdue_count: 0,
      has_more: false,
    });
    apiClient.getAgentDecisionTraceAnalytics.mockResolvedValue({
      window_days: 7,
      total: 1,
      by_source_kind: { job: 1 },
      by_triage_status: { new: 1 },
      top_decision_types: [{ value: 'job_recovery_queued', count: 1 }],
      top_reason_labels: [{ value: 'Execution failure', count: 1 }],
      top_queue_reasons: [{ value: 'execution_failure', count: 1 }],
      daily_trend: [{ day: '2026-03-16', count: 1 }],
    });

    await renderWithProviders('/autonomous-agents?tab=trace&trace_event=evt-recovery-1');

    expect(await screen.findByRole('heading', { name: 'Decision Trace' })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByLabelText('Trace saved view')).toHaveValue('');
      expect(screen.getByLabelText('Default trace view')).not.toBeChecked();
    });
    expect(await screen.findByRole('button', { name: /Copy Event Link/i })).toBeInTheDocument();
    expect(await screen.findByText('Metadata')).toBeInTheDocument();

    const clipboardWriteText = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: clipboardWriteText },
      configurable: true,
    });
    fireEvent.click(screen.getByRole('button', { name: /Copy Event Link/i }));
    await waitFor(() => {
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('tab=trace'));
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('trace_event=evt-recovery-1'));
      expect(clipboardWriteText).toHaveBeenCalledWith(expect.stringContaining('trace_date_range=7d'));
    });
  });

  it('renders derived job recovery trace metadata with a readable reason label', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-recovery-1',
          event_type: 'job_recovery_queued',
          event_time: '2026-03-16T09:30:00Z',
          source_kind: 'job',
          source_id: 'job-scheduler-1',
          source_label: 'Recovery Job',
          customer: null,
          decision_type: 'job_recovery_queued',
          reason_code: 'execution_failure',
          reason_label: 'Execution failure',
          scheduler_state: {
            last_run_status: 'failed',
            failure_streak: 2,
            queue_reason: 'execution_failure',
            last_scheduled_at: '2026-03-16T09:00:00Z',
            last_dispatched_at: '2026-03-16T09:05:00Z',
            current_run_started_at: '2026-03-16T09:06:00Z',
            backoff_until: '2026-03-16T12:00:00Z',
            backoff_seconds: 1800,
          },
          status: 'failed',
          severity: 'high',
          actor_mode: 'autonomous',
          summary: 'Recovery Job: queued for scheduler recovery',
          before_state: null,
          after_state: null,
          deep_link: { target_tab: 'queue', params: { tab: 'queue', job: 'job-scheduler-1' }, label: 'Open Checkpoint Queue' },
          metadata: { reason_label: 'Execution failure' },
          is_derived: true,
          record_origin: 'derived_fallback',
          triage_status: 'new',
          pinned: false,
          escalation_state: 'none',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { job: 1 },
      by_decision_type: { job_recovery_queued: 1 },
      by_status: { failed: 1 },
      by_customer: { Unassigned: 1 },
      by_severity: { high: 1 },
      by_actor_mode: { autonomous: 1 },
      by_triage_status: { new: 1 },
      by_assignee: {},
      by_escalation_state: { none: 1 },
      overdue_count: 0,
      has_more: false,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));
    expect(await screen.findByText('Recovery Job: queued for scheduler recovery')).toBeInTheDocument();
    expect(screen.getAllByText(/Execution failure/).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByText('Recovery Job: queued for scheduler recovery'));

    expect(await screen.findByText(/Reason label: Execution failure/)).toBeInTheDocument();
    expect(screen.getByText('Derived fallback')).toBeInTheDocument();
    expect(screen.getAllByText(/Last run failed/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Failure streak 2/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Queue reason execution failure/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Backoff until/i).length).toBeGreaterThan(0);
  });

  it('updates trace assignee and due date through inline controls', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-inline-1',
          event_type: 'validation_blocked',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'validation_run',
          source_id: 'run-1',
          source_label: 'Validation Run',
          decision_type: 'validation_blocked',
          summary: 'Validation run blocked',
          triage_status: 'new',
          owner_user_id: 'user-1',
          owner_label: 'Test User',
          assigned_to_user_id: null,
          due_at: null,
          pinned: false,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { validation_run: 1 },
      by_decision_type: { validation_blocked: 1 },
      by_status: { unknown: 1 },
      by_customer: { Unassigned: 1 },
      by_severity: { unknown: 1 },
      by_actor_mode: { unknown: 1 },
      by_triage_status: { new: 1 },
      by_assignee: { unassigned: 1 },
      by_escalation_state: { none: 1 },
      overdue_count: 0,
      has_more: false,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));
    fireEvent.click(await screen.findByText('Validation run blocked'));

    fireEvent.change(screen.getByLabelText('Assignee'), { target: { value: 'user-2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply Assignee' }));

    await waitFor(() => {
      expect(apiClient.actionAgentDecisionTraceEvent).toHaveBeenCalledWith(
        'evt-inline-1',
        expect.objectContaining({ action: 'assign', assigned_to_user_id: 'user-2' })
      );
    });

    fireEvent.change(screen.getByLabelText('Due at'), { target: { value: '2026-03-24T09:30' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply Due' }));

    await waitFor(() => {
      expect(apiClient.actionAgentDecisionTraceEvent).toHaveBeenCalledWith(
        'evt-inline-1',
        expect.objectContaining({ action: 'set_due_at', due_at: expect.any(String) })
      );
    });
  });

  it('shows compiler trace context, filters blocked validations, and opens the exact domain target', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-validation-compiler-1',
          event_type: 'validation_blocked',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'validation_run',
          source_id: 'run-1',
          source_label: 'Validation Run: Compiler hotspot',
          decision_type: 'validation_blocked',
          summary: 'Validation Run: Compiler hotspot: validation blocked',
          deep_link: {
            target_tab: 'domain',
            job_id: 'job-validation-1',
            params: { tab: 'domain', profileId: 'profile-1', opportunityId: 'opp-compiler-1', job: 'job-validation-1' },
            label: 'Open Domain',
          },
          track_type: 'compiler',
          domain: 'Compiler',
          objective: 'Validate compiler hotspot',
          source_scope: 'kb_plus_arxiv_plus_repo',
          repo_source_ids: ['repo-source-1'],
          benchmark_queries: ['llvm-test-suite'],
          sandbox_profile_id: 'scientific-compiler-sandbox',
          automation_profile: 'balanced',
          effective_policy: { follow_up_review_mode: 'queue_for_approval' },
          confidence: 0.81,
          readiness: 0.79,
          linked_experiment_plan_ids: ['plan-1'],
          linked_validation_run_ids: ['run-1'],
          child_job_ids: ['job-validation-1'],
          triage_status: 'new',
          pinned: false,
          is_derived: true,
        },
        {
          event_id: 'evt-generic-1',
          event_type: 'job_recovery_queued',
          event_time: '2026-03-22T11:00:00Z',
          source_kind: 'job',
          source_id: 'job-generic-1',
          source_label: 'Generic Job',
          decision_type: 'job_recovery_queued',
          summary: 'Generic recovery event',
          triage_status: 'new',
          pinned: false,
          is_derived: true,
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
      by_source_kind: { validation_run: 1, job: 1 },
      by_decision_type: { validation_blocked: 1, job_recovery_queued: 1 },
      by_status: { blocked: 1, failed: 1 },
      by_customer: {},
      by_severity: { high: 1, medium: 1 },
      by_actor_mode: { autonomous: 2 },
      by_triage_status: { new: 2 },
      by_assignee: { unassigned: 2 },
      by_escalation_state: { none: 2 },
      overdue_count: 0,
      has_more: false,
    });
    apiClient.listDomainResearchProfiles.mockResolvedValue({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Validate compiler hotspot',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'balanced',
          effective_policy: { follow_up_review_mode: 'queue_for_approval' },
          opportunities: [
            {
              opportunity_id: 'opp-compiler-1',
              title: 'Compiler hotspot',
              stage: 'planned',
              confidence: 0.81,
              readiness: 0.79,
              autonomy_state: 'planned',
              linked_experiment_plan_ids: ['plan-1'],
              linked_validation_run_ids: ['run-1'],
              child_job_ids: ['job-validation-1'],
            },
          ],
          latest_summary: {},
          latest_note_ids: ['note-1'],
          latest_experiment_plan_ids: ['plan-1'],
          latest_validation_run_ids: ['run-1'],
          latest_run_job_id: 'job-validation-1',
          active_job_id: 'job-validation-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents?tab=trace', { documentSources: defaultDocumentSources });

    expect(await screen.findByText('Validation Run: Compiler hotspot: validation blocked')).toBeInTheDocument();
    expect(screen.getByText('Domain: Compiler')).toBeInTheDocument();
    expect(screen.getByText('Objective: Validate compiler hotspot')).toBeInTheDocument();
    expect(screen.getByText('Track: compiler')).toBeInTheDocument();
    expect(screen.getByText('Source scope: kb plus arxiv plus repo')).toBeInTheDocument();
    expect(screen.getByText('Repo inputs: repo-source-1')).toBeInTheDocument();
    expect(screen.getByText('Benchmarks: llvm-test-suite')).toBeInTheDocument();
    expect(screen.getByText('Sandbox: scientific-compiler-sandbox')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Blocked validations' }));
    await waitFor(() => {
      expect(screen.queryByText('Generic recovery event')).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Validation Run: Compiler hotspot: validation blocked'));
    fireEvent.click(screen.getByRole('button', { name: 'Open Domain' }));

    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    const row = await screen.findByText('Compiler hotspot');
    expect(row.closest('div.border')?.className).toContain('border-cyan-300');
  });

  it('approves a pending follow-up from the decision trace', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-follow-up-1',
          event_type: 'follow_up_queued_for_approval',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'domain_profile',
          source_id: 'profile-1',
          source_label: 'Compiler Frontier',
          decision_type: 'follow_up_queued_for_approval',
          summary: 'Compiler Frontier: queued follow-up approval for compiler hotspot',
          metadata: { opportunity_id: 'opp-compiler-1' },
          deep_link: { target_tab: 'domain', params: { tab: 'domain', profileId: 'profile-1', opportunityId: 'opp-compiler-1' }, label: 'Open Domain Profiles' },
          triage_status: 'new',
          pinned: false,
          is_derived: false,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { domain_profile: 1 },
      by_decision_type: { follow_up_queued_for_approval: 1 },
      by_status: { eligible: 1 },
      by_customer: {},
      by_severity: {},
      by_actor_mode: {},
      by_triage_status: { new: 1 },
      by_assignee: {},
      by_escalation_state: { none: 1 },
      overdue_count: 0,
      has_more: false,
    });
    apiClient.actionAgentDecisionTraceEvent.mockResolvedValueOnce({
      event: {
        event_id: 'evt-follow-up-1',
        event_type: 'follow_up_approved',
        event_time: '2026-03-22T12:00:00Z',
        source_kind: 'domain_profile',
        source_id: 'profile-1',
        source_label: 'Compiler Frontier',
        decision_type: 'follow_up_approved',
        summary: 'Compiler Frontier: approved queued follow-up',
        triage_status: 'resolved',
        pinned: false,
        metadata: { opportunity_id: 'opp-compiler-1' },
        after_state: {
          opportunity_id: 'opp-compiler-1',
          follow_up_launch_status: 'launched',
          follow_up_operator_decision: 'approved_launch',
          follow_up_job_id: 'job-follow-up-1',
        },
      },
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));
    fireEvent.click(await screen.findByText('Compiler Frontier: queued follow-up approval for compiler hotspot'));
    fireEvent.change(screen.getByPlaceholderText('Approval or rejection note'), { target: { value: 'Looks safe' } });
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(apiClient.actionAgentDecisionTraceEvent).toHaveBeenCalledWith(
        'evt-follow-up-1',
        expect.objectContaining({ action: 'approve_launch', note: 'Looks safe' })
      );
    });
  });

  it('relaunches a failed follow-up from the decision trace', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-follow-up-failed-1',
          event_type: 'follow_up_failed',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'domain_profile',
          source_id: 'profile-1',
          source_label: 'Compiler Frontier',
          decision_type: 'follow_up_failed',
          status: 'failed',
          summary: 'Compiler Frontier: compiler hotspot is follow up failed',
          triage_status: 'new',
          pinned: false,
          is_derived: false,
          after_state: {
            opportunity_id: 'opp-compiler-1',
            follow_up_outcome_status: 'failed',
            follow_up_last_job_id: 'job-follow-up-old',
          },
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { domain_profile: 1 },
      by_decision_type: { follow_up_failed: 1 },
      by_status: { failed: 1 },
      by_customer: {},
      by_severity: {},
      by_actor_mode: {},
      by_triage_status: { new: 1 },
      by_assignee: {},
      by_escalation_state: { none: 1 },
      overdue_count: 0,
      has_more: false,
    });
    apiClient.actionAgentDecisionTraceEvent.mockResolvedValueOnce({
      event: {
        event_id: 'evt-follow-up-failed-1',
        event_type: 'follow_up_launched',
        event_time: '2026-03-22T12:05:00Z',
        source_kind: 'domain_profile',
        source_id: 'profile-1',
        source_label: 'Compiler Frontier',
        decision_type: 'follow_up_launched',
        status: 'active',
        summary: 'Compiler Frontier: relaunched terminal follow-up',
        triage_status: 'resolved',
        pinned: false,
        after_state: {
          opportunity_id: 'opp-compiler-1',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: null,
          follow_up_last_job_id: 'job-follow-up-new',
        },
      },
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));
    fireEvent.click(await screen.findByText('Compiler Frontier: compiler hotspot is follow up failed'));
    fireEvent.change(screen.getByPlaceholderText('Relaunch note'), { target: { value: 'Retry now' } });
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch Follow-up' }));

    await waitFor(() => {
      expect(apiClient.actionAgentDecisionTraceEvent).toHaveBeenCalledWith(
        'evt-follow-up-failed-1',
        expect.objectContaining({ action: 'relaunch_follow_up', note: 'Retry now' })
      );
    });
  });

  it('does not render follow-up controls for derived trace events', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-follow-up-derived',
          event_type: 'follow_up_queued_for_approval',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'portfolio',
          source_id: 'portfolio-1',
          source_label: 'Scientific Fleet',
          decision_type: 'follow_up_queued_for_approval',
          summary: 'Scientific Fleet: queued follow-up approval for compiler hotspot',
          metadata: { opportunity_id: 'opp-compiler-1' },
          triage_status: 'new',
          pinned: false,
          is_derived: true,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { portfolio: 1 },
      by_decision_type: { follow_up_queued_for_approval: 1 },
      by_status: { eligible: 1 },
      by_customer: {},
      by_severity: {},
      by_actor_mode: {},
      by_triage_status: { new: 1 },
      by_assignee: {},
      by_escalation_state: { none: 1 },
      overdue_count: 0,
      has_more: false,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));
    fireEvent.click(await screen.findByText('Scientific Fleet: queued follow-up approval for compiler hotspot'));

    expect(screen.queryByPlaceholderText('Approval or rejection note')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Approve' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reject' })).not.toBeInTheDocument();
  });

  it('does not render relaunch controls for completed follow-up trace events', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-follow-up-completed',
          event_type: 'follow_up_completed',
          event_time: '2026-03-22T12:00:00Z',
          source_kind: 'portfolio',
          source_id: 'portfolio-1',
          source_label: 'Scientific Fleet',
          decision_type: 'follow_up_completed',
          status: 'completed',
          summary: 'Scientific Fleet: hotspot is follow up completed',
          triage_status: 'new',
          pinned: false,
          is_derived: false,
          after_state: {
            opportunity_id: 'opp-fleet-1',
            follow_up_outcome_status: 'completed',
            follow_up_last_job_id: 'job-follow-up-1',
          },
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
      by_source_kind: { portfolio: 1 },
      by_decision_type: { follow_up_completed: 1 },
      by_status: { completed: 1 },
      by_customer: {},
      by_severity: {},
      by_actor_mode: {},
      by_triage_status: { new: 1 },
      by_assignee: {},
      by_escalation_state: { none: 1 },
      overdue_count: 0,
      has_more: false,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Decision Trace'));
    fireEvent.click(await screen.findByText('Scientific Fleet: hotspot is follow up completed'));

    expect(screen.queryByRole('button', { name: 'Relaunch Follow-up' })).not.toBeInTheDocument();
  });

  it('renders coding execution metadata for repo bug triage jobs', async () => {
    const codingJob = makeJob({
      id: 'job-code-1',
      name: 'Repo Bug Triage Job',
      launch_mode: 'quick_start_repo_bug_triage',
      results: {
        code_patch: {
          proposal_id: 'proposal-123',
          title: 'Fix save regression',
          summary: 'Guard save handler before invoking callback',
          risks: ['May miss another caller path'],
          tests_to_run: ['CI=true npm --prefix frontend test -- --watchAll=false'],
        },
        code_patch_execution: {
          mode: 'repo_bug_triage_patch_proposal',
          source_id: 'repo-source-1',
          source_name: 'Knowledge Repo',
          source_type: 'github',
          scope: 'frontend',
          failure_symptom: 'Saving a document returns 500 and leaves the spinner running',
          error_output: 'TypeError: saveDocument is not a function',
          workspace: {
            created: true,
            workspace_id: 'ws-123',
            source_type: 'github',
            file_count: 27,
          },
          inferred_project_profile: {
            detected_stack: ['node', 'python'],
          },
          verification_plan: {
            commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
            bootstrap_commands: ['npm --prefix frontend install'],
            fallback_commands: ['python3 -m pytest -q backend/tests'],
            auto_inferred: true,
          },
          execution_plan: [
            {
              step_id: 'triage_context',
              title: 'Triage failure context',
              status: 'done',
              objective: 'Ground the reported symptom and likely files.',
            },
            {
              step_id: 'verify_patch',
              title: 'Verify candidate patch',
              status: 'pending',
              objective: 'Run bounded verification commands and capture failures.',
              commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
            },
          ],
          proposal_strategy: 'best_passing',
          recovery: {
            recovery_state: 'verification_failed',
            retry_reason: 'Verification failed and needs a refined retry.',
            last_failed_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
            suggested_operator_actions: ['retry_with_refined_plan', 'relaunch_clean_run'],
            can_retry_with_refined_plan: true,
            can_resume_verification: false,
            latest_failed_output: 'TypeError: saveDocument is not a function',
          },
        },
      },
      experiment_run: null,
      experiment_runs: [],
    });

    apiClient.listAgentJobs.mockResolvedValueOnce({
      jobs: [codingJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValueOnce(codingJob);

    await renderWithProviders('/autonomous-agents?job=job-code-1');

    expect((await screen.findAllByText('Repo Bug Triage Job')).length).toBeGreaterThan(0);
    expect(screen.getByText('Workspace 27')).toBeInTheDocument();
    expect(screen.getByText('Planned verify 1')).toBeInTheDocument();
    expect(screen.getByText('Plan 2 steps')).toBeInTheDocument();
    expect(screen.getByText('Stack node, python')).toBeInTheDocument();

    expect(await screen.findByText('Coding execution')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Coding execution'));
    expect(await screen.findByText('Failure symptom')).toBeInTheDocument();
    expect(screen.getByText(/Saving a document returns 500/i)).toBeInTheDocument();
    expect(screen.getByText('Workspace 27 files')).toBeInTheDocument();
    expect(screen.getByText('Verification auto-inferred')).toBeInTheDocument();
    expect(screen.getByText('Detected stack: node, python')).toBeInTheDocument();
    expect(screen.getByText('Recovery')).toBeInTheDocument();
    expect(screen.getByText(/Verification failed and needs a refined retry/i)).toBeInTheDocument();
    expect(screen.getByText('Execution plan')).toBeInTheDocument();
    expect(screen.getByText('Verify candidate patch')).toBeInTheDocument();
    expect(screen.getAllByText(/CI=true npm --prefix frontend test -- --watchAll=false/i).length).toBeGreaterThan(0);
  });

  it('shows coding-specific recovery actions for repo bug triage jobs', async () => {
    const recoveryJob = makeJob({
      id: 'job-code-recovery',
      name: 'Repo Bug Triage Recovery',
      status: 'failed',
      launch_mode: 'quick_start_repo_bug_triage',
      experiment_run: {
        source_id: 'repo-source-1',
        source_name: 'Knowledge Repo',
        ok: false,
        commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
        verification_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
        bootstrap_commands: ['npm --prefix frontend install'],
        fallback_commands: ['python3 -m pytest -q backend/tests'],
        phases: ['primary', 'fallback'],
        final_phase: 'fallback',
        failed_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
        bootstrap_attempted: false,
        bootstrap_ok: null,
        bootstrap_used: false,
        fallback_attempted: true,
        fallback_ok: false,
        fallback_used: true,
      },
      results: {
        code_patch_execution: {
          mode: 'repo_bug_triage_patch_proposal',
          source_id: 'repo-source-1',
          source_name: 'Knowledge Repo',
          scope: 'frontend',
          failure_symptom: 'Saving a document returns 500',
          verification_plan: {
            commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
            auto_inferred: true,
          },
          recovery: {
            recovery_state: 'verification_failed',
            retry_reason: 'Fallback verification still failing.',
            last_failed_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
            suggested_operator_actions: ['retry_with_refined_plan', 'relaunch_clean_run'],
            can_retry_with_refined_plan: true,
            can_resume_verification: false,
            latest_failed_output: 'TypeError: saveDocument is not a function',
          },
        },
      },
      experiment_runs: [],
    });

    apiClient.listAgentJobs.mockResolvedValueOnce({
      jobs: [recoveryJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValueOnce(recoveryJob);
    apiClient.performAgentJobAction.mockResolvedValueOnce({
      ...recoveryJob,
      id: 'job-code-recovery-retry',
      status: 'pending',
    });

    await renderWithProviders('/autonomous-agents?job=job-code-recovery');

    expect((await screen.findAllByText('Repo Bug Triage Recovery')).length).toBeGreaterThan(0);
    expect(screen.getByText('Retry with refined plan')).toBeInTheDocument();
    expect(screen.getByText('Relaunch clean run')).toBeInTheDocument();
    expect(screen.queryByText('Open Checkpoint Queue')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Retry with refined plan'));

    await waitFor(() => {
      expect(apiClient.performAgentJobAction).toHaveBeenCalledWith('job-code-recovery', 'restart', expect.any(Object));
    });
  });

  it('filters the jobs list to unresolved recovery jobs from the summary controls', async () => {
    await renderWithProviders('/autonomous-agents');

    await expectJobHeading('Autonomous Runtime Job');
    await expectJobHeading('Unresolved Recovery Job');

    fireEvent.click(screen.getByText('Open Recovery Jobs'));

    await waitFor(() => {
      const titles = screen.getAllByRole('heading', { level: 3 }).map((node) => node.textContent || '');
      expect(titles).toContain('Unresolved Recovery Job');
      expect(titles).not.toContain('Autonomous Runtime Job');
      expect(titles).not.toContain('Fallback Recovery Job');
      expect(titles).not.toContain('Clean Scope Job');
    });
  });

  it('filters the jobs list to bootstrap-recovered experiment runs from summary controls', async () => {
    await renderWithProviders('/autonomous-agents');

    await expectJobHeading('Autonomous Runtime Job');
    await expectJobHeading('Clean Scope Job');

    fireEvent.click(screen.getAllByRole('button', { name: /Bootstrap/i })[0]);

    await waitFor(() => {
      const titles = screen.getAllByRole('heading', { level: 3 }).map((node) => node.textContent || '');
      expect(titles).toContain('Autonomous Runtime Job');
      expect(titles).not.toContain('Clean Scope Job');
    });
  });

  it('renders live execution graph and scope observability for a deep-linked running job', async () => {
    await renderWithProviders('/autonomous-agents?job=job-1');

    expect(await screen.findByText('Execution Graph')).toBeInTheDocument();
    expect(await screen.findByText('Scope Observability')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Experiment runs'));

    await waitFor(() => {
      expect(screen.getAllByText('Live runtime')).toHaveLength(2);
    });

    expect(screen.getByText('Verification actions: 2')).toBeInTheDocument();
    expect(screen.getByText('Summaries: 1')).toBeInTheDocument();
    expect(screen.getByText('Operator Interventions')).toBeInTheDocument();
    expect(screen.getByText('Recovery Audit')).toBeInTheDocument();
    expect(screen.getByText('2 intervention(s)')).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Latest action: restart (failed -> pending)')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Outcome: applied')
    ).toBeInTheDocument();
    expect(
      screen.getAllByText((_, element) => element?.textContent === 'Outcome reason: Job resumed after intervention').length
    ).toBeGreaterThan(0);
    expect(
      screen.getByText((_, element) => element?.textContent === 'Recovery reason: verification debt building')
    ).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Next step: Re-ground on failing verification output')
    ).toBeInTheDocument();
    expect(screen.getByText(/restart • failed -> pending/i)).toBeInTheDocument();
    expect(screen.getByText(/pause • running -> paused/i)).toBeInTheDocument();
    expect(screen.getByText(/tool python -m pytest -q backend\/tests/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Retry after fallback failure/i).length).toBeGreaterThan(0);
    expect(screen.getByText('Resolved scope: repo-123')).toBeInTheDocument();
    expect(screen.getByText('Scope source: config.source_id')).toBeInTheDocument();
    expect(screen.getByText('Scope events: 2')).toBeInTheDocument();
    expect(screen.getByText('Guard blocks: 1')).toBeInTheDocument();
    expect(screen.getByText(/resolved_scope \| scope repo-123 \| source config\.source_id/i)).toBeInTheDocument();
    expect(screen.getAllByText('Final retry_primary').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Bootstrap ok').length).toBeGreaterThan(0);
    expect(screen.getByText('Phases primary -> bootstrap -> retry_primary')).toBeInTheDocument();
    expect(screen.getAllByText('Source Knowledge Repo').length).toBeGreaterThan(0);
    expect(screen.getAllByText('repo-123').length).toBeGreaterThan(0);
    expect(screen.getByText('Stack node, python')).toBeInTheDocument();
    expect(screen.getAllByText('Last restart (failed -> pending)').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Outcome applied').length).toBeGreaterThan(0);
    expect(screen.getByText('Recent intervention timeline')).toBeInTheDocument();
    expect(screen.getByText(/pause \(running -> paused\): Paused for manual inspection \[superseded\]/i)).toBeInTheDocument();
    expect(screen.getByText(/restart \(failed -> pending\): Retry after fallback failure \[applied\]/i)).toBeInTheDocument();
    expect(
      screen.getAllByText((_, element) => element?.textContent === 'Outcome reason: Job resumed after intervention').length
    ).toBeGreaterThan(0);
    expect(screen.getByText('Fallback verification')).toBeInTheDocument();
    expect(screen.getByText('Failed commands')).toBeInTheDocument();
  });

  it('opens the linked domain opportunity from an inbox row', async () => {
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-domain-target-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'document',
          item_key: 'doc-domain-target-1',
          title: 'Compiler follow-up source',
          summary: 'Open the linked domain opportunity.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_job_id: 'job-follow-up-1',
          follow_up_last_job_id: 'job-follow-up-1',
          origin_source_kind: 'profile',
          origin_source_id: 'profile-1',
          origin_opportunity_id: 'opp-domain-1',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Compiler follow-up source')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open Target' }));

    await waitFor(() => {
      expect(apiClient.listDomainResearchProfiles).toHaveBeenCalled();
    });
  });

  it('opens the linked fleet opportunity from an inbox row', async () => {
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-fleet-target-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'document',
          item_key: 'doc-fleet-target-1',
          title: 'Fleet follow-up source',
          summary: 'Open the linked fleet opportunity.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_job_id: 'job-follow-up-2',
          follow_up_last_job_id: 'job-follow-up-2',
          origin_source_kind: 'portfolio',
          origin_source_id: 'portfolio-1',
          origin_opportunity_id: 'opp-fleet-1',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Fleet follow-up source')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open Target' }));

    await waitFor(() => {
      expect(apiClient.listResearchPortfolios).toHaveBeenCalled();
    });
  });

  it('keeps inbox rows without origin metadata job-centric', async () => {
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-no-target-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'document',
          item_key: 'doc-no-target-1',
          title: 'Detached follow-up source',
          summary: 'No originating opportunity metadata.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_job_id: 'job-follow-up-3',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Detached follow-up source')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open Target' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Follow-up' })).toBeInTheDocument();
  });

  it('approves a pending inbox follow-up recommendation inline', async () => {
    apiClient.listResearchInboxItems
      .mockResolvedValueOnce({
        items: [
          {
            id: 'inbox-follow-up-approve-1',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-approve-1',
            title: 'Queued compiler follow-up',
            summary: 'Needs operator approval.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:00:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'pending_approval',
            created_at: '2026-03-17T11:00:00Z',
            updated_at: '2026-03-17T11:00:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      })
      .mockResolvedValue({
        items: [
          {
            id: 'inbox-follow-up-approve-1',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-approve-1',
            title: 'Queued compiler follow-up',
            summary: 'Needs operator approval.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:00:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'launched',
            follow_up_job_id: 'job-follow-up-approve-1',
            created_at: '2026-03-17T11:00:00Z',
            updated_at: '2026-03-17T11:05:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      });
    apiClient.actionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      ok: true,
      inbox_item_id: 'inbox-follow-up-approve-1',
      follow_up_launch_status: 'launched',
      follow_up_job_id: 'job-follow-up-approve-1',
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Queued compiler follow-up')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Inbox follow-up note for Queued compiler follow-up'), {
      target: { value: 'Looks safe to launch' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve Follow-up' }));

    await waitFor(() => {
      expect(apiClient.actionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        inbox_item_id: 'inbox-follow-up-approve-1',
        action: 'approve_launch',
        operator_note: 'Looks safe to launch',
      });
    });

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Approve Follow-up' })).not.toBeInTheDocument();
    });
  });

  it('rejects a pending inbox follow-up recommendation inline and shows the refreshed operator decision', async () => {
    apiClient.listResearchInboxItems
      .mockResolvedValueOnce({
        items: [
          {
            id: 'inbox-follow-up-reject-1',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-reject-1',
            title: 'Risky compiler follow-up',
            summary: 'Needs operator review.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:00:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'pending_approval',
            created_at: '2026-03-17T11:00:00Z',
            updated_at: '2026-03-17T11:00:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      })
      .mockResolvedValue({
        items: [
          {
            id: 'inbox-follow-up-reject-1',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-reject-1',
            title: 'Risky compiler follow-up',
            summary: 'Needs operator review.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:00:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'pending_approval',
            follow_up_operator_decision: 'rejected',
            follow_up_operator_note: 'Need stronger evidence',
            created_at: '2026-03-17T11:00:00Z',
            updated_at: '2026-03-17T11:05:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      });
    apiClient.actionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      ok: true,
      inbox_item_id: 'inbox-follow-up-reject-1',
      follow_up_launch_status: 'pending_approval',
      follow_up_operator_decision: 'rejected',
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Risky compiler follow-up')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Inbox follow-up note for Risky compiler follow-up'), {
      target: { value: 'Need stronger evidence' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Reject Follow-up' }));

    await waitFor(() => {
      expect(apiClient.actionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        inbox_item_id: 'inbox-follow-up-reject-1',
        action: 'reject_launch',
        operator_note: 'Need stronger evidence',
      });
    });

    expect(await screen.findByText(/Operator: rejected — Need stronger evidence/i)).toBeInTheDocument();
  });

  it('does not render inbox follow-up approval controls for non-pending rows', async () => {
    apiClient.listResearchInboxItems.mockResolvedValueOnce({
      items: [
        {
          id: 'inbox-follow-up-complete-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-complete-1',
          title: 'Completed compiler follow-up',
          summary: 'Already handled.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'launched',
          follow_up_operator_decision: 'approved_launch',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Completed compiler follow-up')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Approve Follow-up' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reject Follow-up' })).not.toBeInTheDocument();
  });

  it('bulk-approves pending inbox follow-up recommendations for one domain profile', async () => {
    apiClient.listResearchInboxItems
      .mockResolvedValueOnce({
        items: [
          {
            id: 'inbox-bulk-domain-1',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-bulk-domain-1',
            title: 'Queued compiler follow-up A',
            summary: 'Needs approval.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:00:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'pending_approval',
            origin_source_kind: 'profile',
            origin_source_id: 'profile-1',
            origin_opportunity_id: 'opp-domain-1',
            created_at: '2026-03-17T11:00:00Z',
            updated_at: '2026-03-17T11:00:00Z',
          },
          {
            id: 'inbox-bulk-domain-2',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-bulk-domain-2',
            title: 'Queued compiler follow-up B',
            summary: 'Needs approval.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:01:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'pending_approval',
            origin_source_kind: 'profile',
            origin_source_id: 'profile-1',
            origin_opportunity_id: 'opp-domain-2',
            created_at: '2026-03-17T11:01:00Z',
            updated_at: '2026-03-17T11:01:00Z',
          },
        ],
        total: 2,
        limit: 100,
        offset: 0,
      })
      .mockResolvedValue({
        items: [
          {
            id: 'inbox-bulk-domain-1',
            user_id: 'user-1',
            job_id: 'monitor-1',
            customer: 'Acme',
            item_type: 'follow_up_recommendation',
            item_key: 'follow-up-bulk-domain-1',
            title: 'Queued compiler follow-up A',
            summary: 'Needs approval.',
            url: null,
            published_at: null,
            discovered_at: '2026-03-17T11:00:00Z',
            status: 'accepted',
            feedback: null,
            metadata: null,
            follow_up_launch_status: 'launched',
            origin_source_kind: 'profile',
            origin_source_id: 'profile-1',
            origin_opportunity_id: 'opp-domain-1',
            created_at: '2026-03-17T11:00:00Z',
            updated_at: '2026-03-17T11:05:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { ok: true, domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-domain-1' },
        { ok: true, domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-domain-2' },
      ],
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Queued compiler follow-up A')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('checkbox')[1]);
    fireEvent.click(screen.getAllByRole('checkbox')[2]);
    fireEvent.change(screen.getByPlaceholderText('Bulk follow-up note (optional)'), {
      target: { value: 'Approve both compiler launches' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve Follow-ups' }));

    await waitFor(() => {
      expect(apiClient.bulkActionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: 'profile-1',
        profile_opportunity_ids: ['opp-domain-1', 'opp-domain-2'],
        portfolio_id: undefined,
        portfolio_opportunity_ids: undefined,
        action: 'approve_launch',
        operator_note: 'Approve both compiler launches',
      });
    });
  });

  it('bulk-rejects pending inbox follow-up recommendations for one research fleet', async () => {
    apiClient.listResearchInboxItems.mockResolvedValueOnce({
      items: [
        {
          id: 'inbox-bulk-fleet-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-bulk-fleet-1',
          title: 'Queued fleet follow-up A',
          summary: 'Needs approval.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'pending_approval',
          origin_source_kind: 'portfolio',
          origin_source_id: 'portfolio-1',
          origin_opportunity_id: 'opp-fleet-1',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-bulk-fleet-2',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-bulk-fleet-2',
          title: 'Queued fleet follow-up B',
          summary: 'Needs approval.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:01:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'pending_approval',
          origin_source_kind: 'portfolio',
          origin_source_id: 'portfolio-1',
          origin_opportunity_id: 'opp-fleet-2',
          created_at: '2026-03-17T11:01:00Z',
          updated_at: '2026-03-17T11:01:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { ok: true, portfolio_id: 'portfolio-1', portfolio_opportunity_id: 'opp-fleet-1' },
        { ok: true, portfolio_id: 'portfolio-1', portfolio_opportunity_id: 'opp-fleet-2' },
      ],
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Queued fleet follow-up A')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('checkbox')[1]);
    fireEvent.click(screen.getAllByRole('checkbox')[2]);
    fireEvent.change(screen.getByPlaceholderText('Bulk follow-up note (optional)'), {
      target: { value: 'Reject until evidence improves' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Reject Follow-ups' }));

    await waitFor(() => {
      expect(apiClient.bulkActionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: undefined,
        profile_opportunity_ids: undefined,
        portfolio_id: 'portfolio-1',
        portfolio_opportunity_ids: ['opp-fleet-1', 'opp-fleet-2'],
        action: 'reject_launch',
        operator_note: 'Reject until evidence improves',
      });
    });
  });

  it('disables inbox bulk follow-up actions for mixed domain and fleet selections', async () => {
    apiClient.listResearchInboxItems.mockResolvedValueOnce({
      items: [
        {
          id: 'inbox-mixed-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-mixed-1',
          title: 'Queued domain follow-up',
          summary: 'Needs approval.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'pending_approval',
          origin_source_kind: 'profile',
          origin_source_id: 'profile-1',
          origin_opportunity_id: 'opp-domain-mixed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-mixed-2',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-mixed-2',
          title: 'Queued fleet follow-up',
          summary: 'Needs approval.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:01:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'pending_approval',
          origin_source_kind: 'portfolio',
          origin_source_id: 'portfolio-1',
          origin_opportunity_id: 'opp-fleet-mixed',
          created_at: '2026-03-17T11:01:00Z',
          updated_at: '2026-03-17T11:01:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Queued domain follow-up')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('checkbox')[1]);
    fireEvent.click(screen.getAllByRole('checkbox')[2]);

    expect(screen.getByRole('button', { name: 'Approve Follow-ups' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reject Follow-ups' })).toBeDisabled();
    expect(screen.getByText('Inbox bulk follow-up actions cannot mix domain and fleet owners.')).toBeInTheDocument();
  });

  it('bulk-relaunches failed inbox follow-ups', async () => {
    apiClient.listResearchInboxItems.mockResolvedValueOnce({
      items: [
        {
          id: 'inbox-relaunch-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-relaunch-1',
          title: 'Failed follow-up A',
          summary: 'Relaunchable.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-relaunch-2',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-relaunch-2',
          title: 'Cancelled follow-up B',
          summary: 'Relaunchable.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:01:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'cancelled',
          created_at: '2026-03-17T11:01:00Z',
          updated_at: '2026-03-17T11:01:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });
    apiClient.bulkRelaunchInboxFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { item_id: 'inbox-relaunch-1', ok: true, follow_up_job_id: 'job-new-1' },
        { item_id: 'inbox-relaunch-2', ok: true, follow_up_job_id: 'job-new-2' },
      ],
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Failed follow-up A')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('checkbox')[1]);
    fireEvent.click(screen.getAllByRole('checkbox')[2]);
    fireEvent.change(screen.getByPlaceholderText('Bulk follow-up note (optional)'), {
      target: { value: 'Retry both terminal follow-ups' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch Follow-ups' }));

    await waitFor(() => {
      expect(apiClient.bulkRelaunchInboxFollowUp).toHaveBeenCalledWith({
        item_ids: ['inbox-relaunch-1', 'inbox-relaunch-2'],
        operator_note: 'Retry both terminal follow-ups',
      });
    });
  });

  it('disables inbox bulk relaunch for mixed terminal and pending follow-up selections', async () => {
    apiClient.listResearchInboxItems.mockResolvedValueOnce({
      items: [
        {
          id: 'inbox-mixed-relaunch-1',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-mixed-relaunch-1',
          title: 'Failed follow-up',
          summary: 'Relaunchable.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-mixed-relaunch-2',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-mixed-relaunch-2',
          title: 'Pending follow-up',
          summary: 'Needs approval.',
          url: null,
          published_at: null,
          discovered_at: '2026-03-17T11:01:00Z',
          status: 'accepted',
          feedback: null,
          metadata: null,
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:01:00Z',
          updated_at: '2026-03-17T11:01:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Inbox'));
    expect(await screen.findByText('Failed follow-up')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('checkbox')[1]);
    fireEvent.click(screen.getAllByRole('checkbox')[2]);

    expect(screen.getByRole('button', { name: 'Relaunch Follow-ups' })).toBeDisabled();
    expect(apiClient.bulkRelaunchInboxFollowUp).not.toHaveBeenCalled();
  });

  it('renders scheduler state in the queue card and job detail panel', async () => {
    const schedulerJob = makeJob({
      id: 'job-scheduler-1',
      name: 'Scheduler Visibility Job',
      status: 'failed',
      schedule_type: 'continuous',
      next_run_at: '2026-03-16T11:00:00Z',
      scheduler_state: {
        last_run_status: 'failed',
        failure_streak: 2,
        last_scheduled_at: '2026-03-16T09:00:00Z',
        last_dispatched_at: '2026-03-16T09:05:00Z',
        current_run_started_at: '2026-03-16T09:06:00Z',
        last_successful_run_at: '2026-03-16T08:00:00Z',
        last_completed_run_at: '2026-03-16T08:01:00Z',
        last_failure_at: '2026-03-16T09:10:00Z',
        backoff_until: '2026-03-16T12:00:00Z',
        backoff_seconds: 1800,
        queue_reason: 'execution_failure',
      },
    });

    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [schedulerJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValue(schedulerJob);

    await renderWithProviders('/autonomous-agents?job=job-scheduler-1');

    expect(await screen.findByText('Scheduler')).toBeInTheDocument();
    expect(screen.getAllByText('Last run failed').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Failure streak 2').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Queue reason execution failure').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Backoff until/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Dispatched/i).length).toBeGreaterThan(0);
  });

  it('hides scheduler state when the payload is malformed', async () => {
    const malformedJob = makeJob({
      id: 'job-scheduler-2',
      name: 'Malformed Scheduler Job',
      status: 'running',
      scheduler_state: 'bad-payload' as any,
    });

    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [malformedJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValue(malformedJob);

    await renderWithProviders('/autonomous-agents?job=job-scheduler-2');

    expect(screen.queryByText('Scheduler')).not.toBeInTheDocument();
    expect(screen.queryByText(/Last run/i)).not.toBeInTheDocument();
  });

  it('ignores late detail-panel loader responses after unmount', async () => {
    const logDeferred = createDeferred<{ entries: any[]; total: number }>();
    const stepEventsDeferred = createDeferred<{ items: any[]; total: number; source: string }>();
    const memoriesDeferred = createDeferred<{ memories: any[]; total: number; page: number; page_size: number }>();
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    apiClient.getAgentJobLog.mockImplementationOnce(() => logDeferred.promise);
    apiClient.getAgentJobStepEvents.mockImplementationOnce(() => stepEventsDeferred.promise);
    apiClient.getJobMemories.mockImplementationOnce(() => memoriesDeferred.promise);

    const view = await renderWithProviders('/autonomous-agents?job=job-1');
    expect(await screen.findByText('Execution Graph')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Show Execution Log'));
    fireEvent.click(screen.getByText('Show Step Events'));
    fireEvent.click(screen.getByText('Show Memories'));

    view.unmount();

    await act(async () => {
      logDeferred.resolve({ entries: [], total: 0 });
      stepEventsDeferred.resolve({ items: [], total: 0, source: 'results.execution_strategy.step_events' });
      memoriesDeferred.resolve({ memories: [], total: 0, page: 1, page_size: 100 });
      await Promise.all([logDeferred.promise, stepEventsDeferred.promise, memoriesDeferred.promise]);
    });

    expect(consoleErrorSpy).not.toHaveBeenCalledWith(
      expect.stringContaining('Warning: An update to JobDetailPanel')
    );

    consoleErrorSpy.mockRestore();
  });

  it('renders unresolved recovery status in the experiment run detail panel', async () => {
    const unresolvedJob = makeJob({
      id: 'job-4',
      name: 'Unresolved Recovery Job',
      experiment_run: {
        source_id: 'repo-999',
        source_name: 'Broken Repo',
        ok: false,
        commands: ['python3 -m pytest -q backend/tests'],
        verification_commands: ['npm --prefix frontend test'],
        bootstrap_commands: ['npm --prefix frontend install'],
        fallback_commands: ['python3 -m pytest -q backend/tests'],
        phases: ['primary', 'bootstrap', 'fallback'],
        final_phase: 'fallback',
        failed_commands: ['npm --prefix frontend test'],
        bootstrap_attempted: true,
        bootstrap_ok: false,
        bootstrap_used: true,
        fallback_attempted: true,
        fallback_ok: false,
        fallback_used: true,
        inferred_project_profile: {
          detected_stack: ['python'],
        },
      },
      results: {
        execution_strategy: {
          execution_graph: {
            graph_health: {
              status: 'critical',
              severity_score: 42,
              blocked_ratio: 0.75,
              reasons: ['fallback verification still failing'],
            },
            dag_stats: {
              total_nodes: 7,
              total_edges: 6,
              critical_path_length: 4,
              blocked_nodes: 3,
              root_nodes: 1,
              leaf_nodes: 2,
              orphan_nodes: 0,
              has_cycle: false,
            },
            verification_actions: [{ id: 'v4' }],
            summarization_actions: [],
            recommended_actions: ['Inspect failing fallback output'],
          },
          scope_observability: {
            resolved_scope_id: 'repo-999',
            scope_source: 'config.source_id',
            events: [
              {
                type: 'resolved_scope',
                source_id: 'repo-999',
                scope_source: 'config.source_id',
              },
            ],
          },
        },
      },
    });
    apiClient.getAgentJob.mockResolvedValueOnce(unresolvedJob);

    await renderWithProviders('/autonomous-agents?job=job-4');

    expect(await screen.findByText('Execution Graph')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Experiment runs'));

    await waitFor(() => {
      expect(screen.getAllByText('Recovery open').length).toBeGreaterThan(0);
      expect(screen.getAllByText(/fallback verification still failing/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/Next Inspect failing fallback output/i)).toBeInTheDocument();
    });
  });

  it('renders autonomy health analytics and filters monitor cards', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));

    await waitFor(() => {
      expect(screen.getAllByText('Acme Monitor').length).toBeGreaterThan(0);
    });
    expect(screen.getByText('Recommendation Performance')).toBeInTheDocument();
    expect(screen.getAllByText('deep_dive_chain').length).toBeGreaterThan(0);
    expect(apiClient.getResearchMonitorAnalytics).toHaveBeenCalled();

    fireEvent.change(screen.getByDisplayValue('All customers'), { target: { value: 'Acme' } });

    await waitFor(() => {
      expect(screen.getAllByText('Acme Monitor').length).toBeGreaterThan(0);
    });
  });

  it('renders customer fleet cards and drills into queue and inbox by customer', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));

    expect(await screen.findByText('Customer Fleet Health')).toBeInTheDocument();
    expect(screen.getAllByText('Acme').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Beta').length).toBeGreaterThan(0);
    expect(screen.getByText('1 monitor(s) are currently throttled.')).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole('button', { name: 'Filter Monitors' })[0]);

    await waitFor(() => {
      expect(screen.getAllByText('Acme Monitor').length).toBeGreaterThan(0);
    });

    fireEvent.click(screen.getAllByRole('button', { name: 'View Queue' })[0]);
    await waitFor(() => {
      expect(apiClient.getAgentCheckpointQueue).toHaveBeenCalledWith(
        expect.objectContaining({ customer: 'Acme' })
      );
    });

    fireEvent.click(screen.getByText('Autonomy Health'));
    fireEvent.click(screen.getAllByRole('button', { name: 'View Inbox' })[0]);

    await waitFor(() => {
      expect(apiClient.listResearchInboxItems).toHaveBeenCalledWith({
        status: 'accepted',
        item_type: undefined,
        customer: 'Acme',
        job_id: undefined,
        q: undefined,
        limit: 100,
        offset: 0,
      });
    });
  });

  it('focuses the top-pressure monitor from a customer fleet card', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    const customerCard = (await screen.findByRole('heading', { name: 'Beta' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(customerCard).getByRole('button', { name: 'Beta Watch' }));

    expect(await screen.findByText('Showing Beta monitors · focused on monitor-2')).toBeInTheDocument();
    const monitorCard = (await screen.findByRole('heading', { name: 'Beta Watch' })).closest('.rounded-lg.p-4') as HTMLElement;
    expect(monitorCard.className).toContain('border-cyan-300');
  });

  it('loads health monitor focus from the url and clears only the focus state', async () => {
    await renderWithProviders('/autonomous-agents?tab=health&health_customer=Beta&health_monitor=monitor-2');

    expect(await screen.findByText('Showing Beta monitors · focused on monitor-2')).toBeInTheDocument();
    const monitorCard = (await screen.findByRole('heading', { name: 'Beta Watch' })).closest('.rounded-lg.p-4') as HTMLElement;
    expect(monitorCard.className).toContain('border-cyan-300');

    fireEvent.click(screen.getByRole('button', { name: 'Clear focus' }));

    await waitFor(() => {
      expect(screen.queryByText('Showing Beta monitors · focused on monitor-2')).not.toBeInTheDocument();
    });
    expect(screen.getByDisplayValue('Beta')).toBeInTheDocument();
  });

  it('drills a monitor failed-outcome metric into accepted inbox follow-ups', async () => {
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-follow-up-failed',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-1',
          title: 'Acme failed follow-up',
          summary: 'Failed outcome',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-follow-up-completed',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-2',
          title: 'Acme completed follow-up',
          summary: 'Completed outcome',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'completed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const monitorCard = (await screen.findByRole('heading', { name: 'Acme Monitor' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(monitorCard).getByRole('button', { name: 'Failed 1' }));

    await waitFor(() => {
      expect(apiClient.listResearchInboxItems).toHaveBeenCalledWith({
        status: 'accepted',
        item_type: undefined,
        customer: 'Acme',
        job_id: 'monitor-1',
        q: undefined,
        limit: 100,
        offset: 0,
      });
    });
    expect(await screen.findByText('Showing accepted follow-ups for Acme · monitor-1 · failed outcomes')).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: 'Acme failed follow-up' })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByRole('heading', { name: 'Acme completed follow-up' })).not.toBeInTheDocument();
    });
  });

  it('drills a customer cancelled-outcome metric into accepted inbox follow-ups', async () => {
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-follow-up-cancelled',
          user_id: 'user-1',
          job_id: 'monitor-2',
          customer: 'Beta',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-3',
          title: 'Beta cancelled follow-up',
          summary: 'Cancelled outcome',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'cancelled',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-follow-up-failed-beta',
          user_id: 'user-1',
          job_id: 'monitor-2',
          customer: 'Beta',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-4',
          title: 'Beta failed follow-up',
          summary: 'Failed outcome',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const customerCard = (await screen.findByRole('heading', { name: 'Beta' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(customerCard).getByRole('button', { name: 'Cancelled 1' }));

    await waitFor(() => {
      expect(apiClient.listResearchInboxItems).toHaveBeenCalledWith({
        status: 'accepted',
        item_type: undefined,
        customer: 'Beta',
        job_id: undefined,
        q: undefined,
        limit: 100,
        offset: 0,
      });
    });
    expect(await screen.findByText('Showing accepted follow-ups for Beta · cancelled outcomes')).toBeInTheDocument();
    expect(await screen.findByText('Beta cancelled follow-up')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText('Beta failed follow-up')).not.toBeInTheDocument();
    });
  });

  it('drills monitor suppressed relaunch pressure into accepted inbox follow-ups', async () => {
    apiClient.listResearchInboxItems.mockResolvedValue({
      items: [
        {
          id: 'inbox-follow-up-suppressed',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-5',
          title: 'Acme suppressed relaunch',
          summary: 'Suppressed relaunch',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          follow_up_operator_decision: 'rejected',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
        {
          id: 'inbox-follow-up-unsuppressed',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-6',
          title: 'Acme ordinary failed relaunch',
          summary: 'Not suppressed',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          follow_up_operator_decision: 'approved_launch',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
    });
    apiClient.getResearchMonitorAnalytics.mockResolvedValue({
      generated_at: '2026-03-17T12:00:00Z',
      totals: {
        total_monitors: 1,
        discovered_count: 2,
        accepted_count: 2,
        rejected_count: 0,
        auto_launched_count: 0,
        approval_launched_count: 0,
        blocked_count: 0,
        follow_up_completed_count: 0,
        follow_up_failed_count: 1,
        follow_up_cancelled_count: 0,
        strong_monitors: 0,
        mixed_monitors: 1,
        weak_monitors: 0,
      },
      customers: [
        {
          customer: 'Acme',
          monitor_count: 1,
          strong_monitor_count: 0,
          mixed_monitor_count: 1,
          weak_monitor_count: 0,
          auto_launch_used_24h: 0,
          auto_launch_capacity_24h: 1,
          approval_queue_used_24h: 0,
          approval_queue_capacity_24h: 1,
          alert_used_24h: 0,
          alert_capacity_24h: 1,
          backlog_used: 0,
          backlog_capacity: 1,
          throttled_monitor_count: 0,
          customer_budget: { auto_launch_limit_24h: 1, approval_queue_limit_24h: 1, alert_limit_24h: 1, queue_backlog_cap: 1 },
          customer_budget_usage: { auto_launch_count_24h: 0, approval_queue_count_24h: 0, alert_count_24h: 0, queue_backlog_count: 0 },
          customer_budget_remaining: { auto_launch_count_24h: 1, approval_queue_count_24h: 1, alert_count_24h: 1, queue_backlog_count: 1 },
          customer_budget_throttle_state: 'normal',
          customer_budget_throttle_reasons: [],
          accepted_count: 2,
          blocked_count: 0,
          follow_up_completed_count: 0,
          follow_up_failed_count: 1,
          follow_up_cancelled_count: 0,
          portfolio_status: 'normal',
          portfolio_reasons: [],
          top_launch_monitors: [],
          top_backlog_monitors: [],
          top_alert_monitors: [],
          throttled_monitors: [],
          rebalance_guidance_status: 'none',
          rebalance_guidance_reasons: [],
          rebalance_guidance_summary: null,
          rebalance_guidance_changes: [],
          latest_rebalance_evaluation_status: undefined,
          latest_rebalance_evaluation_sample_count: 0,
          latest_rebalance_evaluation_target_count: 0,
          latest_rebalance_evaluation_reasons: [],
          recent_rebalance_history: [],
        },
      ],
      monitors: [
        {
          monitor_job_id: 'monitor-1',
          monitor_name: 'Acme Monitor',
          monitor_job_type: 'monitor',
          customer: 'Acme',
          discovered_count: 2,
          accepted_count: 2,
          rejected_count: 0,
          acceptance_rate: 100,
          auto_launched_count: 0,
          approval_launched_count: 0,
          queued_for_approval_count: 0,
          manual_only_count: 0,
          blocked_count: 0,
          follow_up_completed_count: 0,
          follow_up_failed_count: 1,
          follow_up_cancelled_count: 0,
          relaunch_count: 1,
          health_score: 55,
          health_bucket: 'mixed',
          health_reasons: [],
          automation_profile: 'balanced',
          automation_policy: { follow_up_review_mode: 'manual_only', allowed_recommendations: [] },
          effective_policy: { follow_up_review_mode: 'manual_only', allowed_recommendations: [] },
          autonomy_budget: { auto_launch_limit_24h: 1, approval_queue_limit_24h: 1, alert_limit_24h: 1, queue_backlog_cap: 1 },
          budget_usage: { auto_launch_count_24h: 0, approval_queue_count_24h: 0, alert_count_24h: 0, queue_backlog_count: 0 },
          budget_remaining: { auto_launch_count_24h: 1, approval_queue_count_24h: 1, alert_count_24h: 1, queue_backlog_count: 1 },
          budget_throttle_state: 'normal',
          budget_throttle_reasons: [],
          budget_history_count: 0,
          recommended_policy_mode: 'manual_only',
          recommended_allowed_recommendations: [],
          policy_reasons: [],
          policy_confidence: 'low',
          policy_history_count: 0,
          policy_mode_counts: { manual_only: 1 },
          recent_policy_history: [],
          top_recommendations: [],
          scheduler_summary: {
            queued_approvals_count: 0,
            manual_recommendations_count: 0,
            suppressed_relaunches_count: 1,
          },
        },
      ],
      recommendations: [],
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const monitorCard = (await screen.findByRole('heading', { name: 'Acme Monitor' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(monitorCard).getByRole('button', { name: 'Suppressed 1' }));

    await waitFor(() => {
      expect(apiClient.listResearchInboxItems).toHaveBeenCalledWith({
        status: 'accepted',
        item_type: undefined,
        customer: 'Acme',
        job_id: 'monitor-1',
        q: undefined,
        limit: 100,
        offset: 0,
      });
    });
    expect(await screen.findByText('Showing accepted follow-ups for Acme · monitor-1 · suppressed relaunches')).toBeInTheDocument();
    expect(await screen.findByText('Acme suppressed relaunch')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText('Acme ordinary failed relaunch')).not.toBeInTheDocument();
    });
  });

  it('clears the inbox health drilldown context without clearing other inbox filters', async () => {
    apiClient.listResearchInboxItems.mockResolvedValueOnce({
      items: [
        {
          id: 'inbox-follow-up-failed',
          user_id: 'user-1',
          job_id: 'monitor-1',
          customer: 'Acme',
          item_type: 'follow_up_recommendation',
          item_key: 'follow-up-7',
          title: 'Acme failed follow-up',
          summary: 'Failed outcome',
          discovered_at: '2026-03-17T11:00:00Z',
          status: 'accepted',
          follow_up_launch_status: 'launched',
          follow_up_outcome_status: 'failed',
          created_at: '2026-03-17T11:00:00Z',
          updated_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents?tab=inbox&inbox_customer=Acme&inbox_job=monitor-1&inbox_health_drilldown=failed_follow_up');

    expect(await screen.findByText('Showing accepted follow-ups for Acme · monitor-1 · failed outcomes')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear drilldown' }));

    await waitFor(() => {
      expect(screen.queryByText('Showing accepted follow-ups for Acme · monitor-1 · failed outcomes')).not.toBeInTheDocument();
    });
    expect(screen.getByText('Customer filter: Acme')).toBeInTheDocument();
    expect(screen.getByText('Monitor filter: monitor-1')).toBeInTheDocument();
  });

  it('drills a monitor pending-approval metric into checkpoint queue follow-up recommendations', async () => {
    apiClient.getResearchMonitorAnalytics.mockResolvedValueOnce({
      generated_at: '2026-03-17T12:00:00Z',
      totals: {
        total_monitors: 1,
        discovered_count: 4,
        accepted_count: 2,
        rejected_count: 0,
        auto_launched_count: 0,
        approval_launched_count: 0,
        blocked_count: 1,
        follow_up_completed_count: 0,
        follow_up_failed_count: 0,
        follow_up_cancelled_count: 0,
        strong_monitors: 0,
        mixed_monitors: 1,
        weak_monitors: 0,
      },
      customers: [
        {
          customer: 'Acme',
          monitor_count: 1,
          strong_monitor_count: 0,
          mixed_monitor_count: 1,
          weak_monitor_count: 0,
          auto_launch_used_24h: 0,
          auto_launch_capacity_24h: 1,
          approval_queue_used_24h: 1,
          approval_queue_capacity_24h: 3,
          alert_used_24h: 0,
          alert_capacity_24h: 1,
          backlog_used: 0,
          backlog_capacity: 2,
          throttled_monitor_count: 0,
          customer_budget: { auto_launch_limit_24h: 1, approval_queue_limit_24h: 3, alert_limit_24h: 1, queue_backlog_cap: 2 },
          customer_budget_usage: { auto_launch_count_24h: 0, approval_queue_count_24h: 1, alert_count_24h: 0, queue_backlog_count: 0 },
          customer_budget_remaining: { auto_launch_count_24h: 1, approval_queue_count_24h: 2, alert_count_24h: 1, queue_backlog_count: 2 },
          customer_budget_throttle_state: 'normal',
          customer_budget_throttle_reasons: [],
          accepted_count: 2,
          blocked_count: 1,
          follow_up_completed_count: 0,
          follow_up_failed_count: 0,
          follow_up_cancelled_count: 0,
          portfolio_status: 'normal',
          portfolio_reasons: [],
          top_launch_monitors: [],
          top_backlog_monitors: [],
          top_alert_monitors: [],
          throttled_monitors: [],
          rebalance_guidance_status: 'none',
          rebalance_guidance_reasons: [],
          rebalance_guidance_summary: null,
          rebalance_guidance_changes: [],
          latest_rebalance_evaluation_status: undefined,
          latest_rebalance_evaluation_sample_count: 0,
          latest_rebalance_evaluation_target_count: 0,
          latest_rebalance_evaluation_reasons: [],
          recent_rebalance_history: [],
        },
      ],
      monitors: [
        {
          monitor_job_id: 'monitor-1',
          monitor_name: 'Acme Monitor',
          monitor_job_type: 'monitor',
          customer: 'Acme',
          discovered_count: 4,
          accepted_count: 2,
          rejected_count: 0,
          acceptance_rate: 50,
          auto_launched_count: 0,
          approval_launched_count: 0,
          queued_for_approval_count: 1,
          manual_only_count: 0,
          blocked_count: 1,
          follow_up_completed_count: 0,
          follow_up_failed_count: 0,
          follow_up_cancelled_count: 0,
          relaunch_count: 0,
          health_score: 55,
          health_bucket: 'mixed',
          health_reasons: [],
          automation_profile: 'balanced',
          automation_policy: { follow_up_review_mode: 'queue_for_approval', allowed_recommendations: [] },
          effective_policy: { follow_up_review_mode: 'queue_for_approval', allowed_recommendations: [] },
          autonomy_budget: { auto_launch_limit_24h: 1, approval_queue_limit_24h: 3, alert_limit_24h: 1, queue_backlog_cap: 2 },
          budget_usage: { auto_launch_count_24h: 0, approval_queue_count_24h: 1, alert_count_24h: 0, queue_backlog_count: 0 },
          budget_remaining: { auto_launch_count_24h: 1, approval_queue_count_24h: 2, alert_count_24h: 1, queue_backlog_count: 2 },
          budget_throttle_state: 'normal',
          budget_throttle_reasons: [],
          budget_history_count: 0,
          recommended_policy_mode: 'queue_for_approval',
          recommended_allowed_recommendations: [],
          policy_reasons: [],
          policy_confidence: 'medium',
          policy_history_count: 0,
          policy_mode_counts: { queue_for_approval: 1 },
          recent_policy_history: [],
          top_recommendations: [],
          scheduler_summary: {
            queued_approvals_count: 1,
            manual_recommendations_count: 0,
            suppressed_relaunches_count: 0,
          },
        },
      ],
      recommendations: [],
    });
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'queue-pending-match',
          item_type: 'follow_up_recommendation',
          title: 'Acme pending follow-up',
          summary: 'Needs approval',
          customer: 'Acme',
          job_id: 'monitor-1',
          status: 'pending_approval',
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:00:00Z',
        },
        {
          queue_key: 'queue-pending-other-monitor',
          item_type: 'follow_up_recommendation',
          title: 'Acme other monitor follow-up',
          summary: 'Other monitor',
          customer: 'Acme',
          job_id: 'monitor-2',
          status: 'pending_approval',
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { pending_approval: 2 },
      by_customer: { Acme: 2 },
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const monitorCard = (await screen.findByRole('heading', { name: 'Acme Monitor' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(monitorCard).getByRole('button', { name: 'Queue 1' }));

    await waitFor(() => {
      expect(apiClient.getAgentCheckpointQueue).toHaveBeenCalledWith(
        expect.objectContaining({ item_type: 'follow_up_recommendation', customer: 'Acme' })
      );
    });
    expect(await screen.findByText('Showing follow-up recommendations for Acme · monitor-1 · pending approvals')).toBeInTheDocument();
    expect(await screen.findByText('Acme pending follow-up')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText('Acme other monitor follow-up')).not.toBeInTheDocument();
    });
  });

  it('drills a customer manual metric into checkpoint queue follow-up recommendations', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'queue-manual-match',
          item_type: 'follow_up_recommendation',
          title: 'Beta manual follow-up',
          summary: 'Manual recommendation',
          customer: 'Beta',
          job_id: 'monitor-2',
          status: 'blocked',
          follow_up_launch_status: 'blocked',
          follow_up_decision: 'manual_only',
          created_at: '2026-03-17T11:00:00Z',
        },
        {
          queue_key: 'queue-manual-other',
          item_type: 'follow_up_recommendation',
          title: 'Beta queued approval follow-up',
          summary: 'Pending approval',
          customer: 'Beta',
          job_id: 'monitor-2',
          status: 'pending_approval',
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { blocked: 1, pending_approval: 1 },
      by_customer: { Beta: 2 },
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const customerCard = (await screen.findByRole('heading', { name: 'Beta' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(customerCard).getByRole('button', { name: 'Manual 2' }));

    await waitFor(() => {
      expect(apiClient.getAgentCheckpointQueue).toHaveBeenCalledWith(
        expect.objectContaining({ item_type: 'follow_up_recommendation', customer: 'Beta' })
      );
    });
    expect(await screen.findByText('Showing follow-up recommendations for Beta · manual recommendations')).toBeInTheDocument();
    expect(await screen.findByText('Beta manual follow-up')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText('Beta queued approval follow-up')).not.toBeInTheDocument();
    });
  });

  it('drills a monitor blocked metric into checkpoint queue follow-up recommendations', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'queue-blocked-match',
          item_type: 'follow_up_recommendation',
          title: 'Beta blocked follow-up',
          summary: 'Blocked by policy',
          customer: 'Beta',
          job_id: 'monitor-2',
          status: 'blocked',
          reason_code: 'follow_up_blocked',
          follow_up_launch_status: 'blocked',
          follow_up_block_reason: 'Monitor policy is set to manual follow-up launches.',
          created_at: '2026-03-17T11:00:00Z',
        },
        {
          queue_key: 'queue-blocked-other',
          item_type: 'follow_up_recommendation',
          title: 'Beta pending follow-up',
          summary: 'Pending approval',
          customer: 'Beta',
          job_id: 'monitor-2',
          status: 'pending_approval',
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { blocked: 1, pending_approval: 1 },
      by_customer: { Beta: 2 },
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const monitorCard = (await screen.findByRole('heading', { name: 'Beta Watch' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(monitorCard).getByRole('button', { name: 'Blocked 2' }));

    await waitFor(() => {
      expect(apiClient.getAgentCheckpointQueue).toHaveBeenCalledWith(
        expect.objectContaining({ item_type: 'follow_up_recommendation', customer: 'Beta' })
      );
    });
    expect(await screen.findByText('Showing follow-up recommendations for Beta · monitor-2 · blocked follow-ups')).toBeInTheDocument();
    expect(await screen.findByText('Beta blocked follow-up')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText('Beta pending follow-up')).not.toBeInTheDocument();
    });
  });

  it('drills a customer blocked metric into checkpoint queue follow-up recommendations', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'queue-customer-blocked-match',
          item_type: 'follow_up_recommendation',
          title: 'Beta blocked customer follow-up',
          summary: 'Blocked by customer budget',
          customer: 'Beta',
          job_id: 'monitor-2',
          status: 'blocked',
          reason_code: 'follow_up_blocked',
          follow_up_launch_status: 'blocked',
          follow_up_block_reason: 'Customer autonomy budget is currently exhausted.',
          created_at: '2026-03-17T11:00:00Z',
        },
        {
          queue_key: 'queue-customer-pending-other',
          item_type: 'follow_up_recommendation',
          title: 'Beta pending customer follow-up',
          summary: 'Pending approval',
          customer: 'Beta',
          job_id: 'monitor-2',
          status: 'pending_approval',
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { blocked: 1, pending_approval: 1 },
      by_customer: { Beta: 2 },
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByRole('button', { name: 'Autonomy Health' }));
    const customerCard = (await screen.findByRole('heading', { name: 'Beta' })).closest('.border.rounded-lg.p-4') as HTMLElement;
    fireEvent.click(within(customerCard).getByRole('button', { name: 'Blocked 2' }));

    await waitFor(() => {
      expect(apiClient.getAgentCheckpointQueue).toHaveBeenCalledWith(
        expect.objectContaining({ item_type: 'follow_up_recommendation', customer: 'Beta' })
      );
    });
    expect(await screen.findByText('Showing follow-up recommendations for Beta · blocked follow-ups')).toBeInTheDocument();
    expect(await screen.findByText('Beta blocked customer follow-up')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText('Beta pending customer follow-up')).not.toBeInTheDocument();
    });
  });

  it('clears the queue health drilldown context without clearing owner filters', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'queue-clear-pending',
          item_type: 'follow_up_recommendation',
          title: 'Acme pending follow-up',
          summary: 'Needs approval',
          customer: 'Acme',
          job_id: 'monitor-1',
          status: 'pending_approval',
          follow_up_launch_status: 'pending_approval',
          created_at: '2026-03-17T11:00:00Z',
        },
        {
          queue_key: 'queue-clear-manual',
          item_type: 'follow_up_recommendation',
          title: 'Acme manual follow-up',
          summary: 'Manual review',
          customer: 'Acme',
          job_id: 'monitor-1',
          status: 'blocked',
          follow_up_launch_status: 'blocked',
          follow_up_decision: 'manual_only',
          created_at: '2026-03-17T11:00:00Z',
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { blocked: 1, pending_approval: 1 },
      by_customer: { Acme: 2 },
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents?tab=queue&queue_item_type=follow_up_recommendation&queue_customer=Acme&queue_job=monitor-1&queue_health_drilldown=pending_follow_up_approvals');

    expect(await screen.findByText('Showing follow-up recommendations for Acme · monitor-1 · pending approvals')).toBeInTheDocument();
    expect(await screen.findByText('Acme pending follow-up')).toBeInTheDocument();
    expect(screen.queryByText('Acme manual follow-up')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Clear drilldown' }));

    await waitFor(() => {
      expect(screen.queryByText('Showing follow-up recommendations for Acme · monitor-1 · pending approvals')).not.toBeInTheDocument();
    });
    expect(await screen.findByText('Acme pending follow-up')).toBeInTheDocument();
    expect(await screen.findByText('Acme manual follow-up')).toBeInTheDocument();
  });

  it('updates shared customer budget caps from autonomy health', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect(await screen.findByText('Customer Fleet Health')).toBeInTheDocument();

    const autoLaunchInputs = screen.getAllByDisplayValue('6');
    fireEvent.change(autoLaunchInputs[0], { target: { value: '7' } });
    fireEvent.click(screen.getAllByRole('button', { name: 'Save shared caps' })[0]);

    await waitFor(() => {
      expect(apiClient.updateResearchMonitorCustomerBudget).toHaveBeenCalledWith({
        customer: 'Acme',
        auto_launch_limit_24h: 7,
        approval_queue_limit_24h: 10,
        alert_limit_24h: 5,
        queue_backlog_cap: 12,
      });
    });
  });

  it('previews and applies customer rebalance guidance from autonomy health', async () => {
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect(await screen.findByText('Rebalance guidance')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Preview rebalance' }));

    await waitFor(() => {
      expect(apiClient.previewResearchMonitorCustomerRebalance).toHaveBeenCalledWith({
        customer: 'Beta',
        monitor_budget_updates: [
          {
            monitor_job_id: 'monitor-2',
            auto_launch_limit_24h: 2,
            approval_queue_limit_24h: 5,
            alert_limit_24h: 3,
            queue_backlog_cap: 7,
          },
          {
            monitor_job_id: 'monitor-1',
            auto_launch_limit_24h: 4,
            approval_queue_limit_24h: 7,
            alert_limit_24h: 5,
            queue_backlog_cap: 9,
          },
        ],
      });
    });

    fireEvent.click(screen.getByRole('button', { name: 'Apply rebalance' }));

    await waitFor(() => {
      expect(apiClient.applyResearchMonitorCustomerRebalance).toHaveBeenCalledWith({
        customer: 'Beta',
        monitor_budget_updates: [
          {
            monitor_job_id: 'monitor-2',
            auto_launch_limit_24h: 2,
            approval_queue_limit_24h: 5,
            alert_limit_24h: 3,
            queue_backlog_cap: 7,
          },
          {
            monitor_job_id: 'monitor-1',
            auto_launch_limit_24h: 4,
            approval_queue_limit_24h: 7,
            alert_limit_24h: 5,
            queue_backlog_cap: 9,
          },
        ],
        change_reason: 'Shift budget headroom from Beta Watch to Acme Monitor.',
      });
    });

    confirmSpy.mockRestore();
  });

  it('loads customer rebalance outcome details from autonomy health', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect((await screen.findAllByText('Rebalance history')).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole('button', { name: 'Compare outcome' }));

    await waitFor(() => {
      expect(apiClient.getResearchMonitorCustomerRebalanceEvaluation).toHaveBeenCalledWith('Beta', 'rebalance-1');
    });
    expect(screen.getAllByText(/Queue backlog pressure declined/i).length).toBeGreaterThan(0);
  });

  it('applies a guided policy update from autonomy health', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect((await screen.findAllByText('Recommended policy')).length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByText('Use recommendation')[0]);
    fireEvent.click(screen.getAllByText('Apply policy')[0]);

    await waitFor(() => {
      expect(apiClient.updateResearchMonitorPolicy).toHaveBeenCalledWith('monitor-1', {
        automation_profile: 'balanced',
        automation_policy: {
          follow_up_review_mode: 'auto_launch_safe',
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        mode: 'auto_launch_safe',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        change_source: 'guided_recommendation',
        analytics_context: {
          health_bucket: 'strong',
          policy_confidence: 'high',
          accepted_count: 6,
          blocked_count: 0,
          follow_up_completed_count: 3,
          follow_up_failed_count: 1,
          follow_up_cancelled_count: 0,
        },
      });
    });
  });

  it('shows policy history and rolls a monitor back from autonomy health', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect((await screen.findAllByText('Policy history')).length).toBeGreaterThan(0);
    expect(screen.getByText('Strong autonomy health')).toBeInTheDocument();

    const rollbackButton = screen
      .getAllByRole('button', { name: 'Roll back' })
      .find((button) => !(button as HTMLButtonElement).disabled);
    expect(rollbackButton).toBeDefined();
    fireEvent.click(rollbackButton as HTMLButtonElement);

    await waitFor(() => {
      expect(apiClient.rollbackResearchMonitorPolicy).toHaveBeenCalledWith('monitor-1', {
        history_entry_id: 'history-0',
      });
    });
  });

  it('applies a degrading-policy safeguard from autonomy health', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect(await screen.findByText('Policy safeguard recommended')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Apply safeguard' }));

    await waitFor(() => {
      expect(apiClient.rollbackResearchMonitorPolicy).toHaveBeenCalledWith('monitor-2', {
        history_entry_id: 'history-2',
      });
    });
  });

  it('loads before and after policy evaluation details from policy history', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect((await screen.findAllByText('Policy history')).length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByRole('button', { name: 'Compare before/after' })[0]);

    await waitFor(() => {
      expect(apiClient.getResearchMonitorPolicyEvaluation).toHaveBeenCalledWith('monitor-1', 'history-1');
    });

    expect(await screen.findByText('4/8 accepted signals after rollout')).toBeInTheDocument();
    expect(screen.getByText(/Completed 1 · Failed 1 · Blocked 1/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Completed \+2 · Failed 0 · Blocked -1/i).length).toBeGreaterThan(0);
    expect(screen.getByText('Before launch gap')).toBeInTheDocument();
    expect(screen.getByText('After safe launch')).toBeInTheDocument();

    fireEvent.click(screen.getAllByText('Open in Inbox')[0]);

    await waitFor(() => {
      expect(apiClient.listResearchInboxItems).toHaveBeenCalledWith({
        status: 'accepted',
        item_type: undefined,
        job_id: 'monitor-1',
        q: undefined,
        limit: 100,
        offset: 0,
      });
    });

    expect(await screen.findByText('Showing accepted signals for monitor-1 · post-rollout evaluation')).toBeInTheDocument();
  });

  it('previews policy impact from autonomy health', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect((await screen.findAllByText('Use recommendation')).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByText('Use recommendation')[0]);
    fireEvent.click(screen.getAllByText('Preview impact')[0]);

    await waitFor(() => {
      expect(apiClient.simulateResearchMonitorPolicy).toHaveBeenCalledWith('monitor-1', {
        automation_profile: 'balanced',
        automation_policy: {
          follow_up_review_mode: 'auto_launch_safe',
          allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        },
        mode: 'auto_launch_safe',
        allowed_recommendations: ['deep_dive_chain', 'single_research_job'],
        history_limit: 25,
      });
    });

    expect(await screen.findByText('Policy impact preview')).toBeInTheDocument();
    expect(screen.getByText(/Current 0 -> Proposed 3/i)).toBeInTheDocument();
    expect(screen.getByText(/Accepted signal/i)).toBeInTheDocument();
  });

  it('opens previewed signals in the inbox with the monitor filter and policy drilldown applied', async () => {
    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Autonomy Health'));
    expect((await screen.findAllByText('Use recommendation')).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByText('Use recommendation')[0]);
    fireEvent.click(screen.getAllByText('Preview impact')[0]);
    expect(await screen.findByText('Policy impact preview')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Open in Inbox'));

    await waitFor(() => {
      expect(apiClient.listResearchInboxItems).toHaveBeenCalledWith({
        status: 'accepted',
        item_type: undefined,
        job_id: 'monitor-1',
        q: undefined,
        limit: 100,
        offset: 0,
      });
    });

    expect(await screen.findByText('Showing accepted signals for monitor-1 · simulated policy impact')).toBeInTheDocument();
    expect(await screen.findByText('Monitor filter: monitor-1')).toBeInTheDocument();
    expect(screen.getByText('Monitor filter: monitor-1')).toBeInTheDocument();
    expect(screen.getByText('Accepted signal')).toBeInTheDocument();
  });

  it('clears the inbox policy drilldown context without clearing the monitor filter', async () => {
    await renderWithProviders('/autonomous-agents?tab=inbox&inbox_job=monitor-1&inbox_policy_drilldown=simulated_policy_impact');

    expect(await screen.findByText('Showing accepted signals for monitor-1 · simulated policy impact')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear drilldown' }));

    await waitFor(() => {
      expect(screen.queryByText('Showing accepted signals for monitor-1 · simulated policy impact')).not.toBeInTheDocument();
    });
    expect(screen.getByText('Monitor filter: monitor-1')).toBeInTheDocument();
  });

  it('bulk-approves queue follow-up recommendations for one domain profile', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'follow-up:profile-1:opp-a',
          item_type: 'follow_up_recommendation',
          priority: 90,
          title: 'Queue compiler follow-up A',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-profile-1',
          domain_research_profile_id: 'profile-1',
          domain_research_profile_title: 'Compiler Frontier',
          profile_opportunity_id: 'opp-a',
          profile_opportunity_key: 'compiler-a',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
        {
          queue_key: 'follow-up:profile-1:opp-b',
          item_type: 'follow_up_recommendation',
          priority: 85,
          title: 'Queue compiler follow-up B',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-profile-1',
          domain_research_profile_id: 'profile-1',
          domain_research_profile_title: 'Compiler Frontier',
          profile_opportunity_id: 'opp-b',
          profile_opportunity_key: 'compiler-b',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { pending_approval: 2 },
      by_customer: {},
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-a', ok: true, follow_up_launch_status: 'launched' },
        { domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-b', ok: true, follow_up_launch_status: 'launched' },
      ],
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Checkpoint Queue'));
    expect(await screen.findByText('Queue compiler follow-up A')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Select queue item Queue compiler follow-up A'));
    fireEvent.click(screen.getByLabelText('Select queue item Queue compiler follow-up B'));
    fireEvent.change(screen.getByPlaceholderText('Shared note for selected follow-ups'), {
      target: { value: 'Launch the safe compiler follow-ups' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

    await waitFor(() => {
      expect(apiClient.bulkActionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: 'profile-1',
        profile_opportunity_ids: ['opp-a', 'opp-b'],
        portfolio_id: undefined,
        portfolio_opportunity_ids: undefined,
        action: 'approve_launch',
        operator_note: 'Launch the safe compiler follow-ups',
      });
    });
  });

  it('bulk-rejects queue follow-up recommendations for one research fleet', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'follow-up:portfolio-1:opp-a',
          item_type: 'follow_up_recommendation',
          priority: 90,
          title: 'Queue fleet follow-up A',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-portfolio-1',
          portfolio_id: 'portfolio-1',
          portfolio_title: 'Scientific Fleet',
          portfolio_opportunity_id: 'opp-a',
          portfolio_opportunity_key: 'fleet-a',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
        {
          queue_key: 'follow-up:portfolio-1:opp-b',
          item_type: 'follow_up_recommendation',
          priority: 85,
          title: 'Queue fleet follow-up B',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-portfolio-1',
          portfolio_id: 'portfolio-1',
          portfolio_title: 'Scientific Fleet',
          portfolio_opportunity_id: 'opp-b',
          portfolio_opportunity_key: 'fleet-b',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { pending_approval: 2 },
      by_customer: {},
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { portfolio_id: 'portfolio-1', portfolio_opportunity_id: 'opp-a', ok: true, follow_up_launch_status: 'rejected' },
        { portfolio_id: 'portfolio-1', portfolio_opportunity_id: 'opp-b', ok: true, follow_up_launch_status: 'rejected' },
      ],
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Checkpoint Queue'));
    expect(await screen.findByText('Queue fleet follow-up A')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Select queue item Queue fleet follow-up A'));
    fireEvent.click(screen.getByLabelText('Select queue item Queue fleet follow-up B'));
    fireEvent.change(screen.getByPlaceholderText('Shared note for selected follow-ups'), {
      target: { value: 'Hold until more evidence arrives' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Reject selected' }));

    await waitFor(() => {
      expect(apiClient.bulkActionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: undefined,
        profile_opportunity_ids: undefined,
        portfolio_id: 'portfolio-1',
        portfolio_opportunity_ids: ['opp-a', 'opp-b'],
        action: 'reject_launch',
        operator_note: 'Hold until more evidence arrives',
      });
    });
  });

  it('disables queue bulk follow-up actions for mixed domain and fleet selections', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'follow-up:profile-1:opp-a',
          item_type: 'follow_up_recommendation',
          priority: 90,
          title: 'Queue compiler follow-up',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-profile-1',
          domain_research_profile_id: 'profile-1',
          profile_opportunity_id: 'opp-a',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
        {
          queue_key: 'follow-up:portfolio-1:opp-b',
          item_type: 'follow_up_recommendation',
          priority: 85,
          title: 'Queue fleet follow-up',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-portfolio-1',
          portfolio_id: 'portfolio-1',
          portfolio_opportunity_id: 'opp-b',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { pending_approval: 2 },
      by_customer: {},
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Checkpoint Queue'));
    expect(await screen.findByText('Queue compiler follow-up')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Select queue item Queue compiler follow-up'));
    fireEvent.click(screen.getByLabelText('Select queue item Queue fleet follow-up'));

    expect(screen.getByText('Bulk follow-up actions cannot mix domain and fleet owners.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Approve selected' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reject selected' })).not.toBeInTheDocument();
  });

  it('keeps failed queue follow-up selections selected after a partial bulk result', async () => {
    const queuePayload = {
      items: [
        {
          queue_key: 'follow-up:profile-1:opp-a',
          item_type: 'follow_up_recommendation',
          priority: 90,
          title: 'Queue compiler follow-up A',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-profile-1',
          domain_research_profile_id: 'profile-1',
          profile_opportunity_id: 'opp-a',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
        {
          queue_key: 'follow-up:profile-1:opp-b',
          item_type: 'follow_up_recommendation',
          priority: 85,
          title: 'Queue compiler follow-up B',
          summary: 'Queued for approval',
          status: 'pending_approval',
          job_id: 'job-profile-1',
          domain_research_profile_id: 'profile-1',
          profile_opportunity_id: 'opp-b',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { pending_approval: 2 },
      by_customer: {},
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    };
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce(queuePayload);
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce(queuePayload);
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 1,
      failed: 1,
      results: [
        { domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-a', ok: true, follow_up_launch_status: 'launched' },
        { domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-b', ok: false, error: 'Already reviewed' },
      ],
    });

    await renderWithProviders('/autonomous-agents');

    fireEvent.click(await screen.findByText('Checkpoint Queue'));
    expect(await screen.findByText('Queue compiler follow-up A')).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText('Select queue item Queue compiler follow-up A'));
    fireEvent.click(screen.getByLabelText('Select queue item Queue compiler follow-up B'));
    fireEvent.click(screen.getByRole('button', { name: 'Approve selected' }));

    await waitFor(() => {
      expect((screen.getByLabelText('Select queue item Queue compiler follow-up A') as HTMLInputElement).checked).toBe(false);
      expect((screen.getByLabelText('Select queue item Queue compiler follow-up B') as HTMLInputElement).checked).toBe(true);
    });
  });

  it('shows compiler queue context and opens the exact domain opportunity from queue', async () => {
    apiClient.getAgentCheckpointQueue.mockResolvedValueOnce({
      items: [
        {
          queue_key: 'follow-up:profile-1:opp-target',
          item_type: 'follow_up_recommendation',
          priority: 90,
          title: 'Compiler approval target',
          summary: 'Queued compiler follow-up',
          status: 'pending_approval',
          job_id: 'job-domain-1',
          domain_research_profile_id: 'profile-1',
          domain_research_profile_title: 'Compiler Frontier',
          profile_opportunity_id: 'opp-target',
          profile_opportunity_key: 'compiler-hotspot',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          track_type: 'compiler',
          source_scope: 'kb_plus_arxiv_plus_repo',
          repo_source_ids: ['repo-source-1'],
          benchmark_queries: ['llvm-test-suite'],
          sandbox_profile_id: 'scientific-compiler-sandbox',
          automation_profile: 'balanced',
          effective_policy: { follow_up_review_mode: 'queue_for_approval' },
          confidence: 0.82,
          readiness: 0.77,
          linked_note_ids: ['note-1'],
          linked_experiment_plan_ids: ['plan-1'],
          linked_validation_run_ids: ['run-1'],
          child_job_ids: ['job-child-1'],
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
        {
          queue_key: 'follow-up:profile-2:opp-generic',
          item_type: 'follow_up_recommendation',
          priority: 80,
          title: 'Generic queue item',
          summary: 'Generic follow-up',
          status: 'pending_approval',
          job_id: 'job-domain-2',
          domain_research_profile_id: 'profile-2',
          domain_research_profile_title: 'Retrieval Frontier',
          profile_opportunity_id: 'opp-generic',
          track_type: 'generic',
          follow_up_launch_status: 'pending_approval',
          actions: [],
        },
      ],
      total: 2,
      approvals: 0,
      recoveries: 0,
      follow_ups: 2,
      by_type: { follow_up_recommendation: 2 },
      by_status: { pending_approval: 2 },
      by_customer: {},
      by_sla_bucket: {},
      by_escalation_level: {},
      limit: 100,
      offset: 0,
    });
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'balanced',
          effective_policy: { follow_up_review_mode: 'queue_for_approval' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 1,
              manual_follow_up_recommendations_count: 0,
            },
          },
          opportunities: [
            {
              opportunity_id: 'opp-target',
              title: 'Target domain opportunity',
              stage: 'planned',
              confidence: 0.82,
              novelty: 0.74,
              readiness: 0.77,
              autonomy_state: 'eligible',
              linked_experiment_plan_ids: ['plan-1'],
              linked_validation_run_ids: ['run-1'],
              child_job_ids: ['job-child-1'],
            },
          ],
          latest_note_ids: ['note-1'],
          latest_experiment_plan_ids: ['plan-1'],
          latest_validation_run_ids: ['run-1'],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents?tab=queue', { documentSources: defaultDocumentSources });

    expect(await screen.findByText('Compiler approval target')).toBeInTheDocument();
    expect(screen.getByText('Domain: Compiler')).toBeInTheDocument();
    expect(screen.getByText('Objective: Track compiler opportunities')).toBeInTheDocument();
    expect(screen.getByText('Track: compiler')).toBeInTheDocument();
    expect(screen.getByText('Source scope: kb plus arxiv plus repo')).toBeInTheDocument();
    expect(screen.getByText('Repo inputs: repo-source-1')).toBeInTheDocument();
    expect(screen.getByText('Benchmarks: llvm-test-suite')).toBeInTheDocument();
    expect(screen.getByText('Sandbox: scientific-compiler-sandbox')).toBeInTheDocument();
    expect(screen.getByText('Review mode: queue for approval')).toBeInTheDocument();
    expect(screen.getByText('Links:')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Compiler only' }));
    await waitFor(() => {
      expect(screen.queryByText('Generic queue item')).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Open Domain' }));

    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    const row = await screen.findByText('Target domain opportunity');
    expect(row.closest('div.border')?.className).toContain('border-cyan-300');
  });

  it('offers recovery actions and can restart an unresolved recovery run', async () => {
    apiClient.saveAgentJobAsChain.mockResolvedValueOnce({
      id: 'chain-recovery-1',
      name: 'playbook_recovery_unresolved_recovery_job',
      display_name: 'Unresolved Recovery Job (Recovery Playbook)',
      description: 'Saved from recovery job job-4. Recovery reason: Execution failure.',
      chain_steps: [
        {
          step_name: 'Step 1',
          job_type: 'analysis',
          goal_template: 'Investigate the unresolved recovery run',
          trigger_condition: 'on_complete',
        },
      ],
      default_settings: { inherit_results: true },
      owner_user_id: 'user-1',
      is_system: false,
      is_active: true,
      created_at: '2026-03-10T00:00:00Z',
      updated_at: '2026-03-10T00:00:00Z',
    });

    const unresolvedJob = makeJob({
      id: 'job-4',
      status: 'failed',
      launch_mode: 'quick_start_claude_backend',
      name: 'Unresolved Recovery Job',
      experiment_run: {
        source_id: 'repo-999',
        source_name: 'Broken Repo',
        ok: false,
        commands: ['python3 -m pytest -q backend/tests'],
        verification_commands: ['npm --prefix frontend test'],
        bootstrap_commands: ['npm --prefix frontend install'],
        fallback_commands: ['python3 -m pytest -q backend/tests'],
        phases: ['primary', 'bootstrap', 'fallback'],
        final_phase: 'fallback',
        failed_commands: ['npm --prefix frontend test'],
        bootstrap_attempted: true,
        bootstrap_ok: false,
        bootstrap_used: true,
        fallback_attempted: true,
        fallback_ok: false,
        fallback_used: true,
        inferred_project_profile: {
          detected_stack: ['python'],
        },
      },
      results: {
        execution_strategy: {
          execution_graph: {
            graph_health: {
              status: 'critical',
              severity_score: 42,
              blocked_ratio: 0.75,
              reasons: ['fallback verification still failing'],
            },
            dag_stats: {
              total_nodes: 7,
              total_edges: 6,
              critical_path_length: 4,
              blocked_nodes: 3,
              root_nodes: 1,
              leaf_nodes: 2,
              orphan_nodes: 0,
              has_cycle: false,
            },
            verification_actions: [{ id: 'v4' }],
            summarization_actions: [],
            recommended_actions: ['Inspect failing fallback output'],
          },
        },
      },
    });
    apiClient.getAgentJob.mockResolvedValueOnce(unresolvedJob);
    apiClient.performAgentJobAction.mockResolvedValueOnce({
      ...unresolvedJob,
      status: 'pending',
    });

    await renderWithProviders('/autonomous-agents?job=job-4');

    expect(await screen.findByText('Execution Graph')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Experiment runs'));

    await waitFor(() => {
      expect(screen.getByText('Restart recovery')).toBeInTheDocument();
      expect(screen.getByText('Relaunch clean run')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Save recovery playbook' })).toBeInTheDocument();
      expect(screen.getAllByText('Copy failed command').length).toBeGreaterThan(0);
      expect(screen.getByText('Copy next step')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Save recovery playbook' }));
    await waitFor(() => {
      expect(apiClient.saveAgentJobAsChain).toHaveBeenCalledWith(
        'job-4',
        expect.objectContaining({
          name: 'playbook_recovery_unresolved_recovery_job',
          display_name: 'Unresolved Recovery Job (Recovery Playbook)',
        })
      );
    });

    fireEvent.click(screen.getByText('Restart recovery'));

    await waitFor(() => {
      expect(apiClient.performAgentJobAction).toHaveBeenCalledWith('job-4', 'restart', expect.any(Object));
    });
  });

  it('sorts recovery playbooks ahead of generic chains in the chain picker', async () => {
    apiClient.listChainDefinitions.mockResolvedValue({
      total: 2,
      chains: [
        {
          id: 'chain-generic-1',
          name: 'playbook_code_patch_20260408_120000',
          display_name: 'Code Patch Playbook',
          description: 'Saved from job job-1 on 2026-04-08T12:00:00Z.',
          chain_steps: [
            {
              step_name: 'Step 1',
              job_type: 'analysis',
              goal_template: 'Investigate',
              trigger_condition: 'on_complete',
            },
          ],
          default_settings: { inherit_results: true },
          owner_user_id: 'user-1',
          is_system: false,
          is_active: true,
          created_at: '2026-04-08T12:00:00Z',
          updated_at: '2026-04-08T12:00:00Z',
        },
        {
          id: 'chain-recovery-1',
          name: 'playbook_recovery_failed_job_20260408_120100',
          display_name: 'Failed Job (Recovery Playbook)',
          description: 'Saved from recovery job job-4. Saved as a recovery playbook. Recovery reason: Execution failure.',
          chain_steps: [
            {
              step_name: 'Step 1',
              job_type: 'analysis',
              goal_template: 'Recover',
              trigger_condition: 'on_complete',
            },
          ],
          default_settings: { inherit_results: true },
          owner_user_id: 'user-1',
          is_system: false,
          is_active: true,
          created_at: '2026-04-08T12:01:00Z',
          updated_at: '2026-04-08T12:01:00Z',
        },
      ],
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });
    fireEvent.click(screen.getByRole('button', { name: 'Job Chains' }));

    await waitFor(() => {
      expect(screen.getByText('Failed Job (Recovery Playbook)')).toBeInTheDocument();
      expect(screen.getByText('Code Patch Playbook')).toBeInTheDocument();
    });

    const chainHeadings = screen.getAllByRole('heading', { level: 3 }).map((node) => node.textContent || '');
    const recoveryIndex = chainHeadings.indexOf('Failed Job (Recovery Playbook)');
    const genericIndex = chainHeadings.indexOf('Code Patch Playbook');
    expect(recoveryIndex).toBeGreaterThanOrEqual(0);
    expect(genericIndex).toBeGreaterThan(recoveryIndex);
    expect(screen.getByText('Recovery')).toBeInTheDocument();
  });

  it('filters the jobs list to scope-guard blocked jobs from the quick filter', async () => {
    await renderWithProviders('/autonomous-agents');

    await expectJobHeading('Autonomous Runtime Job');
    await expectJobHeading('Clean Scope Job');
    fireEvent.click(screen.getByText('Guard blocked 1'));
    await waitFor(() => {
      expect(screen.queryByRole('heading', { name: 'Clean Scope Job' })).not.toBeInTheDocument();
    });
    expect(screen.getByText('Guard blocked 1')).toBeInTheDocument();
  });

  it('sorts scope-guard blocked jobs first from the sort control', async () => {
    const blockedJob = makeJob();
    const cleanJob = makeJob({
      id: 'job-2',
      name: 'Clean Scope Job',
      results: {
        execution_strategy: {
          execution_graph: {
            graph_health: {
              status: 'ok',
              severity_score: 0,
              blocked_ratio: 0,
              reasons: [],
            },
            dag_stats: {
              total_nodes: 4,
              total_edges: 3,
              critical_path_length: 2,
              blocked_nodes: 0,
              root_nodes: 1,
              leaf_nodes: 1,
              orphan_nodes: 0,
              has_cycle: false,
            },
            verification_actions: [],
            summarization_actions: [],
            recommended_actions: [],
          },
          scope_observability: {
            resolved_scope_id: 'repo-456',
            scope_source: 'config.source_id',
            events: [
              {
                type: 'resolved_scope',
                source_id: 'repo-456',
                scope_source: 'config.source_id',
              },
            ],
          },
        },
      },
    });
    apiClient.listAgentJobs.mockResolvedValueOnce({
      jobs: [cleanJob, blockedJob],
      total: 2,
      page: 1,
      page_size: 50,
      has_more: false,
    });

    await renderWithProviders('/autonomous-agents');

    await expectJobHeading('Autonomous Runtime Job');
    fireEvent.change(screen.getByDisplayValue('Default graph sort'), {
      target: { value: 'scope_guard_blocked_first' },
    });
    const jobTitles = screen.getAllByRole('heading', { level: 3 }).map((node) => node.textContent);
    expect(jobTitles.indexOf('Autonomous Runtime Job')).toBeLessThan(jobTitles.indexOf('Clean Scope Job'));
  });

  it('sorts experiment recovery jobs by fallback before bootstrap recovery', async () => {
    await renderWithProviders('/autonomous-agents');

    await expectJobHeading('Autonomous Runtime Job');
    await expectJobHeading('Fallback Recovery Job');
    await expectJobHeading('Unresolved Recovery Job');

    fireEvent.change(screen.getByDisplayValue('Default graph sort'), {
      target: { value: 'experiment_recovery_priority' },
    });

    await waitFor(() => {
      const jobTitles = screen.getAllByRole('heading', { level: 3 }).map((node) => node.textContent || '');
      expect(jobTitles.indexOf('Unresolved Recovery Job')).toBeLessThan(jobTitles.indexOf('Fallback Recovery Job'));
      expect(jobTitles.indexOf('Fallback Recovery Job')).toBeLessThan(jobTitles.indexOf('Autonomous Runtime Job'));
      expect(jobTitles.indexOf('Autonomous Runtime Job')).toBeLessThan(jobTitles.indexOf('Clean Scope Job'));
    });
  });

  it('builds repo bug triage quick start payload with the expected normalization', () => {
    expect(
      buildRepoBugTriageQuickStartPayload({
        name: 'Repo Bug Triage - 3/24/2026',
        goal: 'Fix the regression without changing successful save flows',
        failureSymptom: 'Saving a document returns 500 and leaves the spinner running',
        selectedSourceId: 'repo-source-1',
        scope: 'frontend',
        searchQuery: '',
        commandsText: 'CI=true npm --prefix frontend test -- --watchAll=false\nnpm --prefix frontend run build',
        filePathsText: 'frontend/src/pages/DocumentsPage.tsx\nfrontend/src/services/api.ts',
        errorOutput: 'TypeError: saveDocument is not a function',
        maxCommands: 6,
        maxFilePaths: 32,
      })
    ).toEqual({
      name: 'Repo Bug Triage - 3/24/2026',
      goal: 'Fix the regression without changing successful save flows',
      failure_symptom: 'Saving a document returns 500 and leaves the spinner running',
      source_id: 'repo-source-1',
      scope: 'frontend',
      search_query: undefined,
      file_paths: ['frontend/src/pages/DocumentsPage.tsx', 'frontend/src/services/api.ts'],
      commands: [
        'CI=true npm --prefix frontend test -- --watchAll=false',
        'npm --prefix frontend run build',
      ],
      error_output: 'TypeError: saveDocument is not a function',
      start_immediately: true,
    });
  });

  it('builds bug triage swarm quick start payload with the expected normalization', () => {
    expect(
      buildBugTriageSwarmQuickStartPayload({
        name: 'Bug Triage Swarm - 3/24/2026',
        goal: 'Auto-launch the strongest repair loop',
        failureSymptom: 'Saving a document returns 500 and leaves the spinner running',
        selectedSourceId: 'repo-source-1',
        scope: 'frontend',
        searchQuery: '',
        commandsText: 'CI=true npm --prefix frontend test -- --watchAll=false\nnpm --prefix frontend run build',
        filePathsText: 'frontend/src/pages/DocumentsPage.tsx\nfrontend/src/services/api.ts',
        errorOutput: 'TypeError: saveDocument is not a function',
        maxAgents: 7,
        maxCommands: 6,
        maxFilePaths: 32,
      })
    ).toEqual({
      name: 'Bug Triage Swarm - 3/24/2026',
      goal: 'Auto-launch the strongest repair loop',
      failure_symptom: 'Saving a document returns 500 and leaves the spinner running',
      source_id: 'repo-source-1',
      scope: 'frontend',
      search_query: undefined,
      file_paths: ['frontend/src/pages/DocumentsPage.tsx', 'frontend/src/services/api.ts'],
      commands: [
        'CI=true npm --prefix frontend test -- --watchAll=false',
        'npm --prefix frontend run build',
      ],
      error_output: 'TypeError: saveDocument is not a function',
      max_agents: 4,
      start_immediately: true,
    });
  });

  it('shows domain research quick start with repo-aware scientific defaults', async () => {
    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Templates'));
    expect(await screen.findByText('Start Domain Research')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Start Domain Research'));
    const title = await screen.findByText('Quick Start Domain Research');
    const modal = title.closest('.fixed') as HTMLElement;
    expect(modal).toBeTruthy();

    expect(screen.getByText('Persist brief/report as Research Notes')).toBeInTheDocument();
    expect(screen.getByText('Auto-launch deep-dive follow-up for the strongest idea when confidence passes')).toBeInTheDocument();
    expect(screen.getByText('Repository evidence sources')).toBeInTheDocument();

    const selects = within(modal).getAllByRole('combobox');
    expect((selects[0] as HTMLSelectElement).value).toBe('compiler');
    expect((selects[1] as HTMLSelectElement).value).toBe('kb_plus_arxiv_plus_repo');
    expect((selects[2] as HTMLSelectElement).value).toBe('brief_and_report');

    fireEvent.change(within(modal).getByPlaceholderText('Compiler optimization and code generation'), {
      target: { value: 'Compiler optimization and code generation' },
    });
    fireEvent.change(
      within(modal).getByPlaceholderText('Rank compiler opportunities, explain the strongest evidence, and propose next experiments'),
      {
        target: { value: 'Rank compiler opportunities and create experiment-ready hypotheses.' },
      }
    );
    fireEvent.change(selects[0], {
      target: { value: 'compiler' },
    });
    fireEvent.change(
      within(modal).getByPlaceholderText(/compile time regression/i),
      {
        target: { value: 'compile time regression\nvectorization benchmark' },
      }
    );
    fireEvent.click(within(modal).getByLabelText(/Knowledge Repo/i));
    await waitFor(() => {
      expect(within(modal).getByLabelText(/Knowledge Repo/i)).toBeChecked();
    });

    expect(
      buildDomainResearchQuickStartPayload({
        name: 'Domain Research - 3/24/2026',
        domain: 'Compiler optimization and code generation',
        objective: 'Rank compiler opportunities and create experiment-ready hypotheses.',
        customerContextValue: '',
        trackType: 'compiler',
        sourceScope: 'kb_plus_arxiv_plus_repo',
        monitorQueriesText: '',
        benchmarkQueriesText: 'compile time regression\nvectorization benchmark',
        selectedRepoSourceIds: ['repo-source-1'],
        sandboxProfileId: 'scientific-compiler-sandbox',
        reportFormat: 'brief_and_report',
        persistArtifacts: true,
        autoLaunchFollowUp: true,
      })
    ).toEqual({
      name: 'Domain Research - 3/24/2026',
      domain: 'Compiler optimization and code generation',
      objective: 'Rank compiler opportunities and create experiment-ready hypotheses.',
      customer_context: undefined,
      track_type: 'compiler',
      source_scope: 'kb_plus_arxiv_plus_repo',
      monitor_queries: undefined,
      repo_source_ids: ['repo-source-1'],
      benchmark_queries: ['compile time regression', 'vectorization benchmark'],
      sandbox_profile_id: 'scientific-compiler-sandbox',
      report_format: 'brief_and_report',
      persist_artifacts: true,
      automation_profile: 'balanced',
      automation_policy: DEFAULT_VALIDATION_POLICY,
      auto_launch_follow_up: true,
      auto_create_experiment_plans: true,
      start_immediately: true,
    });
  });

  it('shows bug triage swarm quick start with coding defaults', async () => {
    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Templates'));
    expect(await screen.findByText('Start Bug Triage Swarm')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Start Bug Triage Swarm'));
    const title = await screen.findByText('Quick Start Bug Triage Swarm');
    const modal = title.closest('.fixed') as HTMLElement;
    expect(modal).toBeTruthy();

    const selects = within(modal).getAllByRole('combobox');
    expect((selects[1] as HTMLSelectElement).value).toBe('auto');
    expect((within(modal).getByRole('spinbutton') as HTMLInputElement).value).toBe('4');
    expect(screen.getByText('Start Swarm')).toBeInTheDocument();
  });

  it('manages coding swarm profiles from the dedicated profiles tab', async () => {
    apiClient.listCodingSwarmProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-bug-default',
          user_id: 'user-1',
          source_id: 'repo-source-1',
          title: 'Bug Triage Default',
          description: 'Default repo profile',
          status: 'active',
          preset_key: 'bug_triage_swarm',
          scope_default: 'frontend',
          default_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
          default_file_paths: ['frontend/src/pages/DocumentsPage.tsx'],
          max_agents: 4,
          safe_command_policy: 'standard',
          saved_search_query: 'save regression',
          is_default: true,
          visibility: 'private',
          shared_with_user_ids: [],
          collaboration_summary: {
            owner_user_id: 'user-1',
            owner_label: 'Repo Owner',
            shared_with_user_ids: [],
            visibility_scope: 'private',
            is_owned_by_current_user: true,
            is_assigned_to_current_user: false,
            is_shared_with_current_user: false,
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
      total: 1,
      limit: 200,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Swarm Profiles'));
    expect(await screen.findByText('Coding Swarm Profiles')).toBeInTheDocument();
    expect(screen.getByText('Bug Triage Default')).toBeInTheDocument();
    expect(screen.getByText(/Owner Repo Owner/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText('Edit'));
    fireEvent.change(screen.getByDisplayValue('Bug Triage Default'), { target: { value: 'Bug Triage Primary' } });
    fireEvent.click(screen.getByText('Save profile'));

    await waitFor(() => {
      expect(apiClient.updateCodingSwarmProfile).toHaveBeenCalledWith(
        'profile-bug-default',
        expect.objectContaining({
          title: 'Bug Triage Primary',
          preset_key: 'bug_triage_swarm',
          is_default: true,
        })
      );
    });
  });

  it('can launch a coding swarm from the dedicated profiles tab', async () => {
    apiClient.listCodingSwarmProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-bug-default',
          user_id: 'user-1',
          source_id: 'repo-source-1',
          title: 'Bug Triage Default',
          description: null,
          status: 'active',
          preset_key: 'bug_triage_swarm',
          scope_default: 'frontend',
          default_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
          default_file_paths: ['frontend/src/pages/DocumentsPage.tsx'],
          max_agents: 4,
          safe_command_policy: 'standard',
          saved_search_query: 'save regression',
          is_default: true,
          visibility: 'private',
          shared_with_user_ids: [],
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
      total: 1,
      limit: 200,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Swarm Profiles'));
    expect(await screen.findByText('Coding Swarm Profiles')).toBeInTheDocument();

    const profileCard = screen.getByText('Bug Triage Default').closest('.border') as HTMLElement;
    fireEvent.click(within(profileCard).getByText('Launch'));

    const title = await screen.findByText('Quick Start Bug Triage Swarm');
    const modal = title.closest('.fixed') as HTMLElement;
    fireEvent.click(within(modal).getByText('Start Swarm'));

    await waitFor(() => {
      expect(apiClient.quickStartBugTriageSwarmJob).toHaveBeenCalledWith(
        expect.objectContaining({
          profile_id: 'profile-bug-default',
          source_id: 'repo-source-1',
          scope: 'frontend',
          search_query: 'save regression',
        })
      );
    });
  });

  it('blocks unsafe repo bug triage commands client-side', () => {
    const commands = parseQuickStartCommands(
      'CI=true npm --prefix frontend test -- --watchAll=false\nrm -rf /tmp/example',
      6
    );

    expect(findUnsafeQuickStartCommands(commands)).toEqual(['rm -rf /tmp/example']);
  });

  it('seeds the compiler and microarchitecture scientific pack with repo-aware defaults', async () => {
    apiClient.createDomainResearchProfile
      .mockResolvedValueOnce({ id: 'profile-compiler' })
      .mockResolvedValueOnce({ id: 'profile-microarch' });
    apiClient.createResearchPortfolio.mockResolvedValueOnce({ id: 'portfolio-science' });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Seed Compiler + Microarch Pack')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Seed Compiler + Microarch Pack'));

    await waitFor(() => {
      expect(apiClient.createDomainResearchProfile).toHaveBeenCalledTimes(2);
      expect(apiClient.createResearchPortfolio).toHaveBeenCalledTimes(1);
    });

    expect(apiClient.createDomainResearchProfile).toHaveBeenNthCalledWith(1, {
      title: 'Compiler Research Pack',
      domain: 'Compiler optimization and code generation',
      objective: 'Identify evidence-backed compiler opportunities, regressions, and validation experiments for the customer codebase.',
      track_type: 'compiler',
      monitor_queries: [
        'llvm optimization pass regression',
        'mlir codegen scheduling',
        'auto-vectorization blocker benchmark',
      ],
      benchmark_queries: [
        'compile time regression',
        'vectorization benchmark',
        'codegen hotspot',
      ],
      source_scope: 'kb_plus_arxiv_plus_repo',
      repo_source_ids: ['repo-source-1', 'repo-source-2'],
      sandbox_profile_id: 'scientific-compiler-sandbox',
      automation_profile: 'max_autonomy',
      automation_policy: {
        ...DEFAULT_VALIDATION_POLICY,
        auto_execute_validation_runs: true,
      },
      interval_minutes: 1440,
      persist_artifacts: true,
      auto_launch_follow_up: true,
      auto_create_experiment_plans: true,
      start_immediately: true,
    });
    expect(apiClient.createDomainResearchProfile).toHaveBeenNthCalledWith(2, {
      title: 'Microarchitecture Research Pack',
      domain: 'CPU microarchitecture performance and bottlenecks',
      objective: 'Surface testable microarchitecture opportunities tied to cache behavior, branch behavior, SIMD usage, and benchmark regressions.',
      track_type: 'microarchitecture',
      monitor_queries: [
        'cache miss bottleneck benchmark',
        'branch predictor workload analysis',
        'simd throughput regression',
      ],
      benchmark_queries: [
        'ipc stall benchmark',
        'branch miss benchmark',
        'memory bandwidth benchmark',
      ],
      source_scope: 'kb_plus_arxiv_plus_repo',
      repo_source_ids: ['repo-source-1', 'repo-source-2'],
      sandbox_profile_id: 'scientific-microarchitecture-sandbox',
      automation_profile: 'max_autonomy',
      automation_policy: {
        ...DEFAULT_VALIDATION_POLICY,
        auto_execute_validation_runs: true,
      },
      interval_minutes: 1440,
      persist_artifacts: true,
      auto_launch_follow_up: true,
      auto_create_experiment_plans: true,
      start_immediately: true,
    });
    expect(apiClient.createResearchPortfolio).toHaveBeenCalledWith({
      title: 'Scientific Research Fleet',
      objective: 'Continuously rank novel and testable compiler and microarchitecture ideas, auto-create validation plans, and auto-launch bounded deep dives within budget.',
      linked_profile_ids: ['profile-compiler', 'profile-microarch'],
      automation_profile: 'max_autonomy',
      automation_policy: {
        ...DEFAULT_VALIDATION_POLICY,
        confidence_threshold: 0.68,
        experiment_readiness_threshold: 0.72,
        max_auto_follow_up_launches: 4,
        auto_execute_validation_runs: true,
        auto_launch_experiment_runs: true,
        max_concurrent_validation_runs: 2,
        max_validation_runtime_minutes: 30,
        max_validation_budget_per_run: 50,
        duplicate_window_items: 120,
      },
      sandbox_profile_id: 'scientific-compiler-sandbox',
      start_immediately: true,
    });
  });

  it('creates a coding backlog item with the expected payload', async () => {
    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Coding Backlog'));
    expect(await screen.findByText('Start Backlog')).toBeInTheDocument();

    fireEvent.input(screen.getByPlaceholderText('Backlog title'), {
      target: { value: 'Stabilize document save flow' },
    });
    fireEvent.input(screen.getByPlaceholderText('Portfolio goal'), {
      target: { value: 'Repair the unstable document save flow and automatically apply low-risk fixes.' },
    });
    fireEvent.input(screen.getByPlaceholderText('Observed failure symptom (optional but recommended)'), {
      target: { value: 'Saving a document sometimes returns 500 after a recent UI change.' },
    });

    const selects = screen.getAllByRole('combobox');
    fireEvent.change(selects[0], {
      target: { value: 'repo-source-1' },
    });
    fireEvent.change(selects[1], {
      target: { value: 'frontend' },
    });

    fireEvent.input(screen.getByPlaceholderText('Verification commands, one per line (optional)'), {
      target: { value: 'CI=true npm --prefix frontend test -- --watchAll=false\nnpm --prefix frontend run build' },
    });
    fireEvent.input(screen.getByPlaceholderText('File path hints, one per line (optional)'), {
      target: { value: 'frontend/src/pages/DocumentsPage.tsx\nfrontend/src/services/api.ts' },
    });

    fireEvent.click(screen.getByText('Start Backlog'));

    await waitFor(() => {
      expect(apiClient.createCodingBacklogItem).toHaveBeenCalledWith({
        title: 'Stabilize document save flow',
        portfolio_goal: 'Repair the unstable document save flow and automatically apply low-risk fixes.',
        source_id: 'repo-source-1',
        scope: 'frontend',
        failure_symptom: 'Saving a document sometimes returns 500 after a recent UI change.',
        commands: [
          'CI=true npm --prefix frontend test -- --watchAll=false',
          'npm --prefix frontend run build',
        ],
        file_paths: ['frontend/src/pages/DocumentsPage.tsx', 'frontend/src/services/api.ts'],
        auto_apply_enabled: true,
        require_patch_pr: false,
        policy: { max_auto_retries: 1 },
        start_immediately: true,
      });
    });
  });

  it('edits backlog notes inline without prompts', async () => {
    const backlogItem = {
      id: 'backlog-inline-1',
      user_id: 'user-1',
      source_id: 'repo-source-1',
      title: 'Manual collaboration backlog',
      portfolio_goal: 'Document an operator note and close reason without a prompt.',
      status: 'draft',
      priority: 40,
      auto_apply_enabled: true,
      require_patch_pr: false,
      collaboration_summary: {
        owner_user_id: 'user-1',
        owner_label: 'Repo Owner',
        shared_with_user_ids: [],
        visibility_scope: 'private',
        is_owned_by_current_user: true,
        is_assigned_to_current_user: false,
        is_shared_with_current_user: false,
      },
      created_at: '2026-03-10T00:00:00Z',
      updated_at: '2026-03-10T00:00:00Z',
    };
    apiClient.listCodingBacklogItems.mockResolvedValueOnce({
      items: [backlogItem],
      total: 1,
      limit: 100,
      offset: 0,
    });
    apiClient.performCodingBacklogAction.mockResolvedValueOnce(backlogItem);
    const promptSpy = jest.spyOn(window, 'prompt').mockImplementation(() => null as any);

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Coding Backlog'));
    const backlogCard = await screen.findByText('Manual collaboration backlog');
    const card = backlogCard.closest('.border') as HTMLElement;
    fireEvent.change(within(card).getByPlaceholderText('Backlog operator note'), {
      target: { value: 'Inline backlog note' },
    });
    fireEvent.change(within(card).getByDisplayValue('Choose close reason'), {
      target: { value: 'duplicate' },
    });
    fireEvent.click(within(card).getByText('Close'));

    await waitFor(() => {
      expect(apiClient.performCodingBacklogAction).toHaveBeenCalledWith('backlog-inline-1', {
        action: 'cancel',
        closure_reason: 'duplicate',
        operator_note: 'Inline backlog note',
      });
    });
    expect(promptSpy).not.toHaveBeenCalled();
    promptSpy.mockRestore();
  });

  it('renders swarm review analytics and linked backlog lineage', async () => {
    const swarmJob = makeJob({
      id: 'swarm-root-1',
      name: 'Frontend Regression Swarm Root',
      goal: 'Review the frontend regression and pick the best repair path.',
      status: 'paused',
      launch_mode: 'quick_start_frontend_regression_swarm' as any,
      config: {
        launch_mode: 'quick_start_frontend_regression_swarm',
        source_id: 'repo-source-1',
        scope: 'frontend',
        failure_symptom: 'Clicking save no longer updates the UI',
        quick_start: {
          source_name: 'Knowledge Repo',
          preset_key: 'frontend_regression_swarm',
        },
      },
      swarm_summary: {
        review_state: 'needs_review',
        review_reason: 'Roles disagree on the affected component cluster.',
        review_required: true,
        winning_role: 'root_cause',
        confidence: { overall: 0.61 },
        candidate_paths: [
          {
            job_id: 'candidate-1',
            role: 'root_cause',
            score: 0.72,
            suspect_files: ['frontend/src/pages/DocumentsPage.tsx'],
            recommended_commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
          },
        ],
      } as any,
    });
    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [swarmJob],
      total: 1,
      page: 1,
      page_size: 200,
      has_more: false,
    });
    apiClient.getAgentJobStats.mockResolvedValue({
      total_jobs: 1,
      running_jobs: 0,
      pending_jobs: 0,
      completed_jobs: 0,
      failed_jobs: 0,
      total_iterations: 0,
      total_tool_calls: 0,
      total_llm_calls: 0,
      launch_mode_counts: {
        quick_start_frontend_regression_swarm: 1,
      },
    });
    apiClient.getAgentJobSwarmAnalytics.mockResolvedValue({
      preset_rows: [
        {
          preset_key: 'frontend_regression_swarm',
          launch_mode: 'quick_start_frontend_regression_swarm',
          label: 'Frontend Regression Swarm',
          total_runs: 3,
          avg_confidence: 0.64,
          high_confidence_runs: 1,
          medium_confidence_runs: 2,
          low_confidence_runs: 0,
          auto_promoted_runs: 1,
          review_needed_runs: 2,
          tie_breaker_runs: 1,
          manual_promotion_runs: 0,
          repair_handoff_runs: 1,
          backlog_handoff_runs: 1,
          promotion_rate: 0.33,
          review_rate: 0.67,
          tie_breaker_rate: 0.33,
        },
      ],
      totals: {
        total_runs: 3,
        repair_handoff_runs: 1,
        review_needed_runs: 2,
        avg_confidence: 0.64,
      },
      filters: {},
    });
    apiClient.listCodingBacklogItems.mockResolvedValue({
      items: [
        {
          id: 'backlog-1',
          user_id: 'user-1',
          source_id: 'repo-source-1',
          title: 'Frontend regression follow-up',
          portfolio_goal: 'Continue the best swarm path manually.',
          status: 'draft',
          priority: 50,
          auto_apply_enabled: true,
          require_patch_pr: false,
          lineage: {
            originating_swarm_job_id: 'swarm-root-1',
            originating_swarm_preset: 'frontend_regression_swarm',
            originating_swarm_review_reason: 'Roles disagree on the affected component cluster.',
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });
    apiClient.performAgentJobAction.mockResolvedValueOnce(swarmJob);
    const promptSpy = jest.spyOn(window, 'prompt').mockImplementation(() => null as any);

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Swarm Review'));

    expect(await screen.findAllByText('Frontend Regression Swarm')).toHaveLength(2);
    expect(screen.getByText('Roles disagree on the affected component cluster.')).toBeInTheDocument();
    expect(screen.getByText(/Backlog linked 1/i)).toBeInTheDocument();
    expect(screen.getByText(/Frontend regression follow-up/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Confidence 61%/i)).toHaveLength(2);
    const swarmCard = screen.getAllByText('Frontend Regression Swarm Root')[0].closest('.border') as HTMLElement;
    fireEvent.change(within(swarmCard).getByPlaceholderText('Swarm review note'), {
      target: { value: 'Inline review note' },
    });
    fireEvent.click(within(swarmCard).getByText('Save review note'));

    await waitFor(() => {
      expect(apiClient.performAgentJobAction).toHaveBeenCalledWith(
        'swarm-root-1',
        'update_swarm_review_note',
        expect.objectContaining({
          action_payload: { review_note: 'Inline review note' },
        })
      );
    });
    expect(promptSpy).not.toHaveBeenCalled();
    promptSpy.mockRestore();
  });

  it('renders swarm outcome funnel analytics and drilldown links', async () => {
    const swarmJob = makeJob({
      id: 'swarm-outcome-1',
      name: 'Frontend Regression Swarm Root',
      launch_mode: 'quick_start_frontend_regression_swarm' as any,
      config: {
        launch_mode: 'quick_start_frontend_regression_swarm',
        source_id: 'repo-source-1',
        scope: 'frontend',
        quick_start: {
          source_name: 'Knowledge Repo',
          preset_key: 'frontend_regression_swarm',
        },
      },
    });
    const repairJob = makeJob({
      id: 'repair-1',
      name: 'Frontend Repair Chain',
      launch_mode: 'quick_start_repo_bug_triage' as any,
      status: 'completed',
      results: {
        code_patch_execution: {
          verification_plan: {
            commands: ['CI=true npm --prefix frontend test -- --watchAll=false'],
          },
          recovery: {
            recovery_state: 'verified_fix',
            retry_reason: 'Verification succeeded against the promoted fix.',
          },
        },
      },
    });
    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [swarmJob, repairJob],
      total: 2,
      page: 1,
      page_size: 200,
      has_more: false,
    });
    apiClient.getAgentJobStats.mockResolvedValue({
      total_jobs: 2,
      running_jobs: 0,
      pending_jobs: 0,
      completed_jobs: 2,
      failed_jobs: 0,
      total_iterations: 0,
      total_tool_calls: 0,
      total_llm_calls: 0,
      launch_mode_counts: {
        quick_start_frontend_regression_swarm: 1,
        quick_start_repo_bug_triage: 1,
      },
    });
    apiClient.getAgentJobSwarmOutcomeAnalytics.mockResolvedValue({
      preset_rows: [
        {
          preset_key: 'frontend_regression_swarm',
          launch_mode: 'quick_start_frontend_regression_swarm',
          label: 'Frontend Regression Swarm',
          total_swarm_roots: 1,
          auto_promoted_runs: 1,
          manual_promoted_runs: 0,
          tie_breaker_runs: 0,
          repair_handoff_runs: 1,
          verified_fix_runs: 1,
          repair_failed_runs: 0,
          backlog_routed_runs: 0,
          needs_review_runs: 0,
          stalled_after_handoff_runs: 0,
          avg_confidence: 0.82,
          avg_handoff_minutes: 14,
        },
      ],
      cases: [
        {
          swarm_job_id: 'swarm-outcome-1',
          swarm_job_name: 'Frontend Regression Swarm Root',
          preset_key: 'frontend_regression_swarm',
          launch_mode: 'quick_start_frontend_regression_swarm',
          source_id: 'repo-source-1',
          source_label: 'Knowledge Repo',
          swarm_status: 'completed',
          swarm_completed_at: '2026-03-10T00:00:00Z',
          review_state: 'auto_promoted',
          promotion_mode: 'auto',
          confidence_overall: 0.82,
          tie_breaker_attempted: false,
          repair_job_id: 'repair-1',
          repair_job_name: 'Frontend Repair Chain',
          repair_status: 'completed',
          repair_handoff_at: '2026-03-10T00:14:00Z',
          verification_status: 'succeeded',
          verification_reason: 'Verification succeeded against the promoted fix.',
          backlog_item_id: null,
          backlog_title: null,
          backlog_status: null,
          latest_downstream_at: '2026-03-10T00:20:00Z',
          handoff_latency_minutes: 14,
          terminal_outcome: 'verified_fix',
          terminal_reason: 'Repair verification succeeded.',
        },
      ],
      totals: {
        total_swarm_roots: 1,
        repair_handoff_runs: 1,
        verified_fix_runs: 1,
        backlog_routed_runs: 0,
        avg_handoff_minutes: 14,
      },
      filters: {},
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Swarm Outcomes'));

    expect(await screen.findByText('Verified fixes')).toBeInTheDocument();
    expect(screen.getAllByText('Frontend Regression Swarm Root').length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Repair verification succeeded\./i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Promotion auto/i)).toBeInTheDocument();
    expect(screen.getByText(/Open repair/i)).toBeInTheDocument();
  });

  it('creates a domain research profile with the expected payload', async () => {
    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Start Monitor')).toBeInTheDocument();

    fireEvent.input(screen.getByPlaceholderText('Profile title'), {
      target: { value: 'Retrieval R&D Monitor' },
    });
    fireEvent.input(screen.getByPlaceholderText('Domain or topic'), {
      target: { value: 'Multimodal retrieval' },
    });
    fireEvent.input(screen.getByPlaceholderText('Research objective'), {
      target: { value: 'Track new evidence, rank ideas, and generate experiment plans.' },
    });
    fireEvent.input(screen.getByPlaceholderText('Monitor queries, one per line'), {
      target: { value: 'multimodal retrieval benchmarks\nretrieval grounding evaluation' },
    });
    fireEvent.input(screen.getByPlaceholderText('Cadence in minutes'), {
      target: { value: '720' },
    });

    fireEvent.click(screen.getByText('Start Monitor'));

    await waitFor(() => {
      expect(apiClient.createDomainResearchProfile).toHaveBeenCalledWith({
        title: 'Retrieval R&D Monitor',
        domain: 'Multimodal retrieval',
        objective: 'Track new evidence, rank ideas, and generate experiment plans.',
        track_type: 'compiler',
        source_scope: 'kb_plus_arxiv_plus_repo',
        monitor_queries: ['multimodal retrieval benchmarks', 'retrieval grounding evaluation'],
        repo_source_ids: undefined,
        benchmark_queries: [],
        sandbox_profile_id: 'scientific-compiler-sandbox',
        scoring_policy: {
          minimum_subscore: 0.6,
          minimum_supporting_sources: 2,
          weights: { novelty: 0.4, evidence: 0.35, testability: 0.25 },
        },
        selection_policy: { max_candidates: 10, max_hypotheses: 3 },
        automation_profile: 'balanced',
        automation_policy: DEFAULT_VALIDATION_POLICY,
        interval_minutes: 720,
        research_mode: 'literature_to_hypothesis',
        persist_artifacts: true,
        auto_launch_follow_up: true,
        auto_create_experiment_plans: true,
        start_immediately: true,
      });
    });
  });

  it('promotes a completed domain research quick-start job into a monitor and fleet', async () => {
    const promotedJob = makeJob({
      id: 'job-promote',
      name: 'Compiler Optimization Scout',
      status: 'completed',
      launch_mode: 'quick_start_domain_research',
      config: {
        launch_mode: 'quick_start_domain_research',
        domain: 'Compiler optimization',
        objective: 'Find benchmark-backed opportunities',
        source_scope: 'kb_plus_arxiv_plus_repo',
        track_type: 'compiler',
        monitor_queries: ['compiler optimization benchmark'],
        benchmark_queries: ['llvm pass regression'],
        automation_profile: 'balanced',
        automation_policy: DEFAULT_VALIDATION_POLICY,
        quick_start: { profile: 'domain_research', version: 'v2' },
      },
    });
    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [promotedJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValue(promotedJob);

    await renderWithProviders('/autonomous-agents?job=job-promote', { documentSources: defaultDocumentSources });

    expect(await screen.findByText('Promote to Monitor')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Promote to Monitor'));

    const promotionSelectsBefore = screen.getAllByRole('combobox');
    fireEvent.change(promotionSelectsBefore[promotionSelectsBefore.length - 1], {
      target: { value: 'profile_with_portfolio' },
    });
    fireEvent.change(screen.getByDisplayValue('Compiler optimization Fleet'), {
      target: { value: 'Compiler Research Fleet' },
    });
    fireEvent.click(screen.getByText('Create monitor'));

    await waitFor(() => {
      expect(apiClient.promoteDomainResearchAgentJob).toHaveBeenCalledWith('job-promote', {
        target_mode: 'profile_with_portfolio',
        profile: {
          title: 'Compiler Optimization Scout',
          interval_minutes: 1440,
          domain: 'Compiler optimization',
          objective: 'Find benchmark-backed opportunities',
          customer_context: undefined,
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          research_mode: undefined,
          monitor_queries: ['compiler optimization benchmark'],
          repo_source_ids: undefined,
          benchmark_queries: ['llvm pass regression'],
          report_format: undefined,
          scoring_policy: undefined,
          selection_policy: undefined,
          sandbox_profile_id: undefined,
          automation_profile: 'balanced',
          automation_policy: DEFAULT_VALIDATION_POLICY,
          persist_artifacts: true,
          auto_launch_follow_up: true,
          auto_create_experiment_plans: true,
          confidence_threshold: undefined,
          max_documents: undefined,
          max_papers: undefined,
        },
        portfolio: {
          title: 'Compiler Research Fleet',
          objective: 'Find benchmark-backed opportunities',
          sandbox_profile_id: undefined,
          automation_profile: 'balanced',
          automation_policy: DEFAULT_VALIDATION_POLICY,
        },
        start_profile_now: true,
        run_portfolio_now: false,
      });
    });
  });

  it('attaches a promoted domain research quick-start job to an existing fleet', async () => {
    const promotedJob = makeJob({
      id: 'job-promote-existing',
      name: 'Compiler Optimization Scout',
      status: 'completed',
      launch_mode: 'quick_start_domain_research',
      config: {
        launch_mode: 'quick_start_domain_research',
        domain: 'Compiler optimization',
        objective: 'Find benchmark-backed opportunities',
        source_scope: 'kb_plus_arxiv_plus_repo',
        track_type: 'compiler',
        monitor_queries: ['compiler optimization benchmark'],
        automation_profile: 'balanced',
        automation_policy: DEFAULT_VALIDATION_POLICY,
        quick_start: { profile: 'domain_research', version: 'v2' },
      },
    });
    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [promotedJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValue(promotedJob);
    apiClient.listResearchPortfolios.mockResolvedValue({
      items: [
        {
          id: 'portfolio-1',
          title: 'Compiler Research Fleet',
          objective: 'Track benchmark-backed optimization opportunities',
          track_type: 'compiler',
          active: true,
          profiles: [],
          latest_summary: {},
          opportunities: [],
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents?job=job-promote-existing', { documentSources: defaultDocumentSources });

    expect(apiClient.listResearchPortfolios).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByText('Promote to Monitor'));

    const getPromotionSelect = (label: string) =>
      screen.getByText(label).parentElement!.querySelector('select') as HTMLSelectElement;

    fireEvent.change(getPromotionSelect('Promotion scope'), {
      target: { value: 'profile_with_portfolio' },
    });
    await waitFor(() => {
      expect(screen.getByText('Fleet target')).toBeInTheDocument();
    });
    fireEvent.change(getPromotionSelect('Fleet target'), {
      target: { value: 'existing' },
    });
    await waitFor(() => {
      expect(apiClient.listResearchPortfolios).toHaveBeenCalledWith({ limit: 100, offset: 0 });
      expect(screen.getByText('Existing fleet')).toBeInTheDocument();
      expect(within(getPromotionSelect('Existing fleet')).getByRole('option', { name: 'Compiler Research Fleet' })).toBeInTheDocument();
    });
    fireEvent.change(getPromotionSelect('Existing fleet'), {
      target: { value: 'portfolio-1' },
    });
    fireEvent.click(screen.getByText('Create monitor'));

    await waitFor(() => {
      expect(apiClient.promoteDomainResearchAgentJob).toHaveBeenCalledWith('job-promote-existing', expect.objectContaining({
        target_mode: 'profile_with_portfolio',
        portfolio_id: 'portfolio-1',
      }));
    });
  });

  it('shows promotion status links and hides the promotion action once already promoted', async () => {
    const promotedJob = makeJob({
      id: 'job-promoted',
      name: 'Compiler Optimization Scout',
      status: 'completed',
      launch_mode: 'quick_start_domain_research',
      promotion_status: 'promoted_to_profile_and_portfolio',
      promoted_domain_research_profile_id: 'profile-promoted-1',
      promoted_research_portfolio_id: 'portfolio-promoted-1',
      config: {
        launch_mode: 'quick_start_domain_research',
        domain: 'Compiler optimization',
        objective: 'Find benchmark-backed opportunities',
        quick_start: {
          profile: 'domain_research',
          version: 'v2',
          promotion: {
            status: 'promoted_to_profile_and_portfolio',
            domain_research_profile_id: 'profile-promoted-1',
            research_portfolio_id: 'portfolio-promoted-1',
          },
        },
      },
    });
    apiClient.listAgentJobs.mockResolvedValue({
      jobs: [promotedJob],
      total: 1,
      page: 1,
      page_size: 50,
      has_more: false,
    });
    apiClient.getAgentJob.mockResolvedValue(promotedJob);

    await renderWithProviders('/autonomous-agents?job=job-promoted', { documentSources: defaultDocumentSources });

    expect(await screen.findByText('Open monitor')).toBeInTheDocument();
    expect(screen.getByText('Open fleet')).toBeInTheDocument();
    expect(screen.queryByText('Promote to Monitor')).not.toBeInTheDocument();
  });

  it('dispatches domain opportunity actions from the opportunity queue', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Retrieval R&D Monitor',
          domain: 'Multimodal retrieval',
          objective: 'Track new evidence, rank ideas, and generate experiment plans.',
          status: 'running',
          source_scope: 'kb_plus_arxiv',
          track_type: 'generic',
          research_mode: 'literature_to_hypothesis',
          monitor_queries: ['retrieval grounding evaluation'],
          repo_source_ids: [],
          benchmark_queries: [],
          report_format: 'brief_and_report',
          automation_profile: 'balanced',
          automation_policy: {},
          effective_policy: {},
          interval_minutes: 720,
          persist_artifacts: true,
          auto_launch_follow_up: true,
          auto_create_experiment_plans: true,
          confidence_threshold: 0.7,
          max_documents: 10,
          max_papers: 8,
          opportunities: [
            {
              opportunity_id: 'opp-1',
              canonical_key: 'retrieval_eval_gap',
              title: 'Retrieval eval gap',
              hypothesis: 'Need a grounded benchmark',
              stage: 'discovered',
              decision_state: 'pending_review',
              decision_source: 'system',
              confidence: 0.83,
              novelty: 0.74,
              readiness: 0.69,
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: [],
            },
          ],
          latest_summary: {},
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_validation_runs: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:05:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Opportunity queue')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Create Plan'));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenCalledWith('profile-1', 'opp-1', {
        action: 'create_plan',
        operator_note: undefined,
      });
    });
  });

  it('materializes a domain opportunity into an experiment and shows the validation job link', async () => {
    apiClient.listDomainResearchProfiles
      .mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Retrieval R&D Monitor',
          domain: 'Multimodal retrieval',
          objective: 'Track new evidence, rank ideas, and generate experiment plans.',
          status: 'running',
          source_scope: 'kb_plus_arxiv',
          track_type: 'generic',
          research_mode: 'literature_to_hypothesis',
          monitor_queries: ['retrieval grounding evaluation'],
          repo_source_ids: [],
          benchmark_queries: [],
          report_format: 'brief_and_report',
          automation_profile: 'balanced',
          automation_policy: {},
          effective_policy: {},
          interval_minutes: 720,
          persist_artifacts: true,
          auto_launch_follow_up: true,
          auto_create_experiment_plans: true,
          confidence_threshold: 0.7,
          max_documents: 10,
          max_papers: 8,
          opportunities: [
            {
              opportunity_id: 'opp-1',
              canonical_key: 'retrieval_eval_gap',
              title: 'Retrieval eval gap',
              hypothesis: 'Need a grounded benchmark',
              stage: 'accepted',
              decision_state: 'accepted',
              decision_source: 'operator',
              confidence: 0.83,
              novelty: 0.74,
              readiness: 0.69,
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: [],
            },
          ],
          latest_summary: {},
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_validation_runs: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:05:00Z',
        },
      ],
      total: 1,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'profile-1',
            user_id: 'user-1',
            title: 'Domain Monitor',
            domain: 'Retrieval',
            objective: 'Track retrieval opportunities',
            status: 'running',
            source_scope: 'kb_plus_arxiv',
            track_type: 'generic',
            research_mode: 'literature_to_hypothesis',
            monitor_queries: ['retrieval benchmarks'],
            repo_source_ids: [],
            benchmark_queries: [],
            report_format: 'brief_and_report',
            automation_profile: 'balanced',
            automation_policy: {},
            effective_policy: {},
            interval_minutes: 1440,
            persist_artifacts: true,
            auto_launch_follow_up: true,
            auto_create_experiment_plans: true,
            confidence_threshold: 0.7,
            max_documents: 10,
            max_papers: 8,
            opportunities: [
              {
                opportunity_id: 'opp-1',
                canonical_key: 'retrieval_eval_gap',
                title: 'Retrieval eval gap',
                hypothesis: 'Need a grounded benchmark',
                stage: 'validating',
                decision_state: 'accepted',
                decision_source: 'operator',
                confidence: 0.83,
                novelty: 0.74,
                readiness: 0.69,
                linked_experiment_plan_ids: ['plan-1'],
                linked_validation_run_ids: ['run-1'],
                latest_experiment_plan_id: 'plan-1',
                latest_validation_run_id: 'run-1',
                latest_validation_job_id: 'job-validation-1',
                latest_validation_status: 'queued',
                latest_validation_blocked_reason_code: null,
                source_note_ids: ['note-1'],
                child_job_ids: [],
              },
            ],
            latest_summary: {},
            latest_note_ids: ['note-1'],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            latest_validation_runs: [],
            latest_run_job_id: 'job-domain-1',
            active_job_id: 'job-domain-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:06:00Z',
          },
        ],
        total: 1,
      });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Opportunity queue')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Run Experiment'));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenCalledWith('profile-1', 'opp-1', {
        action: 'materialize_experiment',
        operator_note: undefined,
        start_immediately: true,
      });
    });

    expect(await screen.findByText('Open plan')).toBeInTheDocument();
    expect(await screen.findByText('Open run')).toBeInTheDocument();
    expect(await screen.findByText('Open validation job')).toBeInTheDocument();
  });

  it('dispatches fleet opportunity suppression with an operator note', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Research Fleet',
          objective: 'Continuously rank novel and testable ideas.',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_policy: {},
          opportunities: [
            {
              opportunity_id: 'opp-portfolio-1',
              canonical_key: 'compiler_hotspot',
              title: 'Compiler hotspot',
              hypothesis: 'Scheduler bottleneck',
              stage: 'discovered',
              decision_state: 'pending_review',
              decision_source: 'system',
              confidence: 0.88,
              novelty: 0.71,
              readiness: 0.79,
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: [],
            },
          ],
          latest_summary: { stage_counts: { discovered: 1, planned: 0, validating: 0, suppressed: 0 } },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_validation_runs: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:05:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Top opportunities')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Suppress'));
    fireEvent.change(screen.getByLabelText('Fleet suppression note'), { target: { value: 'Low signal duplicate' } });
    fireEvent.click(screen.getByText('Save suppression'));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenCalledWith('portfolio-1', 'opp-portfolio-1', {
        action: 'suppress',
        operator_note: 'Low signal duplicate',
      });
    });
  });

  it('materializes a fleet opportunity into an experiment', async () => {
    apiClient.listResearchPortfolios
      .mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Research Fleet',
          objective: 'Continuously rank novel and testable ideas.',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_policy: {},
          opportunities: [
            {
              opportunity_id: 'opp-portfolio-1',
              canonical_key: 'compiler_hotspot',
              title: 'Compiler hotspot',
              hypothesis: 'Scheduler bottleneck',
              stage: 'accepted',
              decision_state: 'accepted',
              decision_source: 'operator',
              confidence: 0.88,
              novelty: 0.71,
              readiness: 0.79,
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: [],
            },
          ],
          latest_summary: { stage_counts: { discovered: 0, planned: 0, validating: 0, suppressed: 0 } },
          latest_note_ids: ['note-1'],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_validation_runs: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:05:00Z',
        },
      ],
      total: 1,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'portfolio-1',
            user_id: 'user-1',
            title: 'Scientific Research Fleet',
            objective: 'Continuously rank novel and testable ideas.',
            status: 'running',
            linked_profile_ids: ['profile-1'],
            automation_policy: {},
            opportunities: [
              {
                opportunity_id: 'opp-portfolio-1',
                canonical_key: 'compiler_hotspot',
                title: 'Compiler hotspot',
                hypothesis: 'Scheduler bottleneck',
                stage: 'validating',
                decision_state: 'accepted',
                decision_source: 'operator',
                confidence: 0.88,
                novelty: 0.71,
                readiness: 0.79,
                linked_experiment_plan_ids: ['plan-1'],
                linked_validation_run_ids: ['run-1'],
                latest_experiment_plan_id: 'plan-1',
                latest_validation_run_id: 'run-1',
                latest_validation_job_id: 'job-validation-1',
                latest_validation_status: 'queued',
                latest_validation_blocked_reason_code: null,
                source_note_ids: ['note-1'],
                child_job_ids: [],
              },
            ],
            latest_summary: { stage_counts: { discovered: 0, planned: 0, validating: 1, suppressed: 0 } },
            latest_note_ids: ['note-1'],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            latest_validation_runs: [],
            child_job_ids: [],
            active_job_id: 'job-portfolio-1',
            latest_run_job_id: 'job-portfolio-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:06:00Z',
          },
        ],
        total: 1,
      });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Research Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Run Experiment'));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenCalledWith('portfolio-1', 'opp-portfolio-1', {
        action: 'materialize_experiment',
        operator_note: undefined,
        start_immediately: true,
      });
    });

    expect(await screen.findByText('Open plan')).toBeInTheDocument();
    expect(await screen.findByText('Open run')).toBeInTheDocument();
  });

  it('lets admins create, edit, and delete scientific sandbox profiles', async () => {
    mockUseAuth.mockReturnValue({
      user: { id: 'user-1', username: 'admin', role: 'admin' },
      loading: false,
    });
    apiClient.listScientificSandboxProfiles.mockResolvedValue({
      items: [
        {
          id: 'scientific-compiler-sandbox',
          name: 'Compiler Validation Sandbox',
          description: 'Compiler sandbox',
          track_type: 'compiler',
          backend: 'docker',
          docker_image: 'ghcr.io/knowledgedb/compiler-research:latest',
          timeout_seconds: 1200,
          resource_caps: { memory_mb: 4096, cpus: 2, pids_limit: 256 },
          allowed_benchmark_families: ['compiler_regression'],
          allowed_perf_collectors: ['benchmark_output', 'compile_time'],
          required_capabilities: ['repo_reconstruction'],
          toolchains: ['clang', 'pytest'],
          budget_limit_default: 35,
          enabled: true,
          system_managed: true,
          is_default: true,
        },
        {
          id: 'custom-generic-sandbox',
          name: 'Custom Generic Sandbox',
          description: 'Custom sandbox',
          track_type: 'generic',
          backend: 'docker',
          docker_image: 'python:3.11-slim',
          timeout_seconds: 900,
          resource_caps: { memory_mb: 2048, cpus: 1.5, pids_limit: 192 },
          allowed_benchmark_families: ['generic_validation'],
          allowed_perf_collectors: ['benchmark_output'],
          required_capabilities: ['repo_reconstruction'],
          toolchains: ['python', 'pytest'],
          budget_limit_default: 25,
          enabled: true,
          system_managed: false,
          is_default: false,
        },
      ],
      total: 2,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Scientific Sandboxes')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Create custom sandbox profile'));
    fireEvent.change(screen.getByPlaceholderText('Profile id'), {
      target: { value: 'custom-new-sandbox' },
    });
    fireEvent.change(screen.getByPlaceholderText('Display name'), {
      target: { value: 'Custom New Sandbox' },
    });
    fireEvent.change(screen.getByPlaceholderText('Docker image'), {
      target: { value: 'python:3.11-slim' },
    });
    fireEvent.click(screen.getByText('Create Profile'));

    await waitFor(() => {
      expect(apiClient.createScientificSandboxProfile).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 'custom-new-sandbox',
          name: 'Custom New Sandbox',
          docker_image: 'python:3.11-slim',
        })
      );
    });

    const customCard = screen.getByText('Custom Generic Sandbox').closest('.border') as HTMLElement;
    fireEvent.click(within(customCard).getByText('Edit'));
    fireEvent.change(screen.getByPlaceholderText('Display name'), {
      target: { value: 'Custom Generic Sandbox v2' },
    });
    fireEvent.click(screen.getByText('Save Profile'));

    await waitFor(() => {
      expect(apiClient.updateScientificSandboxProfile).toHaveBeenCalledWith(
        'custom-generic-sandbox',
        expect.objectContaining({
          name: 'Custom Generic Sandbox v2',
        })
      );
    });

    fireEvent.click(within(customCard).getByText('Delete'));

    await waitFor(() => {
      expect(apiClient.deleteScientificSandboxProfile).toHaveBeenCalledWith('custom-generic-sandbox');
    });
  });

  it('renders recent scientific validation summaries on domain profiles', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: {
            follow_up_review_mode: 'queue_for_approval',
            confidence_threshold: 0.68,
            experiment_readiness_threshold: 0.72,
            auto_create_experiment_plans: true,
            auto_launch_follow_up: true,
            auto_launch_experiment_runs: true,
          },
          monitor_queries: ['llvm pass regression'],
          repo_source_ids: ['repo-source-1'],
          benchmark_queries: ['compile time regression'],
          report_format: 'brief_and_report',
          sandbox_profile_id: 'scientific-compiler-sandbox',
          interval_minutes: 1440,
          persist_artifacts: true,
          auto_launch_follow_up: true,
          auto_create_experiment_plans: true,
          confidence_threshold: 0.7,
          max_documents: 10,
          max_papers: 8,
          latest_summary: {
            autonomy_mode: 'max_autonomy',
            scheduler_summary: {
              next_run_at: '2026-03-25T12:00:00Z',
              pending_follow_up_approvals_count: 1,
              manual_follow_up_recommendations_count: 1,
              suppressed_relaunches_count: 2,
            },
            autonomy_state_counts: {
              eligible: 2,
              active: 1,
              completed_waiting_change: 1,
              blocked_structural: 1,
            },
            queued_operator_reviews_count: 2,
            pending_follow_up_approvals: [
              { opportunity_id: 'opp-1', title: 'Queued follow-up' },
            ],
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-2', title: 'Manual recommendation' },
            ],
            suppressed_relaunches: [
              { opportunity_id: 'opp-3', title: 'Rejected follow-up', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: ['run-1'],
          latest_validation_runs: [
            {
              id: 'run-1',
              name: 'Compiler Validation Run',
              status: 'blocked',
              progress: 100,
              recipe_family: 'compiler_validation',
              recipe_id: 'compiler_validation_v1',
              sandbox_profile_id: 'scientific-compiler-sandbox',
              sandbox_profile_name: 'Compiler Validation Sandbox',
              blocked_reason_code: 'disallowed_image',
              latest_operator_action: 'requeue',
              latest_operator_outcome_status: 'applied',
              retry_count: 2,
              parent_run_id: 'run-root-1',
              latest_child_run_id: 'run-child-1',
              created_at: '2026-03-24T12:00:00Z',
              completed_at: '2026-03-24T12:05:00Z',
            },
          ],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    expect(await screen.findByText('Compiler Validation Run')).toBeInTheDocument();
    expect(screen.getByText(/Recipe compiler_validation/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Sandbox Compiler Validation Sandbox/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Latest action: requeue · applied/i)).toBeInTheDocument();
    expect(screen.getByText(/Retry lineage · attempt 2 · parent run-root-1 · child run-child-1/i)).toBeInTheDocument();
    expect(screen.getByText(/Blocked: disallowed image/i)).toBeInTheDocument();
    expect(screen.getAllByText(/max autonomy/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Pending approvals/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Manual recommendations/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Suppressed relaunches/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Queued follow-up/i)).toBeInTheDocument();
  });

  it('creates a compiler regression explanation from a domain validation summary row', async () => {
    apiClient.createSynthesisJob.mockResolvedValueOnce({
      id: 'syn-job-1',
      job_type: 'compiler_regression_explanation',
      status: 'pending',
    });
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'balanced',
          effective_policy: {
            follow_up_review_mode: 'queue_for_approval',
          },
          repo_source_ids: ['repo-source-1'],
          benchmark_queries: ['compile time regression'],
          latest_summary: {},
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: ['run-1'],
          latest_validation_runs: [
            {
              id: 'run-1',
              agent_job_id: 'job-validation-1',
              name: 'Compiler Validation Run',
              status: 'completed',
              progress: 100,
              recipe_family: 'compiler_validation',
              recipe_id: 'compiler_validation_v1',
              benchmark_family: 'compiler_regression',
              benchmark_suite_id: 'compiler-llvm-regression-core',
              track_type: 'compiler',
              domain_research_profile_id: 'profile-1',
              sandbox_profile_id: 'scientific-compiler-sandbox',
              sandbox_profile_name: 'Compiler Validation Sandbox',
              created_at: '2026-03-24T12:00:00Z',
              completed_at: '2026-03-24T12:05:00Z',
              compiler_artifact_summary: {
                source_run_ids: ['run-1', 'run-0'],
                primary_run_id: 'run-1',
                comparison_run_id: 'run-0',
                available_actions: ['create_regression_explanation'],
              },
            },
          ],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    expect(await screen.findByText('Compiler artifact handoff')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Create explanation' }));

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
        job_type: 'compiler_regression_explanation',
        title: 'Compiler Validation Run Explanation',
        document_ids: [],
        experiment_run_ids: ['run-1', 'run-0'],
        primary_run_id: 'run-1',
        comparison_run_id: 'run-0',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });

  it('creates a compiler patch draft from a domain validation summary row using the profile repo source', async () => {
    apiClient.createSynthesisJob.mockResolvedValueOnce({
      id: 'syn-job-2',
      job_type: 'compiler_patch_draft',
      status: 'pending',
    });
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'balanced',
          effective_policy: {
            follow_up_review_mode: 'queue_for_approval',
          },
          repo_source_ids: ['repo-source-1'],
          benchmark_queries: ['compile time regression'],
          latest_summary: {},
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: ['run-1'],
          latest_validation_runs: [
            {
              id: 'run-1',
              agent_job_id: 'job-validation-1',
              name: 'Compiler Validation Run',
              status: 'completed',
              progress: 100,
              recipe_family: 'compiler_validation',
              recipe_id: 'compiler_validation_v1',
              benchmark_family: 'compiler_regression',
              benchmark_suite_id: 'compiler-llvm-regression-core',
              track_type: 'compiler',
              domain_research_profile_id: 'profile-1',
              sandbox_profile_id: 'scientific-compiler-sandbox',
              sandbox_profile_name: 'Compiler Validation Sandbox',
              created_at: '2026-03-24T12:00:00Z',
              completed_at: '2026-03-24T12:05:00Z',
              compiler_artifact_summary: {
                source_run_ids: ['run-1', 'run-0'],
                primary_run_id: 'run-1',
                comparison_run_id: 'run-0',
                proposal_note_id: 'note-proposal-1',
                available_actions: ['create_patch_draft'],
              },
            },
          ],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    fireEvent.click(screen.getByRole('button', { name: 'Create patch draft' }));

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
        job_type: 'compiler_patch_draft',
        title: 'Compiler Validation Run Patch Draft',
        document_ids: [],
        research_note_id: 'note-proposal-1',
        source_id: 'repo-source-1',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });

  it('saves a completed compiler artifact synthesis job as a research note from a validation row', async () => {
    apiClient.saveSynthesisJobAsResearchNote.mockResolvedValueOnce({
      id: 'note-expl-1',
      title: 'Compiler explanation note',
    });
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'balanced',
          effective_policy: {
            follow_up_review_mode: 'queue_for_approval',
          },
          repo_source_ids: ['repo-source-1'],
          benchmark_queries: ['compile time regression'],
          latest_summary: {},
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: ['run-1'],
          latest_validation_runs: [
            {
              id: 'run-1',
              agent_job_id: 'job-validation-1',
              name: 'Compiler Validation Run',
              status: 'completed',
              progress: 100,
              recipe_family: 'compiler_validation',
              recipe_id: 'compiler_validation_v1',
              benchmark_family: 'compiler_regression',
              benchmark_suite_id: 'compiler-llvm-regression-core',
              track_type: 'compiler',
              domain_research_profile_id: 'profile-1',
              sandbox_profile_id: 'scientific-compiler-sandbox',
              sandbox_profile_name: 'Compiler Validation Sandbox',
              created_at: '2026-03-24T12:00:00Z',
              completed_at: '2026-03-24T12:05:00Z',
              compiler_artifact_summary: {
                source_run_ids: ['run-1', 'run-0'],
                primary_run_id: 'run-1',
                comparison_run_id: 'run-0',
                explanation_synthesis_job_id: 'syn-job-1',
                explanation_synthesis_status: 'completed',
                available_actions: [],
              },
            },
          ],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    fireEvent.click(screen.getByRole('button', { name: 'Save explanation note' }));

    await waitFor(() => {
      expect(apiClient.saveSynthesisJobAsResearchNote).toHaveBeenCalledWith('syn-job-1');
    });
  });

  it('renders follow-up outcome metadata on domain opportunities', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          latest_summary: {
            autonomy_mode: 'max_autonomy',
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 0,
            },
          },
          opportunities: [
            {
              opportunity_id: 'opp-outcome-1',
              title: 'Compiler hotspot',
              stage: 'completed',
              autonomy_state: 'completed_waiting_change',
              confidence: 0.91,
              novelty: 0.72,
              readiness: 0.88,
              child_job_ids: ['job-follow-up-1'],
              follow_up_review_status: 'approved_launch',
              follow_up_outcome_status: 'completed',
              follow_up_outcome_recorded_at: '2026-03-25T12:00:00Z',
              follow_up_outcome_summary: 'Validated the hotspot and documented next steps.',
              follow_up_last_job_id: 'job-follow-up-1',
              last_reevaluation_review_outcome: 'applied_to_source_note',
              last_reevaluation_reviewed_at: '2026-03-25T11:30:00Z',
              last_reevaluation_review_job_id: 'syn-reeval-1',
              last_reevaluation_review_source_note_id: 'note-source-1',
              last_decision_reason_code: 'follow_up_completed',
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-25T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    expect(await screen.findByText(/Outcome completed/i)).toBeInTheDocument();
    expect(screen.getByText(/Validated the hotspot and documented next steps\./i)).toBeInTheDocument();
    expect(screen.getByText(/Job job-follow-up-1/i)).toBeInTheDocument();
    expect(screen.getByText(/Reevaluation applied to source note/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Open reevaluation job/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Open source note/i })).toBeInTheDocument();
  });

  it('renders reevaluation closeout trace actions and preset filtering', async () => {
    apiClient.getAgentDecisionTrace.mockResolvedValue({
      items: [
        {
          event_id: 'evt-reeval-1',
          event_type: 'reevaluation_saved_as_new_note',
          event_time: '2026-03-25T12:00:00Z',
          source_kind: 'domain_profile',
          source_id: 'profile-1',
          source_label: 'Compiler Frontier',
          decision_type: 'reevaluation_saved_as_new_note',
          summary: 'Compiler Frontier: reevaluation saved as new note for compiler hotspot',
          metadata: {
            opportunity_id: 'opp-compiler-1',
            source_note_id: 'note-source-1',
            target_note_id: 'note-saved-1',
            reevaluation_job_id: 'syn-reeval-1',
          },
          deep_link: {
            target_tab: 'domain',
            params: { tab: 'domain', profileId: 'profile-1', opportunityId: 'opp-compiler-1' },
            label: 'Open Domain Opportunity',
          },
          triage_status: 'new',
          pinned: false,
          is_derived: false,
        },
        {
          event_id: 'evt-generic-2',
          event_type: 'job_recovery_queued',
          event_time: '2026-03-25T11:00:00Z',
          source_kind: 'job',
          source_id: 'job-1',
          source_label: 'Generic Job',
          decision_type: 'job_recovery_queued',
          summary: 'Generic recovery event',
          triage_status: 'new',
          pinned: false,
          is_derived: true,
        },
      ],
      total: 2,
      limit: 100,
      offset: 0,
      by_source_kind: { domain_profile: 1, job: 1 },
      by_decision_type: { reevaluation_saved_as_new_note: 1, job_recovery_queued: 1 },
      by_status: {},
      by_customer: {},
      by_severity: { low: 1, medium: 1 },
      by_actor_mode: { operator: 1, autonomous: 1 },
      by_triage_status: { new: 2 },
      by_assignee: { unassigned: 2 },
      by_escalation_state: { none: 2 },
      overdue_count: 0,
      has_more: false,
    });

    await renderWithProviders('/autonomous-agents?tab=trace', { documentSources: defaultDocumentSources });

    expect(await screen.findByText(/reevaluation saved as new note for compiler hotspot/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Reevaluation closeouts/i }));
    await waitFor(() => {
      expect(screen.queryByText('Generic recovery event')).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/reevaluation saved as new note for compiler hotspot/i));
    expect(screen.getAllByRole('button', { name: /Open Domain Opportunity/i }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('button', { name: /Open reevaluation job/i }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('button', { name: /Open source note/i }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('button', { name: /Open saved note/i }).length).toBeGreaterThan(0);
  });

  it('approves pending domain follow-up recommendations inline', async () => {
    apiClient.listDomainResearchProfiles
      .mockResolvedValueOnce({
        items: [
          {
            id: 'profile-1',
            user_id: 'user-1',
            title: 'Compiler Frontier',
            domain: 'Compiler',
            objective: 'Track compiler opportunities',
            status: 'running',
            source_scope: 'kb_plus_arxiv_plus_repo',
            track_type: 'compiler',
            automation_profile: 'max_autonomy',
            effective_policy: { follow_up_review_mode: 'queue_for_approval' },
            latest_summary: {
              scheduler_summary: {
                pending_follow_up_approvals_count: 1,
                manual_follow_up_recommendations_count: 1,
              },
              pending_follow_up_approvals: [
                { opportunity_id: 'opp-approve', title: 'Queued compiler follow-up', reason_code: 'follow_up_pending_approval' },
              ],
              manual_follow_up_recommendations: [
                { opportunity_id: 'opp-manual', title: 'Manual compiler recommendation', reason_code: 'manual_follow_up_recommendation' },
              ],
            },
            opportunities: [],
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            latest_run_job_id: 'job-domain-1',
            active_job_id: 'job-domain-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'profile-1',
            user_id: 'user-1',
            title: 'Compiler Frontier',
            domain: 'Compiler',
            objective: 'Track compiler opportunities',
            status: 'running',
            source_scope: 'kb_plus_arxiv_plus_repo',
            track_type: 'compiler',
            automation_profile: 'max_autonomy',
            effective_policy: { follow_up_review_mode: 'queue_for_approval' },
            latest_summary: {
              scheduler_summary: {
                pending_follow_up_approvals_count: 0,
                manual_follow_up_recommendations_count: 1,
              },
              pending_follow_up_approvals: [],
              manual_follow_up_recommendations: [
                { opportunity_id: 'opp-manual', title: 'Manual compiler recommendation', reason_code: 'manual_follow_up_recommendation' },
              ],
            },
            opportunities: [],
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            latest_run_job_id: 'job-domain-1',
            active_job_id: 'job-domain-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      });
    apiClient.actionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      ok: true,
      detail: 'Follow-up launched',
      domain_research_profile_id: 'profile-1',
      profile_opportunity_id: 'opp-approve',
      follow_up_launch_status: 'launched',
      follow_up_operator_decision: 'approved_launch',
      follow_up_job_id: 'job-follow-1',
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    fireEvent.change(screen.getByLabelText('Operator note for Queued compiler follow-up'), {
      target: { value: 'Ship it now' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));

    await waitFor(() => {
      expect(apiClient.actionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: 'profile-1',
        profile_opportunity_id: 'opp-approve',
        portfolio_id: undefined,
        portfolio_opportunity_id: undefined,
        inbox_item_id: undefined,
        action: 'approve_launch',
        operator_note: 'Ship it now',
      });
    });
    await waitFor(() => {
      expect(screen.queryByText('Queued compiler follow-up')).not.toBeInTheDocument();
    });
  });

  it('bulk-approves pending domain follow-up recommendations within one profile card', async () => {
    apiClient.listDomainResearchProfiles
      .mockResolvedValueOnce({
        items: [
          {
            id: 'profile-1',
            user_id: 'user-1',
            title: 'Compiler Frontier',
            domain: 'Compiler',
            objective: 'Track compiler opportunities',
            status: 'running',
            source_scope: 'kb_plus_arxiv_plus_repo',
            track_type: 'compiler',
            automation_profile: 'max_autonomy',
            effective_policy: { follow_up_review_mode: 'queue_for_approval' },
            latest_summary: {
              scheduler_summary: { pending_follow_up_approvals_count: 2, manual_follow_up_recommendations_count: 0 },
              pending_follow_up_approvals: [
                { opportunity_id: 'opp-1', title: 'Queued compiler follow-up A', reason_code: 'follow_up_pending_approval' },
                { opportunity_id: 'opp-2', title: 'Queued compiler follow-up B', reason_code: 'follow_up_pending_approval' },
              ],
            },
            opportunities: [],
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            latest_run_job_id: 'job-domain-1',
            active_job_id: 'job-domain-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'profile-1',
            user_id: 'user-1',
            title: 'Compiler Frontier',
            domain: 'Compiler',
            objective: 'Track compiler opportunities',
            status: 'running',
            source_scope: 'kb_plus_arxiv_plus_repo',
            track_type: 'compiler',
            automation_profile: 'max_autonomy',
            effective_policy: { follow_up_review_mode: 'queue_for_approval' },
            latest_summary: {
              scheduler_summary: { pending_follow_up_approvals_count: 0, manual_follow_up_recommendations_count: 0 },
              pending_follow_up_approvals: [],
            },
            opportunities: [],
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            latest_run_job_id: 'job-domain-1',
            active_job_id: 'job-domain-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-1', ok: true, follow_up_launch_status: 'launched', follow_up_operator_decision: 'approved_launch', follow_up_job_id: 'job-1' },
        { domain_research_profile_id: 'profile-1', profile_opportunity_id: 'opp-2', ok: true, follow_up_launch_status: 'launched', follow_up_operator_decision: 'approved_launch', follow_up_job_id: 'job-2' },
      ],
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    fireEvent.click(await screen.findByText('Latest research ops state'));
    fireEvent.click(screen.getByLabelText('Select Queued compiler follow-up A'));
    fireEvent.click(screen.getByLabelText('Select Queued compiler follow-up B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk approve' } });
    fireEvent.click(screen.getByRole('button', { name: 'Approve Selected' }));

    await waitFor(() => {
      expect(apiClient.bulkActionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: 'profile-1',
        profile_opportunity_ids: ['opp-1', 'opp-2'],
        portfolio_id: undefined,
        portfolio_opportunity_ids: undefined,
        action: 'approve_launch',
        operator_note: 'Bulk approve',
      });
    });
  });

  it('deep-links directly to a domain opportunity row', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: { follow_up_review_mode: 'queue_for_approval' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
            },
          },
          opportunities: [
            {
              opportunity_id: 'opp-target',
              title: 'Target domain opportunity',
              stage: 'planned',
              confidence: 0.82,
              novelty: 0.74,
              readiness: 0.77,
              autonomy_state: 'eligible',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-target', {
      documentSources: defaultDocumentSources,
    });

    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    const row = await screen.findByText('Target domain opportunity');
    expect(row).toBeInTheDocument();
    expect(row.closest('div.border')?.className).toContain('border-cyan-300');
  });

  it('shows explainability details for blocked domain opportunities', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          latest_summary: {
            blocked_opportunities: [
              {
                opportunity_id: 'opp-blocked',
                title: 'Blocked compiler opportunity',
                last_blocked_reason_code: 'sandbox_policy_rejected',
              },
            ],
          },
          opportunities: [
            {
              opportunity_id: 'opp-blocked',
              title: 'Blocked compiler opportunity',
              stage: 'blocked',
              autonomy_state: 'blocked_structural',
              supporting_evidence: ['Benchmark crash in sandbox', 'Kernel capability denied'],
              last_blocked_reason_code: 'sandbox_policy_rejected',
              evidence_revision: 'rev-42',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));
    const blockedSection = screen.getByText('Blocked opportunities').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(blockedSection).getAllByRole('button', { name: /Show details/i })[0]);

    expect(await within(blockedSection).findByText(/Why blocked/i)).toBeInTheDocument();
    expect(within(blockedSection).getByText(/Reason:/i)).toBeInTheDocument();
    expect(within(blockedSection).getByText(/sandbox policy rejected/i)).toBeInTheDocument();
    expect(within(blockedSection).getByText(/Benchmark crash in sandbox/i)).toBeInTheDocument();
  });

  it('updates domain profile autonomy controls in place', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          research_mode: 'literature_to_hypothesis',
          report_format: 'brief_and_report',
          automation_profile: 'max_autonomy',
          effective_policy: {
            follow_up_review_mode: 'auto_launch_safe',
            confidence_threshold: 0.68,
            experiment_readiness_threshold: 0.72,
            auto_create_experiment_plans: true,
            auto_launch_follow_up: true,
            auto_launch_experiment_runs: true,
          },
          interval_minutes: 1440,
          persist_artifacts: true,
          auto_launch_follow_up: true,
          auto_create_experiment_plans: true,
          confidence_threshold: 0.68,
          max_documents: 10,
          max_papers: 8,
          latest_summary: { effective_policy: { follow_up_review_mode: 'auto_launch_safe', confidence_threshold: 0.68, experiment_readiness_threshold: 0.72, auto_create_experiment_plans: true, auto_launch_follow_up: true, auto_launch_experiment_runs: true } },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    const card = screen.getByText('Compiler Frontier').closest('.border') as HTMLElement;
    fireEvent.click(within(card).getByText('Latest research ops state'));

    const selects = within(card).getAllByRole('combobox');
    fireEvent.change(selects[0], { target: { value: 'balanced' } });
    fireEvent.change(selects[1], { target: { value: 'queue_for_approval' } });
    fireEvent.change(within(card).getByDisplayValue('0.68'), { target: { value: '0.74' } });

    fireEvent.click(within(card).getByText('Save'));

    await waitFor(() => {
      expect(apiClient.updateDomainResearchProfile).toHaveBeenCalledWith('profile-1', {
        automation_profile: 'balanced',
        automation_policy: expect.objectContaining({
          follow_up_review_mode: 'queue_for_approval',
          confidence_threshold: 0.74,
        }),
      });
    });
  });

  it('creates a research fleet portfolio with the expected payload', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Retrieval Monitor',
          domain: 'Multimodal retrieval',
          objective: 'Track retrieval opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv',
          track_type: 'generic',
          monitor_queries: ['retrieval benchmarks'],
          repo_source_ids: [],
          benchmark_queries: [],
          report_format: 'brief_and_report',
          automation_profile: 'balanced',
          automation_policy: {},
          effective_policy: {},
          interval_minutes: 1440,
          persist_artifacts: true,
          auto_launch_follow_up: true,
          auto_create_experiment_plans: true,
          confidence_threshold: 0.7,
          max_documents: 10,
          max_papers: 8,
          latest_summary: {},
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Start Fleet')).toBeInTheDocument();

    fireEvent.input(screen.getByPlaceholderText('Portfolio title'), {
      target: { value: 'Retrieval Experiment Portfolio' },
    });
    fireEvent.input(screen.getByPlaceholderText('Portfolio objective'), {
      target: { value: 'Continuously rank research ideas and convert the strongest ones into experiment plans.' },
    });
    fireEvent.click(screen.getByLabelText(/Retrieval Monitor/i));
    fireEvent.click(screen.getByText('Start Fleet'));

    await waitFor(() => {
      expect(apiClient.createResearchPortfolio).toHaveBeenCalledWith({
        title: 'Retrieval Experiment Portfolio',
        objective: 'Continuously rank research ideas and convert the strongest ones into experiment plans.',
        linked_profile_ids: ['profile-1'],
        automation_profile: 'balanced',
        automation_policy: {
          ...DEFAULT_VALIDATION_POLICY,
          duplicate_window_items: 120,
        },
        sandbox_profile_id: 'scientific-compiler-sandbox',
        start_immediately: true,
      });
    });
  });

  it('renders recent scientific validation summaries on research fleets', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [],
          latest_summary: {
            autonomy_mode: 'max_autonomy',
            autonomy_summary: {
              blocked_opportunities_count: 1,
              suppressed_duplicates_count: 2,
              created_experiment_plan_count: 3,
              launched_follow_up_job_count: 1,
            },
            scheduler_summary: {
              schedule_type: 'continuous',
              next_run_at: '2026-03-25T12:00:00Z',
              pending_follow_up_approvals_count: 1,
              manual_follow_up_recommendations_count: 1,
              suppressed_relaunches_count: 2,
            },
            autonomy_state_counts: {
              eligible: 2,
              cooldown: 1,
              completed_waiting_change: 1,
              blocked_structural: 1,
            },
            blocked_opportunities: [
              {
                opportunity_id: 'opp-1',
                title: 'Blocked opportunity',
                last_blocked_reason_code: 'backoff_cooldown',
              },
            ],
            cooldown_opportunities: [
              {
                opportunity_id: 'opp-2',
                title: 'Cooling opportunity',
                reason_code: 'backoff_cooldown',
              },
            ],
            completed_waiting_change_opportunities: [
              {
                opportunity_id: 'opp-3',
                title: 'Completed opportunity',
                reason_code: 'completed_current_evidence',
              },
            ],
            pending_follow_up_approvals: [
              {
                opportunity_id: 'opp-4',
                title: 'Queued follow-up',
                reason_code: 'follow_up_pending_approval',
              },
            ],
            manual_follow_up_recommendations: [
              {
                opportunity_id: 'opp-5',
                title: 'Manual follow-up',
                reason_code: 'manual_follow_up_recommendation',
              },
            ],
            suppressed_relaunches: [
              {
                opportunity_id: 'opp-6',
                title: 'Rejected follow-up',
                reason_code: 'operator_rejected_follow_up',
              },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: ['run-portfolio-1'],
          latest_validation_runs: [
            {
              id: 'run-portfolio-1',
              name: 'Microarchitecture Validation Run',
              status: 'running',
              progress: 55,
              recipe_family: 'microarchitecture_validation',
              recipe_id: 'microarchitecture_validation_v1',
              sandbox_profile_id: 'scientific-microarchitecture-sandbox',
              sandbox_profile_name: 'Microarchitecture Validation Sandbox',
              latest_operator_action: 'sync',
              latest_operator_outcome_status: 'applied',
              created_at: '2026-03-24T12:00:00Z',
            },
          ],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    expect(await screen.findByText('Microarchitecture Validation Run')).toBeInTheDocument();
    expect(screen.getByText(/Recipe microarchitecture_validation/i)).toBeInTheDocument();
    expect(screen.getByText(/Sandbox Microarchitecture Validation Sandbox/i)).toBeInTheDocument();
    expect(screen.getByText(/Latest action: sync · applied/i)).toBeInTheDocument();
    expect(screen.getByText(/max autonomy/i, { selector: 'span' })).toBeInTheDocument();
    expect(screen.getByText(/Blocked opportunity/i)).toBeInTheDocument();
    expect(screen.getByText(/Eligible now/i)).toBeInTheDocument();
    expect(screen.getByText(/Cooling down/i)).toBeInTheDocument();
    expect(screen.getByText(/Waiting on evidence change/i)).toBeInTheDocument();
    expect(screen.getByText(/Completed opportunity/i)).toBeInTheDocument();
    expect(screen.getByText(/Pending approvals/i)).toBeInTheDocument();
    expect(screen.getByText(/Manual recommendations/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Suppressed relaunches/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Queued follow-up/i)).toBeInTheDocument();
  });

  it('resolves follow-up outcome details for fleet summary rows from the matching opportunity', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-outcome-1',
              title: 'Cooling opportunity',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              confidence: 0.63,
              novelty: 0.51,
              readiness: 0.42,
              child_job_ids: ['job-follow-up-2'],
              follow_up_review_status: 'approved_launch',
              follow_up_outcome_status: 'failed',
              follow_up_outcome_recorded_at: '2026-03-26T12:00:00Z',
              follow_up_outcome_summary: 'Benchmark verification failed.',
              follow_up_last_job_id: 'job-follow-up-2',
              last_decision_reason_code: 'follow_up_failed',
            },
          ],
          latest_summary: {
            autonomy_mode: 'max_autonomy',
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 0,
            },
            cooldown_opportunities: [
              {
                opportunity_id: 'opp-fleet-outcome-1',
                title: 'Cooling opportunity',
                reason_code: 'recent_failed_follow_up',
              },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-26T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));
    fireEvent.click(screen.getAllByRole('button', { name: 'Show details' })[0]);

    expect((await screen.findAllByText(/Outcome:/i)).length).toBeGreaterThan(0);
    expect(screen.getByText(/Outcome failed/i)).toBeInTheDocument();
    expect((await screen.findAllByText(/Benchmark verification failed\./i)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/job-follow-up-2/i)).length).toBeGreaterThan(0);
  });

  it('relaunches a failed domain follow-up from the opportunity row', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValue({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          latest_summary: {},
          opportunities: [
            {
              opportunity_id: 'opp-domain-relaunch-1',
              title: 'Retry compiler hotspot',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'cooldown',
              confidence: 0.71,
              novelty: 0.58,
              readiness: 0.55,
              child_job_ids: ['job-follow-up-old'],
              follow_up_review_status: 'approved_launch',
              follow_up_outcome_status: 'failed',
              follow_up_outcome_recorded_at: '2026-03-26T12:00:00Z',
              follow_up_outcome_summary: 'Compiler benchmark failed.',
              follow_up_last_job_id: 'job-follow-up-old',
              last_decision_reason_code: 'follow_up_failed',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-26T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnDomainResearchOpportunity.mockResolvedValueOnce({
      profile: {
        id: 'profile-1',
        latest_summary: {},
        opportunities: [
          {
            opportunity_id: 'opp-domain-relaunch-1',
            follow_up_last_job_id: 'job-follow-up-new',
            follow_up_outcome_status: null,
          },
        ],
      },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch Follow-up' }));
    fireEvent.change(screen.getByLabelText('Domain relaunch note'), { target: { value: 'Retry with safer guardrails' } });
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch follow-up' }));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenCalledWith(
        'profile-1',
        'opp-domain-relaunch-1',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Retry with safer guardrails',
        })
      );
    });
  });

  it('relaunches a failed fleet follow-up from a summary row via the resolved opportunity', async () => {
    apiClient.listResearchPortfolios.mockResolvedValue({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-relaunch-1',
              title: 'Cooling opportunity',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              confidence: 0.63,
              novelty: 0.51,
              readiness: 0.42,
              child_job_ids: ['job-follow-up-2'],
              follow_up_review_status: 'approved_launch',
              follow_up_outcome_status: 'cancelled',
              follow_up_outcome_recorded_at: '2026-03-26T12:00:00Z',
              follow_up_outcome_summary: 'Operator cancelled verification.',
              follow_up_last_job_id: 'job-follow-up-2',
              last_decision_reason_code: 'follow_up_cancelled',
            },
          ],
          latest_summary: {
            autonomy_mode: 'max_autonomy',
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 0,
            },
            cooldown_opportunities: [
              {
                opportunity_id: 'opp-fleet-relaunch-1',
                title: 'Cooling opportunity',
                reason_code: 'recent_cancelled_follow_up',
              },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-26T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity.mockResolvedValueOnce({
      portfolio: {
        id: 'portfolio-1',
        latest_summary: {},
        opportunities: [
          {
            opportunity_id: 'opp-fleet-relaunch-1',
            follow_up_last_job_id: 'job-follow-up-3',
            follow_up_outcome_status: null,
          },
        ],
      },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));
    const cooldownSection = screen.getByText('Cooldown opportunities').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(cooldownSection).getAllByRole('button', { name: 'Show details' })[0]);
    fireEvent.click(within(cooldownSection).getByRole('button', { name: 'Relaunch Follow-up' }));
    fireEvent.change(within(cooldownSection).getAllByLabelText('Fleet relaunch note')[0], { target: { value: 'Retry the cancelled run' } });
    fireEvent.click(within(cooldownSection).getAllByRole('button', { name: 'Relaunch follow-up' })[0]);

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenCalledWith(
        'portfolio-1',
        'opp-fleet-relaunch-1',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Retry the cancelled run',
        })
      );
    });
  });

  it('launches a manual domain follow-up from the summary row via the resolved opportunity', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: { follow_up_review_mode: 'manual_only' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 1,
            },
            pending_follow_up_approvals: [],
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-manual-launch', title: 'Manual compiler recommendation', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          opportunities: [
            {
              opportunity_id: 'opp-manual-launch',
              title: 'Manual compiler recommendation',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              confidence: 0.78,
              novelty: 0.61,
              readiness: 0.74,
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnDomainResearchOpportunity.mockResolvedValueOnce({
      profile: { id: 'profile-1', latest_summary: {}, opportunities: [] },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    const manualSection = screen.getAllByText('Manual recommendations')[1].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByRole('button', { name: 'Follow-up' }));
    fireEvent.change(within(manualSection).getByLabelText('Domain follow-up note'), { target: { value: 'Launch the manual compiler check' } });
    fireEvent.click(within(manualSection).getByRole('button', { name: 'Launch follow-up' }));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenCalledWith(
        'profile-1',
        'opp-manual-launch',
        expect.objectContaining({
          action: 'launch_follow_up',
          operator_note: 'Launch the manual compiler check',
        })
      );
    });
  });

  it('launches a manual fleet follow-up from the summary row via the resolved opportunity', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-manual-launch',
              title: 'Manual fleet recommendation',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              confidence: 0.72,
              novelty: 0.57,
              readiness: 0.71,
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 1,
            },
            pending_follow_up_approvals: [],
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-fleet-manual-launch', title: 'Manual fleet recommendation', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity.mockResolvedValueOnce({
      portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByRole('button', { name: 'Follow-up' }));
    fireEvent.change(within(manualSection).getByLabelText('Fleet follow-up note'), { target: { value: 'Launch the manual fleet check' } });
    fireEvent.click(within(manualSection).getByRole('button', { name: 'Launch follow-up' }));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenCalledWith(
        'portfolio-1',
        'opp-fleet-manual-launch',
        expect.objectContaining({
          action: 'launch_follow_up',
          operator_note: 'Launch the manual fleet check',
        })
      );
    });
  });

  it('resolves a manual fleet summary row to relaunch when the linked opportunity is terminal', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-manual-relaunch',
              title: 'Manual fleet recommendation',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              confidence: 0.59,
              novelty: 0.41,
              readiness: 0.38,
              child_job_ids: ['job-follow-up-old'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-old',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 1,
            },
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-manual-relaunch', title: 'Manual fleet recommendation', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity.mockResolvedValueOnce({
      portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByRole('button', { name: 'Relaunch Follow-up' }));
    fireEvent.change(within(manualSection).getByLabelText('Fleet relaunch note'), { target: { value: 'Retry the failed manual run' } });
    fireEvent.click(within(manualSection).getByRole('button', { name: 'Relaunch follow-up' }));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenCalledWith(
        'portfolio-1',
        'opp-manual-relaunch',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Retry the failed manual run',
        })
      );
    });
  });

  it('bulk-launches resolved manual domain recommendations within one profile card', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: { follow_up_review_mode: 'manual_only' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 2,
            },
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-manual-bulk-a', title: 'Manual compiler recommendation A', reason_code: 'manual_follow_up_recommendation' },
              { opportunity_id: 'opp-manual-bulk-b', title: 'Manual compiler recommendation B', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          opportunities: [
            {
              opportunity_id: 'opp-manual-bulk-a',
              title: 'Manual compiler recommendation A',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-manual-bulk-b',
              title: 'Manual compiler recommendation B',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnDomainResearchOpportunity
      .mockResolvedValueOnce({ profile: { id: 'profile-1', latest_summary: {}, opportunities: [] } })
      .mockResolvedValueOnce({ profile: { id: 'profile-1', latest_summary: {}, opportunities: [] } });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    const manualSection = screen.getAllByText('Manual recommendations')[1].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByLabelText('Select Manual compiler recommendation A'));
    fireEvent.click(within(manualSection).getByLabelText('Select Manual compiler recommendation B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk launch domain manual recommendations' } });
    fireEvent.click(screen.getByRole('button', { name: 'Launch Selected' }));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenNthCalledWith(
        1,
        'profile-1',
        'opp-manual-bulk-a',
        expect.objectContaining({
          action: 'launch_follow_up',
          operator_note: 'Bulk launch domain manual recommendations',
        })
      );
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenNthCalledWith(
        2,
        'profile-1',
        'opp-manual-bulk-b',
        expect.objectContaining({
          action: 'launch_follow_up',
          operator_note: 'Bulk launch domain manual recommendations',
        })
      );
    });
  });

  it('bulk-launches resolved manual fleet recommendations within one fleet card', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-bulk-a',
              title: 'Manual fleet recommendation A',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-fleet-bulk-b',
              title: 'Manual fleet recommendation B',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 2,
            },
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-fleet-bulk-a', title: 'Manual fleet recommendation A', reason_code: 'manual_follow_up_recommendation' },
              { opportunity_id: 'opp-fleet-bulk-b', title: 'Manual fleet recommendation B', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity
      .mockResolvedValueOnce({ portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] } })
      .mockResolvedValueOnce({ portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] } });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet recommendation A'));
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet recommendation B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk launch fleet manual recommendations' } });
    fireEvent.click(screen.getByRole('button', { name: 'Launch Selected' }));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenNthCalledWith(
        1,
        'portfolio-1',
        'opp-fleet-bulk-a',
        expect.objectContaining({
          action: 'launch_follow_up',
          operator_note: 'Bulk launch fleet manual recommendations',
        })
      );
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenNthCalledWith(
        2,
        'portfolio-1',
        'opp-fleet-bulk-b',
        expect.objectContaining({
          action: 'launch_follow_up',
          operator_note: 'Bulk launch fleet manual recommendations',
        })
      );
    });
  });

  it('bulk-relaunches resolved manual fleet recommendations when all selected rows are terminal', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-relaunch-a',
              title: 'Manual fleet recommendation A',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-a'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-a',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-fleet-relaunch-b',
              title: 'Manual fleet recommendation B',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-b'],
              follow_up_outcome_status: 'cancelled',
              follow_up_last_job_id: 'job-follow-up-b',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 2,
            },
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-fleet-relaunch-a', title: 'Manual fleet recommendation A', reason_code: 'manual_follow_up_recommendation' },
              { opportunity_id: 'opp-fleet-relaunch-b', title: 'Manual fleet recommendation B', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity
      .mockResolvedValueOnce({ portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] } })
      .mockResolvedValueOnce({ portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] } });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet recommendation A'));
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet recommendation B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk relaunch fleet manual recommendations' } });
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch Selected' }));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenNthCalledWith(
        1,
        'portfolio-1',
        'opp-fleet-relaunch-a',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Bulk relaunch fleet manual recommendations',
        })
      );
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenNthCalledWith(
        2,
        'portfolio-1',
        'opp-fleet-relaunch-b',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Bulk relaunch fleet manual recommendations',
        })
      );
    });
  });

  it('disables bulk manual actions when selected recommendations mix launch and relaunch modes', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-launch',
              title: 'Manual fleet recommendation A',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-fleet-relaunch',
              title: 'Manual fleet recommendation B',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-b'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-b',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 2,
            },
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-fleet-launch', title: 'Manual fleet recommendation A', reason_code: 'manual_follow_up_recommendation' },
              { opportunity_id: 'opp-fleet-relaunch', title: 'Manual fleet recommendation B', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet recommendation A'));
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet recommendation B'));

    expect(screen.getByRole('button', { name: 'Launch Selected' })).toBeDisabled();
    expect(screen.getByText('Bulk manual follow-up actions cannot mix launch and relaunch selections.')).toBeInTheDocument();
  });

  it('rejects pending portfolio follow-up recommendations inline', async () => {
    apiClient.listResearchPortfolios
      .mockResolvedValueOnce({
        items: [
          {
            id: 'portfolio-1',
            user_id: 'user-1',
            title: 'Scientific Fleet',
            objective: 'Rank and validate scientific opportunities',
            status: 'running',
            linked_profile_ids: ['profile-1'],
            automation_profile: 'max_autonomy',
            automation_policy: {},
            sandbox_profile_id: 'scientific-generic-sandbox',
            opportunities: [],
            latest_summary: {
              scheduler_summary: {
                pending_follow_up_approvals_count: 1,
                manual_follow_up_recommendations_count: 1,
              },
              pending_follow_up_approvals: [
                { opportunity_id: 'opp-reject', title: 'Queued fleet follow-up', reason_code: 'follow_up_pending_approval' },
              ],
              manual_follow_up_recommendations: [
                { opportunity_id: 'opp-manual', title: 'Manual fleet recommendation', reason_code: 'manual_follow_up_recommendation' },
              ],
            },
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            child_job_ids: [],
            active_job_id: 'job-portfolio-1',
            latest_run_job_id: 'job-portfolio-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'portfolio-1',
            user_id: 'user-1',
            title: 'Scientific Fleet',
            objective: 'Rank and validate scientific opportunities',
            status: 'running',
            linked_profile_ids: ['profile-1'],
            automation_profile: 'max_autonomy',
            automation_policy: {},
            sandbox_profile_id: 'scientific-generic-sandbox',
            opportunities: [],
            latest_summary: {
              scheduler_summary: {
                pending_follow_up_approvals_count: 0,
                manual_follow_up_recommendations_count: 1,
              },
              pending_follow_up_approvals: [],
              manual_follow_up_recommendations: [
                { opportunity_id: 'opp-manual', title: 'Manual fleet recommendation', reason_code: 'manual_follow_up_recommendation' },
              ],
            },
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            child_job_ids: [],
            active_job_id: 'job-portfolio-1',
            latest_run_job_id: 'job-portfolio-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      });
    apiClient.actionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      ok: true,
      detail: 'Follow-up rejected',
      portfolio_id: 'portfolio-1',
      portfolio_opportunity_id: 'opp-reject',
      follow_up_launch_status: 'rejected',
      follow_up_operator_decision: 'rejected',
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    fireEvent.change(screen.getByLabelText('Operator note for Queued fleet follow-up'), {
      target: { value: 'Not enough evidence yet' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Reject' }));

    await waitFor(() => {
      expect(apiClient.actionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: undefined,
        profile_opportunity_id: undefined,
        portfolio_id: 'portfolio-1',
        portfolio_opportunity_id: 'opp-reject',
        inbox_item_id: undefined,
        action: 'reject_launch',
        operator_note: 'Not enough evidence yet',
      });
    });
    await waitFor(() => {
      expect(screen.queryByText('Queued fleet follow-up')).not.toBeInTheDocument();
    });
  });

  it('bulk-rejects pending portfolio follow-up recommendations within one fleet card', async () => {
    apiClient.listResearchPortfolios
      .mockResolvedValueOnce({
        items: [
          {
            id: 'portfolio-1',
            user_id: 'user-1',
            title: 'Scientific Fleet',
            objective: 'Rank and validate scientific opportunities',
            status: 'running',
            linked_profile_ids: ['profile-1'],
            automation_profile: 'max_autonomy',
            automation_policy: {},
            sandbox_profile_id: 'scientific-generic-sandbox',
            opportunities: [],
            latest_summary: {
              scheduler_summary: { pending_follow_up_approvals_count: 2, manual_follow_up_recommendations_count: 0 },
              pending_follow_up_approvals: [
                { opportunity_id: 'opp-a', title: 'Queued fleet follow-up A', reason_code: 'follow_up_pending_approval' },
                { opportunity_id: 'opp-b', title: 'Queued fleet follow-up B', reason_code: 'follow_up_pending_approval' },
              ],
            },
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            child_job_ids: [],
            active_job_id: 'job-portfolio-1',
            latest_run_job_id: 'job-portfolio-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'portfolio-1',
            user_id: 'user-1',
            title: 'Scientific Fleet',
            objective: 'Rank and validate scientific opportunities',
            status: 'running',
            linked_profile_ids: ['profile-1'],
            automation_profile: 'max_autonomy',
            automation_policy: {},
            sandbox_profile_id: 'scientific-generic-sandbox',
            opportunities: [],
            latest_summary: {
              scheduler_summary: { pending_follow_up_approvals_count: 0, manual_follow_up_recommendations_count: 0 },
              pending_follow_up_approvals: [],
            },
            latest_note_ids: [],
            latest_experiment_plan_ids: [],
            latest_validation_run_ids: [],
            child_job_ids: [],
            active_job_id: 'job-portfolio-1',
            latest_run_job_id: 'job-portfolio-1',
            created_at: '2026-03-24T12:00:00Z',
            updated_at: '2026-03-24T12:00:00Z',
          },
        ],
        total: 1,
      });
    apiClient.bulkActionAgentCheckpointQueueFollowUp.mockResolvedValueOnce({
      requested_count: 2,
      applied: 2,
      failed: 0,
      results: [
        { portfolio_id: 'portfolio-1', portfolio_opportunity_id: 'opp-a', ok: true, follow_up_launch_status: 'rejected', follow_up_operator_decision: 'rejected' },
        { portfolio_id: 'portfolio-1', portfolio_opportunity_id: 'opp-b', ok: true, follow_up_launch_status: 'rejected', follow_up_operator_decision: 'rejected' },
      ],
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    fireEvent.click(await screen.findByText('Portfolio state'));
    fireEvent.click(screen.getByLabelText('Select Queued fleet follow-up A'));
    fireEvent.click(screen.getByLabelText('Select Queued fleet follow-up B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk reject' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reject Selected' }));

    await waitFor(() => {
      expect(apiClient.bulkActionAgentCheckpointQueueFollowUp).toHaveBeenCalledWith({
        domain_research_profile_id: undefined,
        profile_opportunity_ids: undefined,
        portfolio_id: 'portfolio-1',
        portfolio_opportunity_ids: ['opp-a', 'opp-b'],
        action: 'reject_launch',
        operator_note: 'Bulk reject',
      });
    });
  });

  it('relaunches a suppressed domain follow-up from the summary row via the resolved opportunity', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: { follow_up_review_mode: 'manual_only' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 1,
            },
            suppressed_relaunches: [
              { opportunity_id: 'opp-domain-suppressed', title: 'Suppressed compiler relaunch', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          opportunities: [
            {
              opportunity_id: 'opp-domain-suppressed',
              title: 'Suppressed compiler relaunch',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-old'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-old',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnDomainResearchOpportunity.mockResolvedValueOnce({
      profile: { id: 'profile-1', latest_summary: {}, opportunities: [] },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(suppressedSection).getByRole('button', { name: 'Relaunch Follow-up' }));
    fireEvent.change(within(suppressedSection).getByLabelText('Domain relaunch note'), { target: { value: 'Retry the suppressed compiler follow-up' } });
    fireEvent.click(within(suppressedSection).getByRole('button', { name: 'Relaunch follow-up' }));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenCalledWith(
        'profile-1',
        'opp-domain-suppressed',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Retry the suppressed compiler follow-up',
        })
      );
    });
  });

  it('relaunches a suppressed fleet follow-up from the summary row via the resolved opportunity', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-suppressed',
              title: 'Suppressed fleet relaunch',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-old'],
              follow_up_outcome_status: 'cancelled',
              follow_up_last_job_id: 'job-follow-up-old',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 1,
            },
            suppressed_relaunches: [
              { opportunity_id: 'opp-fleet-suppressed', title: 'Suppressed fleet relaunch', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity.mockResolvedValueOnce({
      portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] },
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(suppressedSection).getByRole('button', { name: 'Relaunch Follow-up' }));
    fireEvent.change(within(suppressedSection).getByLabelText('Fleet relaunch note'), { target: { value: 'Retry the suppressed fleet follow-up' } });
    fireEvent.click(within(suppressedSection).getByRole('button', { name: 'Relaunch follow-up' }));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenCalledWith(
        'portfolio-1',
        'opp-fleet-suppressed',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Retry the suppressed fleet follow-up',
        })
      );
    });
  });

  it('keeps unresolved suppressed relaunch rows informational', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 1,
            },
            suppressed_relaunches: [
              { opportunity_id: 'opp-fleet-suppressed', title: 'Suppressed fleet relaunch', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    expect(within(suppressedSection).getByText(/Suppressed fleet relaunch/i)).toBeInTheDocument();
    expect(within(suppressedSection).queryByRole('button', { name: 'Relaunch Follow-up' })).not.toBeInTheDocument();
  });

  it('keeps resolved but non-relaunchable suppressed rows read-only', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: { follow_up_review_mode: 'manual_only' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 1,
            },
            suppressed_relaunches: [
              { opportunity_id: 'opp-domain-suppressed-readonly', title: 'Suppressed compiler relaunch', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          opportunities: [
            {
              opportunity_id: 'opp-domain-suppressed-readonly',
              title: 'Suppressed compiler relaunch',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: ['job-follow-up-old'],
              follow_up_outcome_status: 'completed',
              follow_up_last_job_id: 'job-follow-up-old',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    expect(within(suppressedSection).getByText(/Suppressed compiler relaunch/i)).toBeInTheDocument();
    expect(within(suppressedSection).queryByRole('button', { name: 'Relaunch Follow-up' })).not.toBeInTheDocument();
  });

  it('bulk-relaunches resolved suppressed domain rows within one profile card', async () => {
    apiClient.listDomainResearchProfiles.mockResolvedValueOnce({
      items: [
        {
          id: 'profile-1',
          user_id: 'user-1',
          title: 'Compiler Frontier',
          domain: 'Compiler',
          objective: 'Track compiler opportunities',
          status: 'running',
          source_scope: 'kb_plus_arxiv_plus_repo',
          track_type: 'compiler',
          automation_profile: 'max_autonomy',
          effective_policy: { follow_up_review_mode: 'manual_only' },
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 2,
            },
            suppressed_relaunches: [
              { opportunity_id: 'opp-domain-suppressed-a', title: 'Suppressed compiler relaunch A', reason_code: 'operator_rejected_follow_up' },
              { opportunity_id: 'opp-domain-suppressed-b', title: 'Suppressed compiler relaunch B', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          opportunities: [
            {
              opportunity_id: 'opp-domain-suppressed-a',
              title: 'Suppressed compiler relaunch A',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-a'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-a',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-domain-suppressed-b',
              title: 'Suppressed compiler relaunch B',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-b'],
              follow_up_outcome_status: 'cancelled',
              follow_up_last_job_id: 'job-follow-up-b',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_run_job_id: 'job-domain-1',
          active_job_id: 'job-domain-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnDomainResearchOpportunity
      .mockResolvedValueOnce({ profile: { id: 'profile-1', latest_summary: {}, opportunities: [] } })
      .mockResolvedValueOnce({ profile: { id: 'profile-1', latest_summary: {}, opportunities: [] } });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Domain Profiles'));
    expect(await screen.findByText('Compiler Frontier')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Latest research ops state'));

    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(suppressedSection).getByLabelText('Select Suppressed compiler relaunch A'));
    fireEvent.click(within(suppressedSection).getByLabelText('Select Suppressed compiler relaunch B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk relaunch suppressed domain rows' } });
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch Selected' }));

    await waitFor(() => {
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenNthCalledWith(
        1,
        'profile-1',
        'opp-domain-suppressed-a',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Bulk relaunch suppressed domain rows',
        })
      );
      expect(apiClient.actOnDomainResearchOpportunity).toHaveBeenNthCalledWith(
        2,
        'profile-1',
        'opp-domain-suppressed-b',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Bulk relaunch suppressed domain rows',
        })
      );
    });
  });

  it('bulk-relaunches resolved suppressed fleet rows within one fleet card', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-suppressed-a',
              title: 'Suppressed fleet relaunch A',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-a'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-a',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-fleet-suppressed-b',
              title: 'Suppressed fleet relaunch B',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-b'],
              follow_up_outcome_status: 'cancelled',
              follow_up_last_job_id: 'job-follow-up-b',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
              suppressed_relaunches_count: 2,
            },
            suppressed_relaunches: [
              { opportunity_id: 'opp-fleet-suppressed-a', title: 'Suppressed fleet relaunch A', reason_code: 'operator_rejected_follow_up' },
              { opportunity_id: 'opp-fleet-suppressed-b', title: 'Suppressed fleet relaunch B', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });
    apiClient.actOnResearchPortfolioOpportunity
      .mockResolvedValueOnce({ portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] } })
      .mockResolvedValueOnce({ portfolio: { id: 'portfolio-1', latest_summary: {}, opportunities: [] } });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(suppressedSection).getByLabelText('Select Suppressed fleet relaunch A'));
    fireEvent.click(within(suppressedSection).getByLabelText('Select Suppressed fleet relaunch B'));
    fireEvent.change(screen.getByPlaceholderText('Shared operator note (optional)'), { target: { value: 'Bulk relaunch suppressed fleet rows' } });
    fireEvent.click(screen.getByRole('button', { name: 'Relaunch Selected' }));

    await waitFor(() => {
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenNthCalledWith(
        1,
        'portfolio-1',
        'opp-fleet-suppressed-a',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Bulk relaunch suppressed fleet rows',
        })
      );
      expect(apiClient.actOnResearchPortfolioOpportunity).toHaveBeenNthCalledWith(
        2,
        'portfolio-1',
        'opp-fleet-suppressed-b',
        expect.objectContaining({
          action: 'relaunch_follow_up',
          operator_note: 'Bulk relaunch suppressed fleet rows',
        })
      );
    });
  });

  it('disables bulk follow-up actions when selected rows mix manual launch and suppressed relaunch selections', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-fleet-manual-launch',
              title: 'Manual fleet launch',
              stage: 'accepted',
              decision_state: 'accepted',
              autonomy_state: 'eligible',
              child_job_ids: [],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
            {
              opportunity_id: 'opp-fleet-suppressed-relaunch',
              title: 'Suppressed fleet relaunch',
              stage: 'blocked',
              autonomy_state: 'cooldown',
              child_job_ids: ['job-follow-up-b'],
              follow_up_outcome_status: 'failed',
              follow_up_last_job_id: 'job-follow-up-b',
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 1,
              suppressed_relaunches_count: 1,
            },
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-fleet-manual-launch', title: 'Manual fleet launch', reason_code: 'manual_follow_up_recommendation' },
            ],
            suppressed_relaunches: [
              { opportunity_id: 'opp-fleet-suppressed-relaunch', title: 'Suppressed fleet relaunch', reason_code: 'operator_rejected_follow_up' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    const suppressedSection = screen.getAllByText('Suppressed relaunches').slice(-1)[0].closest('.bg-white') as HTMLElement;
    fireEvent.click(within(manualSection).getByLabelText('Select Manual fleet launch'));
    fireEvent.click(within(suppressedSection).getByLabelText('Select Suppressed fleet relaunch'));

    expect(screen.getByRole('button', { name: 'Launch Selected' })).toBeDisabled();
    expect(screen.getByText('Bulk manual follow-up actions cannot mix launch and relaunch selections.')).toBeInTheDocument();
  });

  it('deep-links to a fleet review row when no detailed opportunity row is present', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 1,
            },
            pending_follow_up_approvals: [],
            manual_follow_up_recommendations: [
              {
                opportunity_id: 'opp-manual-target',
                title: 'Manual fleet recommendation',
                reason_code: 'manual_follow_up_recommendation',
                operator_note: 'Needs human review',
              },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents?tab=fleet&fleetId=portfolio-1&opportunityId=opp-manual-target', {
      documentSources: defaultDocumentSources,
    });

    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    const row = await screen.findByText(/Manual fleet recommendation/i);
    expect(row).toBeInTheDocument();
    const manualRow = row.closest('.border') as HTMLElement;
    expect(manualRow?.className).toContain('border-cyan-300');
    expect(within(manualRow).getByText(/Reason:/i)).toBeInTheDocument();
  });

  it('keeps unresolved manual follow-up recommendations informational', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 1,
            },
            pending_follow_up_approvals: [],
            manual_follow_up_recommendations: [
              { opportunity_id: 'opp-manual', title: 'Manual fleet recommendation', reason_code: 'manual_follow_up_recommendation' },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    const manualSection = screen.getByText('Manual follow-up recommendations').closest('.bg-white') as HTMLElement;
    expect(within(manualSection).getByText(/Manual fleet recommendation/i)).toBeInTheDocument();
    expect(within(manualSection).queryByRole('button', { name: /Approve/i })).not.toBeInTheDocument();
    expect(within(manualSection).queryByRole('button', { name: /Reject/i })).not.toBeInTheDocument();
    expect(within(manualSection).queryByLabelText('Select Manual fleet recommendation')).not.toBeInTheDocument();
    expect(within(manualSection).queryByRole('button', { name: 'Follow-up' })).not.toBeInTheDocument();
    expect(within(manualSection).queryByRole('button', { name: 'Relaunch Follow-up' })).not.toBeInTheDocument();
  });

  it('shows explainability details for waiting fleet opportunities from summary rows', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {},
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [
            {
              opportunity_id: 'opp-wait',
              title: 'Completed fleet opportunity',
              stage: 'completed',
              autonomy_state: 'completed_waiting_change',
              last_decision_reason_code: 'completed_current_evidence',
              next_eligible_at: '2026-03-26T12:00:00Z',
              supporting_evidence: ['All current evidence exhausted'],
              linked_experiment_plan_ids: [],
              linked_validation_run_ids: [],
              child_job_ids: ['job-child-1'],
            },
          ],
          latest_summary: {
            scheduler_summary: {
              pending_follow_up_approvals_count: 0,
              manual_follow_up_recommendations_count: 0,
            },
            completed_waiting_change_opportunities: [
              {
                opportunity_id: 'opp-wait',
                title: 'Completed fleet opportunity',
                reason_code: 'completed_current_evidence',
              },
            ],
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));
    const waitingSection = screen.getByText('Waiting on evidence change').closest('.bg-white') as HTMLElement;
    fireEvent.click(within(waitingSection).getAllByRole('button', { name: /Show details/i })[0]);

    expect(await within(waitingSection).findByText(/Reason:/i)).toBeInTheDocument();
    expect(within(waitingSection).getByText(/Next eligible:/i)).toBeInTheDocument();
    expect(within(waitingSection).getByText(/All current evidence exhausted/i)).toBeInTheDocument();
    expect(within(waitingSection).getByText(/job-child-1/i)).toBeInTheDocument();
  });

  it('updates research fleet autonomy controls in place', async () => {
    apiClient.listResearchPortfolios.mockResolvedValueOnce({
      items: [
        {
          id: 'portfolio-1',
          user_id: 'user-1',
          title: 'Scientific Fleet',
          objective: 'Rank and validate scientific opportunities',
          status: 'running',
          linked_profile_ids: ['profile-1'],
          automation_profile: 'max_autonomy',
          automation_policy: {
            ...DEFAULT_VALIDATION_POLICY,
            auto_execute_validation_runs: true,
            auto_launch_experiment_runs: true,
            confidence_threshold: 0.68,
            experiment_readiness_threshold: 0.72,
            max_auto_follow_up_launches: 4,
            max_concurrent_validation_runs: 2,
            max_validation_runtime_minutes: 30,
            max_validation_budget_per_run: 50,
            duplicate_window_items: 120,
          },
          effective_policy: {
            ...DEFAULT_VALIDATION_POLICY,
            auto_execute_validation_runs: true,
            auto_launch_experiment_runs: true,
            confidence_threshold: 0.68,
            experiment_readiness_threshold: 0.72,
            max_auto_follow_up_launches: 4,
            max_concurrent_validation_runs: 2,
            max_validation_runtime_minutes: 30,
            max_validation_budget_per_run: 50,
            duplicate_window_items: 120,
          },
          sandbox_profile_id: 'scientific-generic-sandbox',
          opportunities: [],
          latest_summary: {
            effective_policy: {
              ...DEFAULT_VALIDATION_POLICY,
              auto_execute_validation_runs: true,
              auto_launch_experiment_runs: true,
              confidence_threshold: 0.68,
              experiment_readiness_threshold: 0.72,
              max_auto_follow_up_launches: 4,
              max_concurrent_validation_runs: 2,
              max_validation_runtime_minutes: 30,
              max_validation_budget_per_run: 50,
              duplicate_window_items: 120,
            },
          },
          latest_note_ids: [],
          latest_experiment_plan_ids: [],
          latest_validation_run_ids: [],
          latest_validation_runs: [],
          child_job_ids: [],
          active_job_id: 'job-portfolio-1',
          latest_run_job_id: 'job-portfolio-1',
          created_at: '2026-03-24T12:00:00Z',
          updated_at: '2026-03-24T12:00:00Z',
        },
      ],
      total: 1,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Research Fleet'));
    expect(await screen.findByText('Scientific Fleet')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Portfolio state'));

    fireEvent.change(screen.getByDisplayValue('0.68'), { target: { value: '0.75' } });
    fireEvent.click(screen.getByLabelText(/Auto-launch validation/i));
    fireEvent.click(screen.getByText('Apply settings'));

    await waitFor(() => {
      expect(apiClient.updateResearchPortfolio).toHaveBeenCalledWith('portfolio-1', {
        automation_profile: 'max_autonomy',
        automation_policy: {
          follow_up_review_mode: 'auto_launch_safe',
          confidence_threshold: 0.75,
          experiment_readiness_threshold: 0.72,
          max_auto_follow_up_launches: 4,
          max_concurrent_validation_runs: 2,
          max_validation_runtime_minutes: 30,
          max_validation_budget_per_run: 50,
          duplicate_window_items: 120,
          auto_create_experiment_plans: true,
          auto_launch_follow_up: true,
          auto_launch_experiment_runs: false,
          auto_execute_validation_runs: false,
        },
      });
    });
  });

  it('renders backlog orchestration progress and promotion history', async () => {
    apiClient.listCodingBacklogItems.mockResolvedValueOnce({
      items: [
        {
          id: 'backlog-2',
          user_id: 'user-1',
          source_id: 'repo-source-1',
          title: 'Stabilize save pipeline',
          portfolio_goal: 'Drive the save pipeline through bounded repair slices and auto-apply low-risk fixes.',
          status: 'running',
          priority: 70,
          scope: 'frontend',
          failure_symptom: 'Save requests intermittently return 500.',
          error_output: null,
          file_paths: ['frontend/src/pages/DocumentsPage.tsx', 'frontend/src/services/api.ts'],
          commands: ['npm --prefix frontend run build'],
          auto_apply_enabled: true,
          require_patch_pr: false,
          policy: {
            max_auto_retries: 1,
            max_files_touched: 3,
            require_experiments_ok: true,
            confidence_threshold: 0.55,
            blocked_path_prefixes: [],
          },
          decomposition: {
            strategy: 'portfolio_goal',
            active_slice_id: 'slice_2',
            backlog_timeline: [
              {
                at: '2026-03-23T12:00:00Z',
                actor: 'system',
                action: 'orchestrator_started',
                job_id: 'job-orch',
              },
              {
                at: '2026-03-23T12:05:00Z',
                actor: 'system',
                action: 'auto_apply_completed',
                slice_id: 'slice_1',
                job_id: 'job-a',
              },
            ],
            lineage_summary: {
              repair_job_count: 2,
              apply_job_count: 1,
              patch_pr_count: 0,
              proposal_count: 1,
              operator_action_count: 0,
            },
            planned_slices: [
              {
                slice_id: 'slice_1',
                title: 'Target DocumentsPage.tsx',
                status: 'auto_applied',
                scope: 'frontend',
                retry_count: 0,
                promotion_decision: 'auto_applied',
                proposal_confidence: 0.9,
                file_paths: ['frontend/src/pages/DocumentsPage.tsx'],
                timeline: [
                  {
                    at: '2026-03-23T12:01:00Z',
                    actor: 'system',
                    action: 'repair_job_started',
                    job_id: 'job-a',
                  },
                  {
                    at: '2026-03-23T12:04:00Z',
                    actor: 'system',
                    action: 'auto_apply_completed',
                    job_id: 'job-a',
                  },
                ],
                job_lineage: {
                  repair_job_ids: ['job-a'],
                  apply_job_ids: ['job-apply-a'],
                  patch_pr_ids: [],
                  proposal_ids: ['proposal-a'],
                  retry_from_job_ids: [],
                },
                artifact_history: [
                  {
                    artifact_type: 'proposal',
                    artifact_id: 'proposal-a',
                    label: 'Selected proposal',
                    at: '2026-03-23T12:03:00Z',
                  },
                ],
              },
              {
                slice_id: 'slice_2',
                title: 'Target api.ts',
                status: 'repairing',
                scope: 'frontend',
                retry_count: 1,
                file_paths: ['frontend/src/services/api.ts'],
                timeline: [
                  {
                    at: '2026-03-23T12:06:00Z',
                    actor: 'system',
                    action: 'repair_job_started',
                    job_id: 'job-b',
                  },
                ],
                job_lineage: {
                  repair_job_ids: ['job-b'],
                  apply_job_ids: [],
                  patch_pr_ids: [],
                  proposal_ids: [],
                  retry_from_job_ids: ['job-a'],
                },
              },
            ],
            completed_slices: ['slice_1'],
            failed_slices: [],
            promotion_decisions: [
              {
                slice_id: 'slice_1',
                title: 'Target DocumentsPage.tsx',
                decision: 'auto_applied',
                proposal_confidence: 0.9,
                files_touched_count: 1,
              },
            ],
            portfolio_progress: {
              total_slices: 2,
              pending_slices: 1,
              completed_slices: 1,
              failed_slices: 0,
              auto_applied_slices: 1,
              proposal_only_slices: 0,
            },
          },
          child_job_ids: ['job-a', 'job-b'],
          latest_summary: {
            status: 'repair_started',
            current_child_job_id: 'job-b',
            promotion_decision: 'auto_applied',
            active_slice_id: 'slice_2',
            active_slice_title: 'Target api.ts',
            portfolio_progress: {
              total_slices: 2,
              pending_slices: 1,
              completed_slices: 1,
              failed_slices: 0,
              auto_applied_slices: 1,
              proposal_only_slices: 0,
            },
            promotion_evaluation: {
              decision: 'auto_applied',
              proposal_confidence: 0.9,
              files_touched_count: 1,
              experiment_ok: true,
            },
          },
          orchestrator_job_id: 'job-orch',
          current_job_id: 'job-b',
          latest_apply_job_id: 'job-a',
          latest_proposal_id: 'proposal-a',
          created_at: '2026-03-23T12:00:00Z',
          updated_at: '2026-03-23T12:10:00Z',
          started_at: '2026-03-23T12:00:00Z',
          completed_at: null,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Coding Backlog'));
    expect(await screen.findByText('Stabilize save pipeline')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Orchestration detail'));

    expect(await screen.findByText('Portfolio progress')).toBeInTheDocument();
    expect(screen.getByText('1/2 completed')).toBeInTheDocument();
    expect(screen.getByText('Auto-applied 1')).toBeInTheDocument();
    expect(screen.getByText('Target DocumentsPage.tsx')).toBeInTheDocument();
    expect(screen.getByText('Target api.ts')).toBeInTheDocument();
    expect(screen.getByText(/Target DocumentsPage.tsx: auto applied/i)).toBeInTheDocument();
    expect(screen.getByText('Backlog timeline')).toBeInTheDocument();
    expect(screen.getByText(/Lineage summary/i)).toBeInTheDocument();
    fireEvent.click(screen.getAllByText('Slice timeline')[0]);
    expect((await screen.findAllByText(/repair job started/i)).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByText('Artifacts and lineage')[0]);
    expect(await screen.findByText(/Repair jobs: job-a/i)).toBeInTheDocument();
    expect(screen.getByText(/Selected proposal · proposal-a/i)).toBeInTheDocument();
    expect(screen.getAllByText('Copy ID').length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByText('Open Job')[0]);
    await waitFor(() => {
      expect(apiClient.getAgentJob).toHaveBeenCalledWith('job-a');
    });
  });

  it('sends slice-level operator actions from the backlog tab', async () => {
    apiClient.listCodingBacklogItems.mockResolvedValueOnce({
      items: [
        {
          id: 'backlog-3',
          user_id: 'user-1',
          source_id: 'repo-source-1',
          title: 'Manual promotion backlog',
          portfolio_goal: 'Stop for operator choice when a slice is blocked.',
          status: 'awaiting_operator',
          priority: 60,
          scope: 'frontend',
          failure_symptom: 'Save flow changed after a risky refactor.',
          error_output: null,
          file_paths: ['frontend/src/pages/DocumentsPage.tsx'],
          commands: ['npm --prefix frontend run build'],
          auto_apply_enabled: true,
          require_patch_pr: false,
          policy: { max_auto_retries: 1, max_files_touched: 3, require_experiments_ok: true, confidence_threshold: 0.55 },
          decomposition: {
            strategy: 'portfolio_goal',
            active_slice_id: null,
            backlog_timeline: [
              {
                at: '2026-03-23T12:08:00Z',
                actor: 'system',
                action: 'awaiting_operator',
                slice_id: 'slice_1',
                job_id: 'job-operator-1',
              },
            ],
            lineage_summary: {
              repair_job_count: 1,
              apply_job_count: 0,
              patch_pr_count: 0,
              proposal_count: 1,
              operator_action_count: 1,
            },
            planned_slices: [
              {
                slice_id: 'slice_1',
                title: 'Target DocumentsPage.tsx',
                status: 'proposal_only',
                scope: 'frontend',
                selected_proposal_id: 'proposal-1',
                blocked_reason: 'confidence_below_threshold',
                awaiting_operator_action: true,
                allowed_slice_actions: ['apply_override', 'create_patch_pr', 'keep_proposal_only', 'relaunch_slice', 'skip_slice'],
                recommended_next_action: 'apply_override',
                timeline: [
                  {
                    at: '2026-03-23T12:08:00Z',
                    actor: 'system',
                    action: 'promotion_waiting_on_operator',
                    job_id: 'job-operator-1',
                  },
                ],
                job_lineage: {
                  repair_job_ids: ['job-operator-1'],
                  apply_job_ids: [],
                  patch_pr_ids: [],
                  proposal_ids: ['proposal-1'],
                  retry_from_job_ids: [],
                },
                artifact_history: [
                  {
                    artifact_type: 'proposal',
                    artifact_id: 'proposal-1',
                    label: 'Selected proposal',
                    at: '2026-03-23T12:08:00Z',
                  },
                ],
                manual_promotion_history: [
                  {
                    action: 'keep_proposal_only',
                    operator_note: 'Needs review before apply',
                    at: '2026-03-23T12:09:00Z',
                    proposal_id: 'proposal-1',
                  },
                ],
              },
            ],
            completed_slices: [],
            failed_slices: [],
            promotion_decisions: [],
            portfolio_progress: {
              total_slices: 1,
              pending_slices: 0,
              completed_slices: 0,
              failed_slices: 0,
              auto_applied_slices: 0,
              proposal_only_slices: 1,
            },
          },
          child_job_ids: ['job-operator-1'],
          latest_summary: {
            status: 'awaiting_operator',
            waiting_on_operator_action: true,
            active_slice_id: 'slice_1',
            active_slice_title: 'Target DocumentsPage.tsx',
            allowed_slice_actions: ['apply_override', 'create_patch_pr', 'keep_proposal_only', 'relaunch_slice', 'skip_slice'],
            recommended_next_action: 'apply_override',
            portfolio_progress: {
              total_slices: 1,
              pending_slices: 0,
              completed_slices: 0,
              failed_slices: 0,
              auto_applied_slices: 0,
              proposal_only_slices: 1,
            },
          },
          orchestrator_job_id: 'job-orch-3',
          current_job_id: 'job-operator-1',
          latest_apply_job_id: null,
          latest_proposal_id: 'proposal-1',
          created_at: '2026-03-23T12:00:00Z',
          updated_at: '2026-03-23T12:10:00Z',
          started_at: '2026-03-23T12:00:00Z',
          completed_at: null,
        },
      ],
      total: 1,
      limit: 100,
      offset: 0,
    });

    await renderWithProviders('/autonomous-agents', { documentSources: defaultDocumentSources });

    fireEvent.click(screen.getByText('Coding Backlog'));
    expect(await screen.findByText('Manual promotion backlog')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Orchestration detail'));
    fireEvent.click(screen.getByText('Artifacts and lineage'));
    expect(await screen.findByText(/Operator decisions/i)).toBeInTheDocument();
    fireEvent.click(await screen.findByText('Create Patch PR'));

    await waitFor(() => {
      expect(apiClient.performCodingBacklogAction).toHaveBeenCalledWith('backlog-3', {
        action: 'create_patch_pr',
        slice_id: 'slice_1',
      });
    });
  });
});
