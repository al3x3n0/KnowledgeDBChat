import type {
  AgentJobQuickStartBugTriageSwarmRequest,
  AgentJobQuickStartBuildBreakSwarmRequest,
  AgentJobQuickStartDomainResearchRequest,
  AgentJobQuickStartFrontendRegressionSwarmRequest,
  AgentJobQuickStartRepoBugTriageRequest,
} from '../types';

export const DEFAULT_VALIDATION_POLICY = {
  confidence_threshold: 0.72,
  experiment_readiness_threshold: 0.8,
  max_auto_follow_up_launches: 2,
  auto_create_experiment_plans: true,
  auto_launch_follow_up: true,
  auto_execute_validation_runs: false,
  max_concurrent_validation_runs: 1,
  max_validation_runtime_minutes: 20,
  max_validation_budget_per_run: 25,
  follow_up_review_mode: 'queue_for_approval',
  validation_backoff_policy: {
    max_consecutive_failures: 2,
    cooldown_minutes: 180,
  },
};

export const splitUniqueLines = (value: string, limit = 12): string[] =>
  String(value || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((entry, index, rows) => rows.indexOf(entry) === index)
    .slice(0, limit);

export const parseQuickStartCommands = (raw: string, maxItems: number): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();
  const rows = String(raw || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  for (const row of rows) {
    if (seen.has(row)) continue;
    seen.add(row);
    out.push(row.slice(0, 500));
    if (out.length >= maxItems) break;
  }
  return out;
};

export const parseSafeRelativeFilePaths = (
  raw: string,
  maxItems: number
): { items: string[]; droppedUnsafe: number } => {
  const rows = String(raw || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  const out: string[] = [];
  const seen = new Set<string>();
  let droppedUnsafe = 0;
  for (const row of rows) {
    let path = row.replace(/\\/g, '/').trim();
    while (path.startsWith('./')) path = path.slice(2);
    if (!path || path.startsWith('/') || path.includes(':')) {
      droppedUnsafe += 1;
      continue;
    }
    const parts = path.split('/').filter((seg) => seg && seg !== '.');
    if (parts.some((seg) => seg === '..')) {
      droppedUnsafe += 1;
      continue;
    }
    const normalized = parts.join('/').slice(0, 500);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
    if (out.length >= maxItems) break;
  }
  return { items: out, droppedUnsafe };
};

export const findUnsafeQuickStartCommands = (commands: string[]): string[] => {
  const blockedPatterns = [
    /\brm\s+-rf\b/i,
    /\bsudo\b/i,
    /\bmkfs\b/i,
    /\bdd\s+if=/i,
    /\bshutdown\b/i,
    /\breboot\b/i,
    /\bhalt\b/i,
    /\bpoweroff\b/i,
    /\bchown\b/i,
    /\bchmod\s+777\b/i,
  ];
  return commands.filter((cmd) => blockedPatterns.some((rx) => rx.test(String(cmd || '')))).slice(0, 6);
};

export const buildDomainResearchQuickStartPayload = (args: {
  name: string;
  domain: string;
  objective: string;
  customerContextValue: string;
  trackType: 'compiler' | 'microarchitecture' | 'generic';
  sourceScope: 'kb_only' | 'arxiv_only' | 'kb_plus_arxiv' | 'kb_plus_arxiv_plus_repo';
  monitorQueriesText: string;
  benchmarkQueriesText: string;
  selectedRepoSourceIds: string[];
  sandboxProfileId?: string;
  reportFormat: 'brief_only' | 'report_only' | 'brief_and_report';
  persistArtifacts: boolean;
  autoLaunchFollowUp: boolean;
}): AgentJobQuickStartDomainResearchRequest => {
  const monitorQueries = splitUniqueLines(args.monitorQueriesText, 12);
  const benchmarkQueries = splitUniqueLines(args.benchmarkQueriesText, 16);
  return {
    name: args.name.trim() || undefined,
    domain: args.domain.trim(),
    objective: args.objective.trim(),
    customer_context: args.customerContextValue.trim() || undefined,
    track_type: args.trackType,
    source_scope: args.sourceScope,
    monitor_queries: monitorQueries.length ? monitorQueries : undefined,
    repo_source_ids: args.selectedRepoSourceIds.length ? args.selectedRepoSourceIds : undefined,
    benchmark_queries: benchmarkQueries.length ? benchmarkQueries : undefined,
    sandbox_profile_id: args.sandboxProfileId || undefined,
    report_format: args.reportFormat,
    persist_artifacts: args.persistArtifacts,
    automation_profile: 'balanced',
    automation_policy: {
      ...DEFAULT_VALIDATION_POLICY,
      auto_launch_follow_up: args.autoLaunchFollowUp,
    },
    auto_launch_follow_up: args.autoLaunchFollowUp,
    auto_create_experiment_plans: true,
    start_immediately: true,
  };
};

export const buildRepoBugTriageQuickStartPayload = (args: {
  name: string;
  goal: string;
  failureSymptom: string;
  selectedSourceId: string;
  scope: 'auto' | 'backend' | 'frontend' | 'worker';
  searchQuery: string;
  commandsText: string;
  filePathsText: string;
  errorOutput: string;
  maxCommands: number;
  maxFilePaths: number;
}): AgentJobQuickStartRepoBugTriageRequest => {
  const commands = parseQuickStartCommands(args.commandsText, args.maxCommands);
  const filePaths = parseSafeRelativeFilePaths(args.filePathsText, args.maxFilePaths).items;
  return {
    name: args.name.trim() || undefined,
    goal: args.goal.trim() || undefined,
    failure_symptom: args.failureSymptom.trim() || undefined,
    source_id: args.selectedSourceId,
    scope: args.scope,
    search_query: args.searchQuery.trim() || undefined,
    file_paths: filePaths.length ? filePaths : undefined,
    commands: commands.length ? commands : undefined,
    error_output: args.errorOutput.trim() || undefined,
    start_immediately: true,
  };
};

export const buildBugTriageSwarmQuickStartPayload = (args: {
  name: string;
  goal: string;
  failureSymptom: string;
  selectedSourceId: string;
  scope: 'auto' | 'backend' | 'frontend' | 'worker';
  searchQuery: string;
  commandsText: string;
  filePathsText: string;
  errorOutput: string;
  maxAgents: number;
  maxCommands: number;
  maxFilePaths: number;
  profileId?: string;
}): AgentJobQuickStartBugTriageSwarmRequest => {
  return buildCodingSwarmQuickStartPayload(args);
};

export const buildBuildBreakSwarmQuickStartPayload = (args: {
  name: string;
  goal: string;
  failureSymptom: string;
  selectedSourceId: string;
  scope: 'auto' | 'backend' | 'frontend' | 'worker';
  searchQuery: string;
  commandsText: string;
  filePathsText: string;
  errorOutput: string;
  maxAgents: number;
  maxCommands: number;
  maxFilePaths: number;
  profileId?: string;
}): AgentJobQuickStartBuildBreakSwarmRequest => {
  return buildCodingSwarmQuickStartPayload(args);
};

export const buildFrontendRegressionSwarmQuickStartPayload = (args: {
  name: string;
  goal: string;
  failureSymptom: string;
  selectedSourceId: string;
  scope: 'auto' | 'backend' | 'frontend' | 'worker';
  searchQuery: string;
  commandsText: string;
  filePathsText: string;
  errorOutput: string;
  maxAgents: number;
  maxCommands: number;
  maxFilePaths: number;
  profileId?: string;
}): AgentJobQuickStartFrontendRegressionSwarmRequest => {
  return buildCodingSwarmQuickStartPayload(args);
};

const buildCodingSwarmQuickStartPayload = (args: {
  name: string;
  goal: string;
  failureSymptom: string;
  selectedSourceId: string;
  scope: 'auto' | 'backend' | 'frontend' | 'worker';
  searchQuery: string;
  commandsText: string;
  filePathsText: string;
  errorOutput: string;
  maxAgents: number;
  maxCommands: number;
  maxFilePaths: number;
  profileId?: string;
}):
  | AgentJobQuickStartBugTriageSwarmRequest
  | AgentJobQuickStartBuildBreakSwarmRequest
  | AgentJobQuickStartFrontendRegressionSwarmRequest => {
  const commands = parseQuickStartCommands(args.commandsText, args.maxCommands);
  const filePaths = parseSafeRelativeFilePaths(args.filePathsText, args.maxFilePaths).items;
  return {
    name: args.name.trim() || undefined,
    goal: args.goal.trim() || undefined,
    failure_symptom: args.failureSymptom.trim() || undefined,
    source_id: args.selectedSourceId,
    scope: args.scope,
    search_query: args.searchQuery.trim() || undefined,
    file_paths: filePaths.length ? filePaths : undefined,
    commands: commands.length ? commands : undefined,
    error_output: args.errorOutput.trim() || undefined,
    max_agents: Math.max(1, Math.min(args.maxAgents || 4, 4)),
    profile_id: args.profileId || undefined,
    start_immediately: true,
  };
};
