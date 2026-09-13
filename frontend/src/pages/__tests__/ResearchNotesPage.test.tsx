import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import ResearchNotesPage from '../ResearchNotesPage';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    listResearchNotes: jest.fn(),
    getResearchNote: jest.fn(),
    deleteResearchNote: jest.fn(),
    lintRecentResearchNotes: jest.fn(),
    enforceResearchNoteCitations: jest.fn(),
    lintResearchNoteCitations: jest.fn(),
    listExperimentPlansForNote: jest.fn(),
    listBenchmarkSuites: jest.fn(),
    searchGitDocumentSources: jest.fn(),
    getActiveGitSources: jest.fn(),
    listExperimentRuns: jest.fn(),
    createSynthesisJob: jest.fn(),
    generateExperimentPlan: jest.fn(),
    createExperimentRun: jest.fn(),
    updateExperimentRun: jest.fn(),
    startExperimentRun: jest.fn(),
    syncExperimentRun: jest.fn(),
    performExperimentRunAction: jest.fn(),
    performAgentJobAction: jest.fn(),
    createJobFromChain: jest.fn(),
    appendExperimentRunToNote: jest.fn(),
    updateResearchNote: jest.fn(),
    createResearchNote: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;

const makeNote = () => ({
  id: 'note-1',
  title: 'Autonomous Agent Validation',
  content_markdown: '## Hypothesis\nAgent bootstrap should recover missing envs.',
  structured_payload: {
    research_mode: 'paper_to_hypothesis',
    summary: 'Structured research note.',
    source_paper_ids: ['paper-1'],
    source_document_ids: ['doc-1'],
    hypotheses: [
      {
        id: 'hyp-1',
        rank: 1,
        title: 'Bootstrap recovery',
        claim: 'If the bootstrap phase reconstructs env defaults, backend tests recover.',
        rationale: 'Most failures come from missing envs.',
        novelty_score: 0.7,
        evidence_score: 0.8,
        testability_score: 0.9,
        overall_score: 0.82,
        supporting_sources: [{ id: 'paper-1', title: 'Bootstrap paper' }],
        recommended_next_step: 'Run failing backend suite.',
      },
    ],
  },
  tags: ['agents'],
  created_at: '2026-03-10T00:00:00Z',
  updated_at: '2026-03-10T00:00:00Z',
});

const renderWithProviders = (initialEntry: string = '/research-notes?note=note-1') => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });

  return render(
    <MemoryRouter
      initialEntries={[initialEntry]}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <QueryClientProvider client={queryClient}>
        <ResearchNotesPage />
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe('ResearchNotesPage', () => {
  beforeEach(() => {
    const note = makeNote();
    apiClient.listResearchNotes.mockResolvedValue({
      items: [note],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(note);
    apiClient.listExperimentPlansForNote.mockResolvedValue({
      plans: [
        {
          id: 'plan-1',
          user_id: 'user-1',
          research_note_id: note.id,
          title: 'Bootstrap Validation Plan',
          plan: {},
          generator_details: {
            plan_mode: 'aggregate_note',
            selected_hypothesis_ids: ['hyp-1'],
            source_paper_ids: ['paper-1'],
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });
    apiClient.listBenchmarkSuites.mockResolvedValue({
      items: [
        {
          id: 'compiler-llvm-regression-core',
          name: 'LLVM Regression Core',
          benchmark_family: 'compiler_regression',
          description: 'Compiler regression suite',
          track_type: 'compiler',
          suite_version: 1,
          tags: ['compiler'],
          metadata: {},
          enabled: true,
          system_managed: true,
          cases: [
            {
              id: 'case-instcombine-sroa',
              suite_id: 'compiler-llvm-regression-core',
              name: 'InstCombine + SROA stress',
              rank: 1,
              expected_artifacts: ['compiler_logs'],
              metrics: [{ name: 'compile_time_ms', direction: 'lower_better' }],
              observability: { capture_ir: true, capture_remarks: true, repeat_count: 2 },
              metadata: {},
            },
          ],
          baselines: [
            {
              id: 'baseline-llvm-main',
              suite_id: 'compiler-llvm-regression-core',
              name: 'LLVM main baseline',
              measurements: { compile_time_ms: 1280 },
              environment_snapshot: {},
              enabled: true,
              system_managed: true,
            },
          ],
        },
      ],
      total: 1,
    });
    apiClient.searchGitDocumentSources.mockResolvedValue([]);
    apiClient.getActiveGitSources.mockResolvedValue([]);
    apiClient.listExperimentRuns.mockResolvedValue({
      runs: [
        {
          id: 'run-1',
          user_id: 'user-1',
          experiment_plan_id: 'plan-1',
          agent_job_id: 'job-1',
          name: 'Bootstrap Retry Run',
          status: 'completed',
          progress: 100,
          config: {
            launch_mode: 'quick_start_claude_backend',
            execution_handoff: {
              plan_scope: 'aggregate_note',
              selected_hypothesis_ids: ['hyp-1'],
              source_paper_ids: ['paper-1'],
            },
          },
          results: {
            source_id: 'repo-1',
            execution_strategy: {
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
                },
              ],
              execution_graph: {
                graph_health: {
                  status: 'critical',
                  severity_score: 21,
                  blocked_ratio: 0.4,
                  reasons: ['fallback verification still failing'],
                },
                recommended_actions: ['Inspect failing fallback output'],
              },
            },
          },
          experiment_run: {
            source_id: 'repo-1',
            source_name: 'Knowledge Repo',
            final_phase: 'retry_primary',
            phases: ['primary', 'bootstrap', 'fallback'],
            bootstrap_attempted: true,
            bootstrap_ok: true,
            fallback_attempted: true,
            fallback_ok: false,
            inferred_project_profile: {
              detected_stack: ['node', 'python'],
            },
            verification_commands: [
              'CI=true npm --prefix frontend test -- --watchAll=false',
            ],
            failed_commands: [
              'CI=true npm --prefix frontend test -- --watchAll=false',
            ],
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });
    apiClient.createSynthesisJob.mockResolvedValue({ id: 'syn-1' });
    apiClient.generateExperimentPlan.mockResolvedValue({});
    apiClient.createExperimentRun.mockResolvedValue({});
    apiClient.updateExperimentRun.mockResolvedValue({});
    apiClient.startExperimentRun.mockResolvedValue({});
    apiClient.syncExperimentRun.mockResolvedValue({});
    apiClient.performExperimentRunAction.mockResolvedValue({ run: {} });
    apiClient.performAgentJobAction.mockResolvedValue({});
    apiClient.createJobFromChain.mockResolvedValue({});
    apiClient.appendExperimentRunToNote.mockResolvedValue({
      ...note,
      content_markdown: `${note.content_markdown}\n\n## Experiment Results\n<!-- experiment_run:run-1 -->\n`,
      structured_payload: {
        ...note.structured_payload,
        hypotheses: [
          {
            ...note.structured_payload.hypotheses[0],
            experiment_evidence: [
              {
                run_id: 'run-1',
                plan_scope: 'aggregate_note',
                status: 'completed',
                summary: 'Bootstrap recovered the environment.',
                result_highlights: ['Final phase: retry_primary', 'Bootstrap: ok'],
              },
            ],
          },
        ],
      },
    });
    apiClient.updateResearchNote.mockResolvedValue(note);
    apiClient.createResearchNote.mockResolvedValue(note);
    apiClient.deleteResearchNote.mockResolvedValue({});
    apiClient.lintRecentResearchNotes.mockResolvedValue({ items: [] });
    apiClient.enforceResearchNoteCitations.mockResolvedValue(note);
    apiClient.lintResearchNoteCitations.mockResolvedValue(note);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('highlights deep-linked experiment plan and run targets', async () => {
    renderWithProviders('/research-notes?note=note-1&plan=plan-1&run=run-1');

    const latestPlan = await screen.findByRole('region', { name: 'Experiment plan Bootstrap Validation Plan' });
    const run = await screen.findByRole('article', { name: 'Experiment run Bootstrap Retry Run' });

    expect(latestPlan).toHaveClass('border-primary-400');
    expect(run).toHaveClass('border-primary-400');
  });

  it('renders typed experiment run bootstrap and fallback summary', async () => {
    renderWithProviders();

    expect(await screen.findAllByText('Autonomous Agent Validation')).toHaveLength(2);
    expect(await screen.findByText('Bootstrap Retry Run')).toBeInTheDocument();

    expect(await screen.findByText('Final retry_primary')).toBeInTheDocument();
    expect(screen.getByText('Bootstrap ok')).toBeInTheDocument();
      expect(screen.getByText('Fallback attempted')).toBeInTheDocument();
      expect(screen.getByText('Recovery open')).toBeInTheDocument();
      expect(screen.getAllByText(/fallback verification still failing/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Inspect failing fallback output/i).length).toBeGreaterThan(0);
      expect(screen.getByText('Phases primary -> bootstrap -> fallback')).toBeInTheDocument();
      expect(screen.getByText('Source Knowledge Repo')).toBeInTheDocument();
      expect(screen.getAllByText('repo-1').length).toBeGreaterThan(0);
      expect(screen.getByText('Stack node, python')).toBeInTheDocument();
      expect(screen.getByText('Last restart (failed -> pending)')).toBeInTheDocument();
      expect(screen.getByText('Outcome applied')).toBeInTheDocument();
      expect(screen.getByText('Recovery Audit')).toBeInTheDocument();
      expect(
        screen.getByText((_, element) => element?.textContent === 'Latest action: restart (failed -> pending)')
      ).toBeInTheDocument();
      expect(
        screen.getByText((_, element) => element?.textContent === 'Outcome: applied')
      ).toBeInTheDocument();
      expect(screen.getByText('Recent intervention timeline')).toBeInTheDocument();
      expect(screen.getByText(/pause \(running -> paused\): Paused for manual inspection \[superseded\]/i)).toBeInTheDocument();
      expect(screen.getByText(/restart \(failed -> pending\): Retry after fallback failure \[applied\]/i)).toBeInTheDocument();
      expect(
        screen.getAllByText((_, element) => element?.textContent === 'Outcome reason: Job resumed after intervention').length
      ).toBeGreaterThan(0);
      expect(
        screen.getByText((_, element) => element?.textContent === 'Recovery reason: fallback verification still failing')
      ).toBeInTheDocument();
      expect(
        screen.getByText((_, element) => element?.textContent === 'Next step: Inspect failing fallback output')
      ).toBeInTheDocument();
      expect(screen.getByText('CI=true npm --prefix frontend test -- --watchAll=false')).toBeInTheDocument();
      expect(screen.getByText(/Failed: CI=true npm --prefix frontend test -- --watchAll=false/)).toBeInTheDocument();
      expect(screen.getByText('Restart job')).toBeInTheDocument();
      expect(screen.getByText('Relaunch clean run')).toBeInTheDocument();
      expect(screen.getByText('Copy failed command')).toBeInTheDocument();
    expect(screen.getByText('Copy next step')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Restart job'));

    await waitFor(() => {
      expect(apiClient.performAgentJobAction).toHaveBeenCalledWith('job-1', 'restart', {});
    });

    fireEvent.click(screen.getByText('Relaunch clean run'));

    await waitFor(() => {
      expect(apiClient.performAgentJobAction).toHaveBeenCalledWith('job-1', 'relaunch', {});
    });
  });

  it('passes benchmark suite selection into experiment plan generation', async () => {
    renderWithProviders();

    expect(await screen.findByText('Compiler benchmark suite')).toBeInTheDocument();

    fireEvent.change(screen.getByDisplayValue('No benchmark suite'), {
      target: { value: 'compiler-llvm-regression-core' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Generate aggregate plan' }));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          benchmark_suite_id: 'compiler-llvm-regression-core',
        })
      );
    });
  });

  it('renders scientific validation metadata and hides manual status shortcuts', async () => {
    apiClient.listExperimentRuns.mockResolvedValueOnce({
      runs: [
        {
          id: 'run-2',
          user_id: 'user-1',
          experiment_plan_id: 'plan-1',
          agent_job_id: 'job-2',
          name: 'Scientific Validation Run',
          status: 'blocked',
          progress: 100,
          validation_kind: 'scientific_validation',
          sandbox_profile_id: 'scientific-compiler-sandbox',
          recipe_family: 'compiler_validation',
          recipe_id: 'compiler_validation_v1',
          blocked_reason_code: 'disallowed_image',
          capability_check: {
            ok: false,
            missing: ['repo_reconstruction'],
          },
          profile_snapshot: {
            id: 'scientific-compiler-sandbox',
            name: 'Compiler Validation Sandbox',
          },
          recipe_snapshot: {
            commands: ['python -m pytest -q tests'],
          },
          compiler_artifacts: {
            capture_ir: true,
            capture_asm: true,
            capture_remarks: true,
            capture_perf_stat: true,
            artifact_inventory: ['compiler_logs', 'compiler_remarks', 'ir_or_codegen_artifacts'],
            diff_summary: 'Vectorization remarks changed versus baseline',
            pass_signals: ['loop-vectorize', 'regalloc'],
          },
          perf_counters: {
            instructions: 1024,
            branch_misses: 12,
          },
          measurement_summary: {
            compile_time_ms: 1180,
            artifact_diff_score: 0.14,
            comparison: 'improvement',
            repeat_count: 3,
          },
          operator_actions: [
            {
              action: 'requeue',
              outcome_status: 'applied',
              note: 'Waiting on a fixed image',
            },
          ],
          retry_count: 2,
          parent_run_id: 'run-1',
          experiment_run: {
            source_id: 'repo-2',
            final_phase: 'blocked',
            phases: ['planned', 'blocked'],
            verification_commands: ['python -m pytest -q tests'],
            failed_commands: [],
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });

    renderWithProviders();

    expect(await screen.findByText('Scientific Validation Run')).toBeInTheDocument();
    expect(screen.getByText('Scientific validation')).toBeInTheDocument();
    expect(screen.getByText('Recipe compiler_validation')).toBeInTheDocument();
    expect(screen.getByText('compiler_validation_v1')).toBeInTheDocument();
    expect(screen.getByText('Sandbox scientific-compiler-sandbox')).toBeInTheDocument();
    expect(screen.getByText('Blocked disallowed image')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Scientific validation details'));

    expect(await screen.findByText(/Capability check: blocked/i)).toBeInTheDocument();
    expect(screen.getByText(/Profile snapshot: Compiler Validation Sandbox/i)).toBeInTheDocument();
      expect(screen.getByText(/Recipe snapshot commands: python -m pytest -q tests/i)).toBeInTheDocument();
      expect(screen.getByText(/Measurement summary: compile_time_ms=1180/i)).toBeInTheDocument();
      expect(screen.getByText(/Artifact inventory: compiler_logs, compiler_remarks, ir_or_codegen_artifacts/i)).toBeInTheDocument();
      expect(screen.getByText(/Compiler observability: IR captured · ASM captured · Pass remarks captured · Perf counters captured/i)).toBeInTheDocument();
      expect(screen.getByText(/Perf counters: instructions=1024 · branch_misses=12/i)).toBeInTheDocument();
      expect(screen.getByText(/Latest action: requeue · applied/i)).toBeInTheDocument();
    expect(screen.getByText(/Retry lineage: attempt 2 · parent run-1/i)).toBeInTheDocument();

    expect(screen.queryByText('Run')).not.toBeInTheDocument();
    expect(screen.queryByText('Done')).not.toBeInTheDocument();
    expect(screen.queryByText('Fail')).not.toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
    expect(screen.getByText('Requeue')).toBeInTheDocument();
  });

  it('appends a run to the note and renders latest hypothesis evidence', async () => {
    renderWithProviders();

    expect(await screen.findByText('Bootstrap Retry Run')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Append'));

    await waitFor(() => {
      expect(apiClient.appendExperimentRunToNote).toHaveBeenCalledWith('run-1');
    });

    expect(await screen.findByText('Latest experiment evidence')).toBeInTheDocument();
    expect(screen.getByText(/Run run-1 · completed · aggregate_note/i)).toBeInTheDocument();
      expect(screen.getByText('Bootstrap recovered the environment.')).toBeInTheDocument();
      expect(screen.getByText(/Final phase: retry_primary · Bootstrap: ok/i)).toBeInTheDocument();
    expect(screen.getByText('Appended')).toBeInTheDocument();
  });

  it('starts a compiler regression explanation from comparable benchmark-backed runs', async () => {
    apiClient.listExperimentRuns.mockResolvedValueOnce({
      runs: [
        {
          id: 'run-new',
          user_id: 'user-1',
          experiment_plan_id: 'plan-1',
          agent_job_id: 'job-new',
          name: 'Compiler Run New',
          status: 'completed',
          progress: 100,
          benchmark_family: 'compiler_regression',
          benchmark_suite_id: 'compiler-llvm-regression-core',
          benchmark_case_ids: ['case-instcombine-sroa'],
          config: {
            scientific_validation: {
              benchmark_family: 'compiler_regression',
              benchmark_suite_id: 'compiler-llvm-regression-core',
              benchmark_case_ids: ['case-instcombine-sroa'],
            },
          },
          created_at: '2026-03-11T00:00:00Z',
          updated_at: '2026-03-11T00:00:00Z',
        },
        {
          id: 'run-old',
          user_id: 'user-1',
          experiment_plan_id: 'plan-1',
          agent_job_id: 'job-old',
          name: 'Compiler Run Old',
          status: 'completed',
          progress: 100,
          benchmark_family: 'compiler_regression',
          benchmark_suite_id: 'compiler-llvm-regression-core',
          benchmark_case_ids: ['case-instcombine-sroa'],
          config: {
            scientific_validation: {
              benchmark_family: 'compiler_regression',
              benchmark_suite_id: 'compiler-llvm-regression-core',
              benchmark_case_ids: ['case-instcombine-sroa'],
            },
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });

    renderWithProviders();

    expect(await screen.findByText('Compiler Run New')).toBeInTheDocument();
    expect(screen.getByText('Comparison run run-old')).toBeInTheDocument();

    fireEvent.click(
      screen.getByTitle('Compare against run-old and generate a compiler regression explanation')
    );

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
        job_type: 'compiler_regression_explanation',
        title: 'Compiler Regression Explanation · Autonomous Agent Validation',
        document_ids: [],
        research_note_id: 'note-1',
        experiment_run_ids: ['run-new', 'run-old'],
        primary_run_id: 'run-new',
        comparison_run_id: 'run-old',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });

  it('starts a hypothesis reevaluation job from a note with experiment evidence', async () => {
    const noteWithEvidence = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_synthesis',
        last_appended_at: '2026-03-11T00:00:00Z',
        hypotheses: [
          {
            ...makeNote().structured_payload.hypotheses[0],
            experiment_evidence: [
              {
                run_id: 'run-1',
                status: 'completed',
                summary: 'Positive benchmark evidence.',
              },
            ],
          },
        ],
      },
      updated_at: '2026-03-11T00:00:00Z',
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [noteWithEvidence],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(noteWithEvidence);

    renderWithProviders();

    expect(await screen.findByText('New experiment evidence is available for re-evaluation.')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Re-evaluate hypotheses'));

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
        job_type: 'hypothesis_reevaluation',
        title: 'Hypothesis Re-evaluation · Autonomous Agent Validation',
        document_ids: [],
        research_note_id: 'note-1',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });

  it('renders reevaluation summary and previous snapshot deltas for reevaluated notes', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      updated_at: '2026-03-12T00:00:00Z',
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        reprioritization_summary: 'Bootstrap recovery remains highest priority after positive evidence.',
        scoring_policy: {
          source_job_id: 'syn-1',
          reevaluated_at: '2026-03-12T00:00:00Z',
        },
        previous_summary: 'Previous memo summary.',
        previous_artifact_type: 'hypothesis_synthesis',
        priority_deltas: [
          {
            hypothesis_id: 'hyp-1',
            previous_rank: 2,
            new_rank: 1,
            status: 'up',
            reason: 'Positive benchmark evidence improved confidence.',
          },
        ],
        hypotheses: [
          {
            ...makeNote().structured_payload.hypotheses[0],
            autonomous_origin: {
              source_kind: 'profile',
              source_id: 'profile-1',
              opportunity_id: 'opp-1',
            },
          },
        ],
        previous_hypotheses: [
          {
            ...makeNote().structured_payload.hypotheses[0],
            rank: 2,
            overall_score: 0.61,
            evidence_score: 0.42,
            testability_score: 0.72,
          },
        ],
        reevaluation_history: [
          {
            job_id: 'syn-1',
            saved_at: '2026-03-12T00:00:00Z',
            source_note_id: 'note-1',
            target_note_id: 'note-saved-1',
            origin_source_kind: 'profile',
            origin_source_id: 'profile-1',
            origin_opportunity_id: 'opp-1',
            reprioritization_summary: 'Bootstrap recovery remains highest priority after positive evidence.',
            source_run_ids: ['run-1'],
            outcome_status: 'saved_as_new_note',
            outcome_recorded_at: '2026-03-12T00:05:00Z',
          },
        ],
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);

    renderWithProviders();

    expect(await screen.findByText('Re-evaluation summary')).toBeInTheDocument();
    expect(screen.getAllByText('Bootstrap recovery remains highest priority after positive evidence.').length).toBeGreaterThan(0);
    expect(screen.getByText('Hypothesis ranking is up to date with the latest reevaluation snapshot.')).toBeInTheDocument();
    expect(screen.getByText('Compared with previous snapshot')).toBeInTheDocument();
    expect(screen.getByText(/Rank 2 -> 1 · up/i)).toBeInTheDocument();
    expect(screen.getByText(/Overall 0.61 -> 0.82 · Evidence 0.42 -> 0.80 · Testability 0.72 -> 0.90/i)).toBeInTheDocument();
    expect(screen.getByText('Positive benchmark evidence improved confidence.')).toBeInTheDocument();
    expect(screen.getByText('Recommended')).toBeInTheDocument();
    expect(screen.getByText('Reevaluation history')).toBeInTheDocument();
    expect(screen.getByText('Latest reevaluation')).toBeInTheDocument();
    expect(screen.getByText('Saved as new note')).toBeInTheDocument();
    expect(screen.getByText(/Source runs run-1/i)).toBeInTheDocument();
    expect(screen.getByText('Open reevaluation job')).toBeInTheDocument();
    expect(screen.getByText('Open source note')).toBeInTheDocument();
    expect(screen.getByText('Open saved note')).toBeInTheDocument();
    expect(screen.getAllByText('Open originating opportunity')).toHaveLength(2);
  });

  it('uses the scientific validation run action endpoint for pause and retry controls', async () => {
    apiClient.listExperimentRuns.mockResolvedValueOnce({
      runs: [
        {
          id: 'run-3',
          user_id: 'user-1',
          experiment_plan_id: 'plan-1',
          agent_job_id: 'job-3',
          name: 'Running Scientific Validation',
          status: 'running',
          progress: 45,
          validation_kind: 'scientific_validation',
          sandbox_profile_id: 'scientific-compiler-sandbox',
          recipe_family: 'compiler_validation',
          recipe_id: 'compiler_validation_v1',
          operator_actions: [],
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });
    apiClient.performExperimentRunAction.mockResolvedValue({
      run: {
        id: 'run-3',
        user_id: 'user-1',
        experiment_plan_id: 'plan-1',
        agent_job_id: 'job-3',
        name: 'Running Scientific Validation',
        status: 'paused',
        progress: 45,
        validation_kind: 'scientific_validation',
        created_at: '2026-03-10T00:00:00Z',
        updated_at: '2026-03-10T00:00:00Z',
      },
    });

    renderWithProviders();

    expect(await screen.findByText('Running Scientific Validation')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText(/Operator note for pause\/cancel\/retry\/requeue/i), {
      target: { value: 'Pause for inspection' },
    });
    fireEvent.click(screen.getByText('Pause'));

    await waitFor(() => {
      expect(apiClient.performExperimentRunAction).toHaveBeenCalledWith('run-3', {
        action: 'pause',
        note: 'Pause for inspection',
        start_immediately: undefined,
      });
    });
  });

  it('generates aggregate and single-hypothesis experiment plans from structured notes', async () => {
    renderWithProviders();

    expect(await screen.findByText('Generate aggregate plan')).toBeInTheDocument();
    expect(await screen.findByText('Generate plan for this hypothesis')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Generate aggregate plan'));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          prefer_section: 'hypothesis',
          max_note_chars: 12000,
          plan_mode: 'aggregate_note',
          include_ablations: true,
          include_timeline: true,
          include_risks: true,
          include_repro_checklist: true,
        })
      );
    });

    fireEvent.click(screen.getByText('Generate plan for this hypothesis'));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          prefer_section: 'hypothesis',
          max_note_chars: 12000,
          plan_mode: 'single_hypothesis',
          hypothesis_id: 'hyp-1',
          include_ablations: true,
          include_timeline: true,
          include_risks: true,
          include_repro_checklist: true,
        })
      );
    });

    expect(screen.getAllByText(/Aggregate note/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Hypotheses: hyp-1/i)).toBeInTheDocument();
    expect(screen.getByText(/Source papers: paper-1/i)).toBeInTheDocument();
  });

  it('seeds new experiment runs with execution handoff metadata from the latest plan', async () => {
    apiClient.createExperimentRun.mockResolvedValue({});

    renderWithProviders();

    expect(await screen.findByDisplayValue('Bootstrap Validation Plan · aggregate run')).toBeInTheDocument();
    fireEvent.click(screen.getByText('New run'));

    await waitFor(() => {
      expect(apiClient.createExperimentRun).toHaveBeenCalledWith(
        'plan-1',
        expect.objectContaining({
          name: 'Bootstrap Validation Plan · aggregate run',
          config: expect.objectContaining({
            execution_handoff: expect.objectContaining({
              execution_handoff_version: 1,
              plan_scope: 'aggregate_note',
              selected_hypothesis_ids: ['hyp-1'],
              source_paper_ids: ['paper-1'],
            }),
            commands: ['python -m pytest -q'],
            timeout_seconds: 60,
          }),
        })
      );
    });

    expect(screen.getAllByText(/Scope aggregate note/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Hypotheses hyp-1/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Papers paper-1/i).length).toBeGreaterThan(0);
  });

  it('uses reevaluated notes to default the primary plan action to the top-ranked hypothesis', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        scoring_policy: {
          source_job_id: 'syn-1',
          reevaluated_at: '2026-03-12T00:00:00Z',
        },
        hypotheses: [
          {
            ...makeNote().structured_payload.hypotheses[0],
            id: 'hyp-1',
            rank: 1,
          },
          {
            id: 'hyp-2',
            rank: 2,
            title: 'Second idea',
            claim: 'Another option.',
            rationale: 'Lower ranked.',
            novelty_score: 0.5,
            evidence_score: 0.4,
            testability_score: 0.6,
            overall_score: 0.51,
            recommended_next_step: 'Hold.',
          },
        ],
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);
    apiClient.listExperimentPlansForNote.mockResolvedValue({
      plans: [
        {
          id: 'plan-1',
          user_id: 'user-1',
          research_note_id: 'note-1',
          title: 'Bootstrap Validation Plan',
          plan: {},
          generator_details: {
            plan_mode: 'single_hypothesis',
            selected_hypothesis_ids: ['hyp-1'],
            source_paper_ids: ['paper-1'],
            reevaluation_mode: true,
            reevaluation_source_job_id: 'syn-1',
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });

    renderWithProviders();

    expect(await screen.findByText('Generate next plan')).toBeInTheDocument();
    expect(screen.getByText('Generate aggregate plan')).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.textContent === 'Recommended hypothesis: Bootstrap recovery · hyp-1')
    ).toBeInTheDocument();

    fireEvent.click(screen.getByText('Generate next plan'));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          prefer_section: 'hypothesis',
          max_note_chars: 12000,
          include_ablations: true,
          include_timeline: true,
          include_risks: true,
          include_repro_checklist: true,
        })
      );
    });

    expect(screen.getByText(/Reevaluated ranking/i)).toBeInTheDocument();
    expect(screen.getByText(/Reevaluation job: syn-1/i)).toBeInTheDocument();
  });

  it('creates a recommended run by generating the next reevaluated plan first', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        scoring_policy: {
          source_job_id: 'syn-1',
          reevaluated_at: '2026-03-12T00:00:00Z',
        },
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);
    apiClient.generateExperimentPlan.mockResolvedValue({
      id: 'plan-next',
      title: 'Bootstrap Validation Plan',
      plan: {
        objective: 'Validate the top-ranked bootstrap recovery hypothesis.',
        provenance: {
          source_paper_ids: ['paper-1'],
          source_document_ids: ['doc-1'],
        },
      },
      generator_details: {
        plan_mode: 'single_hypothesis',
        selected_hypothesis_ids: ['hyp-1'],
        source_paper_ids: ['paper-1'],
        source_document_ids: ['doc-1'],
        supporting_sources: [{ id: 'paper-1', title: 'Bootstrap paper' }],
        reevaluation_mode: true,
        reevaluation_source_job_id: 'syn-1',
      },
    });
    apiClient.createExperimentRun.mockResolvedValue({ id: 'run-next' });

    renderWithProviders();

    expect(await screen.findByText('Run recommended hypothesis')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Run recommended hypothesis'));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          prefer_section: 'hypothesis',
          max_note_chars: 12000,
          include_ablations: true,
          include_timeline: true,
          include_risks: true,
          include_repro_checklist: true,
        })
      );
    });

    await waitFor(() => {
      expect(apiClient.createExperimentRun).toHaveBeenCalledWith(
        'plan-next',
        expect.objectContaining({
          name: 'Bootstrap Validation Plan · hyp-1',
          config: expect.objectContaining({
            execution_handoff: expect.objectContaining({
              execution_handoff_version: 1,
              plan_scope: 'single_hypothesis',
              selected_hypothesis_ids: ['hyp-1'],
              source_paper_ids: ['paper-1'],
              source_document_ids: ['doc-1'],
            }),
            post_run_actions: {
              auto_append_to_note: true,
              target_note_id: 'note-1',
              append_status: 'pending',
            },
          }),
          summary: 'Validate the top-ranked bootstrap recovery hypothesis.',
        })
      );
    });
  });

  it('starts the recommended loop by generating the next reevaluated plan first', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        scoring_policy: {
          source_job_id: 'syn-1',
          reevaluated_at: '2026-03-12T00:00:00Z',
        },
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);
    apiClient.generateExperimentPlan.mockResolvedValue({
      id: 'plan-loop',
      title: 'Bootstrap Validation Plan',
      plan: {},
      generator_details: {
        plan_mode: 'single_hypothesis',
        selected_hypothesis_ids: ['hyp-1'],
        reevaluation_mode: true,
        reevaluation_source_job_id: 'syn-1',
      },
    });
    apiClient.createJobFromChain.mockResolvedValue({ id: 'job-loop' });

    renderWithProviders();

    expect(await screen.findByText('Start loop from recommended')).toBeInTheDocument();
    fireEvent.click(await screen.findByText('Runner settings (repo + commands)'));
    fireEvent.change(screen.getByPlaceholderText(/123e4567-e89b-12d3-a456-426614174000/i), {
      target: { value: 'repo-1' },
    });

    fireEvent.click(screen.getByText('Start loop from recommended'));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          prefer_section: 'hypothesis',
          max_note_chars: 12000,
          include_ablations: true,
          include_timeline: true,
          include_risks: true,
          include_repro_checklist: true,
        })
      );
    });

    await waitFor(() => {
      expect(apiClient.createJobFromChain).toHaveBeenCalledWith(
        expect.objectContaining({
          chain_definition_id: '9e267663-48d6-4a69-9679-984d1cdf6205',
          variables: {
            research_note_id: 'note-1',
            experiment_plan_id: 'plan-loop',
          },
          config_overrides: expect.objectContaining({
            research_note_id: 'note-1',
            experiment_plan_id: 'plan-loop',
            selected_hypothesis_ids: ['hyp-1'],
            reevaluation_mode: true,
            reevaluation_source_job_id: 'syn-1',
            post_run_actions: {
              auto_append_to_note: true,
              target_note_id: 'note-1',
              append_status: 'pending',
            },
          }),
          start_immediately: true,
        })
      );
    });
  });

  it('renders automatic append status for seeded runs', async () => {
    apiClient.listExperimentRuns.mockResolvedValueOnce({
      runs: [
        {
          id: 'run-auto',
          user_id: 'user-1',
          experiment_plan_id: 'plan-1',
          name: 'Auto Append Run',
          status: 'completed',
          progress: 100,
          config: {
            execution_handoff: {
              plan_scope: 'single_hypothesis',
              selected_hypothesis_ids: ['hyp-1'],
            },
            post_run_actions: {
              auto_append_to_note: true,
              append_status: 'completed',
              target_note_id: 'note-1',
            },
          },
          created_at: '2026-03-10T00:00:00Z',
          updated_at: '2026-03-10T00:00:00Z',
        },
      ],
    });

    renderWithProviders();

    expect(await screen.findByText('Auto Append Run')).toBeInTheDocument();
    expect(screen.getByText('Auto-appended to note')).toBeInTheDocument();
  });

  it('renders a queued reevaluation draft link for reevaluated notes with fresh appended evidence', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        pending_reevaluation_job_id: 'syn-pending-1',
        pending_reevaluation_created_at: '2026-03-12T02:00:00Z',
        pending_reevaluation_reason: 'new_experiment_evidence',
        pending_reevaluation_source_run_ids: ['run-1'],
        hypotheses: [
          {
            ...makeNote().structured_payload.hypotheses[0],
            autonomous_origin: {
              source_kind: 'profile',
              source_id: 'profile-1',
              opportunity_id: 'opp-1',
            },
          },
        ],
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);

    renderWithProviders();

    expect(await screen.findByText('Reevaluation draft queued')).toBeInTheDocument();
    expect(screen.getByText(/New experiment evidence has already been queued for reevaluation/i)).toBeInTheDocument();
    expect(screen.getByText(/Source runs run-1/i)).toBeInTheDocument();
    expect(screen.getByText('Open queued reevaluation')).toBeInTheDocument();
    expect(screen.getAllByText('Open originating opportunity')).toHaveLength(2);
  });

  it('renders a review-ready CTA when the pending reevaluation draft has completed', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        pending_reevaluation_job_id: 'syn-ready-1',
        pending_reevaluation_status: 'completed',
        pending_reevaluation_completed_at: '2026-03-12T03:00:00Z',
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);

    renderWithProviders();

    expect(await screen.findByText('Reevaluation draft ready for review')).toBeInTheDocument();
    expect(screen.getByText(/finished and is ready to review/i)).toBeInTheDocument();
    expect(screen.getByText('Review reevaluation draft')).toBeInTheDocument();
  });

  it('renders a failed reevaluation draft state with recovery CTA', async () => {
    const reevaluatedNote = {
      ...makeNote(),
      structured_payload: {
        ...makeNote().structured_payload,
        artifact_type: 'hypothesis_reevaluation',
        pending_reevaluation_job_id: 'syn-failed-1',
        pending_reevaluation_status: 'failed',
        pending_reevaluation_error: 'Model timeout during reevaluation',
      },
    };
    apiClient.listResearchNotes.mockResolvedValue({
      items: [reevaluatedNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(reevaluatedNote);

    renderWithProviders();

    expect(await screen.findByText('Reevaluation draft failed')).toBeInTheDocument();
    expect(screen.getByText(/Model timeout during reevaluation/i)).toBeInTheDocument();
    expect(screen.getAllByText('Re-evaluate hypotheses').length).toBeGreaterThan(0);
  });

  it('uses compiler regression explanation notes to generate a follow-up plan by default', async () => {
    const explanationNote = {
      ...makeNote(),
      title: 'Compiler Regression Explanation',
      structured_payload: {
        artifact_type: 'compiler_regression_explanation',
        summary: 'Compile time regressed because vectorization remarks disappeared.',
        regression_type: 'compile_time',
        source_run_ids: ['run-new', 'run-old'],
        primary_run_id: 'run-new',
        comparison_run_id: 'run-old',
        benchmark_family: 'compiler_regression',
        benchmark_suite_id: 'compiler-llvm-regression-core',
        benchmark_case_ids: ['case-instcombine-sroa'],
        benchmark_baseline_id: 'baseline-llvm-main',
        likely_causes: [
          {
            title: 'Vectorizer not firing',
            confidence: 'medium',
            reason: 'loop-vectorize remarks disappeared',
          },
        ],
        recommended_next_steps: ['Diff pass remarks across both builds'],
      },
      updated_at: '2026-03-12T00:00:00Z',
    };

    apiClient.listResearchNotes.mockResolvedValue({
      items: [explanationNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(explanationNote);
    apiClient.listExperimentPlansForNote.mockResolvedValue({
      plans: [
        {
          id: 'plan-followup',
          user_id: 'user-1',
          research_note_id: 'note-1',
          title: 'Experiment Plan: Compiler Regression Explanation · Regression Follow-up',
          plan: {},
          generator_details: {
            plan_mode: 'compiler_regression_followup',
            explanation_mode: true,
            source_run_ids: ['run-new', 'run-old'],
            primary_run_id: 'run-new',
            comparison_run_id: 'run-old',
            regression_type: 'compile_time',
            benchmark_family: 'compiler_regression',
            benchmark_suite_id: 'compiler-llvm-regression-core',
            benchmark_baseline_id: 'baseline-llvm-main',
          },
          benchmark_family: 'compiler_regression',
          benchmark_suite_id: 'compiler-llvm-regression-core',
          benchmark_case_ids: ['case-instcombine-sroa'],
          benchmark_baseline_id: 'baseline-llvm-main',
          created_at: '2026-03-12T00:00:00Z',
          updated_at: '2026-03-12T00:00:00Z',
        },
      ],
    });

    renderWithProviders();

    expect(await screen.findByText('Generate follow-up plan')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Generate follow-up plan'));

    await waitFor(() => {
      expect(apiClient.generateExperimentPlan).toHaveBeenCalledWith(
        expect.objectContaining({
          note_id: 'note-1',
          prefer_section: 'hypothesis',
          max_note_chars: 12000,
          include_ablations: true,
          include_timeline: true,
          include_risks: true,
          include_repro_checklist: true,
        })
      );
    });

    const call = apiClient.generateExperimentPlan.mock.calls[0][0];
    expect(call.plan_mode).toBeUndefined();
    expect(screen.getByText(/Regression type: compile_time/i)).toBeInTheDocument();
    expect(screen.getByText(/Baseline: baseline-llvm-main/i)).toBeInTheDocument();
    expect(screen.getByText(/Benchmark suite: compiler-llvm-regression-core · compiler_regression/i)).toBeInTheDocument();
    expect(screen.getByText(/Compared runs: run-new, run-old/i)).toBeInTheDocument();
  });

  it('uses compiler regression explanation notes to generate a patch proposal', async () => {
    const explanationNote = {
      ...makeNote(),
      title: 'Compiler Regression Explanation',
      structured_payload: {
        artifact_type: 'compiler_regression_explanation',
        summary: 'Compile time regressed because vectorization remarks disappeared.',
        regression_type: 'compile_time',
        source_run_ids: ['run-new', 'run-old'],
        primary_run_id: 'run-new',
        comparison_run_id: 'run-old',
      },
      updated_at: '2026-03-12T00:00:00Z',
    };

    apiClient.listResearchNotes.mockResolvedValue({
      items: [explanationNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(explanationNote);

    renderWithProviders();

    expect(await screen.findByText('Generate patch proposal')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Generate patch proposal' }));

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
        job_type: 'compiler_patch_proposal',
        title: 'Compiler Patch Proposal · Compiler Regression Explanation',
        document_ids: [],
        research_note_id: 'note-1',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });

  it('uses compiler patch proposal notes to generate a repo-aware patch draft', async () => {
    window.localStorage.setItem(
      'research_note_experiment_settings:note-1',
      JSON.stringify({
        source_id: 'repo-1',
        commands_text: 'python -m pytest -q',
        max_runs: 3,
      })
    );

    const proposalNote = {
      ...makeNote(),
      title: 'Compiler Patch Proposal',
      structured_payload: {
        artifact_type: 'compiler_patch_proposal',
        proposal_summary: 'Gate the vectorization heuristic with a narrow profitability check.',
        target_area: 'vectorization heuristic',
        candidate_change: 'Add a guard before enabling the transform.',
      },
      updated_at: '2026-03-12T00:00:00Z',
    };

    apiClient.listResearchNotes.mockResolvedValue({
      items: [proposalNote],
      total: 1,
      page: 1,
      page_size: 20,
    });
    apiClient.getResearchNote.mockResolvedValue(proposalNote);

    renderWithProviders();

    expect(await screen.findByText('Generate patch draft')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Generate patch draft' }));

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith({
        job_type: 'compiler_patch_draft',
        title: 'Compiler Patch Draft · Compiler Patch Proposal',
        document_ids: [],
        research_note_id: 'note-1',
        source_id: 'repo-1',
        output_format: 'markdown',
        output_style: 'technical',
      });
    });
  });
});
