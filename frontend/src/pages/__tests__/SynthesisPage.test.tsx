import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import SynthesisPage from '../SynthesisPage';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    listSynthesisJobs: jest.fn(),
    getSynthesisTypesInfo: jest.fn(),
    getResearchNote: jest.fn(),
    saveSynthesisJobAsResearchNote: jest.fn(),
    reviewSynthesisJob: jest.fn(),
    deleteSynthesisJob: jest.fn(),
    cancelSynthesisJob: jest.fn(),
    downloadSynthesisResult: jest.fn(),
  },
}));

jest.mock('../../contexts/NotificationContext', () => ({
  useNotifications: () => ({
    fetchNotifications: jest.fn().mockResolvedValue(undefined),
    refreshUnreadCount: jest.fn().mockResolvedValue(undefined),
  }),
}));

const apiClient = require('../../services/api').apiClient;

const renderWithProviders = (initialEntry: string = '/synthesis?job=job-1') => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
    },
  });

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <QueryClientProvider client={queryClient}>
        <SynthesisPage />
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe('SynthesisPage', () => {
  const originalLocation = window.location;

  beforeEach(() => {
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { href: '' },
    });

    apiClient.getSynthesisTypesInfo.mockResolvedValue({
      job_types: [],
      output_formats: [],
      output_styles: [],
    });
    apiClient.deleteSynthesisJob.mockResolvedValue({ success: true });
    apiClient.cancelSynthesisJob.mockResolvedValue({ success: true });
    apiClient.downloadSynthesisResult.mockResolvedValue(undefined);
    apiClient.saveSynthesisJobAsResearchNote.mockResolvedValue({ id: 'note-1' });
    apiClient.reviewSynthesisJob.mockResolvedValue({
      id: 'job-1',
      review_outcome_status: 'dismissed',
      can_apply: false,
      can_dismiss: false,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: originalLocation,
    });
  });

  it('renders reevaluation draft diff and applies it back to the source note', async () => {
    apiClient.listSynthesisJobs.mockResolvedValue({
      jobs: [
        {
          id: 'job-1',
          user_id: 'user-1',
          job_type: 'hypothesis_reevaluation',
          title: 'Compiler hypothesis re-evaluation',
          document_ids: [],
          paper_ids: [],
          research_note_id: 'note-source-1',
          output_format: 'markdown',
          output_style: 'technical',
          status: 'completed',
          progress: 100,
          can_apply: true,
          can_dismiss: true,
          result_content: '# Re-evaluation',
          result_metadata: {
            reprioritization_summary: 'Hypothesis one stays on top.',
            priority_deltas: [
              {
                hypothesis_id: 'hyp-1',
                previous_rank: 2,
                new_rank: 1,
                reason: 'Positive evidence improved confidence.',
              },
            ],
            structured_hypotheses: [
              {
                id: 'hyp-1',
                rank: 1,
                title: 'Layout-aware scheduling',
                overall_score: 0.84,
                evidence_score: 0.86,
                testability_score: 0.9,
                recommended_next_step: 'Expand to multi-architecture validation.',
              },
            ],
          },
          created_at: '2026-03-12T00:00:00Z',
          completed_at: '2026-03-12T01:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
    });
    apiClient.getResearchNote.mockResolvedValue({
      id: 'note-source-1',
      user_id: 'user-1',
      title: 'Compiler hypotheses',
      content_markdown: '## Ranked hypotheses',
      structured_payload: {
        artifact_type: 'hypothesis_reevaluation',
        hypotheses: [
          {
            id: 'hyp-1',
            rank: 2,
            title: 'Layout-aware scheduling',
            overall_score: 0.78,
            evidence_score: 0.61,
            testability_score: 0.9,
            recommended_next_step: 'Run kernel benchmarks.',
          },
        ],
      },
    });

    renderWithProviders();

    expect(await screen.findByText('Changes vs source note')).toBeInTheDocument();
    expect(screen.getByText('Apply to source note')).toBeInTheDocument();
    expect(screen.getByText('Save as new note')).toBeInTheDocument();
    expect(screen.getByText(/Rank 2 → 1 · Overall 0.78 → 0.84 · Evidence 0.61 → 0.86 · Testability 0.9 → 0.9/i)).toBeInTheDocument();
    expect(screen.getByText(/Positive evidence improved confidence/i)).toBeInTheDocument();
    expect(screen.getByText(/Next step: Expand to multi-architecture validation/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText('Apply to source note'));

    await waitFor(() => {
      expect(apiClient.saveSynthesisJobAsResearchNote).toHaveBeenCalledWith('job-1', {
        title: 'Compiler hypothesis re-evaluation',
        tags: ['hypothesis-reevaluation', 'hypotheses'],
        target_note_id: 'note-source-1',
      });
    });
  });

  it('dismisses a completed reevaluation draft and hides apply controls after refresh', async () => {
    apiClient.listSynthesisJobs
      .mockResolvedValueOnce({
        jobs: [
          {
            id: 'job-1',
            user_id: 'user-1',
            job_type: 'hypothesis_reevaluation',
            title: 'Compiler hypothesis re-evaluation',
            document_ids: [],
            paper_ids: [],
            research_note_id: 'note-source-1',
            output_format: 'markdown',
            output_style: 'technical',
            status: 'completed',
            progress: 100,
            can_apply: true,
            can_dismiss: true,
            result_content: '# Re-evaluation',
            result_metadata: {
              structured_hypotheses: [{ id: 'hyp-1', rank: 1, title: 'Layout-aware scheduling' }],
            },
            created_at: '2026-03-12T00:00:00Z',
            completed_at: '2026-03-12T01:00:00Z',
          },
        ],
        total: 1,
        page: 1,
        page_size: 50,
      })
      .mockResolvedValueOnce({
        jobs: [
          {
            id: 'job-1',
            user_id: 'user-1',
            job_type: 'hypothesis_reevaluation',
            title: 'Compiler hypothesis re-evaluation',
            document_ids: [],
            paper_ids: [],
            research_note_id: 'note-source-1',
            output_format: 'markdown',
            output_style: 'technical',
            status: 'completed',
            progress: 100,
            can_apply: false,
            can_dismiss: false,
            review_outcome_status: 'dismissed',
            review_recorded_at: '2026-03-12T02:00:00Z',
            result_content: '# Re-evaluation',
            result_metadata: {
              structured_hypotheses: [{ id: 'hyp-1', rank: 1, title: 'Layout-aware scheduling' }],
            },
            created_at: '2026-03-12T00:00:00Z',
            completed_at: '2026-03-12T01:00:00Z',
          },
        ],
        total: 1,
        page: 1,
        page_size: 50,
      });
    apiClient.getResearchNote.mockResolvedValue({
      id: 'note-source-1',
      user_id: 'user-1',
      title: 'Compiler hypotheses',
      content_markdown: '## Ranked hypotheses',
      structured_payload: {
        artifact_type: 'hypothesis_reevaluation',
        hypotheses: [{
          id: 'hyp-1',
          rank: 1,
          title: 'Layout-aware scheduling',
          autonomous_origin: {
            source_kind: 'profile',
            source_id: 'profile-1',
            opportunity_id: 'opp-1',
          },
        }],
      },
    });

    renderWithProviders();

    expect(await screen.findByText('Dismiss draft')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Dismiss draft'));

    await waitFor(() => {
      expect(apiClient.reviewSynthesisJob).toHaveBeenCalledWith('job-1', {
        outcome_status: 'dismissed',
        outcome_note: undefined,
      });
    });

    expect(await screen.findByText('Outcome: Dismissed')).toBeInTheDocument();
    expect(screen.queryByText('Apply to source note')).not.toBeInTheDocument();
    expect(screen.queryByText('Save as new note')).not.toBeInTheDocument();
    expect(await screen.findByText('Open source note')).toBeInTheDocument();
    expect(await screen.findByText('Open originating opportunity')).toBeInTheDocument();
    expect(screen.queryByText('Open saved note')).not.toBeInTheDocument();
  });

  it('shows saved-note handoff actions after saving a reevaluation draft as a new note', async () => {
    apiClient.listSynthesisJobs
      .mockResolvedValueOnce({
        jobs: [
          {
            id: 'job-1',
            user_id: 'user-1',
            job_type: 'hypothesis_reevaluation',
            title: 'Compiler hypothesis re-evaluation',
            document_ids: [],
            paper_ids: [],
            research_note_id: 'note-source-1',
            output_format: 'markdown',
            output_style: 'technical',
            status: 'completed',
            progress: 100,
            can_apply: true,
            can_dismiss: true,
            result_content: '# Re-evaluation',
            result_metadata: {
              structured_hypotheses: [{ id: 'hyp-1', rank: 1, title: 'Layout-aware scheduling' }],
            },
            created_at: '2026-03-12T00:00:00Z',
            completed_at: '2026-03-12T01:00:00Z',
          },
        ],
        total: 1,
        page: 1,
        page_size: 50,
      })
      .mockResolvedValueOnce({
        jobs: [
          {
            id: 'job-1',
            user_id: 'user-1',
            job_type: 'hypothesis_reevaluation',
            title: 'Compiler hypothesis re-evaluation',
            document_ids: [],
            paper_ids: [],
            research_note_id: 'note-source-1',
            output_format: 'markdown',
            output_style: 'technical',
            status: 'completed',
            progress: 100,
            can_apply: false,
            can_dismiss: false,
            review_outcome_status: 'saved_as_new_note',
            review_recorded_at: '2026-03-12T02:00:00Z',
            review_target_note_id: 'note-saved-1',
            result_content: '# Re-evaluation',
            result_metadata: {
              structured_hypotheses: [{ id: 'hyp-1', rank: 1, title: 'Layout-aware scheduling' }],
            },
            created_at: '2026-03-12T00:00:00Z',
            completed_at: '2026-03-12T01:00:00Z',
          },
        ],
        total: 1,
        page: 1,
        page_size: 50,
      });
    apiClient.getResearchNote.mockResolvedValue({
      id: 'note-source-1',
      user_id: 'user-1',
      title: 'Compiler hypotheses',
      content_markdown: '## Ranked hypotheses',
      structured_payload: {
        artifact_type: 'hypothesis_reevaluation',
        hypotheses: [{
          id: 'hyp-1',
          rank: 1,
          title: 'Layout-aware scheduling',
          autonomous_origin: {
            source_kind: 'profile',
            source_id: 'profile-1',
            opportunity_id: 'opp-1',
          },
        }],
      },
    });

    renderWithProviders();

    expect(await screen.findByText('Save as new note')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Save as new note'));

    await waitFor(() => {
      expect(apiClient.saveSynthesisJobAsResearchNote).toHaveBeenCalledWith('job-1', {
        title: 'Compiler hypothesis re-evaluation',
        tags: ['hypothesis-reevaluation', 'hypotheses'],
        target_note_id: undefined,
      });
    });

    expect(await screen.findByText('Outcome: Saved as new note')).toBeInTheDocument();
    expect(await screen.findByText('Open source note')).toBeInTheDocument();
    expect(await screen.findByText('Open saved note')).toBeInTheDocument();
    expect(await screen.findByText('Open originating opportunity')).toBeInTheDocument();
  });

  it('renders compiler regression explanation details and saves an explanation note', async () => {
    apiClient.listSynthesisJobs.mockResolvedValue({
      jobs: [
        {
          id: 'job-exp-1',
          user_id: 'user-1',
          job_type: 'compiler_regression_explanation',
          title: 'Compiler regression explanation',
          document_ids: [],
          paper_ids: [],
          research_note_id: 'note-source-1',
          output_format: 'markdown',
          output_style: 'technical',
          status: 'completed',
          progress: 100,
          result_content: '# Compiler Regression Explanation',
          result_metadata: {
            summary: 'Compile time regressed because vectorization remarks disappeared.',
            regression_type: 'compile_time',
            primary_run_id: 'run-new',
            comparison_run_id: 'run-old',
            metric_deltas: [
              {
                metric: 'compile_time_ms',
                primary: 1400,
                comparison: 1200,
                interpretation: 'regression',
              },
            ],
            artifact_deltas: [
              {
                kind: 'remarks',
                summary: 'loop-vectorize remarks missing in the primary run',
              },
            ],
            likely_causes: [
              {
                title: 'Vectorizer not firing',
                confidence: 'medium',
                reason: 'loop-vectorize remarks disappeared',
              },
            ],
            confounders: ['Single-machine sample'],
            recommended_next_steps: ['Diff pass remarks across both builds'],
          },
          created_at: '2026-03-12T00:00:00Z',
          completed_at: '2026-03-12T01:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
    });
    apiClient.getResearchNote.mockResolvedValue({
      id: 'note-source-1',
      user_id: 'user-1',
      title: 'Compiler hypotheses',
      content_markdown: '## Hypothesis',
      structured_payload: {},
    });

    renderWithProviders('/synthesis?job=job-exp-1');

    const jobTitles = await screen.findAllByText('Compiler regression explanation');
    fireEvent.click(jobTitles[0]);
    expect(
      await screen.findByText('Compile time regressed because vectorization remarks disappeared.')
    ).toBeInTheDocument();
    expect(screen.getByText(/Type: compile_time · Primary run-new · Comparison run-old/i)).toBeInTheDocument();
    expect(screen.getByText(/compile_time_ms: 1200 → 1400 · regression/i)).toBeInTheDocument();
    expect(screen.getByText(/remarks: loop-vectorize remarks missing in the primary run/i)).toBeInTheDocument();
    expect(screen.getByText(/Vectorizer not firing \[medium\] · loop-vectorize remarks disappeared/i)).toBeInTheDocument();
    expect(screen.getByText('Single-machine sample')).toBeInTheDocument();
    expect(screen.getByText('Diff pass remarks across both builds')).toBeInTheDocument();
    expect(screen.getByText('Save Explanation Note')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Save Explanation Note'));

    await waitFor(() => {
      expect(apiClient.saveSynthesisJobAsResearchNote).toHaveBeenCalledWith('job-exp-1', {
        title: 'Compiler regression explanation',
        tags: ['compiler-regression-explanation', 'performance-analysis'],
      });
    });
  });

  it('renders compiler patch proposal details and saves a proposal note', async () => {
    apiClient.listSynthesisJobs.mockResolvedValue({
      jobs: [
        {
          id: 'job-prop-1',
          user_id: 'user-1',
          job_type: 'compiler_patch_proposal',
          title: 'Compiler patch proposal',
          document_ids: [],
          paper_ids: [],
          research_note_id: 'note-exp-1',
          output_format: 'markdown',
          output_style: 'technical',
          status: 'completed',
          progress: 100,
          result_content: '# Compiler Patch Proposal',
          result_metadata: {
            proposal_summary: 'Gate the vectorization heuristic with a narrow profitability check.',
            target_area: 'vectorization heuristic',
            candidate_change: 'Add a guard before enabling the transform.',
            expected_effect: 'Reduce compile-time regressions while preserving wins.',
            mechanism: 'Avoid costly transforms on marginal loops.',
            validation_plan: 'Run the compiler regression suite and diff remarks.',
            risk_assessment: 'May suppress beneficial vectorization in edge cases.',
            rollback_or_guardrail: 'Feature-flag the heuristic and keep a fast rollback path.',
            source_explanation_note_id: 'note-exp-1',
          },
          created_at: '2026-03-12T00:00:00Z',
          completed_at: '2026-03-12T01:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
    });
    apiClient.getResearchNote.mockResolvedValue({
      id: 'note-exp-1',
      user_id: 'user-1',
      title: 'Compiler regression explanation',
      content_markdown: '## Explanation',
      structured_payload: {
        artifact_type: 'compiler_regression_explanation',
      },
    });

    renderWithProviders('/synthesis?job=job-prop-1');

    const jobTitles = await screen.findAllByText('Compiler patch proposal');
    fireEvent.click(jobTitles[0]);
    expect(screen.getByText(/Gate the vectorization heuristic with a narrow profitability check/i)).toBeInTheDocument();
    expect(screen.getByText(/Target area: vectorization heuristic · Source note note-exp-1/i)).toBeInTheDocument();
    expect(screen.getByText(/Add a guard before enabling the transform/i)).toBeInTheDocument();
    expect(screen.getByText(/Reduce compile-time regressions while preserving wins/i)).toBeInTheDocument();
    expect(screen.getByText(/Avoid costly transforms on marginal loops/i)).toBeInTheDocument();
    expect(screen.getByText(/Run the compiler regression suite and diff remarks/i)).toBeInTheDocument();
    expect(screen.getByText(/May suppress beneficial vectorization in edge cases/i)).toBeInTheDocument();
    expect(screen.getByText(/Feature-flag the heuristic and keep a fast rollback path/i)).toBeInTheDocument();
    expect(screen.getByText('Save Proposal Note')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Save Proposal Note'));

    await waitFor(() => {
      expect(apiClient.saveSynthesisJobAsResearchNote).toHaveBeenCalledWith('job-prop-1', {
        title: 'Compiler patch proposal',
        tags: ['compiler-patch-proposal', 'compiler-proposal'],
      });
    });
  });

  it('renders compiler patch draft details and saves a patch draft note', async () => {
    apiClient.listSynthesisJobs.mockResolvedValue({
      jobs: [
        {
          id: 'job-draft-1',
          user_id: 'user-1',
          job_type: 'compiler_patch_draft',
          title: 'Compiler patch draft',
          document_ids: [],
          paper_ids: [],
          research_note_id: 'note-prop-1',
          output_format: 'markdown',
          output_style: 'technical',
          status: 'completed',
          progress: 100,
          result_content: '# Compiler Patch Draft',
          result_metadata: {
            draft_summary: 'Target vectorization profitability hooks in a narrow backend file set.',
            source_name: 'Compiler Repo Source',
            source_id: 'repo-1',
            target_files: ['llvm/lib/Transforms/Vectorize/LoopVectorize.cpp'],
            target_symbols: ['LoopVectorizationPlanner'],
            change_plan: ['Add a profitability guard before enabling the transform.'],
            validation_commands: ['ninja check-llvm'],
            rollback_steps: ['Revert the guard or disable it behind a flag.'],
          },
          created_at: '2026-03-12T00:00:00Z',
          completed_at: '2026-03-12T01:00:00Z',
        },
      ],
      total: 1,
      page: 1,
      page_size: 50,
    });
    apiClient.getResearchNote.mockResolvedValue({
      id: 'note-prop-1',
      user_id: 'user-1',
      title: 'Compiler patch proposal',
      content_markdown: '## Proposal',
      structured_payload: {
        artifact_type: 'compiler_patch_proposal',
      },
    });

    renderWithProviders('/synthesis?job=job-draft-1');

    const jobTitles = await screen.findAllByText('Compiler patch draft');
    fireEvent.click(jobTitles[0]);
    expect(screen.getByText(/Target vectorization profitability hooks in a narrow backend file set/i)).toBeInTheDocument();
    expect(screen.getByText(/Repo source: Compiler Repo Source · repo-1/i)).toBeInTheDocument();
    expect(screen.getByText(/llvm\/lib\/Transforms\/Vectorize\/LoopVectorize\.cpp/i)).toBeInTheDocument();
    expect(screen.getByText(/LoopVectorizationPlanner/i)).toBeInTheDocument();
    expect(screen.getByText(/Add a profitability guard before enabling the transform/i)).toBeInTheDocument();
    expect(screen.getByText(/ninja check-llvm/i)).toBeInTheDocument();
    expect(screen.getByText(/Revert the guard or disable it behind a flag/i)).toBeInTheDocument();
    expect(screen.getByText('Save Patch Draft Note')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Save Patch Draft Note'));

    await waitFor(() => {
      expect(apiClient.saveSynthesisJobAsResearchNote).toHaveBeenCalledWith('job-draft-1', {
        title: 'Compiler patch draft',
        tags: ['compiler-patch-draft', 'compiler-change-plan'],
      });
    });
  });
});
