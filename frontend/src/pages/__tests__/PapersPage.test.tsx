import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import PapersPage from '../PapersPage';

const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

jest.mock('../../services/api', () => ({
  apiClient: {
    searchArxiv: jest.fn(),
    translateArxivQuery: jest.fn(),
    listArxivImports: jest.fn(),
    listResearchPapers: jest.fn(),
    listPaperExtractionJobs: jest.fn(),
    getResearchPaper: jest.fn(),
    extractResearchPapers: jest.fn(),
    saveResearchPaperAsNote: jest.fn(),
    reextractResearchPaper: jest.fn(),
    createSynthesisJob: jest.fn(),
    ingestArxivPapers: jest.fn(),
    summarizeArxivImport: jest.fn(),
    generateReviewForArxivImport: jest.fn(),
    generateSlidesForArxivImport: jest.fn(),
    enrichMetadataForArxivImport: jest.fn(),
    createReadingList: jest.fn(),
    synthesizeWorkflow: jest.fn(),
    createWorkflow: jest.fn(),
    createIngestionProgressWebSocket: jest.fn(),
  },
}));

const apiClient = require('../../services/api').apiClient;

const renderWithProviders = (initialEntry: string = '/papers?q=all:compiler') => {
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
        <PapersPage />
      </QueryClientProvider>
    </MemoryRouter>
  );
};

describe('PapersPage', () => {
  beforeEach(() => {
    apiClient.searchArxiv.mockResolvedValue({
      total_results: 1,
      start: 0,
      max_results: 10,
      items: [
        {
          id: '2401.12345',
          entry_url: 'https://arxiv.org/abs/2401.12345',
          pdf_url: 'https://arxiv.org/pdf/2401.12345.pdf',
          title: 'Compiler Optimization via Layouts',
          summary: 'Paper abstract',
          authors: ['Ada Lovelace'],
          published: '2026-03-01T00:00:00Z',
          updated: '2026-03-02T00:00:00Z',
          categories: ['cs.PL'],
          primary_category: 'cs.PL',
        },
      ],
    });
    apiClient.translateArxivQuery.mockResolvedValue({ query: 'all:compiler' });
    apiClient.listArxivImports.mockResolvedValue({
      items: [
        {
          id: 'source-1',
          name: 'arXiv 2401.12345',
          is_syncing: false,
          display: { paper_ids: ['2401.12345'] },
          document_count: 1,
        },
      ],
      total: 1,
      limit: 10,
      offset: 0,
    });
    apiClient.listResearchPapers.mockResolvedValue({
      items: [
        {
          id: 'paper-1',
          user_id: 'user-1',
          document_id: 'doc-1',
          source_id: 'source-1',
          arxiv_id: '2401.12345',
          title: 'Compiler Optimization via Layouts',
          extraction_status: 'completed',
          summary: 'Structured summary',
          mechanisms: ['layout transform'],
          assumptions: ['regular access patterns'],
          benchmarks: ['PolyBench'],
          metrics: ['runtime'],
          limitations: ['irregular kernels not tested'],
          claims: [
            {
              id: 'claim-1',
              kind: 'performance',
              statement: 'If layout is optimized, runtime improves.',
              target_layer: 'midend',
              confidence: 0.8,
              evidence_summary: 'Measured on PolyBench',
            },
          ],
          latest_job: { id: 'job-1', status: 'completed' },
        },
      ],
      total: 1,
      limit: 500,
      offset: 0,
    });
    apiClient.listPaperExtractionJobs.mockResolvedValue([
      { id: 'job-1', document_id: 'doc-1', source_id: 'source-1', status: 'completed' },
    ]);
    apiClient.getResearchPaper.mockResolvedValue({
      id: 'paper-1',
      user_id: 'user-1',
      document_id: 'doc-1',
      source_id: 'source-1',
      arxiv_id: '2401.12345',
      title: 'Compiler Optimization via Layouts',
      extraction_status: 'completed',
      summary: 'Structured summary',
      mechanisms: ['layout transform'],
      assumptions: ['regular access patterns'],
      benchmarks: ['PolyBench'],
      metrics: ['runtime'],
      limitations: ['irregular kernels not tested'],
      claims: [
        {
          id: 'claim-1',
          kind: 'performance',
          statement: 'If layout is optimized, runtime improves.',
          target_layer: 'midend',
          confidence: 0.8,
          evidence_summary: 'Measured on PolyBench',
        },
      ],
      latest_job: { id: 'job-1', status: 'completed' },
      paper_url: 'https://arxiv.org/abs/2401.12345',
    });
    apiClient.extractResearchPapers.mockResolvedValue([{ id: 'job-2', document_id: 'doc-1', status: 'pending' }]);
    apiClient.saveResearchPaperAsNote.mockResolvedValue({ id: 'note-1', title: 'Paper Extraction: Compiler Optimization via Layouts' });
    apiClient.reextractResearchPaper.mockResolvedValue({ id: 'job-3', status: 'pending' });
    apiClient.createSynthesisJob.mockResolvedValue({ id: 'syn-1' });
    apiClient.createIngestionProgressWebSocket.mockImplementation(() => ({ addEventListener: jest.fn(), removeEventListener: jest.fn(), close: jest.fn() }));
    apiClient.ingestArxivPapers.mockResolvedValue({});
    apiClient.summarizeArxivImport.mockResolvedValue({ queued: 1 });
    apiClient.generateReviewForArxivImport.mockResolvedValue({});
    apiClient.generateSlidesForArxivImport.mockResolvedValue({});
    apiClient.enrichMetadataForArxivImport.mockResolvedValue({});
    apiClient.createReadingList.mockResolvedValue({});
    apiClient.synthesizeWorkflow.mockResolvedValue({ workflow: {} });
    apiClient.createWorkflow.mockResolvedValue({});
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('opens extracted paper details and saves the paper as a note', async () => {
    renderWithProviders();

    expect(await screen.findByText('Open Extracted')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Open Extracted'));

    expect(await screen.findByText('Extracted paper')).toBeInTheDocument();
    expect(await screen.findByText('Structured summary')).toBeInTheDocument();
    expect(await screen.findByText('layout transform')).toBeInTheDocument();
    expect(await screen.findByText('If layout is optimized, runtime improves.')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Save as Note'));

    await waitFor(() => {
      expect(apiClient.saveResearchPaperAsNote).toHaveBeenCalledWith('paper-1', {
        title: 'Paper Extraction: Compiler Optimization via Layouts',
        tags: ['paper-extraction', 'arxiv'],
      });
    });
  });

  it('queues extraction from the import queue', async () => {
    apiClient.listResearchPapers.mockResolvedValueOnce({
      items: [],
      total: 0,
      limit: 500,
      offset: 0,
    });

    renderWithProviders();

    expect(await screen.findByText('Extract Structure')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Extract Structure'));

    await waitFor(() => {
      expect(apiClient.extractResearchPapers).toHaveBeenCalledWith({
        source_id: 'source-1',
        force: false,
        limit: 200,
      });
    });
  });

  it('queues a literature review from the inline topic field', async () => {
    renderWithProviders();

    const card = await screen.findByRole('article', { name: 'arXiv 2401.12345' });
    fireEvent.change(within(card).getByPlaceholderText('Review topic'), {
      target: { value: 'Compiler optimization' },
    });
    fireEvent.click(within(card).getByText('Generate Review'));

    await waitFor(() => {
      expect(apiClient.generateReviewForArxivImport).toHaveBeenCalledWith('source-1', {
        topic: 'Compiler optimization',
      });
    });
  });

  it('creates a hypothesis synthesis job from selected extracted papers', async () => {
    renderWithProviders();

    const generateButton = await screen.findByRole('button', { name: /Generate Hypotheses/i });
    expect(generateButton).toBeDisabled();

    const selectors = await screen.findAllByLabelText('Select paper Compiler Optimization via Layouts for hypothesis generation');
    fireEvent.click(selectors[0]);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Generate Hypotheses \(1\)/i })).not.toBeDisabled();
    });

    fireEvent.click(screen.getByRole('button', { name: /Generate Hypotheses \(1\)/i }));

    await waitFor(() => {
      expect(apiClient.createSynthesisJob).toHaveBeenCalledWith(
        expect.objectContaining({
          job_type: 'gap_analysis_hypotheses',
          document_ids: [],
          paper_ids: ['paper-1'],
        })
      );
    });
  });
});
